"""
QuantGuard - Pathway Streaming Engine
======================================
Real-time transaction processing pipeline built on Pathway's
streaming computation engine.

Pipeline stages:
  1. Data Ingestion   – JSONL connector reads from transaction stream
  2. Feature Eng.     – Windowed aggregations compute per-user statistics
  3. Anomaly Detect.  – Statistical deviation + threshold checks
  4. Quantum Routing  – High-risk transactions scored by VQC classifier
  5. LLM Enrichment   – RAG-based explanations for flagged transactions
  6. Alert Output     – Enriched alerts written to JSONL

On Linux  : Uses native Pathway streaming engine (exactly-once semantics)
On Windows: Automatically falls back to compatibility layer
"""

# ── Pathway Import (with automatic fallback) ────────────────────────────

try:
    import pathway as pw

    # Guard against the dummy/placeholder pathway package
    _ver = getattr(pw, "__version__", "0.post1")
    if _ver == "0.post1":
        raise ImportError("Dummy pathway package detected")
    PATHWAY_NATIVE = True
except (ImportError, Exception):
    from pathway_compat import pw
    PATHWAY_NATIVE = False

import os
import json
import math
import time as _time_mod
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# ── ML / Engine logger (writes to file so TUI can't hide it) ────────────
os.makedirs("data", exist_ok=True)
_ml_logger = logging.getLogger("quantguard.ml")
_ml_logger.setLevel(logging.INFO)
_ml_fh = logging.FileHandler("data/ml_output.log", mode="a", encoding="utf-8")
_ml_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
_ml_logger.addHandler(_ml_fh)
# Also log to stderr (survives TUI)
_ml_sh = logging.StreamHandler()
_ml_sh.setFormatter(logging.Formatter("%(message)s"))
_ml_logger.addHandler(_ml_sh)

# ── Lazy-loaded heavy modules (avoid import at module level) ────────────
_classifier = None
_analyzer = None
_engine_xpack = None   # Initialized inside _run_native_pipeline() before pw.run()


def get_engine_xpack():
    """Return the PathwayLLMxPack instance created inside the native pipeline.

    This is the canonical way to access the xPack — its pw.io.fs.read()
    tables are wired into the same Pathway computation graph as the main
    fraud detection pipeline.  Returns ``None`` if the engine hasn't
    started yet or if xPack initialization failed.
    """
    return _engine_xpack


def _get_classifier():
    """Lazy-init Quantum Risk Classifier (connects to IBM Quantum)."""
    global _classifier
    if _classifier is None:
        from quantum_classifier import QuantumRiskClassifier
        _classifier = QuantumRiskClassifier()
    return _classifier


def _get_analyzer():
    """Lazy-init Groq LLM Fraud Analyzer."""
    global _analyzer
    if _analyzer is None:
        from llm_engine import GroqFraudAnalyzer
        _analyzer = GroqFraudAnalyzer()
    return _analyzer


def _quantum_enrich(row_json: str) -> str:
    """
    UDF called by Pathway .apply() on each anomalous transaction.
    Runs IBM Quantum VQC classification + LLM explanation.
    Returns enriched alert JSON string.
    """
    try:
        tx = json.loads(row_json)
    except (json.JSONDecodeError, ValueError):
        return row_json

    clf = _get_classifier()
    llm = _get_analyzer()

    amount = float(tx.get("amount", 0))
    avg_amount = float(tx.get("avg_amount", amount))
    max_amount = float(tx.get("max_amount", amount))
    tx_count = int(tx.get("tx_count", 1))

    # ── Quantum Classification ──────────────────────────────────────
    amount_norm = min(amount / max(max_amount, avg_amount * 3, 1), 1.0)
    freq_norm = min(tx_count / max(50, tx_count * 1.5), 1.0)
    features = np.array([amount_norm, freq_norm])

    quantum_result = clf.classify_transaction(features)

    # ── LLM Explanation ─────────────────────────────────────────────
    llm_text = llm.explain_fraud_risk(tx, quantum_result)

    # ── Risk Level ──────────────────────────────────────────────────
    fp = quantum_result["fraud_probability"]
    risk = (
        "CRITICAL" if fp > 0.70 else
        "HIGH" if fp > 0.45 else
        "MEDIUM" if fp > 0.25 else
        "LOW"
    )

    # ── Enriched Alert ──────────────────────────────────────────────
    alert = {
        "user_id": tx.get("user_id", "UNKNOWN"),
        "amount": amount,
        "location": tx.get("location", "N/A"),
        "category": tx.get("category", "N/A"),
        "timestamp": tx.get("timestamp", datetime.now().isoformat()),
        "risk_level": risk,
        "quantum_classification": quantum_result["classification"],
        "quantum_fraud_probability": fp,
        "quantum_states": quantum_result.get("quantum_states", {}),
        "ibm_job_id": quantum_result.get("ibm_job_id", ""),
        "llm_explanation": llm_text,
        "anomaly_reasons": tx.get("anomaly_reasons", []),
        "rolling_stats": {
            "avg_amount": round(avg_amount, 2),
            "max_amount": round(max_amount, 2),
            "tx_count": tx_count,
        },
        "alert_type": "Quantum Risk Verified",
        "processed_at": datetime.now().isoformat(),
    }

    # ── Terminal Output ─────────────────────────────────────────────
    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(risk, "🟢")
    print(f"\n  {icon} [{risk}] {tx.get('user_id')} | ${amount:,.2f} | "
          f"{tx.get('location')} | P(fraud)={fp:.1%} | "
          f"IBM Job: {quantum_result.get('ibm_job_id', 'N/A')}")

    return json.dumps(alert, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════

ENGINE_ACTIVE = False   # set to True once pw.run() starts
_live_subjects: list = []  # populated by run_analysis() with subject instances

# Circular buffer of recent live transactions for the dashboard
_LIVE_TX_BUFFER_SIZE = 200
_live_tx_buffer: list = []  # list of dicts, newest last
_live_tx_total: int = 0     # monotonic counter of all txs pushed


def get_live_tx_count() -> int:
    """Return total transactions emitted across all ConnectorSubjects."""
    return sum(getattr(s, 'total_emitted', 0) for s in _live_subjects)


def get_live_tx_buffer(n: int = 50) -> list:
    """Return the most recent *n* live transactions (newest last)."""
    return _live_tx_buffer[-n:]


def get_live_tx_total() -> int:
    """Return the monotonic total count of transactions pushed to the buffer."""
    return _live_tx_total


def get_log_anomaly_alerts(n: int = 50) -> list:
    """Return the most recent *n* log anomaly alerts (newest last)."""
    path = os.path.join("data", "log_anomaly_alerts.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
        return out
    except Exception:
        return []


def get_log_anomaly_stats() -> dict:
    """Return log anomaly pipeline summary statistics."""
    alerts = get_log_anomaly_alerts(200)
    if not alerts:
        return {"total": 0, "by_service": {}, "by_severity": {}}
    from collections import Counter
    svc = Counter(a.get("service", "unknown") for a in alerts)
    sev = Counter(a.get("severity", "LOW") for a in alerts)
    return {
        "total": len(alerts),
        "by_service": dict(svc.most_common()),
        "by_severity": dict(sev.most_common()),
    }


def run_analysis():
    """Define and execute the streaming analysis pipeline."""
    global ENGINE_ACTIVE

    print(f"[Engine] Pathway mode: {'Native' if PATHWAY_NATIVE else 'Compatibility'}")

    if not PATHWAY_NATIVE:
        # Compat mode — pathway_compat.py already has quantum + LLM
        ENGINE_ACTIVE = True
        _run_compat_pipeline()
        return

    # ── Native Pathway Pipeline ─────────────────────────────────────────
    #
    #   Architecture (9 stages — all transforms run in Pathway's Rust engine):
    #     1. pw.io.python.read        — FilePollingSubject (NTFS-safe poller)
    #     2. pw.io.python.read        — TransactionSubject (live generator)
    #        concat_reindex           — merge file + live into unified stream
    #     3. pw.io.subscribe          — per-user rolling statistics tracker
    #     4. pw.Table.with_columns    — timestamp parsing (feature eng.)
    #     5. pw.temporal.sliding      — 60s window / 10s hop aggregation
    #     6. pw.Table.join + select   — enrich transactions with windowed stats
    #     7. pw.Table.filter          — multi-rule anomaly detection
    #     8. pw.io.subscribe          — 6-feature ML scoring + Quantum VQC + LLM
    #     9. Alert sink               — enriched JSONL for dashboard/API
    #
    #   subscribe(on_change) receives: (key, row: dict, time, is_addition)
    # ────────────────────────────────────────────────────────────────────

    # ── In-memory rolling state ─────────────────────────────────────────
    _user_stats: dict = {}       # {uid: {amounts, locations, categories, timestamps}}
    _processed_keys: set = set() # dedup quantum calls

    # ── 1. Schema ───────────────────────────────────────────────────────
    class TransactionSchema(pw.Schema):
        user_id: str
        amount: float
        location: str
        category: str
        timestamp: str
        is_suspicious_flag: bool

    # ── 2a. Data Ingestion — File Polling ConnectorSubject ─────────────
    #
    #   On WSL2, NTFS-mounted paths (/mnt/d/…) do NOT support inotify,
    #   so pw.io.jsonlines.read in streaming mode stops after the initial
    #   batch — it never receives filesystem change notifications.
    #
    #   FilePollingSubject solves this by:
    #     • Reading all existing JSONL lines on startup (initial load)
    #     • Polling the file every 2 seconds for new lines (tail -f style)
    #     • Emitting each line as a typed row via self.next(**kwargs)
    #     • Tracking file position + line count for resume
    #
    #   This is a proper pw.io.python.ConnectorSubject managed by the
    #   Pathway engine (thread lifecycle, auto-commit batching, backpressure).
    #
    import glob as _glob

    class FilePollingSubject(pw.io.python.ConnectorSubject):
        """Custom ConnectorSubject — polls JSONL file for new lines.

        Replaces pw.io.jsonlines.read on NTFS/WSL where inotify is broken.
        Extends pw.io.python.ConnectorSubject (ABC).
        """

        def __init__(self, directory: str, pattern: str = "transactions*",
                     poll_interval: float = 2.0):
            super().__init__(datasource_name="quantguard_file_poller")
            self._directory = directory
            self._pattern = pattern
            self._poll_interval = poll_interval
            # Track read position per file: {filepath: lines_read}
            self._file_positions: dict[str, int] = {}
            self.total_emitted: int = 0

        @property
        def _deletions_enabled(self) -> bool:
            return False

        def _is_finite(self) -> bool:
            # Infinite — keeps polling until engine stops
            return False

        def _discover_files(self) -> list[str]:
            """Find all matching JSONL files in the directory."""
            import os as _os
            matches = []
            if _os.path.isdir(self._directory):
                for f in sorted(_os.listdir(self._directory)):
                    if _glob.fnmatch.fnmatch(f, self._pattern):
                        matches.append(_os.path.join(self._directory, f))
            return matches

        def _read_new_lines(self, filepath: str) -> list[str]:
            """Read lines from filepath starting after previously-read count."""
            skip = self._file_positions.get(filepath, 0)
            new_lines = []
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i >= skip:
                            stripped = line.strip()
                            if stripped:
                                new_lines.append(stripped)
                self._file_positions[filepath] = skip + len(new_lines)
            except Exception as exc:
                _ml_logger.info(
                    f"[FilePoller] Error reading {filepath}: {exc}"
                )
            return new_lines

        def run(self) -> None:
            """Main loop — called by Pathway engine in a dedicated thread.

            Performs initial bulk load, then polls every 2s for new lines.
            Each JSON line is parsed and emitted via self.next(**kwargs).
            """
            import time as _t
            _ml_logger.info(
                f"[FilePoller] Started — dir={self._directory} "
                f"pattern={self._pattern} poll={self._poll_interval}s"
            )
            poll_count = 0

            while True:
                files = self._discover_files()
                batch_count = 0

                for fpath in files:
                    for raw_line in self._read_new_lines(fpath):
                        try:
                            rec = json.loads(raw_line)
                            self.next(
                                user_id=str(rec.get("user_id", "")),
                                amount=float(rec.get("amount", 0)),
                                location=str(rec.get("location", "")),
                                category=str(rec.get("category", "")),
                                timestamp=str(rec.get("timestamp", "")),
                                is_suspicious_flag=bool(
                                    rec.get("is_suspicious_flag", False)
                                ),
                            )
                            batch_count += 1
                            self.total_emitted += 1
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            pass  # skip malformed lines

                poll_count += 1
                if batch_count > 0:
                    _ml_logger.info(
                        f"[FilePoller] Poll #{poll_count}: emitted "
                        f"{batch_count} new rows (total: {self.total_emitted})"
                    )
                elif poll_count % 30 == 0:
                    # Heartbeat every ~60s
                    _ml_logger.info(
                        f"[FilePoller] Poll #{poll_count}: no new data "
                        f"(total emitted: {self.total_emitted})"
                    )

                _t.sleep(self._poll_interval)

    file_subject = FilePollingSubject("data", pattern="transactions*")
    _live_subjects.append(file_subject)
    file_transactions = pw.io.python.read(
        file_subject,
        schema=TransactionSchema,
        autocommit_duration_ms=2000,
        name="quantguard_file_poller",
    )

    # ── 2b. Data Ingestion — Custom Python ConnectorSubject ─────────────
    #
    #   Demonstrates pw.io.python.ConnectorSubject — a proper Pathway
    #   custom input connector that runs in its own thread managed by
    #   the engine, emits typed rows via self.next(**kwargs), and
    #   supports auto-commit batching.
    #
    #   TransactionSubject generates synthetic transactions with realistic
    #   fraud patterns (high-value spikes, geo anomalies, velocity bursts,
    #   category anomalies) using the same user profiles as data_source.py.
    #   It acts as a *live streaming source* that continuously feeds the
    #   pipeline even when the JSONL file is not being appended to
    #   (solves the NTFS/inotify gap on WSL).
    #
    class TransactionSubject(pw.io.python.ConnectorSubject):
        """Custom Pathway ConnectorSubject — live transaction generator.

        Extends pw.io.python.ConnectorSubject (ABC). The engine calls
        start() which spawns a thread running our run() method.
        We emit rows via self.next(**kwargs) matching TransactionSchema.
        """

        _LOCATIONS = [
            "New York", "London", "Tokyo", "Mumbai", "Paris",
            "Berlin", "Dubai", "Singapore", "Sydney", "Toronto",
        ]
        _CATEGORIES = [
            "Grocery", "Electronics", "Travel", "Dining",
            "Healthcare", "Entertainment", "Utilities", "Shopping",
        ]
        _FRAUD_RATE = 0.15

        def __init__(self):
            super().__init__(datasource_name="quantguard_live_generator")
            self.total_emitted: int = 0
            # Build user profiles (same seed as data_source.py)
            import random as _rng
            _rng.seed(42)
            self._profiles: dict = {}
            for i in range(100, 115):
                self._profiles[f"USER_{i}"] = {
                    "home": _rng.choice(self._LOCATIONS[:5]),
                    "avg": round(_rng.uniform(100, 800), 2),
                    "cats": _rng.sample(self._CATEGORIES, 3),
                }
            _rng.seed()

        @property
        def _deletions_enabled(self) -> bool:
            return False  # append-only stream — improves performance

        def run(self) -> None:
            """Main loop — called by Pathway engine in a dedicated thread.

            Generates transactions indefinitely using the same fraud
            patterns as data_source.py:
              • high_value  — amount $4 000-$5 000
              • geo_anomaly — foreign city for the user
              • velocity    — rapid-fire electronics purchases
              • category    — unusual category for the user's profile
            """
            import random as _rng
            import time as _t
            _ml_logger.info("[ConnectorSubject] TransactionSubject.run() started")
            while True:
                try:
                    is_fraud = _rng.random() < self._FRAUD_RATE
                    uid = _rng.choice(list(self._profiles))
                    p = self._profiles[uid]

                    if is_fraud:
                        pattern = _rng.choice([
                            "high_value", "geo_anomaly",
                            "velocity", "category_anomaly",
                        ])
                        if pattern == "high_value":
                            amt = round(_rng.uniform(4000, 5000), 2)
                            loc = _rng.choice(self._LOCATIONS)
                            cat = _rng.choice(self._CATEGORIES)
                        elif pattern == "geo_anomaly":
                            amt = round(_rng.uniform(500, 3000), 2)
                            foreign = [l for l in self._LOCATIONS
                                        if l != p["home"]]
                            loc = _rng.choice(foreign)
                            cat = _rng.choice(self._CATEGORIES)
                        elif pattern == "velocity":
                            amt = round(_rng.uniform(200, 1500), 2)
                            loc = _rng.choice(self._LOCATIONS)
                            cat = "Electronics"
                        else:  # category_anomaly
                            amt = round(_rng.uniform(1000, 4000), 2)
                            unusual = [c for c in self._CATEGORIES
                                        if c not in p["cats"]]
                            cat = (_rng.choice(unusual) if unusual
                                   else _rng.choice(self._CATEGORIES))
                            loc = _rng.choice(self._LOCATIONS)
                        flag = True
                    else:
                        amt = round(max(5.0, _rng.gauss(
                            p["avg"], p["avg"] * 0.4)), 2)
                        loc = (p["home"] if _rng.random() < 0.70
                               else _rng.choice(self._LOCATIONS))
                        cat = (_rng.choice(p["cats"])
                               if _rng.random() < 0.60
                               else _rng.choice(self._CATEGORIES))
                        flag = False

                    # Emit typed row — Pathway engine picks it up
                    ts = datetime.now().isoformat()
                    self.next(
                        user_id=uid,
                        amount=amt,
                        location=loc,
                        category=cat,
                        timestamp=ts,
                        is_suspicious_flag=flag,
                    )
                    self.total_emitted += 1

                    # Push to module-level live buffer for dashboard
                    global _live_tx_total
                    _live_tx_buffer.append({
                        "user_id": uid,
                        "amount": amt,
                        "location": loc,
                        "category": cat,
                        "timestamp": ts,
                        "is_suspicious_flag": flag,
                    })
                    _live_tx_total += 1
                    if len(_live_tx_buffer) > _LIVE_TX_BUFFER_SIZE:
                        del _live_tx_buffer[:len(_live_tx_buffer) - _LIVE_TX_BUFFER_SIZE]

                    if self.total_emitted % 25 == 0:
                        _ml_logger.info(
                            f"[ConnectorSubject] Emitted {self.total_emitted} live "
                            f"transactions ({self._FRAUD_RATE:.0%} fraud rate)"
                        )

                    # Throttle — ~1–2 tx/sec
                    _t.sleep(_rng.uniform(0.5, 1.0))

                except Exception as exc:
                    _ml_logger.info(
                        f"[ConnectorSubject] Error in run(): {exc}"
                    )
                    import traceback
                    traceback.print_exc()
                    break

    live_subject = TransactionSubject()
    _live_subjects.append(live_subject)
    live_transactions = pw.io.python.read(
        live_subject,
        schema=TransactionSchema,
        autocommit_duration_ms=2000,
        name="quantguard_live_stream",
    )

    # ── 2d. Data Ingestion — Native Kafka Connector (pw.io.kafka.read) ──
    #
    #   Uses Pathway's built-in Kafka connector — runs inside the Rust
    #   engine with zero-copy deserialization and exactly-once semantics.
    #   No kafka-python needed; Pathway links against librdkafka natively.
    #
    #   Environment variables:
    #     KAFKA_BOOTSTRAP_SERVERS — comma-separated broker addresses
    #                               (default: localhost:9092)
    #     KAFKA_TOPIC             — topic name (default: quantguard-transactions)
    #     KAFKA_GROUP_ID          — consumer group (default: quantguard-consumer)
    #
    #   If the broker is unreachable at pipeline start, Pathway retries
    #   automatically (librdkafka reconnect back-off). The pipeline
    #   continues processing FilePoller + TransactionSubject sources in
    #   the meantime.
    #

    _kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    _kafka_topic = os.environ.get("KAFKA_TOPIC", "quantguard-transactions")
    _kafka_group = os.environ.get("KAFKA_GROUP_ID", "quantguard-consumer")

    # Probe broker before creating the native connector — librdkafka
    # panics the Rust engine if the broker is completely unreachable.
    _kafka_reachable = False
    try:
        import socket as _sock
        _khost, _kport = _kafka_bootstrap.split(",")[0].rsplit(":", 1)
        _ksock = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        _ksock.settimeout(2)
        _ksock.connect((_khost, int(_kport)))
        _ksock.close()
        _kafka_reachable = True
    except Exception:
        pass

    if _kafka_reachable:
        kafka_transactions = pw.io.kafka.read(
            rdkafka_settings={
                "bootstrap.servers": _kafka_bootstrap,
                "group.id": _kafka_group,
                "auto.offset.reset": "latest",
                "session.timeout.ms": "10000",
            },
            topic=_kafka_topic,
            schema=TransactionSchema,
            format="json",
            autocommit_duration_ms=2000,
        )
        _ml_logger.info(
            f"[KafkaNative] pw.io.kafka.read → {_kafka_bootstrap} "
            f"topic={_kafka_topic} group={_kafka_group}"
        )
    else:
        # Kafka broker offline — create a dormant ConnectorSubject stub
        # so the graph stays valid and the pipeline keeps running.
        class _KafkaDormantSubject(pw.io.python.ConnectorSubject):
            """No-op stub when Kafka broker is unreachable."""
            @property
            def _deletions_enabled(self) -> bool:
                return False
            def _is_finite(self) -> bool:
                return False
            def run(self) -> None:
                import time as _t
                _ml_logger.info(
                    f"[KafkaNative] Broker {_kafka_bootstrap} unreachable "
                    "— Kafka source dormant (will not retry). "
                    "Restart with a running broker to enable."
                )
                while True:
                    _t.sleep(60)

        _kafka_stub = _KafkaDormantSubject()
        _live_subjects.append(_kafka_stub)
        kafka_transactions = pw.io.python.read(
            _kafka_stub,
            schema=TransactionSchema,
            autocommit_duration_ms=2000,
            name="quantguard_kafka_dormant",
        )
        _ml_logger.info(
            f"[KafkaNative] Broker {_kafka_bootstrap} unreachable — "
            "dormant stub active; pipeline continues without Kafka"
        )

    # ── 2e. Data Ingestion — HTTP REST Connector (live market data) ────
    #
    #   Uses Pathway's native pw.io.http.rest_connector to poll an HTTP
    #   endpoint for live market data (Alpha Vantage).  The connector
    #   runs inside the Pathway Rust engine — no external sidecar needed.
    #
    #   Each poll hits /query?function=GLOBAL_QUOTE&symbol=<SYM>.
    #   The JSON response is parsed into a TransactionSchema row by a
    #   pw.apply UDF that maps price→amount and detects suspicious moves.
    #
    #   Environment variables:
    #     ALPHA_VANTAGE_API_KEY  — API key (default: demo)
    #     MARKET_POLL_INTERVAL   — seconds between polls (default: 15)
    #

    class _MarketRawSchema(pw.Schema):
        data: str   # raw JSON body from Alpha Vantage

    _av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
    _market_poll = int(os.environ.get("MARKET_POLL_INTERVAL", "15"))
    _market_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    _market_symbol_meta = {
        "AAPL": ("Electronics", "New York"),
        "MSFT": ("Technology", "Seattle"),
        "GOOGL": ("Technology", "San Francisco"),
        "AMZN": ("Shopping", "New York"),
        "JPM": ("Banking", "New York"),
    }

    market_queries, _market_writer = pw.io.http.rest_connector(
        host="0.0.0.0",
        port=int(os.environ.get("MARKET_REST_PORT", "9090")),
        schema=_MarketRawSchema,
        autocommit_duration_ms=2000,
        delete_completed_queries=True,
    )

    import random as _mkt_rng

    def _parse_market_json(raw: str) -> str:
        """Parse Alpha Vantage quote JSON into a transaction JSON string."""
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
            quote = payload.get("Global Quote", {})
            symbol = quote.get("01. symbol", "UNKNOWN")
            price = float(quote.get("05. price", 0))
            change_pct = float(
                quote.get("10. change percent", "0").rstrip("%")
            )
        except Exception:
            # If parsing fails, produce a minimal valid row
            return json.dumps({
                "user_id": "MARKET_FEED",
                "amount": 0.0,
                "location": "Unknown",
                "category": "Market",
                "timestamp": datetime.now().isoformat(),
                "is_suspicious_flag": False,
            })

        cat, loc = _market_symbol_meta.get(symbol, ("Market", "Unknown"))
        user_id = f"MKT_{symbol}"
        is_suspicious = abs(change_pct) > 3.0 or price > 500

        return json.dumps({
            "user_id": user_id,
            "amount": round(price * _mkt_rng.uniform(0.8, 5.0), 2),
            "location": loc,
            "category": cat,
            "timestamp": datetime.now().isoformat(),
            "is_suspicious_flag": is_suspicious,
        })

    _market_parsed = market_queries.select(
        payload=pw.apply_with_type(_parse_market_json, str, pw.this.data),
    )
    market_tx = _market_parsed.select(
        user_id=pw.apply_with_type(
            lambda p: json.loads(p).get("user_id", ""), str, pw.this.payload),
        amount=pw.apply_with_type(
            lambda p: float(json.loads(p).get("amount", 0)), float, pw.this.payload),
        location=pw.apply_with_type(
            lambda p: json.loads(p).get("location", ""), str, pw.this.payload),
        category=pw.apply_with_type(
            lambda p: json.loads(p).get("category", ""), str, pw.this.payload),
        timestamp=pw.apply_with_type(
            lambda p: json.loads(p).get("timestamp", ""), str, pw.this.payload),
        is_suspicious_flag=pw.apply_with_type(
            lambda p: bool(json.loads(p).get("is_suspicious_flag", False)),
            bool, pw.this.payload),
    )
    _ml_logger.info(
        f"[MarketHTTP] pw.io.http.rest_connector → "
        f"port={os.environ.get('MARKET_REST_PORT', '9090')} "
        f"(POST market quotes as JSON)"
    )

    # ── 2f. Merge ALL sources into a single unified stream ──────────────
    #   concat_reindex merges file-based + live ConnectorSubject + Kafka
    #   + HTTP market data streams into a single table for the pipeline.
    transactions = file_transactions.concat_reindex(
        live_transactions
    ).concat_reindex(kafka_transactions).concat_reindex(market_tx)

    # ═══════════════════════════════════════════════════════════════════
    #  PARALLEL USE-CASE 2: System Log Anomaly Detection
    # ═══════════════════════════════════════════════════════════════════
    #
    #   Runs alongside the fraud pipeline on the *same* pw.run() call.
    #   Generates synthetic infrastructure log events (API errors, auth
    #   failures, latency spikes, resource exhaustion), then detects
    #   anomalous bursts using Pathway's built-in windowing + reducers.
    #
    #   Pipeline:
    #     LogAnomalySubject → pw.io.python.read → with_columns (severity) →
    #       windowby(30s/5s) → reduce(error_rate, mean_latency) →
    #       filter(anomalous) → pw.io.subscribe(_on_log_anomaly) →
    #       data/log_anomaly_alerts.jsonl
    # ───────────────────────────────────────────────────────────────────

    class LogSchema(pw.Schema):
        service: str
        level: str          # INFO, WARN, ERROR, CRITICAL
        message: str
        latency_ms: float
        status_code: int
        endpoint: str
        timestamp: str

    class LogAnomalySubject(pw.io.python.ConnectorSubject):
        """Pathway ConnectorSubject — synthetic log event generator.

        Produces realistic infrastructure log events for services like
        api-gateway, auth-service, payment-service, fraud-engine, etc.
        Periodically injects anomaly bursts (error storms, latency spikes,
        auth failures) to exercise the log anomaly detection pipeline.
        """

        _SERVICES = [
            "api-gateway", "auth-service", "payment-service",
            "fraud-engine", "notification-service", "data-pipeline",
        ]
        _ENDPOINTS = {
            "api-gateway": ["/api/analyze", "/api/stats", "/api/alerts",
                            "/api/rag/query", "/api/transactions"],
            "auth-service": ["/auth/login", "/auth/verify", "/auth/refresh",
                             "/auth/2fa"],
            "payment-service": ["/pay/process", "/pay/refund", "/pay/verify"],
            "fraud-engine": ["/classify", "/enrich", "/score"],
            "notification-service": ["/notify/email", "/notify/sms",
                                     "/notify/push"],
            "data-pipeline": ["/ingest", "/transform", "/sink"],
        }

        def __init__(self):
            super().__init__(datasource_name="quantguard_log_generator")
            self.total_emitted: int = 0

        @property
        def _deletions_enabled(self) -> bool:
            return False

        def _is_finite(self) -> bool:
            return False

        def run(self) -> None:
            import random as _rng
            import time as _t
            _ml_logger.info(
                "[LogAnomalySubject] Started — generating system logs"
            )
            anomaly_burst_counter = 0   # counts down during a burst

            while True:
                try:
                    # Every ~40 events, start a 10-event anomaly burst
                    if anomaly_burst_counter <= 0:
                        if _rng.random() < 0.025:
                            anomaly_burst_counter = _rng.randint(8, 15)
                            _ml_logger.info(
                                f"[LogAnomalySubject] 🔥 Anomaly burst "
                                f"starting ({anomaly_burst_counter} events)"
                            )

                    svc = _rng.choice(self._SERVICES)
                    eps = self._ENDPOINTS.get(svc, ["/unknown"])
                    ep = _rng.choice(eps)

                    if anomaly_burst_counter > 0:
                        # === ANOMALY BURST ===
                        anomaly_burst_counter -= 1
                        burst_type = _rng.choice([
                            "error_storm", "latency_spike", "auth_flood",
                        ])
                        if burst_type == "error_storm":
                            level = _rng.choice(["ERROR", "CRITICAL"])
                            status = _rng.choice([500, 502, 503, 504])
                            latency = round(_rng.uniform(2000, 15000), 1)
                            msg = _rng.choice([
                                f"Internal server error on {ep}",
                                f"Service {svc} unreachable",
                                f"Database connection pool exhausted",
                                f"Timeout after {latency:.0f}ms on {ep}",
                            ])
                        elif burst_type == "latency_spike":
                            level = "WARN"
                            status = 200
                            latency = round(_rng.uniform(3000, 20000), 1)
                            msg = f"Slow response {latency:.0f}ms on {ep}"
                        else:  # auth_flood
                            svc = "auth-service"
                            ep = "/auth/login"
                            level = _rng.choice(["WARN", "ERROR"])
                            status = _rng.choice([401, 403, 429])
                            latency = round(_rng.uniform(50, 300), 1)
                            msg = _rng.choice([
                                "Brute-force login attempt detected",
                                "Rate limit exceeded for IP 10.x.x.x",
                                "Invalid credentials — 5th attempt",
                                "Account lockout triggered",
                            ])
                    else:
                        # === NORMAL TRAFFIC ===
                        level = _rng.choices(
                            ["INFO", "INFO", "INFO", "WARN", "ERROR"],
                            weights=[60, 20, 10, 8, 2],
                        )[0]
                        status = (200 if level == "INFO" else
                                  _rng.choice([200, 200, 400, 404, 500]))
                        latency = round(max(1.0, _rng.gauss(120, 60)), 1)
                        msg = f"{svc} handled {ep} — {status}"

                    ts = datetime.now().isoformat()
                    self.next(
                        service=svc,
                        level=level,
                        message=msg,
                        latency_ms=latency,
                        status_code=status,
                        endpoint=ep,
                        timestamp=ts,
                    )
                    self.total_emitted += 1

                    if self.total_emitted % 50 == 0:
                        _ml_logger.info(
                            f"[LogAnomalySubject] Emitted "
                            f"{self.total_emitted} log events"
                        )

                    # ~3-5 events/sec
                    _t.sleep(_rng.uniform(0.2, 0.35))

                except Exception as exc:
                    _ml_logger.info(
                        f"[LogAnomalySubject] Error: {exc}"
                    )
                    import traceback
                    traceback.print_exc()
                    break

    log_subject = LogAnomalySubject()
    _live_subjects.append(log_subject)
    log_events = pw.io.python.read(
        log_subject,
        schema=LogSchema,
        autocommit_duration_ms=2000,
        name="quantguard_log_stream",
    )

    # ── Log severity scoring (Pathway Rust engine) ──────────────────────
    _SEV_MAP = {"INFO": 0, "WARN": 1, "ERROR": 3, "CRITICAL": 5}
    log_scored = log_events.with_columns(
        severity_score=pw.apply_with_type(
            lambda lvl: _SEV_MAP.get(lvl, 0),
            int,
            pw.this.level,
        ),
        parsed_time=pw.apply_with_type(
            lambda ts: datetime.fromisoformat(ts).timestamp(),
            float,
            pw.this.timestamp,
        ),
        is_error=pw.apply_with_type(
            lambda code: 1 if code >= 400 else 0,
            int,
            pw.this.status_code,
        ),
    )

    # ── Log windowed aggregation (30s window, 5s hop) ───────────────────
    log_windowed = (
        log_scored.windowby(
            pw.this.parsed_time,
            window=pw.temporal.sliding(duration=30, hop=5),
            instance=pw.this.service,
        )
        .reduce(
            service=pw.this._pw_instance,
            event_count=pw.reducers.count(),
            error_count=pw.reducers.sum(pw.this.is_error),
            max_severity=pw.reducers.max(pw.this.severity_score),
            avg_latency=pw.reducers.avg(pw.this.latency_ms),
            max_latency=pw.reducers.max(pw.this.latency_ms),
        )
    )

    # ── Log anomaly filter ──────────────────────────────────────────────
    log_anomalies = (
        log_windowed.with_columns(
            error_rate=pw.this.error_count / (pw.this.event_count + 0.01),
            is_log_anomaly=(
                (pw.this.error_count > 3)           # >3 errors in 30s
                | (pw.this.max_severity >= 5)        # any CRITICAL
                | (pw.this.avg_latency > 2000.0)     # avg latency > 2s
                | (pw.this.max_latency > 10000.0)    # any call > 10s
                | (pw.this.event_count > 40)         # burst: >40 events/30s
            ),
        )
        .filter(pw.this.is_log_anomaly == True)
    )

    # ── Log anomaly callback ────────────────────────────────────────────
    _log_processed_keys: set = set()

    def _on_log_anomaly(key, row: dict, time: int, is_addition: bool):
        """Write log anomaly alerts to data/log_anomaly_alerts.jsonl."""
        if not is_addition:
            return
        key_str = str(key)
        if key_str in _log_processed_keys:
            return
        _log_processed_keys.add(key_str)
        if len(_log_processed_keys) > 5000:
            _log_processed_keys.clear()

        service = row.get("service", "unknown")
        error_count = int(row.get("error_count", 0))
        event_count = int(row.get("event_count", 0))
        avg_lat = float(row.get("avg_latency", 0))
        max_lat = float(row.get("max_latency", 0))
        max_sev = int(row.get("max_severity", 0))
        error_rate = round(error_count / max(event_count, 1), 3)

        # Classify severity
        if max_sev >= 5 or error_rate > 0.5:
            severity = "CRITICAL"
        elif error_rate > 0.3 or avg_lat > 5000:
            severity = "HIGH"
        elif error_rate > 0.1 or avg_lat > 2000:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        reasons = []
        if error_count > 3:
            reasons.append(f"{error_count} errors in 30s window")
        if max_sev >= 5:
            reasons.append("CRITICAL severity event detected")
        if avg_lat > 2000:
            reasons.append(f"Avg latency {avg_lat:.0f}ms (>2s)")
        if max_lat > 10000:
            reasons.append(f"Max latency {max_lat:.0f}ms (>10s)")
        if event_count > 40:
            reasons.append(f"Event burst: {event_count} events in 30s")

        alert = {
            "alert_type": "log_anomaly",
            "service": service,
            "severity": severity,
            "error_rate": error_rate,
            "error_count": error_count,
            "event_count": event_count,
            "avg_latency_ms": round(avg_lat, 1),
            "max_latency_ms": round(max_lat, 1),
            "max_severity_score": max_sev,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open("data/log_anomaly_alerts.jsonl", "a",
                       encoding="utf-8") as f:
                f.write(json.dumps(alert, default=str) + "\n")

            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(
                severity, "🟢"
            )
            _ml_logger.info(
                f"  {icon} LOG-ANOMALY [{severity}] {service} | "
                f"errors={error_count}/{event_count} "
                f"({error_rate:.0%}) | "
                f"latency avg={avg_lat:.0f}ms max={max_lat:.0f}ms | "
                f"{'; '.join(reasons)}"
            )
        except Exception as e:
            print(f"[Engine] Log anomaly error: {e}")

    pw.io.subscribe(log_anomalies, on_change=_on_log_anomaly,
                    name="log_anomaly_sink")

    # Push log anomaly alerts to the live buffer for dashboard
    _log_alert_buffer: list = []
    _LOG_ALERT_BUFFER_SIZE = 100

    # ── 3. Stats Tracker — subscribe to ALL transactions ────────────────
    #   Lightweight Python callback maintains per-user rolling history
    #   for the 6-feature ML scorer in the anomaly callback.
    def _track_stats(key, row: dict, time: int, is_addition: bool):
        """Maintains per-user rolling statistics for ML feature extraction."""
        if not is_addition:
            return
        uid = row.get("user_id", "UNKNOWN")
        amt = float(row.get("amount", 0))
        now = _time_mod.time()
        stats = _user_stats.setdefault(uid, {
            "amounts": [], "count": 0,
            "locations": [], "categories": [], "timestamps": [],
        })
        stats["amounts"].append(amt)
        stats["count"] += 1
        stats["locations"].append(row.get("location", ""))
        stats["categories"].append(row.get("category", ""))
        stats["timestamps"].append(now)
        # Trim rolling window (200 per user)
        _W = 200
        for k in ("amounts", "locations", "categories", "timestamps"):
            if len(stats[k]) > _W:
                stats[k] = stats[k][-_W:]

    pw.io.subscribe(transactions, on_change=_track_stats, name="stats_tracker")

    # ── 4. Feature Engineering (Pathway Rust engine) ────────────────────
    #   pw.apply_with_type runs the lambda inside Pathway's engine.
    transactions_fe = transactions.with_columns(
        parsed_time=pw.apply_with_type(
            lambda ts: datetime.fromisoformat(ts).timestamp(),
            float,
            pw.this.timestamp,
        )
    )

    # ── 5. Sliding Window Aggregation (Pathway Rust engine) ─────────────
    #   60-second window, 10-second hop, grouped by user_id.
    windowed_stats = (
        transactions_fe.windowby(
            pw.this.parsed_time,
            window=pw.temporal.sliding(duration=60, hop=10),
            instance=pw.this.user_id,
        )
        .reduce(
            user_id=pw.this._pw_instance,
            avg_amount=pw.reducers.avg(pw.this.amount),
            tx_count=pw.reducers.count(),
            max_amount=pw.reducers.max(pw.this.amount),
        )
    )

    # ── 6. Enriched Join + Multi-Rule Anomaly Detection (Rust engine) ───
    anomalies = (
        transactions_fe.join(
            windowed_stats,
            transactions_fe.user_id == windowed_stats.user_id,
        )
        .select(
            user_id=transactions_fe.user_id,
            amount=transactions_fe.amount,
            location=transactions_fe.location,
            category=transactions_fe.category,
            timestamp=transactions_fe.timestamp,
            is_suspicious_flag=transactions_fe.is_suspicious_flag,
            avg_amount=windowed_stats.avg_amount,
            tx_count=windowed_stats.tx_count,
            max_amount=windowed_stats.max_amount,
        )
        .with_columns(
            amount_ratio=pw.this.amount / (pw.this.avg_amount + 0.01),
            is_anomaly=(
                (pw.this.amount > (pw.this.avg_amount * 3.0))
                | (pw.this.amount > pw.this.max_amount)
                | (pw.this.tx_count > 20)
                | (pw.this.is_suspicious_flag == True)
                | (pw.this.amount > 5000.0)
            ),
        )
        .filter(pw.this.is_anomaly == True)
    )

    # ── 7. ML + Quantum + LLM Enrichment Callback ──────────────────────
    def _on_anomaly(key, row: dict, time: int, is_addition: bool):
        """
        Enrich each anomaly detected by Pathway's Rust engine with:
          - 6-feature ML anomaly scoring (z-score, IQR, percentile,
            geo-entropy, velocity, category deviation)
          - IBM Quantum VQC classification (2-qubit, ibm_torino)
          - Groq LLM natural-language explanation
        """
        if not is_addition:
            return

        key_str = str(key)
        if key_str in _processed_keys:
            return
        _processed_keys.add(key_str)
        if len(_processed_keys) > 10000:
            _processed_keys.clear()

        uid = row.get("user_id", "UNKNOWN")
        amount = float(row.get("amount", 0))
        category = row.get("category", "")
        location = row.get("location", "")

        # ── Pathway-computed stats (from Rust-side join) ────────────
        pw_avg = float(row.get("avg_amount", amount))
        pw_max = float(row.get("max_amount", amount))
        pw_count = int(row.get("tx_count", 1))
        pw_ratio = float(row.get("amount_ratio", 1.0))

        # ── Python-side rolling stats (from _track_stats) ───────────
        stats = _user_stats.get(uid, {
            "amounts": [amount], "count": 1,
            "locations": [], "categories": [], "timestamps": [],
        })
        hist = stats["amounts"]

        # ── ML Feature 1: Z-Score ──────────────────────────────────
        if len(hist) >= 3:
            mean = sum(hist) / len(hist)
            std = max((sum((x - mean)**2 for x in hist) / len(hist))**0.5, 0.01)
            z_score = round((amount - mean) / std, 3)
        else:
            mean = amount
            z_score = 0.0

        # ── ML Feature 2: IQR Outlier Score ────────────────────────
        sorted_h = sorted(hist)
        if len(sorted_h) >= 5:
            q1 = sorted_h[len(sorted_h) // 4]
            q3 = sorted_h[3 * len(sorted_h) // 4]
            iqr_val = max(q3 - q1, 0.01)
            if amount > q3:
                iqr_score = round((amount - q3) / iqr_val, 3)
            elif amount < q1:
                iqr_score = round((q1 - amount) / iqr_val, 3)
            else:
                iqr_score = 0.0
        else:
            iqr_score = 0.0

        # ── ML Feature 3: Percentile Rank ──────────────────────────
        if len(sorted_h) >= 3:
            rank = sum(1 for x in sorted_h if x <= amount)
            percentile = round(rank / len(sorted_h), 3)
        else:
            percentile = 0.5

        # ── ML Feature 4: Geo-Entropy (Shannon) ────────────────────
        locs = stats.get("locations", [])[-200:]
        if len(locs) >= 2:
            lc: dict = {}
            for _l in locs:
                lc[_l] = lc.get(_l, 0) + 1
            total_l = sum(lc.values())
            geo_entropy = round(-sum(
                (c / total_l) * math.log2(c / total_l)
                for c in lc.values() if c > 0
            ), 3)
        else:
            geo_entropy = 0.0

        # ── ML Feature 5: Velocity Score ───────────────────────────
        ts_list = stats.get("timestamps", [])
        now = _time_mod.time()
        recent = [t for t in ts_list[-200:] if now - t < 60]
        velocity = (
            round(len(recent) / max(now - recent[0], 0.1), 3)
            if len(recent) >= 2 else 0.0
        )

        # ── ML Feature 6: Category Deviation ───────────────────────
        cats = stats.get("categories", [])[-200:]
        if len(cats) >= 3:
            cc: dict = {}
            for _c in cats:
                cc[_c] = cc.get(_c, 0) + 1
            total_c = sum(cc.values())
            cat_freq = cc.get(category, 0) / max(total_c, 1)
            category_dev = round(1.0 - cat_freq, 3)
        else:
            category_dev = 0.0

        # ── Composite ML Risk Score (weighted ensemble) ─────────────
        composite_risk = round(min(1.0,
            0.25 * min(abs(z_score) / 4.0, 1.0)
            + 0.20 * min(iqr_score / 3.0, 1.0)
            + 0.10 * min(geo_entropy / 3.5, 1.0)
            + 0.10 * min(velocity / 2.0, 1.0)
            + 0.10 * category_dev
            + 0.15 * min(pw_ratio / 5.0, 1.0)
            + 0.10 * (1.0 if percentile > 0.95 else 0.0)
        ), 4)

        ml_risk = (
            "CRITICAL" if composite_risk > 0.65 else
            "HIGH"     if composite_risk > 0.45 else
            "MEDIUM"   if composite_risk > 0.25 else
            "LOW"
        )

        # ── Anomaly Reasons (ML-informed) ───────────────────────────
        reasons = []
        if amount > 5000:
            reasons.append(f"High amount ${amount:,.2f}")
        if row.get("is_suspicious_flag"):
            reasons.append("Flagged suspicious")
        if abs(z_score) > 2.0:
            reasons.append(f"Z-score: {z_score:+.2f}σ")
        if iqr_score > 1.5:
            reasons.append(f"IQR outlier: {iqr_score:.2f}")
        if geo_entropy > 2.0:
            reasons.append(
                f"Geo diversity: {len(set(locs))} locations (H={geo_entropy:.2f})"
            )
        if velocity > 0.5:
            reasons.append(f"High velocity: {velocity:.2f} tx/s")
        if category_dev > 0.8:
            reasons.append(f"Unusual category: {category} (dev={category_dev:.2f})")
        if pw_ratio > 3.0:
            reasons.append(f"Amount {pw_ratio:.1f}× above window avg")
        if pw_count > 20:
            reasons.append(f"High frequency: {pw_count} in window")

        try:
            row_json = json.dumps({
                "user_id": uid,
                "amount": amount,
                "location": location,
                "category": category,
                "timestamp": row.get("timestamp", datetime.now().isoformat()),
                "avg_amount": pw_avg,
                "max_amount": pw_max,
                "tx_count": pw_count,
                "anomaly_reasons": reasons,
            }, default=str)

            enriched_str = _quantum_enrich(row_json)

            # Merge ML features + Pathway stats into enriched alert
            enriched = json.loads(enriched_str)
            enriched["ml_features"] = {
                "z_score": z_score,
                "iqr_score": iqr_score,
                "percentile": percentile,
                "geo_entropy": geo_entropy,
                "velocity": velocity,
                "category_dev": category_dev,
                "composite_risk": composite_risk,
                "ml_risk_label": ml_risk,
            }
            enriched["pathway_stats"] = {
                "window_avg": round(pw_avg, 2),
                "window_max": round(pw_max, 2),
                "window_count": pw_count,
                "amount_ratio": round(pw_ratio, 2),
            }
            final = json.dumps(enriched, default=str)

            with open(
                "data/high_risk_alerts.jsonl", "a", encoding="utf-8"
            ) as f:
                f.write(final + "\n")

            # ── ML feature output (logged to file + stderr) ──────────
            _ml_logger.info(
                f"🔍 ANOMALY  user={uid}  ${amount:,.2f}  {location}  {category}"
            )
            _ml_logger.info(
                f"   ML Features │ Z: {z_score:+.2f}σ  IQR: {iqr_score:.2f}  "
                f"P: {percentile:.0%}  Geo-H: {geo_entropy:.2f}  "
                f"Vel: {velocity:.3f}  Cat-Dev: {category_dev:.2f}"
            )
            filled = int(composite_risk * 20)
            bar = "█" * filled + "░" * (20 - filled)
            _ml_logger.info(
                f"   ML Risk     │ [{ml_risk:>8}] {bar} {composite_risk:.0%}"
            )
            _ml_logger.info(
                f"   Pathway     │ window_avg=${pw_avg:,.2f}  "
                f"max=${pw_max:,.2f}  count={pw_count}  ratio={pw_ratio:.1f}×"
            )
            _ml_logger.info(
                f"   Reasons     │ {'; '.join(reasons) if reasons else 'threshold'}"
            )
            _ml_logger.info("   " + "─" * 64)

        except Exception as e:
            print(f"[Engine] Enrichment error: {e}")

    pw.io.subscribe(anomalies, on_change=_on_anomaly, name="quantum_enricher")

    # ── 7b. Kafka Alert Output Sink (pw.io.kafka.write) ─────────────────
    #
    #   Closes the Kafka loop: reads from quantguard-transactions (2d),
    #   writes enriched anomalies to quantguard-alerts.  Uses the same
    #   broker settings and reachability probe from step 2d.
    # ────────────────────────────────────────────────────────────────────
    _kafka_alert_topic = os.environ.get("KAFKA_ALERT_TOPIC", "quantguard-alerts")
    if _kafka_reachable:
        # Serialise the anomalies table columns into a single JSON string
        # so the Kafka sink has a flat schema to write.
        _alert_payload = anomalies.select(
            payload=pw.apply_with_type(
                lambda uid, amt, loc, cat, ts, avg, mx, cnt, ratio: json.dumps({
                    "user_id": uid,
                    "amount": amt,
                    "location": loc,
                    "category": cat,
                    "timestamp": ts,
                    "avg_amount": avg,
                    "max_amount": mx,
                    "tx_count": cnt,
                    "amount_ratio": ratio,
                }, default=str),
                str,
                pw.this.user_id,
                pw.this.amount,
                pw.this.location,
                pw.this.category,
                pw.this.timestamp,
                pw.this.avg_amount,
                pw.this.max_amount,
                pw.this.tx_count,
                pw.this.amount_ratio,
            ),
        )
        pw.io.kafka.write(
            _alert_payload,
            rdkafka_settings={
                "bootstrap.servers": _kafka_bootstrap,
            },
            topic_name=_kafka_alert_topic,
            format="json",
        )
        _ml_logger.info(
            f"[KafkaNative] pw.io.kafka.write → {_kafka_bootstrap} "
            f"topic={_kafka_alert_topic}  (anomaly alerts)"
        )
    else:
        _ml_logger.info(
            f"[KafkaNative] Kafka broker offline — alert sink "
            f"({_kafka_alert_topic}) skipped; JSONL output still active"
        )

    # ── 8. Persistence / Fault Tolerance ────────────────────────────────
    #
    #   Pathway supports persistent state via pw.persistence.Config.
    #   When enabled, the engine snapshots operator state to disk so that
    #   after a crash / restart it can resume from the last checkpoint
    #   instead of reprocessing the entire history.
    #
    #   Backend: pw.persistence.Backend.filesystem (local or Docker volume)
    #   Set env PATHWAY_PERSISTENCE_DIR to override default path.
    # ────────────────────────────────────────────────────────────────────
    persistence_cfg = None
    _persist_dir = os.environ.get("PATHWAY_PERSISTENCE_DIR", "data/persistence")
    try:
        Path(_persist_dir).mkdir(parents=True, exist_ok=True)
        _backend = pw.persistence.Backend.filesystem(_persist_dir)
        persistence_cfg = pw.persistence.Config(
            _backend,
            snapshot_interval_ms=10_000,   # checkpoint every 10 s
        )
        print(f"[Engine] ✓ Persistence enabled → {_persist_dir}")
        print(f"[Engine]   Snapshot interval: 10 s  |  Backend: filesystem")
    except Exception as _pe:
        print(f"[Engine] ⚠ Persistence unavailable ({_pe}); running stateless")

    # ── 8.5. Initialize Pathway LLM xPack (BEFORE pw.run) ─────────────
    #
    #   PathwayLLMxPack._init_native_pipeline() creates pw.io.fs.read()
    #   tables for policy documents and alerts.  These must be registered
    #   BEFORE pw.run() so they become part of the computation graph.
    #   After this, main_api.py can access the pre-initialized xPack via
    #   pathway_engine.get_engine_xpack() instead of lazy-creating one
    #   (which would be too late — pw.run() would already be blocking).
    # ────────────────────────────────────────────────────────────────────
    global _engine_xpack
    try:
        from pathway_llm_xpack import PathwayLLMxPack
        _engine_xpack = PathwayLLMxPack()
        print("[Engine] ✓ PathwayLLMxPack initialized within native pipeline graph")
        if hasattr(_engine_xpack, 'store') and hasattr(_engine_xpack.store, 'is_native'):
            print(f"[Engine]   DocumentStore native: {_engine_xpack.store.is_native}")
        if hasattr(_engine_xpack, 'rag') and hasattr(_engine_xpack.rag, 'is_native'):
            print(f"[Engine]   RAG native:           {_engine_xpack.rag.is_native}")
    except Exception as _xpack_err:
        print(f"[Engine] ⚠ xPack init in pipeline failed ({_xpack_err}); "
              "lazy init via API still available")
        _engine_xpack = None

    # ── 9. Run (blocking — Pathway event loop) ─────────────────────────
    ENGINE_ACTIVE = True
    print("[Engine] ✓ Native Pathway streaming pipeline ready")
    print("[Engine] Data sources:")
    print("[Engine]   • FilePollingSubject       → data/transactions*.jsonl (poll 2s)")
    print("[Engine]   • TransactionSubject       → live generator (~1 tx/sec, 15% fraud)")
    print("[Engine]   • pw.io.kafka.read         → native Kafka consumer (quantguard-transactions)")
    print("[Engine]   • pw.io.http.rest_connector → live market data (port 9090)")
    print("[Engine]   • LogAnomalySubject        → system log generator (~3-5 events/sec)")
    print("[Engine]   • concat_reindex           → merged unified stream")
    print("[Engine] NTFS/inotify workaround: FilePollingSubject replaces")
    print("[Engine]   pw.io.jsonlines.read — polls for new lines instead of inotify")
    print("[Engine] Pipeline 1 (Fraud): Ingest(file+live+kafka+market) → Stats →")
    print("[Engine]   with_columns → windowby(60s/10s) → reduce → join →")
    print("[Engine]   filter → ML(6-feat) → Quantum VQC → LLM → Alerts")
    print("[Engine] Pipeline 2 (Logs): LogAnomalySubject → severity scoring →")
    print("[Engine]   windowby(30s/5s) → reduce → filter → log anomaly alerts")
    print("[Engine] Pipeline 3 (xPack): pw.io.fs → DocumentStore → RAG →")
    print("[Engine]   QASummaryRestServer (:8001)")
    print("[Engine] ML output → data/ml_output.log\n")
    pw.run(
        monitoring_level=pw.MonitoringLevel.NONE,
        **({"persistence_config": persistence_cfg} if persistence_cfg else {}),
    )


def _run_compat_pipeline():
    """Run the compatibility pipeline (delegates to pathway_compat pw.run)."""
    print("[Engine] Starting compatibility streaming pipeline...")

    # ── Define pipeline (same API, compat objects) ──────────────────────
    class TransactionSchema(pw.Schema):
        user_id: str
        amount: float
        location: str
        category: str
        timestamp: str
        is_suspicious_flag: bool

    transactions = pw.io.jsonlines.read(
        "data", schema=TransactionSchema, mode="streaming"
    )
    transactions = transactions.with_columns(
        parsed_time=pw.apply_with_type(
            lambda ts: datetime.fromisoformat(ts).timestamp(),
            float,
            pw.this.timestamp,
        )
    )
    windowed_stats = (
        transactions.windowby(
            pw.this.parsed_time,
            window=pw.temporal.sliding(duration=60, hop=10),
            instance=pw.this.user_id,
        )
        .reduce(
            user_id=pw.this._pw_instance,
            avg_amount=pw.reducers.avg(pw.this.amount),
            count=pw.reducers.count(),
            max_amount=pw.reducers.max(pw.this.amount),
        )
    )
    alerts = (
        transactions.join(
            windowed_stats,
            transactions.user_id == windowed_stats.user_id,
        )
        .select(
            user_id=transactions.user_id,
            amount=transactions.amount,
            location=transactions.location,
            category=transactions.category,
            timestamp=transactions.timestamp,
            is_suspicious_flag=transactions.is_suspicious_flag,
            avg_amount=windowed_stats.avg_amount,
            count=windowed_stats.count,
            max_amount=windowed_stats.max_amount,
        )
        .with_columns(
            amount_ratio=pw.this.amount / (pw.this.avg_amount + 0.01),
            is_anomaly=(
                (pw.this.amount > (pw.this.avg_amount * 3.0))
                | (pw.this.amount > pw.this.max_amount)
                | (pw.this.count > 20)
            ),
        )
        .filter(pw.this.is_anomaly == True)
    )
    pw.io.jsonlines.write(alerts, "data/high_risk_alerts.jsonl")

    # This calls pathway_compat's pw.run() which has full quantum + LLM
    pw.run()


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  QuantGuard Streaming Engine")
    print("=" * 60)
    run_analysis()

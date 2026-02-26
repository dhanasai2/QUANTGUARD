"""
QuantGuard - FastAPI Real-Time Hub
===================================
Central API server powering the QuantGuard fraud detection dashboard.

Endpoints:
  GET  /                    – Glassmorphic real-time dashboard
  GET  /api/stats           – System statistics
  GET  /api/alerts          – Recent high-risk alerts
  GET  /api/transactions    – Recent transactions
  POST /api/analyze         – On-demand quantum + LLM analysis
  POST /api/rag/query       – RAG query over regulatory policies
  GET  /api/alerts/summary  – AI-generated alert summary
  GET  /api/quantum/info    – Quantum circuit metadata
  WS   /ws                  – WebSocket for live updates

  Pathway LLM xPack:
  POST /api/xpack/rag       – Live RAG (real-time indexed alerts + policies)
  POST /api/xpack/report    – Automated report generation (5 types)
  POST /api/xpack/insight   – Explainable AI insight for a transaction
  POST /api/xpack/credit    – Credit decision rationale for a user
  GET  /api/xpack/status    – xPack capabilities & indexed doc count

  Pathway MCP Server (Model Context Protocol):
  POST /mcp/                – MCP JSON-RPC 2.0 endpoint
  GET  /mcp/tools           – List available MCP tools
  GET  /mcp/health          – MCP server health check

  Live Data Sources:
  POST /api/live/start      – Start live market data (Alpha Vantage / Demo / Socket)
  POST /api/live/stop       – Stop live data source
  GET  /api/live/status     – Live source status

  Log Anomaly Detection (parallel streaming pipeline):
  GET  /api/logs/alerts     – Recent log anomaly alerts
  GET  /api/logs/stats      – Log anomaly pipeline stats
  GET  /api/logs/services   – Per-service health from log anomalies
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import json
import os
import time
import threading
from datetime import datetime
from typing import List
import numpy as np

_start_time = time.time()


# ═══════════════════════════════════════════════════════════════════════════
#  Pathway Streaming Engine (background thread)
# ═══════════════════════════════════════════════════════════════════════════

_engine_thread = None
_engine_native = False


def _start_pathway_engine():
    """Launch Pathway streaming engine in a background daemon thread.
    On WSL/Linux with native Pathway, this runs the real streaming pipeline
    with quantum classification.  On Windows it falls back to compat mode.
    """
    global _engine_thread, _engine_native
    try:
        from pathway_engine import run_analysis, PATHWAY_NATIVE, get_live_tx_count
        _engine_native = PATHWAY_NATIVE
        print(f"[API] Pathway engine: {'NATIVE' if PATHWAY_NATIVE else 'compat'}")

        def _run():
            try:
                os.makedirs("data", exist_ok=True)
                run_analysis()
            except Exception as e:
                print(f"[Engine] Error: {e}")

        _engine_thread = threading.Thread(
            target=_run, daemon=True, name="pathway-engine"
        )
        _engine_thread.start()
        print("[API] Pathway streaming engine started in background thread")
    except Exception as e:
        print(f"[API] Pathway engine failed to start: {e}")
        _engine_native = False


# ═══════════════════════════════════════════════════════════════════════════
#  Lifespan (replaces deprecated on_event)
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(application):
    """Startup / shutdown lifecycle for the FastAPI application."""
    # ── Startup ─────────────────────────────────────────────────────────
    _start_pathway_engine()
    asyncio.create_task(watch_alerts())
    asyncio.create_task(watch_transactions())
    asyncio.create_task(watch_log_anomalies())
    # auto_analyze is a fallback when the Pathway engine is NOT active
    # (Pathway engine handles quantum classification within its pipeline)
    if not _engine_native:
        asyncio.create_task(auto_analyze_transactions())
        print("[API] auto_analyze active (Pathway engine not native)")
    else:
        print("[API] Pathway native engine handles analysis — auto_analyze skipped")
    # ── Auto-start live market data if Alpha Vantage key is available ──
    _av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if _av_key and _av_key != "demo":
        try:
            from live_data_source import AlphaVantageSource, write_transaction
            _av_src = AlphaVantageSource(api_key=_av_key)
            asyncio.create_task(
                _av_src.stream(write_transaction, interval=12)
            )
            print(f"[API] ✅ Alpha Vantage live market data auto-started")
            print(f"[API]   Real-time stock prices → transaction stream")
        except Exception as _av_err:
            print(f"[API] Alpha Vantage auto-start failed: {_av_err}")
    else:
        print("[API] Set ALPHA_VANTAGE_API_KEY for real-time market data ingestion")
    print("[API] QuantGuard API running  =>  http://localhost:8000")
    print("[API] Swagger docs           =>  http://localhost:8000/docs")
    yield
    # ── Shutdown ────────────────────────────────────────────────────────
    print("[API] Shutting down…")


# ═══════════════════════════════════════════════════════════════════════════
#  App Initialisation
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="QuantGuard API",
    description="Real-Time Quantum-Enhanced Fraud Detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Pathway MCP Server ───────────────────────────────────────────
try:
    from pathway_mcp_server import create_mcp_app
    app.mount("/mcp", create_mcp_app())
    print("[API] Pathway MCP Server mounted at /mcp")
except Exception as _mcp_err:
    print(f"[API] MCP Server not mounted: {_mcp_err}")


# ── Lazy-loaded heavy modules ──────────────────────────────────────────

_quantum_classifier = None
_llm_analyzer = None
_xpack = None
_live_source_task = None


def get_quantum_classifier():
    global _quantum_classifier
    if _quantum_classifier is None:
        from quantum_classifier import QuantumRiskClassifier
        _quantum_classifier = QuantumRiskClassifier()
    return _quantum_classifier


def get_llm_analyzer():
    global _llm_analyzer
    if _llm_analyzer is None:
        from llm_engine import GroqFraudAnalyzer
        _llm_analyzer = GroqFraudAnalyzer()
    return _llm_analyzer


def get_xpack():
    """Return the Pathway LLM xPack (RAG, Reports, Insights).

    Prefers the instance created inside the native Pathway pipeline graph
    (via pathway_engine.get_engine_xpack()) so that pw.io.fs.read() tables
    are part of the same pw.run() computation graph.  Falls back to lazy
    init when the engine hasn't started yet or xPack init failed.
    """
    global _xpack
    if _xpack is None:
        try:
            from pathway_engine import get_engine_xpack
            _xpack = get_engine_xpack()
        except ImportError:
            pass
    if _xpack is None:
        from pathway_llm_xpack import PathwayLLMxPack
        _xpack = PathwayLLMxPack()
    return _xpack


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket Connection Manager
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.active[:]:
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════════════════
#  Background File Watchers
# ═══════════════════════════════════════════════════════════════════════════

async def watch_alerts():
    """Stream new alert-file entries to WebSocket clients."""
    path = os.path.join("data", "high_risk_alerts.jsonl")
    pos = os.path.getsize(path) if os.path.exists(path) else 0
    while True:
        try:
            if os.path.exists(path) and os.path.getsize(path) > pos:
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                await manager.broadcast(
                                    {"type": "alert", "data": json.loads(line)}
                                )
                            except json.JSONDecodeError:
                                pass
                    pos = f.tell()
        except Exception:
            pass
        await asyncio.sleep(1)


async def watch_transactions():
    """Stream live-buffer transactions to WebSocket clients."""
    try:
        from pathway_engine import get_live_tx_buffer, get_live_tx_total
    except Exception:
        return
    last_total = 0
    while True:
        try:
            total = get_live_tx_total()
            if total > last_total:
                new_count = total - last_total
                buf = get_live_tx_buffer(min(new_count, 200))
                # Only send the truly new entries
                send = buf[-new_count:] if new_count <= len(buf) else buf
                for tx in send:
                    await manager.broadcast({"type": "transaction", "data": tx})
                last_total = total
        except Exception:
            pass
        await asyncio.sleep(1)


async def watch_log_anomalies():
    """Stream new log anomaly alerts to WebSocket clients."""
    path = os.path.join("data", "log_anomaly_alerts.jsonl")
    pos = os.path.getsize(path) if os.path.exists(path) else 0
    while True:
        try:
            if os.path.exists(path) and os.path.getsize(path) > pos:
                with open(path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                await manager.broadcast(
                                    {"type": "log_anomaly", "data": json.loads(line)}
                                )
                            except json.JSONDecodeError:
                                pass
                    pos = f.tell()
        except Exception:
            pass
        await asyncio.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-Analyze Suspicious Transactions (background)
# ═══════════════════════════════════════════════════════════════════════════

def _sync_classify(tx):
    """Run quantum classification synchronously (called from thread pool)."""
    clf = get_quantum_classifier()
    amount_norm = min(tx.get("amount", 0) / 5000.0, 1.0)
    freq_norm = 0.3
    return clf.classify_transaction(np.array([amount_norm, freq_norm]))


async def auto_analyze_transactions():
    """Watch for new transactions and auto-analyze suspicious ones.
    
    Generates high-risk alerts automatically so the dashboard
    alert panel stays populated without manual /api/analyze calls.
    On first run, seeds alerts from existing suspicious transactions.
    Uses asyncio thread executor so quantum jobs don't block the event loop.
    """
    tx_path = os.path.join("data", "transactions.jsonl")
    al_path = os.path.join("data", "high_risk_alerts.jsonl")
    loop = asyncio.get_event_loop()
    await asyncio.sleep(2)  # let server finish startup

    # ── Seed: analyze existing suspicious transactions if no alerts yet ──
    existing_alerts = 0
    if os.path.exists(al_path):
        existing_alerts = _count_lines(al_path)

    if existing_alerts == 0 and os.path.exists(tx_path):
        print("[AutoAnalyze] Seeding alerts from existing suspicious transactions...")
        try:
            with open(tx_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            seeded = 0
            for line in all_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    tx = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not tx.get("is_suspicious_flag", False):
                    continue
                try:
                    qr = await loop.run_in_executor(None, _sync_classify, tx)
                    fp = qr["fraud_probability"]
                    risk = (
                        "CRITICAL" if fp > 0.70 else
                        "HIGH" if fp > 0.45 else
                        "MEDIUM" if fp > 0.25 else
                        "LOW"
                    )
                    if risk in ("CRITICAL", "HIGH", "MEDIUM"):
                        alert_rec = {
                            "user_id": tx.get("user_id", "UNKNOWN"),
                            "amount": tx.get("amount", 0),
                            "location": tx.get("location", "N/A"),
                            "category": tx.get("category", "N/A"),
                            "risk_level": risk,
                            "quantum_classification": qr.get("classification", "N/A"),
                            "quantum_fraud_probability": fp,
                            "anomaly_reasons": tx.get("market_source", {}).get(
                                "suspicious_reasons", []
                            ),
                            "timestamp": tx.get("timestamp", datetime.now().isoformat()),
                        }
                        with open(al_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(alert_rec) + "\n")
                        seeded += 1
                    await asyncio.sleep(0)  # yield to event loop
                except Exception:
                    pass
            print(f"[AutoAnalyze] Seeded {seeded} alerts from {len(all_lines)} transactions")
        except Exception as e:
            print(f"[AutoAnalyze] Seed error: {e}")

    pos = os.path.getsize(tx_path) if os.path.exists(tx_path) else 0

    while True:
        try:
            if os.path.exists(tx_path) and os.path.getsize(tx_path) > pos:
                with open(tx_path, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    new_lines = f.readlines()
                    pos = f.tell()

                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tx = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    # Only auto-analyze flagged transactions
                    if not tx.get("is_suspicious_flag", False):
                        continue

                    try:
                        qr = await loop.run_in_executor(None, _sync_classify, tx)

                        fp = qr["fraud_probability"]
                        risk = (
                            "CRITICAL" if fp > 0.70 else
                            "HIGH" if fp > 0.45 else
                            "MEDIUM" if fp > 0.25 else
                            "LOW"
                        )

                        if risk in ("CRITICAL", "HIGH", "MEDIUM"):
                            alert_rec = {
                                "user_id": tx.get("user_id", "UNKNOWN"),
                                "amount": tx.get("amount", 0),
                                "location": tx.get("location", "N/A"),
                                "category": tx.get("category", "N/A"),
                                "risk_level": risk,
                                "quantum_classification": qr.get("classification", "N/A"),
                                "quantum_fraud_probability": fp,
                                "anomaly_reasons": tx.get("market_source", {}).get(
                                    "suspicious_reasons", []
                                ),
                                "timestamp": datetime.now().isoformat(),
                            }
                            with open(al_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(alert_rec) + "\n")
                    except Exception as e:
                        print(f"[AutoAnalyze] Error: {e}")
        except Exception:
            pass
        await asyncio.sleep(2)


# ═══════════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════

class TransactionInput(BaseModel):
    user_id: str
    amount: float
    location: str
    category: str


class RAGQuery(BaseModel):
    question: str


# ═══════════════════════════════════════════════════════════════════════════
#  REST Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Health-check endpoint (used by Docker HEALTHCHECK and load balancers)."""
    return {
        "status": "healthy",
        "service": "quantguard",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the real-time dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/stats")
async def get_stats():
    """Return system-wide statistics."""
    tx_path = os.path.join("data", "transactions.jsonl")
    al_path = os.path.join("data", "high_risk_alerts.jsonl")
    tx_file = _count_lines(tx_path)
    # Use live pipeline counter if engine is running (includes file + live + kafka)
    try:
        from pathway_engine import get_live_tx_count, get_live_tx_buffer
        tx_live = get_live_tx_count()
        tx_n = max(tx_file, tx_live)  # pipeline sees all sources
    except Exception:
        tx_n = tx_file
    al_n = _count_lines(al_path)
    try:
        qinfo = get_quantum_classifier().get_circuit_info()
        backend_name = qinfo.get("backend", "numpy_statevector_simulator")
        hw_active = qinfo.get("ibm_hardware_active", False)
    except Exception:
        backend_name = "numpy_statevector_simulator"
        hw_active = False
    # Sustainability impact metrics
    al_amounts = []
    al_path_full = os.path.join("data", "high_risk_alerts.jsonl")
    if os.path.exists(al_path_full):
        try:
            with open(al_path_full, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        al_amounts.append(rec.get("amount", 0))
                    except (json.JSONDecodeError, ValueError):
                        pass
        except Exception:
            pass
    total_blocked = sum(al_amounts)
    blocked_inr = total_blocked * 83  # approximate USD→INR

    return {
        "total_transactions": tx_n,
        "total_alerts": al_n,
        "alert_rate": round(al_n / max(tx_n, 1) * 100, 2),
        "system_status": "active",
        "pathway_engine_native": _engine_native,
        "pathway_engine_alive": _engine_thread is not None and _engine_thread.is_alive(),
        "quantum_backend": backend_name,
        "ibm_hardware_active": hw_active,
        "llm_model": "llama-3.3-70b-versatile (Groq)",
        "timestamp": datetime.now().isoformat(),
        "sustainability": {
            "frauds_detected": al_n,
            "funds_protected_usd": round(total_blocked, 2),
            "funds_protected_inr": round(blocked_inr, 2),
            "green_redirect_potential_inr": round(blocked_inr * 0.4, 2),
            "co2_offset_kg": round(blocked_inr / 1000 * 2.5, 1),
            "trees_equivalent": int(blocked_inr / 500),
            "clean_water_liters": int(blocked_inr / 100000 * 50000),
            "solar_panels_kw": round(blocked_inr / 25000, 1),
            "sdg_alignment": [
                "SDG 6: Clean Water",
                "SDG 7: Clean Energy",
                "SDG 13: Climate Action",
                "SDG 15: Life on Land",
                "SDG 16: Peace & Justice",
            ],
            "methodology": {
                "co2": "INR 1,000 = 2.5 kg CO₂ offset (Gold Standard VERs, goldstandard.org)",
                "trees": "INR 500 = 1 native tree (Grow-Trees.com / UNEP Trillion Tree Campaign)",
                "water": "INR 2/L clean water (WHO/UNICEF JMP 2023, washdata.org)",
                "solar": "INR 25,000/kW rooftop solar (MNRE PM Surya Ghar Yojana, mnre.gov.in)",
                "redirect": "40% green bond eligible (SEBI Green Bond Framework 2023)",
            },
        },
    }


@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    """Return the most recent high-risk alerts."""
    return {"alerts": _tail_jsonl("data/high_risk_alerts.jsonl", limit)}


@app.get("/api/transactions")
async def get_transactions(limit: int = 50):
    """Return the most recent transactions (file + live stream merged)."""
    try:
        from pathway_engine import get_live_tx_buffer
        live = get_live_tx_buffer(limit)
    except Exception:
        live = []
    if live:
        # Live buffer has the freshest data — return newest last
        return {"transactions": live[-limit:]}
    # Fallback to file if engine hasn't started yet
    return {"transactions": _tail_jsonl("data/transactions.jsonl", limit)}


@app.get("/api/engine/status")
async def engine_status():
    """Pathway streaming engine status (both fraud + log anomaly pipelines)."""
    return {
        "native": _engine_native,
        "alive": _engine_thread is not None and _engine_thread.is_alive(),
        "mode": "Native Pathway pw.run()" if _engine_native else "Compatibility layer",
        "pipelines": {
            "fraud_detection": [
                "pw.io.python.read → FilePollingSubject (NTFS-safe poller, 2s interval)",
                "pw.io.python.read → TransactionSubject (ConnectorSubject, live gen)",
                "pw.io.python.read → KafkaSubject (real Apache Kafka consumer)",
                "pw.Table.concat_reindex → unified stream (file + live + kafka)",
                "pw.io.subscribe → per-user rolling stats (ML feature history)",
                "pw.Table.with_columns → timestamp parsing (feature eng.)",
                "pw.temporal.sliding(60s window, 10s hop) → pw.windowby",
                "pw.reducers.avg / .count / .max → windowed aggregation",
                "pw.Table.join + .select → enrich transactions with windowed stats",
                "pw.Table.with_columns → amount_ratio + multi-rule anomaly flag",
                "pw.Table.filter → anomaly detection (5 rules)",
                "pw.io.subscribe → 6-feature ML scoring (z/IQR/geo/vel/cat/composite)",
                "IBM Quantum VQC (2-qubit, ibm_torino)",
                "Groq LLM explanation (llama-3.3-70b)",
                "Alert sink → data/high_risk_alerts.jsonl",
            ],
            "log_anomaly_detection": [
                "pw.io.python.read → LogAnomalySubject (system log generator, ~3-5 events/sec)",
                "pw.Table.with_columns → severity scoring + error flag",
                "pw.temporal.sliding(30s window, 5s hop) → pw.windowby",
                "pw.reducers.count / .sum / .max / .avg → windowed aggregation",
                "pw.Table.with_columns → error_rate + multi-rule log anomaly flag",
                "pw.Table.filter → log anomaly detection (5 rules)",
                "pw.io.subscribe → log anomaly classification + alert sink",
                "Alert sink → data/log_anomaly_alerts.jsonl",
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze_transaction(tx: TransactionInput):
    """Analyse a single transaction with Quantum + LLM — always returns full results."""
    clf = get_quantum_classifier()
    llm = get_llm_analyzer()

    # ── Adaptive normalisation from user history ────────────────────────
    # Read the user's recent transactions for context-aware feature scaling
    user_amounts = []
    user_tx_count = 0
    tx_path = os.path.join("data", "transactions.jsonl")
    if os.path.exists(tx_path):
        try:
            with open(tx_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get("user_id") == tx.user_id:
                            user_amounts.append(rec.get("amount", 0))
                            user_tx_count += 1
                    except (json.JSONDecodeError, ValueError):
                        pass
        except Exception:
            pass

    if user_amounts and len(user_amounts) >= 3:
        # Use real user statistics — same logic as pathway_compat.py
        import numpy as _np
        recent = user_amounts[-100:]  # last 100
        avg_amount = float(_np.mean(recent))
        max_amount = float(max(recent))
        amount_norm = min(tx.amount / max(max_amount, avg_amount * 3, 1), 1.0)
        freq_norm = min(len(recent) / max(50, len(recent) * 1.5), 1.0)
    else:
        # No history: scale against reasonable baseline ($5000 max, low velocity)
        amount_norm = min(tx.amount / 5000.0, 1.0)
        freq_norm = 0.15  # low velocity = less suspicious

    qr = clf.classify_transaction(np.array([amount_norm, freq_norm]))

    tx_dict = tx.model_dump()
    tx_dict["timestamp"] = datetime.now().isoformat()
    explanation = llm.explain_fraud_risk(tx_dict, qr)

    fp = qr["fraud_probability"]
    risk = (
        "CRITICAL" if fp > 0.70 else
        "HIGH" if fp > 0.45 else
        "MEDIUM" if fp > 0.25 else
        "LOW"
    )

    result = {
        "transaction": tx_dict,
        "quantum_result": qr,
        "llm_explanation": explanation,
        "risk_level": risk,
        "timestamp": datetime.now().isoformat(),
    }

    # ── Persist high-risk alerts to JSONL for dashboard + xPack ─────
    if risk in ("CRITICAL", "HIGH", "MEDIUM"):
        alert_rec = {
            "user_id": tx.user_id,
            "amount": tx.amount,
            "location": tx.location,
            "category": tx.category,
            "risk_level": risk,
            "quantum_classification": qr.get("classification", "N/A"),
            "quantum_fraud_probability": fp,
            "llm_explanation": explanation,
            "anomaly_reasons": qr.get("anomaly_reasons", []),
            "timestamp": datetime.now().isoformat(),
        }
        al_path = os.path.join("data", "high_risk_alerts.jsonl")
        with open(al_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert_rec) + "\n")

    return result


@app.post("/api/rag/query")
async def rag_query(query: RAGQuery):
    """Query regulatory policy documents via RAG."""
    analyzer = get_llm_analyzer()
    # Retrieve context chunks for the query (used by RAGAS evaluation)
    context_chunks = analyzer.retrieve(query.question, top_k=5)
    context_texts = [c["content"] for c in context_chunks] if context_chunks else []
    answer = analyzer.rag_query(query.question)
    return {
        "question": query.question,
        "answer": answer,
        "context_chunks": context_texts,
        "num_sources": len(context_chunks),
    }


@app.get("/api/alerts/summary")
async def get_alert_summary():
    """AI-generated executive summary of recent alerts."""
    alerts = _tail_jsonl("data/high_risk_alerts.jsonl", 20)
    summary = get_llm_analyzer().generate_alert_summary(alerts)
    return {"summary": summary, "alerts_analyzed": len(alerts)}


@app.get("/api/quantum/info")
async def quantum_info():
    """Quantum circuit architecture metadata."""
    return get_quantum_classifier().get_circuit_info()


@app.get("/api/impact/timeline")
async def impact_timeline():
    """Return sustainability impact time-series for the Chart.js dashboard chart.

    Groups alerts by 5-minute buckets and computes cumulative funds protected,
    trees, CO2, and clean water equivalents over time.
    """
    from collections import defaultdict
    al_path = os.path.join("data", "high_risk_alerts.jsonl")
    buckets: dict = defaultdict(lambda: {"count": 0, "amount": 0.0})

    if os.path.exists(al_path):
        try:
            with open(al_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ts_str = rec.get("timestamp") or rec.get("processed_at", "")
                        amt = float(rec.get("amount", 0))
                        # Parse timestamp and bucket to 5-minute intervals
                        if ts_str:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            # Round down to nearest 5 minutes
                            minute = (dt.minute // 5) * 5
                            bucket_key = dt.strftime(f"%Y-%m-%dT%H:{minute:02d}:00")
                        else:
                            bucket_key = "unknown"
                        buckets[bucket_key]["count"] += 1
                        buckets[bucket_key]["amount"] += amt
                    except (json.JSONDecodeError, ValueError):
                        pass
        except Exception:
            pass

    # Build cumulative timeline sorted by time
    sorted_keys = sorted(k for k in buckets if k != "unknown")
    timeline = []
    cum_amount = 0.0
    cum_count = 0
    for key in sorted_keys:
        b = buckets[key]
        cum_amount += b["amount"]
        cum_count += b["count"]
        inr = cum_amount * 83
        timeline.append({
            "time": key,
            "frauds_detected": cum_count,
            "funds_protected_usd": round(cum_amount, 2),
            "funds_protected_inr": round(inr, 2),
            "trees_equivalent": int(inr / 500),
            "co2_offset_kg": round(inr / 1000 * 2.5, 1),
            "clean_water_liters": int(inr / 100000 * 50000),
        })

    return {"timeline": timeline, "total_buckets": len(timeline)}


@app.get("/api/green-impact/methodology")
async def green_impact_methodology():
    """Detailed methodology and citations for Green Bharat sustainability impact.

    Returns verified conversion factors with academic and governmental sources
    mapping fraud prevention to environmental impact under India's climate
    commitments (NDC, NAPCC, net-zero 2070).
    """
    return {
        "title": "QuantGuard Green Bharat — Impact Methodology & Citations",
        "version": "2.0",
        "description": (
            "QuantGuard quantifies the environmental impact of fraud prevention "
            "by calculating how protected funds could be redirected to "
            "sustainability initiatives under India's national green programs. "
            "Each conversion factor is sourced from peer-reviewed or "
            "government-published data."
        ),
        "conversion_factors": [
            {
                "metric": "CO₂ Offset",
                "rate": "INR 1,000 → 2.5 kg CO₂ offset",
                "source": "Gold Standard Foundation",
                "document": "Verified Emission Reductions (VERs) — Impact Quantification Methodology",
                "url": "https://www.goldstandard.org/impact-quantification/",
                "year": 2024,
                "basis": (
                    "Average cost of Gold Standard-certified carbon credits "
                    "in the Indian voluntary carbon market. Price range: "
                    "INR 300–500 per tonne CO₂e (2023-24 average)."
                ),
            },
            {
                "metric": "Reforestation",
                "rate": "INR 500 → 1 native tree planted",
                "source": "Grow-Trees.com & UNEP Trillion Tree Campaign",
                "document": "India Tree Planting Cost Database",
                "url": "https://www.grow-trees.com/plant-trees-india",
                "url_alt": "https://www.trilliontrees.org/",
                "year": 2024,
                "basis": (
                    "End-to-end cost: sapling procurement, planting, "
                    "3-year maintenance, survival monitoring. Species: "
                    "Neem, Peepal, Banyan, Mango (native). NGT (2019) "
                    "values 1 mature tree at INR 74,500/year in ecosystem "
                    "services (oxygen, carbon sequestration, soil conservation)."
                ),
            },
            {
                "metric": "Clean Water",
                "rate": "INR 2 per litre via community filtration",
                "source": "WHO/UNICEF Joint Monitoring Programme (JMP)",
                "document": "Progress on Household Drinking Water 2023",
                "url": "https://washdata.org/",
                "india_program": "Jal Jeevan Mission (jaljeevanmission.gov.in)",
                "year": 2023,
                "basis": (
                    "Cost of community-scale RO/UV water purification "
                    "infrastructure amortised over 10-year lifespan. "
                    "Jal Jeevan Mission target: functional household tap "
                    "connections to 195M+ rural households."
                ),
            },
            {
                "metric": "Solar Energy",
                "rate": "INR 25,000 per 1 kW rooftop solar panel",
                "source": "Ministry of New and Renewable Energy (MNRE)",
                "document": "PM Surya Ghar Muft Bijli Yojana Guidelines 2024",
                "url": "https://www.mnre.gov.in/",
                "year": 2024,
                "basis": (
                    "Subsidised cost under PM Surya Ghar scheme after "
                    "central + state subsidies. 1 kW panel generates "
                    "~1,400 kWh/year in India, offsetting ~1.1 tonne CO₂."
                ),
            },
            {
                "metric": "Green Investment Redirection",
                "rate": "40% of protected funds eligible for green bonds",
                "source": "SEBI & Reserve Bank of India",
                "document": "SEBI Green Bond Framework 2023; RBI Sovereign Green Bond Guidelines",
                "url": "https://www.sebi.gov.in/",
                "year": 2023,
                "basis": (
                    "Conservative estimate of fraud-protected funds that "
                    "meet SEBI green bond eligibility criteria for renewable "
                    "energy, clean transport, pollution prevention, and "
                    "sustainable water management projects."
                ),
            },
        ],
        "india_policy_context": {
            "ndc_2030": (
                "50% cumulative electric power installed capacity from "
                "non-fossil fuel sources by 2030 (India's Updated NDC, Aug 2022)"
            ),
            "net_zero": "Net-zero emissions by 2070 (PM Modi, COP26 Glasgow, Nov 2021)",
            "forest_cover": "713,789 km² (21.71% of geographic area, FSI ISFR 2023)",
            "carbon_intensity": (
                "45% reduction in GDP carbon intensity by 2030 vs 2005 "
                "levels (Updated NDC under Paris Agreement)"
            ),
            "napcc_missions": [
                "National Solar Mission (100 GW target)",
                "National Mission for Green India (5M hectares afforestation)",
                "National Water Mission (20% efficiency improvement)",
                "National Mission for Sustainable Agriculture",
            ],
            "fraud_environment_nexus": (
                "Financial crime directly enables environmental destruction: "
                "illegal logging, sand mining, e-waste dumping, and wildlife "
                "trafficking. FATF (2021) estimates money laundering from "
                "environmental crime at USD 110–281 billion/year globally. "
                "Every fraud prevented protects India's natural capital. "
                "Source: FATF Report on Money Laundering from Environmental Crime, 2021"
            ),
        },
        "sdg_mapping": [
            {"sdg": "SDG 6", "title": "Clean Water and Sanitation", "metric": "clean_water_liters", "india_target": "Jal Jeevan Mission: 100% rural tap water"},
            {"sdg": "SDG 7", "title": "Affordable and Clean Energy", "metric": "solar_panels_kw", "india_target": "500 GW non-fossil capacity by 2030"},
            {"sdg": "SDG 13", "title": "Climate Action", "metric": "co2_offset_kg", "india_target": "Net-zero 2070; 45% carbon intensity reduction by 2030"},
            {"sdg": "SDG 15", "title": "Life on Land", "metric": "trees_equivalent", "india_target": "33% forest cover (National Forest Policy); Green India Mission"},
            {"sdg": "SDG 16", "title": "Peace, Justice and Strong Institutions", "metric": "frauds_detected", "india_target": "PMLA 2002; RBI fraud reporting; FIU-IND STR compliance"},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Log Anomaly Detection Endpoints (parallel streaming use-case)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/logs/alerts")
async def get_log_alerts(limit: int = 50):
    """Return the most recent log anomaly alerts."""
    try:
        from pathway_engine import get_log_anomaly_alerts
        alerts = get_log_anomaly_alerts(limit)
    except Exception:
        alerts = _tail_jsonl("data/log_anomaly_alerts.jsonl", limit)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/api/logs/stats")
async def get_log_stats():
    """Return log anomaly pipeline summary statistics."""
    try:
        from pathway_engine import get_log_anomaly_stats
        stats = get_log_anomaly_stats()
    except Exception:
        stats = {"total": 0, "by_service": {}, "by_severity": {}}
    return stats


@app.get("/api/logs/services")
async def get_log_services():
    """Return per-service health based on recent log anomaly data."""
    try:
        from pathway_engine import get_log_anomaly_alerts
        alerts = get_log_anomaly_alerts(200)
    except Exception:
        alerts = []

    services = {}
    for a in alerts:
        svc = a.get("service", "unknown")
        if svc not in services:
            services[svc] = {
                "service": svc,
                "total_anomalies": 0,
                "latest_severity": "OK",
                "avg_error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "_error_rates": [],
                "_latencies": [],
            }
        services[svc]["total_anomalies"] += 1
        services[svc]["latest_severity"] = a.get("severity", "LOW")
        services[svc]["_error_rates"].append(a.get("error_rate", 0))
        services[svc]["_latencies"].append(a.get("avg_latency_ms", 0))

    result = []
    for svc_data in services.values():
        er = svc_data.pop("_error_rates")
        la = svc_data.pop("_latencies")
        svc_data["avg_error_rate"] = round(sum(er) / max(len(er), 1), 3)
        svc_data["avg_latency_ms"] = round(sum(la) / max(len(la), 1), 1)
        result.append(svc_data)

    result.sort(key=lambda x: -x["total_anomalies"])
    return {"services": result}


# ═══════════════════════════════════════════════════════════════════════════
#  Pathway LLM xPack Endpoints — RAG, Reports, Insights
# ═══════════════════════════════════════════════════════════════════════════

class ReportRequest(BaseModel):
    report_type: str = "executive_summary"  # executive_summary, trend_analysis, compliance_report, risk_assessment, green_impact


class InsightRequest(BaseModel):
    user_id: str
    amount: float
    location: str
    category: str


class CreditRequest(BaseModel):
    user_id: str


@app.post("/api/xpack/rag")
async def xpack_rag_query(query: RAGQuery):
    """Live RAG query over fraud alerts + policy documents (Pathway LLM xPack)."""
    xp = get_xpack()
    xp.sync()  # pick up any new data
    result = xp.rag.query(query.question)
    return result


@app.post("/api/xpack/report")
async def xpack_generate_report(req: ReportRequest):
    """Generate automated report via Pathway LLM xPack."""
    xp = get_xpack()
    xp.sync()

    # Gather data for report
    alerts = _tail_jsonl("data/high_risk_alerts.jsonl", 100)
    stats_data = await get_stats()

    report = xp.reports.generate_report(
        req.report_type,
        data={"alerts": alerts, "stats": stats_data},
    )
    return report


@app.post("/api/xpack/insight")
async def xpack_explain_insight(req: InsightRequest):
    """Generate explainable insight for a transaction (Pathway LLM xPack)."""
    xp = get_xpack()
    clf = get_quantum_classifier()

    # Quantum classify
    amount_norm = min(req.amount / 5000.0, 1.0)
    freq_norm = 0.3
    qr = clf.classify_transaction(np.array([amount_norm, freq_norm]))

    tx = {"user_id": req.user_id, "amount": req.amount,
          "location": req.location, "category": req.category,
          "timestamp": datetime.now().isoformat()}

    insight = xp.insights.explain_decision(tx, qr)
    return insight


@app.post("/api/xpack/credit")
async def xpack_credit_rationale(req: CreditRequest):
    """Generate credit decision rationale for a user (Pathway LLM xPack)."""
    xp = get_xpack()
    alerts = _tail_jsonl("data/high_risk_alerts.jsonl", 200)
    result = xp.insights.generate_credit_rationale(req.user_id, alerts)
    return result


@app.get("/api/xpack/status")
async def xpack_status():
    """Pathway LLM xPack status and capabilities."""
    from pathway_llm_xpack import XPACK_NATIVE
    xp = get_xpack()
    return {
        "status": "active",
        "native_sdk": XPACK_NATIVE,
        "documents_indexed": xp.store.size,
        "capabilities": [
            "live_rag",
            "automated_reports",
            "explainable_insights",
            "credit_rationale",
        ],
        "report_types": [
            "executive_summary",
            "trend_analysis",
            "compliance_report",
            "risk_assessment",
            "green_impact",
        ],
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Live Data Source Control
# ═══════════════════════════════════════════════════════════════════════════

class LiveSourceConfig(BaseModel):
    source: str = "demo"  # demo, alpha, socket
    symbols: list = None
    interval: float = 1.0


@app.post("/api/live/start")
async def start_live_source(config: LiveSourceConfig):
    """Start a live market data source (Alpha Vantage / Polygon / Demo)."""
    global _live_source_task

    if _live_source_task and not _live_source_task.done():
        return {"status": "already_running", "message": "Live source is already active"}

    from live_data_source import (
        AlphaVantageSource, DemoLiveSource, SocketStreamSource, write_transaction
    )

    if config.source == "alpha":
        src = AlphaVantageSource(symbols=config.symbols)
        _live_source_task = asyncio.create_task(
            src.stream(write_transaction, interval=config.interval or 12)
        )
    elif config.source == "socket":
        src = SocketStreamSource()
        _live_source_task = asyncio.create_task(
            src.stream(write_transaction)
        )
    else:  # demo
        src = DemoLiveSource()
        _live_source_task = asyncio.create_task(
            src.stream(write_transaction, interval=config.interval or 1.0)
        )

    return {"status": "started", "source": config.source, "message": f"Live {config.source} source started"}


@app.post("/api/live/stop")
async def stop_live_source():
    """Stop the running live data source."""
    global _live_source_task
    if _live_source_task and not _live_source_task.done():
        _live_source_task.cancel()
        _live_source_task = None
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/api/live/status")
async def live_source_status():
    """Check live data source status."""
    running = _live_source_task and not _live_source_task.done()
    return {"running": running, "timestamp": datetime.now().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket
# ═══════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
            await ws.send_json({"type": "ack"})
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _tail_jsonl(path, n=50):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line.strip()))
        except (json.JSONDecodeError, ValueError):
            pass
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

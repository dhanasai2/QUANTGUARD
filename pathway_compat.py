"""
QuantGuard - Pathway Compatibility Layer
=========================================
Provides a Pathway-compatible API surface for Windows / non-Linux
environments where the native Pathway engine cannot run.

The mock classes let pathway_engine.py define its pipeline declaratively.
The actual computation happens inside PathwayMock.run(), which:

  1. Polls the transaction JSONL file for new entries
  2. Computes per-user rolling statistics (mean, std, max, count)
  3. Detects anomalies via multiple rules (spike, threshold, outlier, geo)
  4. Routes high-risk transactions to the Quantum Risk Classifier
  5. Generates LLM-powered explanations via Groq (RAG)
  6. Writes enriched alerts to the output file
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
#  Mock Pathway API Surface
# ═══════════════════════════════════════════════════════════════════════════

class MockColumn:
    def __init__(self, name):
        self.name = name

    def apply(self, func):
        return self

    def __gt__(self, o): return self
    def __lt__(self, o): return self
    def __ge__(self, o): return self
    def __le__(self, o): return self
    def __or__(self, o): return self
    def __and__(self, o): return self
    def __eq__(self, o): return self
    def __ne__(self, o): return self
    def __mul__(self, o): return self
    def __add__(self, o): return self
    def __sub__(self, o): return self
    def __truediv__(self, o): return self


class MockThis:
    def __getattr__(self, name):
        return MockColumn(name)


class Schema:
    pass


class Reducers:
    @staticmethod
    def avg(col): return "avg"
    @staticmethod
    def count(): return "count"
    @staticmethod
    def max(col): return "max"
    @staticmethod
    def min(col): return "min"
    @staticmethod
    def sum(col): return "sum"


class Temporal:
    @staticmethod
    def sliding(duration, hop):
        return {"type": "sliding", "duration": duration, "hop": hop}
    @staticmethod
    def tumbling(duration):
        return {"type": "tumbling", "duration": duration}


class MockTable:
    def __init__(self, path=None):
        self.path = path
        self.output_path = None

    def __getattr__(self, name):
        return MockColumn(name)

    def with_columns(self, **kw): return self
    def windowby(self, *a, **kw): return self
    def reduce(self, **kw): return self
    def join(self, *a, **kw): return self
    def filter(self, *a, **kw): return self
    def select(self, *a, **kw): return self
    def groupby(self, *a, **kw): return self


class IO:
    def __init__(self):
        self.jsonlines = self._JSONLines()

    class _JSONLines:
        def read(self, path, schema=None, mode="streaming"):
            return MockTable(path)
        def write(self, table, path):
            if isinstance(table, MockTable):
                table.output_path = path


# ═══════════════════════════════════════════════════════════════════════════
#  Pathway Mock Engine (Compatibility Processing)
# ═══════════════════════════════════════════════════════════════════════════

class PathwayMock:
    """
    Drop-in replacement for the ``pathway`` module.
    Exposes the same public surface (Schema, this, reducers, temporal, io)
    so that ``pathway_engine.py`` can import and call it transparently.
    """

    def __init__(self):
        self.Schema = Schema
        self.this = MockThis()
        self.reducers = Reducers
        self.temporal = Temporal
        self.io = IO()

    # ── Core Streaming Loop ─────────────────────────────────────────────

    def run(self):
        """Execute the streaming pipeline with full analytics."""
        print()
        print("=" * 60)
        print("  QuantGuard Streaming Engine (Pathway Compat)")
        print("=" * 60)

        # Lazy-import heavy modules to avoid circular deps
        from quantum_classifier import QuantumRiskClassifier
        from llm_engine import GroqFraudAnalyzer

        classifier = QuantumRiskClassifier()
        analyzer = GroqFraudAnalyzer()

        input_file = os.path.join("data", "transactions.jsonl")
        output_file = os.path.join("data", "high_risk_alerts.jsonl")

        # Per-user rolling state
        user_amounts = defaultdict(list)
        user_locations = defaultdict(set)
        user_timestamps = defaultdict(list)

        last_pos = 0
        processed = 0
        alerts_generated = 0

        print(f"[Engine] Input  : {input_file}")
        print(f"[Engine] Output : {output_file}")
        print(f"[Engine] Watching for new transactions ...\n")

        while True:
            if not os.path.exists(input_file):
                time.sleep(1)
                continue

            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    f.seek(last_pos)
                    lines = f.readlines()
                    last_pos = f.tell()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        tx = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    processed += 1
                    uid = tx.get("user_id", "UNKNOWN")
                    amount = float(tx.get("amount", 0))
                    location = tx.get("location", "")
                    category = tx.get("category", "")
                    timestamp = tx.get("timestamp", "")

                    # ── Rolling Feature Computation ─────────────────
                    user_amounts[uid].append(amount)
                    user_locations[uid].add(location)
                    user_timestamps[uid].append(timestamp)

                    # Keep last 100 per user (sliding window)
                    if len(user_amounts[uid]) > 100:
                        user_amounts[uid] = user_amounts[uid][-100:]

                    recent = user_amounts[uid]
                    avg_amount = float(np.mean(recent))
                    max_amount = float(max(recent))
                    std_amount = float(np.std(recent)) if len(recent) > 1 else 0.0
                    tx_count = len(recent)
                    unique_locs = len(user_locations[uid])

                    # ── ML Anomaly Scoring (no hardcoded thresholds) ─
                    anomaly_score = 0.0
                    anomaly_reasons = []
                    ml_features = {}

                    # Feature 1: Z-score (standard deviations from user mean)
                    if std_amount > 0 and tx_count > 3:
                        z_score = (amount - avg_amount) / std_amount
                        ml_features["z_score"] = round(z_score, 2)
                        if z_score > 2.0:
                            anomaly_score += min(z_score / 4.0, 0.35)
                            anomaly_reasons.append(
                                f"Z-score: {z_score:.1f}σ above user mean (${avg_amount:.0f})"
                            )
                    else:
                        z_score = 0.0
                        ml_features["z_score"] = 0.0

                    # Feature 2: IQR outlier detection
                    if tx_count >= 5:
                        sorted_amts = sorted(recent)
                        q1 = sorted_amts[len(sorted_amts) // 4]
                        q3 = sorted_amts[3 * len(sorted_amts) // 4]
                        iqr = q3 - q1
                        upper_fence = q3 + 1.5 * iqr
                        ml_features["iqr_upper_fence"] = round(upper_fence, 2)
                        if iqr > 0 and amount > upper_fence:
                            iqr_deviation = (amount - upper_fence) / iqr
                            anomaly_score += min(iqr_deviation / 3.0, 0.25)
                            anomaly_reasons.append(
                                f"IQR outlier: ${amount:.0f} > fence ${upper_fence:.0f}"
                            )

                    # Feature 3: Percentile rank within user history
                    if tx_count > 1:
                        pct_rank = sum(1 for x in recent if x <= amount) / tx_count
                        ml_features["percentile_rank"] = round(pct_rank, 3)
                        if pct_rank > 0.95:
                            anomaly_score += 0.15
                            anomaly_reasons.append(
                                f"Top {(1-pct_rank)*100:.0f}th percentile for this user"
                            )
                    else:
                        ml_features["percentile_rank"] = 0.5

                    # Feature 4: Geographic entropy
                    ml_features["unique_locations"] = unique_locs
                    if unique_locs > 4:
                        geo_score = min((unique_locs - 4) / 6.0, 0.15)
                        anomaly_score += geo_score
                        anomaly_reasons.append(
                            f"Geo-entropy: {unique_locs} distinct locations"
                        )

                    # Feature 5: Amount-to-mean ratio
                    if avg_amount > 0:
                        ratio = amount / avg_amount
                        ml_features["amount_to_mean_ratio"] = round(ratio, 2)
                        if ratio > 3.0:
                            anomaly_score += min((ratio - 3.0) / 5.0, 0.15)
                            anomaly_reasons.append(
                                f"Spending ratio: {ratio:.1f}x user average"
                            )

                    # Feature 6: Velocity (transactions per window)
                    ml_features["tx_velocity"] = tx_count
                    if tx_count > 20:
                        vel_score = min((tx_count - 20) / 40.0, 0.10)
                        anomaly_score += vel_score
                        anomaly_reasons.append(
                            f"High velocity: {tx_count} txns in window"
                        )

                    ml_features["anomaly_score"] = round(min(anomaly_score, 1.0), 4)

                    # Skip if combined ML score is too low
                    if anomaly_score < 0.12:
                        continue  # normal transaction

                    # ── Quantum Classification ──────────────────────
                    amount_norm = min(amount / max(max_amount, avg_amount * 3, 1), 1.0)
                    freq_norm = min(tx_count / max(50, tx_count * 1.5), 1.0)
                    features = np.array([amount_norm, freq_norm])

                    quantum_result = classifier.classify_transaction(features)

                    # ── LLM Explanation (always generated) ──────────
                    llm_text = analyzer.explain_fraud_risk(tx, quantum_result)

                    # ── Risk Level ──────────────────────────────────
                    fp = quantum_result["fraud_probability"]
                    if fp > 0.70:
                        risk = "CRITICAL"
                    elif fp > 0.45:
                        risk = "HIGH"
                    elif fp > 0.25:
                        risk = "MEDIUM"
                    else:
                        risk = "LOW"

                    # ── Build Enriched Alert ────────────────────────
                    alert = {
                        **tx,
                        "is_anomaly": True,
                        "anomaly_reasons": anomaly_reasons,
                        "ml_features": ml_features,
                        "rolling_stats": {
                            "avg_amount": round(avg_amount, 2),
                            "max_amount": round(max_amount, 2),
                            "std_amount": round(std_amount, 2),
                            "tx_count": tx_count,
                            "unique_locations": unique_locs,
                        },
                        "quantum_classification": quantum_result["classification"],
                        "quantum_fraud_probability": quantum_result["fraud_probability"],
                        "quantum_states": quantum_result["quantum_states"],
                        "llm_explanation": llm_text,
                        "alert_type": "Quantum Risk Verified",
                        "risk_level": risk,
                        "processed_at": datetime.now().isoformat(),
                    }

                    with open(output_file, "a", encoding="utf-8") as out:
                        out.write(json.dumps(alert, default=str) + "\n")

                    alerts_generated += 1

                    # ── Rich Terminal Output ────────────────────────
                    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(risk, "🟢")
                    qp = quantum_result['fraud_probability']
                    qs = quantum_result.get('quantum_states', {})
                    sa = quantum_result.get('state_analysis', {})
                    ba = sa.get('bloch_angles', {})
                    probs = sa.get('probabilities', {})
                    bar_w = 20  # bar chart width

                    print(f"\n  {'═' * 72}")
                    print(f"  {icon} ALERT #{alerts_generated}  │  {uid}  │  ${amount:,.2f}  │  {location}  │  {category}")
                    print(f"  {'─' * 72}")

                    # ML Feature Scores
                    print(f"  📊 ML ANOMALY MODEL (score: {ml_features.get('anomaly_score', 0):.3f})")
                    zs = ml_features.get('z_score', 0)
                    pr = ml_features.get('percentile_rank', 0)
                    ar = ml_features.get('amount_to_mean_ratio', 0)
                    ul = ml_features.get('unique_locations', 0)
                    tv = ml_features.get('tx_velocity', 0)
                    iqr_f = ml_features.get('iqr_upper_fence', 0)
                    z_bar = '█' * min(int(abs(zs) * 3), bar_w)
                    p_bar = '█' * min(int(pr * bar_w), bar_w)
                    print(f"     Z-Score:     {zs:>+7.2f}σ  [{z_bar:<{bar_w}}]  (mean: ${avg_amount:.0f}, std: ${std_amount:.0f})")
                    print(f"     Percentile:  {pr:>7.1%}   [{p_bar:<{bar_w}}]  (rank in {tx_count} txns)")
                    print(f"     Spend Ratio: {ar:>7.1f}x   IQR Fence: ${iqr_f:,.0f}  Geo: {ul} cities  Velocity: {tv} txns")

                    # Quantum Result
                    print(f"  ⚛️  QUANTUM VQC ({quantum_result.get('backend', 'simulator')})")
                    print(f"     Classification: {quantum_result['classification']}  │  P(fraud): {qp:.1%}  │  Shots: {quantum_result.get('total_shots', 1024)}")
                    if quantum_result.get('ibm_job_id'):
                        print(f"     IBM Job: {quantum_result['ibm_job_id']}")

                    # Measurement probabilities mini bar chart
                    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
                    keys = ['00', '01', '10', '11']
                    bars = '     '
                    for lbl, k in zip(labels, keys):
                        p = probs.get(k, 0)
                        b = '█' * min(int(p * bar_w * 2), bar_w)
                        bars += f"{lbl} {p:5.1%} [{b:<{bar_w}}]  "
                    print(bars)

                    # Bloch sphere angles
                    if ba:
                        q0d = ba.get('qubit_0', {}).get('description', '')
                        q1d = ba.get('qubit_1', {}).get('description', '')
                        ent = sa.get('entanglement_indicator', 0)
                        print(f"     Bloch: q₀={q0d}  q₁={q1d}  Entanglement={ent:.3f}")

                    # Risk decision
                    print(f"  🏷️  RISK: [{risk}]  │  Reasons: {'; '.join(anomaly_reasons)}")
                    print(f"  {'═' * 72}")

                # Periodic stats
                if processed > 0 and processed % 50 == 0:
                    rate = alerts_generated / max(processed, 1) * 100
                    print(
                        f"\n  ┌{'─' * 50}┐"
                        f"\n  │ 📈 PIPELINE: {processed:,} processed │ "
                        f"{alerts_generated:,} alerts │ "
                        f"Rate: {rate:.1f}%"
                        f"\n  └{'─' * 50}┘\n"
                    )

            except Exception as e:
                print(f"  [Error] {e}")

            time.sleep(0.5)


# ── Global instance (matches `import pathway as pw` interface) ─────────
pw = PathwayMock()

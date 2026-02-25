"""
QuantGuard - ML Transaction Data Generator
=============================================
Generates synthetic financial transaction data with realistic fraud
patterns and performs real-time ML anomaly scoring as each transaction
is produced.

ML Features computed per transaction:
  • Z-Score             – standard deviations from user mean spend
  • IQR Outlier Score   – distance from interquartile range bounds
  • Geo-Entropy         – Shannon entropy of user location history
  • Velocity Score      – time-weighted transaction frequency
  • Category Deviation  – divergence from user's spending profile
  • Composite Risk      – weighted ensemble of all features

Fraud patterns injected:
  • High-value spikes       – amount > $4 000
  • Geographic anomalies    – transactions from unexpected cities
  • Category anomalies      – unusual spending categories for the user
  • Velocity / rapid-fire   – bursts of quick successive transactions

Each synthetic user has a persistent profile (home location, average
spend, preferred categories) so normal behaviour is coherent.
"""

import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime
import os

# ── Configuration ───────────────────────────────────────────────────────

DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.jsonl")

LOCATIONS = [
    "New York", "London", "Tokyo", "Mumbai", "Paris",
    "Berlin", "Dubai", "Singapore", "Sydney", "Toronto",
]
CATEGORIES = [
    "Grocery", "Electronics", "Travel", "Dining",
    "Healthcare", "Entertainment", "Utilities", "Shopping",
]

FRAUD_RATE = 0.15  # 15 % of transactions will be suspicious
WINDOW_SIZE = 50   # rolling window for ML stats

# ── User Profiles ───────────────────────────────────────────────────────

random.seed(42)  # reproducible profiles
USER_PROFILES = {}
for i in range(100, 115):
    USER_PROFILES[f"USER_{i}"] = {
        "home_location": random.choice(LOCATIONS[:5]),
        "avg_spend": round(random.uniform(100, 800), 2),
        "primary_categories": random.sample(CATEGORIES, 3),
    }
random.seed()  # re-randomise from here


# ═══════════════════════════════════════════════════════════════════════════
#  ML Anomaly Scoring Engine (runs inline with generator)
# ═══════════════════════════════════════════════════════════════════════════

class MLAnomalyScorer:
    """Real-time rolling statistics and anomaly feature extraction."""

    def __init__(self):
        self._amounts: dict[str, list[float]] = defaultdict(list)
        self._timestamps: dict[str, list[float]] = defaultdict(list)
        self._locations: dict[str, list[str]] = defaultdict(list)
        self._categories: dict[str, list[str]] = defaultdict(list)

    # ── Feature: Z-Score ────────────────────────────────────────────
    def z_score(self, uid: str, amount: float) -> float:
        """Standard deviations from the user's rolling mean."""
        hist = self._amounts.get(uid, [])
        if len(hist) < 3:
            return 0.0
        mean = sum(hist) / len(hist)
        std = max((sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5, 0.01)
        return round((amount - mean) / std, 3)

    # ── Feature: IQR Outlier Score ──────────────────────────────────
    def iqr_score(self, uid: str, amount: float) -> float:
        """Distance from interquartile range (>1.5 = outlier)."""
        hist = sorted(self._amounts.get(uid, []))
        if len(hist) < 5:
            return 0.0
        q1 = hist[len(hist) // 4]
        q3 = hist[3 * len(hist) // 4]
        iqr = max(q3 - q1, 0.01)
        if amount > q3:
            return round((amount - q3) / iqr, 3)
        elif amount < q1:
            return round((q1 - amount) / iqr, 3)
        return 0.0

    # ── Feature: Geo-Entropy ────────────────────────────────────────
    def geo_entropy(self, uid: str) -> float:
        """Shannon entropy of location diversity (0 = single location)."""
        locs = self._locations.get(uid, [])
        if len(locs) < 2:
            return 0.0
        counts: dict[str, int] = {}
        for loc in locs[-WINDOW_SIZE:]:
            counts[loc] = counts.get(loc, 0) + 1
        total = sum(counts.values())
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values() if c > 0
        )
        return round(entropy, 3)

    # ── Feature: Velocity Score ─────────────────────────────────────
    def velocity_score(self, uid: str) -> float:
        """Transaction frequency in last 60s window (higher = faster)."""
        ts = self._timestamps.get(uid, [])
        if len(ts) < 2:
            return 0.0
        now = time.time()
        recent = [t for t in ts[-WINDOW_SIZE:] if now - t < 60]
        return round(len(recent) / max(now - recent[0], 0.1), 3) if recent else 0.0

    # ── Feature: Category Deviation ─────────────────────────────────
    def category_deviation(self, uid: str, category: str) -> float:
        """How unusual this category is for the user (0-1)."""
        cats = self._categories.get(uid, [])
        if len(cats) < 3:
            return 0.0
        counts: dict[str, int] = {}
        for c in cats[-WINDOW_SIZE:]:
            counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values())
        freq = counts.get(category, 0) / max(total, 1)
        return round(1.0 - freq, 3)

    # ── Composite Risk Score ────────────────────────────────────────
    def composite_risk(self, z: float, iqr: float, geo: float,
                       vel: float, cat_dev: float) -> float:
        """Weighted ensemble: 0.0 (safe) to 1.0 (critical)."""
        raw = (
            0.30 * min(abs(z) / 4.0, 1.0)      # Z-score (capped at 4σ)
            + 0.25 * min(iqr / 3.0, 1.0)        # IQR outlier
            + 0.15 * min(geo / 3.5, 1.0)         # Geo entropy (max ~3.3 for 10 locs)
            + 0.15 * min(vel / 2.0, 1.0)         # Velocity
            + 0.15 * cat_dev                      # Category deviation
        )
        return round(min(raw, 1.0), 4)

    # ── Update & Score ──────────────────────────────────────────────
    def score_transaction(self, tx: dict) -> dict:
        """Compute all ML features for the transaction, then update history."""
        uid = tx["user_id"]
        amount = tx["amount"]
        category = tx["category"]
        now = time.time()

        # Compute features BEFORE updating history (test against history)
        z = self.z_score(uid, amount)
        iqr = self.iqr_score(uid, amount)
        geo = self.geo_entropy(uid)
        vel = self.velocity_score(uid)
        cat_dev = self.category_deviation(uid, category)
        risk = self.composite_risk(z, iqr, geo, vel, cat_dev)

        # Update rolling history AFTER scoring
        self._amounts[uid].append(amount)
        self._timestamps[uid].append(now)
        self._locations[uid].append(tx["location"])
        self._categories[uid].append(category)
        # Trim to window
        for store in (self._amounts, self._timestamps,
                      self._locations, self._categories):
            if len(store[uid]) > WINDOW_SIZE:
                store[uid] = store[uid][-WINDOW_SIZE:]

        risk_label = (
            "CRITICAL" if risk > 0.65 else
            "HIGH"     if risk > 0.45 else
            "MEDIUM"   if risk > 0.25 else
            "LOW"
        )

        return {
            "z_score": z,
            "iqr_score": iqr,
            "geo_entropy": geo,
            "velocity": vel,
            "category_dev": cat_dev,
            "composite_risk": risk,
            "risk_label": risk_label,
            "history_depth": len(self._amounts[uid]),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Transaction Generators
# ═══════════════════════════════════════════════════════════════════════════

def generate_normal_transaction():
    """Generate a legitimate-looking transaction."""
    uid = random.choice(list(USER_PROFILES))
    p = USER_PROFILES[uid]

    amount = round(max(5.0, random.gauss(p["avg_spend"], p["avg_spend"] * 0.4)), 2)
    location = p["home_location"] if random.random() < 0.70 else random.choice(LOCATIONS)
    category = (
        random.choice(p["primary_categories"])
        if random.random() < 0.60
        else random.choice(CATEGORIES)
    )

    return {
        "user_id": uid,
        "amount": amount,
        "location": location,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "is_suspicious_flag": False,
    }


def generate_fraudulent_transaction():
    """Generate a transaction exhibiting a fraud pattern."""
    uid = random.choice(list(USER_PROFILES))
    p = USER_PROFILES[uid]

    pattern = random.choice(["high_value", "geo_anomaly", "velocity", "category_anomaly"])

    if pattern == "high_value":
        amount = round(random.uniform(4000, 5000), 2)
        location = random.choice(LOCATIONS)
        category = random.choice(CATEGORIES)
    elif pattern == "geo_anomaly":
        amount = round(random.uniform(500, 3000), 2)
        foreign = [loc for loc in LOCATIONS if loc != p["home_location"]]
        location = random.choice(foreign)
        category = random.choice(CATEGORIES)
    elif pattern == "velocity":
        amount = round(random.uniform(200, 1500), 2)
        location = random.choice(LOCATIONS)
        category = "Electronics"
    else:  # category_anomaly
        amount = round(random.uniform(1000, 4000), 2)
        unusual = [c for c in CATEGORIES if c not in p["primary_categories"]]
        category = random.choice(unusual) if unusual else random.choice(CATEGORIES)
        location = random.choice(LOCATIONS)

    return {
        "user_id": uid,
        "amount": amount,
        "location": location,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "is_suspicious_flag": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Terminal Display
# ═══════════════════════════════════════════════════════════════════════════

_RISK_COLORS = {
    "CRITICAL": "\033[91m",  # bright red
    "HIGH":     "\033[93m",  # yellow
    "MEDIUM":   "\033[33m",  # dark yellow
    "LOW":      "\033[32m",  # green
}
_RESET = "\033[0m"
_DIM   = "\033[2m"
_BOLD  = "\033[1m"
_CYAN  = "\033[36m"
_WHITE = "\033[97m"


def _risk_bar(risk: float, width: int = 20) -> str:
    """Visual risk bar: ████████░░░░ 65%"""
    filled = int(risk * width)
    if risk > 0.65:
        color = "\033[91m"
    elif risk > 0.45:
        color = "\033[93m"
    elif risk > 0.25:
        color = "\033[33m"
    else:
        color = "\033[32m"
    return f"{color}{'█' * filled}{'░' * (width - filled)}{_RESET} {risk:.0%}"


def _print_ml_output(count: int, tx: dict, ml: dict, fraud_count: int):
    """Print rich ML feature output to terminal."""
    uid = tx["user_id"]
    amt = tx["amount"]
    loc = tx["location"]
    cat = tx["category"]
    rl = ml["risk_label"]
    color = _RISK_COLORS.get(rl, "")
    flag = "🚨" if tx["is_suspicious_flag"] else "✅"

    # ── Header Line ─────────────────────────────────────────────────
    print(f"\n  {flag} {_BOLD}[{count:>5}]{_RESET} {_WHITE}{uid}{_RESET} | "
          f"{_CYAN}${amt:>9,.2f}{_RESET} | {loc:>12} | {cat}")

    # ── ML Feature Vector ───────────────────────────────────────────
    z = ml["z_score"]
    z_indicator = "▲" if z > 2 else "▼" if z < -2 else "─"
    print(f"        {_DIM}├─ Z-Score:      {z:>+7.3f} {z_indicator}  "
          f"│ IQR:          {ml['iqr_score']:>6.3f}  "
          f"│ Depth: {ml['history_depth']}{_RESET}")
    print(f"        {_DIM}├─ Geo-Entropy:  {ml['geo_entropy']:>7.3f}    "
          f"│ Velocity:     {ml['velocity']:>6.3f}  "
          f"│ Cat-Dev: {ml['category_dev']:.3f}{_RESET}")

    # ── Composite Risk ──────────────────────────────────────────────
    risk = ml["composite_risk"]
    print(f"        {_DIM}└─ Risk: {color}[{rl:>8}]{_RESET} "
          f"{_risk_bar(risk)}")


def _print_summary(count: int, fraud_count: int, scorer: MLAnomalyScorer):
    """Print periodic summary stats."""
    crit = high = med = low = 0
    for uid in scorer._amounts:
        last_amt = scorer._amounts[uid][-1] if scorer._amounts[uid] else 0
        last_cat = scorer._categories[uid][-1] if scorer._categories[uid] else ""
        ml = scorer.score_transaction({
            "user_id": uid, "amount": last_amt,
            "location": "", "category": last_cat})
        lbl = ml["risk_label"]
        if lbl == "CRITICAL": crit += 1
        elif lbl == "HIGH": high += 1
        elif lbl == "MEDIUM": med += 1
        else: low += 1

    print(f"\n  {'─' * 70}")
    print(f"  {_BOLD}📊 ML Summary @ tx #{count}{_RESET}  │  "
          f"Generated: {count}  │  Fraud injected: {fraud_count} "
          f"({fraud_count/max(count,1):.1%})")
    print(f"  Risk Distribution:  "
          f"\033[91m● CRIT: {crit}\033[0m  "
          f"\033[93m● HIGH: {high}\033[0m  "
          f"\033[33m● MED: {med}\033[0m  "
          f"\033[32m● LOW: {low}\033[0m")
    print(f"  {'─' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_generator():
    os.makedirs(DATA_DIR, exist_ok=True)

    scorer = MLAnomalyScorer()

    print(f"\033[96m{'═' * 70}")
    print(f"  QuantGuard ML Transaction Generator")
    print(f"{'═' * 70}\033[0m")
    print(f"  Output     : {TRANSACTIONS_FILE}")
    print(f"  Users      : {len(USER_PROFILES)}")
    print(f"  Fraud rate : ~{FRAUD_RATE:.0%}")
    print(f"  ML Window  : {WINDOW_SIZE} transactions per user")
    print(f"  Features   : Z-Score │ IQR │ Geo-Entropy │ Velocity │ Cat-Dev")
    print(f"  Scoring    : Weighted ensemble → 0.0 (safe) to 1.0 (critical)")
    print(f"  Press Ctrl+C to stop\n")

    count = 0
    fraud_count = 0

    while True:
        if random.random() < FRAUD_RATE:
            tx = generate_fraudulent_transaction()
            fraud_count += 1
        else:
            tx = generate_normal_transaction()

        # ── ML Scoring ──────────────────────────────────────────────
        ml = scorer.score_transaction(tx)

        # ── Write Transaction ───────────────────────────────────────
        tx_out = {**tx, "ml_risk": ml["composite_risk"],
                  "ml_label": ml["risk_label"]}
        with open(TRANSACTIONS_FILE, "a") as f:
            f.write(json.dumps(tx_out) + "\n")

        count += 1

        # ── Terminal Output ─────────────────────────────────────────
        # Show ALL suspicious/high-risk + every 5th normal
        if (tx["is_suspicious_flag"]
                or ml["risk_label"] in ("CRITICAL", "HIGH", "MEDIUM")
                or count % 5 == 0):
            _print_ml_output(count, tx, ml, fraud_count)

        # ── Periodic Summary ────────────────────────────────────────
        if count % 50 == 0:
            _print_summary(count, fraud_count, scorer)

        time.sleep(random.uniform(0.3, 1.2))


if __name__ == "__main__":
    run_generator()

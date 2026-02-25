"""
QuantGuard — Kafka Producer Sidecar
=====================================
Publishes synthetic fraud transactions to a Kafka topic so the
end-to-end Kafka pipeline is demonstrated out of the box.

Runs as a Docker sidecar container alongside the main QuantGuard app.
Waits for the Kafka broker to become available, then continuously
produces JSON-encoded transaction messages to the configured topic.

Environment:
  KAFKA_BOOTSTRAP_SERVERS  — broker address (default: kafka:9092)
  KAFKA_TOPIC              — target topic (default: quantguard-transactions)
  PRODUCER_INTERVAL        — seconds between messages (default: 2.0)
"""

import json
import os
import random
import time
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "quantguard-transactions")
INTERVAL = float(os.environ.get("PRODUCER_INTERVAL", "2.0"))

LOCATIONS = [
    "New York", "London", "Tokyo", "Mumbai", "Paris",
    "Berlin", "Dubai", "Singapore", "Sydney", "Toronto",
]
CATEGORIES = [
    "Grocery", "Electronics", "Travel", "Dining",
    "Healthcare", "Entertainment", "Utilities", "Shopping",
]
FRAUD_RATE = 0.18  # slightly higher to exercise the pipeline


def _build_profiles():
    """Build 15 synthetic user profiles (same seed as data_source.py)."""
    rng = random.Random(42)
    profiles = {}
    for i in range(100, 115):
        profiles[f"USER_{i}"] = {
            "home": rng.choice(LOCATIONS[:5]),
            "avg": round(rng.uniform(100, 800), 2),
            "cats": rng.sample(CATEGORIES, 3),
        }
    return profiles


def generate_transaction(profiles: dict) -> dict:
    """Generate a single transaction (normal or fraudulent)."""
    is_fraud = random.random() < FRAUD_RATE
    uid = random.choice(list(profiles))
    p = profiles[uid]

    if is_fraud:
        pattern = random.choice([
            "high_value", "geo_anomaly", "velocity", "category_anomaly",
        ])
        if pattern == "high_value":
            amt = round(random.uniform(4000, 5000), 2)
            loc = random.choice(LOCATIONS)
            cat = random.choice(CATEGORIES)
        elif pattern == "geo_anomaly":
            amt = round(random.uniform(500, 3000), 2)
            foreign = [l for l in LOCATIONS if l != p["home"]]
            loc = random.choice(foreign)
            cat = random.choice(CATEGORIES)
        elif pattern == "velocity":
            amt = round(random.uniform(200, 1500), 2)
            loc = random.choice(LOCATIONS)
            cat = "Electronics"
        else:  # category_anomaly
            amt = round(random.uniform(1000, 4000), 2)
            unusual = [c for c in CATEGORIES if c not in p["cats"]]
            cat = random.choice(unusual) if unusual else random.choice(CATEGORIES)
            loc = random.choice(LOCATIONS)
        flag = True
    else:
        amt = round(max(5.0, random.gauss(p["avg"], p["avg"] * 0.4)), 2)
        loc = p["home"] if random.random() < 0.70 else random.choice(LOCATIONS)
        cat = (random.choice(p["cats"]) if random.random() < 0.60
               else random.choice(CATEGORIES))
        flag = False

    return {
        "user_id": uid,
        "amount": amt,
        "location": loc,
        "category": cat,
        "timestamp": datetime.now().isoformat(),
        "is_suspicious_flag": flag,
    }


def main():
    """Main producer loop — waits for Kafka, then publishes transactions."""
    print(f"[KafkaProducer] Starting — broker={BOOTSTRAP} topic={TOPIC}")
    print(f"[KafkaProducer] Interval={INTERVAL}s  fraud_rate={FRAUD_RATE:.0%}")

    # ── Import kafka-python ────────────────────────────────────────────
    try:
        from kafka import KafkaProducer
    except ImportError:
        print("[KafkaProducer] ERROR: kafka-python not installed")
        print("[KafkaProducer] Install with: pip install kafka-python")
        return

    # ── Wait for broker ────────────────────────────────────────────────
    producer = None
    retry_delay = 3
    while producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            print(f"[KafkaProducer] Connected to {BOOTSTRAP}")
        except Exception as exc:
            print(f"[KafkaProducer] Broker not ready ({exc}). "
                  f"Retrying in {retry_delay}s…")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)

    # ── Produce loop ───────────────────────────────────────────────────
    profiles = _build_profiles()
    count = 0

    try:
        while True:
            tx = generate_transaction(profiles)
            producer.send(TOPIC, value=tx)
            count += 1

            if count % 25 == 0:
                producer.flush()
                print(f"[KafkaProducer] Published {count} messages to {TOPIC}")

            time.sleep(INTERVAL + random.uniform(-0.5, 0.5))
    except KeyboardInterrupt:
        print(f"\n[KafkaProducer] Stopped after {count} messages")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()

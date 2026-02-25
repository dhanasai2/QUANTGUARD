"""
QuantGuard — FastAPI Endpoint Tests
=====================================
Tests the REST API endpoints using FastAPI's TestClient.
"""

import json
import pytest


class TestHealthEndpoints:
    def test_stats_endpoint(self, api_client):
        """GET /api/stats returns 200 with expected structure."""
        resp = api_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_transactions" in data
        assert "total_alerts" in data
        assert "alert_rate" in data
        assert "sustainability" in data
        assert "quantum_backend" in data

    def test_sustainability_fields(self, api_client):
        """Sustainability metrics include all Green Bharat fields."""
        resp = api_client.get("/api/stats")
        s = resp.json()["sustainability"]
        required = [
            "frauds_detected", "funds_protected_usd",
            "funds_protected_inr", "co2_offset_kg",
            "trees_equivalent", "clean_water_liters",
        ]
        for key in required:
            assert key in s, f"Missing sustainability field: {key}"

    def test_engine_status(self, api_client):
        """GET /api/engine/status returns pipeline description."""
        resp = api_client.get("/api/engine/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "pipelines" in data
        assert "fraud_detection" in data["pipelines"]
        assert "log_anomaly_detection" in data["pipelines"]
        assert isinstance(data["pipelines"]["fraud_detection"], list)
        assert len(data["pipelines"]["fraud_detection"]) >= 10
        assert len(data["pipelines"]["log_anomaly_detection"]) >= 5

    def test_quantum_info(self, api_client):
        """GET /api/quantum/info returns circuit metadata."""
        resp = api_client.get("/api/quantum/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_parameters"] == 4
        assert data["hilbert_space_dimension"] == 4


class TestAlertsTransactions:
    def test_alerts_endpoint(self, api_client):
        """GET /api/alerts returns list."""
        resp = api_client.get("/api/alerts?limit=5")
        assert resp.status_code == 200
        assert "alerts" in resp.json()

    def test_transactions_endpoint(self, api_client):
        """GET /api/transactions returns list."""
        resp = api_client.get("/api/transactions?limit=5")
        assert resp.status_code == 200
        assert "transactions" in resp.json()


class TestAnalyze:
    def test_analyze_returns_quantum_result(self, api_client):
        """POST /api/analyze returns quantum classification + LLM."""
        resp = api_client.post("/api/analyze", json={
            "user_id": "USER_TEST",
            "amount": 2500.0,
            "location": "Dubai",
            "category": "Electronics",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "quantum_result" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        qr = data["quantum_result"]
        assert qr["classification"] in ("FRAUD", "SAFE")
        assert 0.0 <= qr["fraud_probability"] <= 1.0

    def test_analyze_low_amount(self, api_client):
        """Low amount → lower risk."""
        resp = api_client.post("/api/analyze", json={
            "user_id": "USER_TEST",
            "amount": 15.0,
            "location": "Mumbai",
            "category": "Grocery",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Very low amount should not be CRITICAL
        assert data["risk_level"] != "CRITICAL"


class TestImpactTimeline:
    def test_impact_timeline_endpoint(self, api_client):
        """GET /api/impact/timeline returns time-series data."""
        resp = api_client.get("/api/impact/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert "timeline" in data
        assert isinstance(data["timeline"], list)


class TestDashboard:
    def test_dashboard_loads(self, api_client):
        """GET / returns HTML dashboard."""
        resp = api_client.get("/")
        assert resp.status_code == 200
        assert "QuantGuard" in resp.text
        assert "Green Bharat" in resp.text
        assert "System Log Anomalies" in resp.text


class TestXPack:
    def test_xpack_status(self, api_client):
        """GET /api/xpack/status returns capabilities."""
        resp = api_client.get("/api/xpack/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data
        assert "live_rag" in data["capabilities"]


class TestLogAnomalyEndpoints:
    """Tests for the log anomaly detection pipeline API endpoints."""

    def test_log_alerts_endpoint(self, api_client):
        """GET /api/logs/alerts returns 200 with expected structure."""
        resp = api_client.get("/api/logs/alerts?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert "count" in data
        assert isinstance(data["alerts"], list)

    def test_log_stats_endpoint(self, api_client):
        """GET /api/logs/stats returns summary statistics."""
        resp = api_client.get("/api/logs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "by_service" in data
        assert "by_severity" in data

    def test_log_services_endpoint(self, api_client):
        """GET /api/logs/services returns per-service health."""
        resp = api_client.get("/api/logs/services")
        assert resp.status_code == 200
        data = resp.json()
        assert "services" in data
        assert isinstance(data["services"], list)


class TestKafkaProducerModule:
    """Tests for the kafka_producer.py module."""

    def test_generate_transaction(self):
        """kafka_producer.generate_transaction returns valid structure."""
        from kafka_producer import generate_transaction, _build_profiles
        profiles = _build_profiles()
        tx = generate_transaction(profiles)
        assert "user_id" in tx
        assert "amount" in tx
        assert "location" in tx
        assert "category" in tx
        assert "timestamp" in tx
        assert "is_suspicious_flag" in tx
        assert isinstance(tx["amount"], float)
        assert tx["amount"] > 0

    def test_build_profiles(self):
        """_build_profiles returns 15 user profiles."""
        from kafka_producer import _build_profiles
        profiles = _build_profiles()
        assert len(profiles) == 15
        assert "USER_100" in profiles
        assert "USER_114" in profiles
        for uid, p in profiles.items():
            assert "home" in p
            assert "avg" in p
            assert "cats" in p

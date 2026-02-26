"""
QuantGuard Test Suite — Shared Fixtures
========================================
Provides reusable test fixtures for the quantum classifier,
ML anomaly scorer, and FastAPI application.
"""

import os
import sys
import json
import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def quantum_simulator():
    """Session-scoped numpy QuantumSimulator instance."""
    from quantum_classifier import QiskitSimulator
    return QiskitSimulator(num_qubits=2)


@pytest.fixture(scope="session")
def quantum_classifier():
    """Session-scoped QuantumRiskClassifier (uses numpy sim fallback)."""
    # Force simulator mode (no IBM key needed for tests)
    old_key = os.environ.pop("IBMQ_API_KEY", None)
    from quantum_classifier import QuantumRiskClassifier
    clf = QuantumRiskClassifier()
    if old_key:
        os.environ["IBMQ_API_KEY"] = old_key
    return clf


@pytest.fixture
def sample_transactions():
    """List of sample transactions covering normal + fraud patterns."""
    return [
        {"user_id": "USER_100", "amount": 42.50, "location": "Mumbai",
         "category": "Grocery", "timestamp": "2026-02-10T10:00:00",
         "is_suspicious_flag": False},
        {"user_id": "USER_101", "amount": 4800.00, "location": "Dubai",
         "category": "Electronics", "timestamp": "2026-02-10T10:01:00",
         "is_suspicious_flag": True},
        {"user_id": "USER_102", "amount": 150.00, "location": "London",
         "category": "Dining", "timestamp": "2026-02-10T10:02:00",
         "is_suspicious_flag": False},
        {"user_id": "USER_103", "amount": 9500.00, "location": "Singapore",
         "category": "Travel", "timestamp": "2026-02-10T10:03:00",
         "is_suspicious_flag": True},
    ]


@pytest.fixture
def api_client():
    """FastAPI TestClient — imports main_api.app."""
    from fastapi.testclient import TestClient
    from main_api import app
    return TestClient(app)

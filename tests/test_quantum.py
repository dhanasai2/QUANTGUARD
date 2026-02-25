"""
QuantGuard — Quantum Simulator + Classifier Tests
===================================================
Tests the numpy statevector simulator gates, VQC circuit,
and fraud classification decision boundary.
"""

import numpy as np
import pytest


class TestQuantumSimulator:
    """Test individual quantum gates on the numpy simulator."""

    def test_initial_state_is_zero(self, quantum_simulator):
        """Simulator starts in |00⟩ state."""
        quantum_simulator.reset()
        assert abs(quantum_simulator.state[0] - 1.0) < 1e-10
        assert sum(abs(quantum_simulator.state[i]) for i in range(1, 4)) < 1e-10

    def test_hadamard_creates_superposition(self, quantum_simulator):
        """H|0⟩ = (|0⟩ + |1⟩)/√2 — equal probabilities."""
        quantum_simulator.reset()
        quantum_simulator.h(0)
        probs = quantum_simulator.get_probabilities()
        # After H on q0: |00⟩ and |10⟩ should each be ~50%
        assert abs(probs["00"] - 0.5) < 1e-10
        assert abs(probs["10"] - 0.5) < 1e-10

    def test_rz_gate_phase(self, quantum_simulator):
        """Rz(π) on |0⟩ only adds global phase — probabilities unchanged."""
        quantum_simulator.reset()
        quantum_simulator.rz(np.pi, 0)
        probs = quantum_simulator.get_probabilities()
        assert abs(probs["00"] - 1.0) < 1e-10

    def test_ry_gate_rotation(self, quantum_simulator):
        """Ry(π)|0⟩ = |1⟩ — full rotation to excited state."""
        quantum_simulator.reset()
        quantum_simulator.ry(np.pi, 0)
        probs = quantum_simulator.get_probabilities()
        assert abs(probs["10"] - 1.0) < 1e-9  # qubit 0 flipped

    def test_rx_gate_rotation(self, quantum_simulator):
        """Rx(π)|0⟩ = -i|1⟩ — full rotation to excited state."""
        quantum_simulator.reset()
        quantum_simulator.rx(np.pi, 0)
        probs = quantum_simulator.get_probabilities()
        assert abs(probs["10"] - 1.0) < 1e-9

    def test_cnot_entanglement(self, quantum_simulator):
        """H + CNOT creates Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
        quantum_simulator.reset()
        quantum_simulator.h(0)
        quantum_simulator.cx(0, 1)
        probs = quantum_simulator.get_probabilities()
        assert abs(probs["00"] - 0.5) < 1e-10
        assert abs(probs["11"] - 0.5) < 1e-10
        assert abs(probs["01"]) < 1e-10
        assert abs(probs["10"]) < 1e-10

    def test_measurement_shot_distribution(self, quantum_simulator):
        """1024 shots of |0⟩ state should all yield '00'."""
        quantum_simulator.reset()
        counts = quantum_simulator.measure(shots=1024)
        assert counts.get("00", 0) == 1024

    def test_measurement_stochastic(self, quantum_simulator):
        """H|0⟩ measurement over many shots should be ~50/50."""
        quantum_simulator.reset()
        quantum_simulator.h(0)
        counts = quantum_simulator.measure(shots=10000)
        p00 = counts.get("00", 0) / 10000
        p10 = counts.get("10", 0) / 10000
        assert abs(p00 - 0.5) < 0.05  # within 5% tolerance
        assert abs(p10 - 0.5) < 0.05

    def test_state_normalization(self, quantum_simulator):
        """State vector should remain normalized after any gate sequence."""
        quantum_simulator.reset()
        quantum_simulator.h(0)
        quantum_simulator.ry(1.23, 1)
        quantum_simulator.cx(0, 1)
        quantum_simulator.rz(2.5, 0)
        norm = np.linalg.norm(quantum_simulator.state)
        assert abs(norm - 1.0) < 1e-10


class TestQuantumClassifier:
    """Test the VQC fraud classifier decision boundary and output format."""

    def test_safe_classification(self, quantum_classifier):
        """Low-risk features → classified as SAFE."""
        result = quantum_classifier.classify_transaction(np.array([0.05, 0.05]))
        assert result["classification"] == "SAFE"
        assert result["fraud_probability"] < 0.45

    def test_fraud_classification(self, quantum_classifier):
        """High-risk features → classified as FRAUD."""
        result = quantum_classifier.classify_transaction(np.array([0.95, 0.90]))
        assert result["classification"] == "FRAUD"
        assert result["fraud_probability"] > 0.45

    def test_output_structure(self, quantum_classifier):
        """Verify all required fields are present in classification result."""
        result = quantum_classifier.classify_transaction(np.array([0.5, 0.5]))
        required = [
            "classification", "fraud_probability", "quantum_states",
            "total_shots", "circuit_diagram", "gate_sequence",
            "state_analysis", "input_features", "circuit_depth",
            "total_gates", "num_qubits", "backend",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_probability_range(self, quantum_classifier):
        """Fraud probability should be in [0, 1]."""
        for amt in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = quantum_classifier.classify_transaction(np.array([amt, 0.3]))
            assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_state_analysis_bloch(self, quantum_classifier):
        """State analysis should include Bloch angles for both qubits."""
        result = quantum_classifier.classify_transaction(np.array([0.5, 0.5]))
        sa = result["state_analysis"]
        assert "bloch_angles" in sa
        assert "qubit_0" in sa["bloch_angles"]
        assert "qubit_1" in sa["bloch_angles"]
        assert "theta" in sa["bloch_angles"]["qubit_0"]

    def test_feature_clipping(self, quantum_classifier):
        """Features outside [0,1] are clipped — no crash."""
        result = quantum_classifier.classify_transaction(np.array([-0.5, 1.5]))
        assert result["classification"] in ("FRAUD", "SAFE")

    def test_probabilities_sum_to_one(self, quantum_classifier):
        """Measurement probabilities across all 4 basis states sum to 1."""
        result = quantum_classifier.classify_transaction(np.array([0.6, 0.4]))
        probs = result["state_analysis"]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.02  # within 2% (shot noise)

    def test_circuit_info(self, quantum_classifier):
        """get_circuit_info() returns correct metadata."""
        info = quantum_classifier.get_circuit_info()
        assert info["total_parameters"] == 4
        assert info["hilbert_space_dimension"] == 4
        assert "ZZFeatureMap" in info["feature_map"]
        assert "RealAmplitudes" in info["ansatz"]

    def test_deterministic_decision(self, quantum_classifier):
        """Same features should produce same classification (sim is seeding-independent)."""
        r1 = quantum_classifier.classify_transaction(np.array([0.1, 0.1]))
        r2 = quantum_classifier.classify_transaction(np.array([0.1, 0.1]))
        assert r1["classification"] == r2["classification"]

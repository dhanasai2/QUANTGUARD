"""
QuantGuard — VQC Weight Optimiser
==================================
Trains the 2-qubit Variational Quantum Classifier using a synthetic
labelled dataset and COBYLA optimisation (500 iterations).

The output weights are used in quantum_classifier.py.

Usage:
    python train_vqc.py              # Run training (prints optimal weights)
    python train_vqc.py --eval       # Evaluate existing weights on test set

The training dataset mirrors the data_source.py distribution:
  • Normal transactions  → label 0 (SAFE)
  • Fraudulent patterns  → label 1 (FRAUD)

Features (normalised to [0, 1]):
  • amount_norm  — transaction amount relative to user max
  • freq_norm    — transaction velocity relative to window size
"""

import numpy as np
from scipy.optimize import minimize
import argparse
import json

# ═══════════════════════════════════════════════════════════════════════════
#  Statevector Simulator (same math as quantum_classifier.py)
# ═══════════════════════════════════════════════════════════════════════════

def statevector_vqc(features, weights, num_qubits=2):
    """
    Evaluate the VQC circuit and return P(qubit_0 = |1⟩).
    Uses exact statevector math (no sampling noise) for clean gradients.
    """
    dim = 2 ** num_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    def _apply_single(st, gate, qubit):
        ops = [np.eye(2, dtype=complex) for _ in range(num_qubits)]
        ops[qubit] = gate
        full = ops[0]
        for op in ops[1:]:
            full = np.kron(full, op)
        return full @ st

    def H():
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def Rz(theta):
        return np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]], dtype=complex)

    def Ry(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def CX(st, ctrl, tgt):
        cnot = np.eye(dim, dtype=complex)
        for i in range(dim):
            bits = list(format(i, f"0{num_qubits}b"))
            if bits[ctrl] == "1":
                bits[tgt] = "0" if bits[tgt] == "1" else "1"
                j = int("".join(bits), 2)
                cnot[i, i] = 0
                cnot[i, j] = 1
        return cnot @ st

    # ZZFeatureMap (2 reps)
    for _ in range(2):
        for i in range(num_qubits):
            state = _apply_single(state, H(), i)
            state = _apply_single(state, Rz(2.0 * features[i]), i)
        state = CX(state, 0, 1)
        state = _apply_single(state, Rz(2.0 * (np.pi - features[0]) * (np.pi - features[1])), 1)
        state = CX(state, 0, 1)

    # RealAmplitudes ansatz (4 weights)
    state = _apply_single(state, Ry(weights[0]), 0)
    state = _apply_single(state, Ry(weights[1]), 1)
    state = CX(state, 0, 1)
    state = _apply_single(state, Ry(weights[2]), 0)
    state = _apply_single(state, Ry(weights[3]), 1)

    # P(qubit_0 = |1⟩) = P(|10⟩) + P(|11⟩)
    probs = np.abs(state) ** 2
    p_fraud = probs[2] + probs[3]  # |10⟩ and |11⟩
    return float(p_fraud)


# ═══════════════════════════════════════════════════════════════════════════
#  Training Dataset (mirrors data_source.py distribution)
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_data(n_samples=200, seed=42):
    """
    Generate labelled training pairs: (features, label).
    Features: [amount_norm, freq_norm] ∈ [0, 1]
    Label: 0 = SAFE, 1 = FRAUD
    """
    rng = np.random.RandomState(seed)
    X, y = [], []

    n_safe = int(n_samples * 0.65)
    n_fraud = n_samples - n_safe

    # Safe transactions: low-to-moderate amount, moderate frequency
    for _ in range(n_safe):
        amt = rng.uniform(0.02, 0.45)
        freq = rng.uniform(0.05, 0.50)
        X.append([amt, freq])
        y.append(0)

    # Fraudulent transactions: higher amount, varied velocity
    for _ in range(n_fraud):
        pattern = rng.choice(["high_val", "velocity", "mixed"])
        if pattern == "high_val":
            amt = rng.uniform(0.55, 1.0)
            freq = rng.uniform(0.10, 0.60)
        elif pattern == "velocity":
            amt = rng.uniform(0.30, 0.80)
            freq = rng.uniform(0.55, 1.0)
        else:
            amt = rng.uniform(0.50, 0.95)
            freq = rng.uniform(0.40, 0.90)
        X.append([amt, freq])
        y.append(1)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  Cost Function (Binary Cross-Entropy)
# ═══════════════════════════════════════════════════════════════════════════

def cost_function(weights, X, y):
    """Binary cross-entropy loss over the dataset."""
    eps = 1e-8
    total_loss = 0.0
    for xi, yi in zip(X, y):
        p_fraud = statevector_vqc(xi, weights)
        p_fraud = np.clip(p_fraud, eps, 1 - eps)
        loss = -(yi * np.log(p_fraud) + (1 - yi) * np.log(1 - p_fraud))
        total_loss += loss
    return total_loss / len(X)


# ═══════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train(max_iter=500, n_samples=200, seed=42):
    """Train VQC weights using COBYLA optimiser."""
    print("=" * 60)
    print("  QuantGuard VQC Training")
    print("=" * 60)

    X_train, y_train = generate_training_data(n_samples, seed)
    print(f"  Dataset: {n_samples} samples ({sum(y_train == 0)} safe, {sum(y_train == 1)} fraud)")

    # Initial random weights
    rng = np.random.RandomState(seed + 1)
    w0 = rng.uniform(-np.pi, np.pi, size=4)
    print(f"  Initial weights: {np.round(w0, 4).tolist()}")

    iteration_log = []

    def callback(xk):
        loss = cost_function(xk, X_train, y_train)
        iteration_log.append(loss)
        if len(iteration_log) % 50 == 0:
            print(f"    Iteration {len(iteration_log):>4d}  |  Loss: {loss:.6f}")

    print(f"\n  Optimising (COBYLA, max_iter={max_iter})...\n")
    result = minimize(
        cost_function,
        w0,
        args=(X_train, y_train),
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 0.5},
        callback=callback,
    )

    optimal_weights = result.x.tolist()
    final_loss = result.fun

    print(f"\n  {'─' * 50}")
    print(f"  ✅ Training complete")
    print(f"  Final loss:    {final_loss:.6f}")
    print(f"  Optimal weights: {[round(w, 4) for w in optimal_weights]}")
    print(f"  Iterations:    {len(iteration_log)}")

    return optimal_weights, final_loss, X_train, y_train


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(weights, X, y, threshold=0.45):
    """Evaluate accuracy of given weights on a dataset."""
    correct = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for xi, yi in zip(X, y):
        p_fraud = statevector_vqc(xi, weights)
        pred = 1 if p_fraud > threshold else 0
        if pred == yi:
            correct += 1
        if pred == 1 and yi == 1: tp += 1
        if pred == 1 and yi == 0: fp += 1
        if pred == 0 and yi == 0: tn += 1
        if pred == 0 and yi == 1: fn += 1

    acc = correct / len(y)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n  📊 Evaluation (threshold={threshold})")
    print(f"  {'─' * 40}")
    print(f"  Accuracy:  {acc:.1%}  ({correct}/{len(y)})")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuantGuard VQC")
    parser.add_argument("--eval", action="store_true", help="Evaluate existing weights only")
    parser.add_argument("--iter", type=int, default=500, help="Max COBYLA iterations")
    parser.add_argument("--samples", type=int, default=200, help="Training samples")
    args = parser.parse_args()

    if args.eval:
        # Evaluate the production weights from quantum_classifier.py
        production_weights = [-1.1549, -1.5882, -2.1251, -2.3295]
        print("  Evaluating production weights:", production_weights)
        X, y = generate_training_data(args.samples)
        evaluate(production_weights, X, y)
    else:
        weights, loss, X, y = train(max_iter=args.iter, n_samples=args.samples)
        evaluate(weights, X, y)

        # Save weights
        out = {
            "weights": [round(w, 4) for w in weights],
            "loss": round(loss, 6),
            "method": "COBYLA",
            "iterations": args.iter,
            "samples": args.samples,
            "threshold": 0.45,
        }
        print(f"\n  Paste into quantum_classifier.py:")
        print(f"    self.weights = {out['weights']}")
        print(f"\n  Full result: {json.dumps(out, indent=2)}")

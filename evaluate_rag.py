"""
QuantGuard — RAGAS Evaluation Suite
=====================================
Evaluates the quality of the RAG pipeline using RAGAS-inspired metrics.

Metrics computed:
  • Faithfulness     — Is the answer grounded in the retrieved context?
  • Answer Relevancy — Is the answer relevant to the question asked?
  • Context Precision — Are the retrieved documents relevant to the query?
  • Context Recall   — Does the context contain information needed for the answer?
  • Overall Score    — Weighted harmonic mean of all 4 metrics

Each metric is scored 0.0–1.0.

Usage:
  python evaluate_rag.py                  # Run full evaluation
  python evaluate_rag.py --json           # Output machine-readable JSON
  python evaluate_rag.py --verbose        # Show per-question details

Reference:  RAGAS — https://docs.ragas.io/en/latest/concepts/metrics/
"""

import json
import os
import sys
import re
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
#  Golden Test Set — Ground-Truth QA Pairs for Evaluation
# ═══════════════════════════════════════════════════════════════════════════

GOLDEN_QA_SET: List[Dict[str, Any]] = [
    {
        "question": "What is the threshold amount that triggers mandatory fraud review?",
        "ground_truth": "Transactions exceeding ₹50,000 require mandatory manual review and enhanced due diligence checks.",
        "expected_context_keywords": ["threshold", "₹50,000", "review", "mandatory"],
    },
    {
        "question": "What are the penalties for non-compliance with RBI fraud reporting guidelines?",
        "ground_truth": "Non-compliance can result in fines up to ₹1 crore, license suspension, and mandatory audit requirements per RBI Master Directions.",
        "expected_context_keywords": ["penalty", "fine", "non-compliance", "RBI"],
    },
    {
        "question": "How should suspicious transaction reports (STR) be filed?",
        "ground_truth": "STRs must be filed with the Financial Intelligence Unit (FIU-IND) within 7 working days of detecting suspicious activity.",
        "expected_context_keywords": ["STR", "suspicious", "FIU", "report"],
    },
    {
        "question": "What risk factors classify a transaction as high-risk?",
        "ground_truth": "High-risk factors include unusual transaction amounts, cross-border transfers, new account holders, velocity anomalies, and transactions in flagged categories.",
        "expected_context_keywords": ["risk", "high-risk", "unusual", "cross-border", "anomaly"],
    },
    {
        "question": "What customer due diligence (CDD) measures are required for high-value transactions?",
        "ground_truth": "Enhanced CDD includes identity verification, source of funds documentation, beneficial ownership identification, and ongoing transaction monitoring.",
        "expected_context_keywords": ["CDD", "due diligence", "identity", "verification", "KYC"],
    },
    {
        "question": "How does QuantGuard classify transaction risk using quantum computing?",
        "ground_truth": "QuantGuard uses a 2-qubit Variational Quantum Circuit (VQC) with ZZFeatureMap encoding and RealAmplitudes ansatz, classifying transactions as FRAUD when P(|1⟩) exceeds 0.45.",
        "expected_context_keywords": ["quantum", "VQC", "qubit", "classification", "fraud"],
    },
    {
        "question": "What is the Green Bharat sustainability impact of detecting fraud?",
        "ground_truth": "Each fraud detected protects funds that are quantified as sustainability impact: ₹1 lakh ≈ 50,000L clean water, ₹500 ≈ 1 tree planted, ₹1,000 ≈ 2.5 kg CO₂ offset.",
        "expected_context_keywords": ["green", "sustainability", "tree", "CO₂", "water"],
    },
    {
        "question": "What ML features does QuantGuard use for anomaly detection?",
        "ground_truth": "QuantGuard uses 6 ML features: Z-score, IQR outlier score, percentile rank, geo-entropy, spending velocity, and category deviation from user profile.",
        "expected_context_keywords": ["ML", "anomaly", "Z-score", "IQR", "feature"],
    },
    {
        "question": "How does the RAG pipeline retrieve regulatory context?",
        "ground_truth": "The RAG pipeline chunks policy documents, scores them by keyword overlap with the query, retrieves top-k relevant chunks, and augments the LLM prompt with this context.",
        "expected_context_keywords": ["RAG", "retrieval", "policy", "context", "document"],
    },
    {
        "question": "What compliance regulations apply to real-time fraud monitoring systems?",
        "ground_truth": "Systems must comply with RBI Master Directions on KYC, PML Act 2002, Information Technology Act 2000, and SEBI guidelines for market surveillance.",
        "expected_context_keywords": ["compliance", "RBI", "regulation", "KYC", "PML"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation Metrics (RAGAS-inspired)
# ═══════════════════════════════════════════════════════════════════════════

def _keyword_overlap(text_a: str, keywords: List[str]) -> float:
    """Fraction of keywords found in text (case-insensitive)."""
    if not keywords:
        return 1.0
    text_lower = text_a.lower()
    found = sum(1 for k in keywords if k.lower() in text_lower)
    return found / len(keywords)


def _sentence_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF-like cosine similarity between two texts."""
    words_a = set(re.findall(r'\w+', text_a.lower()))
    words_b = set(re.findall(r'\w+', text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / (len(words_a | words_b) ** 0.5)


def evaluate_faithfulness(answer: str, context_chunks: List[str]) -> float:
    """
    Faithfulness: Is the generated answer grounded in the retrieved context?

    Checks what fraction of claim-words in the answer appear in the context.
    Score 1.0 = fully grounded.   Score 0.0 = hallucinated.
    """
    if not answer or not context_chunks:
        return 0.0
    context_text = " ".join(context_chunks).lower()
    answer_words = set(re.findall(r'\w{3,}', answer.lower()))
    # Remove common stopwords
    stopwords = {"the", "and", "for", "that", "this", "with", "are", "was",
                 "has", "have", "been", "from", "they", "their", "which",
                 "about", "into", "more", "also", "can", "should", "would",
                 "must", "may", "not", "all", "any", "each", "such", "than"}
    answer_words -= stopwords
    if not answer_words:
        return 1.0
    grounded = sum(1 for w in answer_words if w in context_text)
    return min(1.0, grounded / len(answer_words))


def evaluate_answer_relevancy(answer: str, question: str) -> float:
    """
    Answer Relevancy: Is the answer relevant to the question?

    Uses keyword overlap between question and answer as a proxy.
    """
    if not answer or not question:
        return 0.0
    return _sentence_similarity(answer, question)


def evaluate_context_precision(
    context_chunks: List[str], expected_keywords: List[str]
) -> float:
    """
    Context Precision: Are the retrieved chunks relevant to the query?

    Measures what fraction of retrieved chunks contain expected keywords.
    """
    if not context_chunks:
        return 0.0
    relevant = 0
    for chunk in context_chunks:
        overlap = _keyword_overlap(chunk, expected_keywords)
        if overlap >= 0.3:  # at least 30% keyword match
            relevant += 1
    return relevant / len(context_chunks)


def evaluate_context_recall(
    context_chunks: List[str], ground_truth: str
) -> float:
    """
    Context Recall: Does the context contain information needed for the answer?

    Checks if key phrases from the ground truth appear in the retrieved context.
    """
    if not context_chunks or not ground_truth:
        return 0.0
    context_text = " ".join(context_chunks).lower()
    gt_words = set(re.findall(r'\w{4,}', ground_truth.lower()))
    stopwords = {"the", "and", "for", "that", "this", "with", "are", "was",
                 "from", "their", "which", "about", "into", "more", "also"}
    gt_words -= stopwords
    if not gt_words:
        return 1.0
    found = sum(1 for w in gt_words if w in context_text)
    return found / len(gt_words)


def harmonic_mean(scores: List[float], weights: Optional[List[float]] = None) -> float:
    """Weighted harmonic mean of scores (RAGAS composite)."""
    if not scores:
        return 0.0
    if weights is None:
        weights = [1.0] * len(scores)
    total_w = sum(weights)
    denom = sum(w / max(s, 1e-6) for s, w in zip(scores, weights))
    return total_w / denom if denom > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  RAG Pipeline Interface
# ═══════════════════════════════════════════════════════════════════════════

def _get_rag_answer(question: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Query the QuantGuard RAG pipeline and return answer + context.
    Retries on rate-limit (429) errors with exponential backoff.

    Returns:
        {"answer": str, "context_chunks": list[str], "latency_ms": float}
    """
    # Try live API first (returns answer + retrieved context chunks)
    for attempt in range(max_retries + 1):
        try:
            import urllib.request
            payload = json.dumps({"question": question}).encode()
            req = urllib.request.Request(
                "http://localhost:8000/api/rag/query",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                answer = data.get("answer", data.get("response", ""))
                # If rate-limited, the answer contains "429" — retry
                if "429" in answer and "rate limit" in answer.lower() and attempt < max_retries:
                    wait = 8 * (attempt + 1)
                    print(f"         [Rate limited — waiting {wait}s before retry]")
                    time.sleep(wait)
                    continue
                return {
                    "answer": answer,
                    "context_chunks": data.get("context_chunks", data.get("context", [])),
                    "latency_ms": data.get("latency_ms", 0),
                }
        except Exception:
            break

    # Fallback: call llm_engine directly (no server needed)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llm_engine import GroqFraudAnalyzer
        analyzer = GroqFraudAnalyzer()
        chunks = analyzer.retrieve(question, top_k=5)
        context_texts = [c["content"] for c in chunks] if chunks else []
        answer = analyzer.rag_query(question)
        return {
            "answer": answer,
            "context_chunks": context_texts,
            "latency_ms": 0,
        }
    except Exception as e:
        return {"answer": "", "context_chunks": [], "latency_ms": 0, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  Main Evaluation Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(verbose: bool = False) -> Dict[str, Any]:
    """
    Execute RAGAS evaluation over the golden test set.

    Returns comprehensive results dict with per-question and aggregate scores.
    """
    results = []
    totals = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    print("=" * 70)
    print("  QuantGuard — RAGAS Evaluation Suite")
    print("  Golden test set: {} questions".format(len(GOLDEN_QA_SET)))
    print("=" * 70)
    print()

    for i, qa in enumerate(GOLDEN_QA_SET, 1):
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        expected_kw = qa["expected_context_keywords"]

        print(f"  [{i}/{len(GOLDEN_QA_SET)}] {question[:65]}...")

        t0 = time.time()
        response = _get_rag_answer(question)
        elapsed = (time.time() - t0) * 1000

        answer = response["answer"]
        context = response["context_chunks"]
        # Normalize context to list of strings
        if context and isinstance(context[0], dict):
            context = [c.get("content", str(c)) for c in context]

        # Compute RAGAS metrics
        faith = evaluate_faithfulness(answer, context)
        relevancy = evaluate_answer_relevancy(answer, question)
        precision = evaluate_context_precision(context, expected_kw)
        recall = evaluate_context_recall(context, ground_truth)
        overall = harmonic_mean([faith, relevancy, precision, recall])

        totals["faithfulness"].append(faith)
        totals["answer_relevancy"].append(relevancy)
        totals["context_precision"].append(precision)
        totals["context_recall"].append(recall)

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer[:300],
            "num_context_chunks": len(context),
            "metrics": {
                "faithfulness": round(faith, 3),
                "answer_relevancy": round(relevancy, 3),
                "context_precision": round(precision, 3),
                "context_recall": round(recall, 3),
                "overall": round(overall, 3),
            },
            "latency_ms": round(elapsed, 1),
        }
        results.append(result)

        if verbose:
            print(f"         Answer:    {answer[:80]}...")
            print(f"         Context:   {len(context)} chunks")
        print(f"         Faith={faith:.2f}  Rel={relevancy:.2f}  "
              f"Prec={precision:.2f}  Recall={recall:.2f}  "
              f"Overall={overall:.2f}  ({elapsed:.0f}ms)")
        print()

        # Delay between questions to avoid LLM rate limits (Groq free tier)
        if i < len(GOLDEN_QA_SET):
            time.sleep(3)

    # ── Aggregate Scores ────────────────────────────────────────────────
    agg = {}
    for metric, scores in totals.items():
        agg[metric] = round(sum(scores) / len(scores), 3) if scores else 0.0
    agg["overall"] = round(harmonic_mean(list(agg.values())), 3)

    print("=" * 70)
    print("  AGGREGATE RAGAS SCORES")
    print("=" * 70)
    print(f"  Faithfulness      : {agg['faithfulness']:.3f}")
    print(f"  Answer Relevancy  : {agg['answer_relevancy']:.3f}")
    print(f"  Context Precision : {agg['context_precision']:.3f}")
    print(f"  Context Recall    : {agg['context_recall']:.3f}")
    print(f"  ─────────────────────────────")
    print(f"  Overall (H-mean)  : {agg['overall']:.3f}")
    print("=" * 70)

    # Quality verdict
    score = agg["overall"]
    if score >= 0.8:
        verdict = "EXCELLENT — Production-ready RAG pipeline"
    elif score >= 0.6:
        verdict = "GOOD — Solid retrieval with room for improvement"
    elif score >= 0.4:
        verdict = "FAIR — Retrieval needs chunking/embedding tuning"
    else:
        verdict = "POOR — Major RAG pipeline issues detected"
    print(f"  Verdict: {verdict}")
    print()

    return {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(GOLDEN_QA_SET),
        "aggregate_scores": agg,
        "verdict": verdict,
        "per_question": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantGuard RAGAS Evaluation")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Save results to file")
    args = parser.parse_args()

    results = run_evaluation(verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {args.output}")

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))

"""
QuantGuard - Groq LLM Engine with RAG
======================================
Provides AI-powered fraud analysis using Groq's ultra-fast inference.
Implements Retrieval-Augmented Generation (RAG) over regulatory policy
documents for explainable, compliance-aware fraud detection insights.

Features:
- Fraud risk explanation using transaction context + quantum results
- RAG-based regulatory query answering
- Alert summary generation for executive dashboards
- Fallback explanations when LLM is unavailable
"""

import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ── Sentence-Transformer Embedding Model (lazy-loaded) ─────────────────
_embed_model = None


def _get_embed_model():
    """Lazy-load sentence-transformers model for semantic retrieval."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[RAG] Loaded embedding model: all-MiniLM-L6-v2 (384d)")
        except ImportError:
            print("[RAG] sentence-transformers not installed — using BM25 fallback")
            _embed_model = False  # sentinel: tried and failed
    return _embed_model if _embed_model is not False else None


class GroqFraudAnalyzer:
    """Groq-powered LLM engine for fraud analysis with RAG capabilities."""

    def __init__(self):
        self.client = None
        self.model = "llama-3.3-70b-versatile"
        self.policy_store = []
        self._init_client()
        self._load_policy_documents()

    def _init_client(self):
        """Initialize Groq API client."""
        if not GROQ_AVAILABLE:
            print("[LLM] groq package not installed. Run: pip install groq")
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[LLM] GROQ_API_KEY not found in .env file. LLM features disabled.")
            return

        try:
            self.client = Groq(api_key=api_key)
            print(f"[LLM] Connected to Groq API (model: {self.model})")
        except Exception as e:
            print(f"[LLM] Failed to initialize Groq client: {e}")

    def _load_policy_documents(self):
        """Load and chunk regulatory policy documents for RAG retrieval."""
        policy_dir = os.path.join("data", "policies")
        if not os.path.exists(policy_dir):
            print("[RAG] No policy documents found in data/policies/")
            return

        for filename in sorted(os.listdir(policy_dir)):
            filepath = os.path.join(policy_dir, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = self._chunk_document(content, chunk_size=400, overlap=50)
                for i, chunk in enumerate(chunks):
                    self.policy_store.append({
                        "source": filename,
                        "chunk_id": i,
                        "content": chunk,
                        "keywords": set(chunk.lower().split()),
                        "embedding": None,  # computed lazily below
                    })
            except Exception as e:
                print(f"[RAG] Error loading {filename}: {e}")

        # Pre-compute embeddings for all chunks if model is available
        model = _get_embed_model()
        if model and self.policy_store:
            texts = [d["content"] for d in self.policy_store]
            embeddings = model.encode(texts, show_progress_bar=False,
                                      normalize_embeddings=True)
            for doc, emb in zip(self.policy_store, embeddings):
                doc["embedding"] = emb

        print(f"[RAG] Loaded {len(self.policy_store)} document chunks from {policy_dir}")

    @staticmethod
    def _chunk_document(text, chunk_size=400, overlap=50):
        """Split document into overlapping word-level chunks."""
        words = text.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]

    # ── Retrieval (Hybrid: Semantic Embedding + BM25 Keyword) ───────────

    def retrieve(self, query, top_k=3):
        """Retrieve relevant policy chunks using hybrid semantic + keyword scoring.

        Uses sentence-transformer cosine similarity when available,
        combined with BM25-style keyword overlap for robust retrieval.
        Falls back to pure keyword scoring if embeddings unavailable.
        """
        if not self.policy_store:
            return []

        # ── Semantic scoring (cosine similarity via embeddings) ─────────
        model = _get_embed_model()
        has_embeddings = model and self.policy_store[0].get("embedding") is not None

        if has_embeddings:
            query_emb = model.encode([query], normalize_embeddings=True)[0]
            sem_scores = []
            for doc in self.policy_store:
                emb = doc.get("embedding")
                if emb is not None:
                    sim = float(np.dot(query_emb, emb))  # cosine (pre-normalized)
                    sem_scores.append(sim)
                else:
                    sem_scores.append(0.0)
        else:
            sem_scores = [0.0] * len(self.policy_store)

        # ── BM25-style keyword scoring ──────────────────────────────────
        query_keywords = set(query.lower().split())
        kw_scores = []
        for doc in self.policy_store:
            overlap = len(query_keywords & doc["keywords"])
            kw_scores.append(overlap)

        # ── Hybrid fusion: 0.7 × semantic + 0.3 × keyword (normalized) ──
        max_sem = max(sem_scores) if any(s > 0 for s in sem_scores) else 1.0
        max_kw = max(kw_scores) if any(s > 0 for s in kw_scores) else 1.0

        scored = []
        for i, doc in enumerate(self.policy_store):
            sem_norm = sem_scores[i] / max(max_sem, 1e-9)
            kw_norm = kw_scores[i] / max(max_kw, 1e-9)
            hybrid = 0.7 * sem_norm + 0.3 * kw_norm if has_embeddings else kw_norm
            if hybrid > 0.01:
                scored.append((hybrid, doc))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored[:top_k]]

    # ── Fraud Explanation ───────────────────────────────────────────────

    def explain_fraud_risk(self, transaction, quantum_result):
        """Generate an LLM-powered explanation for a fraud detection result."""
        if not self.client:
            return self._fallback_explanation(transaction, quantum_result)

        # Retrieve relevant regulatory context
        query = (
            f"fraud detection {transaction.get('category', '')} "
            f"transaction amount {transaction.get('amount', 0)} "
            f"{transaction.get('location', '')}"
        )
        relevant_docs = self.retrieve(query)
        context = "\n---\n".join(
            [f"[{d['source']}]: {d['content'][:400]}" for d in relevant_docs]
        )

        prompt = f"""You are QuantGuard, an AI fraud detection analyst for the Hack for Green Bharat initiative. Your mission is to protect India's financial ecosystem so that resources can flow toward renewable energy, clean water, and sustainable development.

Analyze this transaction and provide a concise risk assessment.

TRANSACTION:
- User: {transaction.get('user_id')}
- Amount: ${transaction.get('amount')}
- Location: {transaction.get('location')}
- Category: {transaction.get('category')}
- Timestamp: {transaction.get('timestamp', 'N/A')}

QUANTUM CIRCUIT RESULT:
- Classification: {quantum_result.get('classification', 'N/A')}
- Fraud Probability: {quantum_result.get('fraud_probability', 'N/A')}
- Quantum States: {quantum_result.get('quantum_states', {})}

REGULATORY CONTEXT:
{context}

Provide a concise risk analysis (3-4 sentences) covering:
1. Why this transaction was flagged
2. Interpretation of the quantum classification
3. Recommended action per regulatory guidelines
4. Brief note on how preventing this fraud protects resources for India's green economy"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial fraud detection AI assistant for Green Bharat. Be concise and actionable. Every fraud you detect protects India's sustainable development resources.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=250,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] Groq API error in explain_fraud_risk: {type(e).__name__}: {e}")
            return self._fallback_explanation(transaction, quantum_result)

    # ── RAG Query ───────────────────────────────────────────────────────

    def rag_query(self, question):
        """Answer questions using RAG over regulatory policy documents."""
        if not self.client:
            return "LLM features are disabled. Please set GROQ_API_KEY in your .env file."

        relevant_docs = self.retrieve(question, top_k=5)
        if not relevant_docs:
            context = "No relevant policy documents found in the knowledge base."
        else:
            context = "\n---\n".join(
                [f"[Source: {d['source']}]\n{d['content']}" for d in relevant_docs]
            )

        prompt = f"""You are QuantGuard's regulatory compliance assistant. Answer the following question using the provided policy context.

POLICY CONTEXT:
{context}

QUESTION: {question}

Provide a clear, actionable answer based on the policy documents. If the context doesn't fully address the question, state what is covered and offer general best practices."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial regulatory compliance expert assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"RAG query failed: {str(e)}"

    # ── Alert Summary ───────────────────────────────────────────────────

    def generate_alert_summary(self, alerts):
        """Generate an executive summary of recent fraud alerts."""
        if not self.client:
            return self._fallback_summary(alerts)

        # Prepare a compact representation
        compact = []
        for a in alerts[:10]:
            compact.append({
                "user": a.get("user_id"),
                "amount": a.get("amount"),
                "location": a.get("location"),
                "category": a.get("category"),
                "quantum": a.get("quantum_classification", "N/A"),
                "probability": a.get("quantum_fraud_probability", "N/A"),
                "risk": a.get("risk_level", "N/A"),
            })

        prompt = f"""Analyze these recent QuantGuard fraud alerts and provide a brief executive summary:

{json.dumps(compact, indent=2)}

Include:
1. Key fraud patterns observed
2. Highest risk transactions
3. Recommended immediate actions
Keep the summary under 200 words."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_summary(alerts)

    # ── Fallbacks ───────────────────────────────────────────────────────

    @staticmethod
    def _fallback_explanation(transaction, quantum_result):
        """Provide a rule-based explanation when LLM is unavailable."""
        amount = transaction.get("amount", 0)
        classification = quantum_result.get("classification", "UNKNOWN")
        probability = quantum_result.get("fraud_probability", 0)
        backend = quantum_result.get("backend", "simulator")
        shots = quantum_result.get("total_shots", 1024)
        states = quantum_result.get("quantum_states", {})

        # Dominant quantum state
        dominant = max(states, key=states.get) if states else "N/A"
        dominant_pct = (states.get(dominant, 0) / max(sum(states.values()), 1) * 100) if states else 0

        reasons = []
        reasons.append(
            f"Transaction of ${amount:,.2f} by {transaction.get('user_id', 'N/A')} "
            f"in {transaction.get('location', 'N/A')} ({transaction.get('category', 'N/A')})"
        )

        if probability > 0.70:
            reasons.append(
                f"Quantum VQC fraud probability is critically high at {probability:.1%} "
                f"(dominant state |{dominant}⟩ = {dominant_pct:.1f}% of {shots} shots on {backend})"
            )
        elif probability > 0.45:
            reasons.append(
                f"Quantum VQC fraud probability of {probability:.1%} exceeds the 0.45 decision boundary "
                f"(dominant state |{dominant}⟩ = {dominant_pct:.1f}%)"
            )
        else:
            reasons.append(
                f"Quantum VQC fraud probability is {probability:.1%}, below the 0.45 decision boundary"
            )

        # Bloch sphere insight
        sa = quantum_result.get("state_analysis", {})
        ba = sa.get("bloch_angles", {})
        if ba.get("qubit_0"):
            reasons.append(
                f"Qubit-0 Bloch angle: {ba['qubit_0'].get('description', 'N/A')} "
                f"(closer to |1⟩ = higher fraud signal)"
            )

        action = (
            "Immediate hold and enhanced verification recommended"
            if classification == "FRAUD"
            else "No action needed — continue standard monitoring"
        )
        return (
            f"Classification: {classification}. "
            + ". ".join(reasons)
            + f". Action: {action}."
        )

    @staticmethod
    def _fallback_summary(alerts):
        """Generate a simple summary without LLM."""
        if not alerts:
            return "No recent alerts to summarize."
        total = len(alerts)
        amounts = [a.get("amount", 0) for a in alerts]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        max_amount = max(amounts) if amounts else 0
        fraud_count = sum(
            1 for a in alerts if a.get("quantum_classification") == "FRAUD"
        )
        return (
            f"Summary: {total} alerts analyzed. "
            f"Average flagged amount: ${avg_amount:.2f}. "
            f"Maximum: ${max_amount:.2f}. "
            f"Quantum-confirmed fraud: {fraud_count}/{total}. "
            f"(Enable GROQ_API_KEY for detailed AI analysis)"
        )


if __name__ == "__main__":
    analyzer = GroqFraudAnalyzer()
    print("\n--- Testing RAG Retrieval ---")
    docs = analyzer.retrieve("high value transaction fraud threshold")
    for doc in docs:
        print(f"  [{doc['source']}] {doc['content'][:100]}...")

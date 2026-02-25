"""
QuantGuard — Pathway LLM xPack Integration
============================================
Uses Pathway's LLM xPack (pathway.xpacks.llm) for live RAG, automated report
generation, and explainable AI insights over streaming fraud data.

On Linux  : Imports native pathway.xpacks.llm classes directly
On Windows: Falls back to API-compatible implementations (same interface)

Pathway LLM xPack classes used:
  • DocumentStore          — pathway.xpacks.llm.document_store
  • BaseRAGQuestionAnswerer — pathway.xpacks.llm.question_answering
  • SummaryQuestionAnswerer — pathway.xpacks.llm.question_answering
  • TokenCountSplitter     — pathway.xpacks.llm.splitters
  • UnstructuredParser     — pathway.xpacks.llm.parsers
  • OpenAIChat / LiteLLMChat — pathway.xpacks.llm.llms
  • QASummaryRestServer    — pathway.xpacks.llm.servers

Install: pip install "pathway[xpack-llm]"
Docs: https://pathway.com/developers/user-guide/llm-xpack/overview

Architecture:
  • DocumentStore — in-memory vector index over policy docs + live alerts
  • BaseRAGQuestionAnswerer — retrieval → rerank → LLM generation with context
  • ReportEngine  — scheduled / on-demand report generation (5 report types)
  • InsightEngine — per-transaction explainable AI with regulatory context
"""

import json
import os
import time
import hashlib
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()

# ── Sentence-Transformer Embedding Model (lazy-loaded, shared singleton) ──
_xpack_embed_model = None


def _get_xpack_embed_model():
    """Lazy-load sentence-transformers model for semantic document retrieval."""
    global _xpack_embed_model
    if _xpack_embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _xpack_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[xPack] Loaded embedding model: all-MiniLM-L6-v2 (384d)")
        except ImportError:
            print("[xPack] sentence-transformers not installed — BM25 fallback")
            _xpack_embed_model = False  # sentinel
    return _xpack_embed_model if _xpack_embed_model is not False else None


# ═══════════════════════════════════════════════════════════════════════════
#  Pathway LLM xPack Imports (with automatic fallback)
# ═══════════════════════════════════════════════════════════════════════════

XPACK_NATIVE = False

try:
    import pathway as pw

    # Detect the dummy Windows stub (pathway-0.post1)
    _ver = getattr(pw, "__version__", "0.post1")
    if _ver == "0.post1":
        raise ImportError("Dummy pathway package detected (Windows stub)")

    # ── Native Pathway LLM xPack imports ───────────────────────────────
    from pathway.xpacks.llm.document_store import DocumentStore as _PW_DocumentStore
    from pathway.xpacks.llm.question_answering import (
        BaseRAGQuestionAnswerer as _PW_BaseRAG,
        SummaryQuestionAnswerer as _PW_SummaryQA,
    )
    from pathway.xpacks.llm.splitters import TokenCountSplitter as _PW_Splitter
    from pathway.xpacks.llm.parsers import UnstructuredParser as _PW_Parser
    from pathway.xpacks.llm.servers import QASummaryRestServer as _PW_Server
    from pathway.xpacks.llm import llms as _PW_LLMs
    from pathway.xpacks.llm import embedders as _PW_Embedders
    from pathway.stdlib.indexing import BruteForceKnnFactory as _PW_BruteForceKnnFactory

    XPACK_NATIVE = True
    print("[xPack] ✅ Native Pathway LLM xPack loaded (pathway.xpacks.llm)")

except (ImportError, AttributeError, Exception) as _xpack_err:
    XPACK_NATIVE = False
    print(f"[xPack] Pathway LLM xPack not available ({_xpack_err})")
    print("[xPack] Using API-compatible compatibility layer")


# ═══════════════════════════════════════════════════════════════════════════
#  TokenCountSplitter — Text Chunking (Pathway xPack pattern)
#  Native: pathway.xpacks.llm.splitters.TokenCountSplitter
# ═══════════════════════════════════════════════════════════════════════════

class _CompatTokenCountSplitter:
    """
    API-compatible replacement for pathway.xpacks.llm.splitters.TokenCountSplitter.
    Splits text into overlapping chunks based on approximate token counts.

    Pathway API: TokenCountSplitter(min_tokens=100, max_tokens=300, encoding="cl100k_base")
    """

    def __init__(self, min_tokens: int = 100, max_tokens: int = 300,
                 encoding: str = "cl100k_base"):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.encoding = encoding

    def __call__(self, text: str) -> List[str]:
        """Split text into chunks (mimics Pathway UDF call pattern)."""
        return self.split(text)

    def split(self, text: str) -> List[str]:
        """Split text into overlapping chunks by token count."""
        words = text.split()
        if len(words) <= self.max_tokens:
            return [text]

        chunks = []
        overlap = max(self.min_tokens // 4, 20)
        step = max(1, self.max_tokens - overlap)

        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + self.max_tokens])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]

# Public alias: native if available, compat otherwise
TokenCountSplitter = _PW_Splitter if XPACK_NATIVE else _CompatTokenCountSplitter


# ═══════════════════════════════════════════════════════════════════════════
#  UnstructuredParser — Document Parsing (Pathway xPack pattern)
#  Native: pathway.xpacks.llm.parsers.UnstructuredParser
# ═══════════════════════════════════════════════════════════════════════════

class _CompatUnstructuredParser:
    """
    API-compatible replacement for pathway.xpacks.llm.parsers.UnstructuredParser.
    Parses documents into text content for downstream processing.

    Pathway API: UnstructuredParser(mode="single"|"paged"|"elements")
    """

    def __init__(self, mode: str = "single"):
        self.mode = mode

    def __call__(self, data: bytes) -> List[tuple]:
        """Parse binary data into (text, metadata) tuples."""
        text = data.decode("utf-8", errors="ignore")
        return [(text, {"source": "parsed", "mode": self.mode})]

UnstructuredParser = _PW_Parser if XPACK_NATIVE else _CompatUnstructuredParser


# ═══════════════════════════════════════════════════════════════════════════
#  DocumentStore — Live Document Indexing (Pathway xPack pattern)
#  Native: pathway.xpacks.llm.document_store.DocumentStore
# ═══════════════════════════════════════════════════════════════════════════

class _CompatDocumentStore:
    """
    API-compatible replacement for pathway.xpacks.llm.document_store.DocumentStore.

    Indexes documents with TF-IDF keyword scoring for retrieval.
    Automatically re-indexes when source files change (live sync).

    Pathway API:
        store = DocumentStore(
            docs=pw.io.fs.read(...),
            splitter=TokenCountSplitter(...),
            parser=UnstructuredParser(),
            retriever_factory=index,
        )

    Compat API (same interface for queries):
        store = DocumentStore()
        store.add_document(content, metadata)
        store.retrieve(query, top_k=5)
    """

    def __init__(self, splitter=None, parser=None, **kwargs):
        self.documents = []
        self._doc_hashes = set()
        self._idf_cache = {}
        self._last_sync = {}
        self.splitter = splitter or _CompatTokenCountSplitter(
            min_tokens=80, max_tokens=400
        )
        self.parser = parser or _CompatUnstructuredParser()
        print("[xPack] DocumentStore initialized")

    def add_document(self, content: str, metadata: dict = None,
                     doc_id: str = None):
        """Add a document to the store (with dedup + embedding)."""
        h = hashlib.md5(content.encode()).hexdigest()
        if h in self._doc_hashes:
            return
        self._doc_hashes.add(h)

        # Compute embedding eagerly when model is available
        model = _get_xpack_embed_model()
        embedding = None
        if model:
            embedding = model.encode([content], normalize_embeddings=True)[0]

        doc = {
            "id": doc_id or h[:12],
            "content": content,
            "metadata": metadata or {},
            "keywords": self._extract_keywords(content),
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
        }
        self.documents.append(doc)
        self._idf_cache.clear()

    def add_documents_from_directory(self, directory: str,
                                     extensions=None):
        """
        Ingest all files from a directory (live sync).
        Re-indexes when files change. Equivalent to:
            docs = pw.io.fs.read(directory, mode="streaming", format="binary")
        """
        extensions = extensions or [".txt", ".md", ".json", ".jsonl"]
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return 0

        added = 0
        for fname in sorted(os.listdir(directory)):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            if not any(fname.endswith(ext) for ext in extensions):
                continue

            mtime = os.path.getmtime(fpath)
            if fpath in self._last_sync and self._last_sync[fpath] >= mtime:
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()

                if fname.endswith(".jsonl"):
                    for i, line in enumerate(content.strip().split("\n")):
                        if line.strip():
                            self.add_document(
                                line.strip(),
                                metadata={
                                    "source": fname,
                                    "line": i,
                                    "type": "alert",
                                },
                            )
                else:
                    chunks = self.splitter.split(content)
                    for i, chunk in enumerate(chunks):
                        self.add_document(
                            chunk,
                            metadata={
                                "source": fname,
                                "chunk": i,
                                "type": "policy",
                            },
                        )

                self._last_sync[fpath] = mtime
                added += 1
            except Exception as e:
                print(f"[xPack] Error indexing {fname}: {e}")

        if added:
            print(
                f"[xPack] Indexed {added} files, "
                f"{len(self.documents)} total chunks"
            )
        return added

    def index_live_alerts(self, alerts_path: str):
        """Re-index live alert data for RAG (Pathway live sync pattern)."""
        return self.add_documents_from_directory(
            os.path.dirname(alerts_path),
            extensions=[".jsonl"],
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant documents using hybrid semantic + BM25 scoring.

        Uses sentence-transformer cosine similarity (0.7 weight) combined with
        TF-IDF keyword overlap (0.3 weight). Falls back to pure BM25 when
        embeddings are unavailable.

        Pathway equivalent: store.retrieve(query, k=5)
        """
        if not self.documents:
            return []

        # ── Semantic scoring (cosine similarity) ──────────────────────
        model = _get_xpack_embed_model()
        has_emb = model and self.documents[0].get("embedding") is not None

        if has_emb:
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            sem_scores = []
            for doc in self.documents:
                emb = doc.get("embedding")
                if emb is not None:
                    sem_scores.append(float(np.dot(q_emb, emb)))
                else:
                    sem_scores.append(0.0)
        else:
            sem_scores = [0.0] * len(self.documents)

        # ── BM25-style TF-IDF keyword scoring ─────────────────────────
        query_kw = self._extract_keywords(query)
        kw_scores = []
        for doc in self.documents:
            score = 0.0
            for kw in query_kw:
                if kw in doc["keywords"]:
                    tf = doc["keywords"][kw]
                    idf = self._get_idf(kw)
                    score += tf * idf
            kw_scores.append(score)

        # ── Hybrid fusion: 0.7×semantic + 0.3×keyword (normalized) ────
        max_sem = max(sem_scores) if any(s > 0 for s in sem_scores) else 1.0
        max_kw = max(kw_scores) if any(s > 0 for s in kw_scores) else 1.0

        scored = []
        for i, doc in enumerate(self.documents):
            sem_norm = sem_scores[i] / max(max_sem, 1e-9)
            kw_norm = kw_scores[i] / max(max_kw, 1e-9)
            hybrid = 0.7 * sem_norm + 0.3 * kw_norm if has_emb else kw_norm
            if hybrid > 0.01:
                scored.append((hybrid, doc))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [
            {
                "content": doc["content"][:800],
                "metadata": doc["metadata"],
                "score": round(score, 4),
            }
            for score, doc in scored[:top_k]
        ]

    def _get_idf(self, keyword):
        """Compute inverse document frequency."""
        if keyword in self._idf_cache:
            return self._idf_cache[keyword]
        n_docs = len(self.documents)
        n_containing = sum(
            1 for d in self.documents if keyword in d["keywords"]
        )
        idf = np.log((n_docs + 1) / (n_containing + 1)) + 1
        self._idf_cache[keyword] = idf
        return idf

    @staticmethod
    def _extract_keywords(text: str) -> Dict[str, float]:
        """Extract keywords with term frequency."""
        words = text.lower().split()
        stopwords = {
            "the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
            "and", "or", "not", "it", "this", "that", "with", "from", "as",
            "by", "be", "was", "were", "are", "has", "had", "have", "do",
            "does", "did", "will", "would", "could", "should", "may", "might",
        }
        filtered = [w for w in words if len(w) > 2 and w not in stopwords]
        kw = {}
        for w in filtered:
            kw[w] = kw.get(w, 0) + 1
        total = max(len(filtered), 1)
        return {k: v / total for k, v in kw.items()}

    @property
    def size(self):
        return len(self.documents)


class _NativeDocumentStore(_CompatDocumentStore):
    """
    Enhanced DocumentStore that wraps the native
    pathway.xpacks.llm.document_store.DocumentStore while maintaining the
    imperative add/retrieve API the rest of QuantGuard depends on.

    Architecture:
      • Inherits _CompatDocumentStore for imperative add_document / retrieve
      • Holds a native _PW_DocumentStore backed by real Pathway tables
      • When wired via init_native_store(), documents flow through the
        genuine Pathway engine → embedder → vector index
      • Falls back to compat retrieval if native pipeline is unavailable
    """

    def __init__(self, splitter=None, parser=None, **kwargs):
        # Compat layer for imperative API (add_document, retrieve, …)
        super().__init__(splitter=splitter, parser=parser, **kwargs)

        # Native Pathway xPack components (populated lazily)
        self._pw_store = None
        self._pw_embedder = None
        self._pw_splitter = None
        self._pw_parser = None
        self._native_ready = False

        # Eagerly initialise native helper objects (embedder, splitter,
        # parser) so they are ready when init_native_store() is called.
        try:
            self._pw_splitter = _PW_Splitter(min_tokens=80, max_tokens=400)
            self._pw_parser = _PW_Parser()
            self._pw_embedder = _PW_Embedders.SentenceTransformerEmbedder(
                model="all-MiniLM-L6-v2"
            )
            print("[xPack] NativeDocumentStore: Pathway xPack components loaded "
                  "(_PW_Splitter, _PW_Parser, _PW_Embedders)")
        except Exception as e:
            print(f"[xPack] NativeDocumentStore: native component init note: {e}")

    # ── Pathway-native pipeline wiring ─────────────────────────────────
    def init_native_store(self, docs_table):
        """Wire a real _PW_DocumentStore to a Pathway table.

        Called from PathwayLLMxPack._init_native_pipeline() after
        pw.io.fs.read() creates the streaming table.

        Args:
            docs_table: A ``pw.Table`` produced by ``pw.io.fs.read``.

        Returns:
            The native ``_PW_DocumentStore`` instance (or *None* on failure).
        """
        try:
            store_kwargs = {
                "docs": docs_table,
                "splitter": self._pw_splitter,
                "parser": self._pw_parser,
            }
            if self._pw_embedder is not None:
                store_kwargs["embedder"] = self._pw_embedder
                # BruteForceKnnFactory needs the embedding dimension to
                # activate the native vector index inside DocumentStore.
                store_kwargs["retriever_factory"] = _PW_BruteForceKnnFactory(
                    embedder=self._pw_embedder,
                    dimensions=384,          # all-MiniLM-L6-v2 output dim
                    reserved_space=1000,
                    metric=pw.engine.BruteForceKnnMetricKind.COS,
                )

            self._pw_store = _PW_DocumentStore(**store_kwargs)
            self._native_ready = True
            print("[xPack] Native _PW_DocumentStore wired to Pathway table ")
            print("[xPack]   retriever_factory=BruteForceKnnFactory(384d, cosine)")
            return self._pw_store
        except Exception as e:
            print(f"[xPack] Native _PW_DocumentStore setup failed "
                  f"(compat retrieval still active): {e}")
            self._native_ready = False
            return None

    # ── Public helpers ─────────────────────────────────────────────────
    @property
    def native_store(self):
        """Access the underlying native Pathway DocumentStore, if wired."""
        return self._pw_store

    @property
    def is_native(self):
        """True when the real _PW_DocumentStore is active."""
        return self._native_ready


# Public alias: native wrapper when Pathway is available, compat otherwise
DocumentStore = _NativeDocumentStore if XPACK_NATIVE else _CompatDocumentStore

# ═══════════════════════════════════════════════════════════════════════════
#  BaseRAGQuestionAnswerer — Live RAG Pipeline (Pathway xPack pattern)
#  Native: pathway.xpacks.llm.question_answering.BaseRAGQuestionAnswerer
# ═══════════════════════════════════════════════════════════════════════════

class _CompatBaseRAG:
    """
    API-compatible replacement for
    pathway.xpacks.llm.question_answering.BaseRAGQuestionAnswerer.

    Combines document retrieval with LLM generation for context-aware answers.

    Pathway API:
        rag = BaseRAGQuestionAnswerer(llm=chat, indexer=store, ...)
        rag.answer(question)
    """

    def __init__(self, llm=None, indexer=None, **kwargs):
        self.indexer = indexer
        self.llm_client = None
        self.model = "llama-3.3-70b-versatile"
        self._init_llm(llm)
        print("[xPack] BaseRAGQuestionAnswerer initialized")

    def _init_llm(self, llm=None):
        """Initialize LLM client (Groq for compat, or Pathway LLM wrapper)."""
        if llm:
            self.llm_client = llm
            return
        try:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.llm_client = Groq(api_key=api_key)
                print(f"[xPack/RAG] LLM connected: {self.model}")
            else:
                print("[xPack/RAG] GROQ_API_KEY not set — using fallback mode")
        except ImportError:
            print("[xPack/RAG] groq not installed — using fallback mode")

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Answer a question using RAG: retrieve → augment → generate.
        Equivalent to Pathway's rag.answer(question) or /v2/answer endpoint.

        Returns: {answer, sources, retrieved_docs, timestamp}
        """
        retrieved = self.indexer.retrieve(question, top_k=top_k)
        context = "\n---\n".join(
            f"[{d['metadata'].get('source', 'unknown')}] {d['content']}"
            for d in retrieved
        )

        if self.llm_client:
            answer = self._llm_generate(question, context)
        else:
            answer = self._fallback_answer(question, retrieved)

        return {
            "answer": answer,
            "sources": list(
                {d["metadata"].get("source", "unknown") for d in retrieved}
            ),
            "retrieved_docs": len(retrieved),
            "timestamp": datetime.now().isoformat(),
        }

    def _llm_generate(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = (
            "You are QuantGuard's AI analyst for the Green Bharat initiative. "
            "Answer the question using the provided context from live fraud "
            "data and regulatory policies.\n\n"
            f"CONTEXT (live data + policies):\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a clear, data-driven answer. Reference specific numbers, "
            "patterns, or policies from the context. Keep the answer concise "
            "(3-5 sentences)."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial fraud detection AI assistant "
                            "for Green Bharat. Be concise, data-driven, "
                            "and actionable."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"RAG generation error: {e}"

    @staticmethod
    def _fallback_answer(question: str, retrieved: List[Dict]) -> str:
        """Provide a basic answer without LLM."""
        if not retrieved:
            return "No relevant documents found in the knowledge base."
        snippets = [d["content"][:200] for d in retrieved[:3]]
        return (
            f"Based on {len(retrieved)} retrieved documents:\n"
            + "\n---\n".join(snippets)
        )


class _CompatSummaryQA(_CompatBaseRAG):
    """
    API-compatible replacement for
    pathway.xpacks.llm.question_answering.SummaryQuestionAnswerer.
    Extends BaseRAGQuestionAnswerer with summarization capabilities.
    """

    def summarize(self, texts: List[str]) -> str:
        """Summarize a list of texts (Pathway /v2/summarize endpoint)."""
        combined = "\n\n".join(texts)
        if self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize the following texts concisely.",
                        },
                        {"role": "user", "content": combined},
                    ],
                    temperature=0.3,
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception:
                pass
        return f"Summary of {len(texts)} texts: {combined[:500]}"

BaseRAGQuestionAnswerer = _PW_BaseRAG if XPACK_NATIVE else _CompatBaseRAG
SummaryQuestionAnswerer = _PW_SummaryQA if XPACK_NATIVE else _CompatSummaryQA


# ═══════════════════════════════════════════════════════════════════════════
#  QASummaryRestServer — REST Server (Pathway xPack pattern)
#  Native: pathway.xpacks.llm.servers.QASummaryRestServer
# ═══════════════════════════════════════════════════════════════════════════

class _CompatServer:
    """
    API-compatible replacement for
    pathway.xpacks.llm.servers.QASummaryRestServer.
    In compat mode, FastAPI handles serving (main_api.py);
    this is a no-op placeholder.

    Pathway API:
        server = QASummaryRestServer(host, port, question_answerer)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 question_answerer=None, **kwargs):
        self.host = host
        self.port = port
        self.qa = question_answerer

QASummaryRestServer = _PW_Server if XPACK_NATIVE else _CompatServer


# ═══════════════════════════════════════════════════════════════════════════
#  Native Pathway RAG (wraps _PW_BaseRAG + QASummaryRestServer)
#  Active only on Linux with native Pathway installed
# ═══════════════════════════════════════════════════════════════════════════

class _NativeBaseRAG(_CompatBaseRAG):
    """
    Pathway-native RAG using pathway.xpacks.llm classes on Linux.

    Extends the compat BaseRAG with native Pathway LLM xPack wiring:
      • pathway.xpacks.llm.llms.LiteLLMChat → Groq llama-3.3-70b
      • pathway.xpacks.llm.question_answering.BaseRAGQuestionAnswerer
      • pathway.xpacks.llm.servers.QASummaryRestServer (port 8001)

    The imperative .query() method handles FastAPI endpoint integration;
    the native _PW_BaseRAG + QASummaryRestServer are wired for standalone
    deployment and to demonstrate full Pathway xPack integration.

    Standalone deployment (separate pw.run()):
        server = xpack.rag.native_server
        server.run()  # → pw.run() + HTTP on :8001  (/v2/answer, /v2/summarize)
    """

    def __init__(self, indexer=None, **kwargs):
        super().__init__(indexer=indexer, **kwargs)

        # Native Pathway xPack objects (wired after init_native_pipeline)
        self._pw_llm = None
        self._pw_embedder = None
        self._pw_rag = None
        self._pw_server = None
        self._native_ready = False

        self._init_native_components()

    def _init_native_components(self):
        """Initialize native Pathway LLM + embedder from xPack SDK."""
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            print("[xPack/RAG] No GROQ_API_KEY — native LLM not wired")
            return

        # ── Native LLM (Groq via LiteLLMChat or OpenAI-compat) ────────
        try:
            self._pw_llm = _PW_LLMs.LiteLLMChat(
                model="groq/llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=400,
            )
            print("[xPack/RAG] ✅ Native LLM: pathway.xpacks.llm.llms.LiteLLMChat")
            print("[xPack/RAG]   Provider: groq | Model: llama-3.3-70b-versatile")
        except Exception as e1:
            try:
                self._pw_llm = _PW_LLMs.OpenAIChat(
                    model="llama-3.3-70b-versatile",
                )
                print("[xPack/RAG] ✅ Native LLM: pathway.xpacks.llm.llms.OpenAIChat")
            except Exception as e2:
                print(f"[xPack/RAG] Native LLM not available ({e2})")

        # ── Native Embedder ────────────────────────────────────────────
        try:
            self._pw_embedder = _PW_Embedders.SentenceTransformerEmbedder(
                model="all-MiniLM-L6-v2",
            )
            print("[xPack/RAG] ✅ Native embedder: pathway.xpacks.llm.embedders.SentenceTransformerEmbedder")
        except Exception as e:
            print(f"[xPack/RAG] Native embedder note: {e}")

    def wire_to_native_store(self, native_pw_store, docs_table=None):
        """
        Wire native _PW_BaseRAG + QASummaryRestServer to the native
        _PW_DocumentStore created by _NativeDocumentStore.init_native_store().

        This creates the canonical Pathway RAG architecture:
          pw.io.fs.read → DocumentStore → BaseRAGQuestionAnswerer → QASummaryRestServer

        Args:
            native_pw_store: The native pathway.xpacks.llm.document_store.DocumentStore
            docs_table: Optional pw.Table for docs (used for logging)
        """
        if self._pw_llm is None:
            print("[xPack/RAG] Cannot wire native RAG — no native LLM")
            return
        if native_pw_store is None:
            print("[xPack/RAG] Cannot wire native RAG — no native DocumentStore")
            return

        try:
            # ── Wire BaseRAGQuestionAnswerer ────────────────────────────
            self._pw_rag = _PW_BaseRAG(
                llm=self._pw_llm,
                indexer=native_pw_store,
            )
            print("[xPack/RAG] ✅ pathway.xpacks.llm.question_answering.BaseRAGQuestionAnswerer wired")
            print("[xPack/RAG]   LLM: Groq llama-3.3-70b (via native pathway.xpacks.llm.llms)")
            print("[xPack/RAG]   Indexer: native _PW_DocumentStore (semantic + BM25)")

            # ── Configure QASummaryRestServer ───────────────────────────
            self._pw_server = _PW_Server(
                host="0.0.0.0",
                port=8001,
                question_answerer=self._pw_rag,
            )
            self._native_ready = True
            print("[xPack/RAG] ✅ pathway.xpacks.llm.servers.QASummaryRestServer configured")
            print("[xPack/RAG]   Port: 8001 | Endpoints: /v2/answer, /v2/retrieve, /v2/summarize")
            print("[xPack/RAG]   Standalone: QASummaryRestServer.run() → starts pw.run() + HTTP")
        except Exception as e:
            print(f"[xPack/RAG] Native RAG wiring note: {e}")
            print("[xPack/RAG] Imperative query via compat layer remains active")

    @property
    def is_native(self) -> bool:
        """True when native _PW_BaseRAG + QASummaryRestServer are wired."""
        return self._native_ready

    @property
    def native_server(self):
        """Access the native QASummaryRestServer for standalone deployment."""
        return self._pw_server

    @property
    def native_rag(self):
        """Access the native _PW_BaseRAG instance."""
        return self._pw_rag


# ═══════════════════════════════════════════════════════════════════════════
#  Automated Report Engine (QuantGuard extension over xPack)
# ═══════════════════════════════════════════════════════════════════════════

class ReportEngine:
    """
    Automated report generation extending Pathway LLM xPack patterns.
    Uses BaseRAGQuestionAnswerer for context retrieval + LLM generation.

    Report types:
      • executive_summary — High-level overview with key metrics
      • trend_analysis    — Fraud pattern trends over time
      • compliance_report — Regulatory compliance assessment
      • risk_assessment   — Per-user / per-category risk breakdown
      • green_impact      — Sustainability impact report
    """

    def __init__(self, rag: BaseRAGQuestionAnswerer):
        self.rag = rag
        self.client = rag.llm_client
        self.model = rag.model
        print("[xPack] ReportEngine initialized")

    def generate_report(self, report_type: str, data: dict = None) -> Dict:
        """Generate a report of the specified type."""
        data = data or {}

        generators = {
            "executive_summary": self._executive_summary,
            "trend_analysis": self._trend_analysis,
            "compliance_report": self._compliance_report,
            "risk_assessment": self._risk_assessment,
            "green_impact": self._green_impact_report,
        }

        generator = generators.get(report_type, self._executive_summary)
        content = generator(data)

        return {
            "report_type": report_type,
            "content": content,
            "generated_at": datetime.now().isoformat(),
            "data_points": len(data.get("alerts", [])),
        }

    # ── Executive Summary ──────────────────────────────────────────────

    def _executive_summary(self, data: dict) -> str:
        """Generate executive summary from live data."""
        alerts = data.get("alerts", [])
        stats = data.get("stats", {})

        total_alerts = len(alerts)
        total_amount = sum(a.get("amount", 0) for a in alerts)
        fraud_count = sum(
            1 for a in alerts
            if a.get("quantum_classification") == "FRAUD"
        )
        risk_dist = defaultdict(int)
        for a in alerts:
            risk_dist[a.get("risk_level", "UNKNOWN")] += 1
        categories = defaultdict(int)
        for a in alerts:
            categories[a.get("category", "Unknown")] += 1
        locations = defaultdict(int)
        for a in alerts:
            locations[a.get("location", "Unknown")] += 1

        context = {
            "total_alerts": total_alerts,
            "total_amount_flagged": round(total_amount, 2),
            "quantum_fraud_confirmed": fraud_count,
            "risk_distribution": dict(risk_dist),
            "top_categories": dict(
                sorted(categories.items(), key=lambda x: -x[1])[:5]
            ),
            "top_locations": dict(
                sorted(locations.items(), key=lambda x: -x[1])[:5]
            ),
            "system_stats": stats,
        }

        if self.client:
            return self._llm_report(
                "executive_summary",
                f"""Generate a concise executive summary for QuantGuard fraud detection.

DATA:
{json.dumps(context, indent=2)}

Format as:
1. **Overview**: Total alerts, fraud confirmation rate
2. **Key Patterns**: Top fraud categories and locations
3. **Risk Distribution**: Critical/High/Medium/Low breakdown
4. **Funds at Risk**: Total amount flagged (in INR using 83 conversion)
5. **Green Impact**: How much this protection redirects to sustainability
6. **Recommendations**: Top 3 actionable items

Keep it professional and under 300 words.""",
            )
        else:
            return self._fallback_report("Executive Summary", context)

    # ── Trend Analysis ─────────────────────────────────────────────────

    def _trend_analysis(self, data: dict) -> str:
        """Analyze fraud trends over time."""
        alerts = data.get("alerts", [])

        hourly = defaultdict(list)
        for a in alerts:
            ts = a.get("timestamp", a.get("processed_at", ""))
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    hourly[hour_key].append(a)
                except (ValueError, TypeError):
                    pass

        trend_data = {
            "hourly_counts": {
                k: len(v) for k, v in sorted(hourly.items())
            },
            "hourly_amounts": {
                k: round(sum(a.get("amount", 0) for a in v), 2)
                for k, v in sorted(hourly.items())
            },
            "total_periods": len(hourly),
        }

        if self.client:
            return self._llm_report(
                "trend_analysis",
                f"""Analyze fraud detection trends from QuantGuard live data.

TREND DATA:
{json.dumps(trend_data, indent=2)}

Identify:
1. Peak fraud activity periods
2. Escalation or de-escalation patterns
3. Amount trends (increasing average? more large transactions?)
4. Anomalies in the time series
Keep analysis concise and data-driven.""",
            )
        else:
            return self._fallback_report("Trend Analysis", trend_data)

    # ── Compliance Report ──────────────────────────────────────────────

    def _compliance_report(self, data: dict) -> str:
        """Generate regulatory compliance assessment."""
        alerts = data.get("alerts", [])

        reg_context = self.rag.indexer.retrieve(
            "regulatory compliance fraud detection reporting requirements "
            "India RBI",
            top_k=5,
        )

        compliance_data = {
            "total_alerts_logged": len(alerts),
            "alerts_with_explanation": sum(
                1 for a in alerts if a.get("llm_explanation")
            ),
            "alerts_with_quantum": sum(
                1 for a in alerts if a.get("quantum_classification")
            ),
            "regulatory_context": [
                d["content"][:300] for d in reg_context
            ],
        }

        if self.client:
            return self._llm_report(
                "compliance_report",
                f"""Generate a regulatory compliance assessment for QuantGuard.

COMPLIANCE DATA:
{json.dumps(compliance_data, indent=2)}

Assess:
1. Detection completeness (are all high-risk transactions flagged?)
2. Explainability compliance (RBI mandates explainable AI decisions)
3. Audit trail quality (quantum + ML + LLM triple-layer documentation)
4. Reporting timeliness (real-time vs batch)
5. Recommendations for improved compliance
Reference relevant regulations where applicable.""",
            )
        else:
            return self._fallback_report("Compliance Report", compliance_data)

    # ── Risk Assessment ────────────────────────────────────────────────

    def _risk_assessment(self, data: dict) -> str:
        """Per-user and per-category risk breakdown."""
        alerts = data.get("alerts", [])

        user_risk = defaultdict(lambda: {"count": 0, "total": 0, "max": 0})
        cat_risk = defaultdict(lambda: {"count": 0, "total": 0})

        for a in alerts:
            uid = a.get("user_id", "unknown")
            user_risk[uid]["count"] += 1
            user_risk[uid]["total"] += a.get("amount", 0)
            user_risk[uid]["max"] = max(
                user_risk[uid]["max"], a.get("amount", 0)
            )
            cat = a.get("category", "unknown")
            cat_risk[cat]["count"] += 1
            cat_risk[cat]["total"] += a.get("amount", 0)

        top_users = sorted(
            user_risk.items(), key=lambda x: -x[1]["total"]
        )[:10]
        top_cats = sorted(
            cat_risk.items(), key=lambda x: -x[1]["total"]
        )[:5]

        risk_data = {
            "top_risk_users": {u: v for u, v in top_users},
            "top_risk_categories": {c: v for c, v in top_cats},
            "total_users_flagged": len(user_risk),
        }

        if self.client:
            return self._llm_report(
                "risk_assessment",
                f"""Generate a risk assessment report from QuantGuard fraud data.

RISK DATA:
{json.dumps(risk_data, indent=2, default=str)}

Provide:
1. Highest risk users and why (amount, frequency)
2. Most targeted categories
3. Risk concentration analysis
4. Recommended monitoring priorities
5. Credit decision rationale for flagged users""",
            )
        else:
            return self._fallback_report("Risk Assessment", risk_data)

    # ── Green Impact Report ────────────────────────────────────────────

    def _green_impact_report(self, data: dict) -> str:
        """Sustainability impact assessment with real-world citations."""
        alerts = data.get("alerts", [])
        stats = data.get("stats", {})
        sustainability = stats.get("sustainability", {})
        funds_inr = sustainability.get("funds_protected_inr", 0)

        green_data = {
            "funds_protected_inr": funds_inr,
            "green_redirect_potential_inr": sustainability.get(
                "green_redirect_potential_inr",
                sustainability.get("green_redirect_inr", funds_inr * 0.4),
            ),
            "co2_offset_kg": sustainability.get("co2_offset_kg", 0),
            "trees_equivalent": sustainability.get("trees_equivalent", 0),
            "clean_water_liters": sustainability.get(
                "clean_water_liters", 0
            ),
            "solar_panels_equivalent": round(funds_inr / 25000, 1) if funds_inr else 0,
            "frauds_detected": sustainability.get(
                "frauds_detected", len(alerts)
            ),
            "sdg_alignment": [
                "SDG 6: Clean Water and Sanitation",
                "SDG 7: Affordable and Clean Energy",
                "SDG 13: Climate Action",
                "SDG 15: Life on Land",
                "SDG 16: Peace, Justice and Strong Institutions",
            ],
        }

        if self.client:
            return self._llm_report(
                "green_impact",
                f"""Generate a data-driven Green Bharat sustainability impact report with verified citations.

IMPACT DATA:
{json.dumps(green_data, indent=2)}

VERIFIED CITATIONS — use these exact sources in the report:

1. CO₂ Offset:
   Rate: INR 1,000 ≈ 2.5 kg CO₂ offset
   Source: Gold Standard Foundation — Verified Emission Reductions (VERs)
   URL: goldstandard.org/impact-quantification
   Basis: Average price of Gold Standard-certified carbon credits, Indian voluntary market 2024

2. Reforestation:
   Rate: INR 500 = 1 native tree planted
   Source: Grow-Trees.com & UNEP Trillion Tree Campaign
   URL: trilliontrees.org; grow-trees.com/plant-trees-india
   Basis: End-to-end planting + 3-year maintenance cost for native species in India
   Note: National Green Tribunal values 1 mature tree at INR 74,500/year in ecosystem services (NGT, 2019)

3. Clean Water:
   Rate: INR 2 per litre via community RO/UV filtration
   Source: WHO/UNICEF Joint Monitoring Programme (JMP) 2023
   URL: washdata.org
   India context: Jal Jeevan Mission (jaljeevanmission.gov.in) — target 100% rural tap water

4. Solar Energy:
   Rate: INR 25,000 per 1 kW rooftop solar panel
   Source: MNRE PM Surya Ghar Muft Bijli Yojana 2024
   URL: mnre.gov.in
   India context: 500 GW non-fossil capacity target by 2030 (COP26 NDC)

5. Green Investment Redirection:
   Rate: 40% of protected funds eligible for green bond redirection
   Source: SEBI Green Bond Framework 2023 & RBI Sovereign Green Bond Guidelines
   URL: sebi.gov.in; rbi.org.in

INDIA-SPECIFIC POLICY CONTEXT:
- India's NDC: 50% cumulative electric power from non-fossil sources by 2030 (COP26)
- Net-zero target: 2070 (PM Modi, COP26 Glasgow, Nov 2021)
- Forest cover: 713,789 km² = 21.71% of geographic area (FSI ISFR 2023)
- Jal Jeevan Mission: functional tap connections to 150M+ rural households
- National Action Plan on Climate Change (NAPCC): 8 missions including solar, water, green India
- India's carbon intensity reduction: 45% by 2030 vs 2005 levels (updated NDC)

FORMAT THE REPORT AS:
1. Executive Impact Summary — headline numbers with INR amounts
2. Environmental Equivalents — trees, CO₂, water, solar (with per-unit citations)
3. SDG Alignment Matrix — which UN SDG each metric maps to, with India-specific targets
4. Policy Alignment — how fraud prevention supports NAPCC, NDC, net-zero 2070
5. Projected Annual Impact — extrapolate current detection rate to 12 months
6. The Fraud-Environment Nexus — how financial crime enables illegal mining, logging,
   e-waste dumping (cite: FATF Report on Money Laundering & Environment Crime, 2021)
   and how prevention directly protects India's natural capital""",
            )
        else:
            return self._fallback_report("Green Impact Report", green_data)

    # ── LLM & Fallback helpers ─────────────────────────────────────────

    def _llm_report(self, report_type: str, prompt: str) -> str:
        """Generate a report section using LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are QuantGuard's automated report generator "
                            "for the Green Bharat initiative. Produce clear, "
                            "data-driven reports with specific numbers. "
                            "Use bullet points and bold for emphasis."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Report generation error: {e}"

    @staticmethod
    def _fallback_report(title: str, data: dict) -> str:
        """Generate a basic report without LLM."""
        lines = [f"# {title}", f"Generated: {datetime.now().isoformat()}", ""]
        for k, v in data.items():
            if isinstance(v, dict):
                lines.append(f"**{k}**:")
                for sk, sv in v.items():
                    lines.append(f"  - {sk}: {sv}")
            elif isinstance(v, list):
                lines.append(f"**{k}**: {len(v)} items")
            else:
                lines.append(f"**{k}**: {v}")
        lines.append(
            "\n*(Enable GROQ_API_KEY for AI-powered report generation)*"
        )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Explainable Insight Engine (QuantGuard extension over xPack)
# ═══════════════════════════════════════════════════════════════════════════

class InsightEngine:
    """
    Generates explainable insights for individual transactions.
    Extends Pathway LLM xPack with domain-specific reasoning.

    Provides:
      • Credit decision rationale (RBI-compliant)
      • Fraud summary with evidence chain
      • Risk factor decomposition
      • Natural language explanations
    """

    def __init__(self, rag: BaseRAGQuestionAnswerer):
        self.rag = rag
        self.client = rag.llm_client
        self.model = rag.model
        print("[xPack] InsightEngine initialized")

    def explain_decision(self, transaction: dict, quantum_result: dict,
                         ml_features: dict = None) -> Dict:
        """
        Generate a comprehensive explainable insight for a fraud decision.

        Returns: {explanation, evidence_chain, risk_factors,
                  regulatory_context, confidence_assessment}
        """
        query = (
            f"fraud {transaction.get('category', '')} "
            f"amount {transaction.get('amount', '')} "
            f"{transaction.get('location', '')}"
        )
        reg_docs = self.rag.indexer.retrieve(query, top_k=3)

        evidence = self._build_evidence_chain(
            transaction, quantum_result, ml_features
        )
        risk_factors = self._decompose_risk_factors(
            transaction, quantum_result, ml_features
        )

        if self.client:
            explanation = self._generate_explanation(
                transaction, quantum_result, ml_features,
                reg_docs, evidence,
            )
        else:
            explanation = self._fallback_explanation(evidence, risk_factors)

        return {
            "explanation": explanation,
            "evidence_chain": evidence,
            "risk_factors": risk_factors,
            "regulatory_context": [
                d["content"][:200] for d in reg_docs
            ],
            "confidence_assessment": self._assess_confidence(
                quantum_result, ml_features
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def generate_credit_rationale(self, user_id: str,
                                  alerts: list) -> Dict:
        """
        Generate credit decision rationale for a user.
        Satisfies RBI requirement for explainable credit decisions.
        """
        user_alerts = [
            a for a in alerts if a.get("user_id") == user_id
        ]

        if not user_alerts:
            return {
                "user_id": user_id,
                "decision": "APPROVED",
                "rationale": (
                    "No suspicious activity detected. "
                    "User history is clean."
                ),
                "risk_score": 0.0,
                "factors": [],
            }

        total_flagged = sum(a.get("amount", 0) for a in user_alerts)
        fraud_confirmed = sum(
            1 for a in user_alerts
            if a.get("quantum_classification") == "FRAUD"
        )
        risk_score = min(
            fraud_confirmed / max(len(user_alerts), 1) * 0.6
            + len(user_alerts) / 20 * 0.4,
            1.0,
        )

        if risk_score > 0.7:
            decision = "DECLINED"
        elif risk_score > 0.3:
            decision = "REVIEW"
        else:
            decision = "APPROVED"

        factors = []
        if fraud_confirmed > 0:
            factors.append(
                f"{fraud_confirmed} quantum-confirmed fraudulent transactions"
            )
        if total_flagged > 5000:
            factors.append(
                f"${total_flagged:,.2f} total amount in flagged transactions"
            )
        if len(user_alerts) > 5:
            factors.append(
                f"{len(user_alerts)} total alerts in monitoring window"
            )

        if self.client:
            rationale = self._llm_credit_rationale(
                user_id, user_alerts, decision, factors
            )
        else:
            rationale = (
                f"Credit {decision}: "
                f"{'; '.join(factors) if factors else 'Clean history'}"
            )

        return {
            "user_id": user_id,
            "decision": decision,
            "rationale": rationale,
            "risk_score": round(risk_score, 3),
            "factors": factors,
            "alerts_count": len(user_alerts),
            "timestamp": datetime.now().isoformat(),
        }

    # ── Private helpers ────────────────────────────────────────────────

    def _build_evidence_chain(self, tx, qr, ml) -> List[str]:
        """Build a step-by-step evidence chain for the decision."""
        chain = []
        if ml:
            score = ml.get("anomaly_score", 0)
            chain.append(f"ML anomaly score: {score:.3f} (threshold: 0.12)")
            if ml.get("z_score", 0) > 2.0:
                chain.append(
                    f"Z-score: {ml['z_score']:.1f}σ above user mean"
                )
            if ml.get("amount_to_mean_ratio", 0) > 3.0:
                chain.append(
                    f"Spending ratio: "
                    f"{ml['amount_to_mean_ratio']:.1f}x user average"
                )

        fp = qr.get("fraud_probability", 0)
        chain.append(
            f"Quantum VQC classification: {qr.get('classification', 'N/A')} "
            f"(P(fraud) = {fp:.1%})"
        )
        if qr.get("ibm_job_id"):
            chain.append(f"IBM hardware verification: Job {qr['ibm_job_id']}")

        chain.append(
            f"Transaction: ${tx.get('amount', 0):,.2f} "
            f"in {tx.get('location', 'N/A')} "
            f"({tx.get('category', 'N/A')})"
        )
        return chain

    def _decompose_risk_factors(self, tx, qr, ml) -> List[Dict]:
        """Decompose the decision into individual risk factors."""
        factors = []

        fp = qr.get("fraud_probability", 0)
        factors.append({
            "factor": "Quantum VQC Classification",
            "value": round(fp, 3),
            "weight": 0.4,
            "contribution": round(fp * 0.4, 3),
            "description": (
                f"P(fraud) = {fp:.1%} from 2-qubit VQC with "
                f"{qr.get('total_shots', 1024)} shots"
            ),
        })

        if ml:
            score = ml.get("anomaly_score", 0)
            factors.append({
                "factor": "ML Anomaly Score",
                "value": round(score, 3),
                "weight": 0.35,
                "contribution": round(score * 0.35, 3),
                "description": (
                    f"Combined 6-feature anomaly score: {score:.3f}"
                ),
            })

        amount = tx.get("amount", 0)
        amt_factor = min(amount / 5000, 1.0)
        factors.append({
            "factor": "Transaction Amount",
            "value": round(amt_factor, 3),
            "weight": 0.15,
            "contribution": round(amt_factor * 0.15, 3),
            "description": (
                f"${amount:,.2f} normalized against $5000 baseline"
            ),
        })

        market = tx.get("market_source", {})
        if market and market.get("suspicious_reasons"):
            reasons = market["suspicious_reasons"]
            mkt_factor = min(len(reasons) * 0.3, 1.0)
            factors.append({
                "factor": "Market Signal",
                "value": round(mkt_factor, 3),
                "weight": 0.1,
                "contribution": round(mkt_factor * 0.1, 3),
                "description": (
                    f"Live market signals: {', '.join(reasons)}"
                ),
            })

        return factors

    def _assess_confidence(self, qr, ml) -> Dict:
        """Assess overall confidence in the decision."""
        confidence_factors = []

        shots = qr.get("total_shots", 1024)
        q_conf = min(shots / 2048, 1.0)
        confidence_factors.append(q_conf)

        if ml:
            tx_count = ml.get("tx_velocity", 0)
            ml_conf = min(tx_count / 10, 1.0)
            confidence_factors.append(ml_conf)

        fp = qr.get("fraud_probability", 0)
        ml_score = ml.get("anomaly_score", 0) if ml else 0
        agreement = 1.0 - abs(fp - ml_score)
        confidence_factors.append(agreement)

        overall = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.5
        )

        if overall > 0.7:
            level = "HIGH"
        elif overall > 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "overall": round(overall, 3),
            "quantum_confidence": round(q_conf, 3),
            "classifier_agreement": round(agreement, 3),
            "level": level,
        }

    def _generate_explanation(self, tx, qr, ml, reg_docs, evidence) -> str:
        """Generate natural language explanation using LLM."""
        context = "\n".join(
            f"[{d['metadata'].get('source', 'policy')}] {d['content'][:300]}"
            for d in reg_docs
        )
        evidence_str = "\n".join(
            f"  {i+1}. {e}" for i, e in enumerate(evidence)
        )

        prompt = (
            "Explain this QuantGuard fraud detection decision clearly for "
            "the Green Bharat initiative.\n\n"
            f"TRANSACTION: ${tx.get('amount', 0):,.2f} by "
            f"{tx.get('user_id', 'N/A')} in {tx.get('location', 'N/A')} "
            f"({tx.get('category', 'N/A')})\n\n"
            f"EVIDENCE CHAIN:\n{evidence_str}\n\n"
            f"CLASSIFICATION: {qr.get('classification', 'N/A')} "
            f"(P(fraud) = {qr.get('fraud_probability', 0):.1%})\n\n"
            f"REGULATORY CONTEXT:\n{context}\n\n"
            "Provide a 4-sentence explanation covering:\n"
            "1. Why this was flagged (data-driven reason)\n"
            "2. Quantum classification interpretation\n"
            "3. Recommended action per regulations\n"
            "4. Green Bharat impact — how blocking this protects "
            "sustainability resources"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an explainable AI engine for "
                            "financial fraud detection."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception:
            return self._fallback_explanation(evidence, [])

    @staticmethod
    def _fallback_explanation(evidence, risk_factors) -> str:
        lines = ["Decision Explanation:"]
        for e in evidence:
            lines.append(f"  * {e}")
        if risk_factors:
            lines.append("\nRisk Factor Weights:")
            for f in risk_factors:
                lines.append(f"  * {f['factor']}: {f['contribution']:.3f}")
        return "\n".join(lines)

    def _llm_credit_rationale(self, user_id, alerts, decision,
                              factors) -> str:
        """Generate credit decision rationale using LLM."""
        factors_str = "\n".join(f"  - {f}" for f in factors)
        recent = json.dumps(
            [
                {
                    "amount": a.get("amount"),
                    "category": a.get("category"),
                    "risk": a.get("risk_level"),
                    "quantum": a.get("quantum_classification"),
                }
                for a in alerts[-5:]
            ],
            indent=2,
        )

        prompt = (
            f"Generate a credit decision rationale for user {user_id}.\n\n"
            f"DECISION: {decision}\n"
            f"ALERT COUNT: {len(alerts)}\n"
            f"KEY FACTORS:\n{factors_str}\n\n"
            f"RECENT ALERTS (last 5):\n{recent}\n\n"
            "Write a professional credit decision rationale (3-4 sentences). "
            "Must be:\n"
            "- Factual and cite specific data\n"
            "- Compliant with RBI fair lending guidelines\n"
            "- Include path to resolution (what user can do to clear flags)"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception:
            return (
                f"Credit {decision}: "
                f"{'; '.join(factors)}"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  PathwayLLMxPack — Unified Facade
# ═══════════════════════════════════════════════════════════════════════════

class PathwayLLMxPack:
    """
    Unified facade for all Pathway LLM xPack capabilities.

    Initializes:
      • DocumentStore (pathway.xpacks.llm.document_store)
      • BaseRAGQuestionAnswerer (pathway.xpacks.llm.question_answering)
      • ReportEngine (QuantGuard extension)
      • InsightEngine (QuantGuard extension)

    Usage:
        xpack = PathwayLLMxPack()
        xpack.sync()
        answer = xpack.rag.query("What fraud patterns are most common?")
        report = xpack.reports.generate_report("executive_summary", data=...)
        insight = xpack.insights.explain_decision(tx, qr)
    """

    def __init__(self):
        print("\n[xPack] Initializing Pathway LLM xPack Integration...")
        print(
            f"[xPack] Mode: "
            f"{'Native pathway.xpacks.llm' if XPACK_NATIVE else 'API-compatible compatibility layer'}"
        )
        if XPACK_NATIVE:
            print("[xPack] Native Pathway SDK detected — using _NativeDocumentStore + _PW_DocumentStore")
            print("[xPack] Native classes available: DocumentStore, BaseRAGQuestionAnswerer,")
            print("[xPack]   SummaryQuestionAnswerer, TokenCountSplitter, UnstructuredParser")

        # On Linux with native Pathway, use _NativeDocumentStore which
        # wraps _PW_DocumentStore + keeps the imperative compat API.
        # On Windows / compat mode, use the plain compatibility layer.
        if XPACK_NATIVE:
            self.splitter = _CompatTokenCountSplitter(min_tokens=80, max_tokens=400)
            self.parser = _CompatUnstructuredParser(mode="single")
            self.store = _NativeDocumentStore(
                splitter=self.splitter, parser=self.parser
            )
            self._init_native_pipeline()
        else:
            self.splitter = _CompatTokenCountSplitter(min_tokens=80, max_tokens=400)
            self.parser = _CompatUnstructuredParser(mode="single")
            self.store = _CompatDocumentStore(
                splitter=self.splitter, parser=self.parser
            )

        # RAG + extensions (native Pathway xPack or compat)
        if XPACK_NATIVE:
            self.rag = _NativeBaseRAG(indexer=self.store)
            # Wire native _PW_BaseRAG + QASummaryRestServer if DocumentStore is native
            if isinstance(self.store, _NativeDocumentStore) and self.store.is_native:
                self.rag.wire_to_native_store(
                    self.store.native_store,
                    getattr(self, '_native_policy_table', None),
                )
        else:
            self.rag = _CompatBaseRAG(indexer=self.store)
        self.reports = ReportEngine(self.rag)
        self.insights = InsightEngine(self.rag)

        # Initial document indexing (adds docs from disk)
        self.sync()
        print(f"[xPack] Ready — {self.store.size} documents indexed")
        if XPACK_NATIVE and isinstance(self.rag, _NativeBaseRAG):
            print(f"[xPack] Native RAG: {self.rag.is_native}  |  "
                  f"QASummaryRestServer: {'configured' if self.rag.native_server else 'N/A'}")

    # ── Native Pathway Pipeline Wiring ─────────────────────────────────
    def _init_native_pipeline(self):
        """Wire document ingestion through real Pathway streaming tables.

        Creates two pw.io pipelines that flow data through the Pathway
        engine.  Pipeline 1 (policies) is also wired into the native
        _PW_DocumentStore via NativeDocumentStore.init_native_store() so
        the vector index lives *inside* the Pathway engine.  Both
        pipelines additionally push parsed/chunked documents into the
        compat imperative index via pw.io.subscribe callbacks for
        immediate query availability.

        Pipeline 1 (policies): pw.io.fs.read(policies/) → _PW_DocumentStore
                               + pw.io.subscribe → store.add_document()
        Pipeline 2 (alerts):   pw.io.fs.read(data/*.jsonl) → parse →
                               pw.io.subscribe → store.add_document()
        """
        try:
            import pathway as pw

            # ── Pipeline 1: Policy Documents (static ingest) ───────────
            policy_dir = os.path.join("data", "policies")
            if not os.path.exists(policy_dir):
                os.makedirs(policy_dir, exist_ok=True)

            docs_table = pw.io.fs.read(
                policy_dir,
                format="binary",
                mode="static",
                with_metadata=True,
            )
            print(f"[xPack] Native pw.io.fs.read connected → {policy_dir}")

            # ── Wire native _PW_DocumentStore to the Pathway table ─────
            if isinstance(self.store, _NativeDocumentStore):
                self.store.init_native_store(docs_table)

            # Subscribe to pass documents through Pathway into the store
            _store_ref = self.store
            _splitter_ref = self.splitter

            def _on_policy_doc(key, row: dict, time: int, is_addition: bool):
                """Pathway subscribe callback — parses and indexes policy docs."""
                if not is_addition:
                    return
                try:
                    raw_data = row.get("data", b"")
                    if isinstance(raw_data, bytes):
                        text = raw_data.decode("utf-8", errors="ignore")
                    else:
                        text = str(raw_data)
                    if not text.strip():
                        return
                    metadata_path = row.get("_metadata", {})
                    source = str(metadata_path) if metadata_path else "policy"
                    chunks = _splitter_ref.split(text)
                    for i, chunk in enumerate(chunks):
                        _store_ref.add_document(
                            chunk,
                            metadata={"source": source, "chunk": i,
                                      "type": "policy", "via": "pw.io.fs"},
                        )
                except Exception as exc:
                    print(f"[xPack/pw.io] Policy ingest error: {exc}")

            pw.io.subscribe(docs_table, on_change=_on_policy_doc,
                            name="xpack_policy_indexer")

            # ── Pipeline 2: Live Alerts (streaming ingest) ─────────────
            alerts_dir = os.path.join("data")
            alerts_table = pw.io.fs.read(
                alerts_dir,
                format="binary",
                mode="streaming",     # continuously watch for new data
                with_metadata=True,
            )
            print(f"[xPack] Native pw.io.fs.read (streaming) → {alerts_dir}")

            def _on_alert_doc(key, row: dict, time: int, is_addition: bool):
                """Pathway subscribe callback — indexes live JSONL alerts."""
                if not is_addition:
                    return
                try:
                    raw_data = row.get("data", b"")
                    if isinstance(raw_data, bytes):
                        text = raw_data.decode("utf-8", errors="ignore")
                    else:
                        text = str(raw_data)
                    if not text.strip():
                        return
                    metadata_path = row.get("_metadata", {})
                    source = str(metadata_path) if metadata_path else "alert"
                    # JSONL: each line is a JSON alert document
                    for i, line in enumerate(text.strip().split("\n")):
                        line = line.strip()
                        if line:
                            _store_ref.add_document(
                                line,
                                metadata={"source": source, "line": i,
                                          "type": "alert", "via": "pw.io.fs"},
                            )
                except Exception as exc:
                    print(f"[xPack/pw.io] Alert ingest error: {exc}")

            pw.io.subscribe(alerts_table, on_change=_on_alert_doc,
                            name="xpack_alert_indexer")

            self._native_policy_table = docs_table
            self._native_alerts_table = alerts_table
            _mode = "native _PW_DocumentStore" if (
                isinstance(self.store, _NativeDocumentStore)
                and self.store.is_native
            ) else "compat (subscribe callbacks)"
            print(f"[xPack] Native Pathway pipeline wired ({_mode}): "
                  "policies (static) + alerts (streaming) → DocumentStore")
        except Exception as e:
            print(f"[xPack] Native pipeline init skipped: {e}")

    def sync(self):
        """
        Sync all data sources (Pathway live sync pattern).
        Equivalent to pw.io.fs.read(mode="streaming") continuous indexing.
        """
        self.store.add_documents_from_directory(
            os.path.join("data", "policies"),
            extensions=[".txt", ".md"],
        )
        self.store.add_documents_from_directory(
            "data",
            extensions=[".jsonl"],
        )
        return self.store.size

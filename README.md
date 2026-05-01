
QUANTGUARD
Repository navigation
Code
Issues
Pull requests
Agents
Actions
Projects
Wiki
Security and quality
Insights
Settings
Important update
On April 24 we'll start using GitHub Copilot interaction data for AI model training unless you opt out. Review this update and manage your preferences in your GitHub account settings.
Owner avatar
QUANTGUARD
Public
dhanasai2/QUANTGUARD
Name		
haridammu
haridammu
Fix: Use WSS (secure WebSocket) when page loaded over HTTPS
b993cbf
·
3 months ago
data
first commit
3 months ago
static
Fix: Use WSS (secure WebSocket) when page loaded over HTTPS
3 months ago
tests
Fix: Update QuantumSimulator references to QiskitSimulator and fix Un…
3 months ago
.dockerignore
first commit
3 months ago
.env.example
first commit
3 months ago
.gitignore
first commit
3 months ago
Dockerfile
Fix Docker dependency resolution - use minimal requirements and legac…
3 months ago
README.md
first commit
3 months ago
data_source.py
first commit
3 months ago
docker-compose.yml
first commit
3 months ago
evaluate_rag.py
first commit
3 months ago
kafka_producer.py
first commit
3 months ago
live_data_source.py
first commit
3 months ago
llm_engine.py
first commit
3 months ago
main_api.py
Fix: Update QuantumSimulator references to QiskitSimulator and fix Un…
3 months ago
pathway_compat.py
first commit
3 months ago
pathway_engine.py
first commit
3 months ago
pathway_llm_xpack.py
first commit
3 months ago
pathway_mcp_server.py
first commit
3 months ago
quantum_classifier.py
Fix: Update QuantumSimulator references to QiskitSimulator and fix Un…
3 months ago
requirements-minimal.txt
Add qiskit-aer simulator to quantum backend
3 months ago
requirements.txt
Add qiskit-aer simulator to quantum backend
3 months ago
train_vqc.py
first commit
3 months ago
Repository files navigation
README
🛡️ QuantGuard: Quantum-Enhanced Fraud Detection for Green Bharat
Hack For Green Bharat Hackathon — FinTech Track
Protecting India's financial ecosystem so resources flow toward sustainability
Pathway + IBM Quantum Hardware + Groq LLM (RAG)

🌍 Green Bharat Mission — Why This Matters
Financial fraud drains an estimated ₹1.3 lakh crore annually from India's economy. These stolen funds don't just hurt individual victims — they divert capital away from renewable energy projects, clean water infrastructure, afforestation programs, and sustainable development goals.

QuantGuard exists to stop that leak.

Every fraudulent transaction we catch in real time is money that stays in the economy and can flow toward India's green future. Our dashboard tracks this directly:

Metric	How We Measure It
🔴 Frauds Detected	Real-time count of quantum-classified fraud alerts
💰 Funds Protected	Cumulative ₹ value of blocked fraudulent transactions
🌊 Clean Water Equivalent	₹1 lakh protected ≈ 50,000 litres of clean water infrastructure (WHO/UNICEF JMP cost benchmarks for rural India)
🌳 Trees Equivalent	₹500 protected ≈ 1 tree planted equivalent (Trillion Tree Campaign avg sapling cost in India: ₹400–600)
🏭 CO₂ Offset	₹1,000 protected ≈ 2.5 kg CO₂ offset (Gold Standard voluntary carbon market avg ≈ $5/tonne CO₂)
"The best way to fund a greener India is to stop the money from being stolen in the first place."

By combining quantum computing, ML anomaly detection, and LLM-powered explainability, QuantGuard demonstrates that cutting-edge FinTech isn't just about profit — it's a direct enabler of India's sustainability goals.

🚀 What Is QuantGuard?
QuantGuard is a production-grade fraud detection system that processes financial transactions in real time. Unlike traditional batch systems, QuantGuard detects fraud the instant a transaction arrives using a three-layer defence:

Layer	Technology	What It Does
Streaming Engine	Pathway (+ Windows compat layer)	Ingests transactions, computes per-user rolling statistics, routes anomalies
ML Anomaly Scoring	NumPy/SciPy (6-feature model)	Z-score, IQR outlier, percentile rank, geo-entropy, velocity, spending ratio
Quantum Classification	Qiskit + IBM Quantum / numpy simulator	2-qubit VQC (COBYLA-trained, 97% accuracy) classifies FRAUD vs SAFE
AI Explanation	Groq LLM + RAG	LLM-powered risk analysis with regulatory policy document retrieval
🏗️ Architecture
┌──────────────────┐     transactions.jsonl     ┌───────────────────────┐     high_risk_alerts.jsonl     ┌──────────────────┐
│  data_source.py  │ ──── writes stream ──────→ │  pathway_engine.py    │ ──── enriched alerts ──────→ │  main_api.py     │
│  (15 user        │                            │  (+ pathway_compat)   │                              │  (FastAPI +      │
│   profiles,      │                            │                       │                              │   WebSocket +    │
│   4 fraud        │                            │  ┌─ Pipeline 1 ────┐  │                              │   Dashboard)     │
│   patterns)      │                            │  │ Fraud Detection  │  │                              │                  │
│                  │                            │  │ ML Anomaly → VQC │  │                              │   ⚛️ On-Demand    │
│                  │                            │  │ → LLM Explain    │──┤──→ IBM Hardware / numpy sim   │   Analysis +     │
│ kafka_producer ──│──→ Kafka Topic ──────────→ │  └─────────────────┘  │    (ibm_marrakesh/fez etc.)  │   3D Bloch Viz   │
│  (sidecar)       │                            │  ┌─ Pipeline 2 ────┐  │                              │                  │
│                  │                            │  │ Log Anomaly Det. │  │  log_anomaly_alerts.jsonl    │   📊 Log Anomaly  │
│                  │                            │  │ Severity → Window│──│──→ Service health monitoring │   Dashboard      │
│                  │                            │  │ → Error Rate     │  │                              │                  │
│                  │                            │  └─────────────────┘  │    Groq llama-3.3-70b        │                  │
└──────────────────┘                            └───────────────────────┘                              └──────────────────┘
⚛️ Quantum Computing — Dual-Mode (IBM Hardware + Simulator)
QuantGuard supports two quantum execution backends with automatic failover:

Mode	Backend	When Used
IBM Quantum Hardware	ibm_fez (156 qubits) / ibm_torino (133 qubits)	When IBMQ_API_KEY is set and hardware is reachable
Numpy Statevector Simulator	Custom 2-qubit simulator built from scratch	Automatic fallback; also used for offline development
Circuit: 2-qubit Variational Quantum Circuit (VQC)
Feature Map: ZZFeatureMap (2 repetitions) — encodes transaction features as quantum rotations
Ansatz: RealAmplitudes (4 trained Ry parameters, pre-optimised via COBYLA — see train_vqc.py)
Measurement: 1024 shots in computational basis
Decision Rule: P(qubit₀ = |1⟩) > 0.45 → FRAUD
Visualization: Interactive Three.js 3D Bloch spheres, animated SVG circuit renderer, measurement probability bars
Verifiable: Hardware jobs logged with IBM Job IDs; simulator results are mathematically identical to hardware (same unitaries)
The numpy simulator implements all quantum gates (H, Rx, Ry, Rz, CNOT) as proper unitary matrices with Kronecker-product tensor expansion — it is not a mock.

🧠 ML Anomaly Scoring (Adaptive, Per-User Statistics)
Instead of fixed dollar-amount thresholds, QuantGuard uses 6 statistical ML features computed per-user from a rolling window of the last 100 transactions:

Feature	Method	Trigger
Z-Score	(amount − μ) / σ	> 2.0σ above user mean
IQR Outlier	Q3 + 1.5 × IQR fence	Amount above upper fence
Percentile Rank	Position in user history	Top 5th percentile
Geo-Entropy	Distinct location count	> 4 unique cities
Spending Ratio	amount / user_mean	> 3× personal average
Velocity	Transaction burst count	> 20 in sliding window
Features are combined into a weighted anomaly_score (0.0–1.0). Only scores ≥ 0.12 proceed to quantum classification.

🤖 LLM + RAG (Groq)
Model: llama-3.3-70b-versatile via Groq API (ultra-fast inference)
RAG: Retrieves relevant chunks from 3 regulatory policy documents:
fraud_detection_policy.txt
risk_assessment_guidelines.txt
compliance_regulations.txt
Output: 3–4 sentence risk analysis citing quantum results + regulatory context
Fallback: Quantum-data-driven explanation when LLM is unavailable
� Live Market Data Sources
QuantGuard ingests transaction data from multiple configurable sources, as recommended by Pathway's architecture:

Source	Description	API Key Required
Alpha Vantage	Real-time stock & forex quotes (AAPL, MSFT, GOOGL…) — price movements converted to transactions	Free key at alphavantage.co
Polygon.io	WebSocket trade stream for stocks/crypto with real-time volume data	Free key at polygon.io
Socket / Kafka	TCP socket listener for Kafka consumers or custom event producers	None
Demo Mode	Simulated market events with intraday price curves, volume clustering, flash crashes	None
Market events are transformed into fraud-relevant transactions:

Price spikes (>3% change) → high-value suspicious transactions
Volume bursts (>2.5× rolling average) → velocity fraud patterns
Price deviation (>2σ from mean) → statistical anomaly patterns

Run live data source standalone

python live_data_source.py --source demo              # No API key needed
python live_data_source.py --source alpha --symbols AAPL MSFT GOOGL
python live_data_source.py --source socket             # Listen on port 9999
Or control it directly from the dashboard using the Live Data Stream panel.

📋 Pathway LLM xPack Integration
QuantGuard integrates Pathway's LLM xPack for live retrieval, automated reporting, and explainable AI:

Capability	Description
Live RAG	Real-time Retrieval-Augmented Generation over fraud alerts + policy documents. DocumentStore auto-indexes new data as it arrives
Automated Reports	5 report types generated on demand: Executive Summary, Trend Analysis, Compliance Report, Risk Assessment, Green Impact
Explainable Insights	Per-transaction evidence chains: ML features → quantum states → risk factor decomposition → regulatory context
Credit Rationale	Per-user credit decision rationale with risk scoring, compliant with RBI fair lending guidelines
Architecture follows the Pathway LLM xPack pattern:

DocumentStore — in-memory vector index wired through pw.io.fs.read + pw.io.subscribe (data flows through Pathway's engine, not side-loaded)
LiveRAGPipeline — retrieval → augment → generate (equivalent to BaseRAGQuestionAnswerer)
ReportEngine — automated report generation with LLM summarization
InsightEngine — explainable AI with evidence chains and risk factor decomposition
�🖥️ Dashboard Features
Real-time transaction feed via WebSocket (auto-updates on new data)
Live alert stream with risk-level colour coding (CRITICAL / HIGH / MEDIUM / LOW)
Green Bharat sustainability tracker — live metrics: frauds detected, funds protected, clean water/trees/CO₂ equivalents
On-demand quantum analysis panel:
Interactive Three.js 3D Bloch spheres (OrbitControls, auto-rotate, state vector arrow)
Animated SVG quantum circuit renderer with ZZFeatureMap/Ansatz section labels
Measurement probability bar chart (|00⟩, |01⟩, |10⟩, |11⟩) with shimmer animation
Risk gauge with animated SVG arc
LLM-powered AI explanation (Groq RAG)
Particle background and glassmorphic dark-mode design
IBM hardware badge showing active quantum backend
System stats: total transactions, alert rate, backend status, LLM model
Live Data Stream controls — start/stop Alpha Vantage, Demo, or Socket feeds from the UI
Pathway LLM xPack panel — generate automated reports (5 types), credit decision rationale, all from the dashboard
AI Regulatory Assistant — RAG-powered Q&A over fraud policy documents
📂 Project Structure
QuantGuard/
├── data_source.py          # Synthetic transaction generator (15 users, 4 fraud patterns)
├── live_data_source.py     # Live market API integration (Alpha Vantage, Polygon.io, Socket/Kafka, Demo)
├── pathway_engine.py       # Pathway streaming pipeline — dual use-case: fraud detection + log anomaly detection
├── pathway_compat.py       # Cross-platform processing engine + ML scoring + quantum + LLM
├── pathway_llm_xpack.py    # Pathway LLM xPack: DocumentStore (pw.io wired), Live RAG, Reports, Insights
├── train_vqc.py            # VQC weight optimiser (COBYLA, 500 iterations — produces weights for quantum_classifier)
├── quantum_classifier.py   # VQC classifier (IBM hardware + numpy fallback)
├── llm_engine.py           # Groq LLM engine with RAG over policy documents
├── main_api.py             # FastAPI server + WebSocket + dashboard + xPack API + log anomaly endpoints
├── kafka_producer.py       # Kafka producer sidecar — publishes synthetic transactions to Kafka topic
├── evaluate_rag.py         # RAGAS evaluation suite (10 golden QA pairs, 4 metrics)
├── Dockerfile              # Multi-stage Docker build (builder + runtime)
├── docker-compose.yml      # Compose: app + Kafka + Zookeeper + kafka-producer (all default profile)
├── .dockerignore           # Docker build exclusions
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── tests/                  # Pytest test suite (41+ tests)
│   ├── conftest.py
│   ├── test_quantum.py
│   ├── test_ml_scoring.py
│   └── test_api.py
├── data/
│   ├── transactions.jsonl         # Live transaction stream
│   ├── high_risk_alerts.jsonl     # Enriched fraud alerts
│   ├── log_anomaly_alerts.jsonl   # Log anomaly detection alerts
│   ├── persistence/               # Pathway state snapshots (fault tolerance)
│   └── policies/                  # Regulatory documents for RAG
│       ├── fraud_detection_policy.txt
│       ├── risk_assessment_guidelines.txt
│       └── compliance_regulations.txt
🚦 How to Run

1. Set up environment

python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install qiskit qiskit-ibm-runtime   # For IBM Quantum hardware

2. Configure API keys

cp .env.example .env

Edit .env → add IBMQ_API_KEY, GROQ_API_KEY

Optional: ALPHA_VANTAGE_API_KEY (free at alphavantage.co)

Optional: POLYGON_API_KEY (free at polygon.io)

3. Start services (choose data source)

Option A: Synthetic generator

python data_source.py          # Terminal 1: Synthetic transactions

Option B: Live market data (Alpha Vantage / Polygon / Demo)

python live_data_source.py --source demo     # Simulated market events
python live_data_source.py --source alpha    # Real Alpha Vantage API
python live_data_source.py --source socket   # TCP/Kafka socket listener

4. Start processing + API

python pathway_engine.py       # Terminal 2: Streaming engine + ML + Quantum + LLM
python main_api.py             # Terminal 3: Dashboard API (has built-in live source controls)

5. Run RAGAS evaluation

python evaluate_rag.py --verbose   # Full RAG quality evaluation (10 golden QA pairs)

5. Open dashboard → http://localhost:8000

Use the "Live Data Stream" panel to start/stop market feeds from the UI

🔑 API Endpoints
Method	Endpoint	Description
GET	/	Real-time dashboard (SPA)
GET	/api/stats	System statistics + quantum backend info
GET	/api/alerts	Recent high-risk alerts
GET	/api/transactions	Recent transactions
POST	/api/analyze	On-demand quantum + LLM analysis
POST	/api/rag/query	RAG query over policy documents
GET	/api/alerts/summary	AI-generated executive summary
GET	/api/quantum/info	Quantum circuit metadata
WS	/ws	WebSocket for live updates
Pathway LLM xPack	
POST	/api/xpack/rag	Live RAG — real-time indexed alerts + policies
POST	/api/xpack/report	Automated report generation (5 types)
POST	/api/xpack/insight	Explainable AI insight for a transaction
POST	/api/xpack/credit	Credit decision rationale for a user
GET	/api/xpack/status	xPack capabilities & indexed doc count
Log Anomaly Detection	
GET	/api/logs/alerts	Recent log anomaly alerts
GET	/api/logs/stats	Log anomaly pipeline statistics
GET	/api/logs/services	Per-service health from log anomalies
Live Data Sources	
POST	/api/live/start	Start live market data (Alpha Vantage / Demo / Socket)
POST	/api/live/stop	Stop live data source
GET	/api/live/status	Live source running status
� Docker Deployment

Quick start with Docker

docker build -t quantguard .
docker run -p 8000:8000 --env-file .env quantguard

Or with Docker Compose (full stack: API + Kafka + producer)

docker compose up --build
Full Kafka end-to-end by default — docker compose up starts Zookeeper, Kafka, the QuantGuard app, and a Kafka producer sidecar that publishes synthetic transactions to the quantguard-transactions topic. The Pathway engine's KafkaSubject consumes from this topic automatically.

Multi-stage build — builder stage compiles native wheels, runtime stage is slim (~200MB). Persistent volumes — alert data + Pathway state snapshots survive container restarts. Health check — built-in /api/health endpoint polled every 30s.

🔄 Fault Tolerance (Pathway Persistence)
QuantGuard uses pw.persistence.Config with a filesystem backend to checkpoint streaming operator state:

Configured in pathway_engine.py

persistence_cfg = pw.persistence.Config(
pw.persistence.Backend.filesystem("data/persistence"),
snapshot_interval_ms=10_000,   # checkpoint every 10 seconds
)
pw.run(persistence_config=persistence_cfg)
On crash/restart: The engine resumes from the last checkpoint instead of reprocessing the full history
Docker Compose: Persistence directory is mounted as a named volume (quantguard-persistence)
Configurable: Set PATHWAY_PERSISTENCE_DIR env var to override the storage path
📊 RAGAS Evaluation
QuantGuard includes a RAGAS-inspired evaluation suite (evaluate_rag.py) that measures RAG pipeline quality using 4 standard metrics:

Metric	What It Measures	Range
Faithfulness	Is the answer grounded in retrieved context?	0.0 – 1.0
Answer Relevancy	Is the answer relevant to the question?	0.0 – 1.0
Context Precision	Are retrieved documents relevant to the query?	0.0 – 1.0
Context Recall	Does context contain information needed for the answer?	0.0 – 1.0
Overall	Weighted harmonic mean of all 4 metrics	0.0 – 1.0
python evaluate_rag.py --verbose              # Full evaluation with details
python evaluate_rag.py --json --output results.json  # Machine-readable output
The golden test set contains 10 hand-crafted QA pairs covering regulatory compliance, fraud detection policies, quantum classification, ML features, and Green Bharat impact.

�💎 Innovation Factor
Dual Streaming Use-Cases: Two parallel Pathway pipelines in a single pw.run() — fraud detection AND log anomaly detection, demonstrating real multi-stream architecture
Pathway-Native DocumentStore: Vector index wired through pw.io.fs.read → pw.io.subscribe — data flows through Pathway's Rust engine, not side-loaded into Python
End-to-End Kafka: docker compose up starts Zookeeper + Kafka + producer sidecar + app — full Kafka pipeline demonstrated out of the box
Live Market API Integration: Real-time transaction streams from Alpha Vantage stock API, Polygon.io WebSocket, TCP/Kafka sockets — with demo mode for offline testing. Market price movements (spikes, volume bursts) are transformed into fraud-relevant transaction events
Pathway LLM xPack: Full implementation of Pathway's LLM extension — DocumentStore with live indexing, Live RAG pipeline, automated report generation (5 types: executive summary, trend analysis, compliance, risk assessment, green impact), credit decision rationale, and explainable AI insights
Green Bharat Impact: Every fraud blocked is quantified as sustainability impact (clean water, trees, CO₂ offset) with cited conversion benchmarks — visible live on the dashboard
Dual Quantum Backend: Supports real IBM QPU (ibm_fez/ibm_torino) with automatic fallback to a mathematically correct custom numpy simulator
Adaptive ML: No hardcoded dollar thresholds — 6-feature anomaly detection learns each user's personal spending profile via rolling statistics
Hybrid Architecture: Live Market APIs → Pathway streams → ML scores → Quantu

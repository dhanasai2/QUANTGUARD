QuantGuard: Real-Time Fraud Detection System

A real-time fraud detection system combining streaming pipelines, statistical anomaly detection, quantum classification, and LLM-based explainability.

---

1. Overview

QuantGuard is designed to detect fraudulent financial transactions in real time using a layered architecture. Unlike traditional batch-based systems, it processes events as they arrive and produces immediate risk assessments.

The system integrates distributed streaming, adaptive anomaly detection, quantum-enhanced classification, and language-model-based explanations to provide both detection and interpretability.

---

2. System Architecture

Architecture Diagram

"System Architecture" (./assets/architecture.png)

Data Flow Overview

Data Source
   │
   ▼
Streaming Engine (Pathway / Kafka)
   │
   ▼
ML Anomaly Scoring
   │
   ▼
Quantum Classification (Qiskit)
   │
   ▼
LLM Analysis (RAG)
   │
   ▼
API Layer (FastAPI)
   │
   ▼
Dashboard (WebSocket UI)

---

3. Core Components

3.1 Data Ingestion

- Synthetic transaction generator
- Live data streams (configurable sources)
- Kafka-based ingestion

---

3.2 Streaming Engine

- Real-time processing pipeline
- Rolling user statistics
- Event routing and enrichment

---

3.3 Machine Learning Layer

Adaptive anomaly detection based on user behavior:

- Z-score deviation
- Interquartile range (IQR)
- Percentile rank
- Transaction velocity
- Spending ratio
- Location entropy

Only high-risk transactions proceed to deeper analysis.

---

3.4 Quantum Classification

- Variational Quantum Circuit (VQC) model
- Supports IBM Quantum hardware and local simulation
- Binary classification: fraud vs safe

---

3.5 LLM-Based Explanation

- Context-aware explanations for flagged transactions
- Retrieval over policy and compliance documents
- Provides interpretable reasoning

---

3.6 API and Dashboard

- FastAPI-based backend
- WebSocket-based real-time updates
- Visualization of transactions and alerts

---

4. Technology Stack

Backend

- Python
- FastAPI
- Apache Kafka
- Pathway

ML / Quantum

- NumPy / SciPy
- IBM Qiskit

Infrastructure

- Docker
- WebSockets

---

5. Project Structure

QuantGuard/
├── data_source.py
├── live_data_source.py
├── pathway_engine.py
├── quantum_classifier.py
├── llm_engine.py
├── main_api.py
├── kafka_producer.py
├── train_vqc.py
├── evaluate_rag.py
├── tests/
├── data/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

---

6. Running the System

6.1 Local Setup

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

6.2 Start Services

python data_source.py
python pathway_engine.py
python main_api.py

6.3 Access Dashboard

http://localhost:8000

---

7. Docker Deployment

docker build -t quantguard .
docker run -p 8000:8000 --env-file .env quantguard

Or:

docker compose up --build

---

8. API Overview

Method| Endpoint| Description
GET| /| Dashboard
GET| /api/stats| System metrics
GET| /api/alerts| Fraud alerts
GET| /api/transactions| Transaction stream
POST| /api/analyze| On-demand analysis
WS| /ws| Real-time updates

---

9. System Characteristics

- Real-time event processing
- Modular architecture
- Scalable streaming pipeline
- Hybrid ML + quantum approach
- Explainable outputs via LLM

---

10. Limitations

- Quantum model operates on low-dimensional features
- Performance depends on data quality and feature engineering
- Designed for experimentation and demonstration

---

11. Future Work

- Improved anomaly detection models
- Expanded quantum feature space
- Distributed deployment at scale
- Advanced explainability and reporting

---

12. Author

Gundumogula Dhana Sai
B.Tech Information Technology

Email: saigundumogula5@gmail.com
GitHub: https://github.com/dhanasai2

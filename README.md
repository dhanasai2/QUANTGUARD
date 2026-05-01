# QuantGuard: Real-Time Fraud Detection System

A real-time fraud detection system combining streaming pipelines, statistical anomaly detection, quantum classification, and LLM-based explainability.

## Overview

QuantGuard processes financial transactions in real time and detects fraud using a layered architecture.

Unlike batch-based systems, it evaluates transactions as they arrive and produces immediate risk assessments.

## Architecture

### System Diagram

![System Architecture](Assets/architecture.png)

### Data Flow
Data Source
     ↓ 
Streaming Engine (Pathway / Kafka) 
     ↓ 
ML Anomaly Scoring ↓ Quantum Classification (Qiskit) 
     ↓ 
LLM Analysis (RAG) 
     ↓ 
API Layer (FastAPI) 
     ↓ 
Dashboard (WebSocket UI)

## Core Components

### Data Ingestion
- Synthetic transaction generator  
- Live data streams  
- Kafka-based ingestion  

### Streaming Engine
- Real-time processing  
- Rolling user statistics  
- Event routing  

### Machine Learning Layer
- Z-score deviation  
- Interquartile range (IQR)  
- Percentile rank  
- Transaction velocity  
- Spending ratio  
- Location entropy  

### Quantum Classification
- Variational Quantum Circuit (VQC)  
- IBM Quantum + simulator support  
- Binary classification (fraud / safe)  

### LLM Analysis
- Context-aware explanations  
- Retrieval from policy documents  
- Explainable outputs  

### API and Dashboard
- FastAPI backend  
- WebSocket real-time updates  
- Visualization interface  

## Technology Stack

**Backend**
- Python  
- FastAPI  
- Apache Kafka  
- Pathway  

**ML / Quantum**
- NumPy / SciPy  
- IBM Qiskit  

**Infrastructure**
- Docker  
- WebSockets  

## Project Structure
QuantGuard/ ├──data_source.py ├── live_data_source.py ├── pathway_engine.py ├── quantum_classifier.py ├── llm_engine.py ├── main_api.py ├── kafka_producer.py ├── train_vqc.py ├── evaluate_rag.py ├── tests/ ├── data/ ├── Dockerfile ├── docker-compose.yml └── requirements.txt

## Running the System

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Start Services
Bash
python data_source.py
python pathway_engine.py
python main_api.py
Open Dashboard
http://localhost:8000⁠�
Docker
Bash
docker build -t quantguard .
docker run -p 8000:8000 --env-file .env quantguard
Or:
Bash
docker compose up --build
API Endpoints
Method
Endpoint
Description
GET
/
Dashboard
GET
/api/stats
System metrics
GET
/api/alerts
Fraud alerts
GET
/api/transactions
Transactions
POST
/api/analyze
Analysis
WS
/ws
Real-time updates
Key Characteristics
Real-time processing
Modular architecture
Scalable pipeline
ML + quantum hybrid
Explainable outputs
Limitations
Quantum model uses low-dimensional features
Performance depends on data quality
Designed for experimentation
Author
Gundumogula Dhana Sai
B.Tech Information Technology
Email: saigundumogula5@gmail.com
GitHub: https://github.com/dhanasai2⁠�

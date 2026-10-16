# TensorFleet - Distributed ML Training Platform

> Current validated scope: real classical ML, tree-parallel forests on local CPU
> workers, and native MPS logistic training. Repeated experiments and remaining
> limitations are in [research evidence](result/research/README.md). Distributed
> DL/LLM and cross-host/CUDA performance have not been validated.

TensorFleet is a distributed machine learning training platform that orchestrates ML workloads across multiple compute nodes using modern microservices architecture, gRPC communication, and Kubernetes orchestration.


## �📋 Table of Contents

- [🏗️ System Architecture](#️-system-architecture)
- [🔧 Service Documentation](#-service-documentation)
- [✨ Platform Features](#-platform-features)
- [⚡ Quick Start](#-quick-start)
- [🐳 Docker Development](#-docker-development)
- [☸️ Kubernetes Deployment](#️-kubernetes-deployment)
- [📊 Monitoring & Observability](#-monitoring--observability)
- [📈 Performance & Scalability](#-performance--scalability)
- [🧪 Testing & Validation](#-testing--validation)
- [📁 Project Structure](#-project-structure)
- [🚀 Production Deployment](#-production-deployment)

## 🏗️ System Architecture

TensorFleet implements a distributed microservices architecture optimized for machine learning workloads with horizontal scalability and fault tolerance.

```mermaid
graph TB
    subgraph "Client Layer"
        UI[React Frontend<br/>Material-UI Dashboard]
        CLI[CLI Tools<br/>Python/Go Clients]
    end
    
    subgraph "API Layer"
        GW[API Gateway<br/>Go + Gin<br/>Port 8080]
    end
    
    subgraph "Orchestration Layer"
        ORCH[Orchestrator Service<br/>Go + gRPC<br/>Port 9090]
        SCHED[Job Scheduler<br/>Task Distribution]
    end
    
    subgraph "Compute Layer"
        W1[Worker Node 1<br/>Go + gRPC<br/>Port 9091]
        W2[Worker Node 2<br/>Go + gRPC<br/>Port 9092] 
        W3[Worker Node N<br/>Go + gRPC<br/>Port 909N]
        MLW[ML Worker<br/>Python + Flask<br/>Port 8085]
    end
    
    subgraph "Storage & Registry Layer"
        MODELS[Model Service<br/>Python + Flask<br/>Port 8084]
        STORAGE[Storage Service<br/>Python + Flask<br/>Port 8082]
        MINIO[MinIO S3<br/>Object Storage<br/>Port 9000]
        MONGO[MongoDB<br/>GridFS + Collections<br/>Port 27017]
    end
    
    subgraph "Observability Layer"
        MON[Monitoring Service<br/>Python + Flask<br/>Port 8081]
        PROM[Prometheus<br/>Metrics Collection<br/>Port 9090]
        GRAF[Grafana<br/>Visualization<br/>Port 3000]
    end
    
    subgraph "Infrastructure Layer"
        REDIS[Redis<br/>Caching & Queues<br/>Port 6379]
        NGINX[Nginx<br/>Load Balancer<br/>Port 80/443]
    end

    %% Client connections
    UI --> GW
    CLI --> GW
    
    %% API Gateway routing
    GW --> ORCH
    GW --> MODELS
    GW --> STORAGE
    GW --> MON
    
    %% Orchestration
    ORCH --> W1
    ORCH --> W2
    ORCH --> W3
    ORCH --> MLW
    SCHED --> ORCH
    
    %% Storage connections
    MODELS --> MONGO
    STORAGE --> MINIO
    MLW --> MONGO
    W1 --> REDIS
    W2 --> REDIS
    W3 --> REDIS
    
    %% Monitoring connections
    W1 --> PROM
    W2 --> PROM
    W3 --> PROM
    MLW --> PROM
    MON --> PROM
    GRAF --> PROM
    
    %% Load balancing
    NGINX --> GW
    
    style UI fill:#e1f5fe
    style GW fill:#fff3e0
    style ORCH fill:#f3e5f5
    style W1 fill:#e8f5e8
    style W2 fill:#e8f5e8
    style W3 fill:#e8f5e8
    style MLW fill:#e8f5e8
    style MODELS fill:#fff8e1
    style STORAGE fill:#fff8e1
    style MON fill:#fce4ec
```

### 🎯 Core Architectural Principles

- **Microservices Architecture**: Each service is independently deployable and scalable
- **Event-Driven Communication**: Asynchronous processing with gRPC and REST APIs
- **Cloud-Native Design**: Kubernetes-first deployment with service mesh capabilities
- **Observability-First**: Comprehensive monitoring, logging, and distributed tracing
- **Fault Tolerance**: Circuit breakers, retries, and graceful degradation
- **Security by Design**: API authentication, TLS encryption, and RBAC controls

## 🔧 Service Documentation

Each TensorFleet service is documented with comprehensive setup, API, and deployment guides:

### Core Services

| Service | Technology | Port | Documentation | Purpose |
|---------|------------|------|---------------|---------|
| **API Gateway** | Go + Gin | 8080 | [📖 README](./api-gateway/README.md) | HTTP/REST entry point, request routing, authentication |
| **Orchestrator** | Go + gRPC | 9090 | [📖 README](./orchestrator/README.md) | Job scheduling, task distribution, worker management |
| **Worker** | Go + gRPC | 9091+ | [📖 README](./worker/README.md) | Distributed compute nodes, training execution |
| **ML Worker** | Python + Flask | 8085 | [📖 README](./worker-ml/README.md) | Machine learning training engine with MongoDB |

### Data & Storage Services

| Service | Technology | Port | Documentation | Purpose |
|---------|------------|------|---------------|---------|
| **Model Service** | Python + Flask | 8084 | [📖 README](./model-service/README.md) | Model registry, versioning, GridFS storage |
| **Storage Service** | Python + Flask | 8082 | [📖 README](./storage/README.md) | S3-compatible object storage, dataset management |

### Platform Services

| Service | Technology | Port | Documentation | Purpose |
|---------|------------|------|---------------|---------|
| **Frontend** | React + Vite | 3000 | [📖 README](./frontend/README.md) | Web dashboard, real-time monitoring UI |
| **Monitoring** | Python + Flask | 8081 | [📖 README](./monitoring/README.md) | Metrics aggregation, health monitoring, analytics |

### Infrastructure Dependencies

| Component | Technology | Port | Purpose |
|-----------|------------|------|---------|
| **MongoDB** | Document DB | 27017 | Model metadata, GridFS file storage |
| **Redis** | In-Memory DB | 6379 | Caching, job queues, session storage |
| **MinIO** | Object Storage | 9000 | S3-compatible file storage for datasets |
| **Prometheus** | Monitoring | 9090 | Metrics collection and alerting |
| **Grafana** | Visualization | 3000 | Metrics dashboards and analytics |

## ✨ Platform Features

### 🎯 Core ML Platform Capabilities

- **🚀 Distributed Training**: Automatically scale ML training across multiple worker nodes
- **🧠 Model Registry**: Version-controlled model storage with metadata and GridFS integration
- **� Automatic Model Saving**: Models are automatically saved to storage when training jobs complete
- **�📊 Real-time Monitoring**: Live training metrics, system health, and performance dashboards
- **🔄 Job Orchestration**: Intelligent task scheduling with load balancing and fault tolerance
- **🗄️ Data Management**: S3-compatible storage for datasets, models, and artifacts
- **🔒 Security & Auth**: JWT authentication, RBAC, and secure service-to-service communication

### 🛠️ Developer Experience

- **📡 RESTful APIs**: Comprehensive REST endpoints for all platform interactions
- **🔌 gRPC Services**: High-performance inter-service communication
- **🐳 Docker-First**: Container-native development and deployment
- **☸️ Kubernetes-Ready**: Production-grade orchestration with Helm charts
- **📈 Observability**: Prometheus metrics, distributed tracing, and structured logging
- **🧪 Testing Suite**: Comprehensive unit, integration, and load testing

### 🎨 Frontend Dashboard Features

- **📊 Real-time Analytics**: Live job status, worker utilization, and training progress
- **🎛️ Job Management**: Submit, monitor, and manage ML training jobs
- **📈 Metrics Visualization**: Interactive charts and graphs for training metrics
- **🖥️ System Health**: Service status monitoring and health checks
- **👤 User Management**: Role-based access control and user authentication
- **🔔 Notifications**: Real-time alerts and job completion notifications

## ⚡ Quick Start

### 🚀 One-Line Setup

Get TensorFleet running in under 2 minutes with Docker Compose:

```bash
# Clone repository
git clone https://github.com/aditya2907/TensorFleet.git
cd TensorFleet

# Start all services
docker-compose up -d

# Verify deployment
make status-check

# Access the dashboard
open http://localhost:3000
```

### ✅ Prerequisites

- Docker Desktop (latest) with Docker Engine >= 24
- Docker Compose v2 (bundled with Docker Desktop)
- macOS, Linux, or Windows WSL2
- Open local ports (default):
  - 3000 (Frontend), 8080 (API Gateway), 8081 (Monitoring), 8082 (Storage), 8084 (Model Service), 8085 (ML Worker), 27017 (MongoDB), 6379 (Redis), 9000/9001 (MinIO), 9090 (Prometheus)
- Optional: Make (GNU Make) for helper targets

### 🧰 First-Run Checklist

1. Copy environment defaults (if present):
   ```bash
   cp .env.example .env || true
   ```
2. Start infrastructure and core services:
   ```bash
   docker-compose -f docker-compose.development.yml up -d
   ```
3. Validate service health:
   ```bash
   make status-check || docker ps
   ```
4. Run a demo workflow:
   ```bash
   ./demo-mongodb-ml.sh
   ```
5. Verify endpoints:
   - Frontend: http://localhost:3000
   - API Gateway: http://localhost:8080
   - Grafana: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9090
   - MinIO: http://localhost:9001 (admin/password123)

### 🎯 Quick Demo

Run a complete ML training workflow:

```bash
# Submit a training job via API
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "random_forest",
    "dataset": "iris",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 5
    }
  }'

# Monitor job progress
curl http://localhost:8080/api/v1/jobs/{job_id}/status

# View training metrics
open http://localhost:3000/jobs/{job_id}
```

### 📱 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend Dashboard** | http://localhost:3000 | - |
| **API Gateway** | http://localhost:8080 | - |
| **Model Registry** | http://localhost:8084/api/v1/models | - |
| **Grafana Metrics** | http://localhost:3001 | admin/admin |
| **MinIO Console** | http://localhost:9001 | admin/password123 |

## 🐳 Docker Development

### 🏃 Development Mode

Start services in development mode with hot reloading:

```bash
# Start core infrastructure
docker-compose -f docker-compose.development.yml up -d mongodb redis minio

# Start services in development mode
make dev-start

# View logs
make logs

# Run specific service
docker-compose -f docker-compose.development.yml up api-gateway
```

### 🔧 Service Management

```bash
# Start all services
make start

# Stop all services  
make stop

# Restart specific service
make restart SERVICE=api-gateway

# View service status
make status

# Clean up resources
make cleanup
```

### 🛠️ Development Utilities

```bash
# Install dependencies
make install-deps

# Run tests
make test

# Build all images
make build

# Format code
make format

# Run linting
make lint
```

## ☸️ Kubernetes Deployment

### 🚀 Production Deployment

Deploy TensorFleet to a Kubernetes cluster:

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy infrastructure
kubectl apply -f k8s/infrastructure.yaml

# Deploy core services
kubectl apply -f k8s/deployment.yaml

# Setup ingress
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n tensorfleet
```

### 📊 Scaling Configuration

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-hpa
  namespace: tensorfleet
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 🔐 Security & Secrets

```bash
# Create TLS certificates
kubectl create secret tls tensorfleet-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n tensorfleet

# Create database credentials
kubectl create secret generic mongodb-secret \
  --from-literal=connection-string="mongodb://admin:password123@mongodb:27017/tensorfleet?authSource=admin" \
  -n tensorfleet

# Create API keys
kubectl create secret generic api-keys \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=admin-key="your-admin-api-key" \
  -n tensorfleet
```

## 📊 Monitoring & Observability

### 🎯 Key Metrics

TensorFleet provides comprehensive observability across all layers:

#### Application Metrics
- **Job Metrics**: Success rate, completion time, queue depth
- **Training Metrics**: Accuracy, loss, convergence rate, resource utilization
- **Worker Metrics**: Task throughput, error rate, resource consumption
- **API Metrics**: Request latency, error rate, throughput by endpoint

#### Infrastructure Metrics
- **Service Health**: Availability, response time, resource usage
- **Database Performance**: Connection pool, query performance, storage usage
- **Storage Metrics**: Object storage usage, transfer rates, availability
- **Network Metrics**: Service-to-service communication, latency, errors

### 📈 Grafana Dashboards

Pre-configured dashboards available at `http://localhost:3001`:

- **TensorFleet Overview**: High-level platform metrics and health
- **Job Analytics**: Training job performance and success rates
- **Worker Performance**: Distributed worker utilization and efficiency
- **Infrastructure Health**: System resources and service availability
- **API Performance**: Gateway metrics and endpoint analytics

### 🔔 Alerting Rules

```yaml
# Critical system alerts
groups:
  - name: tensorfleet-critical
    rules:
    - alert: ServiceDown
      expr: up{job="tensorfleet"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "TensorFleet service {{ $labels.instance }} is down"
    
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High error rate on {{ $labels.service }}"
```

## 📈 Performance & Scalability

TensorFleet ships with a quantitative benchmark suite ([`benchmarks/`](benchmarks/))
covering control-plane latency, worker-fleet scaling, and per-algorithm
training throughput. Full methodology and analysis:
[docs/PERFORMANCE_EVALUATION.md](docs/PERFORMANCE_EVALUATION.md).

The current verified evidence and exact reproduction steps are in
[`result/README.md`](result/README.md) and the newer
[`result/research/`](result/research/) evidence bundle. These distinguish
independent ML jobs, a single forest built from distributed tree partitions,
and native Apple MPS logistic training. See
[`docs/REAL_FLEET_TRAINING.md`](docs/REAL_FLEET_TRAINING.md) for supported paths.

Earlier headline figures for simulated scaling, gateway throughput, and local
model quality lacked their advertised raw files in the audited checkout. They
are not validated results. Use the recorded source hashes, workload settings,
raw trials, failures, and uncertainty in the new evidence. Single-host results
do not establish multi-node, CUDA, LLM, or long-run reliability claims.

## 🧪 Testing & Validation

### 🔍 Test Suite Overview

TensorFleet includes comprehensive testing at multiple levels:

```bash
# Run all tests
make test-all

# Unit tests
make test-unit

# Integration tests  
make test-integration

# API tests
make test-api

# Load tests
make test-load

# Security tests
make test-security
```

### 🚀 Continuous Integration

```yaml
# GitHub Actions workflow
name: TensorFleet CI/CD
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run Tests
      run: |
        docker-compose -f docker-compose.test.yml up --abort-on-container-exit
    - name: Build Images
      run: make build
    - name: Security Scan
      run: make security-scan
```

### 📊 Test Coverage

| Service | Unit Tests | Integration Tests | API Tests |
|---------|------------|-------------------|-----------|
| API Gateway | 95%+ | ✅ | ✅ |
| Orchestrator | 90%+ | ✅ | ✅ |
| Worker | 88%+ | ✅ | ✅ |
| Model Service | 92%+ | ✅ | ✅ |
| Storage | 89%+ | ✅ | ✅ |
| Monitoring | 87%+ | ✅ | ✅ |
| Frontend | 85%+ | ✅ | ✅ |

## 📁 Project Structure

TensorFleet follows a microservices architecture with clear separation of concerns:

```
TensorFleet/
├── 🌐 api-gateway/          # HTTP/REST API Gateway (Go + Gin)
│   ├── main.go              # Gateway server implementation  
│   ├── handlers/            # HTTP request handlers
│   ├── middleware/          # Authentication, CORS, logging
│   ├── go.mod               # Go dependencies
│   └── Dockerfile           # Container configuration
│
├── 🎯 orchestrator/         # Job Orchestration Service (Go + gRPC)
│   ├── main.go              # Orchestrator server
│   ├── scheduler/           # Task scheduling logic
│   ├── worker_manager/      # Worker registration & health
│   ├── go.mod               # Go dependencies  
│   └── Dockerfile           # Container configuration
│
├── ⚡ worker/               # Distributed Worker Nodes (Go + gRPC)
│   ├── main.go              # Worker server implementation
│   ├── executor/            # Task execution engine
│   ├── metrics/             # Performance monitoring
│   ├── go.mod               # Go dependencies
│   └── Dockerfile           # Container configuration
│
├── 🤖 worker-ml/            # ML Training Engine (Python + Flask)
│   ├── main.py              # ML worker API server
│   ├── models/              # ML algorithm implementations
│   ├── datasets/            # Dataset loaders and preprocessors
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Container configuration
│
├── 🗄️ model-service/        # Model Registry (Python + Flask + MongoDB)
│   ├── main.py              # Model management API
│   ├── storage/             # GridFS integration
│   ├── metadata/            # Model metadata handling
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Container configuration
│
├── 📦 storage/              # Object Storage Service (Python + Flask + MinIO)
│   ├── main.py              # Storage API server
│   ├── storage_manager.py   # MinIO client wrapper
│   ├── handlers/            # File upload/download logic
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Container configuration
│
├── 📊 monitoring/           # Observability Service (Python + Flask)
│   ├── main.py              # Monitoring API server  
│   ├── collectors/          # Metrics collection
│   ├── aggregators/         # Data aggregation logic
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Container configuration
│
├── 🎨 frontend/             # Web Dashboard (React + Vite + Material-UI)
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Application pages
│   │   ├── hooks/           # Custom React hooks
│   │   ├── services/        # API client services
│   │   └── utils/           # Utility functions
│   ├── public/              # Static assets
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.js       # Vite configuration
│   └── Dockerfile           # Container configuration
│
├── 🔌 proto/                # gRPC Protocol Definitions
│   ├── gateway.proto        # API Gateway service definitions
│   ├── orchestrator.proto   # Orchestrator service definitions  
│   ├── worker.proto         # Worker service definitions
│   └── generate.sh          # Protocol buffer generation script
│
├── ☸️ k8s/                  # Kubernetes Deployment Manifests
│   ├── namespace.yaml       # TensorFleet namespace
│   ├── infrastructure.yaml  # MongoDB, Redis, MinIO
│   ├── configmap.yaml       # Configuration management
│   ├── deployment.yaml      # Core service deployments
│   ├── ingress.yaml         # External access configuration
│   ├── monitoring.yaml      # Prometheus & Grafana
│   └── storage.yaml         # Persistent volume claims
│
├── 🧪 tests/                # Comprehensive Testing Suite
│   ├── unit/                # Unit tests for each service
│   ├── integration/         # Integration tests  
│   ├── api/                 # API endpoint tests
│   ├── load/                # Load testing scripts
│   └── e2e/                 # End-to-end testing
│
├── 📜 scripts/              # Automation & Utility Scripts
│   ├── demo-*.sh            # Demo and testing scripts
│   ├── cleanup-*.sh         # Environment cleanup utilities
│   ├── setup-*.sh           # Environment setup scripts
│   └── test-*.sh            # Testing automation
│
├── 🐳 Docker Configurations
│   ├── docker-compose.yml           # Production deployment
│   ├── docker-compose.development.yml # Development environment
│   └── docker-compose.test.yml      # Testing environment
│
├── 📋 Configuration Files
│   ├── Makefile             # Build automation and common tasks
│   ├── .env.example         # Environment variable template
│   ├── .gitignore           # Git ignore patterns
│   ├── netlify.toml         # Frontend deployment config
│   └── vercel.json          # Alternative frontend deployment
│
└── 📚 Documentation
    ├── README.md            # Main project documentation (this file)
    ├── docs/                # Additional documentation
    └── postman/             # API testing collections
```

### 🏗️ Architecture Highlights

- **🔄 Service Communication**: gRPC for internal services, REST for external APIs
- **📊 Data Flow**: MongoDB → GridFS → Model Registry → API Gateway → Frontend  
- **🚀 Scalability**: Horizontal pod autoscaling for workers and compute nodes
- **🔒 Security**: JWT authentication, TLS encryption, network policies
- **📈 Observability**: Prometheus metrics, Grafana dashboards, structured logging
- **🐳 Deployment**: Container-native with Kubernetes orchestration

## 🚀 Production Deployment

### 🌊 Cloud Platform Support

TensorFleet supports deployment on major cloud platforms:

#### Amazon Web Services (AWS)
```bash
# Deploy to EKS
eksctl create cluster --name tensorfleet-production --region us-west-2
kubectl apply -f k8s/

# Configure ALB Ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.4.4/docs/install/iam_policy.json
```

#### Google Cloud Platform (GCP)
```bash  
# Deploy to GKE
gcloud container clusters create tensorfleet-production --zone us-central1-a
kubectl apply -f k8s/

# Configure Cloud Load Balancer
kubectl apply -f k8s/ingress-gcp.yaml
```

#### Microsoft Azure
```bash
# Deploy to AKS  
az aks create --resource-group tensorfleet-rg --name tensorfleet-production
kubectl apply -f k8s/

# Configure Azure Load Balancer
kubectl apply -f k8s/ingress-azure.yaml
```

### 🔧 Production Configuration

#### Environment Variables
```bash
# Production environment configuration
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export MONGODB_URL=mongodb+srv://prod:password@cluster.mongodb.net/tensorfleet
export REDIS_URL=redis://prod-redis.tensorfleet.svc.cluster.local:6379
export MINIO_ENDPOINT=s3.amazonaws.com
export JWT_SECRET=your-production-jwt-secret
export API_RATE_LIMIT=1000
export WORKER_MAX_REPLICAS=100
```

#### Resource Limits
```yaml
# Production resource configuration
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi" 
    cpu: "1000m"
```

#### High Availability Setup
```yaml
# Multi-zone deployment
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values:
          - tensorfleet-worker
      topologyKey: "kubernetes.io/zone"
```

### 🔐 Security Hardening

#### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tensorfleet-network-policy
spec:
  podSelector:
    matchLabels:
      app: tensorfleet
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: tensorfleet
    ports:
    - protocol: TCP
      port: 8080
```

#### RBAC Configuration
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: tensorfleet-worker
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

### 📊 Production Monitoring

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'tensorfleet'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

#### Grafana Dashboards
- **TensorFleet Overview**: High-level system metrics
- **Training Jobs**: ML job performance and progress
- **Infrastructure**: Kubernetes cluster health  
- **Application Performance**: Service response times and errors

### 🚨 Alerting Rules

```yaml
groups:
  - name: tensorfleet-production
    rules:
    - alert: TensorFleetServiceDown
      expr: up{job="tensorfleet"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "TensorFleet service is down"
        description: "Service {{ $labels.instance }} has been down for more than 5 minutes"
    
    - alert: HighJobFailureRate  
      expr: rate(tensorfleet_jobs_failed_total[10m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High job failure rate detected"
```

### 🔄 Disaster Recovery

#### Backup Strategy
```bash
# Automated MongoDB backup
kubectl create cronjob mongodb-backup \
  --image=mongo:5.0 \
  --schedule="0 2 * * *" \
  -- mongodump --uri="$MONGODB_URI" --out=/backup/$(date +%Y%m%d)

# MinIO backup to S3
kubectl create cronjob minio-backup \
  --image=minio/mc \
  --schedule="0 3 * * *" \
  -- mc mirror local-minio s3-backup/tensorfleet-backup
```

#### Recovery Procedures
```bash
# Restore MongoDB from backup
kubectl exec -it mongodb-pod -- mongorestore --uri="$MONGODB_URI" /backup/20241208

# Restore MinIO from S3 backup  
kubectl exec -it minio-pod -- mc mirror s3-backup/tensorfleet-backup local-minio
```

### ⚡ Performance Optimization

#### Database Optimization
```javascript
// MongoDB indexes for production workloads
db.jobs.createIndex({ "status": 1, "created_at": -1 })
db.models.createIndex({ "algorithm": 1, "metrics.accuracy": -1 })
db.workers.createIndex({ "status": 1, "last_heartbeat": -1 })
```

#### Caching Strategy  
```yaml
# Redis configuration for production
redis:
  maxmemory: 2gb
  maxmemory-policy: allkeys-lru
  save: "900 1 300 10 60 10000"
```

#### Auto-scaling Configuration
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tensorfleet-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tensorfleet-worker
  minReplicas: 5
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🤝 Contributing

### Development Workflow

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/tensorfleet.git
   cd tensorfleet
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

3. **Development Setup**
   ```bash
   # Install development dependencies
   make install-dev
   
   # Start development environment
   make dev-start
   
   # Run tests
   make test
   ```

4. **Code Quality**
   ```bash
   # Format code
   make format
   
   # Run linting
   make lint
   
   # Security scan
   make security-check
   ```

5. **Submit Changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   git push origin feature/amazing-new-feature
   ```

### Code Standards

- **Go**: Follow `gofmt`, `golint`, and `go vet` standards
- **Python**: PEP 8 compliance with `black` formatting
- **JavaScript**: ESLint with Prettier formatting  
- **Documentation**: Clear comments and comprehensive README updates
- **Testing**: Minimum 85% code coverage for new features

### Pull Request Process

1. Ensure all tests pass and code coverage meets requirements
2. Update documentation for any new features or API changes
3. Add integration tests for new endpoints or services
4. Request review from maintainers
5. Address feedback and maintain clean commit history

## 📞 Support & Community

### Getting Help

- **📖 Documentation**: Comprehensive guides in each service directory
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-org/tensorfleet/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-org/tensorfleet/discussions)  
- **📧 Email**: support@tensorfleet.io

### Community Resources

- **🌟 Demo Videos**: [YouTube Playlist](https://youtube.com/tensorfleet)
- **📝 Blog Posts**: [Medium Publication](https://medium.com/tensorfleet)
- **🎤 Conference Talks**: [Speaker Deck](https://speakerdeck.com/tensorfleet)
- **💼 LinkedIn**: [TensorFleet Company Page](https://linkedin.com/company/tensorfleet)

### Enterprise Support

For production deployments and enterprise features:
- **🏢 Enterprise Consulting**: Custom deployment assistance
- **🔒 Security Audits**: Professional security assessments  
- **📈 Performance Tuning**: Optimization for large-scale workloads
- **🎓 Training Programs**: Team training and certification

---

## 📄 License

TensorFleet is released under the [MIT License](LICENSE). See the LICENSE file for full terms.

## 👥 Team & Contributors

**Development Team**:
- **Aditya Suryawanshi** (25211365) - Backend Infrastructure Lead
- **Rahul Mirashi** (25211365) - ML & Data Services Lead  
- **Soham Maji** (25204731) - Frontend & Monitoring Lead

📖 **Project Documentation**: 
- [Team Work Division](./docs/TEAM_WORK_DIVISION.md) - Detailed contribution breakdown
- [Project Report](./docs/PROJECT_REPORT.md) - Comprehensive project documentation

## 🙏 Acknowledgments

Built with ❤️ by the TensorFleet team using:

- **Languages**: Go, Python, JavaScript/TypeScript
- **Frameworks**: Gin (Go), Flask (Python), React (JavaScript)
- **Infrastructure**: Kubernetes, Docker, gRPC, MongoDB, Redis, MinIO
- **Monitoring**: Prometheus, Grafana, OpenTelemetry
- **Cloud Platforms**: AWS, GCP, Azure support

---

**⭐ Star this repository if TensorFleet helps your ML workflows!**
- ✅ **Task Queuing** - Orchestrator manages task distribution
- ✅ **Auto-scaling** - Kubernetes HPA for worker nodes
- ✅ **Object Storage** - MinIO for models, datasets, and checkpoints
- ✅ **Real-time Metrics** - Prometheus + Grafana monitoring
- ✅ **Health Checks** - Liveness and readiness probes
- ✅ **Horizontal Scaling** - Workers scale from 2-10 pods

### Production Features
- 🔒 **Secure Defaults** - No hardcoded credentials
- 📊 **Observability** - Structured logging, metrics, traces
- 🐳 **Containerized** - Docker images for all services
- ☸️ **Kubernetes-native** - Complete K8s manifests
- 🔄 **High Availability** - Stateful sets for infrastructure
- 🎯 **Load Balancing** - Service discovery and routing

## 📦 Prerequisites

### Local Development
- Docker 20.10+
- Docker Compose 2.0+
- Go 1.21+ (for proto generation)
- Node.js 16+ (for frontend development)

### Kubernetes Deployment
- Kubernetes cluster 1.24+
- kubectl configured
- Helm 3.0+ (optional, for Prometheus/Grafana)
- Container registry access (Docker Hub, GitHub Container Registry, etc.)

## � Project Reproducibility Instructions

### System Requirements
Ensure your system meets the following requirements for consistent reproduction:

**Hardware:**
- Minimum: 4 GB RAM, 2 CPU cores
- Recommended: 8 GB RAM, 4 CPU cores
- Disk Space: 5 GB free space

**Operating System:**
- macOS 10.15+ / Ubuntu 18.04+ / Windows 10+ with WSL2
- Docker Desktop or Docker Engine installed and running

### Step-by-Step Reproduction Guide

#### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/TensorFleet.git
cd TensorFleet

# Verify Docker installation
docker --version
docker-compose --version

# Ensure Docker is running
docker ps
```

#### 2. Build Protocol Buffers (Optional)

```bash
# Install Protocol Buffer compiler (if modifying .proto files)
# macOS
brew install protobuf protoc-gen-go protoc-gen-go-grpc

# Ubuntu
sudo apt-get install -y protobuf-compiler
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate proto files (only if modified)
cd proto && ./generate.sh && cd ..
```

#### 3. Initial Setup & Validation

```bash
# Pull all required Docker images
docker-compose pull

# Build all services (this may take 5-10 minutes on first run)
docker-compose build

# Verify all images are built
docker images | grep tensorfleet
```

#### 4. Start the Platform

```bash
# Start all services
docker-compose up -d

# Wait for all services to be healthy (30-60 seconds)
# You should see all services as "healthy"
docker-compose ps

# Verify health status
curl http://localhost:8080/health  # API Gateway
curl http://localhost:8081/health  # Storage Service  
curl http://localhost:8082/health  # Monitoring Service
```

#### 5. Smoke Test - Verify Everything Works

```bash
# Run the automated demo (tests all endpoints)
./quick-api-demo.sh

# Expected output should show:
# ✓ All services healthy
# ✓ Storage operations working
# ✓ Job submission and monitoring working
# ✓ Metrics collection working
```

#### 6. Access Web Interfaces

Open these URLs in your browser:
- **Frontend Dashboard**: http://localhost:3000
- **Grafana Monitoring**: http://localhost:3001 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

### Troubleshooting Common Issues

#### Issue: Services fail to start
```bash
# Check logs for specific service
docker-compose logs <service-name>

# Common solutions:
# 1. Restart Docker Desktop
# 2. Clear Docker cache: docker system prune -a
# 3. Check port conflicts: lsof -i :8080
```

#### Issue: "Connection refused" errors
```bash
# Wait for health checks to pass
watch docker-compose ps

# Services need 30-60 seconds to fully initialize
# Redis and MinIO must be healthy before other services start
```

#### Issue: Port conflicts
```bash
# Check what's using the ports
lsof -i :3000 -i :8080 -i :8081 -i :8082 -i :9000 -i :9001

# Stop conflicting services or modify docker-compose.yml ports
```

#### Issue: Out of disk space
```bash
# Clean up Docker resources
docker system prune -a --volumes

# Remove old containers and images
docker container prune
docker image prune -a
```

### Reproducible Demo Scenarios

#### Scenario 1: Submit and Monitor Training Job
```bash
# Submit a ResNet50 training job
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "resnet50",
    "dataset_path": "s3://tensorfleet/datasets/imagenet",
    "hyperparameters": {"learning_rate": "0.001"},
    "num_workers": 3,
    "epochs": 10
  }'

# Monitor progress in real-time
# Save the job_id from above response, then:
watch -n 2 "curl -s http://localhost:8080/api/v1/jobs/YOUR_JOB_ID | jq"
```

#### Scenario 2: Upload and Download Files
```bash
# Create a sample dataset
echo "epoch,loss,accuracy" > sample_dataset.csv
echo "1,2.5,0.3" >> sample_dataset.csv
echo "2,1.8,0.5" >> sample_dataset.csv

# Upload to storage
curl -X POST http://localhost:8081/api/v1/upload/datasets/sample.csv \
  -F "file=@sample_dataset.csv"

# List files
curl http://localhost:8081/api/v1/list/datasets | jq

# Download file
curl http://localhost:8081/api/v1/download/datasets/sample.csv
```

#### Scenario 3: View Monitoring Dashboard
1. Open http://localhost:3001 in browser
2. Login with admin/admin
3. Navigate to "TensorFleet Dashboard"
4. Submit some jobs and watch metrics update

### Environment Variables for Customization

Create a `.env` file to customize settings:
```bash
# Optional: Customize ports
API_GATEWAY_PORT=8080
STORAGE_PORT=8081
MONITORING_PORT=8082
FRONTEND_PORT=3000

# Optional: Customize MinIO credentials
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password123

# Optional: Worker scaling
WORKER_REPLICAS=3
```

### Data Persistence

All data is persisted in Docker volumes:
```bash
# View volumes
docker volume ls | grep tensorfleet

# To reset all data (WARNING: destroys all jobs/files)
docker-compose down -v

# To backup data
docker run --rm -v tensorfleet_minio_data:/data -v $(pwd):/backup ubuntu tar czf /backup/minio_backup.tar.gz /data
```

### Cleanup Instructions

```bash
# Stop all services
docker-compose down

# Remove all data (optional)
docker-compose down -v

# Clean up Docker resources
docker system prune -a

# Remove built images
docker rmi $(docker images | grep tensorfleet | awk '{print $3}')
```

### Testing Reproducibility

To verify the setup works on a clean system:
```bash
# Test script for CI/CD or clean environment
#!/bin/bash
set -e

echo "Testing TensorFleet reproducibility..."
docker-compose up -d
sleep 60  # Wait for services to initialize

# Test health endpoints
curl -f http://localhost:8080/health
curl -f http://localhost:8081/health  
curl -f http://localhost:8082/health

# Test job submission
JOB_ID=$(curl -s -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"model_type":"test","dataset_path":"test","num_workers":1,"epochs":1}' | jq -r .job_id)

# Verify job was created
curl -f "http://localhost:8080/api/v1/jobs/$JOB_ID"

echo "✅ Reproducibility test passed!"
```

## �🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd TensorFleet
```

### 2. Generate gRPC Stubs

```bash
make proto
```

### 3. Run Locally with Docker Compose

```bash
# Build and start all services
make compose-up

# Or manually:
docker-compose up --build
```

### 4. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | - |
| API Gateway | http://localhost:8080 | - |
| Storage API | http://localhost:8081 | - |
| Monitoring API | http://localhost:8082 | - |
| Model Service API | http://localhost:8083 | - |
| ML Worker API | http://localhost:8000 | - |
| Grafana | http://localhost:3001 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| MinIO Console | http://localhost:9001 | minioadmin/minioadmin |
| MongoDB | localhost:27017 | admin/password123 |

## 💻 Local Development

### Start Services

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
make compose-down
```

### Submit a Training Job

```bash
# Using curl
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "X-User-ID: demo-user" \
  -d '{
    "model_type": "cnn",
    "dataset_path": "/data/mnist",
    "num_workers": 3,
    "epochs": 10,
    "hyperparameters": {
      "learning_rate": "0.001",
      "batch_size": "64"
    }
  }'

# Response
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "RUNNING",
  "num_tasks": 100,
  "message": "Job created with 100 tasks"
}
```

### Check Job Status

```bash
curl http://localhost:8080/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```

### View Metrics

```bash
# Dashboard metrics
curl http://localhost:8082/api/v1/dashboard

# Prometheus metrics
curl http://localhost:8082/metrics
```

## 🤖 MongoDB ML Training

TensorFleet now supports machine learning training with MongoDB for dataset storage and model persistence.

### Quick ML Training Demo

```bash
# Run the automated ML training demo
./demo-mongodb-ml.sh
```

This demo will:
1. ✅ Train 3 different ML models (RandomForest, LogisticRegression, SVM)
2. ✅ Store trained models in MongoDB using GridFS
3. ✅ Save model metadata (hyperparameters, metrics, version)
4. ✅ Download and save models locally
5. ✅ Display model statistics

### Manual ML Training

#### 1. List Available Datasets

```bash
curl http://localhost:8000/datasets | jq
```

**Response:**
```json
{
  "datasets": [
    {
      "name": "iris",
      "description": "Iris flower dataset",
      "n_samples": 150,
      "n_features": 4,
      "target_column": "species"
    },
    {
      "name": "wine",
      "description": "Wine classification dataset",
      "n_samples": 178,
      "n_features": 13,
      "target_column": "wine_class"
    }
  ]
}
```

#### 2. Train a Model

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "my_training_job_001",
    "dataset_name": "iris",
    "algorithm": "random_forest",
    "target_column": "species",
    "model_name": "iris_rf_model",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 5,
      "random_state": 42
    }
  }' | jq
```

**Response:**
```json
{
  "job_id": "my_training_job_001",
  "model_id": "507f1f77bcf86cd799439011",
  "status": "completed",
  "metrics": {
    "train_accuracy": 0.9833,
    "test_accuracy": 0.9667,
    "training_time": 0.234
  },
  "model_name": "iris_rf_model",
  "version": "v1701964800"
}
```

#### 3. List Trained Models

```bash
curl "http://localhost:8083/api/v1/models?page=1&limit=10" | jq
```

**Response:**
```json
{
  "models": [
    {
      "id": "507f1f77bcf86cd799439011",
      "name": "iris_rf_model",
      "algorithm": "random_forest",
      "metrics": {
        "test_accuracy": 0.9667,
        "train_accuracy": 0.9833
      },
      "version": "v1701964800",
      "created_at": "2025-12-07T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 1,
    "pages": 1
  }
}
```

#### 4. Get Model Metadata

```bash
curl http://localhost:8083/api/v1/models/<model_id> | jq
```

**Response:**
```json
{
  "id": "507f1f77bcf86cd799439011",
  "name": "iris_rf_model",
  "algorithm": "random_forest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
  },
  "metrics": {
    "train_accuracy": 0.9833,
    "test_accuracy": 0.9667,
    "training_time": 0.234
  },
  "version": "v1701964800",
  "dataset_name": "iris",
  "target_column": "species",
  "features": ["sepal length", "sepal width", "petal length", "petal width"],
  "created_at": "2025-12-07T10:30:00Z"
}
```

#### 5. Download a Model

```bash
# Download model file
curl http://localhost:8083/api/v1/models/<model_id>/download \
  -o my_model.pkl

# Verify download
ls -lh my_model.pkl
```

#### 6. Use Downloaded Model (Python)

```python
import pickle
import numpy as np

# Load the model
with open('my_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample_data)
print(f"Prediction: {prediction}")
```

### Supported ML Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `random_forest` | Random Forest Classifier | Classification, robust to overfitting |
| `logistic_regression` | Logistic Regression | Binary/multi-class classification |
| `svm` | Support Vector Machine | Non-linear classification |
| `decision_tree` | Decision Tree Classifier | Interpretable models |

### Model Hyperparameters

**Random Forest:**
```json
{
  "n_estimators": 100,
  "max_depth": null,
  "min_samples_split": 2,
  "random_state": 42
}
```

**Logistic Regression:**
```json
{
  "max_iter": 1000,
  "C": 1.0,
  "random_state": 42
}
```

**SVM:**
```json
{
  "kernel": "rbf",
  "C": 1.0,
  "gamma": "scale",
  "random_state": 42
}
```

**Decision Tree:**
```json
{
  "max_depth": null,
  "min_samples_split": 2,
  "random_state": 42
}
```

### Python Client for ML Operations

Use the provided Python client for easy interaction:

```bash
# Run complete demo
python ml_client.py demo

# List all models
python ml_client.py list

# Get statistics
python ml_client.py stats

# Download a specific model
python ml_client.py download <model_id> output.pkl
```

### Model Storage Architecture

- **Datasets**: Stored in MongoDB collections or GridFS for large files
- **Trained Models**: Serialized with pickle and stored in GridFS
- **Metadata**: Model information stored in MongoDB collections
  - Model name, algorithm, hyperparameters
  - Training metrics (accuracy, loss)
  - Version information
  - Dataset reference
  - Feature names
  - Training timestamp

### Model Versioning

Each trained model is automatically versioned using a timestamp:
- Format: `v{unix_timestamp}`
- Example: `v1701964800`
- Allows multiple versions of the same model
- Easy rollback to previous versions

## ☸️ Kubernetes Deployment

### 🚀 Production Deployment

Deploy TensorFleet to a Kubernetes cluster:

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy infrastructure
kubectl apply -f k8s/infrastructure.yaml

# Deploy core services
kubectl apply -f k8s/deployment.yaml

# Setup ingress
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n tensorfleet
```

### 📊 Scaling Configuration

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-hpa
  namespace: tensorfleet
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 🔐 Security & Secrets

```bash
# Create TLS certificates
kubectl create secret tls tensorfleet-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n tensorfleet

# Create database credentials
kubectl create secret generic mongodb-secret \
  --from-literal=connection-string="mongodb://admin:password123@mongodb:27017/tensorfleet?authSource=admin" \
  -n tensorfleet

# Create API keys
kubectl create secret generic api-keys \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=admin-key="your-admin-api-key" \
  -n tensorfleet
```

## 📊 Monitoring & Observability

### 🎯 Key Metrics

TensorFleet provides comprehensive observability across all layers:

#### Application Metrics
- **Job Metrics**: Success rate, completion time, queue depth
- **Training Metrics**: Accuracy, loss, convergence rate, resource utilization
- **Worker Metrics**: Task throughput, error rate, resource consumption
- **API Metrics**: Request latency, error rate, throughput by endpoint

#### Infrastructure Metrics
- **Service Health**: Availability, response time, resource usage
- **Database Performance**: Connection pool, query performance, storage usage
- **Storage Metrics**: Object storage usage, transfer rates, availability
- **Network Metrics**: Service-to-service communication, latency, errors

### 📈 Grafana Dashboards

Pre-configured dashboards available at `http://localhost:3001`:

- **TensorFleet Overview**: High-level platform metrics and health
- **Job Analytics**: Training job performance and success rates
- **Worker Performance**: Distributed worker utilization and efficiency
- **Infrastructure Health**: System resources and service availability
- **API Performance**: Gateway metrics and endpoint analytics

### 🔔 Alerting Rules

```yaml
# Critical system alerts
groups:
  - name: tensorfleet-critical
    rules:
    - alert: ServiceDown
      expr: up{job="tensorfleet"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "TensorFleet service {{ $labels.instance }} is down"
    
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High error rate on {{ $labels.service }}"
```

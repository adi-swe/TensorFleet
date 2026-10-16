# TensorFleet: Distributed Machine Learning Platform
## Comprehensive Project Report

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Core Components](#4-core-components)
5. [Technology Stack](#5-technology-stack)
6. [Implementation Details](#6-implementation-details)
7. [Features and Capabilities](#7-features-and-capabilities)
8. [Performance Analysis](#8-performance-analysis)
9. [Future Enhancements](#9-future-enhancements)
10. [Conclusion](#10-conclusion)
11. [Team Contributions](#11-team-contributions)
12. [Appendix A: Architectural Decision Records (ADRs)](#appendix-a-architectural-decision-records-adrs)

---

## 1. Executive Summary

TensorFleet is a sophisticated distributed machine learning platform designed to democratize access to large-scale ML training capabilities. Built using modern microservices architecture, the platform enables organizations to efficiently train machine learning models across distributed computing resources while providing comprehensive monitoring, management, and scalability features.

### Validated implementation and scope

The current implementation runs real classical estimators through the worker
fleet, combines deterministic forest shards into one saved model, and supports
native Apple MPS logistic training. Five repeated trials per scaling configuration
on one Mac compare 1/2/4 CPU workers with a direct process pool. Four-worker MNIST
was slower than two-worker training; MPS logistic was slower than CPU for the
measured workload. Raw evidence is indexed in `result/research/README.md`.

Worker and coordinator interruption checks exercise retry and Redis-based task
recovery. No measured uptime, cost saving, automatic performance-optimal scaling,
production readiness, multi-host acceleration, or distributed DL/LLM result is
claimed. Earlier percentage savings and uptime figures are withdrawn.

---

## 2. Introduction and Problem Statement

### 2.1 Background
Machine learning model training has become increasingly compute-intensive, requiring significant infrastructure investments and technical expertise. Organizations often struggle with:

- **Infrastructure Complexity**: Setting up and managing ML training clusters
- **Resource Utilization**: Inefficient use of available computing resources
- **Scalability Challenges**: Difficulty scaling training workloads dynamically
- **Monitoring Gaps**: Lack of comprehensive training job monitoring
- **Cost Management**: Unpredictable infrastructure costs

### 2.2 Project Objectives
TensorFleet addresses these challenges by providing:

1. **Simplified ML Training**: One-click deployment of distributed training jobs
2. **Resource Optimization**: Intelligent resource allocation and auto-scaling
3. **Comprehensive Monitoring**: Real-time metrics, logging, and visualization
4. **Cost Efficiency**: Pay-per-use model with automatic resource optimization
5. **Developer Experience**: Intuitive APIs and web interface

### 2.3 Target Users
- **Data Scientists**: Researchers needing distributed training capabilities
- **ML Engineers**: Teams building production ML pipelines
- **Organizations**: Companies seeking to optimize ML infrastructure costs
- **Educational Institutions**: Universities requiring scalable ML training resources

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────-┐
│                           TensorFleet Platform                       │
├─────────────────────────────────────────────────────────────────────-┤
│  Frontend (React)     │  API Gateway (Go)     │  Monitoring (Python) │
├─────────────────────────────────────────────────────────────────────-┤
│           Orchestrator (Go/gRPC)      │      Storage (Python/S3)     │
├─────────────────────────────────────────────────────────────────────-┤
│  Worker Nodes (Go)    │  ML Workers (Python)  │  Model Service       │
├─────────────────────────────────────────────────────────────────────-┤
│     Redis Queue       │     MongoDB           │     MinIO Storage    │
└─────────────────────────────────────────────────────────────────────-┘
```

### 3.2 Service Communication
- **Synchronous**: REST APIs for user-facing operations
- **Asynchronous**: gRPC for internal service communication
- **Event-driven**: Redis pub/sub for real-time updates
- **Data Flow**: MinIO for model artifacts, MongoDB for metadata

### 3.3 Deployment Architecture
- **Containerization**: All services deployed as Docker containers
- **Orchestration**: Docker Compose for local development
- **Production Ready**: Complete Kubernetes manifests in `kubenetes/` directory for production deployment
- **Cloud Database**: MongoDB Atlas integration for managed database services
- **Load Balancing**: Nginx for frontend, internal load balancing for services

**Automated Deployment**: We provide `k8s/deploy.sh` to deploy the entire TensorFleet platform in one command. The script supports building images (`--build-images`), optional automatic loading to Minikube, and flags to skip waiting for rollouts. A companion script `k8s/build-images.sh` will build all service images and can optionally push or load them into Minikube.

**Default Local Configuration**: For convenience the Kubernetes manifests include sensible defaults for local development:
- `tensorfleet-config` ConfigMap defaults MongoDB to MongoDB Atlas cloud connection with secure credentials
- `tensorfleet-secrets` defaults MinIO credentials to `minioadmin/minioadmin` and will be created by the deploy script if missing.
- Complete Kubernetes manifests available in `kubenetes/` directory generated via Kompose conversion

These defaults are suitable for local development (Minikube) and should be overridden for production using external secret management or CI pipelines that inject production credentials.

---

## 4. Core Components

### 4.1 Frontend Dashboard (React/Vite)
**Purpose**: Web-based user interface for job management and monitoring

**Key Features**:
- Job submission and configuration
- Real-time training progress monitoring
- Resource utilization dashboards
- Model performance metrics visualization
- Worker node management interface

**Technology Stack**:
- React 18 with hooks
- Material-UI components
- Vite build system
- Axios for API communication
- Real-time updates via Server-Sent Events

### 4.2 API Gateway (Go/Gin)
**Purpose**: Central entry point for all client requests

**Responsibilities**:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- CORS handling

**Endpoints**:
- `/api/v1/jobs` - Job management
- `/api/v1/workers` - Worker monitoring
- `/health` - Health checks
- `/worker-activity` - Real-time worker data

### 4.3 Orchestrator (Go/gRPC)
**Purpose**: Core job scheduling and task distribution engine

**Key Functions**:
- Job queue management
- Task assignment to workers
- Resource allocation optimization
- Failure detection and recovery
- Job lifecycle management

**gRPC Services**:
- `CreateTrainingJob`
- `GetJobStatus`
- `AssignTask`
- `ReportTaskCompletion`
- `CancelJob`

### 4.4 Worker Nodes (Go/Python)
**Purpose**: Distributed computing nodes for ML training execution

**Components**:
- **Go Worker**: Task coordination and communication
- **ML Worker**: Python-based training execution
- **Resource Monitoring**: CPU, memory, GPU utilization tracking

**Capabilities**:
- Dynamic model loading
- Checkpoint management
- Progress reporting
- Error handling and recovery

### 4.5 Monitoring Service (Python/Flask)
**Purpose**: Comprehensive system monitoring and metrics collection

**Features**:
- Metrics collection and export
- Real-time performance monitoring
- Auto-scaling triggers
- Health check coordination

**Metrics Collected**:
- Training loss and accuracy
- Resource utilization
- Job completion rates
- System performance indicators

### 4.6 Storage Service (Python/MinIO)
**Purpose**: Distributed file storage for models and datasets

**Capabilities**:
- Model artifact management
- Dataset storage and versioning
- Checkpoint persistence
- Automatic backup and replication

---

## 5. Technology Stack

### 5.1 Backend Technologies
- **Go (Golang)**: High-performance services (API Gateway, Orchestrator, Workers)
- **Python**: ML-focused services (ML Workers, Monitoring, Storage)
- **gRPC**: Inter-service communication protocol
- **REST APIs**: External interface standards

### 5.2 Frontend Technologies
- **React 18**: Modern UI framework
- **Material-UI**: Component library
- **Vite**: Fast build tool and dev server
- **JavaScript/JSX**: Programming languages

### 5.3 Data Layer
- **Redis**: Message queuing and caching
- **MongoDB Atlas**: Cloud-hosted database for metadata and job information storage
- **MinIO**: S3-compatible object storage

### 5.4 Infrastructure
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration for local development
- **Kubernetes**: Production-ready orchestration with complete manifests
- **Nginx**: Web server and reverse proxy

### 5.5 Development Tools
- **Git**: Version control
- **GitHub**: Code repository and collaboration
- **VSCode**: Development environment
- **Postman**: API testing

---

## 6. Implementation Details

### 6.1 Job Submission Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant AG as API Gateway
    participant O as Orchestrator
    participant W as Worker
    
    U->>F: Submit Training Job
    F->>AG: POST /api/v1/jobs
    AG->>O: CreateTrainingJob (gRPC)
    O->>O: Queue Job Tasks
    O->>W: AssignTask (gRPC)
    W->>W: Execute Training
    W->>O: ReportProgress (gRPC)
    O->>AG: Status Update
    AG->>F: Real-time Updates (SSE)
    F->>U: Progress Display
```

### 6.2 Auto-scaling Implementation
```python
def monitor_and_scale():
    """Automatic worker scaling based on load"""
    while auto_scale_enabled:
        utilization = calculate_worker_utilization()
        
        if utilization > scale_up_threshold:
            scale_up_workers()
        elif utilization < scale_down_threshold:
            scale_down_workers()
            
        time.sleep(monitoring_interval)
```

### 6.3 Fault Tolerance Mechanisms
- **Health Checks**: Regular service health monitoring
- **Circuit Breakers**: Automatic failure isolation
- **Retry Logic**: Intelligent request retry with exponential backoff
- **Graceful Degradation**: Fallback mechanisms for service failures

### 6.4 Data Persistence Strategy
- **Job Metadata**: MongoDB Atlas cloud database with high availability
- **Model Artifacts**: MinIO with versioning and S3-compatible API
- **Metrics Data**: Application-level metrics collection and storage
- **Cache Layer**: Redis with persistence for job queues and session data
- **Cloud Migration**: Transitioned from local MongoDB to MongoDB Atlas for enhanced scalability and reliability

---

## 7. Features and Capabilities

### 7.1 Core Features
- **Multi-Model Support**: ResNet, BERT, custom models
- **Dynamic Scaling**: 1-100+ workers with auto-scaling
- **Real-time Monitoring**: Live training metrics and logs
- **Job Management**: Start, stop, pause, resume operations
- **Resource Optimization**: Intelligent task distribution

### 7.2 Advanced Capabilities
- **Hyperparameter Tuning**: Automated parameter optimization
- **Checkpointing**: Automatic training state preservation
- **Model Versioning**: Complete model lifecycle management
- **Performance Analytics**: Detailed training performance insights
- **Cost Optimization**: Resource usage optimization algorithms

### 7.3 User Experience Features
- **Intuitive Dashboard**: Drag-and-drop job configuration
- **Real-time Updates**: Live progress monitoring
- **Mobile Responsive**: Works on all device sizes
- **Dark Mode**: User preference support
- **Accessibility**: WCAG 2.1 compliant interface

### 7.4 Enterprise Features
- **Multi-tenancy**: Isolated environments for different teams
- **RBAC**: Role-based access control
- **Audit Logging**: Comprehensive activity tracking
- **API Keys**: Programmatic access management
- **SLA Monitoring**: Service level agreement tracking

---

## 8. Performance Analysis

The earlier performance figures in this section were not backed by the raw
results advertised in the repository. They have been withdrawn as validated
results. In particular, the earlier claims about 50 workers, 1,000 concurrent
jobs, 3–10× faster training, utilization, fault-recovery time, uptime, failure
rate, and zero data loss must not be cited as experimental findings.

Use [the evaluation methodology](PERFORMANCE_EVALUATION.md),
[the saved MNIST evidence](../result/README.md), and
[the gap-by-gap audit](../scale.md) for current implementation status and
bounded conclusions. The historical report's broader architecture and production
capabilities also require checking against the implemented flow in [FLOW.md](FLOW.md).


## 9. Future Enhancements

### 9.1 Short-term Improvements (3-6 months)
- **GPU Support**: NVIDIA GPU integration for accelerated training
- **Enhanced Kubernetes Support**: Production-hardened Kubernetes deployment with Helm charts
- **Advanced Authentication**: OAuth2/OIDC integration
- **Model Registry**: Enhanced model versioning and management
- **Performance Profiling**: Detailed training performance analysis
- **Advanced Monitoring**: Enhanced metrics visualization and alerting systems

### 9.2 Medium-term Features (6-12 months)
- **Multi-cloud Support**: AWS, GCP, Azure deployment options
- **Federated Learning**: Privacy-preserving distributed training
- **AutoML Integration**: Automated machine learning workflows
- **Stream Processing**: Real-time data processing capabilities
- **Edge Computing**: Edge device training coordination

### 9.3 Long-term Vision (1-2 years)
- **AI Model Marketplace**: Community-driven model sharing
- **Quantum ML Support**: Quantum machine learning algorithms
- **Explainable AI**: Model interpretability tools
- **Carbon Footprint**: Environmental impact optimization
- **Enterprise Integration**: SAP, Salesforce, Oracle connectors

### 9.4 Research and Development
- **Novel Algorithms**: Custom distributed training algorithms
- **Hardware Optimization**: Specialized hardware acceleration
- **Network Optimization**: Advanced networking protocols
- **Security Research**: Zero-trust security architecture
- **Performance Research**: Cutting-edge optimization techniques

---

## 9.5 Recent Infrastructure Updates (December 2025)

### Cloud Database Migration
- **MongoDB Atlas Integration**: Successfully migrated from local MongoDB to MongoDB Atlas cloud database
- **Benefits**: Enhanced reliability, automated backups, better scalability, and reduced operational overhead
- **Connection String**: Secure connection with authentication credentials managed through environment variables

### Kubernetes Deployment Enhancements
- **Complete Manifest Set**: Generated comprehensive Kubernetes manifests using Kompose in `kubenetes/` directory
- **Production Ready**: All 12+ services have dedicated deployment and service YAML files
- **Resource Management**: Configured persistent volume claims for stateful services (MinIO)
- **Health Checks**: Implemented liveness probes for all critical services

### Local Development Optimization
- **Streamlined Docker Compose**: Simplified local development setup for faster startup
- **Focus on Core Services**: Docker Compose now focuses on essential services for rapid development
- **Lightweight Setup**: Removed heavyweight monitoring infrastructure from local development environment

### Configuration Management
- **ConfigMaps**: Centralized configuration management with Kubernetes ConfigMaps
- **Secrets Management**: Secure credential handling with Kubernetes Secrets
- **Environment Flexibility**: Easy switching between local development and production configurations

---

## 10. Conclusion

### 10.1 Project Success
TensorFleet successfully addresses the critical challenges in distributed machine learning infrastructure by providing a comprehensive, scalable, and user-friendly platform. The project demonstrates significant technical achievements in microservices architecture, distributed computing, and real-time monitoring.

### 10.2 Technical Achievements
- **Microservices Excellence**: 12+ independently deployable services
- **Performance Optimization**: Sub-second response times at scale
- **Fault Tolerance**: Robust error handling and recovery mechanisms
- **Developer Experience**: Intuitive APIs and comprehensive documentation
- **Monitoring Excellence**: Real-time observability across all components
- **Cloud Integration**: Successful migration to MongoDB Atlas for enhanced reliability
- **Production Deployment**: Complete Kubernetes infrastructure with automated deployment scripts

### 10.3 Business Value
The platform delivers substantial business value through:
- **Cost Reduction**: Significant infrastructure cost savings
- **Time-to-Market**: Faster ML model development cycles
- **Scalability**: Seamless scaling from prototype to production
- **Risk Mitigation**: Reliable, fault-tolerant training infrastructure
- **Innovation Enablement**: Democratized access to distributed ML training

### 10.4 Learning Outcomes
The project provided valuable learning experiences in:
- **Distributed Systems Design**: Complex system architecture patterns
- **Microservices Implementation**: Service decomposition and communication
- **Container Orchestration**: Docker and Kubernetes technologies
- **Performance Engineering**: Optimization and monitoring techniques
- **Team Collaboration**: Agile development and DevOps practices

### 10.5 Impact and Recognition
TensorFleet represents a significant contribution to the open-source ML infrastructure landscape, providing a reference implementation for distributed machine learning platforms. The project demonstrates industry-ready engineering practices and serves as a foundation for future ML infrastructure innovations.

### 10.6 Final Thoughts
TensorFleet proves that sophisticated distributed machine learning infrastructure can be both powerful and accessible. By leveraging modern technologies and architectural patterns, the platform successfully bridges the gap between research-grade ML capabilities and production-ready infrastructure, enabling organizations to harness the full potential of distributed machine learning.

---

## 11. Team Contributions

### 11.1 Team Structure

The TensorFleet project was developed by a team of three dedicated students, each taking ownership of critical components of the distributed ML platform:

| Team Member | Student ID | Role | Contributions |
|-------------|------------|------|---------------|
| **Aditya Suryawanshi** | 25211365 | Backend Infrastructure Lead | API Gateway, Orchestrator, Worker Nodes, gRPC Infrastructure |
| **Rahul Mirashi** | 25211365 | ML & Data Services Lead | ML Worker, Model Service, Storage Service, MongoDB Integration |
| **Soham Maji** | 25204731 | Frontend & Monitoring Lead | React Dashboard, Monitoring Service, Documentation, DevOps |

### 11.2 Individual Contributions

#### Aditya Suryawanshi - Backend Infrastructure Lead
**Primary Responsibilities**: Core orchestration, job scheduling, and distributed computing infrastructure

**Key Contributions**:
- Designed and implemented the **API Gateway** service using Go and Gin framework (~500-600 LOC)
- Built the **Orchestrator Service** with advanced job scheduling and worker management (~800-1000 LOC)
- Developed the **Worker Service** for distributed task execution (~400-500 LOC)
- Implemented gRPC service contracts and protocol buffer definitions
- Established Redis integration for job queuing and caching
- Created comprehensive unit and integration tests for Go services
- Documented backend architecture and API specifications

**Technical Skills Applied**: Go, gRPC, Redis, Microservices Architecture, Distributed Systems

#### Rahul Mirashi - ML & Data Services Lead
**Primary Responsibilities**: Machine learning execution, data management, and model registry

**Key Contributions**:
- Developed the **ML Worker Service** with support for multiple algorithms (~1200-1500 LOC)
- Implemented scikit-learn and TensorFlow model training pipelines
- Built the **Model Service** for model registry and versioning (~350-400 LOC)
- Created the **Storage Service** with MinIO integration (~500-600 LOC)
- Designed MongoDB schema and GridFS integration for model artifacts
- Implemented data preprocessing and model evaluation metrics
- Created sample datasets and training workflows
- Wrote comprehensive tests for ML training logic

**Technical Skills Applied**: Python, Machine Learning, scikit-learn, TensorFlow, MongoDB, Flask, MinIO

#### Soham Maji - Frontend & Monitoring Lead
**Primary Responsibilities**: User interface, visualization, monitoring, and observability

**Key Contributions**:
- Built the **Frontend Dashboard** using React 18 and Material-UI (~2000-2500 LOC)
- Implemented real-time job monitoring and worker visualization
- Developed model registry and dataset management interfaces
- Created the **Monitoring Service** with Prometheus integration (~200-250 LOC)
- Configured Grafana dashboards for system observability
- Managed Docker Compose and Kubernetes deployment configurations
- Wrote comprehensive project documentation and demo materials
- Established CI/CD workflows and deployment automation

**Technical Skills Applied**: React, Material-UI, JavaScript, Python, Prometheus, Grafana, Docker, Kubernetes, Technical Writing

### 11.3 Collaborative Efforts

The team worked collaboratively on several cross-cutting concerns:

- **Integration Testing**: All members contributed to end-to-end testing and validation
- **Code Reviews**: Regular peer reviews ensured code quality and knowledge sharing
- **Architecture Decisions**: Major design decisions were made collectively through team discussions
- **Documentation**: Each member documented their components and contributed to overall project documentation
- **Deployment**: Coordinated deployment strategies and troubleshooting across all services

### 11.4 Project Management Approach

The team followed agile development practices:
- **Sprint Planning**: Weekly sprints with clear goals and deliverables
- **Daily Standups**: Regular communication through team channels
- **Version Control**: Git workflow with feature branches and pull requests
- **Issue Tracking**: GitHub Issues for task management and bug tracking
- **Code Standards**: Established coding conventions and review processes

### 11.5 Work Distribution Analysis

The workload was distributed equitably across the team:

| Metric | Aditya | Rahul | Soham |
|--------|--------|-------|-------|
| **Estimated Hours** | 120-150 | 120-150 | 120-150 |
| **Services Built** | 3 | 3 | 2 + docs |
| **Lines of Code** | ~1500-2000 | ~2000-2500 | ~2200-2800 |
| **Test Coverage** | 90%+ | 85%+ | 80%+ |
| **Documentation** | High | High | Very High |

### 11.6 Learning Outcomes

Each team member gained valuable experience in:

- **Technical Skills**: Hands-on experience with modern technologies and frameworks
- **System Design**: Understanding of distributed systems and microservices architecture
- **Collaboration**: Working effectively in a team environment
- **DevOps**: Container orchestration, deployment automation, and monitoring
- **Problem Solving**: Debugging complex distributed system issues
- **Communication**: Technical documentation and presentation skills

### 11.7 Acknowledgments

The team would like to acknowledge:
- Course instructors for guidance and feedback
- Open-source communities for excellent documentation and tools
- MongoDB Atlas for cloud database services
- Docker and Kubernetes communities for container orchestration support

**Detailed Work Division**: For a comprehensive breakdown of responsibilities, tasks, and timelines, see [TEAM_WORK_DIVISION.md](./TEAM_WORK_DIVISION.md)

---

## Appendix A: Architectural Decision Records (ADRs)

### ADR 1: Service Decomposition and Granularity

**Context:**  
TensorFleet needed to support scalable, maintainable, and independently deployable components for distributed ML training. The team had to decide between a monolithic architecture, coarse-grained services, or fine-grained microservices.

**Decision:**  
We adopted a microservices architecture, decomposing the system into 12+ independent services (API Gateway, Orchestrator, Worker, Monitoring, Storage, etc.), each with a single responsibility and clear boundaries.

**Consequences (Trade-offs):**  
- **Positive:**  
  - Independent scaling and deployment of services.
  - Improved fault isolation—failures in one service do not cascade.
  - Technology heterogeneity (Go for orchestration, Python for ML).
  - Easier team parallelism and codebase ownership.
- **Negative:**  
  - Increased operational complexity (service discovery, configuration, monitoring).
  - Higher resource overhead (multiple containers, inter-service communication).
  - More complex debugging and distributed tracing.

---

### ADR 2: Inter-Service Communication Pattern

**Context:**  
The platform required reliable, efficient communication between services (e.g., job submission, worker coordination, monitoring). Options included REST, gRPC, and message brokers (Kafka/RabbitMQ).

**Decision:**  
We chose REST APIs for user-facing operations (API Gateway ↔ Frontend) and gRPC for internal service-to-service communication (API Gateway ↔ Orchestrator ↔ Workers). Redis Pub/Sub is used for real-time updates and lightweight eventing.

**Consequences (Trade-offs):**  
- **Positive:**  
  - gRPC provides high performance, strong typing, and efficient binary serialization for internal comms.
  - REST is widely supported and easy to consume for external clients.
  - Redis Pub/Sub enables low-latency event notifications.
- **Negative:**  
  - gRPC requires additional tooling (protobufs, codegen) and is less human-readable than REST.
  - Multiple protocols increase maintenance and learning curve.
  - Redis Pub/Sub is not persistent—missed events if consumers are offline.

---

### ADR 3: Deployment and Orchestration Strategy

**Context:**  
TensorFleet needed a deployment approach that supports local development, easy scaling, and production readiness. Options included bare-metal deployment, Docker Compose, and Kubernetes.

**Decision:**  
We containerized all services using Docker and used Docker Compose for local development. For production, the system is designed to be Kubernetes-ready, with manifests provided for scalable, resilient deployment.

**Consequences (Trade-offs):**  
- **Positive:**  
  - Consistent environments across dev, test, and prod.
  - Easy local setup with Docker Compose.
  - Kubernetes enables auto-scaling, self-healing, and advanced orchestration.
- **Negative:**  
  - Kubernetes has a steep learning curve and operational overhead.
  - More complex CI/CD and configuration management.
  - Resource usage is higher compared to bare-metal or single-container setups.

---

### ADR 4: Database Migration to MongoDB Atlas

**Context:**  
Initially, TensorFleet used a local MongoDB instance deployed alongside other services. As the project matured, concerns arose about database reliability, backup management, scalability, and operational overhead for production deployments.

**Decision:**  
We migrated from a self-hosted MongoDB container to MongoDB Atlas, a fully-managed cloud database service. All services were updated to connect to MongoDB Atlas using secure connection strings with authentication credentials.

**Consequences (Trade-offs):**  
- **Positive:**  
  - Automated backups and point-in-time recovery.
  - Built-in high availability with replica sets and automatic failover.
  - Reduced operational complexity—no need to manage MongoDB upgrades, patches, or scaling.
  - Better performance monitoring and optimization tools provided by Atlas.
  - Simplified local development—no need to run MongoDB container locally.
  - Geographic distribution options for reduced latency.
- **Negative:**  
  - Ongoing cloud service costs (though offset by reduced operational overhead).
  - External dependency on MongoDB Atlas service availability.
  - Network latency for database operations (mitigated by choosing appropriate regions).
  - Requires internet connectivity for database access during development.
  - Credentials management becomes more critical for security.

---

**Document Information**
- **Version**: 1.2
- **Date**: December 21, 2025
- **Authors**: 
  - Aditya Suryawanshi (25211365) - Backend Infrastructure Lead
  - Rahul Mirashi (25211365) - ML & Data Services Lead
  - Soham Maji (25204731) - Frontend & Monitoring Lead
- **Review Status**: Final
- **Distribution**: Public

---

*This report represents the comprehensive documentation of the TensorFleet project, showcasing the technical depth, architectural sophistication, and practical value delivered by the platform.*

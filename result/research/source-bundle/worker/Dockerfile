# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Install protoc and Go plugins
RUN apk add --no-cache protobuf-dev git
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.31.0
RUN go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.3.0

# Copy proto files first
COPY proto/ ./proto/

# Generate proto files for worker
RUN mkdir -p worker/proto/worker worker/proto/orchestrator && \
    protoc --proto_path=proto \
    --go_out=worker/proto/worker --go_opt=paths=source_relative \
    --go-grpc_out=worker/proto/worker --go-grpc_opt=paths=source_relative \
    proto/worker.proto && \
    protoc --proto_path=proto \
    --go_out=worker/proto/orchestrator --go_opt=paths=source_relative \
    --go-grpc_out=worker/proto/orchestrator --go-grpc_opt=paths=source_relative \
    proto/orchestrator.proto

# Copy go mod files
COPY worker/go.mod worker/go.sum* ./worker/
WORKDIR /app/worker
RUN go mod download || true

# Copy source code
COPY worker/ ./

RUN go mod tidy || true
# Build only the main package to avoid conflict with generated proto packages
RUN CGO_ENABLED=0 GOOS=linux go build -o worker .

# Runtime stage
FROM python:3.11-slim

COPY worker/fleet-requirements.txt /tmp/fleet-requirements.txt
RUN pip install --no-cache-dir -r /tmp/fleet-requirements.txt

WORKDIR /root/

COPY --from=builder /app/worker/worker .
COPY worker/fleet_trainer.py .
COPY worker/accelerated_logistic.py .

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EXPOSE 50052 2112

CMD ["./worker"]

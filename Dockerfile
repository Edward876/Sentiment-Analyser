# Multi-stage Dockerfile for sentiment-simple-ml
# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /build

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Ensure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Expose HTTP port
EXPOSE 8000

# Default: Start HTTP server
CMD ["python", "-m", "src.service"]

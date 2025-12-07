# Use NVIDIA CUDA base image with Python
# Try these in order if one doesn't work:
#FROM nvidia/cuda:12.1-devel-ubuntu22.04
# FROM nvidia/cuda:12.2-devel-ubuntu22.04
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/model
ENV HOST=0.0.0.0
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies using uv
RUN uv pip install --system -e .

# Copy application code
COPY app.py ./
COPY *.py ./

# Create model directory
RUN mkdir -p /app/model

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

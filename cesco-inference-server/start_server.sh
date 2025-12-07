#!/bin/bash

# Production startup script for CESCO Inference Server
# Usage: ./start_server.sh [port] [log_level]

# Set default values
DEFAULT_PORT=8000
DEFAULT_LOG_LEVEL=info
DEFAULT_HOST=0.0.0.0

# Get parameters
PORT=${1:-$DEFAULT_PORT}
LOG_LEVEL=${2:-$DEFAULT_LOG_LEVEL}
HOST=${3:-$DEFAULT_HOST}

# Check if MODEL_PATH is set
if [ -z "$MODEL_PATH" ]; then
    echo "⚠️  MODEL_PATH not set. Using default: ./outputs_final/best_model"
    export MODEL_PATH="./outputs_final/best_model"
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model directory not found: $MODEL_PATH"
    echo "Please ensure the model is available or set the correct MODEL_PATH"
    exit 1
fi

echo "🚀 Starting CESCO Inference Server..."
echo "   Model Path: $MODEL_PATH"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: $LOG_LEVEL"
echo ""

# Start the server with uv
exec uv run uvicorn app:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    --loop uvloop \
    --http httptools

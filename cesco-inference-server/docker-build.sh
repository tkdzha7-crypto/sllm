#!/bin/bash

# Docker build and run script for CESCO Inference Server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cesco-inference"
CONTAINER_NAME="cesco-api"
PORT=${1:-8000}
MODEL_PATH=${2:-"./outputs_final"}
USE_CPU=${3:-false}

# Choose Dockerfile
if [ "$USE_CPU" = "true" ]; then
    DOCKERFILE="Dockerfile.cpu"
    echo -e "${YELLOW}Using CPU-only build${NC}"
else
    DOCKERFILE="Dockerfile"
    echo -e "${YELLOW}Using GPU-enabled build${NC}"
fi

echo -e "${GREEN}🐋 Building CESCO Inference Docker Image${NC}"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model directory not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please ensure your model is in the correct directory${NC}"
    exit 1
fi

# Build the image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t $IMAGE_NAME .

# Stop existing container if running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop $CONTAINER_NAME
fi

# Remove existing container if exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm $CONTAINER_NAME
fi

# Run the container
echo -e "${GREEN}🚀 Starting CESCO Inference Server on port $PORT${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    -p $PORT:8000 \
    -v "$(pwd)/$MODEL_PATH":/app/model:ro \
    -v "$(pwd)/logs":/app/logs \
    -e MODEL_PATH=/app/model \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    --restart unless-stopped \
    $IMAGE_NAME

# Wait a moment for startup
sleep 5

# Check if container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo -e "${GREEN}✅ Container started successfully!${NC}"
    echo -e "${GREEN}📋 API Documentation: http://localhost:$PORT/docs${NC}"
    echo -e "${GREEN}🔍 Health Check: http://localhost:$PORT/health${NC}"
    echo ""
    echo -e "${YELLOW}Useful commands:${NC}"
    echo "  View logs: docker logs -f $CONTAINER_NAME"
    echo "  Stop server: docker stop $CONTAINER_NAME"
    echo "  Shell access: docker exec -it $CONTAINER_NAME bash"
else
    echo -e "${RED}❌ Failed to start container${NC}"
    echo "Check logs with: docker logs $CONTAINER_NAME"
    exit 1
fi

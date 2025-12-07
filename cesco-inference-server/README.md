# CESCO Inference Server

A FastAPI-based inference server for the CESCO customer complaint analysis model.

## Features

- **FastAPI Web Server**: Modern, fast web framework with automatic API documentation
- **Model Loading**: Loads fine-tuned CESCO model with LoRA adapters
- **JSON Response Parsing**: Automatic parsing and validation of model outputs
- **Batch Processing**: Support for single and batch inference requests
- **Health Checks**: Built-in health check endpoints
- **Auto Documentation**: Swagger UI and ReDoc available at `/docs` and `/redoc`

## Installation

1. Install dependencies:
```bash
pip install -e .
```

2. Set the model path (optional, defaults to `./outputs_final/best_model`):
```bash
export MODEL_PATH=/path/to/your/model
```

## Usage

### Start the Server

#### Local Development
```bash
python app.py
```

#### Production with uv (Recommended for remote servers)
```bash
# Install dependencies
uv sync

# Run the server (accessible from external connections)
uv run uvicorn app:app --host 0.0.0.0 --port 8000

# Or with custom configuration
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --log-level info --access-log

# Using the startup script
./start_server.sh 8000 info
```

**⚠️ Important for Remote Access**: Use `--host 0.0.0.0` to make the server accessible from your laptop. See [REMOTE_ACCESS.md](REMOTE_ACCESS.md) for detailed instructions.

#### Alternative with uvicorn directly
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info --access-log
```

#### Environment Configuration
Copy `.env.example` to `.env` and modify as needed:
```bash
cp .env.example .env
```

### API Endpoints

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "바퀴벌레가 나와서 해지하고 싶습니다.",
    "max_new_tokens": 512,
    "temperature": 0.1
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "input_text": "바퀴벌레가 나와서 해지하고 싶습니다.",
      "max_new_tokens": 512,
      "temperature": 0.1
    },
    {
      "input_text": "향이 너무 강해서 줄여주세요.",
      "max_new_tokens": 512,
      "temperature": 0.1
    }
  ]'
```

#### Health Check
```bash
curl "http://localhost:8000/health"
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs (or http://YOUR_SERVER_IP:8000/docs)
- **ReDoc**: http://localhost:8000/redoc (or http://YOUR_SERVER_IP:8000/redoc)
- **OpenAPI JSON**: http://localhost:8000/openapi.json

For remote server access, see [REMOTE_ACCESS.md](REMOTE_ACCESS.md)

## Request/Response Format

### Request
```json
{
  "input_text": "고객 민원 텍스트",
  "max_new_tokens": 512,
  "temperature": 0.1,
  "top_p": 0.9
}
```

### Response
```json
{
  "success": true,
  "raw_response": "모델의 원시 출력",
  "parsed_response": {
    "is_claim": "claim",
    "summary": "민원 내용 요약",
    "bug_type": "바퀴",
    "keywords": ["바퀴벌레", "해지"],
    "categories": [
      {
        "대분류": "해충 문제",
        "중분류": "방문 요청",
        "소분류": "바퀴",
        "근거": "바퀴벌레 출현으로 인한 해지 요청"
      }
    ]
  },
  "error": null
}
```

## Environment Variables

- `MODEL_PATH`: Path to the model directory (default: `./outputs_final/best_model`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)

## Model Requirements

The server expects a model directory with:
- LoRA adapter files (`adapter_config.json`, `adapter_model.bin` or `adapter_model.safetensors`)
- OR a merged model in `merged_model/` subdirectory
- Compatible with Unsloth FastLanguageModel

## Development

The FastAPI application includes:
- Automatic model loading on startup
- Proper error handling and logging
- Pydantic models for request/response validation
- Async support for better performance
- Graceful shutdown handling

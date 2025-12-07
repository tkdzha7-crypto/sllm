"""
FastAPI application for CESCO model inference.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Try to import vLLM for batch inference
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
    print("✅ vLLM imported successfully - batch inference enabled")
except ImportError:
    VLLM_AVAILABLE = False
    print("ℹ️ vLLM not available - using Unsloth for inference")

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth imported successfully")
except ImportError as e:
    UNSLOTH_AVAILABLE = False
    print(f"❌ Failed to import unsloth: {e}")
    if not VLLM_AVAILABLE:
        raise ImportError("Neither vLLM nor Unsloth is available. Please install one of them.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variable to store the model
model_instance = None


class InferenceBatchRequest(BaseModel):
    """Request model for batch inference."""

    input_texts: list[str] = Field(..., description="List of texts to analyze")
    input_categories: str | None = Field(None, description="The category dictionary as a string")
    max_new_tokens: int = Field(default=2056)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class InferenceRequest(BaseModel):
    """Request model for inference."""

    input_text: str = Field(..., description="The text to analyze")
    input_categories: str | None = Field(None, description="The category dictionary as a string")
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    """Request model for chat."""

    instruction: str = Field(..., description="The instruction for the chat")
    input_text: str = Field(..., description="The text to chat about")
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    """Response model for inference."""

    success: bool
    raw_response: str
    parsed_response: dict[str, Any] | None = None
    error: str | None = None
    confidence_score: float = 0.0


class CESCOInference:
    """CESCO model inference class."""

    def __init__(self, model_path: str, max_seq_length: int = 8096, use_vllm: bool = True):
        """Initialize the inference model."""
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.use_vllm = use_vllm and VLLM_AVAILABLE

        # Load model and tokenizer
        self._load_model()

        # Alpaca prompt template
        self.alpaca_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
 {}

### Input:
 {}

### Response:
"""

        # Default instruction
        self.default_instruction = """
    You are a helpful assistant that analyzes customer interactions for CESCO. Based on the customer's interaction summary, please respond in JSON format including the following information:
    1. is_claim: 클레임 여부 (claim 또는 non-claim) # it is a claim if the customer is expressing dissatisfaction; otherwise, it is a non-claim
    2. summary: 민원 내용 요약 # a brief summary of the customer interaction
    3. bug_type: 해충 종류 (해충 관련인 경우만, 없으면 null)
    4. keywords: 주요 키워드 리스트
    5. categories: 대분류, 중분류, 소분류 조합 리스트 (복수 선택 가능, 최대 5개)

    The response should be in the following JSON format:
    {
        "is_claim": "claim/non-claim",
        "summary": "민원 내용 요약문",
        "bug_type": "바퀴벌레/쥐/기타", # null only if not pest-related
        "keywords": ["주방", "바퀴", "악취"],
        "categories": [
            {
                "대분류": "해충방제",
                "중분류": "바퀴방제",
                "소분류": "바퀴벌레",
                "근거": "고객이 주방에서 바퀴벌레가 나왔다고 주장함"
            }
        ]
    }

    ** HARD REQUIREMENTS **
    - CAUTION: You can only choose categories from the provided [Category Dict].
    - NEVER respond with categories that are not in the dictionary. You must strictly adhere to the provided categories.
    - The category dictionary is in the following format:
   {{대분류: {{중분류: [소분류 리스트]}},
   대분류: {{중분류: [소분류 리스트]}},
   대분류: {{중분류: [소분류 리스트]}}}}
   - Follow the hierarchy strictly and do not invent categories.
   - Ensure the JSON is properly formatted.
   - Make sure the response is in Korean, and the JSON output is properly structured.
    """
        self.default_category_dict = """
{{
  "배송문제": {{"기타": ["기타"], "반품/교환 요청": ["반품/교환 요청"]}},
  "서비스 품질": {{"작업 품질": ["서비스 품질 미흡", "기타"], "담당자 불만": ["고객 응대 미흡", "기타"]}},
  "요금/계약 문제": {{"계약 조건": ["계약 문의/변경 요청", "기타"], "비용 산정/변동": ["청구 금액 불만", "기타"]}},
  "제품": {{"에어제닉": ["향 강함", "분사안됨", "기타"], "정수기": ["누수", "소음", "기타"]}},
  "해약/환불": {{"해약 요청": ["고객 요청 해약", "기타"], "환불 요청": ["서비스 미제공 환불 요청", "기타"]}},
  "해충 문제": {{"방문 요청": ["바퀴", "쥐", "개미", "기타"], "방제 효과 미흡/불만": ["바퀴", "쥐", "기타"]}}
}}
"""

    def input_text_cleaning(self, text: str) -> str:
        """Clean input text by keeping only alphabets, Korean characters, and numbers."""
        import re

        # Keep only alphabets (a-z, A-Z), Korean characters (가-힣, ㄱ-ㅎ, ㅏ-ㅣ), and numbers (0-9)
        # Also keep spaces to maintain word separation
        cleaned_text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", text)
        # Clean up multiple spaces and strip
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        # Check if path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Check for LoRA adapter files
        lora_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
        has_lora = any(os.path.exists(os.path.join(self.model_path, f)) for f in lora_files)

        if not has_lora:
            # Check for merged model
            merged_path = os.path.join(self.model_path, "merged_model")
            if os.path.exists(merged_path):
                self.model_path = merged_path
            else:
                logger.warning("No LoRA adapters or merged model found")

        if self.use_vllm:
            self._load_vllm_model()
        else:
            self._load_unsloth_model()

    def _load_vllm_model(self):
        """Load model using vLLM for efficient batch inference."""
        from vllm import LLM, SamplingParams  # noqa: F401

        logger.info("Loading model with vLLM...")
        try:
            self.llm = LLM(
                model=self.model_path,
                max_model_len=self.max_seq_length,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
            )
            # Get tokenizer from vLLM
            self.tokenizer = self.llm.get_tokenizer()
            logger.info("Model loaded successfully with vLLM")
        except Exception as e:
            logger.error(f"Failed to load model with vLLM: {e}")
            raise

    def _load_unsloth_model(self):
        """Load model using Unsloth."""
        try:
            # Load with Unsloth
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            # Set model to inference mode
            FastLanguageModel.for_inference(self.model)
            logger.info("Model loaded successfully with Unsloth")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded successfully! Vocab size: {len(self.tokenizer)}")

    def generate_response(
        self,
        input_text: str,
        input_categories: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response for input text."""

        # Clean input text
        cleaned_text = self.input_text_cleaning(input_text)

        # Format input with category dict
        formatted_input = f"""[Voice-of-Customer]: {cleaned_text}

[Category Dict]
{input_categories.strip() if input_categories else self.default_category_dict}
"""

        # Format prompt
        prompt = self.alpaca_prompt_template.format(self.default_instruction, formatted_input)

        if self.use_vllm:
            return self._generate_vllm(prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_unsloth(prompt, max_new_tokens, temperature, top_p)

    def _generate_vllm(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> tuple[str, float]:
        """Generate response using vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        # vLLM returns only the generated part, so no need to split
        # But let's check for safety
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text

        # Calculate confidence from cumulative log prob if available
        cumulative_logprob = outputs[0].outputs[0].cumulative_logprob
        if cumulative_logprob is not None:
            # Normalize by number of tokens
            num_tokens = len(outputs[0].outputs[0].token_ids)
            avg_logprob = cumulative_logprob / max(num_tokens, 1)
            confidence_score = min(1.0, max(0.0, (avg_logprob + 5) / 5))  # Normalize to 0-1
        else:
            confidence_score = 0.0

        return response, confidence_score

    def _generate_unsloth(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> tuple[str, float]:
        """Generate response using Unsloth."""
        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length - max_new_tokens
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        probabilities = torch.exp(transition_scores)
        confidence_score = probabilities.mean().item()

        # Decode and extract response
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Extract only the response part
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text[len(prompt) :].strip()

        return response, confidence_score

    def generate_free_response(
        self, instruction: str, input_text: str, max_new_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.9
    ):
        # Clean input text
        cleaned_text = self.input_text_cleaning(input_text)
        prompt = self.alpaca_prompt_template.format(instruction, cleaned_text)

        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_length - max_new_tokens
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode and extract response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text[len(prompt) :].strip()

        return response

    def parse_json_response(self, response: str) -> dict[str, Any] | None:
        """Parse JSON response from the model output."""
        try:
            # Try to find JSON in the response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None

    def generate_batch_response(
        self,
        input_texts: list[str],
        input_categories: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> list[tuple[str, float]]:
        """Generate responses for multiple input texts in batch."""

        # Prepare all prompts in Alpaca style
        prompts = []
        for input_text in input_texts:
            # Clean input text
            cleaned_text = self.input_text_cleaning(input_text)

            # Format input with category dict
            formatted_input = f"""[Voice-of-Customer]: {cleaned_text}

[Category Dict]
{input_categories.strip() if input_categories else self.default_category_dict}
"""
            # Format prompt
            prompt = self.alpaca_prompt_template.format(self.default_instruction, formatted_input)
            prompts.append(prompt)

        if self.use_vllm:
            return self._generate_batch_vllm(prompts, max_new_tokens, temperature, top_p)
        else:
            return self._generate_batch_unsloth(prompts, max_new_tokens, temperature, top_p)

    def _generate_batch_vllm(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> list[tuple[str, float]]:
        """Generate batch responses using vLLM - true parallel batch processing."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # vLLM handles batching efficiently with continuous batching
        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()

            # Extract response
            if "### Response:" in generated_text:
                response = generated_text.split("### Response:")[-1].strip()
            else:
                response = generated_text

            # Calculate confidence from cumulative log prob
            cumulative_logprob = output.outputs[0].cumulative_logprob
            if cumulative_logprob is not None:
                num_tokens = len(output.outputs[0].token_ids)
                avg_logprob = cumulative_logprob / max(num_tokens, 1)
                confidence_score = min(1.0, max(0.0, (avg_logprob + 5) / 5))
            else:
                confidence_score = 0.0

            results.append((response, confidence_score))

        return results

    def _generate_batch_unsloth(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> list[tuple[str, float]]:
        """Generate batch responses using Unsloth - sequential processing for reliability."""
        results = []
        for i, prompt in enumerate(prompts):
            try:
                response, confidence_score = self._generate_unsloth(prompt, max_new_tokens, temperature, top_p)
                results.append((response, confidence_score))
            except Exception as e:
                logger.error(f"Error processing input {i}: {e}")
                results.append(("", 0.0))

        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global model_instance

    # Load model on startup
    model_path = os.getenv("MODEL_PATH", "./outputs_final/best_model")
    use_vllm = os.getenv("USE_VLLM", "true").lower() == "true"

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Using vLLM: {use_vllm and VLLM_AVAILABLE}")

    try:
        model_instance = CESCOInference(model_path=model_path, use_vllm=use_vllm)
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise

    yield

    # Cleanup on shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title="CESCO Inference API",
    description="API for CESCO customer complaint analysis using fine-tuned language model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CESCO Inference API",
        "status": "running",
        "model_loaded": model_instance is not None,
        "backend": "vllm" if (model_instance and model_instance.use_vllm) else "unsloth",
        "vllm_available": VLLM_AVAILABLE,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "healthy", "model_loaded": True}


@app.post("/chat", response_model=InferenceResponse)
async def chat(request: ChatRequest):
    """Generate chat response for customer complaint text."""

    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate response
        raw_response = model_instance.generate_free_response(
            instruction=request.instruction,
            input_text=request.input_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Parse JSON
        parsed_response = model_instance.parse_json_response(raw_response)

        return InferenceResponse(success=True, raw_response=raw_response, parsed_response=parsed_response)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return InferenceResponse(success=False, raw_response="", error=str(e))


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Generate prediction for customer complaint text."""

    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate response
        raw_response, confidence_score = model_instance.generate_response(
            input_text=request.input_text,
            input_categories=request.input_categories if request.input_categories else "",
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Parse JSON
        parsed_response = model_instance.parse_json_response(raw_response)

        return InferenceResponse(
            success=True, raw_response=raw_response, parsed_response=parsed_response, confidence_score=confidence_score
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return InferenceResponse(success=False, raw_response="", error=str(e))


@app.post("/batch", response_model=list[InferenceResponse])
async def batch_predict(request: InferenceBatchRequest):
    """Generate predictions for multiple texts in batch using Alpaca-style processing."""

    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate batch responses using Alpaca-style batch processing
        batch_results = model_instance.generate_batch_response(
            input_texts=request.input_texts,
            input_categories=request.input_categories,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Parse JSON for each response
        results = []
        for raw_response, confidence_score in batch_results:
            parsed_response = model_instance.parse_json_response(raw_response)
            results.append(
                InferenceResponse(
                    success=True,
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    confidence_score=confidence_score,
                )
            )

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        # Return error for all inputs
        return [InferenceResponse(success=False, raw_response="", error=str(e)) for _ in request.input_texts]


if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=True)

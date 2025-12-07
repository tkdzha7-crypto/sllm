import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any

from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported

    print("✅ Unsloth imported successfully")
except ImportError as e:
    print(f"❌ Failed to import unsloth: {e}")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CESCOTrainer:
    """CESCO model trainer with support for full fine-tuning and LoRA."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen-3-8B",
        max_seq_length: int = 8096,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list | None = None,
    ):
        """Initialize the trainer."""
        self.base_model_name = base_model_name
        self.max_seq_length = max_seq_length
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        # Alpaca prompt template (same as in app.py)
        self.alpaca_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

        # Default instruction (same as in app.py)
        self.default_instruction = """You are a helpful assistant that analyzes customer interactions for CESCO. Based on the customer's interaction summary, please respond in JSON format including the following information:
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
- Make sure the response is in Korean, and the JSON output is properly structured."""

        self.model = None
        self.tokenizer = None

    def input_text_cleaning(self, text: str) -> str:
        """Clean input text by keeping only alphabets, Korean characters, and numbers."""
        import re

        # Keep only alphabets (a-z, A-Z), Korean characters (가-힣, ㄱ-ㅎ, ㅏ-ㅣ), and numbers (0-9)
        # Also keep spaces to maintain word separation
        cleaned_text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", text)
        # Clean up multiple spaces and strip
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def load_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading base model: {self.base_model_name}")
        logger.info(f"Training mode: {'LoRA' if self.use_lora else 'Full Fine-tuning'}")

        # Load model with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization for efficiency
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded successfully! Vocab size: {len(self.tokenizer)}")

        if self.use_lora:
            # Apply LoRA adapters
            logger.info("Applying LoRA adapters...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_r,
                target_modules=self.target_modules,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",  # Enable for long context
                random_state=42,
                use_rslora=False,  # Set to True for rank-stabilized LoRA
                loftq_config=None,
            )
            logger.info("LoRA adapters applied successfully!")

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
            )
        else:
            logger.info("Using full fine-tuning mode")
            # For full fine-tuning, enable all parameters
            for param in self.model.parameters():
                param.requires_grad = True

    def format_prompt(self, example: dict[str, Any]) -> str:
        """Format a training example into Alpaca prompt format."""
        # Clean the input text
        cleaned_input = self.input_text_cleaning(example.get("input_text", ""))

        # Get category dict if provided
        category_dict = example.get("category_dict", "")

        # Format input with category dict
        formatted_input = f"""[Voice-of-Customer]: {cleaned_input}

[Category Dict]
{category_dict}"""

        # Get the response (should be JSON string)
        response = example.get("response", "")

        # Format using Alpaca template
        prompt = self.alpaca_prompt_template.format(self.default_instruction, formatted_input, response)

        return prompt

    def prepare_dataset(self, data_path: str, validation_split: float = 0.1):
        """Prepare dataset from JSON or JSONL file."""
        logger.info(f"Loading dataset from: {data_path}")

        # Load the data
        if data_path.endswith(".jsonl"):
            with open(data_path, encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        elif data_path.endswith(".json"):
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError("Data file must be .json or .jsonl format")

        logger.info(f"Loaded {len(data)} examples")

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(data)

        # Split into train and validation
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f"Train size: {len(train_dataset)}, No validation split")

        return train_dataset, eval_dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        output_dir: str = "./outputs",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
        save_total_limit: int = 3,
        fp16: bool = False,
        bf16: bool = False,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        seed: int = 42,
        early_stopping_patience: int = 3,
    ):
        """Train the model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Auto-detect bf16 support
        if bf16 is None:
            bf16 = is_bfloat16_supported()

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            save_total_limit=save_total_limit,
            fp16=fp16 and not bf16,
            bf16=bf16,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            logging_dir=f"{output_dir}/logs",
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_pin_memory=True,
        )

        # Callbacks
        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=self.format_prompt,
            max_seq_length=self.max_seq_length,
            dataset_text_field=None,  # We use formatting_func instead
            packing=False,  # Set to True for better efficiency with short sequences
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final model
        logger.info("Saving final model...")
        if self.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(f"{output_dir}/final_model")
            self.tokenizer.save_pretrained(f"{output_dir}/final_model")

            # Optionally save merged model
            logger.info("Saving merged model (LoRA + base)...")
            self.model.save_pretrained_merged(
                f"{output_dir}/final_model_merged",
                self.tokenizer,
                save_method="merged_16bit",  # or "merged_4bit", "lora"
            )
        else:
            # Save full model
            self.model.save_pretrained(f"{output_dir}/final_model")
            self.tokenizer.save_pretrained(f"{output_dir}/final_model")

        # Save training metrics
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        logger.info(f"Training completed! Metrics: {metrics}")

        return trainer, train_result


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CESCO model with LoRA or full fine-tuning")

    # Model arguments
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-3-8B", help="Base model name or path")
    parser.add_argument("--max_seq_length", type=int, default=8096, help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for training (default: True)")
    parser.add_argument(
        "--full_finetune", action="store_true", default=False, help="Use full fine-tuning instead of LoRA"
    )

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data (JSON or JSONL)")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split ratio")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Override use_lora if full_finetune is specified
    if args.full_finetune:
        args.use_lora = False

    # Initialize trainer
    trainer = CESCOTrainer(
        base_model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Load model
    trainer.load_model()

    # Prepare dataset
    train_dataset, eval_dataset = trainer.prepare_dataset(args.data_path, validation_split=args.validation_split)

    # Train
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

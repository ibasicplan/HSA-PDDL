from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional


CONTROL_TOKENS = (
    "<|eot_id|>", "<|endoftext|>", "<|end|>", "<|return|>",
    "<|start|>", "<|channel|>", "<|message|>", "<|assistant|>",
    "<|user|>", "<|system|>", "<|start_header_id|>", "<|end_header_id|>",
)


def sanitize_output(text: str) -> str:
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


class LocalLLM:
    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        device: str = "auto",
        seed: int = 123,
        max_context_tokens: int = 16384,
        mock_response_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.seed = seed
        self.max_context_tokens = max_context_tokens
        self.mock_response_dir = Path(mock_response_dir).resolve() if mock_response_dir else None
        self.tokenizer = None
        self.model = None
        random.seed(seed)
        if backend == "transformers":
            self._load()
        elif backend != "mock":
            raise ValueError("generator backend must be 'transformers' or 'mock'")

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for the real generator") from exc
        model_path = Path(self.model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {model_path}")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.device == "cpu" or not torch.cuda.is_available():
            resolved_device = "cpu"
            dtype = torch.float32
            device_map = None
        else:
            resolved_device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), dtype=dtype, trust_remote_code=True, device_map=device_map, local_files_only=True
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=dtype, trust_remote_code=True, device_map=device_map, local_files_only=True
            )
        if resolved_device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    def _mock(self, stage: str, sample_key: str, attempt: int) -> Dict[str, Any]:
        if self.mock_response_dir is None:
            raise RuntimeError("--mock_response_dir is required with generator_backend=mock")
        sample_dir = self.mock_response_dir / sample_key
        candidates = [sample_dir / f"{stage}_attempt{attempt}.txt", sample_dir / f"{stage}.txt"]
        path = next((x for x in candidates if x.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No mock response for stage={stage}, sample={sample_key}: {candidates}")
        raw = path.read_text(encoding="utf-8")
        return {
            "raw_output": raw,
            "clean_output": sanitize_output(raw),
            "stats": {"backend": "mock", "source": str(path), "input_tokens": None, "generated_tokens": None, "generation_time_s": 0.0},
        }

    def generate(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        sample_key: str,
        attempt: int,
    ) -> Dict[str, Any]:
        if self.backend == "mock":
            return self._mock(stage, sample_key, attempt)
        import torch

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens >= self.max_context_tokens:
            raise RuntimeError(f"Prompt has {input_tokens} tokens, exceeding context limit {self.max_context_tokens}")
        effective_new = min(max_new_tokens, self.max_context_tokens - input_tokens - 1)
        if effective_new <= 0:
            raise RuntimeError("No token budget remains for generation")
        try:
            model_device = next(x.device for x in self.model.parameters() if x.device.type != "meta")
        except (StopIteration, AttributeError):
            model_device = self.model.device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        kwargs = {
            "max_new_tokens": effective_new,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        elapsed = time.time() - started
        generated_ids = output[0, input_tokens:]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return {
            "raw_output": raw,
            "clean_output": sanitize_output(raw),
            "stats": {
                "backend": "transformers",
                "input_tokens": input_tokens,
                "generated_tokens": int(generated_ids.shape[0]),
                "effective_max_new_tokens": effective_new,
                "generation_time_s": round(elapsed, 4),
            },
        }

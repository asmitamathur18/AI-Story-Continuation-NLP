"""Shared DistilGPT2 inference code used by the UI, local API, and TorchServe handler."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilgpt2"
MAX_PROMPT_CHARS = 4000


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any, str]:
    """Load model/tokenizer once and cache them for the process lifetime."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, device


def generate_story(
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.8,
) -> str:
    """Validate input and return only the newly generated continuation."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please enter a story opening.")
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Story opening is too long. Maximum is {MAX_PROMPT_CHARS} characters.")
    if not 10 <= int(max_new_tokens) <= 180:
        raise ValueError("max_new_tokens must be between 10 and 180.")
    if not 0.2 <= float(temperature) <= 1.5:
        raise ValueError("temperature must be between 0.2 and 1.5.")

    tokenizer, model, device = load_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.92,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return continuation or "(The model did not produce a continuation. Try another prompt.)"

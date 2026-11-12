"""TorchServe custom handler for DistilGPT2 story continuation."""
import json
import os
import torch
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_PROMPT_CHARS = 4000


class StoryHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # The archive includes tokenizer/config files. The model weights are downloaded
        # during packaging or supplied in the model archive directory.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.initialized = True

    def preprocess(self, data):
        item = data[0]
        if isinstance(item, (bytes, bytearray)):
            item = item.decode("utf-8")
        if isinstance(item, str):
            item = json.loads(item)
        if isinstance(item, dict) and "body" in item:
            item = item["body"]
            if isinstance(item, (bytes, bytearray)):
                item = item.decode("utf-8")
            if isinstance(item, str):
                item = json.loads(item)
        if not isinstance(item, dict):
            raise ValueError("Request must be a JSON object.")
        prompt = item.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must be at most {MAX_PROMPT_CHARS} characters.")
        max_new_tokens = int(item.get("max_new_tokens", 60))
        temperature = float(item.get("temperature", 0.8))
        if not 10 <= max_new_tokens <= 180:
            raise ValueError("max_new_tokens must be between 10 and 180.")
        if not 0.2 <= temperature <= 1.5:
            raise ValueError("temperature must be between 0.2 and 1.5.")
        return {
            "prompt": prompt.strip(),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

    def inference(self, data):
        encoded = self.tokenizer(
            data["prompt"], return_tensors="pt", truncation=True, max_length=900
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=data["max_new_tokens"],
                do_sample=True,
                temperature=data["temperature"],
                top_p=0.92,
                repetition_penalty=1.12,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def postprocess(self, inference_output):
        return [{"generated_text": inference_output}]


_service = StoryHandler()
_service.handle = StoryHandler.handle.__get__(_service, StoryHandler)

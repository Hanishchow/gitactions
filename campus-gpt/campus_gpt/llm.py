"""Qwen2.5 loader for Colab (4-bit NF4, optional LoRA adapter). Needs a GPU; not used in tests."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_generator(
    adapter_dir: Optional[str] = None, model_name: str = MODEL_NAME, max_new_tokens: int = 300
) -> Callable[[List[Dict[str, str]]], str]:
    """Return ``generate(messages) -> str`` using greedy decoding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto")
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    @torch.inference_mode()
    def generate(messages: List[Dict[str, str]]) -> str:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
        )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate

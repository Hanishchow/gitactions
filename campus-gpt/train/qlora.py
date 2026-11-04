"""QLoRA fine-tune Qwen2.5-1.5B-Instruct on data/sft_train.jsonl (run on a Colab GPU).

Loss is computed only on the assistant turn. Usage:
    python train/qlora.py --train data/sft_train.jsonl --out adapters/campus-gpt
"""
from __future__ import annotations

import argparse
import json

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="adapters/campus-gpt")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(line) for line in open(a.train, encoding="utf-8") if line.strip()]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            msgs = rows[i]["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = prompt + msgs[-1]["content"] + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False, truncation=True, max_length=a.max_len)["input_ids"]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
            return {"input_ids": f_ids, "labels": labels}

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        fp16=True, logging_steps=5, save_strategy="no", report_to="none",
        optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.05,
        gradient_checkpointing=True, remove_unused_columns=False, seed=7)
    Trainer(model=model, args=args, train_dataset=DS(), data_collator=collate).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved adapter to", a.out)


if __name__ == "__main__":
    main()

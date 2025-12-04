# src/finetune_llm.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.2.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import typer
from datasets import Dataset
from jinja2 import Template
from tqdm import tqdm

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from utils import Config, set_global_seed, get_model_config

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------


def resolve_device(device_arg: str = "auto") -> str:
    """
    Map a user-friendly device string to an actual device:
    - 'auto' -> cuda if available, else mps, else cpu
    - 'cuda', 'cpu', 'mps' -> returned if available
    """
    device_arg = device_arg.lower()

    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_arg in {"cuda", "cpu"}:
        return device_arg

    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available on this machine.")

    raise ValueError(f"Unknown device '{device_arg}'. Use 'auto', 'cuda', 'mps', or 'cpu'.")


# ---------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------


@dataclass
class TrainConfig:
    model_name: str
    output_dir: Path
    use_qlora: bool  # effective (after considering device)
    max_seq_length: int
    num_train_epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    seed: int
    device: str  # 'cuda', 'mps', or 'cpu'


def load_prompts_yaml(path: Path) -> Dict[str, str]:
    import yaml

    with path.open("r") as f:
        data = yaml.safe_load(f)
    system = data["system"]
    prompts = data["prompts"]
    return {"system": system, **prompts}


def build_token_block(tokens: List[str]) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))


def build_gold_json(tokens: List[str], word_labels: List[int], ciu_labels: List[int]) -> str:
    import json

    records = []
    for i, (tok, w, c) in enumerate(zip(tokens, word_labels, ciu_labels)):
        records.append(
            {"index": i, "token": tok, "word_label": int(w), "ciu_label": int(c)}
        )
    return json.dumps(records, indent=2)


def make_sft_examples(
    df: pd.DataFrame,
    prompt_ids: List[str],
    prompts_path: Path,
    mode: str = "z_shot_local",
) -> List[Dict[str, str]]:
    """
    Build SFT-style training examples at utterance or transcript level.

    Input text format:
      SYSTEM MESSAGE
      USER MESSAGE
      ASSISTANT MESSAGE (gold JSON labels)
    """
    prompts_dict = load_prompts_yaml(prompts_path)
    system_prompt = prompts_dict["system"]
    user_template = Template(prompts_dict[mode])

    df = df[df["transcript_id"].isin(prompt_ids)].copy()
    df = df.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    if "utterance_id" in df.columns:
        group_cols = ["transcript_id", "utterance_id"]
        print("[finetune_llm] Grouping by (transcript_id, utterance_id)")
    else:
        group_cols = ["transcript_id"]
        print("[finetune_llm] Grouping by transcript_id only")

    examples: List[Dict[str, str]] = []

    for group_vals, g in tqdm(df.groupby(group_cols), desc="Building SFT examples"):
        if isinstance(group_vals, tuple):
            transcript_id = group_vals[0]
            utterance_id = group_vals[1] if len(group_vals) > 1 else None
        else:
            transcript_id = group_vals
            utterance_id = None

        tokens = g["token_text"].tolist()
        word_labels = g["word_label"].tolist()
        ciu_labels = g["ciu_label"].tolist()

        token_block = build_token_block(tokens)
        group_id = (
            f"{transcript_id}__utt-{utterance_id}" if utterance_id is not None else transcript_id
        )

        user_prompt = user_template.render(
            utterance_id=group_id,
            transcript_id=transcript_id,
            token_block=token_block,
            few_shot_examples="",  # could embed examples here too
        )

        gold_json = build_gold_json(tokens, word_labels, ciu_labels)

        full_text = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{user_prompt}\n\n"
            "ASSISTANT MESSAGE:\n"
            f"{gold_json}\n"
        )

        examples.append(
            {
                "transcript_id": transcript_id,
                "utterance_id": utterance_id,
                "text": full_text,
            }
        )

    return examples


def get_train_config(
    cfg: Config,
    model_key: str,
    output_dir: Path,
    seed: int,
    device: str,
) -> TrainConfig:
    """
    Build TrainConfig from global config and resolved device.

    NOTE: use_qlora is only enabled when device == 'cuda'. On mps/cpu we fall
    back to standard full-model finetuning.
    """
    model_cfg = get_model_config(cfg, model_key)
    ft = model_cfg.get("finetune", {})

    raw_use_qlora = bool(ft.get("use_qlora", False))

    if raw_use_qlora and device != "cuda":
        print(
            f"[finetune_llm] WARNING: use_qlora=True in config, but device='{device}'. "
            "QLoRA requires CUDA; falling back to standard full-model finetuning."
        )

    effective_use_qlora = raw_use_qlora and (device == "cuda")

    return TrainConfig(
        model_name=model_cfg["model_name"],
        output_dir=output_dir,
        use_qlora=effective_use_qlora,
        max_seq_length=int(ft.get("max_seq_length", 1024)),
        num_train_epochs=int(ft.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(ft.get("batch_size", 1)),
        learning_rate=float(ft.get("learning_rate", 2e-4)),
        seed=seed,
        device=device,
    )


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


def load_model_and_tokenizer(train_cfg: TrainConfig):
    """
    Load base model + tokenizer, with QLoRA on CUDA when enabled,
    otherwise standard full-model load.

    - On CUDA + use_qlora=True -> 4-bit QLoRA via bitsandbytes.
    - On CUDA + use_qlora=False -> full model in bfloat16.
    - On MPS -> full model in float16.
    - On CPU -> full model in default dtype (float32).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = train_cfg.device

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if train_cfg.use_qlora:
        # QLoRA path (CUDA only)
        from transformers import BitsAndBytesConfig

        print("[finetune_llm] Loading model with 4-bit QLoRA (CUDA)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Base 4-bit model, device_map=auto to spread across GPU if needed
        model = AutoModelForCausalLM.from_pretrained(
            train_cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Prepare for k-bit training (sets up norms, cast, grad, etc.)
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration for causal LM
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # adjust if needed per architecture
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Sanity check: which parameters are trainable
        model.print_trainable_parameters()

    else:
        # Standard full-model load (no QLoRA)
        print(f"[finetune_llm] Loading full model on device='{device}' (no QLoRA)...")

        # Choose dtype based on device to reduce memory footprint
        if device == "cuda":
            torch_dtype = torch.bfloat16  # good default on A100/V100 class GPUs
        elif device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = None  # CPU: default float32

        if torch_dtype is not None:
            model = AutoModelForCausalLM.from_pretrained(
                train_cfg.model_name,
                torch_dtype=torch_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(train_cfg.model_name)

        model.to(device)

    return model, tokenizer


# ---------------------------------------------------------------------
# Dataset tokenization
# ---------------------------------------------------------------------


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(
        _tok,
        batched=True,
        remove_columns=["text", "transcript_id", "utterance_id"],
    )


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


@app.command()
def main(
    config_path: Path = typer.Option(Path("config.yaml"), help="Config file."),
    model_key: str = typer.Option(
        "phi3-mini",
        help="Key in config.yaml under 'models' (or model_zoo) to fine-tune.",
    ),
    prompts_path: Path = typer.Option(
        Path("prompts/ciu_prompts.yaml"), help="YAML with CIU prompt templates."
    ),
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Normalized labeled tokens.",
    ),
    prompt_ids_path: Path = typer.Option(
        Path("data/splits/prompt_ids.txt"),
        help="Transcript IDs used to build fine-tuning set.",
    ),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompt template key to use for SFT examples.",
    ),
    device: str = typer.Option(
        "auto",
        help="Compute device: 'auto', 'cuda', 'mps', or 'cpu'.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    """
    LoRA / QLoRA fine-tuning script for CIU SFT examples.

    - Builds SFT-style examples from prompt-support transcripts.
    - Fine-tunes a causal LM with QLoRA (on CUDA) or full-model finetuning.
    - Saves adapters under models/llm/<model_key>-ciu-lora/.
    """
    set_global_seed(seed)

    device_resolved = resolve_device(device)
    print(f"[finetune_llm] Using device: {device_resolved}")

    cfg = Config.load(config_path)

    df = pd.read_parquet(labeled_path)
    prompt_ids = prompt_ids_path.read_text().splitlines()

    print(f"[finetune_llm] Building SFT examples for model_key='{model_key}'")
    examples = make_sft_examples(
        df=df,
        prompt_ids=prompt_ids,
        prompts_path=prompts_path,
        mode=mode,
    )
    dataset = Dataset.from_pandas(pd.DataFrame(examples))

    adapter_dir = Path("models/llm") / f"{model_key}-ciu-lora"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_config(cfg, model_key, adapter_dir, seed, device_resolved)
    print(
        f"[finetune_llm] model_name={train_cfg.model_name}, "
        f"use_qlora={train_cfg.use_qlora}, "
        f"output_dir={adapter_dir}, device={train_cfg.device}"
    )
    model, tokenizer = load_model_and_tokenizer(train_cfg)

    tokenized = tokenize_dataset(dataset, tokenizer, train_cfg.max_seq_length)

    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Decide on fp16/bf16 flags for Trainer for the non-QLoRA path.
    use_fp16 = False
    use_bf16 = False
    if device_resolved == "cuda" and not train_cfg.use_qlora:
        # For full-precision CUDA training, use bf16/fp16 to save memory
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        num_train_epochs=train_cfg.num_train_epochs,
        learning_rate=train_cfg.learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        seed=train_cfg.seed,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        # Recommended optimizer for QLoRA
        optim="paged_adamw_8bit" if train_cfg.use_qlora else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[finetune_llm] Starting training...")
    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"[finetune_llm] Fine-tuned adapters saved to {adapter_dir}")


if __name__ == "__main__":
    app()

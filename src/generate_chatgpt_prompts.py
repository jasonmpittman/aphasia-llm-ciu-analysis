# src/generate_chatgpt_prompts.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Generate prompts for running CIU labeling experiments with ChatGPT (web UI).

Supports:
  - Zero-shot prompts (mode='z_shot'): instructions + TARGET utterance only.
  - Few-shot prompts (mode='few_shot'): instructions + K EXAMPLE utterances
    (with gold labels) + TARGET utterance.

For each eval transcript_id, writes a .txt file you can paste directly
into the ChatGPT UI.

Default I/O:
  - Labeled data: data/labeled/ciu_tokens_normalized.parquet
  - Prompt-support IDs: data/splits/prompt_ids.txt
  - Eval IDs: data/splits/eval_ids.txt
  - Output directory: prompts/chatgpt/<mode>/*.txt
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import typer

from utils import set_global_seed

app = typer.Typer(add_completion=False)


INSTRUCTIONS = """You are an expert speech-language pathologist performing Correct Information Unit (CIU) analysis
following Nicholas & Brookshire (1993). You receive pre-tokenized utterances from a clinical
picture description task. Each token is a single word. For each token, you must decide if it is:
(1) a WORD (intelligible lexical item) or NOT WORD; and
(2) if WORD, whether it is a Correct Information Unit (CIU).

A CIU is a single WORD that is accurate, relevant, and informative about the pictured scene, and
understandable in context. Fillers ("uh", "um"), repetitions, false starts, and off-topic words are
NOT CIUs.

You will see:
- (Optionally) one or more EXAMPLE utterances with their correct labels.
- Then a TARGET utterance to label.

For each utterance, tokens are shown as:
index: token

You MUST output labels for the TARGET utterance ONLY, in the following JSON format:

[
  {
    "index": 0,
    "token": "the",
    "word_label": 1,
    "ciu_label": 0
  },
  ...
]

Where:
- index is the integer token index
- token is exactly the token string as given (case-sensitive, do not normalize or change it)
- word_label is 1 if the token is an intelligible WORD, else 0
- ciu_label is 1 if and only if:
    * word_label == 1, and
    * the word is accurate about the scene,
    * relevant (on-topic and adds information),
    * understandable in context
  otherwise ciu_label is 0

Important formatting rules:
- Output a SINGLE JSON array.
- Do NOT include any explanation, commentary, or additional text.
- Do NOT include keys other than "index", "token", "word_label", "ciu_label".
- The array MUST contain one object per token in the TARGET utterance, in order.
""".strip()


def load_id_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"ID file not found: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return ids


def format_tokens_block(df_sub: pd.DataFrame) -> str:
    """Render a 'Tokens:' block as:
       0: token0
       1: token1
       ...
    """
    lines = []
    for _, row in df_sub.sort_values("token_index").iterrows():
        idx = int(row["token_index"])
        tok = str(row["token_text"])
        lines.append(f"{idx}: {tok}")
    return "\n".join(lines)


def format_labels_json(df_sub: pd.DataFrame) -> str:
    """Render gold labels as a pretty-printed JSON array."""
    records = []
    for _, row in df_sub.sort_values("token_index").iterrows():
        records.append(
            {
                "index": int(row["token_index"]),
                "token": str(row["token_text"]),
                "word_label": int(row["word_label"]),
                "ciu_label": int(row["ciu_label"]),
            }
        )
    return json.dumps(records, indent=2, ensure_ascii=False)


@app.command()
def main(
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Gold labeled tokens parquet.",
    ),
    prompt_ids_path: Path = typer.Option(
        Path("data/splits/prompt_ids.txt"),
        help="File with transcript_ids for prompt-support (few-shot examples).",
    ),
    eval_ids_path: Path = typer.Option(
        Path("data/splits/eval_ids.txt"),
        help="File with transcript_ids for evaluation (targets).",
    ),
    out_dir: Path = typer.Option(
        Path("prompts/chatgpt"),
        help="Base output directory for prompts.",
    ),
    mode: str = typer.Option(
        "z_shot",
        help="Prompt mode: 'z_shot' (no examples) or 'few_shot' (with examples).",
    ),
    n_examples: int = typer.Option(
        2,
        help="Number of example utterances to include in few-shot mode (max = #prompt_ids).",
    ),
    seed: int = typer.Option(
        2025,
        help="Random seed for selecting example transcripts.",
    ),
) -> None:
    """
    Generate ChatGPT prompts for CIU labeling experiments.

    One .txt file per eval transcript_id is written to:
      out_dir / mode / <transcript_id>.txt
    """
    mode = mode.strip().lower()
    if mode not in {"z_shot", "few_shot"}:
        raise ValueError("mode must be 'z_shot' or 'few_shot'.")

    set_global_seed(seed)

    # Load labeled data
    df = pd.read_parquet(labeled_path)

    # Basic sanity
    required_cols = {"transcript_id", "token_index", "token_text", "word_label", "ciu_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Labeled data missing required columns: {missing}")

    # Load ID lists
    eval_ids = load_id_list(eval_ids_path)

    if not eval_ids:
        raise ValueError(f"No eval_ids found in {eval_ids_path}")

    prompt_ids: List[str] = []
    example_ids: List[str] = []

    if mode == "few_shot":
        prompt_ids = load_id_list(prompt_ids_path)
        if not prompt_ids:
            raise ValueError(
                f"few_shot mode requested but no prompt_ids found in {prompt_ids_path}"
            )
        if n_examples <= 0:
            raise ValueError("n_examples must be >= 1 for few_shot mode.")

        # Choose a fixed set of example transcripts for all prompts (simpler & reproducible)
        import random

        n_examples_eff = min(n_examples, len(prompt_ids))
        example_ids = random.sample(prompt_ids, k=n_examples_eff)
        print(
            f"[generate_chatgpt_prompts] few_shot mode: using {n_examples_eff} example transcripts: "
            + ", ".join(example_ids)
        )

    # Prepare output directory
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[generate_chatgpt_prompts] Generating prompts for mode='{mode}' "
        f"into {mode_dir} for {len(eval_ids)} eval transcripts."
    )

    # Pre-slice per transcript for efficiency
    grouped = {tid: g.copy() for tid, g in df.groupby("transcript_id")}

    # Pre-build example blocks (once) for few_shot
    example_blocks: List[str] = []
    if mode == "few_shot":
        for i, ex_tid in enumerate(example_ids, start=1):
            if ex_tid not in grouped:
                print(
                    f"[generate_chatgpt_prompts] WARNING: example transcript_id '{ex_tid}' "
                    "not found in labeled data; skipping."
                )
                continue

            df_ex = grouped[ex_tid]
            ex_tokens_block = format_tokens_block(df_ex)
            ex_labels_json = format_labels_json(df_ex)

            block = (
                "----------------\n"
                f"EXAMPLE UTTERANCE {i}\n"
                f"Example Utterance ID: {ex_tid}\n\n"
                "Tokens:\n"
                f"{ex_tokens_block}\n\n"
                "Labels:\n"
                f"{ex_labels_json}\n"
            )
            example_blocks.append(block)

    # Generate one prompt per eval transcript
    for tid in eval_ids:
        if tid not in grouped:
            print(
                f"[generate_chatgpt_prompts] WARNING: eval transcript_id '{tid}' "
                "not found in labeled data; skipping."
            )
            continue

        df_tgt = grouped[tid]
        tgt_tokens_block = format_tokens_block(df_tgt)

        parts: List[str] = []
        parts.append(INSTRUCTIONS)

        # Few-shot examples (if any)
        if mode == "few_shot" and example_blocks:
            parts.append("\n")
            parts.extend(example_blocks)

        # Target block
        target_block = (
            "----------------\n"
            "TARGET UTTERANCE\n"
            f"Utterance ID: {tid}\n\n"
            "Tokens:\n"
            f"{tgt_tokens_block}\n\n"
            f"Now, using the exact same rules and output format, output ONLY the JSON array of labels\n"
            f"for the TARGET utterance (Utterance ID: {tid}).\n"
            "Remember: no explanations, just the JSON array.\n"
        )
        parts.append(target_block)

        prompt_text = "\n".join(parts).strip() + "\n"

        out_path = mode_dir / f"{tid}.txt"
        out_path.write_text(prompt_text, encoding="utf-8")

    print("[generate_chatgpt_prompts] Done.")


if __name__ == "__main__":
    app()


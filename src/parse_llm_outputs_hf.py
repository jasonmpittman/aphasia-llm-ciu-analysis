# src/parse_llm_outputs_hf.py
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""
Parse raw HF LLM outputs (saved by run_llm_inference.py) into token-level predictions
and merge with gold labels.

Assumes each raw JSON file looks like:

{
  "transcript_id": "...",
  "utterance_id": null,
  "group_id": "...",
  "mode": "few_shot_local" | "z_shot_local" | ...,
  "model_name": "...",
  "model_key": "...",
  "use_lora": false,
  "adapter_dir": null,
  "response_text": "SYSTEM MESSAGE: ... Example Utterance ... Labels: [ {...}, ... ] ... Now, label the following utterance ... Labels: [ {...}, ... ]"
}

We extract the **last** 'Labels: [...]' block from response_text, which should correspond
to the model's labels for the *target* utterance (not the few-shot examples).
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import typer
from tqdm import tqdm

from utils import set_global_seed

app = typer.Typer(add_completion=False)


def extract_last_labels_array(text: str) -> List[Dict[str, Any]]:
    """
    Find the last 'Labels:' block in the response_text and parse the JSON array
    that immediately follows it.

    We:
      1) locate the last occurrence of 'Labels:'
      2) from there, find the first '[' and the matching last ']'
      3) json.loads that slice
    """
    labels_pos = text.rfind("Labels:")
    if labels_pos == -1:
        raise ValueError("No 'Labels:' marker found in response_text.")

    # Slice starting from the last 'Labels:'
    sub = text[labels_pos:]

    start = sub.find("[")
    end = sub.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Could not find a JSON array following the last 'Labels:' marker."
        )

    snippet = sub[start : end + 1]

    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode labels JSON array: {e}") from e

    if not isinstance(obj, list):
        raise ValueError(
            f"Expected a JSON array for labels, got {type(obj)} instead."
        )

    return obj


@app.command()
def main(
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Gold labeled tokens parquet.",
    ),
    raw_dir: Path = typer.Option(
        Path("results/raw/hf_local/llama3-8b/few_shot_local"),
        help="Directory with raw HF JSON outputs from run_llm_inference.py.",
    ),
    out_path: Path = typer.Option(
        Path("results/parsed/llm_predictions.parquet"),
        help="Output path for merged predictions + gold labels.",
    ),
    seed: int = typer.Option(
        2025,
        help="Random seed (not heavily used here, but kept for reproducibility).",
    ),
) -> None:
    set_global_seed(seed)

    # Load gold labels
    df_gold = pd.read_parquet(labeled_path)
    df_gold = df_gold.sort_values(
        ["transcript_id", "token_index"]
    ).reset_index(drop=True)

    # Check for duplicate keys in gold (for debugging + safety)
    dup_mask = df_gold.duplicated(subset=["transcript_id", "token_index"], keep=False)
    if dup_mask.any():
        dup_df = df_gold.loc[dup_mask, ["transcript_id", "token_index"]]
        n_dup_rows = dup_mask.sum()
        n_dup_keys = dup_df.drop_duplicates().shape[0]
        print(
            f"[parse_llm_outputs_hf] WARNING: found {n_dup_rows} rows in gold labels "
            f"with non-unique (transcript_id, token_index) across {n_dup_keys} keys. "
            f"Keeping the first occurrence of each and dropping the rest."
        )
        df_gold = df_gold.drop_duplicates(
            subset=["transcript_id", "token_index"], keep="first"
        ).reset_index(drop=True)

    pred_rows: List[Dict[str, Any]] = []

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No .json files found in {raw_dir}")

    print(f"[parse_llm_outputs_hf] Found {len(raw_files)} raw JSON files in {raw_dir}")

    for fp in tqdm(raw_files, desc="Parsing HF outputs"):
        with fp.open("r") as f:
            data = json.load(f)

        # Expect wrapper dict
        if not isinstance(data, dict):
            print(
                f"[parse_llm_outputs_hf] WARNING: {fp.name} is not a dict "
                f"(type={type(data)}); skipping (likely from an older run)."
            )
            continue

        transcript_id = data.get("transcript_id")
        utterance_id = data.get("utterance_id")
        group_id = data.get("group_id")
        mode = data.get("mode")
        model_name = data.get("model_name")
        model_key = data.get("model_key")
        use_lora = data.get("use_lora")
        adapter_dir = data.get("adapter_dir")
        response_text = data.get("response_text", "")

        try:
            records = extract_last_labels_array(response_text)
        except Exception as e:
            print(f"[parse_llm_outputs_hf] WARNING: skipping {fp.name}: {e}")
            continue

        for rec in records:
            if not isinstance(rec, dict):
                continue

            try:
                idx = int(rec["index"])
            except (KeyError, ValueError, TypeError):
                continue

            try:
                pred_word = int(rec.get("word_label", 0))
            except (ValueError, TypeError):
                pred_word = 0

            try:
                pred_ciu = int(rec.get("ciu_label", 0))
            except (ValueError, TypeError):
                pred_ciu = 0

            token_pred = rec.get("token")

            pred_rows.append(
                {
                    "transcript_id": transcript_id,
                    "utterance_id": utterance_id,
                    "group_id": group_id,
                    "mode": mode,
                    "model_name": model_name,
                    "model_key": model_key,
                    "use_lora": use_lora,
                    "adapter_dir": adapter_dir,
                    "token_index": idx,
                    "token_pred": token_pred,
                    "pred_word_label": pred_word,
                    "pred_ciu_label": pred_ciu,
                }
            )

    if not pred_rows:
        raise RuntimeError(
            "[parse_llm_outputs_hf] No prediction rows were parsed. "
            "Check that the model outputs contain a final 'Labels: [...]' block "
            "for the target utterance."
        )

    df_pred = pd.DataFrame(pred_rows)

    # Deduplicate predictions as well, just in case
    dup_p_mask = df_pred.duplicated(
        subset=["transcript_id", "token_index"], keep=False
    )
    if dup_p_mask.any():
        dup_p_df = df_pred.loc[dup_p_mask, ["transcript_id", "token_index"]]
        n_dup_p_rows = dup_p_mask.sum()
        n_dup_p_keys = dup_p_df.drop_duplicates().shape[0]
        print(
            f"[parse_llm_outputs_hf] WARNING: found {n_dup_p_rows} prediction rows "
            f"sharing (transcript_id, token_index) across {n_dup_p_keys} keys. "
            f"Keeping the last occurrence."
        )
        df_pred = (
            df_pred.sort_values(["transcript_id", "token_index"])
            .drop_duplicates(subset=["transcript_id", "token_index"], keep="last")
            .reset_index(drop=True)
        )

    # Final merge (allowing left 1-to-0/1, right 0/1; we don't enforce validate)
    df_merged = df_gold.merge(
        df_pred,
        on=["transcript_id", "token_index"],
        how="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(out_path)
    print(f"[parse_llm_outputs_hf] Wrote merged predictions to {out_path}")


if __name__ == "__main__":
    app()
from __future__ import annotations

__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import typer
from tqdm import tqdm

from utils import set_global_seed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    labeled_path: Path = typer.Option(
        Path("data/labeled/ciu_tokens_normalized.parquet"),
        help="Gold labeled tokens parquet.",
    ),
    raw_dir: Path = typer.Option(
        Path("results/raw/chatgpt/few_shot_ui"),
        help="Directory with ChatGPT UI JSON outputs (one file per transcript).",
    ),
    out_path: Path = typer.Option(
        Path("results/parsed/llm_predictions_chatgpt_few_shot_ui.parquet"),
        help="Output path for merged predictions + gold labels.",
    ),
    seed: int = typer.Option(
        2025,
        help="Random seed (not really used, kept for reproducibility).",
    ),
) -> None:
    set_global_seed(seed)

    # Load gold labels
    df_gold = pd.read_parquet(labeled_path)
    df_gold = df_gold.sort_values(
        ["transcript_id", "token_index"]
    ).reset_index(drop=True)

    # Ensure uniqueness in gold
    df_gold = df_gold.drop_duplicates(
        subset=["transcript_id", "token_index"], keep="first"
    ).reset_index(drop=True)

    pred_rows: List[Dict[str, Any]] = []

    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No .json files found in {raw_dir}")

    print(f"[parse_chatgpt_ui_outputs] Found {len(raw_files)} raw JSON files in {raw_dir}")

    for fp in tqdm(raw_files, desc="Parsing ChatGPT UI outputs"):
        # transcript_id from filename, e.g. CR_001.json -> CR_001
        transcript_id = fp.stem

        with fp.open("r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(
                f"[parse_chatgpt_ui_outputs] WARNING: {fp.name} is not a JSON array; "
                "skipping."
            )
            continue

        for rec in data:
            if not isinstance(rec, dict):
                continue

            try:
                idx = int(rec["index"])
            except (KeyError, ValueError, TypeError):
                continue

            token_pred = rec.get("token")

            try:
                pred_word = int(rec.get("word_label", 0))
            except (ValueError, TypeError):
                pred_word = 0

            try:
                pred_ciu = int(rec.get("ciu_label", 0))
            except (ValueError, TypeError):
                pred_ciu = 0

            pred_rows.append(
                {
                    "transcript_id": transcript_id,
                    "token_index": idx,
                    "token_pred": token_pred,
                    "pred_word_label": pred_word,
                    "pred_ciu_label": pred_ciu,
                    "mode": "few_shot_chatgpt_ui",
                    "model_name": "chatgpt-ui",
                }
            )

    if not pred_rows:
        raise RuntimeError(
            "[parse_chatgpt_ui_outputs] No prediction rows were parsed. "
            "Check that the ChatGPT outputs are valid JSON arrays."
        )

    df_pred = pd.DataFrame(pred_rows)

    # Deduplicate predictions just in case
    df_pred = (
        df_pred.sort_values(["transcript_id", "token_index"])
        .drop_duplicates(subset=["transcript_id", "token_index"], keep="last")
        .reset_index(drop=True)
    )

    # Merge with gold labels
    df_merged = df_gold.merge(
        df_pred,
        on=["transcript_id", "token_index"],
        how="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(out_path)
    print(f"[parse_chatgpt_ui_outputs] Wrote merged predictions to {out_path}")


if __name__ == "__main__":
    app()
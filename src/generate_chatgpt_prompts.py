# src/generate_chatgpt_prompts.py
__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import typer
from jinja2 import Template
from tqdm import tqdm

from utils import Config, set_global_seed

app = typer.Typer(add_completion=False)


def load_prompts_yaml(path: Path) -> Dict[str, str]:
    import yaml
    with path.open("r") as f:
        data = yaml.safe_load(f)
    system = data["system"]
    prompts = data["prompts"]
    return {"system": system, **prompts}


def build_token_block(tokens: List[str]) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens))


def choose_grouping_cols(df: pd.DataFrame) -> Tuple[List[str], bool]:
    """
    Decide how to group tokens for prompting.

    If 'utterance_id' exists, group by (transcript_id, utterance_id).
    Otherwise, group by transcript_id only.
    """
    if "utterance_id" in df.columns:
        print("ChatGPT prompts: grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
    else:
        print("ChatGPT prompts: no 'utterance_id'; grouping by transcript_id only.")
        return ["transcript_id"], False


@app.command()
def main(
    config_path: Path = typer.Option(Path("config.yaml"), help="Config file."),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompting mode: z_shot_local | few_shot_local | few_shot_global",
    ),
    out_dir: Path = typer.Option(
        Path("results/prompts/chatgpt"),
        help="Where to save prompt text files.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    """
    Generate one prompt text file per utterance (if available) or per transcript,
    for manual use in the ChatGPT web UI.
    """
    set_global_seed(seed)
    cfg = Config.load(config_path)

    prompts_dict = load_prompts_yaml(Path("prompts/ciu_prompts.yaml"))
    system_prompt = prompts_dict["system"]
    user_template = Template(prompts_dict[mode])

    labeled_path = Path(cfg["data"]["labeled_normalized"])
    eval_ids_path = Path(cfg["data"]["eval_ids"])

    df = pd.read_parquet(labeled_path)
    eval_ids = set(eval_ids_path.read_text().splitlines())
    df = df[df["transcript_id"].isin(eval_ids)].copy()
    df = df.sort_values(["transcript_id", "token_index"]).reset_index(drop=True)

    group_cols, has_utter = choose_grouping_cols(df)

    out_mode_dir = out_dir / mode
    out_mode_dir.mkdir(parents=True, exist_ok=True)

    for group_vals, g in tqdm(df.groupby(group_cols), desc="Generating ChatGPT prompts"):
        if isinstance(group_vals, tuple):
            transcript_id = group_vals[0]
            utterance_id = group_vals[1] if len(group_vals) > 1 else None
        else:
            transcript_id = group_vals
            utterance_id = None

        tokens = g["token_text"].tolist()
        token_block = build_token_block(tokens)

        group_id = (
            f"{transcript_id}__utt-{utterance_id}" if has_utter and utterance_id is not None
            else transcript_id
        )

        rendered_user = user_template.render(
            utterance_id=group_id,
            transcript_id=transcript_id,
            token_block=token_block,
            few_shot_examples="",  # wire in examples later if you want
        )

        prompt_text = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{rendered_user}\n"
        )

        (out_mode_dir / f"{group_id}.txt").write_text(prompt_text)

    print(f"ChatGPT prompts written to {out_mode_dir}")


if __name__ == "__main__":
    app()

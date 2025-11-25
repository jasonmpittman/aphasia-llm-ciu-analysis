# src/run_llm_inference.py
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

from utils import Config, save_json, set_global_seed

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


def load_hf_pipeline(model_name: str, max_new_tokens: int):
    """
    Load a local HF model on Apple Silicon (MPS if available, else CPU).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return text_gen


def choose_grouping_cols(df: pd.DataFrame) -> Tuple[List[str], bool]:
    if "utterance_id" in df.columns:
        print("HF inference: grouping by (transcript_id, utterance_id).")
        return ["transcript_id", "utterance_id"], True
    else:
        print("HF inference: no 'utterance_id'; grouping by transcript_id only.")
        return ["transcript_id"], False


@app.command()
def main(
    config_path: Path = typer.Option(Path("config.yaml"), help="Config file."),
    mode: str = typer.Option(
        "z_shot_local",
        help="Prompting mode: z_shot_local | few_shot_local | few_shot_global",
    ),
    out_root: Path = typer.Option(
        Path("results/raw/hf_local"),
        help="Root directory for raw HF model responses.",
    ),
    seed: int = typer.Option(2025, help="Random seed."),
) -> None:
    set_global_seed(seed)
    cfg = Config.load(config_path)

    model_name = cfg["llm"]["model_name"]
    max_new_tokens = int(cfg["llm"].get("max_new_tokens", 2048))

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

    model_basename = model_name.split("/")[-1]
    out_dir = out_root / model_basename / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    text_gen = load_hf_pipeline(model_name, max_new_tokens=max_new_tokens)

    for group_vals, g in tqdm(
        df.groupby(group_cols), desc=f"Running HF LLM: {model_basename} | {mode}"
    ):
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
            few_shot_examples="",  # wire in examples later
        )

        prompt = (
            "SYSTEM MESSAGE:\n"
            f"{system_prompt}\n\n"
            "USER MESSAGE:\n"
            f"{rendered_user}\n"
        )

        gen = text_gen(prompt)[0]["generated_text"]

        save_json(
            {
                "transcript_id": transcript_id,
                "utterance_id": utterance_id,
                "group_id": group_id,
                "mode": mode,
                "model_name": model_name,
                "response_text": gen,
            },
            out_dir / f"{group_id}.json",
        )

    print(f"Completed HF inference. Outputs in {out_dir}")


if __name__ == "__main__":
    app()

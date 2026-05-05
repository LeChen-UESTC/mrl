#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import load_eval_model, pad_token_id
from qwen_omni_retrieval.data.cache_dataset import CachedRetrievalDataset
from qwen_omni_retrieval.data.collator import QwenOmniCachedCollator
from qwen_omni_retrieval.data.modality import parse_modalities, validate_modalities
from qwen_omni_retrieval.utils.config import load_config
from qwen_omni_retrieval.utils.tensor import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cached dataset embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modalities", required=True)
    parser.add_argument("--output_pt", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    modalities = parse_modalities(args.modalities)
    dataset_name = cfg["eval"].get("name", "eval")
    validate_modalities(modalities, dataset_name=dataset_name, allow_vast_cap=dataset_name.lower() == "vast")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_eval_model(cfg, device)
    dataset = CachedRetrievalDataset(
        cfg["eval"]["cache_dir"],
        required_modalities=modalities,
        caption_selection=cfg["eval"].get("caption_selection", "random"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["eval"].get("batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg["eval"].get("num_workers", 2)),
        collate_fn=QwenOmniCachedCollator(modalities=modalities, pad_token_id=pad_token_id(processor)),
    )
    rows = []
    for batch in loader:
        modality_inputs = {
            modality: move_to_device(batch["modalities"][modality], device)
            for modality in modalities
        }
        embeddings = {
            modality: tensor.detach().float().cpu()
            for modality, tensor in model(modality_inputs).items()
        }
        for i, video_id in enumerate(batch["video_ids"]):
            rows.append(
                {
                    "index": int(batch["indices"][i]),
                    "video_id": video_id,
                    "embeddings": {modality: embeddings[modality][i] for modality in modalities},
                }
            )
    rows.sort(key=lambda item: item["index"])
    output_path = Path(args.output_pt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, output_path)
    print({"saved": str(output_path), "rows": len(rows), "modalities": modalities})


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.data.cache_dataset import CachedRetrievalDataset
from qwen_omni_retrieval.data.collator import QwenOmniCachedCollator
from qwen_omni_retrieval.data.serialization import encode_jsonable
from qwen_omni_retrieval.models.embedding import thinker_final_hidden_state


def test_encoder_feature_cache_loads_json_tokens_and_pt_features() -> None:
    with TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        token_dir = cache_dir / "text_tokens"
        feature_dir = cache_dir / "feature_shards"
        token_dir.mkdir()
        feature_dir.mkdir()

        token_modalities = {
            "vision_cap": [
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            ],
            "video": {
                "input_ids": torch.tensor([[1, 9, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
            },
        }
        with (token_dir / "shard_00000.jsonl").open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "cache_key": "video0::0",
                        "modalities": encode_jsonable(token_modalities),
                    }
                )
                + "\n"
            )

        torch.save(
            {
                "video0::0": {
                    "video": {
                        "video_features": torch.tensor([[0.25, 0.75]], dtype=torch.float16),
                    }
                }
            },
            feature_dir / "shard_00000.pt",
        )

        with (cache_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "cache_format": "encoder_features_v1",
                        "sample_id": "video0",
                        "video_id": "video0",
                        "available_modalities": ["vision_cap", "video"],
                        "token_shard": "text_tokens/shard_00000.jsonl",
                        "feature_shard": "feature_shards/shard_00000.pt",
                        "cache_key": "video0::0",
                    }
                )
                + "\n"
            )

        dataset = CachedRetrievalDataset(
            cache_dir,
            required_modalities=["vision_cap", "video"],
            caption_selection="first",
        )
        sample = dataset[0]
        assert sample["modalities"]["video"]["input_ids"].shape == (1, 3)
        assert sample["modalities"]["video"]["video_features"].dtype == torch.float16

        batch = QwenOmniCachedCollator(modalities=["vision_cap", "video"])([sample])
        assert batch["modalities"]["video"]["video_features"].shape == (1, 2)


class FakeThinker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 2)
        self.video_encoder_called = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_video_features(self, *args, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        self.video_encoder_called = True
        raise AssertionError("video encoder should be skipped when video_features is cached")

    def get_placeholder_mask(self, input_ids: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch.zeros(input_ids.shape + (2,), dtype=torch.bool, device=input_ids.device)
        video_mask = mask.clone()
        video_mask[:, 1, :] = True
        return mask.clone(), video_mask, mask.clone()

    def get_rope_index(self, **kwargs) -> tuple[None, None]:
        return None, None

    def model(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])


def test_precomputed_video_features_skip_video_encoder() -> None:
    thinker = FakeThinker()
    inputs = {
        "input_ids": torch.tensor([[1, 9, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "video_grid_thw": torch.tensor([[1, 1, 1]]),
        "video_features": torch.tensor([[0.5, 0.75]]),
    }
    hidden = thinker_final_hidden_state(thinker, inputs, use_audio_in_video=False)
    assert not thinker.video_encoder_called
    assert torch.allclose(hidden[0, 1], torch.tensor([0.5, 0.75]))


if __name__ == "__main__":
    test_encoder_feature_cache_loads_json_tokens_and_pt_features()
    test_precomputed_video_features_skip_video_encoder()
    print("ok")

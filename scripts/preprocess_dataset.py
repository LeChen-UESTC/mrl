#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_omni_retrieval.data.modality import parse_modalities
from qwen_omni_retrieval.data.preprocess import preprocess_dataset
from qwen_omni_retrieval.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Qwen2.5-Omni retrieval cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--anno_path")
    parser.add_argument("--video_dir")
    parser.add_argument("--audio_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--modalities", help="Comma separated modalities to cache.")
    parser.add_argument("--nframes", type=int, help="Fixed number of video frames. Omit for no frame limit.")
    parser.add_argument("--max_samples", type=int, help="Only preprocess the first N records for sanity checks.")
    parser.add_argument("--log_every", type=int, help="Fallback progress log interval when tqdm is unavailable.")
    parser.add_argument("--use_audio_in_video", choices=["true", "false"])
    parser.add_argument("--cache_media_features", choices=["true", "false"])
    parser.add_argument("--cache_raw_processor_tensors", choices=["true", "false"])
    parser.add_argument("--feature_dtype", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.anno_path:
        cfg["dataset"]["anno_path"] = args.anno_path
    if args.video_dir:
        cfg["dataset"]["video_dir"] = args.video_dir
    if args.audio_dir:
        cfg["dataset"]["audio_dir"] = args.audio_dir
    if args.output_dir:
        cfg["cache"]["output_dir"] = args.output_dir
    if args.modalities is not None:
        cfg["cache"]["modalities_to_cache"] = parse_modalities(args.modalities)
    if args.nframes is not None:
        cfg.setdefault("video", {})["nframes"] = args.nframes
    if args.max_samples is not None:
        cfg["cache"]["max_samples"] = args.max_samples
    if args.log_every is not None:
        cfg["cache"]["log_every"] = args.log_every
    if args.use_audio_in_video is not None:
        cfg["dataset"]["use_audio_in_video"] = args.use_audio_in_video == "true"
    if args.cache_media_features is not None:
        cfg["cache"]["cache_media_features"] = args.cache_media_features == "true"
    if args.cache_raw_processor_tensors is not None:
        cfg["cache"]["cache_raw_processor_tensors"] = args.cache_raw_processor_tensors == "true"
    if args.feature_dtype is not None:
        cfg["cache"]["feature_dtype"] = args.feature_dtype

    summary = preprocess_dataset(cfg)
    print(summary)


if __name__ == "__main__":
    main()

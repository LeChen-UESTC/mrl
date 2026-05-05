# Gram-Volume Retrieval Based on Qwen2.5-Omni

This project finetunes the Qwen2.5-Omni Thinker with LoRA for multimodal embedding and retrieval. The trainable parts are LoRA adapters on `q_proj/k_proj/v_proj/o_proj` and an optional projection head.

## Preprocess

The default cache format is `manifest.jsonl + text_tokens/*.jsonl + feature_shards/*.pt`.
Token-side tensors such as `input_ids`, attention masks, and media grid metadata are stored as
JSONL. Video/audio modalities store frozen encoder features in fp16 `.pt` shards, so training
skips tokenization, media processing, and the frozen media encoders.

```bash
bash scripts_sh/preprocess_vast.sh
bash scripts_sh/preprocess_downstreams.sh
```

Set a fixed video frame count when matching a baseline:

```bash
bash scripts_sh/preprocess_vast.sh --nframes 8
bash scripts_sh/preprocess_downstreams.sh --nframes 8
```

Run a quick sanity check before full preprocessing:

```bash
bash scripts_sh/preprocess_vast.sh --nframes 8 --max_samples 10
```

Omit `--nframes` for the default Qwen processor sampling. When `--nframes 8` is set, cache directories are written with an `_n_frames_8` suffix, for example `/mnt/d/cl/mrl/data_cache/vast_train_n_frames_8`.

Change the stored media feature precision when needed:

```bash
bash scripts_sh/preprocess_vast.sh --nframes 8 --feature_dtype bf16
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts_sh/train_vast.sh --extra audio,subtitle,vast_cap
```

## Evaluate

```bash
bash scripts_sh/eval_msrvtt.sh \
  --checkpoint_dir /mnt/d/cl/mrl/outputs/vast_lora_volume/step_0000500 \
  --query vision_cap \
  --target video \
  --aux audio
```

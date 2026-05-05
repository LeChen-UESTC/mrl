# Gram-Volume Retrieval Based on Qwen2.5-Omni

This project finetunes the Qwen2.5-Omni Thinker with LoRA for multimodal embedding and retrieval. The trainable parts are LoRA adapters on `q_proj/k_proj/v_proj/o_proj` and an optional projection head.

## Preprocess

The cache format is `manifest.jsonl + shards/*.pt`. The `.pt` shards contain processor tensors, so training does not repeat tokenization or media processing.

```bash
bash scripts_sh/preprocess_vast.sh
bash scripts_sh/preprocess_downstreams.sh
```

Set a fixed video frame count when matching a baseline:

```bash
bash scripts_sh/preprocess_vast.sh --nframes 8
bash scripts_sh/preprocess_downstreams.sh --nframes 8
```

Omit `--nframes` for the default Qwen processor sampling. When `--nframes 8` is set, cache directories are written with an `_n_frames_8` suffix, for example `data_cache/vast_train_n_frames_8`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts_sh/train_vast.sh --extra audio,subtitle,vast_cap
```

## Evaluate

```bash
bash scripts_sh/eval_msrvtt.sh \
  --checkpoint_dir /Users/bytedance/Qwen2.5Omni/outputs/vast_lora_volume/step_0000500 \
  --query vision_cap \
  --target video \
  --aux audio
```

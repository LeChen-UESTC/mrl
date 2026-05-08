# Gram-Volume Retrieval Based on Qwen2.5-Omni

This project finetunes the Qwen2.5-Omni Thinker with LoRA for multimodal embedding and retrieval. The trainable parts are LoRA adapters on `q_proj/k_proj/v_proj/o_proj` and an optional projection head.

## Preprocess

The default cache format is `manifest.jsonl + text_tokens/*.jsonl + feature_shards/*.pt`.
Token-side tensors such as `input_ids`, attention masks, and media grid metadata are stored as
JSONL. Video/audio modalities store frozen encoder features in fp16 `.pt` shards, so training
skips tokenization, media processing, and the frozen media encoders.

```bash
bash scripts_sh/preprocess_vast.sh --nframes 2 --max_samples 10 --feature_dtype fp16
bash scripts_sh/preprocess_downstreams.sh --nframes 8 --max_samples 10 --feature_dtype fp16
```

## Train

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 \
NPROC_PER_NODE=4 \
bash scripts_sh/train_vast.sh \
  --modality video audio vision_cap \
  --epochs 3 \
  --learning_rate 5e-5 \
  --batch_size 16 \
  --eval_batch_size 4 \
  --eval_nframes 8 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --do_eval false \
  --eval_steps 500 \
  --save_steps 500 \
  --wandb_mode offline 
```

## Evaluate

Standalone eval uses the same cache-first, raw-media fallback behavior.
When `--nframes` is omitted, raw-video fallback uses the processor's default 2 fps sampling and
the auto JSON filename ends with `_2fps.json`; fixed frame counts end with suffixes such as
`_8frames.json`.

```bash
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=6,7 \
NPROC_PER_NODE=2 \
bash scripts_sh/eval_msrvtt.sh \
  --checkpoint_dir /mnt/d/cl/mrl/outputs/models/train_vast_inverse_volume_video-audio-vision_cap_lr5e-5_lora-r16-a32-d0.05_proj-shared-1024/step_0001000 \
  --query vision_cap \
  --target video \
  --aux audio \
  --nframes 8 \
  --batch_size 4 \
  --num_workers 8 \
  2>&1 | tee /mnt/d/cl/mrl/outputs/eval_msrvtt_step_0001000.log
```

## Visualize

Plot the default offline W&B run metrics configured in the script: `train/loss`, `train/lr`,
and `train/volume_mean`.

```bash
python visualization/plot_wandb_curve.py
```

By default it reads `/mnt/d/cl/mrl/wandb/offline-run-20260506_055038-8b0mj82x` and writes
`wandb_metrics.svg` plus `wandb_metrics.csv` under `/mnt/d/cl/mrl/visualizations`. If
`matplotlib` already exists in the environment, it also writes `wandb_metrics.png`.

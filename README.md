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
  --max_steps 10000 \
  --learning_rate 5e-5 \
  --batch_size 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --do_eval false \
  --wandb_mode offline
```

`--epochs` controls full dataset passes. `--max_steps` caps total optimizer steps; set it to
`0` or omit it for no step cap. Common runtime overrides also include `--eval_steps`,
`--save_steps`, `--log_steps`, `--weight_decay`, `--max_grad_norm`, `--eval_batch_size`,
`--num_workers`, `--lora_target_modules`, and `--lora_bias`. Rank 0 shows a tqdm training
progress bar when `tqdm` is available; otherwise it falls back to `--log_steps` prints.

During training-time evaluation, each eval dataset first tries `cache_dir/manifest.jsonl`.
If that cache is missing or lacks the requested modalities, the script falls back to the raw
`anno_path` and video/audio directories configured under `eval_datasets` and processes media
on the fly without writing a cache.

## Evaluate

Standalone eval uses the same cache-first, raw-media fallback behavior.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts_sh/eval_msrvtt.sh \
  --checkpoint_dir /mnt/d/cl/mrl/outputs/vast_lora_volume/step_0001000 \
  --query vision_cap \
  --target video \
  --aux audio \
  --batch_size 4 \
  --num_workers 8
```

`eval_msrvtt.sh` uses `torchrun`; set `NPROC_PER_NODE` to the number of visible GPUs.
Rank 0 prints model-loading status, cache/raw source, and a tqdm progress bar.

Unless `--output_json` is passed explicitly, the result filename appends the last two
checkpoint path parts, for example `eval_msrvtt_vast_lora_volume_step_0001000.json`.

## Visualize

Plot the default offline W&B run metrics configured in the script: `train/loss`, `train/lr`,
and `train/volume_mean`.

```bash
python visualization/plot_wandb_curve.py
```

By default it reads `/mnt/d/cl/mrl/wandb/offline-run-20260506_055038-8b0mj82x` and writes
`wandb_metrics.svg` plus `wandb_metrics.csv` under `/mnt/d/cl/mrl/visualizations`. If
`matplotlib` already exists in the environment, it also writes `wandb_metrics.png`.

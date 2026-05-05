# Gram-Volume Retrieval Based on Qwen2.5-Omni

This project fine-tunes the Qwen2.5-Omni Thinker with LoRA for multimodal embedding and retrieval. The trainable parts are LoRA adapters on `q_proj/k_proj/v_proj/o_proj` and an optional projection head.

## Preprocess

The cache format is `manifest.jsonl + shards/*.pt`. The `.pt` shards contain processor tensors, so training does not repeat tokenization or media processing.

```bash
bash scripts_sh/preprocess_vast.sh
bash scripts_sh/preprocess_downstreams.sh
```

Optional cache modalities can be overridden:

```bash
bash scripts_sh/preprocess_vast.sh --modalities vision_cap,video,audio,subtitle,vast_cap
```

Rows missing optional modalities are still cached. Training/eval datasets filter rows according to the modalities requested at runtime.

## Train

Training always uses:

```text
anchor = vision_cap
primary candidate = video
extra modalities = audio/subtitle/vast_cap subset
```

Run on 8 GPUs:

```bash
bash scripts_sh/train_vast.sh --extra audio
```

Other modality combinations:

```bash
bash scripts_sh/train_vast.sh --extra subtitle
bash scripts_sh/train_vast.sh --extra audio,subtitle
bash scripts_sh/train_vast.sh --extra audio,subtitle,vast_cap
```

Override GPU count:

```bash
NPROC_PER_NODE=4 bash scripts_sh/train_vast.sh --extra audio
```

Training logs `train/loss` and downstream eval metrics to wandb when `wandb.enabled: true`. If `wandb` is not already installed, the script fails with a dependency message and does not install anything.

## Evaluate

Evaluation explicitly chooses query, target, and auxiliary modalities:

```bash
bash scripts_sh/eval_msrvtt.sh \
  --checkpoint_dir /Users/bytedance/Qwen2.5Omni/outputs/vast_lora_volume/step_0000500 \
  --query vision_cap \
  --target video \
  --aux audio
```

`vast_cap` is only valid for VAST. MSRVTT, DiDeMo, and VATEX configs reject it.

## Loss

The default score matches the provided loss style:

```python
logits = volume_scale / (gram_volume + eps)
```

The loss is symmetric over local anchors against global candidates and global anchors against local candidates, using DDP `all_gather` with gradients.

## Environment

The shell scripts activate:

```bash
eval "$(conda shell.bash hook)"
conda activate cl
```

No script installs or modifies dependencies.

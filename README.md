# Qwen2.5-Omni Gram-Volume Retrieval

This project fine-tunes the Qwen2.5-Omni Thinker with LoRA for multimodal embedding and retrieval. The multimodal encoders are frozen. The trainable parts are LoRA adapters on `q_proj/k_proj/v_proj/o_proj` and an optional projection head.

## Layout

```text
configs/      Preprocess, train, and eval configs
scripts/      Python entrypoints
scripts_sh/   Shell wrappers
src/          Python package
data_cache/   Generated processor-input cache
outputs/      Adapters, projection heads, and metrics
```

## Projection Modes

Set `projection.mode` in the train/eval config:

```yaml
projection:
  mode: shared      # none, shared, or per_modality
  embed_dim: 1024
  normalize: true
```

`none` uses the final hidden state directly. `shared` uses one linear head for all modalities. `per_modality` uses one linear head per modality.

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
source /Users/bytedance/.pyenv/versions/3.10.15/envs/env310/bin/activate
```

No script installs or modifies dependencies.

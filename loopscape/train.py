from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from loopscape.adapters import configure_trainable
from loopscape.audio import read_stereo
from loopscape.checkpoint import TEXT_ENCODER_REVISION, load_base, read_adapter, save_adapter
from loopscape.config import load_config
from loopscape.losses import flow_and_boundary_loss
from loopscape.manifest import manifest_sha256, records


UPSTREAM_SHA = "164420faa2b3f8a28bb013ccde7006422a616fc7"


class PreparedLoops(Dataset):
    def __init__(self, manifest: Path, split: str, duration_s: float):
        self.items = [item for item in records(manifest) if item.get("split") == split]
        if not self.items:
            raise ValueError(f"No {split} records in {manifest}")
        if any(item.get("ai_training_allowed") is not True for item in self.items):
            raise ValueError("Training manifest contains an unapproved source")
        self.expected_samples = round(duration_s * 44_100)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        audio = read_stereo(item["path"])
        if audio.shape[-1] != self.expected_samples:
            raise ValueError(f"Wrong audio length: {item['path']}")
        return audio, str(item["caption"])


def deterministic_batch(dataset: PreparedLoops, micro_index: int, batch_size: int, seed: int):
    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    epoch, batch_in_epoch = divmod(micro_index, batches_per_epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(len(dataset), generator=generator)
    indices = permutation[batch_in_epoch * batch_size : (batch_in_epoch + 1) * batch_size]
    items = [dataset[int(index)] for index in indices]
    return torch.stack([audio for audio, _ in items]), [caption for _, caption in items]


def train(config_path: Path, args: argparse.Namespace) -> None:
    config = load_config(config_path)
    settings = config["train"]
    if args.output_dir:
        settings["output_dir"] = str(args.output_dir)
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.boundary_weight is not None:
        settings["boundary_weight"] = args.boundary_weight
    if int(settings["max_steps"]) <= 0 or float(settings["boundary_weight"]) < 0:
        raise ValueError("max_steps must be positive and boundary_weight nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Training needs a CUDA GPU; use --help to inspect options")
    device = torch.device("cuda")
    if settings["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; select fp16 in the config")
    if settings["precision"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16, or fp32")

    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, vae = load_base(config["model"]["repo_id"], config["model"]["revision"], device)
    projection_count = configure_trainable(
        model, settings["mode"], int(settings["lora_rank"]), float(settings["lora_alpha"])
    )
    model.to(device)
    model.train()
    model.text_encoder.eval()
    vae.eval()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = PreparedLoops(
        Path(settings["manifest"]), "train", float(config["model"]["duration_s"])
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(settings["learning_rate"]),
    )
    use_fp16 = settings["precision"] == "fp16"
    use_autocast = settings["precision"] in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if settings["precision"] == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    micro_index = 0
    latent_frames = None
    if args.resume:
        metadata = read_adapter(model, output_dir)
        if metadata["manifest_sha256"] != manifest_sha256(settings["manifest"]):
            raise ValueError("Training manifest changed since the checkpoint")
        if metadata["config"]["model"] != config["model"]:
            raise ValueError("Base model settings changed since the checkpoint")
        saved_settings = metadata["config"]["train"]
        allowed_changes = {"max_steps", "output_dir", "save_every"}
        changed = [
            key for key in saved_settings
            if key not in allowed_changes and saved_settings[key] != settings.get(key)
        ]
        if changed:
            raise ValueError(f"Training settings changed since the checkpoint: {changed}")
        state = torch.load(output_dir / "state.pt", map_location="cpu", weights_only=False)
        if int(state["step"]) != int(metadata["step"]):
            raise ValueError("Optimizer and adapter checkpoints have different steps")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        completed = int(state["step"])
        micro_index = int(state["micro_index"])
        latent_frames = int(metadata["latent_frames"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        log_path = output_dir / "train.jsonl"
        if log_path.exists():
            retained = [
                item for item in records(log_path) if int(item["step"]) <= completed
            ]
            with log_path.open("w", encoding="utf-8") as stream:
                for item in retained:
                    stream.write(json.dumps(item) + "\n")
    manifest_hash = manifest_sha256(settings["manifest"])
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={settings['mode']} trainable={trainable_count} LoRA_layers={projection_count}")
    print(f"GPU={torch.cuda.get_device_name()} records={len(dataset)} start_step={completed}")
    started = time.monotonic()
    accumulated = 0
    optimizer.zero_grad(set_to_none=True)

    while completed < int(settings["max_steps"]):
        audio, prompts = deterministic_batch(
            dataset, micro_index, int(settings["batch_size"]), seed
        )
        micro_index += 1
        audio = audio.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=use_autocast
        ):
            latents = vae.encode(audio).latent_dist.sample().transpose(1, 2)
        if latent_frames is None:
            latent_frames = int(latents.shape[1])
            print(f"latent_frames={latent_frames} duration_s={config['model']['duration_s']}")
        elif latent_frames != latents.shape[1]:
            raise ValueError("VAE produced inconsistent latent lengths")
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            loss, flow, boundary = flow_and_boundary_loss(
                model,
                latents,
                list(prompts),
                float(config["model"]["duration_s"]),
                float(settings["boundary_weight"]),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at optimizer step {completed}")
        scaler.scale(loss / int(settings["grad_accum"])).backward()
        accumulated += 1
        if accumulated < int(settings["grad_accum"]):
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if use_fp16 and scaler.get_scale() < scale_before:
            accumulated = 0
            continue
        completed += 1
        accumulated = 0
        log_item = {
            "step": completed,
            "loss": float(loss.detach()),
            "flow_loss": float(flow),
            "boundary_loss": float(boundary),
            "elapsed_s": round(time.monotonic() - started, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        }
        with (output_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log_item) + "\n")
        if completed == 1 or completed % 20 == 0:
            print(json.dumps(log_item))
        if completed % int(settings["save_every"]) == 0 or completed == int(settings["max_steps"]):
            metadata = {
                "config": config,
                "step": completed,
                "latent_frames": latent_frames,
                "manifest_sha256": manifest_hash,
                "upstream_sha": UPSTREAM_SHA,
                "text_encoder_revision": TEXT_ENCODER_REVISION,
                "trainable_parameters": trainable_count,
                "gpu_name": torch.cuda.get_device_name(),
                "peak_vram_gb": log_item["peak_vram_gb"],
                "elapsed_s": log_item["elapsed_s"],
            }
            save_adapter(model, output_dir, metadata)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": completed,
                    "micro_index": micro_index,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                },
                output_dir / "state.pt",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune baseline or loop-aware TangoFlux")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--boundary-weight", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.config, args)


if __name__ == "__main__":
    main()

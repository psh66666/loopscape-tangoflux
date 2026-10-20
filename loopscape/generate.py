from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from loopscape.adapters import configure_trainable
from loopscape.audio import SAMPLE_RATE, audio_metrics
from loopscape.checkpoint import load_base, read_adapter
from loopscape.config import load_config
from loopscape.manifest import records


def declick_same_length(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """A fixed-length inference-only control for isolated sample jumps."""
    width = min(sample_rate // 20, audio.shape[-1] // 10)
    if width < 2:
        return audio
    result = audio.clone()
    delta = result[:, -1] - result[:, 0]
    ramp = torch.linspace(0.0, 1.0, width, dtype=result.dtype)
    result[:, :width] += 0.5 * delta[:, None] * (1.0 - ramp)
    result[:, -width:] -= 0.5 * delta[:, None] * ramp
    return result


def crossfade_same_length(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Blend both sides toward one common boundary without changing duration."""
    width = min(sample_rate // 4, audio.shape[-1] // 10)
    if width < 2:
        return audio
    result = audio.clone()
    head = audio[:, :width]
    reversed_tail = audio[:, -width:].flip(-1)
    bridge = (head + reversed_tail) * 0.5
    boundary_weight = torch.linspace(1.0, 0.0, width, dtype=audio.dtype)
    result[:, :width] = head * (1.0 - boundary_weight) + bridge * boundary_weight
    reversed_new_tail = reversed_tail * (1.0 - boundary_weight) + bridge * boundary_weight
    result[:, -width:] = reversed_new_tail.flip(-1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched environmental sound loops")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="Omit for zero-shot pretrained model")
    parser.add_argument("--prompts", type=Path, default=Path("configs/eval_prompts.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=1200)
    postprocessing = parser.add_mutually_exclusive_group()
    postprocessing.add_argument("--declick", action="store_true")
    postprocessing.add_argument("--crossfade", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Generation needs a CUDA GPU")
    config = load_config(args.config)
    device = torch.device("cuda")
    model, vae = load_base(config["model"]["repo_id"], config["model"]["revision"], device)
    metadata = None
    if args.run_dir:
        metadata = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
        saved = metadata["config"]
        if saved["model"] != config["model"] or saved["train"]["mode"] != config["train"]["mode"]:
            raise ValueError("Generation config differs from the training checkpoint")
        configure_trainable(
            model,
            saved["train"]["mode"],
            int(saved["train"]["lora_rank"]),
            float(saved["train"]["lora_alpha"]),
        )
        model.to(device)
        read_adapter(model, args.run_dir)
        model.audio_seq_len = int(metadata["latent_frames"])
    else:
        duration_samples = round(float(config["model"]["duration_s"]) * SAMPLE_RATE)
        with torch.inference_mode():
            model.audio_seq_len = int(
                vae.encode(torch.zeros(1, 2, duration_samples, device=device)).latent_dist.mean.shape[-1]
            )
    model.eval()
    vae.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, item in enumerate(records(args.prompts)):
        prompt_id = str(item["id"])
        if not prompt_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Unsafe prompt id: {prompt_id}")
        seed = args.seed + index
        with torch.inference_mode():
            latents = model.inference_flow(
                str(item["caption"]),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                duration=float(config["model"]["duration_s"]),
                seed=seed,
            )
            audio = vae.decode(latents.transpose(2, 1)).sample[0].float().cpu()
        audio = audio[:, : round(float(config["model"]["duration_s"]) * SAMPLE_RATE)]
        if args.declick:
            audio = declick_same_length(audio)
        elif args.crossfade:
            audio = crossfade_same_length(audio)
        filename = f"{prompt_id}__seed{seed}.wav"
        torchaudio.save(str(args.output_dir / filename), audio, SAMPLE_RATE)
        results.append(
            {"id": prompt_id, "caption": item["caption"], "seed": seed, "file": filename, "metrics": audio_metrics(audio)}
        )
    summary = {
        "config": config,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "training_step": metadata["step"] if metadata else 0,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "declick": args.declick,
        "crossfade": args.crossfade,
        "samples": results,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"generated={len(results)} output={args.output_dir}")


if __name__ == "__main__":
    main()

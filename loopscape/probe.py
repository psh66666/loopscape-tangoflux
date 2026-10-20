from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from loopscape.audio import SAMPLE_RATE, audio_metrics, read_stereo
from loopscape.checkpoint import load_vae
from loopscape.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the VAE preserves a known loop seam")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vae_probe"))
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(config["model"]["repo_id"], config["model"]["revision"], device)
    audio = read_stereo(args.audio)
    with torch.inference_mode():
        encoded = vae.encode(audio.unsqueeze(0).to(device)).latent_dist.mode()
        reconstructed = vae.decode(encoded).sample[0].float().cpu()
    reconstructed = reconstructed[:, : audio.shape[-1]]
    if reconstructed.shape[-1] < audio.shape[-1]:
        raise ValueError("VAE reconstruction is shorter than the input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_dir / "input.wav"), audio, SAMPLE_RATE)
    torchaudio.save(str(args.output_dir / "reconstructed.wav"), reconstructed, SAMPLE_RATE)
    result = {
        "input": audio_metrics(audio),
        "reconstructed": audio_metrics(reconstructed),
        "latent_frames": int(encoded.shape[-1]),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

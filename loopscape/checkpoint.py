from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderOobleck
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from tangoflux.model import TangoFlux

from loopscape.adapters import load_trainable_state, trainable_state


TEXT_ENCODER_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


def load_base(repo_id: str, revision: str, device: torch.device):
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["config.json", "tangoflux.safetensors", "vae.safetensors"],
        )
    )
    with (snapshot / "config.json").open(encoding="utf-8") as stream:
        model_config = json.load(stream)
    text_encoder = snapshot_download(
        repo_id="google/flan-t5-large",
        revision=TEXT_ENCODER_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    model = TangoFlux(model_config, text_encoder_dir=text_encoder)
    model.load_state_dict(load_file(str(snapshot / "tangoflux.safetensors")), strict=False)
    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(str(snapshot / "vae.safetensors")))
    model.to(device)
    vae.to(device)
    vae.requires_grad_(False).eval()
    return model, vae


def load_vae(repo_id: str, revision: str, device: torch.device):
    snapshot = Path(
        snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=["vae.safetensors"])
    )
    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(str(snapshot / "vae.safetensors")))
    return vae.to(device).requires_grad_(False).eval()


def save_adapter(model: torch.nn.Module, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), str(output_dir / "adapter.safetensors"))
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_adapter(model: torch.nn.Module, output_dir: Path) -> dict:
    metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    load_trainable_state(model, load_file(str(output_dir / "adapter.safetensors")))
    return metadata

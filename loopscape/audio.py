from __future__ import annotations

import math
from pathlib import Path

import torch
import torchaudio


SAMPLE_RATE = 44_100


def read_stereo(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, original_rate = torchaudio.load(str(path))
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if original_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, original_rate, sample_rate)
    return audio.float()


def make_continuous_loop(
    audio: torch.Tensor, duration_s: float, overlap_s: float, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """Join the end of a continuous source interval to its own beginning."""
    target = round(duration_s * sample_rate)
    overlap = round(overlap_s * sample_rate)
    if overlap <= 1 or overlap >= target // 2:
        raise ValueError("overlap must be positive and shorter than half the loop")
    needed = target + overlap
    if audio.shape[-1] < needed:
        raise ValueError(f"source needs at least {needed} samples")
    start = (audio.shape[-1] - needed) // 2
    source = audio[..., start : start + needed]
    result = source[..., :target].clone()
    ramp = torch.linspace(0.0, 1.0, overlap, dtype=result.dtype, device=result.device)
    result[..., :overlap] = (
        source[..., target : target + overlap] * (1.0 - ramp)
        + source[..., :overlap] * ramp
    )
    return result


def audio_metrics(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> dict[str, float]:
    """Cheap seam checks; perceptual listening remains necessary."""
    if audio.ndim == 2:
        mono = audio.mean(dim=0)
    elif audio.ndim == 1:
        mono = audio
    else:
        raise ValueError("audio must have shape [samples] or [channels, samples]")
    if mono.numel() < sample_rate:
        raise ValueError("audio needs at least one second")
    difference = mono[1:] - mono[:-1]
    seam_jump = (mono[0] - mono[-1]).abs()
    interior_jump = difference.abs().median().clamp_min(1e-8)
    half_second = min(sample_rate // 2, mono.numel() // 4)
    edges = torch.cat((mono[:half_second], mono[-half_second:]))
    center_start = (mono.numel() - 2 * half_second) // 2
    center = mono[center_start : center_start + 2 * half_second]
    edge_rms = edges.square().mean().sqrt()
    center_rms = center.square().mean().sqrt().clamp_min(1e-8)

    # Compare spectral flux at the join with the median flux inside the clip.
    window = min(2048, 2 ** int(math.log2(max(512, sample_rate // 20))))
    hop = window // 4
    extended = torch.cat((mono[-window:], mono, mono[:window]))
    spectrum = torch.stft(
        extended, n_fft=window, hop_length=hop, return_complex=True, center=False
    ).abs().clamp_min(1e-7).log1p()
    flux = (spectrum[:, 1:] - spectrum[:, :-1]).abs().mean(dim=0)
    seam_frame = (window + mono.numel()) // hop
    seam_flux = flux[max(0, seam_frame - 2) : seam_frame + 2].mean()
    interior_flux = flux[4:-4].median().clamp_min(1e-8)
    return {
        "seam_jump": float(seam_jump),
        "seam_jump_ratio": float(seam_jump / interior_jump),
        "seam_spectral_flux_ratio": float(seam_flux / interior_flux),
        "edge_to_center_rms": float(edge_rms / center_rms),
        "rms": float(mono.square().mean().sqrt()),
        "peak": float(mono.abs().max()),
    }

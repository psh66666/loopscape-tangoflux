from __future__ import annotations

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.nn import functional as F


def boundary_continuity_loss(clean_estimate: torch.Tensor) -> torch.Tensor:
    if clean_estimate.shape[1] < 4:
        raise ValueError("At least four latent frames are needed")
    first_step = clean_estimate[:, 1] - clean_estimate[:, 0]
    last_step = clean_estimate[:, -1] - clean_estimate[:, -2]
    seam_step = clean_estimate[:, 0] - clean_estimate[:, -1]
    return F.mse_loss(seam_step, (first_step + last_step) * 0.5) + 0.5 * F.mse_loss(
        first_step, last_step
    )


def flow_and_boundary_loss(
    model, latents: torch.Tensor, prompts: list[str], duration_s: float, boundary_weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, frames, _ = latents.shape
    device = latents.device
    encoder_hidden_states, encoder_mask = model.encode_text(prompts)
    mask = encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
    pooled = torch.nanmean(
        torch.where(mask, encoder_hidden_states, torch.full_like(encoder_hidden_states, float("nan"))),
        dim=1,
    )
    pooled_projection = model.fc(pooled)
    durations = torch.full((batch_size,), duration_s, device=device)
    duration_hidden_states = model.encode_duration(durations)
    encoder_hidden_states = torch.cat((encoder_hidden_states, duration_hidden_states), dim=1)
    txt_ids = torch.zeros(batch_size, encoder_hidden_states.shape[1], 3, device=device)
    audio_ids = torch.arange(frames, device=device).view(1, frames, 1).repeat(batch_size, 1, 3)

    noise = torch.randn_like(latents)
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=batch_size,
        logit_mean=0,
        logit_std=1,
        mode_scale=None,
    )
    indices = (u * model.noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = model.noise_scheduler_copy.timesteps[indices].to(device=device)
    sigmas = model.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy = (1.0 - sigmas) * latents + sigmas * noise
    prediction = model.transformer(
        hidden_states=noisy,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projection,
        img_ids=audio_ids,
        txt_ids=txt_ids,
        guidance=None,
        timestep=timesteps / 1000,
        return_dict=False,
    )[0]
    target = noise - latents
    flow = F.mse_loss(prediction.float(), target.float())
    if boundary_weight:
        clean_estimate = noisy.float() - sigmas.float() * prediction.float()
        boundary = boundary_continuity_loss(clean_estimate)
    else:
        boundary = flow.new_zeros(())
    return flow + boundary_weight * boundary, flow.detach(), boundary.detach()

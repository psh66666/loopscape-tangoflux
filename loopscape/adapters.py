from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ATTENTION_PROJECTIONS = {
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scale


def install_lora(transformer: nn.Module, rank: int, alpha: float) -> int:
    count = 0
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in ATTENTION_PROJECTIONS):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = transformer.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        count += 1
    if count == 0:
        raise RuntimeError("No Flux attention projections were found for LoRA")
    return count


class BoundaryContext(nn.Module):
    """Local residual context; circular and zero-padded modes share parameters."""

    def __init__(self, channels: int = 64, kind: str = "circular"):
        super().__init__()
        if kind not in {"circular", "plain_adapter"}:
            raise ValueError(f"Unknown boundary context: {kind}")
        self.kind = kind
        self.depthwise = nn.Conv1d(channels, channels, 5, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels_first = hidden_states.transpose(1, 2)
        padding = "circular" if self.kind == "circular" else "constant"
        context = F.pad(channels_first, (2, 2), mode=padding)
        context = self.pointwise(F.gelu(self.depthwise(context)))
        return hidden_states + self.scale * context.transpose(1, 2)


class TransformerWithBoundaryContext(nn.Module):
    def __init__(self, inner: nn.Module, kind: str, channels: int = 64):
        super().__init__()
        self.inner = inner
        self.context = BoundaryContext(channels, kind)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return self.inner(hidden_states=self.context(hidden_states), **kwargs)


def configure_trainable(model: nn.Module, mode: str, rank: int, alpha: float) -> int:
    model.requires_grad_(False)
    count = install_lora(model.transformer, rank, alpha)
    if mode in {"circular", "plain_adapter"}:
        model.transformer = TransformerWithBoundaryContext(
            model.transformer, mode, channels=model.in_channels
        )
    elif mode != "baseline":
        raise ValueError(f"Unknown mode: {mode}")
    return count


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(f"Adapter keys differ: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.load_state_dict(state, strict=False)

# src/snn/lif_neuron.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class LIFConfig:
    threshold: float = 1.0
    reset_value: float = 0.0
    tau_mem: float = 20.0
    dt: float = 1.0
    reset_mode: str = "hard"  # "hard" or "soft"
    surrogate_scale: float = 10.0
    learn_threshold: bool = False
    learn_tau: bool = False
    detach_reset: bool = True
    refractory_steps: int = 0
    membrane_clamp: Optional[Tuple[float, float]] = None


class SurrogateSpike(torch.autograd.Function):
    """
    Binary spike in forward pass and smooth surrogate gradient in backward pass.
    """

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: Tensor, scale: Tensor):
        ctx.save_for_backward(membrane, threshold, scale)
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold, scale = ctx.saved_tensors
        x = scale * (membrane - threshold)
        grad = 1.0 / (1.0 + x.abs()).pow(2)
        return grad_output * grad, None, None


spike_fn = SurrogateSpike.apply


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer with:
    - batch support
    - optional learnable threshold and tau
    - hard/soft reset
    - refractory period
    - single-step and sequence mode
    - spike statistics helpers

    Input shapes:
      - step mode:     [batch, neurons]
      - sequence mode: [batch, time, neurons]

    Outputs:
      - step mode:     spikes, membrane
      - sequence mode: spikes_seq, membrane
    """

    def __init__(self, config: Optional[LIFConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = LIFConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword arguments, not both.")

        if config.reset_mode not in {"hard", "soft"}:
            raise ValueError("reset_mode must be either 'hard' or 'soft'.")

        if config.tau_mem <= 0:
            raise ValueError("tau_mem must be > 0.")

        if config.dt <= 0:
            raise ValueError("dt must be > 0.")

        if config.refractory_steps < 0:
            raise ValueError("refractory_steps must be >= 0.")

        self.config = config

        threshold = torch.tensor(float(config.threshold))
        if config.learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

        if config.learn_tau:
            self.log_tau_mem = nn.Parameter(torch.log(torch.tensor(float(config.tau_mem))))
        else:
            self.register_buffer("tau_mem", torch.tensor(float(config.tau_mem)))

        self.register_buffer("reset_value", torch.tensor(float(config.reset_value)))
        self.register_buffer("dt", torch.tensor(float(config.dt)))
        self.register_buffer("surrogate_scale", torch.tensor(float(config.surrogate_scale)))

    @property
    def tau_value(self) -> Tensor:
        if hasattr(self, "log_tau_mem"):
            return torch.exp(self.log_tau_mem)
        return self.tau_mem

    @property
    def beta(self) -> Tensor:
        # Discrete decay factor
        return torch.exp(-self.dt / self.tau_value)

    def extra_repr(self) -> str:
        return (
            f"threshold={float(self.threshold.detach().cpu()):.4f}, "
            f"tau_mem={float(self.tau_value.detach().cpu()):.4f}, "
            f"dt={float(self.dt.detach().cpu()):.4f}, "
            f"reset_mode='{self.config.reset_mode}', "
            f"refractory_steps={self.config.refractory_steps}, "
            f"learn_threshold={self.config.learn_threshold}, "
            f"learn_tau={self.config.learn_tau}"
        )

    def init_state(
        self,
        batch_size: int,
        num_neurons: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            membrane: [batch, neurons]
            refractory_counter: [batch, neurons] or None
        """
        membrane = torch.zeros(
            batch_size,
            num_neurons,
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )

        refractory_counter = None
        if self.config.refractory_steps > 0:
            refractory_counter = torch.zeros(
                batch_size,
                num_neurons,
                device=device,
                dtype=torch.long,
            )

        return membrane, refractory_counter

    def _apply_clamp(self, membrane: Tensor) -> Tensor:
        if self.config.membrane_clamp is None:
            return membrane
        min_v, max_v = self.config.membrane_clamp
        return torch.clamp(membrane, min=min_v, max=max_v)

    def forward_step(
        self,
        input_current: Tensor,
        membrane: Tensor,
        refractory_counter: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Single timestep update.

        Args:
            input_current: [batch, neurons]
            membrane: [batch, neurons]
            refractory_counter: [batch, neurons] or None

        Returns:
            spikes: [batch, neurons]
            membrane: [batch, neurons]
            refractory_counter: updated counter or None
        """
        if input_current.dim() != 2:
            raise ValueError("input_current must have shape [batch, neurons].")

        beta = self.beta.to(device=membrane.device, dtype=membrane.dtype)
        threshold = self.threshold.to(device=membrane.device, dtype=membrane.dtype)
        scale = self.surrogate_scale.to(device=membrane.device, dtype=membrane.dtype)

        if refractory_counter is not None:
            active_mask = (refractory_counter <= 0).to(membrane.dtype)
        else:
            active_mask = torch.ones_like(membrane)

        membrane = beta * membrane + input_current * active_mask
        membrane = self._apply_clamp(membrane)

        spikes = spike_fn(membrane, threshold, scale)

        if refractory_counter is not None:
            refractory_counter = torch.clamp(refractory_counter - 1, min=0)
            fired = spikes.bool()
            refractory_counter = torch.where(
                fired,
                torch.full_like(refractory_counter, self.config.refractory_steps),
                refractory_counter,
            )
            spikes = spikes * (refractory_counter == self.config.refractory_steps).to(spikes.dtype)

        if self.config.detach_reset:
            reset_spikes = spikes.detach()
        else:
            reset_spikes = spikes

        if self.config.reset_mode == "hard":
            membrane = torch.where(
                reset_spikes.bool(),
                torch.full_like(membrane, float(self.reset_value.item())),
                membrane,
            )
        else:
            membrane = membrane - reset_spikes * threshold

        membrane = self._apply_clamp(membrane)

        return spikes, membrane, refractory_counter

    def forward_sequence(
        self,
        input_current: Tensor,
        membrane: Tensor,
        refractory_counter: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Sequence mode.

        Args:
            input_current: [batch, time, neurons]
            membrane: [batch, neurons]
            refractory_counter: [batch, neurons] or None

        Returns:
            spikes_seq: [batch, time, neurons]
            membrane: final membrane [batch, neurons]
            refractory_counter: final counter or None
        """
        if input_current.dim() != 3:
            raise ValueError("input_current must have shape [batch, time, neurons].")

        batch_size, time_steps, num_neurons = input_current.shape
        spikes_over_time = []

        for t in range(time_steps):
            spikes, membrane, refractory_counter = self.forward_step(
                input_current[:, t, :],
                membrane,
                refractory_counter,
            )
            spikes_over_time.append(spikes.unsqueeze(1))

        spikes_seq = torch.cat(spikes_over_time, dim=1)
        return spikes_seq, membrane, refractory_counter

    def forward(
        self,
        input_current: Tensor,
        membrane: Tensor,
        refractory_counter: Optional[Tensor] = None,
    ):
        """
        Dispatches to step or sequence mode based on input dimension.
        """
        if input_current.dim() == 2:
            return self.forward_step(input_current, membrane, refractory_counter)
        if input_current.dim() == 3:
            return self.forward_sequence(input_current, membrane, refractory_counter)
        raise ValueError("input_current must be 2D or 3D.")

    @torch.no_grad()
    def firing_rate(self, spikes: Tensor) -> float:
        return float(spikes.float().mean().item())

    @torch.no_grad()
    def spike_density(self, spikes: Tensor) -> float:
        return float(spikes.float().mean().item())

    @torch.no_grad()
    def silent_neuron_ratio(self, spikes: Tensor) -> float:
        """
        Fraction of neurons that never fired.
        Works for [B, N] or [B, T, N].
        """
        if spikes.dim() == 2:
            neuron_activity = spikes.sum(dim=0)
        elif spikes.dim() == 3:
            neuron_activity = spikes.sum(dim=(0, 1))
        else:
            raise ValueError("spikes must be 2D or 3D.")

        silent = (neuron_activity == 0).float()
        return float(silent.mean().item())

    @torch.no_grad()
    def event_density(self, spikes: Tensor) -> float:
        """
        Average number of events per timestep.
        """
        if spikes.dim() == 2:
            return float(spikes.float().sum(dim=1).mean().item())
        if spikes.dim() == 3:
            return float(spikes.float().sum(dim=2).mean().item())
        raise ValueError("spikes must be 2D or 3D.")

    @torch.no_grad()
    def membrane_stability(self, membrane_history: Tensor) -> float:
        """
        Variance of membrane potential.
        Works on a full history tensor such as [B, T, N].
        """
        return float(membrane_history.float().var(unbiased=False).item())

    @torch.no_grad()
    def summarize_spikes(self, spikes: Tensor) -> dict:
        return {
            "firing_rate": self.firing_rate(spikes),
            "spike_density": self.spike_density(spikes),
            "silent_neuron_ratio": self.silent_neuron_ratio(spikes),
            "event_density": self.event_density(spikes),
        }
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SpikeEncoderConfig:
    """
    Global configuration shared by all encoders.
    """

    num_steps: int = 50

    min_value: float = 0.0
    max_value: float = 1.0

    threshold: float = 0.5

    normalize_input: bool = True

    deterministic: bool = False

    seed: Optional[int] = None


# ============================================================
# BASE ENCODER
# ============================================================

class BaseSpikeEncoder(nn.Module, ABC):
    """
    Abstract base class for all spike encoders.

    Input:
        [B, F]

    Output:
        [B, T, F]
    """

    def __init__(self, config: SpikeEncoderConfig):
        super().__init__()

        self.config = config

        if config.num_steps <= 0:
            raise ValueError(
                "num_steps must be > 0"
            )

        if config.seed is not None:
            torch.manual_seed(config.seed)

    @property
    def num_steps(self) -> int:
        return self.config.num_steps

    def normalize(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Normalize into [0,1].
        """

        if not self.config.normalize_input:
            return x

        x_min = x.min()
        x_max = x.max()

        denom = x_max - x_min

        if denom.abs() < 1e-8:
            return torch.zeros_like(x)

        return (x - x_min) / denom

    def validate_input(
        self,
        x: Tensor,
    ) -> None:
        """
        Validate shape and dtype.
        """

        if not torch.is_tensor(x):
            raise TypeError(
                "Input must be torch.Tensor"
            )

        if x.dim() != 2:
            raise ValueError(
                "Expected shape [batch, features]"
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "Input must be floating point"
            )

    @torch.no_grad()
    def spike_rate(
        self,
        spikes: Tensor,
    ) -> float:
        """
        Global spike rate.
        """

        return float(
            spikes.float().mean().item()
        )

    @torch.no_grad()
    def spike_density(
        self,
        spikes: Tensor,
    ) -> float:
        """
        Ds(t) from paper.
        """

        return float(
            spikes.float().mean().item()
        )

    @torch.no_grad()
    def silent_feature_ratio(
        self,
        spikes: Tensor,
    ) -> float:
        """
        Features that never spike.
        """

        feature_activity = spikes.sum(
            dim=(0, 1)
        )

        silent = (
            feature_activity == 0
        ).float()

        return float(
            silent.mean().item()
        )

    @torch.no_grad()
    def summary(
        self,
        spikes: Tensor,
    ) -> Dict[str, float]:

        return {
            "spike_rate":
                self.spike_rate(spikes),

            "spike_density":
                self.spike_density(spikes),

            "silent_feature_ratio":
                self.silent_feature_ratio(
                    spikes
                ),
        }

    @abstractmethod
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Convert analog input into spike trains.

        Input:
            [B,F]

        Output:
            [B,T,F]
        """
        pass


# ============================================================
# RATE ENCODER
# ============================================================

class RateEncoder(BaseSpikeEncoder):
    """
    Rate Encoding

    Higher input values produce higher spike probability.

    Example:

        x = 0.9
        -> many spikes

        x = 0.1
        -> few spikes

    Input:
        [B, F]

    Output:
        [B, T, F]
    """

    def __init__(
        self,
        config: SpikeEncoderConfig,
    ):
        super().__init__(config)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        self.validate_input(x)

        x = self.normalize(x)

        batch_size, num_features = x.shape

        probs = x.unsqueeze(1).expand(
            batch_size,
            self.num_steps,
            num_features,
        )

        if self.config.deterministic:

            threshold = 0.5

            spikes = (
                probs > threshold
            ).float()

        else:

            random_tensor = torch.rand(
                batch_size,
                self.num_steps,
                num_features,
                device=x.device,
                dtype=x.dtype,
            )

            spikes = (
                random_tensor < probs
            ).float()

        return spikes
    

# ============================================================
# TEMPORAL ENCODER
# ============================================================

class TemporalEncoder(BaseSpikeEncoder):
    """
    Temporal Encoding

    Larger values spike earlier.

    Smaller values spike later.

    Example:

        x = 1.0
        -> spike at t=0

        x = 0.5
        -> spike around middle

        x = 0.0
        -> spike near end

    Input:
        [B, F]

    Output:
        [B, T, F]
    """

    def __init__(
        self,
        config: SpikeEncoderConfig,
    ):
        super().__init__(config)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        self.validate_input(x)

        x = self.normalize(x)

        batch_size, num_features = x.shape

        spikes = torch.zeros(
            batch_size,
            self.num_steps,
            num_features,
            device=x.device,
            dtype=x.dtype,
        )

        # Convert value -> spike time
        spike_times = (
            (1.0 - x)
            * (self.num_steps - 1)
        ).long()

        spike_times = torch.clamp(
            spike_times,
            min=0,
            max=self.num_steps - 1,
        )

        batch_idx = torch.arange(
            batch_size,
            device=x.device,
        ).unsqueeze(1)

        feature_idx = torch.arange(
            num_features,
            device=x.device,
        ).unsqueeze(0)

        spikes[
            batch_idx,
            spike_times,
            feature_idx,
        ] = 1.0

        return spikes
    
# ============================================================
# THRESHOLD ENCODER
# ============================================================

class ThresholdEncoder(BaseSpikeEncoder):
    """
    Threshold Encoding

    Fires whenever input exceeds a fixed threshold.

    Input:
        [B, F]

    Output:
        [B, T, F]

    Notes:
        The same thresholded spike pattern is repeated
        across all timesteps.
    """

    def __init__(
        self,
        config: SpikeEncoderConfig,
    ):
        super().__init__(config)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        self.validate_input(x)

        x = self.normalize(x)

        batch_size, num_features = x.shape

        base_spikes = (x > self.config.threshold).float()

        spikes = base_spikes.unsqueeze(1).expand(
            batch_size,
            self.num_steps,
            num_features,
        ).contiguous()

        return spikes
    

# ============================================================
# POISSON ENCODER
# ============================================================

class PoissonEncoder(BaseSpikeEncoder):
    """
    Poisson Rate Encoding

    Input value represents firing probability.

    Higher values:
        -> more spikes

    Lower values:
        -> fewer spikes

    Input:
        [B, F]

    Output:
        [B, T, F]

    This is one of the most widely used SNN
    encoding schemes.
    """

    def __init__(
        self,
        config: SpikeEncoderConfig,
    ):
        super().__init__(config)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        self.validate_input(x)

        x = self.normalize(x)

        batch_size, num_features = x.shape

        probs = x.unsqueeze(1).expand(
            batch_size,
            self.num_steps,
            num_features,
        )

        spikes = torch.bernoulli(
            probs
        )

        return spikes.float()
    
# ============================================================
# ENCODER REGISTRY
# ============================================================

ENCODER_REGISTRY = {
    "rate": RateEncoder,
    "temporal": TemporalEncoder,
    "threshold": ThresholdEncoder,
    "poisson": PoissonEncoder,
}

# ============================================================
# FACTORY
# ============================================================

class SpikeEncoderFactory:
    """
    Factory for constructing spike encoders.

    Example
    -------
    encoder = SpikeEncoderFactory.create(
        "temporal",
        config,
    )
    """

    @staticmethod
    def available_encoders():
        return sorted(
            ENCODER_REGISTRY.keys()
        )

    @staticmethod
    def create(
        encoder_type: str,
        config: SpikeEncoderConfig,
    ) -> BaseSpikeEncoder:

        encoder_type = encoder_type.lower()

        if encoder_type not in ENCODER_REGISTRY:

            supported = ", ".join(
                SpikeEncoderFactory.available_encoders()
            )

            raise ValueError(
                f"Unknown encoder '{encoder_type}'. "
                f"Supported encoders: {supported}"
            )

        return ENCODER_REGISTRY[
            encoder_type
        ](config)
    
# ============================================================
# PUBLIC WRAPPER
# ============================================================

class SpikeEncoder(nn.Module):
    """
    Unified interface.

    Example
    -------
    encoder = SpikeEncoder(
        method="temporal",
        num_steps=50,
    )

    spikes = encoder(x)
    """

    def __init__(
        self,
        method: str = "temporal",
        **kwargs,
    ):
        super().__init__()

        config = SpikeEncoderConfig(
            **kwargs
        )

        self.encoder = (
            SpikeEncoderFactory.create(
                method,
                config,
            )
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.encoder(x)

    def summary(
        self,
        spikes: Tensor,
    ):
        return self.encoder.summary(
            spikes
        )
    
# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "SpikeEncoderConfig",

    "BaseSpikeEncoder",

    "RateEncoder",
    "TemporalEncoder",
    "ThresholdEncoder",
    "PoissonEncoder",

    "SpikeEncoderFactory",
    "SpikeEncoder",
]



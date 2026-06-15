from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TemporalEncoderConfig:
    """
    Configuration for temporal feature extraction.
    """

    hidden_dim: int = 128

    variance_window: int = 10

    use_projection: bool = True

    dropout: float = 0.0


# ============================================================
# BASE ENCODER
# ============================================================

class BaseTemporalEncoder(nn.Module, ABC):
    """
    Base class for temporal feature encoders.

    Input:
        [B, T, F]

    Output:
        [B, T, H]
    """

    def __init__(
        self,
        config: TemporalEncoderConfig,
    ):
        super().__init__()

        self.config = config

        if config.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be > 0"
            )

        if config.variance_window <= 0:
            raise ValueError(
                "variance_window must be > 0"
            )

    def validate_input(
        self,
        spikes: Tensor,
    ) -> None:

        if not torch.is_tensor(spikes):
            raise TypeError(
                "Input must be torch.Tensor"
            )

        if spikes.dim() != 3:
            raise ValueError(
                "Expected shape [B,T,F]"
            )

    @torch.no_grad()
    def spike_density(
        self,
        spikes: Tensor,
    ) -> Tensor:
        """
        Ds(t)

        Output:
            [B,T,1]
        """

        return spikes.mean(
            dim=-1,
            keepdim=True,
        )

    @torch.no_grad()
    def temporal_variance(
        self,
        spikes: Tensor,
    ) -> Tensor:
        """
        Vt(t)

        Rolling temporal variance.

        Output:
            [B,T,1]
        """

        B, T, F = spikes.shape

        window = self.config.variance_window

        variance = []

        for t in range(T):

            start = max(
                0,
                t - window + 1,
            )

            chunk = spikes[
                :,
                start:t + 1,
                :
            ]

            var = chunk.var(
                dim=1,
                unbiased=False,
            ).mean(
                dim=-1,
                keepdim=True,
            )

            variance.append(var)

        return torch.stack(
            variance,
            dim=1,
        )

    @torch.no_grad()
    def summary(
        self,
        spikes: Tensor,
    ) -> Dict[str, float]:

        density = self.spike_density(
            spikes
        )

        variance = self.temporal_variance(
            spikes
        )

        return {
            "avg_spike_density":
                density.mean().item(),

            "avg_temporal_variance":
                variance.mean().item(),
        }

    @abstractmethod
    def forward(
        self,
        spikes: Tensor,
    ) -> Tensor:
        pass

# ============================================================
# TEMPORAL ENCODER
# ============================================================

class TemporalEncoder(BaseTemporalEncoder):
    """
    Main temporal encoder.

    Input:
        spikes [B,T,F]

    Output:
        temporal_features [B,T,H]

    Features used:
        - spike density Ds(t)
        - temporal variance Vt(t)
        - raw spike activity
        - learnable projection
    """

    def __init__(
        self,
        input_dim: int,
        config: TemporalEncoderConfig,
    ):
        super().__init__(config)

        self.input_dim = input_dim

        self.extractor = TemporalFeatureExtractor(
        variance_window=config.variance_window
        )

        feature_dim = (
            input_dim     # raw spikes
            + 4
        )

        if config.use_projection:

            self.projection = nn.Sequential(
                nn.Linear(
                    feature_dim,
                    config.hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(
                    config.dropout
                ),
            )

        else:

            self.projection = nn.Identity()

            config.hidden_dim = feature_dim

    def forward(
        self,
        spikes: Tensor,
    ) -> Tensor:

        self.validate_input(
            spikes
        )

        stats = self.extractor(
            spikes
        )

        temporal_features = torch.cat(
            [
                spikes,
                stats,
            ],
            dim =- 1,
        )

        temporal_features = (
            self.projection(
                temporal_features
            )
        )

        return temporal_features
    

# ============================================================
# TEMPORAL FEATURE EXTRACTOR
# ============================================================

class TemporalFeatureExtractor(nn.Module):
    """
    Additional temporal statistics extracted from spike trains.

    Input:
        spikes [B,T,F]

    Output:
        features [B,T,4]

    Features:
        1. Spike Density
        2. Temporal Variance
        3. Rolling Activity
        4. Temporal Smoothness
    """

    def __init__(
        self,
        variance_window: int = 10,
    ):
        super().__init__()

        self.variance_window = variance_window

    def spike_density(
        self,
        spikes: Tensor,
    ) -> Tensor:

        return spikes.mean(
            dim=-1,
            keepdim=True,
        )
    
    

    def temporal_variance(
        self,
        spikes: Tensor,
    ) -> Tensor:

        B, T, F = spikes.shape

        output = []

        for t in range(T):

            start = max(
                0,
                t - self.variance_window + 1,
            )

            window = spikes[
                :,
                start:t + 1,
                :
            ]

            var = window.var(
                dim=1,
                unbiased=False,
            ).mean(
                dim=-1,
                keepdim=True,
            )

            output.append(var)

        return torch.stack(
            output,
            dim=1,
        )

    def rolling_activity(
        self,
        spikes: Tensor,
    ) -> Tensor:

        B, T, F = spikes.shape

        output = []

        for t in range(T):

            start = max(
                0,
                t - self.variance_window + 1,
            )

            window = spikes[
                :,
                start:t + 1,
                :
            ]

            activity = window.mean(
                dim=(1,2)
            ).view(-1,1)

            output.append(activity)

        return torch.stack(
            output,
                dim=1,
        )

    def temporal_smoothness(
        self,
        spikes: Tensor,
    ) -> Tensor:

        diff = torch.diff(
            spikes.float(),
            dim=1,
        )

        smoothness = diff.abs().mean(
            dim=-1,
            keepdim=True,
        )

        pad = torch.zeros(
            spikes.size(0),
            1,
            1,
            device=spikes.device,
            dtype=spikes.dtype,
        )

        return torch.cat(
            [pad, smoothness],
            dim=1,
        )

    def forward(
        self,
        spikes: Tensor,
    ) -> Tensor:

        density = self.spike_density(
            spikes
        )

        variance = self.temporal_variance(
            spikes
        )

        activity = self.rolling_activity(
            spikes
        )

        smoothness = self.temporal_smoothness(
            spikes
        )

        return torch.cat(
            [
                density,
                variance,
                activity,
                smoothness,
            ],
            dim=-1,
        )
    

# ============================================================
# TEMPORAL ENCODER REGISTRY
# ============================================================

TEMPORAL_ENCODER_REGISTRY = {
    "default": TemporalEncoder,
}


# ============================================================
# TEMPORAL ENCODER FACTORY
# ============================================================

class TemporalEncoderFactory:
    """
    Factory for temporal encoders.
    """

    @staticmethod
    def available_encoders():

        return sorted(
            TEMPORAL_ENCODER_REGISTRY.keys()
        )

    @staticmethod
    def create(
        encoder_type: str,
        input_dim: int,
        config: TemporalEncoderConfig,
    ) -> BaseTemporalEncoder:

        encoder_type = encoder_type.lower()

        if encoder_type not in TEMPORAL_ENCODER_REGISTRY:

            available = ", ".join(
                TemporalEncoderFactory.available_encoders()
            )

            raise ValueError(
                f"Unknown temporal encoder '{encoder_type}'. "
                f"Available encoders: {available}"
            )

        return TEMPORAL_ENCODER_REGISTRY[
            encoder_type
        ](
            input_dim=input_dim,
            config=config,
        )
    
# ============================================================
# PUBLIC WRAPPER
# ============================================================

class TemporalRepresentationEncoder(nn.Module):
    """
    Unified public interface.

    Example
    -------

    encoder = TemporalRepresentationEncoder(
        input_dim=128,
        hidden_dim=128,
    )

    features = encoder(spikes)
    """

    def __init__(
        self,
        input_dim: int,
        method: str = "default",
        **kwargs,
    ):
        super().__init__()

        config = TemporalEncoderConfig(
            **kwargs
        )

        self.encoder = (
            TemporalEncoderFactory.create(
                encoder_type=method,
                input_dim=input_dim,
                config=config,
            )
        )

    def forward(
        self,
        spikes: Tensor,
    ) -> Tensor:

        return self.encoder(
            spikes
        )

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
    "TemporalEncoderConfig",

    "BaseTemporalEncoder",

    "TemporalFeatureExtractor",

    "TemporalEncoder",

    "TemporalEncoderFactory",

    "TemporalRepresentationEncoder",
]
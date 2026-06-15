from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class ComplexityScoreConfig:
    """
    Configuration for temporal complexity scoring.

    Paper Equation:
        δ(t) = αDs(t) + βVt(t)

    Notes
    -----
    The raw complexity score is always preserved.

    Optional normalization is provided only for:
    - visualization
    - logging
    - monitoring

    and MUST NOT replace the raw score used by the
    event-trigger policy.
    """

    alpha: float = 0.5
    beta: float = 0.5

    normalize_score: bool = False

    eps: float = 1e-8

    def __post_init__(self):
        object.__setattr__(
            self,
            "alpha",
            float(self.alpha),
        )

        object.__setattr__(
            self,
            "beta",
            float(self.beta),
        )


# ============================================================
# RESULT
# ============================================================

@dataclass
class ComplexityScoreResult:
    """
    Output container for complexity scoring.
    """

    density: Tensor
    variance: Tensor

    raw_score: Tensor

    normalized_score: Optional[Tensor]

    # --------------------------------------------------------
    # Score Statistics
    # --------------------------------------------------------


    @property
    def mean_score(self) -> float:
        return float(self.raw_score.mean().item())
    
    @property
    def score(self) -> Tensor:
        return self.raw_score

    @property
    def max_score(self) -> float:
        return float(self.raw_score.max().item())

    @property
    def min_score(self) -> float:
        return float(self.raw_score.min().item())
    
    @property
    def std_score(self) -> float:
        return float(
            self.raw_score.std(
                unbiased=False
            ).item()
        )

    @property
    def score_variance(self) -> float:
        return float(
            self.raw_score.var(
                unbiased=False
            ).item()
        )

    # --------------------------------------------------------
    # Paper Metrics
    # --------------------------------------------------------

    @property
    def mean_density(self) -> float:
        return float(self.density.mean().item())

    @property
    def mean_variance(self) -> float:
        return float(self.variance.mean().item())

    @property
    def spike_rate(self) -> float:
        """
        Paper Metric:
            Spike Rate

        Since density is derived from spike activity,
        mean density provides a stable estimate of
        global spike activity.
        """
        return float(self.density.mean().item())


# ============================================================
# COMPLEXITY SCORE
# ============================================================

class ComplexityScore(nn.Module):
    """
    Temporal Complexity Score

    Paper Equations
    ---------------

    Ds(t):
        Spike Density

    Vt(t):
        Temporal Variance

    δ(t):
        αDs(t) + βVt(t)

    Input Shape
    -----------
    [B, T, 1]

    Output Shape
    ------------
    [B, T, 1]
    """

    def __init__(
        self,
        config: ComplexityScoreConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self._validate_config()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_config(self) -> None:

        if self.config.alpha < 0:
            raise ValueError(
                f"alpha must be non-negative. "
                f"Got {self.config.alpha}."
            )

        if self.config.beta < 0:
            raise ValueError(
                f"beta must be non-negative. "
                f"Got {self.config.beta}."
            )

        if (self.config.alpha + self.config.beta) <= 0:
            raise ValueError(
                "alpha + beta must be positive."
            )

        if self.config.eps <= 0:
            raise ValueError(
                f"eps must be positive. "
                f"Got {self.config.eps}."
            )

    def _validate_inputs(
        self,
        density: Tensor,
        variance: Tensor,
    ) -> None:

        if not torch.is_tensor(density):
            raise TypeError(
                "density must be a torch.Tensor."
            )

        if not torch.is_tensor(variance):
            raise TypeError(
                "variance must be a torch.Tensor."
            )

        if density.shape != variance.shape:
            raise ValueError(
                "density and variance must have "
                f"identical shapes. "
                f"Got {density.shape} and "
                f"{variance.shape}."
            )

        if density.dim() != 3:
            raise ValueError(
                "Expected input shape [B, T, 1]. "
                f"Got {density.shape}."
            )

        if density.size(-1) != 1:
            raise ValueError(
                "Last dimension must equal 1. "
                f"Got shape {density.shape}."
            )

        if density.device != variance.device:
            raise ValueError(
                "density and variance must be on "
                "the same device."
            )

        if density.dtype != variance.dtype:
            raise ValueError(
                "density and variance must have "
                "identical dtypes."
            )

        if not torch.is_floating_point(density):
            raise TypeError(
                "density must be floating point."
            )

        if not torch.is_floating_point(variance):
            raise TypeError(
                "variance must be floating point."
            )

        if not torch.isfinite(density).all():
            raise ValueError(
                "density contains NaN or Inf."
            )

        if not torch.isfinite(variance).all():
            raise ValueError(
                "variance contains NaN or Inf."
            )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize(
        self,
        score: Tensor,
    ) -> Tensor:
        """
        Reporting-only normalization.

        Never replaces the raw paper score.
        """

        score_min = score.amin()

        score_max = score.amax()

        denom = score_max - score_min

        if torch.abs(denom) < self.config.eps:
            return torch.zeros_like(score)

        return (
            score - score_min
        ) / (
            denom + self.config.eps
        )

    # ========================================================
    # COMPUTE
    # ========================================================

    def _compute_raw_score(
        self,
        density: Tensor,
        variance: Tensor,
    ) -> Tensor:
        """
        Paper Equation:

        δ(t) = αDs(t) + βVt(t)
        """

        return (
            self.config.alpha * density
            + self.config.beta * variance
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        density: Tensor,
        variance: Tensor,
    ) -> ComplexityScoreResult:

        self._validate_inputs(
            density,
            variance,
        )

        raw_score = self._compute_raw_score(
            density,
            variance,
        )

        normalized_score = None

        if self.config.normalize_score:
            normalized_score = self._normalize(
                raw_score
            )

        return ComplexityScoreResult(
            density=density,
            variance=variance,
            raw_score=raw_score,
            normalized_score=normalized_score,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    @torch.no_grad()
    def summary(
        self,
        result: ComplexityScoreResult,
    ) -> dict:

        summary = {
            "mean_complexity_score":
                result.mean_score,

            "max_complexity_score":
                result.max_score,

            "min_complexity_score":
                result.min_score,

            "mean_spike_density":
                result.mean_density,

            "mean_temporal_variance":
                result.mean_variance,

            "spike_rate":
                result.spike_rate,

            "std_complexity_score":
                result.std_score,

            "complexity_score_variance":
                result.score_variance,
        }

        if result.normalized_score is not None:
            summary.update(
                {
                    "mean_normalized_score":
                        float(
                            result.normalized_score
                            .mean()
                            .item()
                        )
                }
            )

        return summary


# ============================================================
# REGISTRY
# ============================================================

COMPLEXITY_SCORE_REGISTRY = {
    "default": ComplexityScore,
}


# ============================================================
# FACTORY
# ============================================================

class ComplexityScoreFactory:
    """
    Factory for complexity score modules.
    """

    @staticmethod
    def available_scorers() -> list[str]:
        return sorted(
            COMPLEXITY_SCORE_REGISTRY.keys()
        )

    @staticmethod
    def create(
        scorer_type: str,
        config: ComplexityScoreConfig,
    ) -> ComplexityScore:

        scorer_type = scorer_type.lower()

        if scorer_type not in COMPLEXITY_SCORE_REGISTRY:

            available = ", ".join(
                ComplexityScoreFactory.available_scorers()
            )

            raise ValueError(
                f"Unknown scorer '{scorer_type}'. "
                f"Available scorers: {available}"
            )

        return COMPLEXITY_SCORE_REGISTRY[
            scorer_type
        ](config)


# ============================================================
# PUBLIC WRAPPER
# ============================================================

class ComplexityScorer(nn.Module):
    """
    Thin public wrapper.

    All logic remains inside ComplexityScore.
    """

    def __init__(
        self,
        config: ComplexityScoreConfig,
        scorer_type: str = "default",
    ) -> None:
        super().__init__()

        self.scorer = ComplexityScoreFactory.create(
            scorer_type=scorer_type,
            config=config,
        )

    def forward(
        self,
        density: Tensor,
        variance: Tensor,
    ) -> ComplexityScoreResult:

        return self.scorer(
            density,
            variance,
        )

    def summary(
        self,
        result: ComplexityScoreResult,
    ) -> dict:

        return self.scorer.summary(
            result
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ComplexityScoreConfig",
    "ComplexityScoreResult",
    "ComplexityScore",
    "ComplexityScoreFactory",
    "ComplexityScorer",
]
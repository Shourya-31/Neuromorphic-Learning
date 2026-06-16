from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn


# ============================================================
# Decision Container
# ============================================================

@dataclass
class TriggerDecision:
    """
    Output container for trigger decisions.

    Attributes
    ----------
    trigger_mask:
        Boolean tensor indicating whether ODE refinement
        should be activated.

    complexity_score:
        Raw complexity score δ(t).

    threshold:
        Threshold θT used for triggering.

    refine_ratio:
        Fraction of samples routed to ODE.

    bypass_ratio:
        Fraction of samples bypassing ODE.
    """

    trigger_mask: torch.Tensor
    complexity_score: torch.Tensor
    threshold: torch.Tensor
    refine_ratio: float
    bypass_ratio: float


# ============================================================
# Threshold Modes
# ============================================================

class ThresholdStrategy(str, Enum):
    STATIC = "static"
    MEAN_STD = "mean_std"
    EMA = "ema"
    PERCENTILE = "percentile"


# ============================================================
# Trigger Policy
# ============================================================

class TriggerPolicy(nn.Module):
    """
    Event-trigger decision module.

    Implements the paper condition:

        δ(t) > θT

    If true:
        route to Neural ODE refinement

    Else:
        bypass refinement

    Supports:
    - Static threshold
    - Adaptive threshold
    - Rolling threshold estimation

    Paper Equation:
        δ(t) > θT
    """

    def __init__(
        self,
        threshold: float = 0.3,
        strategy: str = "ema",
        k: float = 1.0,
        percentile: float = 90.0,
        ema_alpha: float = 0.1,
        history_size: int = 100,
        warmup_steps: int = 20,
        hysteresis_factor: float = 0.9,
        min_threshold: float = 0.05,
        max_threshold: float = 5.0,
    ):
        super().__init__()

        self.base_threshold = threshold

        self.strategy = ThresholdStrategy(strategy)

        self.k = k
        self.percentile = percentile
        self.ema_alpha = ema_alpha

        self.history_size = history_size
        self.warmup_steps = warmup_steps

        self.hysteresis_factor = hysteresis_factor

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.step_count = 0

        self.score_history = deque(maxlen=history_size)
        self.threshold_history = deque(maxlen=history_size)

        self.ema_mean = None
        self.ema_var = None

        self.previous_mask = None

        self.total_samples = 0
        self.total_triggers = 0
        self.total_bypasses = 0

    # ========================================================
    # Public API
    # ========================================================

    def forward(
        self,
        complexity_score: torch.Tensor,
    ) -> TriggerDecision:
        """
        Parameters
        ----------
        complexity_score:
            δ(t)
            Shape:
                [B]
                [B, T]
                [*]

        Returns
        -------
        TriggerDecision
        """

        self._validate_input(complexity_score)

        threshold = self.compute_threshold(complexity_score)

        high = threshold

        low = (
            threshold
            * self.hysteresis_factor
        )

        if self.previous_mask is None:

            trigger_mask = (
                complexity_score > high
            )

        else:

            trigger_mask = torch.where(
                self.previous_mask,
                complexity_score > low,
                complexity_score > high,
            )

        self.previous_mask = trigger_mask.detach()

        refine_ratio = (
            trigger_mask.float().mean().item()
        )

        num_samples = trigger_mask.numel()

        num_triggers = (
            trigger_mask.sum().item()
        )

        num_bypasses = (
            num_samples
            - num_triggers
        )

        self.total_samples += num_samples
        self.total_triggers += num_triggers
        self.total_bypasses += num_bypasses

        bypass_ratio = 1.0 - refine_ratio

        return TriggerDecision(
            trigger_mask=trigger_mask,
            complexity_score=complexity_score,
            threshold=threshold,
            refine_ratio=refine_ratio,
            bypass_ratio=bypass_ratio,
        )
    

    def get_runtime_metrics(self):

        if self.total_samples == 0:

            return {}

        return {
            "activation_ratio":
                self.total_triggers
                / self.total_samples,

            "bypass_efficiency":
                self.total_bypasses
                / self.total_samples,

            "total_samples":
                self.total_samples,

            "total_triggers":
                self.total_triggers,

            "total_bypasses":
                self.total_bypasses,
        }

    def to_config(self):

        return {
            "strategy":
                self.strategy.value,

            "base_threshold":
                self.base_threshold,

            "k":
                self.k,

            "percentile":
                self.percentile,

            "ema_alpha":
                self.ema_alpha,

            "warmup_steps":
                self.warmup_steps,

            "history_size":
                self.history_size,

            "hysteresis_factor":
                self.hysteresis_factor,
        }

    # ========================================================
    # Threshold Computation
    # ========================================================

    def compute_threshold(
        self,
        complexity_score: torch.Tensor,
    ) -> torch.Tensor:

        self.step_count += 1

        if self.step_count <= self.warmup_steps:

            threshold_value = self.base_threshold

        elif self.strategy == ThresholdStrategy.STATIC:

            threshold_value = self.base_threshold

        elif self.strategy == ThresholdStrategy.MEAN_STD:

            threshold_value = self._mean_std_threshold(
                complexity_score
            )

        elif self.strategy == ThresholdStrategy.EMA:

            threshold_value = self._ema_threshold(
                complexity_score
            )

        elif self.strategy == ThresholdStrategy.PERCENTILE:

            threshold_value = self._percentile_threshold(
                complexity_score
            )

        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}"
            )

        threshold_value = max(
            self.min_threshold,
            threshold_value,
        )

        threshold_value = min(
            self.max_threshold,
            threshold_value,
        )

        self.threshold_history.append(
            threshold_value
        )

        return torch.full_like(
            complexity_score,
            threshold_value,
        )
    # ========================================================
    # Adaptive Threshold
    # ========================================================

    # def _adaptive_threshold(
    #     self,
    #     complexity_score: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Adaptive threshold based on rolling statistics.

    #     θT = μ + kσ
    #     """

    #     current_mean = complexity_score.mean().item()
    #     current_std = complexity_score.std().item()

    #     self.score_history.append(current_mean)

    #     if len(self.score_history) == 1:

    #         rolling_mean = current_mean
    #         rolling_std = current_std

    #     else:

    #         history_tensor = torch.tensor(
    #             list(self.score_history),
    #             dtype=torch.float32,
    #         )

    #         rolling_mean = history_tensor.mean().item()
    #         rolling_std = history_tensor.std().item()

    #     threshold_value = (
    #         rolling_mean +
    #         self.k * rolling_std
    #     )

    #     self._update_threshold_history(
    #         threshold_value
    #     )

    #     return torch.full_like(
    #         complexity_score,
    #         fill_value=threshold_value,
    #     )
    
    def _mean_std_threshold(
        self,
        complexity_score: torch.Tensor,
    ) -> float:

        score = complexity_score.detach()

        mean = score.mean().item()
        std = score.std().item()

        self.score_history.append(mean)

        return mean + self.k * std


    def _ema_threshold(
        self,
        complexity_score: torch.Tensor,
    ) -> float:

        score = complexity_score.detach()

        mean = score.mean().item()
        var = score.var().item()

        if self.ema_mean is None:

            self.ema_mean = mean
            self.ema_var = var

        else:

            self.ema_mean = (
                self.ema_alpha * mean
                + (1 - self.ema_alpha)
                * self.ema_mean
            )

            self.ema_var = (
                self.ema_alpha * var
                + (1 - self.ema_alpha)
                * self.ema_var
            )

        std = self.ema_var ** 0.5

        return self.ema_mean + self.k * std


    def _percentile_threshold(
        self,
        complexity_score: torch.Tensor,
    ) -> float:

        score = complexity_score.detach()

        return torch.quantile(
            score,
            self.percentile / 100.0,
        ).item()
    # ========================================================
    # Metrics
    # ========================================================

    def get_statistics(self) -> Dict[str, float]:
        """
        Returns threshold statistics useful
        for optimization metrics.

        Includes support for:

        - Threshold Stability
        - Mean Threshold
        - Threshold Variance
        """

        if len(self.threshold_history) == 0:

            return {
                "mean_threshold": 0.0,
                "std_threshold": 0.0,
                "threshold_stability": 0.0,
            }

        values = torch.tensor(
            list(self.threshold_history),
            dtype=torch.float32,
        )

        mean_val = values.mean().item()
        std_val = values.std().item()

        return {
            "mean_threshold": mean_val,
            "std_threshold": std_val,
            "threshold_stability": std_val,
        }

    # ========================================================
    # History Management
    # ========================================================

    def reset_history(self) -> None:
        """
        Clears rolling statistics.
        """

        self.score_history.clear()
        self.threshold_history.clear()

    def _update_threshold_history(
        self,
        threshold: float,
    ) -> None:
        self.threshold_history.append(
            float(threshold)
        )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_input(
        complexity_score: torch.Tensor,
    ) -> None:

        if not isinstance(
            complexity_score,
            torch.Tensor,
        ):
            raise TypeError(
                "complexity_score must be torch.Tensor"
            )

        if complexity_score.numel() == 0:
            raise ValueError(
                "complexity_score cannot be empty"
            )

        if torch.isnan(complexity_score).any():
            raise ValueError(
                "complexity_score contains NaNs"
            )

        if torch.isinf(complexity_score).any():
            raise ValueError(
                "complexity_score contains Infs"
            )

    # ========================================================
    # Properties
    # ========================================================

    @property
    def current_threshold(self) -> Optional[float]:

        if len(self.threshold_history) == 0:
            return None

        return self.threshold_history[-1]

    @property
    def threshold_stability(self) -> float:

        stats = self.get_statistics()

        return stats["threshold_stability"]
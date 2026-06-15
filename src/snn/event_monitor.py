from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ============================================================
# CONFIG
# ============================================================

@dataclass
class EventMonitorConfig:
    """
    Event-trigger configuration.
    """

    alpha: float = 0.5
    beta: float = 0.5

    threshold: float = 0.25

    adaptive_threshold: bool = False

    ema_momentum: float = 0.95

    variance_window: int = 10


# ============================================================
# EVENT MONITOR
# ============================================================

class EventMonitor(nn.Module):
    """
    Computes temporal complexity score.

    δ(t)
    =
    αDs(t)
    +
    βVt(t)

    and generates trigger decisions.
    """

    def __init__(
        self,
        config: EventMonitorConfig,
    ):
        super().__init__()

        self.config = config

        self.register_buffer(
            "running_threshold",
            torch.tensor(
                float(config.threshold)
            )
        )

    def validate_input(
        self,
        spikes: Tensor,
    ):

        if spikes.dim() != 3:
            raise ValueError(
                "Expected spikes shape [B,T,F]"
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

        Output:
            [B,T,1]
        """

        _, T, _  = spikes.shape

        output = []

        for t in range(T):

            start = max(
                0,
                t - self.config.variance_window + 1,
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
    
        # ============================================================
        # COMPLEXITY SCORE
        # ============================================================

    @torch.no_grad()
    def compute_complexity_score(
        self,
        spikes: Tensor,
    ) -> Tensor:
        """
        δ(t)

        δ(t)
        =
        αDs(t)
        +
        βVt(t)

        Output:
            [B,T,1]
        """

        density = self.spike_density(
            spikes
        )

        variance = self.temporal_variance(
            spikes
        )

        complexity = (
            self.config.alpha * density
            +
            self.config.beta * variance
        )

        return complexity

    # ============================================================
    # TRIGGER MASK
    # ============================================================

    @torch.no_grad()
    def compute_trigger_mask(
        self,
        complexity_score: Tensor,
    ) -> Tensor:
        """
        Trigger decision.

        δ(t) > θT

        Output:
            [B,T,1]
        """

        threshold = self.running_threshold

        trigger_mask = (
            complexity_score > threshold
        )

        return trigger_mask.float()

    # ============================================================
    # ADAPTIVE THRESHOLD
    # ============================================================

    @torch.no_grad()
    def update_threshold(
        self,
        complexity_score: Tensor,
    ):
        """
        EMA threshold update.
        """

        if not self.config.adaptive_threshold:
            return

        batch_mean = (
            complexity_score.mean()
        )

        self.running_threshold.mul_(
            self.config.ema_momentum
        )

        self.running_threshold.add_(
            (
                1
                -
                self.config.ema_momentum
            )
            * batch_mean
        )


        # ============================================================
        # SUMMARY
        # ============================================================

    @torch.no_grad()
    def summary(
        self,
        complexity_score: Tensor,
        trigger_mask: Tensor,
    ) -> Dict[str, float]:

        return {
            "avg_complexity":
                complexity_score.mean().item(),

            "trigger_rate":
                trigger_mask.mean().item(),

            "current_threshold":
                self.running_threshold.item(),
        }


    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        spikes: Tensor,
    ):
        """
        Returns:

        complexity_score
            [B,T,1]

        trigger_mask
            [B,T,1]
        """

        self.validate_input(
            spikes
        )

        complexity_score = (
            self.compute_complexity_score(
                spikes
            )
        )

        self.update_threshold(
            complexity_score
        )

        trigger_mask = (
            self.compute_trigger_mask(
                complexity_score
            )
        )

        return EventTriggerDecision(
            complexity_score=complexity_score,
            trigger_mask=trigger_mask,
            threshold=self.running_threshold.clone(),
    )



# ============================================================
# EVENT TRIGGER DECISION
# ============================================================

@dataclass
class EventTriggerDecision:
    """
    Trigger result container.
    """

    complexity_score: Tensor

    trigger_mask: Tensor

    threshold: Tensor

    @property
    def activation_ratio(self) -> float:

        return float(
            self.trigger_mask.float()
            .mean()
            .item()
        )

    @property
    def num_triggered(self) -> int:

        return int(
            self.trigger_mask.sum()
            .item()
        )
    

    # ============================================================
    # REGISTRY
    # ============================================================

EVENT_MONITOR_REGISTRY = {
    "default": EventMonitor,
}


# ============================================================
# FACTORY
# ============================================================

class EventMonitorFactory:

    @staticmethod
    def available_monitors():

        return sorted(
            EVENT_MONITOR_REGISTRY.keys()
        )

    @staticmethod
    def create(
        monitor_type: str,
        config: EventMonitorConfig,
    ) -> EventMonitor:

        monitor_type = monitor_type.lower()

        if monitor_type not in EVENT_MONITOR_REGISTRY:

            available = ", ".join(
                EventMonitorFactory.available_monitors()
            )

            raise ValueError(
                f"Unknown monitor '{monitor_type}'. "
                f"Available: {available}"
            )

        return EVENT_MONITOR_REGISTRY[
            monitor_type
        ](config)
    

    # ============================================================
# PUBLIC WRAPPER
# ============================================================

class EventTrigger(nn.Module):
    """
    High-level trigger interface.

    Example
    -------

    trigger = EventTrigger()

    decision = trigger(
        spikes
    )
    """

    def __init__(
        self,
        method: str = "default",
        **kwargs,
    ):
        super().__init__()

        config = EventMonitorConfig(
            **kwargs
        )

        self.monitor = (
            EventMonitorFactory.create(
                method,
                config,
            )
        )

    def forward(
        self,
        spikes: Tensor,
    ):

        return self.monitor(
            spikes
        )

    def summary(
        self,
        complexity_score: Tensor,
        trigger_mask: Tensor,
    ):

        return self.monitor.summary(
            complexity_score,
            trigger_mask,
        )
    


    # ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "EventMonitorConfig",

    "EventTriggerDecision",

    "EventMonitor",

    "EventMonitorFactory",

    "EventTrigger",
]
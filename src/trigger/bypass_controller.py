"""Bypass controller for event-triggered neuromorphic refinement.

This module decides whether a spike representation should:
1) bypass the continuous-time refinement stage, or
2) be routed into the Neural ODE block.

The implementation is research-paper oriented:
- explicit routing decision object
- trigger/ground-truth bookkeeping for precision/recall
- ODE activation ratio and bypass efficiency
- threshold stability tracking
- FLOPs accounting for efficiency claims
- warmup, hysteresis, and cooldown controls
- optional integration with upstream trigger decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from src.trigger.decision import TriggerDecision

import math

import torch


class RoutingMode(str, Enum):
    """Supported routing outcomes."""

    BYPASS = "bypass"
    REFINE = "refine"


class ScoreProvider(Protocol):
    """Protocol for objects that can produce a routing score."""

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class BypassDecision:
    """Structured output of a bypass decision."""

    route: RoutingMode
    score: float
    threshold: float
    should_bypass: bool
    reason: str
    step_index: int
    trigger_mask_fraction: Optional[float] = None
    ground_truth_trigger: Optional[bool] = None
    predicted_trigger: Optional[bool] = None
    threshold_history_index: Optional[int] = None
    flops_bypass: Optional[float] = None
    flops_refine: Optional[float] = None
    estimated_flops_saved: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["route"] = self.route.value
        return payload


@dataclass
class BypassStats:
    """Running statistics for routing behavior and paper metrics."""

    total_calls: int = 0
    bypass_calls: int = 0
    refine_calls: int = 0
    warmup_calls: int = 0
    cooldown_calls: int = 0

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    total_score: float = 0.0
    total_saved_flops: float = 0.0
    total_refine_flops: float = 0.0
    threshold_history: List[float] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    route_history: List[str] = field(default_factory=list)

    @property
    def bypass_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.bypass_calls / self.total_calls
    
    @property
    def routing_entropy(self) -> float:
        probs = [self.bypass_rate, self.refine_rate]

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy
    

    @property
    def routing_stability(self) -> float:
        if len(self.route_history) < 2:
            return 1.0

        switches = 0

        for i in range(1, len(self.route_history)):
            if self.route_history[i] != self.route_history[i - 1]:
                switches += 1

        return 1.0 - (
            switches / (len(self.route_history) - 1)
        )

    @property
    def refine_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.refine_calls / self.total_calls

    @property
    def ode_activation_ratio(self) -> float:
        return self.refine_rate

    @property
    def bypass_efficiency(self) -> float:
        return self.bypass_rate

    @property
    def trigger_precision(self) -> float:
        denom = self.true_positive + self.false_positive
        if denom == 0:
            return 0.0
        return self.true_positive / denom

    @property
    def trigger_recall(self) -> float:
        denom = self.true_positive + self.false_negative
        if denom == 0:
            return 0.0
        return self.true_positive / denom

    @property
    def trigger_f1(self) -> float:
        p = self.trigger_precision
        r = self.trigger_recall
        if p + r == 0:
            return 0.0
        return 2.0 * p * r / (p + r)

    @property
    def mean_score(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_score / self.total_calls

    @property
    def threshold_variance(self) -> float:
        values = self.threshold_history
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    @property
    def estimated_flops_saved(self) -> float:
        return self.total_saved_flops

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_calls": float(self.total_calls),
            "bypass_calls": float(self.bypass_calls),
            "refine_calls": float(self.refine_calls),
            "warmup_calls": float(self.warmup_calls),
            "cooldown_calls": float(self.cooldown_calls),
            "true_positive": float(self.true_positive),
            "false_positive": float(self.false_positive),
            "true_negative": float(self.true_negative),
            "false_negative": float(self.false_negative),
            "bypass_rate": float(self.bypass_rate),
            "refine_rate": float(self.refine_rate),
            "ode_activation_ratio": float(self.ode_activation_ratio),
            "bypass_efficiency": float(self.bypass_efficiency),
            "trigger_precision": float(self.trigger_precision),
            "trigger_recall": float(self.trigger_recall),
            "trigger_f1": float(self.trigger_f1),
            "mean_score": float(self.mean_score),
            "threshold_variance": float(self.threshold_variance),
            "estimated_flops_saved": float(self.estimated_flops_saved),
            "total_refine_flops": float(self.total_refine_flops),
            "routing_entropy": float(self.routing_entropy),
            "routing_stability": float(self.routing_stability), 
        }


class BypassController(torch.nn.Module):
    """Event-triggered controller that decides bypass vs. refinement.

    The controller implements the paper routing rule:
        delta(t) > theta_T  -> refine
        otherwise           -> bypass

    It also supports practical research instrumentation:
    - warmup support
    - optional hysteresis
    - optional cooldown
    - ground-truth trigger bookkeeping for precision/recall
    - FLOPs accounting
    - threshold stability tracking
    - compatibility with upstream trigger decisions
    """

    def __init__(
        self,
        threshold: float = 0.5,
        warmup_steps: int = 0,
        hysteresis: bool = True,
        refine_margin: float = 0.0,
        bypass_margin: float = 0.0,
        cooldown_steps: int = 0,
        score_provider: Optional[ScoreProvider] = None,
    ) -> None:
        super().__init__()

        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if refine_margin < 0 or bypass_margin < 0:
            raise ValueError("margins must be non-negative")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")

        self.threshold = float(threshold)
        self.warmup_steps = int(warmup_steps)
        self.hysteresis = bool(hysteresis)
        self.refine_margin = float(refine_margin)
        self.bypass_margin = float(bypass_margin)
        self.cooldown_steps = int(cooldown_steps)
        self.score_provider = score_provider

        self.stats = BypassStats()
        self.event_log: List[BypassDecision] = []
        self._step_index: int = 0
        self._cooldown_remaining: int = 0
        self._last_route: RoutingMode = RoutingMode.BYPASS

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def last_route(self) -> RoutingMode:
        return self._last_route

    @property
    def active_threshold(self) -> float:
        return self.threshold

    def reset_state(self) -> None:
        """Reset internal counters but keep configuration intact."""
        self.stats = BypassStats()
        self._step_index = 0
        self._cooldown_remaining = 0
        self._last_route = RoutingMode.BYPASS

    def set_threshold(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)
        self.stats.threshold_history.append(self.threshold)

    def export_config(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "warmup_steps": self.warmup_steps,
            "hysteresis": self.hysteresis,
            "refine_margin": self.refine_margin,
            "bypass_margin": self.bypass_margin,
            "cooldown_steps": self.cooldown_steps,
        }

    def get_stats(self) -> Dict[str, float]:
        return self.stats.to_dict()

    def get_metric_snapshot(self) -> Dict[str, float]:
        """Alias for paper-facing metric logging."""
        return self.get_stats()

    def _to_scalar(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("routing score must be a scalar tensor")
            return float(value.detach().item())
        if isinstance(value, (int, float)):
            return float(value)
        return float(value)

    def _infer_trigger_fraction(self, trigger_mask: Optional[torch.Tensor]) -> Optional[float]:
        if trigger_mask is None:
            return None
        if trigger_mask.numel() == 0:
            return 0.0
        return float(trigger_mask.float().mean().detach().item())

    def _extract_score_and_trigger_state(
        self,
        score: Optional[Any] = None,
        trigger_decision: Optional[TriggerDecision] = None,
        **score_kwargs: Any,
    ) -> Tuple[float, Optional[bool], Optional[float]]:
        """Resolve input from a raw score or an upstream trigger decision.

        Supported trigger decision shapes:
        - object with .score / .triggered / .threshold
        - dict with keys score / triggered / threshold
        - plain scalar score
        """
        predicted_trigger: Optional[bool] = None
        upstream_threshold: Optional[float] = None

        if trigger_decision is not None:
            if score is None:
                score = trigger_decision.score

            predicted_trigger = trigger_decision.triggered
            upstream_threshold = trigger_decision.threshold

        if score is None:
            if self.score_provider is None:
                raise ValueError("score must be provided when score_provider is not set")
            score = self.score_provider(**score_kwargs)

        return self._to_scalar(score), predicted_trigger, upstream_threshold

    def _route_from_score(self, score: float) -> Tuple[RoutingMode, str]:
        """Map a scalar score to a route decision."""
        base_threshold = self.threshold

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RoutingMode.BYPASS, "cooldown active"

        if self._step_index < self.warmup_steps:
            return RoutingMode.BYPASS, "warmup phase"

        if not self.hysteresis:
            if score > base_threshold:
                return RoutingMode.REFINE, "score above threshold"
            return RoutingMode.BYPASS, "score below threshold"

        refine_threshold = base_threshold + self.refine_margin
        bypass_threshold = max(0.0, base_threshold - self.bypass_margin)

        if self._last_route == RoutingMode.REFINE:
            if score <= bypass_threshold:
                return RoutingMode.BYPASS, "fell below bypass threshold"
            return RoutingMode.REFINE, "hysteresis hold"

        if score > refine_threshold:
            return RoutingMode.REFINE, "score above refine threshold"
        return RoutingMode.BYPASS, "score below refine threshold"

    def _update_ground_truth_counters(
        self,
        predicted_trigger: Optional[bool],
        actual_trigger: Optional[bool],
    ) -> None:
        if actual_trigger is None:
            return

        if predicted_trigger is None:
            # If the caller only provides ground truth, map it to the realized route.
            predicted_trigger = False

        if predicted_trigger and actual_trigger:
            self.stats.true_positive += 1
        elif predicted_trigger and not actual_trigger:
            self.stats.false_positive += 1
        elif (not predicted_trigger) and actual_trigger:
            self.stats.false_negative += 1
        else:
            self.stats.true_negative += 1

    def decide(
        self,
        score: Optional[Any] = None,
        *,
        trigger_decision: Optional[TriggerDecision] = None,
        trigger_mask: Optional[torch.Tensor] = None,
        ground_truth_trigger: Optional[bool] = None,
        flops_bypass: Optional[float] = None,
        flops_refine: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **score_kwargs: Any,
    ) -> BypassDecision:
        """Compute a bypass/refine decision.

        Parameters
        ----------
        score:
            Explicit scalar score.
        trigger_decision:
            Optional upstream decision object/dict that may contain score,
            triggered flag, and threshold.
        trigger_mask:
            Optional binary mask for active events; used only for reporting.
        ground_truth_trigger:
            Optional label indicating whether refinement was actually needed.
            This enables trigger precision/recall accounting.
        flops_bypass:
            Estimated FLOPs when bypassing refinement.
        flops_refine:
            Estimated FLOPs when refinement is executed.
        metadata:
            Extra experiment info to attach to the decision.
        score_kwargs:
            Forwarded to `score_provider` if score is not supplied.
        """
        scalar_score, predicted_trigger, upstream_threshold = self._extract_score_and_trigger_state(
            score=score,
            trigger_decision=trigger_decision,
            **score_kwargs,
        )
        if not math.isfinite(scalar_score):
            raise ValueError(
                f"Invalid trigger score detected: {scalar_score}"
            )

        # If the upstream module already decided on trigger/bypass, respect it.
        if predicted_trigger is not None:
            route = RoutingMode.REFINE if bool(predicted_trigger) else RoutingMode.BYPASS
            reason = "upstream trigger decision"
        else:
            route, reason = self._route_from_score(scalar_score)
            predicted_trigger = route == RoutingMode.REFINE

        should_bypass = route == RoutingMode.BYPASS
        trigger_fraction = self._infer_trigger_fraction(trigger_mask)

        estimated_flops_saved: Optional[float] = None
        if flops_bypass is not None and flops_refine is not None:
            if flops_bypass < 0 or flops_refine < 0:
                raise ValueError("flops estimates must be non-negative")
            if should_bypass:
                estimated_flops_saved = max(0.0, float(flops_refine) - float(flops_bypass))
            else:
                estimated_flops_saved = 0.0

        threshold_used = upstream_threshold if upstream_threshold is not None else self.threshold

        decision = BypassDecision(
            route=route,
            score=scalar_score,
            threshold=threshold_used,
            should_bypass=should_bypass,
            reason=reason,
            step_index=self._step_index,
            trigger_mask_fraction=trigger_fraction,
            ground_truth_trigger=ground_truth_trigger,
            predicted_trigger=predicted_trigger,
            threshold_history_index=len(self.stats.threshold_history),
            flops_bypass=flops_bypass,
            flops_refine=flops_refine,
            estimated_flops_saved=estimated_flops_saved,
            metadata=metadata,
        )
        self.event_log.append(decision)

        # Update running state after the decision is formed.
        self._step_index += 1
        self._last_route = route
        self.stats.total_calls += 1
        self.stats.threshold_history.append(self.threshold)
        self.stats.score_history.append(scalar_score)
        self.stats.route_history.append(route.value)
        self.stats.total_score += scalar_score

        if self._step_index <= self.warmup_steps:
            self.stats.warmup_calls += 1

        if self._cooldown_remaining > 0:
            self.stats.cooldown_calls += 1

        if should_bypass:
            self.stats.bypass_calls += 1
            if self._step_index > self.warmup_steps and self.cooldown_steps > 0:
                self._cooldown_remaining = self.cooldown_steps
            if flops_bypass is not None:
                self.stats.total_refine_flops += float(flops_bypass)
            if estimated_flops_saved is not None:
                self.stats.total_saved_flops += float(estimated_flops_saved)
        else:
            self.stats.refine_calls += 1
            if flops_refine is not None:
                self.stats.total_refine_flops += float(flops_refine)

        self._update_ground_truth_counters(predicted_trigger=predicted_trigger, actual_trigger=ground_truth_trigger)
        return decision

    def forward(
        self,
        x: torch.Tensor,
        *,
        score: Optional[Any] = None,
        trigger_decision: Optional[TriggerDecision] = None,
        refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        trigger_mask: Optional[torch.Tensor] = None,
        ground_truth_trigger: Optional[bool] = None,
        flops_bypass: Optional[float] = None,
        flops_refine: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **score_kwargs: Any,
    ) -> Tuple[torch.Tensor, BypassDecision]:
        """Route the tensor through bypass or refinement."""
        decision = self.decide(
            score=score,
            trigger_decision=trigger_decision,
            trigger_mask=trigger_mask,
            ground_truth_trigger=ground_truth_trigger,
            flops_bypass=flops_bypass,
            flops_refine=flops_refine,
            metadata=metadata,
            **score_kwargs,
        )

        if decision.should_bypass:
            return x, decision

        if refine_fn is None:
            raise ValueError("refine_fn must be provided when the controller selects refinement")

        refined = refine_fn(x)
        return refined, decision

    def route_batch(
        self,
        batch: torch.Tensor,
        *,
        score: Optional[Any] = None,
        trigger_decision: Optional[TriggerDecision] = None,
        refine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        trigger_mask: Optional[torch.Tensor] = None,
        ground_truth_trigger: Optional[bool] = None,
        flops_bypass: Optional[float] = None,
        flops_refine: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **score_kwargs: Any,
    ) -> Tuple[torch.Tensor, BypassDecision]:
        """Convenience wrapper for batched inputs."""
        return self.forward(
            batch,
            score=score,
            trigger_decision=trigger_decision,
            refine_fn=refine_fn,
            trigger_mask=trigger_mask,
            ground_truth_trigger=ground_truth_trigger,
            flops_bypass=flops_bypass,
            flops_refine=flops_refine,
            metadata=metadata,
            **score_kwargs,
        )

    def summary(self) -> Dict[str, float]:
        """Paper-facing summary of the controller behavior."""
        return {
            "ode_activation_ratio": self.stats.ode_activation_ratio,
            "bypass_efficiency": self.stats.bypass_efficiency,
            "trigger_precision": self.stats.trigger_precision,
            "trigger_recall": self.stats.trigger_recall,
            "trigger_f1": self.stats.trigger_f1,
            "threshold_variance": self.stats.threshold_variance,
            "estimated_flops_saved": self.stats.estimated_flops_saved,
            "mean_score": self.stats.mean_score,
            "routing_entropy": self.stats.routing_entropy,
            "routing_stability": self.stats.routing_stability,
        }
    
    def export_history(self):
        """
        Export all routing events for analysis,
        visualization, and ablation studies.
        """
        return [
            event.to_dict()
            for event in self.event_log
        ]


class ConstantBypassController(BypassController):
    """Simple controller variant for ablation experiments."""

    def __init__(self, always_bypass: bool = True) -> None:
        super().__init__(threshold=0.0, warmup_steps=0, hysteresis=False)
        self.always_bypass = bool(always_bypass)

    def _route_from_score(self, score: float) -> Tuple[RoutingMode, str]:  # noqa: D401
        if self.always_bypass:
            return RoutingMode.BYPASS, "constant bypass baseline"
        return RoutingMode.REFINE, "constant refine baseline"


__all__ = [
    "RoutingMode",
    "BypassDecision",
    "BypassStats",
    "BypassController",
    "ConstantBypassController",
]


from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import math
import statistics
import hashlib
import json
from collections import Counter

import torch
import torch.nn as nn


try:
    from .lora_layer import LoRAConfig, LoRALayer
except Exception:  # pragma: no cover
    LoRAConfig = Any  # type: ignore
    LoRALayer = Any  # type: ignore


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _validate_integer(name: str, value: Any) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _validate_positive_integer(name: str, value: int) -> None:
    _validate_integer(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative_integer(name: str, value: int) -> None:
    _validate_integer(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_numeric(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")


def _validate_finite_numeric(name: str, value: Any) -> None:
    _validate_numeric(name, value)
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _is_power_of_two(value: float) -> bool:
    if value <= 0:
        return False
    if abs(value - round(value)) > 1e-12:
        return False
    iv = int(round(value))
    return (iv & (iv - 1)) == 0


def _canonical_float_list(values: Iterable[float]) -> List[float]:
    output: List[float] = []
    for value in values:
        _validate_finite_numeric("alpha", value)
        output.append(float(value))
    unique = sorted(set(output))
    return unique


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _estimate_lora_parameter_count(rank: int, in_features: int, out_features: int) -> int:
    return int(rank * (in_features + out_features))


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _safe_median(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(statistics.median(values))


def _module_name_fallback(module: nn.Module) -> str:
    return module.__class__.__name__

def _alpha_entropy(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    counts = Counter(values)
    total = float(len(values))

    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return float(entropy)


def _alpha_variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.pvariance(values))


def _alpha_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0

    return float(statistics.pstdev(values))


def _alpha_diversity_ratio(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return len(set(values)) / float(len(values))


def _alpha_concentration(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    counts = Counter(values)

    return max(counts.values()) / float(len(values))


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ScalingConfig:
    """
    Global scaling-management configuration.

    Centralizes all LoRA alpha decisions and future-proofing for:
    - trigger-aware scaling
    - ODE-aware scaling
    - DE search
    - scaling ablations
    - layer-wise studies
    """

    default_alpha: float = 1.0
    min_alpha: float = 0.01
    max_alpha: float = 64.0

    allowed_alphas: Optional[List[float]] = None

    layerwise: bool = False
    inverse_rank: bool = False
    adaptive: bool = False
    trigger_aware: bool = False
    ode_aware: bool = False

    strict: bool = True
    verbose: bool = False

    allow_zero_alpha: bool = False
    alpha_step: float = 0.25
    power_of_two_scaling: bool = False

    def __post_init__(self) -> None:
        _validate_finite_numeric("default_alpha", self.default_alpha)
        _validate_finite_numeric("min_alpha", self.min_alpha)
        _validate_finite_numeric("max_alpha", self.max_alpha)
        _validate_finite_numeric("alpha_step", self.alpha_step)

        if self.allow_zero_alpha:
            if self.default_alpha < 0:
                raise ValueError("default_alpha must be >= 0.")
            if self.min_alpha < 0:
                raise ValueError("min_alpha must be >= 0.")
            if self.max_alpha < 0:
                raise ValueError("max_alpha must be >= 0.")
        else:
            if self.default_alpha <= 0:
                raise ValueError("default_alpha must be positive.")
            if self.min_alpha <= 0:
                raise ValueError("min_alpha must be positive.")
            if self.max_alpha <= 0:
                raise ValueError("max_alpha must be positive.")

        if self.min_alpha > self.max_alpha:
            raise ValueError("min_alpha must be <= max_alpha.")

        if self.alpha_step <= 0:
            raise ValueError("alpha_step must be positive.")

        if self.layerwise and self.inverse_rank:
            raise ValueError("layerwise and inverse_rank cannot both be enabled.")

        if self.layerwise and self.adaptive:
            raise ValueError("layerwise and adaptive cannot both be enabled.")

        if self.allowed_alphas is not None:
            if not isinstance(self.allowed_alphas, list):
                raise TypeError("allowed_alphas must be a list.")
            normalized: List[float] = []
            for alpha in self.allowed_alphas:
                _validate_finite_numeric("allowed_alpha", alpha)
                alpha = float(alpha)
                if alpha < 0:
                    raise ValueError("allowed alphas cannot be negative.")
                if alpha == 0:
                    if not self.allow_zero_alpha:
                        raise ValueError("alpha 0 requires allow_zero_alpha=True.")
                else:
                    if alpha < self.min_alpha:
                        raise ValueError("allowed alpha below min_alpha.")
                    if alpha > self.max_alpha:
                        raise ValueError("allowed alpha above max_alpha.")
                    if self.power_of_two_scaling and not _is_power_of_two(alpha):
                        raise ValueError("allowed alpha violates power-of-two constraint.")
                normalized.append(alpha)
            normalized = _canonical_float_list(normalized)
            if len(normalized) == 0:
                raise ValueError("allowed_alphas cannot be empty.")
            self.allowed_alphas = normalized
            if self.strict and self.default_alpha not in self.allowed_alphas:
                raise ValueError("default_alpha must exist in allowed_alphas.")

        if (
            self.power_of_two_scaling
            and self.default_alpha > 0
            and not _is_power_of_two(self.default_alpha)
        ):
            raise ValueError("default_alpha violates power-of-two constraint.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# STATISTICS
# ============================================================================


@dataclass
class ScalingStatistics:
    layers_seen: int = 0
    layers_assigned: int = 0
    distinct_scalings_used: int = 0
    minimum_alpha: float = 0.0
    maximum_alpha: float = 0.0
    mean_alpha: float = 0.0
    median_alpha: float = 0.0
    recommendation_calls: int = 0
    ablation_calls: int = 0
    search_space_generation_calls: int = 0
    constraint_violations: int = 0
    trigger_adjustments: int = 0
    ode_adjustments: int = 0
    de_encoding_calls: int = 0
    metadata_exports: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# POLICY TYPES
# ============================================================================


class ScalingPolicyType(str, Enum):
    STATIC = "static"
    LAYERWISE = "layerwise"
    INVERSE_RANK = "inverse_rank"
    ADAPTIVE = "adaptive"
    TRIGGER_AWARE = "trigger_aware"
    ODE_AWARE = "ode_aware"


# ============================================================================
# RECOMMENDATION OBJECTS
# ============================================================================


@dataclass
class ScalingRecommendation:
    layer_name: str
    alpha: float
    reason: str

    confidence: float = 1.0

    estimated_cost: Optional[float] = None
    adaptation_strength: Optional[float] = None
    score: Optional[float] = None

    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class ScalingAblationReport:
    tested_alphas: List[float]

    recommended_alpha: float

    minimum_alpha: float
    maximum_alpha: float

    alpha_budget: float

    alpha_variance: float = 0.0
    alpha_std: float = 0.0
    alpha_entropy: float = 0.0
    alpha_diversity_ratio: float = 0.0

    estimated_adaptation_budget: float = 0.0

    metadata: Dict[str, Any] = None

    def paper_table(self) -> Dict[str, Any]:
        return {
            "recommended_alpha": self.recommended_alpha,
            "alpha_range": self.maximum_alpha - self.minimum_alpha,
            "alpha_budget": self.alpha_budget,
            "alpha_variance": self.alpha_variance,
            "alpha_std": self.alpha_std,
            "alpha_entropy": self.alpha_entropy,
            "alpha_diversity_ratio": self.alpha_diversity_ratio,
            "estimated_adaptation_budget":
                self.estimated_adaptation_budget,
            # "configuration_hash":
            #     self.configuration_hash(),

            # "assignment_hash":
            #     self.assignment_hash(),

            # "search_space_hash":
            #     self.search_space_hash(),

            # "scaling_budget":
            #     self.scaling_budget_report(),

            # "candidate_encoding_metadata":
            #     self.candidate_encoding_metadata(),

            # "placement_coupling_ready":
            #     True,

            # "joint_search_ready":
            #     True,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SEARCH SPACE
# ============================================================================


class ScalingSearchSpace:
    """
    Generates legal alpha candidates under repository constraints.
    """

    def __init__(self, config: ScalingConfig) -> None:
        if not isinstance(config, ScalingConfig):
            raise TypeError("config must be ScalingConfig.")
        self.config = config

    def candidate_alphas(self) -> List[float]:
        if self.config.allowed_alphas is not None:
            return list(self.config.allowed_alphas)

        candidates: List[float] = []

        if self.config.allow_zero_alpha:
            candidates.append(0.0)

        if self.config.power_of_two_scaling:
            start = max(self.config.min_alpha, 1e-12)
            value = 1.0
            while value < start:
                value *= 2.0
            while value <= self.config.max_alpha + 1e-12:
                if value >= self.config.min_alpha:
                    candidates.append(float(value))
                value *= 2.0
        else:
            value = float(self.config.min_alpha)
            while value <= self.config.max_alpha + 1e-12:
                candidates.append(float(value))
                value += float(self.config.alpha_step)

        candidates = _canonical_float_list(candidates)

        if self.config.strict and not candidates:
            raise RuntimeError("No legal candidate alphas available.")

        return candidates

    def filter_candidates(self, candidates: Sequence[float], max_budget: float) -> List[float]:
        _validate_finite_numeric("max_budget", max_budget)
        if max_budget < 0:
            raise ValueError("max_budget must be non-negative.")
        valid: List[float] = []
        for alpha in candidates:
            _validate_finite_numeric("alpha", alpha)
            alpha = float(alpha)
            if 0 <= alpha <= max_budget:
                valid.append(alpha)
        return _canonical_float_list(valid)

    def layerwise_candidates(self, layer_names: Sequence[str]) -> Dict[str, List[float]]:
        output: Dict[str, List[float]] = {}
        candidates = self.candidate_alphas()
        for name in layer_names:
            output[str(name)] = list(candidates)
        return output

    def search_metadata(self) -> Dict[str, Any]:
        candidates = self.candidate_alphas()
        return {
            "candidate_count": len(candidates),
            "minimum_alpha": min(candidates) if candidates else None,
            "maximum_alpha": max(candidates) if candidates else None,
            "candidates": candidates,
            "power_of_two_scaling": self.config.power_of_two_scaling,
            "allow_zero_alpha": self.config.allow_zero_alpha,
        }


# ============================================================================
# POLICIES
# ============================================================================


class BaseScalingPolicy:
    def __init__(self, config: ScalingConfig) -> None:
        if not isinstance(config, ScalingConfig):
            raise TypeError("config must be ScalingConfig.")
        self.config = config

    @property
    def policy_name(self) -> str:
        raise RuntimeError("Policy name unavailable.")

    def alpha_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        raise RuntimeError("Policy does not implement alpha_for_layer.")


class StaticScalingPolicy(BaseScalingPolicy):
    @property
    def policy_name(self) -> str:
        return ScalingPolicyType.STATIC.value

    def alpha_for_layer(self, layer_name: str, **kwargs: Any) -> float:
        return float(self.config.default_alpha)


class LayerwiseScalingPolicy(BaseScalingPolicy):
    def __init__(self, config: ScalingConfig, assignments: Optional[Dict[str, float]] = None) -> None:
        super().__init__(config)
        self.assignments = assignments or {}

    @property
    def policy_name(self) -> str:
        return ScalingPolicyType.LAYERWISE.value

    def alpha_for_layer(self, layer_name: str, **kwargs: Any) -> float:
        if layer_name in self.assignments:
            return float(self.assignments[layer_name])
        return float(self.config.default_alpha)


class InverseRankScalingPolicy(BaseScalingPolicy):
    @property
    def policy_name(self) -> str:
        return ScalingPolicyType.INVERSE_RANK.value

    def alpha_for_layer(self, layer_name: str, *, rank: Optional[int] = None, **kwargs: Any) -> float:
        if rank is None or rank <= 0:
            return float(self.config.default_alpha)
        alpha = float(self.config.default_alpha / float(rank))
        return alpha


class AdaptiveScalingPolicy(BaseScalingPolicy):
    @property
    def policy_name(self) -> str:
        return ScalingPolicyType.ADAPTIVE.value

    def alpha_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        base = float(self.config.default_alpha)
        if rank is not None and rank > 0:
            base = base / math.sqrt(float(rank))
        dims = None
        if in_features is not None and out_features is not None:
            dims = float(max(1, min(int(in_features), int(out_features))))
            if dims <= 256:
                base *= 1.15
            elif dims <= 1024:
                base *= 1.0
            else:
                base *= 0.85
        if layer is not None:
            if hasattr(layer, "base_layer"):
                try:
                    base_layer = getattr(layer, "base_layer")
                    if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                        size = min(int(base_layer.in_features), int(base_layer.out_features))
                        if size > 2048:
                            base *= 0.9
                except Exception:
                    pass
        if context:
            if "parameter_ratio" in context:
                try:
                    ratio = float(context["parameter_ratio"])
                    if ratio < 0.05:
                        base *= 1.15
                    elif ratio > 0.25:
                        base *= 0.9
                except Exception:
                    pass
        return base


class TriggerAwareScalingPolicy(BaseScalingPolicy):
    """
    Uses duck-typed complexity / trigger information.
    """

    @property
    def policy_name(self) -> str:
        return ScalingPolicyType.TRIGGER_AWARE.value

    def compute_trigger_scaling(self, context: Any = None, *, layer_name: str = "") -> float:
        alpha = float(self.config.default_alpha)

        if context is None:
            return alpha

        score = None
        density = None
        variance = None

        if isinstance(context, Mapping):
            score = context.get("complexity_score", context.get("score", None))
            density = context.get("density", None)
            variance = context.get("variance", None)
            if score is None and hasattr(context.get("result", None), "raw_score"):
                result = context["result"]
                score = getattr(result, "raw_score", None)
                density = getattr(result, "density", density)
                variance = getattr(result, "variance", variance)
        else:
            if hasattr(context, "raw_score"):
                score = getattr(context, "raw_score")
            elif hasattr(context, "score"):
                score = getattr(context, "score")
            density = getattr(context, "density", density)
            variance = getattr(context, "variance", variance)
            if score is None and hasattr(context, "mean_score"):
                try:
                    score = float(getattr(context, "mean_score"))
                except Exception:
                    score = None

        scalar = None
        if score is not None:
            try:
                if torch.is_tensor(score):
                    scalar = float(score.mean().item())
                else:
                    scalar = float(score)
            except Exception:
                scalar = None

        density_v = None
        variance_v = None
        if density is not None:
            try:
                density_v = float(density.mean().item()) if torch.is_tensor(density) else float(density)
            except Exception:
                density_v = None
        if variance is not None:
            try:
                variance_v = float(variance.mean().item()) if torch.is_tensor(variance) else float(variance)
            except Exception:
                variance_v = None

        if scalar is None:
            if density_v is not None or variance_v is not None:
                scalar = 0.5 * float(density_v or 0.0) + 0.5 * float(variance_v or 0.0)
            else:
                return alpha

        if scalar < 0.25:
            alpha *= 0.8
        elif scalar < 0.75:
            alpha *= 1.0
        else:
            alpha *= 1.35

        if density_v is not None and density_v < 0.15:
            alpha *= 0.9
        if variance_v is not None and variance_v > 0.75:
            alpha *= 1.15

        return alpha

    def recommend_from_complexity(self, context: Any, layer_name: str = "") -> ScalingRecommendation:
        alpha = self.compute_trigger_scaling(context, layer_name=layer_name)
        reason = "trigger-aware scaling"
        confidence = 0.9
        metadata: Dict[str, Any] = {"policy": self.policy_name, "layer_name": layer_name}
        if isinstance(context, Mapping):
            for key in ("complexity_score", "score", "density", "variance"):
                if key in context:
                    value = context[key]
                    metadata[key] = float(value.mean().item()) if torch.is_tensor(value) else value
        else:
            for key in ("raw_score", "density", "variance"):
                if hasattr(context, key):
                    value = getattr(context, key)
                    metadata[key] = float(value.mean().item()) if torch.is_tensor(value) else value
        return ScalingRecommendation(layer_name=layer_name, alpha=alpha, reason=reason, confidence=confidence, metadata=metadata)

    def trigger_metadata(self, context: Any = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {"policy": self.policy_name, "default_alpha": self.config.default_alpha}
        if context is None:
            return data
        try:
            rec = self.recommend_from_complexity(context)
            data.update({"recommended_alpha": rec.alpha, "reason": rec.reason, "confidence": rec.confidence})
        except Exception as exc:
            data["error"] = str(exc)
        return data

    def alpha_for_layer(self, layer_name: str, *, context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> float:
        return float(self.compute_trigger_scaling(context, layer_name=layer_name))


class ODEAwareScalingPolicy(BaseScalingPolicy):
    """
    Uses duck-typed ODEBlockStatistics information.
    """

    @property
    def policy_name(self) -> str:
        return ScalingPolicyType.ODE_AWARE.value

    def compute_ode_scaling(self, context: Any = None, *, layer_name: str = "") -> float:
        alpha = float(self.config.default_alpha)
        if context is None:
            return alpha

        avg_steps = None
        avg_fev = None
        avg_state_norm = None
        avg_step_norm = None
        avg_state_change = None

        def _as_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        keys = {
            "average_steps": "average_steps",
            "average_function_evaluations": "average_function_evaluations",
            "average_state_norm": "average_state_norm",
            "average_step_norm": "average_step_norm",
            "average_state_change": "average_state_change",
        }

        if isinstance(context, Mapping):
            avg_steps = _as_float(context.get(keys["average_steps"]))
            avg_fev = _as_float(context.get(keys["average_function_evaluations"]))
            avg_state_norm = _as_float(context.get(keys["average_state_norm"]))
            avg_step_norm = _as_float(context.get(keys["average_step_norm"]))
            avg_state_change = _as_float(context.get(keys["average_state_change"]))
        else:
            avg_steps = _as_float(getattr(context, "average_steps", None))
            avg_fev = _as_float(getattr(context, "average_function_evaluations", None))
            avg_state_norm = _as_float(getattr(context, "average_state_norm", None))
            avg_step_norm = _as_float(getattr(context, "average_step_norm", None))
            avg_state_change = _as_float(getattr(context, "average_state_change", None))

        signals = [v for v in [avg_steps, avg_fev, avg_state_norm, avg_step_norm, avg_state_change] if v is not None]
        if not signals:
            return alpha

        complexity = 0.0
        if avg_steps is not None:
            complexity += min(1.0, avg_steps / 32.0)
        if avg_fev is not None:
            complexity += min(1.0, avg_fev / 128.0)
        if avg_state_norm is not None:
            complexity += min(1.0, avg_state_norm / 16.0)
        if avg_step_norm is not None:
            complexity += min(1.0, avg_step_norm / 4.0)
        if avg_state_change is not None:
            complexity += min(1.0, avg_state_change / 4.0)

        complexity /= float(len(signals))

        if complexity < 0.25:
            alpha *= 0.85
        elif complexity < 0.75:
            alpha *= 1.0
        else:
            alpha *= 1.25

        if avg_state_change is not None and avg_state_change > 1.0:
            alpha *= 1.1
        if avg_steps is not None and avg_steps > 64:
            alpha *= 1.05
        if avg_fev is not None and avg_fev > 256:
            alpha *= 1.05

        return alpha

    def recommend_from_ode_statistics(self, context: Any, layer_name: str = "") -> ScalingRecommendation:
        alpha = self.compute_ode_scaling(context, layer_name=layer_name)
        metadata: Dict[str, Any] = {"policy": self.policy_name, "layer_name": layer_name}
        if isinstance(context, Mapping):
            for key in ("average_steps", "average_function_evaluations", "average_state_norm", "average_step_norm", "average_state_change"):
                if key in context:
                    metadata[key] = context[key]
        else:
            for key in ("average_steps", "average_function_evaluations", "average_state_norm", "average_step_norm", "average_state_change"):
                if hasattr(context, key):
                    metadata[key] = getattr(context, key)
        return ScalingRecommendation(layer_name=layer_name, alpha=alpha, reason="ODE-aware scaling", confidence=0.9, metadata=metadata)

    def ode_metadata(self, context: Any = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {"policy": self.policy_name, "default_alpha": self.config.default_alpha}
        if context is None:
            return data
        try:
            rec = self.recommend_from_ode_statistics(context)
            data.update({"recommended_alpha": rec.alpha, "reason": rec.reason, "confidence": rec.confidence})
        except Exception as exc:
            data["error"] = str(exc)
        return data

    def alpha_for_layer(self, layer_name: str, *, context: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> float:
        return float(self.compute_ode_scaling(context, layer_name=layer_name))


# ============================================================================
# MANAGER
# ============================================================================


class ScalingManager:
    """
    Central scaling-management API.
    """

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        *,
        layer_assignments: Optional[Dict[str, float]] = None,
    ) -> None:
        if config is None:
            config = ScalingConfig()
        if not isinstance(config, ScalingConfig):
            raise TypeError("config must be ScalingConfig.")

        self.config = config
        self.statistics_tracker = ScalingStatistics()
        self.active_assignments: Dict[str, float] = {}
        self.search_space = ScalingSearchSpace(config)
        self.layer_assignments = layer_assignments or {}
        self.policy = self._build_policy()

    # ------------------------------------------------------------------
    # POLICY
    # ------------------------------------------------------------------

    def configuration_hash(self) -> str:
        return _stable_hash(
            self.config.to_dict()
        )


    def assignment_hash(self) -> str:
        return _stable_hash(
            self.active_assignments
        )


    def search_space_hash(self) -> str:
        return _stable_hash(
            self.search_space_metadata()
        )

    def _build_policy(self) -> BaseScalingPolicy:
        if self.config.trigger_aware:
            return TriggerAwareScalingPolicy(self.config)
        if self.config.ode_aware:
            return ODEAwareScalingPolicy(self.config)
        if self.config.layerwise:
            return LayerwiseScalingPolicy(self.config, self.layer_assignments)
        if self.config.inverse_rank:
            return InverseRankScalingPolicy(self.config)
        if self.config.adaptive:
            return AdaptiveScalingPolicy(self.config)
        return StaticScalingPolicy(self.config)

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_alpha(self, alpha: float) -> float:
        _validate_finite_numeric("alpha", alpha)
        alpha = float(alpha)

        if alpha == 0:
            if not self.config.allow_zero_alpha:
                raise ValueError("alpha 0 is disabled by allow_zero_alpha=False.")
            return 0.0

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        if alpha < self.config.min_alpha:
            raise ValueError("alpha below min_alpha.")

        if alpha > self.config.max_alpha:
            raise ValueError("alpha above max_alpha.")

        if self.config.power_of_two_scaling and not _is_power_of_two(alpha):
            raise ValueError("alpha violates power-of-two constraint.")

        return float(alpha)

    def normalize_alpha(self, alpha: float) -> float:
        _validate_finite_numeric("alpha", alpha)
        alpha = float(alpha)

        if alpha == 0 and self.config.allow_zero_alpha:
            return 0.0
        if alpha < 0 and self.config.allow_zero_alpha:
            return 0.0

        alpha = _clamp_float(alpha, self.config.min_alpha, self.config.max_alpha)

        if self.config.power_of_two_scaling and alpha > 0 and not _is_power_of_two(alpha):
            candidates = self.search_space.candidate_alphas()
            if not candidates:
                raise RuntimeError("No legal alpha candidates exist.")
            alpha = min(candidates, key=lambda value: abs(value - alpha))

        return float(alpha)

    # ------------------------------------------------------------------
    # DIMENSIONS
    # ------------------------------------------------------------------

    def module_dimensions(self, module: nn.Module) -> Tuple[int, int]:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be nn.Module.")

        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            try:
                return int(module.in_features), int(module.out_features)
            except Exception:
                pass

        if hasattr(module, "base_layer"):
            base = getattr(module, "base_layer")
            if hasattr(base, "in_features") and hasattr(base, "out_features"):
                return int(base.in_features), int(base.out_features)

        raise ValueError("Unable to infer module dimensions.")

    def _extract_layer_rank(self, module: nn.Module) -> Optional[int]:
        for attr in ("rank", "adaptation_rank"):
            if hasattr(module, attr):
                try:
                    value = int(getattr(module, attr))
                    if value > 0:
                        return value
                except Exception:
                    pass
        if hasattr(module, "base_layer"):
            base = getattr(module, "base_layer")
            for attr in ("rank", "adaptation_rank"):
                if hasattr(base, attr):
                    try:
                        value = int(getattr(base, attr))
                        if value > 0:
                            return value
                    except Exception:
                        pass
        return None

    # ------------------------------------------------------------------
    # ASSIGNMENT
    # ------------------------------------------------------------------

    def alpha_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        alpha = self.policy.alpha_for_layer(
            layer_name,
            rank=rank,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )
        alpha = self.normalize_alpha(alpha)
        self.active_assignments[layer_name] = alpha
        if isinstance(
            self.policy,
            TriggerAwareScalingPolicy,
        ):
            self.statistics_tracker.trigger_adjustments += 1

        if isinstance(
            self.policy,
            ODEAwareScalingPolicy,
        ):
            self.statistics_tracker.ode_adjustments += 1
        return alpha

    def alpha_for_module(self, module_name: str, module: nn.Module, context: Optional[Mapping[str, Any]] = None) -> float:
        in_features, out_features = self.module_dimensions(module)
        rank = self._extract_layer_rank(module)
        return self.alpha_for_layer(
            module_name,
            rank=rank,
            in_features=in_features,
            out_features=out_features,
            layer=module,
            context=context,
        )

    def assign_scalings(self, modules: Mapping[str, nn.Module]) -> Dict[str, float]:
        if not isinstance(modules, Mapping):
            raise TypeError("modules must be a mapping.")

        assignments: Dict[str, float] = {}
        dimensions: Dict[str, Tuple[int, int]] = {}
        skipped: List[str] = []

        for module_name, module in modules.items():
            try:
                dimensions[module_name] = self.module_dimensions(module)
                assignments[module_name] = self.alpha_for_module(module_name, module)
            except Exception:
                if self.config.strict:
                    raise
                skipped.append(module_name)

        self.active_assignments = dict(assignments)
        self._update_assignment_statistics(dimensions=dimensions)
        self.statistics_tracker.constraint_violations = len(skipped)
        return dict(assignments)

    # ------------------------------------------------------------------
    # SEARCH SPACE
    # ------------------------------------------------------------------

    def generate_search_space(self, layer_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        self.statistics_tracker.search_space_generation_calls += 1
        candidates = self.search_space.candidate_alphas()
        if layer_names is None:
            return {"global": candidates, "candidate_count": len(candidates)}
        return self.search_space.layerwise_candidates(layer_names)

    def candidate_alphas(self) -> List[float]:
        return self.search_space.candidate_alphas()

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------

    def recommend_alpha(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        in_features: int,
        out_features: int,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ScalingRecommendation:
        self.statistics_tracker.recommendation_calls += 1
        alpha = self.alpha_for_layer(
            layer_name,
            rank=rank,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )

        if self.config.trigger_aware:
            reason = "trigger-aware policy"
        elif self.config.ode_aware:
            reason = "ODE-aware policy"
        elif self.config.inverse_rank:
            reason = "inverse-rank policy"
        elif self.config.adaptive:
            reason = "adaptive policy"
        elif self.config.layerwise:
            reason = "layerwise policy"
        else:
            reason = "static policy"

        metadata = {
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "policy": self.policy.policy_name,
        }
        return ScalingRecommendation(
            layer_name=layer_name,
            alpha=alpha,
            reason=reason,
            confidence=min(
                1.0,
                max(
                    0.5,
                    alpha / max(
                        self.config.default_alpha,
                        1e-8,
                    ),
                ),
            ),

            estimated_cost=float(
                alpha * max(
                    1,
                    (in_features + out_features),
                )
            ),

            adaptation_strength=float(alpha),

            score=float(alpha),
            metadata=metadata,
        )

    def recommend_alphas(self, modules: Mapping[str, nn.Module]) -> Dict[str, ScalingRecommendation]:
        recommendations: Dict[str, ScalingRecommendation] = {}
        for module_name, module in modules.items():
            try:
                in_features, out_features = self.module_dimensions(module)
                rank = self._extract_layer_rank(module)
            except Exception:
                continue
            recommendations[module_name] = self.recommend_alpha(
                module_name,
                rank=rank,
                in_features=in_features,
                out_features=out_features,
                layer=module,
            )
        return recommendations

    # ------------------------------------------------------------------
    # ABLATION
    # ------------------------------------------------------------------

    def ablation_report(
        self,
        alphas: Optional[Sequence[float]] = None,
        *,
        rank: Optional[int] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer_name: str = "layer",
    ) -> ScalingAblationReport:
        self.statistics_tracker.ablation_calls += 1

        if alphas is None:
            alphas = self.candidate_alphas()

        tested = _canonical_float_list([self.normalize_alpha(float(a)) for a in alphas])
        if not tested:
            raise ValueError("ablation_report requires at least one legal alpha.")

        recommended_alpha = self.normalize_alpha(float(self.config.default_alpha))

        if rank is not None and rank > 0 and self.config.inverse_rank:
            recommended_alpha = self.normalize_alpha(self.config.default_alpha / float(rank))

        parameter_budget = sum(tested)

        return ScalingAblationReport(
            tested_alphas=list(tested),
            recommended_alpha=recommended_alpha,
            minimum_alpha=min(tested),
            maximum_alpha=max(tested),
            alpha_budget=float(parameter_budget),
            metadata={
                "policy": self.policy.policy_name,
                "candidate_count": len(tested),
                "layer_name": layer_name,
                "rank": rank,
                "in_features": in_features,
                "out_features": out_features,
            },
            alpha_variance=_alpha_variance(tested),
            alpha_std=_alpha_std(tested),
            alpha_entropy=_alpha_entropy(tested),
            alpha_diversity_ratio=
                _alpha_diversity_ratio(tested),

            estimated_adaptation_budget=
                float(sum(tested)),
        )

    # ------------------------------------------------------------------
    # BUDGET / DISTRIBUTION
    # ------------------------------------------------------------------

    def scaling_distribution(self) -> Dict[float, int]:
        distribution: Dict[float, int] = {}
        for alpha in self.active_assignments.values():
            distribution[alpha] = distribution.get(alpha, 0) + 1
        return distribution
    
    def alpha_variance(self) -> float:
        return _alpha_variance(
            list(self.active_assignments.values())
        )


    def alpha_std(self) -> float:
        return _alpha_std(
            list(self.active_assignments.values())
        )


    def alpha_entropy(self) -> float:
        return _alpha_entropy(
            list(self.active_assignments.values())
        )


    def alpha_diversity_ratio(self) -> float:
        return _alpha_diversity_ratio(
            list(self.active_assignments.values())
        )


    def alpha_concentration(self) -> float:
        return _alpha_concentration(
            list(self.active_assignments.values())
        )
    
    def total_alpha_budget(self) -> float:
        return float(
            sum(self.active_assignments.values())
        )


    def average_alpha_budget(self) -> float:
        if not self.active_assignments:
            return 0.0

        return (
            self.total_alpha_budget()
            / float(len(self.active_assignments))
        )


    def scaling_budget_report(self) -> Dict[str, Any]:

        return {
            "total_alpha_budget":
                self.total_alpha_budget(),

            "average_alpha":
                self.average_alpha_budget(),

            "minimum_alpha":
                self.statistics_tracker.minimum_alpha,

            "maximum_alpha":
                self.statistics_tracker.maximum_alpha,

            "alpha_variance":
                self.alpha_variance(),

            "alpha_std":
                self.alpha_std(),

            "alpha_entropy":
                self.alpha_entropy(),

            "budget_utilization":
                self.total_alpha_budget(),

            "active_layers":
                len(self.active_assignments),
        }

    def scaling_summary(self) -> str:
        if not self.active_assignments:
            return "No scaling assignments."
        alphas = sorted(self.active_assignments.values())
        return f"layers={len(alphas)}, min={min(alphas):.4f}, max={max(alphas):.4f}, mean={statistics.mean(alphas):.4f}"

    def estimated_parameter_budget(
        self,
        dimensions: Optional[
            Mapping[str, Tuple[int, int]]
        ] = None,
        rank: int = 8,
    ) -> int:

        if not dimensions:
            return 0

        total = 0

        for in_f, out_f in dimensions.values():
            total += _estimate_lora_parameter_count(
                rank,
                in_f,
                out_f,
            )

        return int(total)


    def parameter_efficiency_report(
        self,
        dimensions: Optional[
            Mapping[str, Tuple[int, int]]
        ] = None,
        rank: int = 8,
    ) -> Dict[str, Any]:

        budget = self.estimated_parameter_budget(
            dimensions,
            rank,
        )

        return {
            "estimated_parameter_budget":
                budget,

            "alpha_budget":
                self.total_alpha_budget(),

            "parameter_efficiency_ratio":
                (
                    self.total_alpha_budget()
                    / max(1, budget)
                ),
        }
    # ------------------------------------------------------------------
    # CONSTRAINTS
    # ------------------------------------------------------------------

    def check_constraints(self, assignments: Optional[Mapping[str, float]] = None) -> bool:
        if assignments is None:
            assignments = self.active_assignments

        violations = 0
        for alpha in assignments.values():
            try:
                self.validate_alpha(float(alpha))
            except Exception:
                violations += 1

        self.statistics_tracker.constraint_violations = violations
        return violations == 0

    def constraint_report(self) -> Dict[str, Any]:
        valid = self.check_constraints()
        return {
            "constraints_satisfied": valid,
            "violations": self.statistics_tracker.constraint_violations,
            "power_of_two_scaling": self.config.power_of_two_scaling,
            "minimum_alpha": self.config.min_alpha,
            "maximum_alpha": self.config.max_alpha,
        }

    # ------------------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------------------

    def encoding(self) -> List[float]:
        self.statistics_tracker.de_encoding_calls += 1
        ordered_names = sorted(self.active_assignments.keys())
        return [self.active_assignments[name] for name in ordered_names]

    def encode_for_de(self) -> List[float]:
        self.statistics_tracker.de_encoding_calls += 1
        return self.encoding()

    def decode_encoding(self, vector: Sequence[float], layer_names: Sequence[str]) -> Dict[str, float]:
        self.statistics_tracker.de_encoding_calls += 1
        if len(vector) != len(layer_names):
            raise ValueError("encoding size mismatch.")
        assignments: Dict[str, float] = {}
        for layer_name, alpha in zip(layer_names, vector):
            assignments[layer_name] = self.normalize_alpha(float(alpha))
        return assignments

    def decode_from_de(self, vector: Sequence[float], layer_names: Sequence[str]) -> Dict[str, float]:
        self.statistics_tracker.de_encoding_calls += 1
        return self.decode_encoding(vector, layer_names)

    def encoded_dimension(self) -> int:
        return len(self.active_assignments)

    def candidate_encodings(self, layer_names: Sequence[str]) -> List[List[float]]:
        self.statistics_tracker.de_encoding_calls += 1
        candidates = self.candidate_alphas()
        return [[alpha for _ in layer_names] for alpha in candidates]

    def search_space_cardinality(self) -> float:

        candidates = len(
            self.candidate_alphas()
        )

        layers = max(
            1,
            len(self.active_assignments),
        )

        try:
            return float(
                candidates ** layers
            )
        except OverflowError:
            return float("inf")


    def candidate_encoding_metadata(
        self,
    ) -> Dict[str, Any]:
        
        self.statistics_tracker.de_encoding_calls += 1

        return {
            "de_search_dimensions":
                self.encoded_dimension(),

            "de_candidate_count":
                len(self.candidate_alphas()),

            "estimated_search_space_size":
                self.search_space_cardinality(),
        }
    
    # ------------------------------------------------------------------
    # STATISTICS UPDATE
    # ------------------------------------------------------------------

    def _update_assignment_statistics(self, dimensions: Optional[Mapping[str, Tuple[int, int]]] = None) -> None:
        alphas = list(self.active_assignments.values())

        self.statistics_tracker.layers_seen = len(alphas)
        self.statistics_tracker.layers_assigned = len(alphas)

        if not alphas:
            self.statistics_tracker.distinct_scalings_used = 0
            self.statistics_tracker.minimum_alpha = 0.0
            self.statistics_tracker.maximum_alpha = 0.0
            self.statistics_tracker.mean_alpha = 0.0
            self.statistics_tracker.median_alpha = 0.0
            return

        self.statistics_tracker.distinct_scalings_used = len(set(alphas))
        self.statistics_tracker.minimum_alpha = min(alphas)
        self.statistics_tracker.maximum_alpha = max(alphas)
        self.statistics_tracker.mean_alpha = float(statistics.mean(alphas))
        self.statistics_tracker.median_alpha = float(statistics.median(alphas))

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def search_space_metadata(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        candidates = self.candidate_alphas()
        return {
            "candidate_count": len(candidates),
            "candidate_alphas": candidates,

            "search_ready": True,
            "de_ready": True,
            "ablation_ready": True,

            "encoded_dimension": self.encoded_dimension(),
            "de_search_dimensions":
                self.encoded_dimension(),

            "de_candidate_count":
                len(candidates),

            "estimated_search_space_size":
                self.search_space_cardinality(),
        }

    def metadata(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        self._update_assignment_statistics()
        return {
            "module": "ScalingManager",
            "policy": self.policy.policy_name,
            "policy_flags": {
                "layerwise": self.config.layerwise,
                "inverse_rank": self.config.inverse_rank,
                "adaptive": self.config.adaptive,
                "trigger_aware": self.config.trigger_aware,
                "ode_aware": self.config.ode_aware,
                "strict": self.config.strict,
                "allow_zero_alpha": self.config.allow_zero_alpha,
                "power_of_two_scaling": self.config.power_of_two_scaling,
            },
            "configuration": self.config.to_dict(),
            "active_assignments": dict(self.active_assignments),
            "candidate_alphas": self.candidate_alphas(),
            "scaling_distribution": self.scaling_distribution(),
            "scaling_summary": self.scaling_summary(),
            "search_space": self.search_space_metadata(),
            "global_scaling_report":
                self.global_scaling_report(),

            "layerwise_scaling_report":
                self.layerwise_scaling_report(),
            "configuration_hash":
                self.configuration_hash(),

            "assignment_hash":
                self.assignment_hash(),

            "search_space_hash":
                self.search_space_hash(),

            "scaling_budget":
                self.scaling_budget_report(),

            "candidate_encoding_metadata":
                self.candidate_encoding_metadata(),
            "encoding": self.encoding_metadata(),
            "statistics": self.statistics().to_dict(),
            "constraint_report": self.constraint_report(),
            "scaling_search_ready": True,
            "ablation_ready": True,
            "de_optimization_ready": True,
            "placement_ready": True,
            "placement_coupling_ready": True,
            "joint_search_ready": True,
            "trigger_ready": True,
            "ode_ready": True,
        }

    def diagnostics(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        alphas = list(self.active_assignments.values())
        if not alphas:
            return {
                "num_layers": 0,
                "scaling_distribution": {},
                "budget_usage": 0.0,
                "candidate_count": len(self.candidate_alphas()),
                "constraints_satisfied": True,
                "policy": self.policy.policy_name,
            }

        return {
            "num_layers": len(alphas),
            "minimum_alpha": min(alphas),
            "maximum_alpha": max(alphas),
            "mean_alpha": float(statistics.mean(alphas)),
            "median_alpha": float(statistics.median(alphas)),
            "scaling_distribution": self.scaling_distribution(),
            "budget_usage": float(sum(alphas)),
            "candidate_count": len(self.candidate_alphas()),
            "constraints_satisfied": self.check_constraints(),
            "policy": self.policy.policy_name,
            "alpha_variance":
                self.alpha_variance(),

            "alpha_std":
                self.alpha_std(),

            "alpha_entropy":
                self.alpha_entropy(),

            "alpha_diversity_ratio":
                self.alpha_diversity_ratio(),

            "alpha_concentration":
                self.alpha_concentration(),

            "trigger_adjustments":
                self.statistics_tracker.trigger_adjustments,

            "ode_adjustments":
                self.statistics_tracker.ode_adjustments,

            "search_space_size":
                self.search_space_cardinality(),

            "budget_utilization":
                self.total_alpha_budget(),

            "estimated_parameter_budget":
                self.parameter_efficiency_report().get(
                    "estimated_parameter_budget",
                    0,
                ),
        }

    def statistics(self) -> ScalingStatistics:
        self._update_assignment_statistics()
        return self.statistics_tracker

    def export_configuration(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        return {
            "scaling_configuration": self.config.to_dict(),
            "statistics": self.statistics().to_dict(),
            "metadata": self.metadata(),
            "diagnostics": self.diagnostics(),
            "search_space": self.search_space_metadata(),
            "encoding": self.encoding(),
        }
    
    def layerwise_scaling_report(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        report: Dict[str, Dict[str, Any]] = {}

        for layer_name, alpha in self.active_assignments.items():

            report[layer_name] = {
                "alpha": alpha,
                "active": alpha > 0.0,
                "normalized_alpha":
                    alpha / max(
                        self.config.max_alpha,
                        1e-8,
                    ),
            }

        return report


    def global_scaling_report(
        self,
    ) -> Dict[str, Any]:

        return {
            "num_layers":
                len(self.active_assignments),

            "active_layers":
                sum(
                    1
                    for alpha
                    in self.active_assignments.values()
                    if alpha > 0.0
                ),

            "policy":
                self.policy.__class__.__name__,

            "alpha_budget":
                self.total_alpha_budget(),

            "average_alpha":
                self.average_alpha_budget(),

            "alpha_variance":
                self.alpha_variance(),

            "alpha_std":
                self.alpha_std(),

            "alpha_entropy":
                self.alpha_entropy(),

            "alpha_diversity_ratio":
                self.alpha_diversity_ratio(),

            "alpha_concentration":
                self.alpha_concentration(),

            "search_space_size":
                self.search_space_cardinality(),

            "trigger_adjustments":
                self.statistics_tracker.trigger_adjustments,

            "ode_adjustments":
                self.statistics_tracker.ode_adjustments,
        }

    def export_assignments(self) -> Dict[str, float]:
        return dict(self.active_assignments)

    def load_assignments(self, assignments: Mapping[str, float]) -> None:
        if not isinstance(assignments, Mapping):
            raise TypeError("assignments must be a mapping.")

        normalized: Dict[str, float] = {}
        for layer_name, alpha in assignments.items():
            alpha_value = float(alpha)
            if self.config.strict:
                alpha_value = self.validate_alpha(alpha_value)
            else:
                alpha_value = self.normalize_alpha(alpha_value)
            normalized[str(layer_name)] = alpha_value

        self.active_assignments = normalized
        self._update_assignment_statistics()

    # ------------------------------------------------------------------
    # FACTORY METHODS
    # ------------------------------------------------------------------

    @classmethod
    def from_adapter(cls, adapter: nn.Module, config: Optional[ScalingConfig] = None) -> "ScalingManager":
        if not isinstance(adapter, nn.Module):
            raise TypeError("adapter must be nn.Module.")

        manager = cls(config=config)
        assignments: Dict[str, float] = {}

        if hasattr(adapter, "scaling_configuration"):
            try:
                raw = adapter.scaling_configuration()
            except Exception:
                raw = None
            if isinstance(raw, Mapping):
                for k, v in raw.items():
                    try:
                        assignments[str(k)] = float(v)
                    except Exception:
                        continue

        if not assignments and hasattr(adapter, "get_lora_layers"):
            try:
                layers = adapter.get_lora_layers()
            except Exception:
                layers = {}
            if isinstance(layers, Mapping):
                for name, layer in layers.items():
                    alpha = None
                    if hasattr(layer, "alpha"):
                        try:
                            alpha = float(getattr(layer, "alpha"))
                        except Exception:
                            alpha = None
                    elif hasattr(layer, "adaptation_alpha"):
                        try:
                            alpha = float(getattr(layer, "adaptation_alpha"))
                        except Exception:
                            alpha = None
                    if alpha is not None:
                        assignments[str(name)] = alpha

        if assignments:
            manager.load_assignments(assignments)

        return manager

    @classmethod
    def from_model(cls, model: nn.Module, config: Optional[ScalingConfig] = None) -> "ScalingManager":
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module.")

        manager = cls(config=config)
        modules: Dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, "base_layer") or hasattr(module, "alpha") or hasattr(module, "adaptation_alpha"):
                modules[name] = module

        if modules:
            try:
                manager.assign_scalings(modules)
            except Exception:
                if manager.config.strict:
                    raise

        return manager

    # ------------------------------------------------------------------
    # VALIDATION / INTEGRITY
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        try:
            self.check_constraints()
            self.statistics()
            self.metadata()
            return True
        except Exception:
            return False

    def validate_state(self) -> None:
        if not self.verify_integrity():
            raise RuntimeError("ScalingManager integrity check failed.")

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_statistics(self) -> None:
        self.statistics_tracker = ScalingStatistics()

    @torch.no_grad()
    def reset(self) -> None:
        self.active_assignments.clear()
        self.reset_statistics()

    # ------------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------------

    def recommendation_summary(self, recommendations: Mapping[str, ScalingRecommendation]) -> Dict[str, Any]:
        alphas = [rec.alpha for rec in recommendations.values()]
        if not alphas:
            return {}
        return {
            "num_layers": len(alphas),
            "average_alpha": float(sum(alphas) / len(alphas)),
            "minimum_alpha": min(alphas),
            "maximum_alpha": max(alphas),
        }

    def encoding_metadata(self) -> Dict[str, Any]:
        return {
            "dimension": self.encoded_dimension(),
            "candidate_alphas": self.candidate_alphas(),
            "search_space_size": len(self.candidate_alphas()),
            "encoding_ready": True,
            "de_ready": True,
        }

    def extra_repr(self) -> str:
        return (
            f"policy={self.policy.policy_name}, "
            f"assignments={len(self.active_assignments)}, "
            f"alpha_budget={sum(self.active_assignments.values()) if self.active_assignments else 0.0:.4f}"
        )

    # ------------------------------------------------------------------
    # DIAGNOSTIC HELPERS
    # ------------------------------------------------------------------

    def rank_report(self) -> Dict[str, Any]:
        """
        Convenience report for context symmetry with adapter.py.
        """
        return {
            "note": "ScalingManager does not own ranks directly.",
            "candidate_alpha_count": len(self.candidate_alphas()),
            "policy": self.policy.policy_name,
        }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "ScalingConfig",
    "ScalingStatistics",
    "ScalingPolicyType",
    "ScalingRecommendation",
    "ScalingAblationReport",
    "ScalingSearchSpace",
    "StaticScalingPolicy",
    "LayerwiseScalingPolicy",
    "InverseRankScalingPolicy",
    "AdaptiveScalingPolicy",
    "TriggerAwareScalingPolicy",
    "ODEAwareScalingPolicy",
    "ScalingManager",
]

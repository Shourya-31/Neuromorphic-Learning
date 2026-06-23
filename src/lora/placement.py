from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import hashlib
import json
import math
import statistics
from collections import Counter

import torch
import torch.nn as nn


# ============================================================================
# OPTIONAL IMPORTS
# ============================================================================

try:
    from .lora_layer import LoRALayer  # type: ignore
except Exception:  # pragma: no cover
    LoRALayer = Any  # type: ignore

try:
    from .rank_config import RankManager, RankConfig  # type: ignore
except Exception:  # pragma: no cover
    RankManager = Any  # type: ignore
    RankConfig = Any  # type: ignore

try:
    from .scaling import ScalingManager, ScalingConfig  # type: ignore
except Exception:  # pragma: no cover
    ScalingManager = Any  # type: ignore
    ScalingConfig = Any  # type: ignore


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


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _safe_median(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(statistics.median(values))


def _safe_std(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) <= 1:
        return float(default)
    return float(statistics.pstdev(values))


def _safe_variance(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) <= 1:
        return float(default)
    return float(statistics.pvariance(values))


def _entropy_from_binary(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(int(v) for v in values)
    total = float(len(values))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def _diversity_ratio_from_binary(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return len(set(int(v) for v in values)) / float(len(values))


def _concentration_from_binary(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    counts = Counter(int(v) for v in values)
    return max(counts.values()) / float(len(values))


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _module_name_fallback(module: nn.Module) -> str:
    return module.__class__.__name__


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _module_dimensions(module: nn.Module) -> Optional[Tuple[int, int]]:
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        try:
            return int(getattr(module, "in_features")), int(getattr(module, "out_features"))
        except Exception:
            pass

    if hasattr(module, "base_layer"):
        base = _safe_getattr(module, "base_layer")
        if base is not None and hasattr(base, "in_features") and hasattr(base, "out_features"):
            try:
                return int(base.in_features), int(base.out_features)
            except Exception:
                pass

    return None


def _module_rank(module: nn.Module) -> Optional[int]:
    for attr in ("rank", "adaptation_rank"):
        value = _safe_getattr(module, attr, None)
        if value is not None:
            try:
                value = int(value)
                if value > 0:
                    return value
            except Exception:
                pass
    if hasattr(module, "base_layer"):
        base = _safe_getattr(module, "base_layer")
        for attr in ("rank", "adaptation_rank"):
            value = _safe_getattr(base, attr, None)
            if value is not None:
                try:
                    value = int(value)
                    if value > 0:
                        return value
                except Exception:
                    pass
    return None


def _module_alpha(module: nn.Module) -> Optional[float]:
    for attr in ("alpha", "adaptation_alpha", "scaling", "adaptation_scaling"):
        value = _safe_getattr(module, attr, None)
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value):
                    return value
            except Exception:
                pass
    if hasattr(module, "config"):
        cfg = _safe_getattr(module, "config")
        for attr in ("alpha", "scaling"):
            value = _safe_getattr(cfg, attr, None)
            if value is not None:
                try:
                    value = float(value)
                    if math.isfinite(value):
                        return value
                except Exception:
                    pass
    return None


def _module_enabled(module: nn.Module, default: bool = True) -> bool:
    for attr in ("enabled", "is_enabled"):
        value = _safe_getattr(module, attr, None)
        if value is not None:
            b = _as_bool(value)
            if b is not None:
                return b
    if hasattr(module, "config"):
        cfg = _safe_getattr(module, "config")
        for attr in ("enabled", "default_enabled"):
            value = _safe_getattr(cfg, attr, None)
            if value is not None:
                b = _as_bool(value)
                if b is not None:
                    return b
    return bool(default)


def _is_module_like(obj: Any) -> bool:
    return isinstance(obj, nn.Module) or hasattr(obj, "named_modules") or hasattr(obj, "base_layer")


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or (isinstance(value, (int, float)) and value in (0, 1))


def _coerce_bool(value: Any, *, name: str = "assignment") -> bool:
    b = _as_bool(value)
    if b is None:
        raise TypeError(f"{name} must be boolean-like.")
    return bool(b)


def _coerce_layers_mapping(arg: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
    has_module = False
    out: Dict[str, Any] = {}
    for k, v in arg.items():
        if _is_module_like(v):
            has_module = True
        out[str(k)] = v
    return out, has_module


def _binary_density(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(int(v) for v in values) / len(values))


def _binary_variance(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pvariance([int(v) for v in values]))


def _binary_std(values: Sequence[int]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev([int(v) for v in values]))


def _safe_cardinality(n: int, minimum_layers: int, maximum_layers: int, allow_empty: bool) -> Union[int, float]:
    if n < 0:
        return 0
    if n == 0:
        return 1 if allow_empty else 0

    lower = 0 if allow_empty else max(1, minimum_layers)
    upper = min(n, maximum_layers) if maximum_layers > 0 else n
    if upper < lower:
        return 0

    total = 0
    threshold = 10 ** 18
    for k in range(lower, upper + 1):
        total += math.comb(n, k)
        if total > threshold:
            return float("inf")
    return int(total)


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    denom = mx - mn
    if abs(denom) < 1e-12:
        return [0.5 for _ in scores]
    return [float((s - mn) / denom) for s in scores]


def _top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return [idx for idx, _ in indexed[:k]]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PlacementConfig:
    default_enabled: bool = True
    allow_empty_placement: bool = False
    layerwise: bool = True
    rank_aware: bool = True
    scaling_aware: bool = True
    trigger_aware: bool = True
    ode_aware: bool = True
    adaptive: bool = True
    sparse_mode: bool = False
    strict: bool = True
    verbose: bool = False
    minimum_layers: int = 1
    maximum_layers: Optional[int] = None
    maximum_density: float = 1.0
    minimum_density: float = 0.0
    support_joint_search: bool = True
    support_de_encoding: bool = True
    support_ablations: bool = True

    def __post_init__(self) -> None:
        _validate_bool("default_enabled", self.default_enabled)
        _validate_bool("allow_empty_placement", self.allow_empty_placement)
        _validate_bool("layerwise", self.layerwise)
        _validate_bool("rank_aware", self.rank_aware)
        _validate_bool("scaling_aware", self.scaling_aware)
        _validate_bool("trigger_aware", self.trigger_aware)
        _validate_bool("ode_aware", self.ode_aware)
        _validate_bool("adaptive", self.adaptive)
        _validate_bool("sparse_mode", self.sparse_mode)
        _validate_bool("strict", self.strict)
        _validate_bool("verbose", self.verbose)
        _validate_bool("support_joint_search", self.support_joint_search)
        _validate_bool("support_de_encoding", self.support_de_encoding)
        _validate_bool("support_ablations", self.support_ablations)

        _validate_positive_integer("minimum_layers", self.minimum_layers)
        if self.maximum_layers is not None:
            _validate_positive_integer("maximum_layers", self.maximum_layers)
            if self.maximum_layers < self.minimum_layers:
                raise ValueError("maximum_layers must be >= minimum_layers.")

        _validate_finite_numeric("maximum_density", self.maximum_density)
        _validate_finite_numeric("minimum_density", self.minimum_density)
        if not (0.0 <= self.minimum_density <= 1.0):
            raise ValueError("minimum_density must satisfy 0 <= minimum_density <= 1.")
        if not (0.0 <= self.maximum_density <= 1.0):
            raise ValueError("maximum_density must satisfy 0 <= maximum_density <= 1.")
        if self.minimum_density > self.maximum_density:
            raise ValueError("minimum_density must be <= maximum_density.")

        if self.allow_empty_placement:
            if self.minimum_layers < 0:
                raise ValueError("minimum_layers must be non-negative.")
        else:
            if self.minimum_layers < 1:
                raise ValueError("minimum_layers must be positive when empty placement is disabled.")

        if self.maximum_layers is not None and self.maximum_layers == 0 and not self.allow_empty_placement:
            raise ValueError("maximum_layers=0 requires allow_empty_placement=True.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# STATISTICS
# ============================================================================

@dataclass
class PlacementStatistics:
    layers_seen: int = 0
    layers_enabled: int = 0
    layers_disabled: int = 0
    active_placements: int = 0
    placement_density: float = 0.0
    recommendation_calls: int = 0
    ablation_calls: int = 0
    search_space_generation_calls: int = 0
    trigger_adjustments: int = 0
    ode_adjustments: int = 0
    de_encoding_calls: int = 0
    metadata_exports: int = 0
    constraint_violations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# POLICY TYPE
# ============================================================================

class PlacementPolicyType(str, Enum):
    STATIC = "static"
    LAYERWISE = "layerwise"
    RANK_AWARE = "rank_aware"
    SCALING_AWARE = "scaling_aware"
    TRIGGER_AWARE = "trigger_aware"
    ODE_AWARE = "ode_aware"
    SPARSE = "sparse"
    ADAPTIVE = "adaptive"


# ============================================================================
# RECOMMENDATION
# ============================================================================

@dataclass
class PlacementRecommendation:
    layer_name: str
    enabled: bool
    reason: str
    confidence: float = 1.0
    estimated_cost: Optional[float] = None
    adaptation_strength: Optional[float] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# ABLATION REPORT
# ============================================================================

@dataclass
class PlacementAblationReport:
    tested_placements: List[Dict[str, bool]]
    recommended_placement: Dict[str, bool]
    active_layers: int
    placement_density: float
    placement_entropy: float
    placement_diversity: float
    estimated_adaptation_budget: float
    metadata: Dict[str, Any]

    def paper_table(self) -> Dict[str, Any]:
        return {
            "active_layers": self.active_layers,
            "placement_density": self.placement_density,
            "placement_entropy": self.placement_entropy,
            "placement_diversity": self.placement_diversity,
            "estimated_adaptation_budget": self.estimated_adaptation_budget,
        }
    
    def summary(self) -> Dict[str, Any]:
        return {
            "active_layers": self.active_layers,
            "placement_density": self.placement_density,
            "placement_entropy": self.placement_entropy,
            "placement_diversity": self.placement_diversity,
            "estimated_adaptation_budget": self.estimated_adaptation_budget,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SEARCH SPACE
# ============================================================================

class PlacementSearchSpace:
    def __init__(self, config: PlacementConfig) -> None:
        if not isinstance(config, PlacementConfig):
            raise TypeError("config must be PlacementConfig.")
        self.config = config

    def _normalize_layer_names(self, layer_names: Sequence[str]) -> List[str]:
        names = [str(name) for name in layer_names if str(name).strip()]
        if len(names) == 0 and not self.config.allow_empty_placement:
            raise ValueError("layer_names cannot be empty when empty placement is disabled.")
        return names

    def candidate_masks(self, layer_names: Sequence[str]) -> List[List[int]]:
        names = self._normalize_layer_names(layer_names)
        n = len(names)

        if n == 0:
            return [[]]

        min_active = 0 if self.config.allow_empty_placement else max(1, self.config.minimum_layers)
        max_active = self.config.maximum_layers if self.config.maximum_layers is not None else n
        max_active = min(max_active, n)
        min_active = min(min_active, n)
        if max_active < min_active:
            return []

        cardinality = _safe_cardinality(n, min_active, max_active, self.config.allow_empty_placement)
        exhaustive = isinstance(cardinality, int) and cardinality <= 1024 and n <= 16

        masks: List[List[int]] = []

        if exhaustive:
            for k in range(min_active, max_active + 1):
                for combo in combinations(range(n), k):
                    mask = [0] * n
                    for idx in combo:
                        mask[idx] = 1
                    masks.append(mask)
            if self.config.allow_empty_placement:
                masks.append([0] * n)
            if self.config.default_enabled and n > 0 and max_active == n:
                masks.append([1] * n)
            unique: List[List[int]] = []
            seen = set()
            for mask in masks:
                tup = tuple(mask)
                if tup not in seen:
                    unique.append(mask)
                    seen.add(tup)
            return unique

        # Heuristic fallback for large spaces.
        densities = {
            self.config.minimum_density,
            self.config.maximum_density,
            0.0,
            1.0,
            _clamp_float(float(self.config.minimum_layers) / max(1, n), 0.0, 1.0),
            _clamp_float(float(max_active) / max(1, n), 0.0, 1.0),
        }

        masks.append([0] * n)
        masks.append([1] * n)

        for density in sorted(densities):
            k = int(round(density * n))
            k = max(min_active, min(max_active, k))
            if k <= 0:
                if self.config.allow_empty_placement:
                    masks.append([0] * n)
                continue
            mask = [0] * n
            for idx in range(k):
                mask[idx] = 1
            masks.append(mask)

        # top / bottom variants
        if n > 1:
            first = [0] * n
            first[0] = 1
            masks.append(first)
            last = [0] * n
            last[-1] = 1
            masks.append(last)

        unique = []
        seen = set()
        for mask in masks:
            tup = tuple(int(v) for v in mask)
            if tup not in seen:
                unique.append(list(int(v) for v in mask))
                seen.add(tup)
        return unique

    def candidate_placements(self, layer_names: Sequence[str]) -> List[Dict[str, bool]]:
        names = self._normalize_layer_names(layer_names)
        return [
            {name: bool(bit) for name, bit in zip(names, mask)}
            for mask in self.candidate_masks(names)
        ]

    def filter_candidates(
        self,
        candidates: Sequence[Sequence[int]],
        max_budget: Union[int, float],
    ) -> List[List[int]]:
        _validate_finite_numeric("max_budget", max_budget)
        budget = float(max_budget)
        if budget < 0:
            raise ValueError("max_budget must be non-negative.")

        valid: List[List[int]] = []
        for mask in candidates:
            bits = [1 if int(v) else 0 for v in mask]
            active = sum(bits)
            density = active / max(1, len(bits))
            if budget <= 1.0:
                if density <= budget + 1e-12:
                    valid.append(bits)
            else:
                if active <= int(math.floor(budget + 1e-12)):
                    valid.append(bits)

        unique: List[List[int]] = []
        seen = set()
        for mask in valid:
            tup = tuple(mask)
            if tup not in seen:
                unique.append(mask)
                seen.add(tup)
        return unique

    def search_space_cardinality(self, layer_names: Sequence[str]) -> Union[int, float]:
        names = self._normalize_layer_names(layer_names)
        return _safe_cardinality(
            len(names),
            0 if self.config.allow_empty_placement else max(1, self.config.minimum_layers),
            self.config.maximum_layers if self.config.maximum_layers is not None else len(names),
            self.config.allow_empty_placement,
        )

    def search_metadata(self, layer_names: Sequence[str]) -> Dict[str, Any]:
        names = self._normalize_layer_names(layer_names)
        masks = self.candidate_masks(names)
        cardinality = self.search_space_cardinality(names)
        return {
            "candidate_count": len(masks),
            "candidate_names": names,
            "minimum_layers": self.config.minimum_layers,
            "maximum_layers": self.config.maximum_layers,
            "minimum_density": self.config.minimum_density,
            "maximum_density": self.config.maximum_density,
            "cardinality": cardinality,
            "support_de_encoding": self.config.support_de_encoding,
            "support_joint_search": self.config.support_joint_search,
        }


# ============================================================================
# POLICIES
# ============================================================================

class BasePlacementPolicy:
    def __init__(self, config: PlacementConfig) -> None:
        if not isinstance(config, PlacementConfig):
            raise TypeError("config must be PlacementConfig.")
        self.config = config

    @property
    def policy_name(self) -> str:
        raise RuntimeError("Policy name unavailable.")

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        raise RuntimeError("Policy does not implement placement_for_layer.")

    def score_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        return 1.0 if self.placement_for_layer(
            layer_name,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        ) else 0.0


class StaticPlacementPolicy(BasePlacementPolicy):
    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.STATIC.value

    def placement_for_layer(self, layer_name: str, **kwargs: Any) -> bool:
        return bool(self.config.default_enabled)


class LayerwisePlacementPolicy(BasePlacementPolicy):
    def __init__(self, config: PlacementConfig, assignments: Optional[Dict[str, bool]] = None) -> None:
        super().__init__(config)
        self.assignments = dict(assignments or {})

    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.LAYERWISE.value

    def placement_for_layer(self, layer_name: str, **kwargs: Any) -> bool:
        if layer_name in self.assignments:
            return bool(self.assignments[layer_name])
        return bool(self.config.default_enabled)

    def layerwise_metadata(self) -> Dict[str, Any]:
        values = list(int(v) for v in self.assignments.values())
        return {
            "num_assignments": len(self.assignments),
            "enabled_assignments": int(sum(values)),
            "disabled_assignments": int(len(values) - sum(values)),
        }


class RankAwarePlacementPolicy(BasePlacementPolicy):
    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.RANK_AWARE.value

    def _score_from_rank(self, rank: Optional[int], in_features: Optional[int], out_features: Optional[int]) -> float:
        if rank is None:
            return 0.5 if self.config.default_enabled else 0.0
        if in_features is not None and out_features is not None:
            denom = max(1, min(int(in_features), int(out_features)))
            return _clamp_float(float(rank) / float(denom), 0.0, 1.0)
        return _clamp_float(float(rank) / float(rank + 1.0), 0.0, 1.0)

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        score = self._score_from_rank(rank, in_features, out_features)
        if context and "rank_threshold" in context:
            try:
                thr = float(context["rank_threshold"])
                if thr <= 0:
                    thr = 0.5
                score = _clamp_float(score / max(thr, 1e-8), 0.0, 1.0)
            except Exception:
                pass
        return score >= 0.5 if self.config.default_enabled else score > 0.75

    def recommend_from_rank(
        self,
        layer_name: str,
        *,
        rank: Optional[int],
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PlacementRecommendation:
        score = self._score_from_rank(rank, in_features, out_features)
        enabled = self.placement_for_layer(
            layer_name,
            rank=rank,
            in_features=in_features,
            out_features=out_features,
            context=context,
        )
        return PlacementRecommendation(
            layer_name=layer_name,
            enabled=enabled,
            reason="rank-aware placement",
            confidence=_clamp_float(score, 0.5, 1.0),
            estimated_cost=float(rank or 0),
            adaptation_strength=float(score),
            score=float(score),
            metadata={
                "policy": self.policy_name,
                "rank": rank,
                "in_features": in_features,
                "out_features": out_features,
            },
        )

    def rank_metadata(self, rank: Optional[int] = None) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "rank": rank,
            "default_enabled": self.config.default_enabled,
        }


class ScalingAwarePlacementPolicy(BasePlacementPolicy):
    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.SCALING_AWARE.value

    def _score_from_alpha(self, alpha: Optional[float]) -> float:
        if alpha is None:
            return 0.5 if self.config.default_enabled else 0.0
        alpha = max(0.0, float(alpha))
        return _clamp_float(alpha / (alpha + 1.0), 0.0, 1.0)

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        score = self._score_from_alpha(alpha)
        if context and "alpha_threshold" in context:
            try:
                thr = float(context["alpha_threshold"])
                if thr > 0:
                    score = _clamp_float(float(alpha or 0.0) / thr, 0.0, 1.0)
            except Exception:
                pass
        return score >= 0.5 if self.config.default_enabled else score > 0.75

    def recommend_from_scaling(
        self,
        layer_name: str,
        *,
        alpha: Optional[float],
        rank: Optional[int] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PlacementRecommendation:
        score = self._score_from_alpha(alpha)
        enabled = self.placement_for_layer(layer_name, rank=rank, alpha=alpha, context=context)
        return PlacementRecommendation(
            layer_name=layer_name,
            enabled=enabled,
            reason="scaling-aware placement",
            confidence=_clamp_float(score, 0.5, 1.0),
            estimated_cost=float(alpha or 0.0),
            adaptation_strength=float(score),
            score=float(score),
            metadata={
                "policy": self.policy_name,
                "alpha": alpha,
                "rank": rank,
            },
        )

    def scaling_metadata(self, alpha: Optional[float] = None) -> Dict[str, Any]:
        return {
            "policy": self.policy_name,
            "alpha": alpha,
            "default_enabled": self.config.default_enabled,
        }


class TriggerAwarePlacementPolicy(BasePlacementPolicy):
    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.TRIGGER_AWARE.value

    def _extract_context_score(self, context: Any = None) -> Tuple[float, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        if context is None:
            return 0.0, metadata

        score = None
        density = None
        variance = None
        spike_rate = None

        if isinstance(context, Mapping):
            score = context.get("score", context.get("raw_score", context.get("complexity_score", None)))
            density = context.get("density", None)
            variance = context.get("variance", None)
            spike_rate = context.get("spike_rate", None)
            for key in ("score", "raw_score", "complexity_score", "density", "variance", "spike_rate"):
                if key in context:
                    metadata[key] = context[key]
        else:
            for key in ("score", "raw_score", "complexity_score", "density", "variance", "spike_rate"):
                if hasattr(context, key):
                    metadata[key] = getattr(context, key)
            score = _safe_getattr(context, "score", None)
            if score is None:
                score = _safe_getattr(context, "raw_score", None)
            if score is None:
                score = _safe_getattr(context, "complexity_score", None)
            density = _safe_getattr(context, "density", None)
            variance = _safe_getattr(context, "variance", None)
            spike_rate = _safe_getattr(context, "spike_rate", None)

        def _to_scalar(x: Any) -> Optional[float]:
            if x is None:
                return None
            try:
                if torch.is_tensor(x):
                    return float(x.mean().item())
                return float(x)
            except Exception:
                return None

        s = _to_scalar(score)
        d = _to_scalar(density)
        v = _to_scalar(variance)
        r = _to_scalar(spike_rate)

        signals = [x for x in (s, d, v, r) if x is not None]
        if not signals:
            return 0.0, metadata

        value = _safe_mean([float(x) for x in signals])
        return float(value), metadata

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        score, _ = self._extract_context_score(context)
        if layer is not None and score == 0.0:
            # fallback to scale of layer activity if possible
            score = _clamp_float(float(_module_rank(layer) or 0) / max(1.0, float(min(_module_dimensions(layer) or (1, 1)))), 0.0, 1.0)
        if score < 0.25:
            return False if not self.config.default_enabled else False
        if score < 0.75:
            return bool(self.config.default_enabled)
        return True

    def recommend_from_complexity(self, context: Any, layer_name: str = "") -> PlacementRecommendation:
        score, metadata = self._extract_context_score(context)
        enabled = self.placement_for_layer(layer_name, context=context)
        return PlacementRecommendation(
            layer_name=layer_name,
            enabled=enabled,
            reason="trigger-aware placement",
            confidence=_clamp_float(max(0.5, score), 0.5, 1.0),
            estimated_cost=float(score),
            adaptation_strength=float(score),
            score=float(score),
            metadata={
                "policy": self.policy_name,
                **metadata,
            },
        )

    def placement_from_complexity(self, context: Any) -> bool:
        return self.placement_for_layer("layer", context=context)

    def trigger_metadata(self, context: Any = None) -> Dict[str, Any]:
        score, metadata = self._extract_context_score(context)
        return {
            "policy": self.policy_name,
            "score": score,
            "default_enabled": self.config.default_enabled,
            **metadata,
        }


class ODEAwarePlacementPolicy(BasePlacementPolicy):
    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.ODE_AWARE.value

    def _extract_ode_signal(self, context: Any = None) -> Tuple[float, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        if context is None:
            return 0.0, metadata

        keys = (
            "average_steps",
            "average_function_evaluations",
            "average_state_norm",
            "average_step_norm",
            "average_state_change",
        )

        values: List[float] = []
        if isinstance(context, Mapping):
            for key in keys:
                if key in context:
                    metadata[key] = context[key]
                    try:
                        values.append(float(context[key]))
                    except Exception:
                        pass
        else:
            for key in keys:
                if hasattr(context, key):
                    value = getattr(context, key)
                    metadata[key] = value
                    try:
                        values.append(float(value))
                    except Exception:
                        pass

        if not values:
            return 0.0, metadata

        # normalize to [0,1]-ish using gentle saturation
        signals = []
        for v in values:
            v = max(0.0, float(v))
            signals.append(v / (v + 1.0))
        return _safe_mean(signals), metadata

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        score, _ = self._extract_ode_signal(context)
        if score < 0.25:
            return False if not self.config.default_enabled else False
        if score < 0.75:
            return bool(self.config.default_enabled)
        return True

    def recommend_from_ode_statistics(self, context: Any, layer_name: str = "") -> PlacementRecommendation:
        score, metadata = self._extract_ode_signal(context)
        enabled = self.placement_for_layer(layer_name, context=context)
        return PlacementRecommendation(
            layer_name=layer_name,
            enabled=enabled,
            reason="ODE-aware placement",
            confidence=_clamp_float(max(0.5, score), 0.5, 1.0),
            estimated_cost=float(score),
            adaptation_strength=float(score),
            score=float(score),
            metadata={
                "policy": self.policy_name,
                **metadata,
            },
        )

    def placement_from_ode(self, context: Any) -> bool:
        return self.placement_for_layer("layer", context=context)

    def ode_metadata(self, context: Any = None) -> Dict[str, Any]:
        score, metadata = self._extract_ode_signal(context)
        return {
            "policy": self.policy_name,
            "score": score,
            "default_enabled": self.config.default_enabled,
            **metadata,
        }


class SparsePlacementPolicy(BasePlacementPolicy):
    def __init__(self, config: PlacementConfig, top_k: Optional[int] = None) -> None:
        super().__init__(config)
        self.top_k = top_k

    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.SPARSE.value

    def _score(self, layer_name: str, *, rank: Optional[int] = None, alpha: Optional[float] = None, in_features: Optional[int] = None, out_features: Optional[int] = None, layer: Optional[nn.Module] = None, context: Optional[Mapping[str, Any]] = None) -> float:
        score = 0.0
        if rank is not None and rank > 0:
            score += _clamp_float(float(rank) / float(rank + 1.0), 0.0, 1.0)
        if alpha is not None:
            score += _clamp_float(float(alpha) / (float(alpha) + 1.0), 0.0, 1.0)
        if in_features is not None and out_features is not None:
            size = min(int(in_features), int(out_features))
            score += _clamp_float(float(size) / float(size + 128.0), 0.0, 1.0)
        if layer is not None:
            dims = _module_dimensions(layer)
            if dims is not None:
                size = min(dims)
                score += _clamp_float(float(size) / float(size + 128.0), 0.0, 1.0)
        if context and "importance" in context:
            try:
                score += _clamp_float(float(context["importance"]), 0.0, 1.0)
            except Exception:
                pass
        return float(score / 4.0)

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        score = self._score(layer_name, rank=rank, alpha=alpha, in_features=in_features, out_features=out_features, layer=layer, context=context)
        threshold = 0.5
        if context and "threshold" in context:
            try:
                threshold = float(context["threshold"])
                threshold = _clamp_float(threshold, 0.0, 1.0)
            except Exception:
                pass
        return score >= threshold if self.config.default_enabled else score > max(threshold, 0.75)

    def sparsity_report(self, placements: Mapping[str, bool]) -> Dict[str, Any]:
        values = [1 if bool(v) else 0 for v in placements.values()]
        active = sum(values)
        total = len(values)
        return {
            "active": active,
            "inactive": total - active,
            "density": active / max(1, total),
            "sparsity": 1.0 - (active / max(1, total)),
        }

    def density_report(self, placements: Mapping[str, bool]) -> Dict[str, Any]:
        values = [1 if bool(v) else 0 for v in placements.values()]
        return {
            "density": _binary_density(values),
            "variance": _binary_variance(values),
            "std": _binary_std(values),
        }


class AdaptivePlacementPolicy(BasePlacementPolicy):
    @property
    def policy_name(self) -> str:
        return PlacementPolicyType.ADAPTIVE.value

    def placement_priority_score(
        self,
        *,
        rank_score: float,
        scaling_score: float,
        trigger_score: float,
        ode_score: float,
        efficiency_score: float,
        structure_score: float,
    ) -> float:

        components = [
            _clamp_float(rank_score, 0.0, 1.0),
            _clamp_float(scaling_score, 0.0, 1.0),
            _clamp_float(trigger_score, 0.0, 1.0),
            _clamp_float(ode_score, 0.0, 1.0),
            _clamp_float(efficiency_score, 0.0, 1.0),
            _clamp_float(structure_score, 0.0, 1.0),
        ]

        weights = [0.20, 0.20, 0.20, 0.15, 0.15, 0.10]

        return float(
            sum(
                w * c
                for w, c in zip(weights, components)
            )
        )


    def priority_breakdown(
        self,
        *,
        rank_score: float,
        scaling_score: float,
        trigger_score: float,
        ode_score: float,
        efficiency_score: float,
        structure_score: float,
    ) -> Dict[str, float]:

        final_score = self.placement_priority_score(
            rank_score=rank_score,
            scaling_score=scaling_score,
            trigger_score=trigger_score,
            ode_score=ode_score,
            efficiency_score=efficiency_score,
            structure_score=structure_score,
        )

        return {
            "rank_component": rank_score,
            "scaling_component": scaling_score,
            "trigger_component": trigger_score,
            "ode_component": ode_score,
            "efficiency_component": efficiency_score,
            "structure_component": structure_score,
            "final_priority_score": final_score,
        }

    def _score_from_module(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        scores: List[float] = []

        if rank is not None:
            if in_features is not None and out_features is not None:
                scores.append(_clamp_float(float(rank) / max(1.0, float(min(in_features, out_features))), 0.0, 1.0))
            else:
                scores.append(_clamp_float(float(rank) / (float(rank) + 1.0), 0.0, 1.0))

        if alpha is not None:
            scores.append(_clamp_float(float(alpha) / (float(alpha) + 1.0), 0.0, 1.0))

        if in_features is not None and out_features is not None:
            size = min(int(in_features), int(out_features))
            scores.append(_clamp_float(float(size) / float(size + 256.0), 0.0, 1.0))

        if layer is not None:
            dims = _module_dimensions(layer)
            if dims is not None:
                size = min(dims)
                scores.append(_clamp_float(float(size) / float(size + 256.0), 0.0, 1.0))
            if _module_enabled(layer, default=self.config.default_enabled):
                scores.append(1.0)
            else:
                scores.append(0.0)

        if context:
            trigger_like = None
            ode_like = None
            if "trigger_score" in context:
                try:
                    trigger_like = float(context["trigger_score"])
                except Exception:
                    pass
            if "ode_score" in context:
                try:
                    ode_like = float(context["ode_score"])
                except Exception:
                    pass
            if trigger_like is not None:
                scores.append(_clamp_float(trigger_like, 0.0, 1.0))
            if ode_like is not None:
                scores.append(_clamp_float(ode_like, 0.0, 1.0))

        if not scores:
            return 1.0 if self.config.default_enabled else 0.0
        return _safe_mean(scores)

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        score = self._score_from_module(
            layer_name,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )
        return score >= 0.5 if self.config.default_enabled else score > 0.75

    def recommend_from_module(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PlacementRecommendation:
        score = self._score_from_module(
            layer_name,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )
        enabled = self.placement_for_layer(
            layer_name,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )
        return PlacementRecommendation(
            layer_name=layer_name,
            enabled=enabled,
            reason="adaptive placement",
            confidence=_clamp_float(score, 0.5, 1.0),
            estimated_cost=float(max(0.0, score)),
            adaptation_strength=float(score),
            score=float(score),
            metadata={
                "policy": self.policy_name,
                "rank": rank,
                "alpha": alpha,
                "in_features": in_features,
                "out_features": out_features,
            },
        )


# ============================================================================
# MANAGER
# ============================================================================

class PlacementManager(nn.Module):
    def __init__(
        self,
        config: Optional[PlacementConfig] = None,
        *,
        layer_assignments: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = PlacementConfig()
        if not isinstance(config, PlacementConfig):
            raise TypeError("config must be PlacementConfig.")

        self.config = config
        self.statistics_tracker = PlacementStatistics()
        self.search_space = PlacementSearchSpace(config)

        self.active_assignments: Dict[str, bool] = {}
        self.layer_assignments = dict(layer_assignments or {})
        self._layer_info: Dict[str, Dict[str, Any]] = {}
        self.policy = self._build_policy()

    # ------------------------------------------------------------------
    # POLICY
    # ------------------------------------------------------------------

    def _build_policy(self) -> BasePlacementPolicy:
        if self.config.adaptive:
            return AdaptivePlacementPolicy(self.config)
        if self.config.sparse_mode:
            return SparsePlacementPolicy(self.config)
        if self.config.ode_aware:
            return ODEAwarePlacementPolicy(self.config)
        if self.config.trigger_aware:
            return TriggerAwarePlacementPolicy(self.config)
        if self.config.scaling_aware:
            return ScalingAwarePlacementPolicy(self.config)
        if self.config.rank_aware:
            return RankAwarePlacementPolicy(self.config)
        if self.config.layerwise:
            return LayerwisePlacementPolicy(self.config, self.layer_assignments)
        return StaticPlacementPolicy(self.config)

    # ------------------------------------------------------------------
    # HASHES
    # ------------------------------------------------------------------

    def configuration_hash(self) -> str:
        return _stable_hash(self.config.to_dict())

    def assignment_hash(self) -> str:
        return _stable_hash(self.active_assignments)

    def search_space_hash(self) -> str:
        return _stable_hash(self.search_space_metadata())

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_assignment(self, enabled: Any) -> bool:
        b = _as_bool(enabled)
        if b is None:
            raise TypeError("placement assignment must be boolean-like.")
        return bool(b)

    def validate_placement(self, placement: Mapping[str, Any]) -> Dict[str, bool]:
        if not isinstance(placement, Mapping):
            raise TypeError("placement must be a mapping.")
        normalized: Dict[str, bool] = {}
        for name, value in placement.items():
            normalized[str(name)] = self.validate_assignment(value)
        return normalized

    def _validate_density_constraints(self, active_count: int, total_count: int) -> None:
        density = active_count / max(1, total_count)
        if density < self.config.minimum_density - 1e-12:
            raise ValueError("placement density is below minimum_density.")
        if density > self.config.maximum_density + 1e-12:
            raise ValueError("placement density is above maximum_density.")
        if active_count < self.config.minimum_layers and total_count > 0:
            raise ValueError("active layers are below minimum_layers.")
        if self.config.maximum_layers is not None and active_count > self.config.maximum_layers:
            raise ValueError("active layers exceed maximum_layers.")
        if not self.config.allow_empty_placement and total_count > 0 and active_count == 0:
            raise ValueError("empty placement is disabled by configuration.")

    # ------------------------------------------------------------------
    # ACCESS
    # ------------------------------------------------------------------

    def _refresh_statistics(self) -> None:
        values = list(self.active_assignments.values())
        total = len(values)
        active = sum(1 for v in values if v)
        self.statistics_tracker.layers_seen = total
        self.statistics_tracker.layers_enabled = active
        self.statistics_tracker.layers_disabled = total - active
        self.statistics_tracker.active_placements = active
        self.statistics_tracker.placement_density = active / max(1, total)

    def active_layers(self) -> List[str]:
        return [name for name, enabled in self.active_assignments.items() if enabled]

    def inactive_layers(self) -> List[str]:
        return [name for name, enabled in self.active_assignments.items() if not enabled]

    def active_placement_map(self) -> Dict[str, bool]:
        return dict(self.active_assignments)

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------

    def placement_density(self) -> float:
        values = list(self.active_assignments.values())
        return _binary_density([1 if v else 0 for v in values])

    def active_layer_ratio(self) -> float:
        return self.placement_density()

    def inactive_layer_ratio(self) -> float:
        return 1.0 - self.placement_density()

    def placement_entropy(self) -> float:
        values = [1 if v else 0 for v in self.active_assignments.values()]
        return _entropy_from_binary(values)

    def placement_diversity_ratio(self) -> float:
        values = [1 if v else 0 for v in self.active_assignments.values()]
        return _diversity_ratio_from_binary(values)

    def placement_concentration(self) -> float:
        values = [1 if v else 0 for v in self.active_assignments.values()]
        return _concentration_from_binary(values)

    def placement_variance(self) -> float:
        values = [1 if v else 0 for v in self.active_assignments.values()]
        return _binary_variance(values)

    def placement_std(self) -> float:
        values = [1 if v else 0 for v in self.active_assignments.values()]
        return _binary_std(values)
    
    def placement_utilization(self) -> float:
        return self.placement_density()


    def placement_sparsity_score(self) -> float:
        return 1.0 - self.placement_density()


    def placement_budget_utilization(self) -> float:

        active = len(self.active_layers())

        if self.config.maximum_layers is None:
            return self.placement_density()

        return min(
            1.0,
            active / float(
                max(1, self.config.maximum_layers)
            ),
        )


    def placement_activation_ratio(self) -> float:
        return self.placement_density()


    def placement_redundancy_score(self) -> float:

        density = self.placement_density()

        if density <= 0.5:
            return 0.0

        return min(
            1.0,
            (density - 0.5) * 2.0,
        )


    def placement_efficiency_score(self) -> float:

        utilization = self.placement_utilization()

        redundancy = (
            self.placement_redundancy_score()
        )

        return utilization * (
            1.0 - redundancy
        )


    def placement_search_efficiency(self) -> float:

        cardinality = (
            self.search_space_cardinality()
        )

        if cardinality in (0, 1):
            return 1.0

        if math.isinf(cardinality):
            return 0.0

        return 1.0 / math.log2(
            float(cardinality) + 1.0
        )

    def placement_budget_report(self) -> Dict[str, Any]:
        total = len(self.active_assignments)
        active = sum(1 for v in self.active_assignments.values() if v)
        density = active / max(1, total)
        return {
            "active_layers": active,
            "inactive_layers": total - active,
            "total_layers": total,
            "placement_density": density,
            "minimum_layers": self.config.minimum_layers,
            "maximum_layers": self.config.maximum_layers,
            "minimum_density": self.config.minimum_density,
            "maximum_density": self.config.maximum_density,
            "placement_entropy": self.placement_entropy(),
            "placement_diversity_ratio": self.placement_diversity_ratio(),
            "placement_concentration": self.placement_concentration(),
            "placement_variance": self.placement_variance(),
            "placement_std": self.placement_std(),
            "placement_efficiency_ratio": 1.0 - density,
            "adaptation_budget": active,
            "placement_utilization":
                self.placement_utilization(),

            "placement_sparsity":
                self.placement_sparsity_score(),

            "placement_budget_utilization":
                self.placement_budget_utilization(),

            "placement_activation_ratio":
                self.placement_activation_ratio(),

            "placement_redundancy":
                self.placement_redundancy_score(),

            "placement_efficiency":
                self.placement_efficiency_score(),
        }

    def estimated_parameter_cost(self, layer_name: str, layer: Optional[nn.Module] = None) -> int:
        info = self._layer_info.get(layer_name, {})
        dims = info.get("dimensions")
        rank = info.get("rank")
        if layer is not None:
            dims = _module_dimensions(layer) or dims
            rank = _module_rank(layer) or rank
        if dims is None:
            return 0
        if rank is None:
            rank = 0
        in_features, out_features = dims
        return int(max(0, rank) * (int(in_features) + int(out_features)))

    def estimated_parameter_budget(self, modules: Optional[Mapping[str, nn.Module]] = None) -> int:
        if modules is not None:
            self._capture_layer_info(modules)
        total = 0
        for name, enabled in self.active_assignments.items():
            if enabled:
                total += self.estimated_parameter_cost(name)
        return int(total)

    def parameter_efficiency_report(self, modules: Optional[Mapping[str, nn.Module]] = None) -> Dict[str, Any]:
        if modules is not None:
            self._capture_layer_info(modules)
        layer_costs: Dict[str, int] = {}
        total_cost = 0
        active_cost = 0
        for name in self.active_assignments:
            cost = self.estimated_parameter_cost(name)
            layer_costs[name] = cost
            total_cost += cost
            if self.active_assignments[name]:
                active_cost += cost
        ratio = 0.0
        if total_cost > 0:
            ratio = 1.0 - (active_cost / float(total_cost))
        return {
            "layerwise_cost": layer_costs,
            "total_cost": total_cost,
            "active_cost": active_cost,
            "estimated_parameter_budget": active_cost,
            "placement_efficiency_ratio": ratio,
            "adaptation_budget": sum(1 for v in self.active_assignments.values() if v),
            "placement_utilization":
                self.placement_utilization(),

            "placement_efficiency":
                self.placement_efficiency_score(),

            "adaptation_coverage_ratio":
                self.placement_density(),
        }

    # ------------------------------------------------------------------
    # CAPTURE / REFLECTION
    # ------------------------------------------------------------------

    def _capture_layer_info(self, modules: Mapping[str, nn.Module]) -> None:
        if not isinstance(modules, Mapping):
            raise TypeError("modules must be a mapping.")
        for name, module in modules.items():
            if not _is_module_like(module):
                continue
            dims = _module_dimensions(module)
            rank = _module_rank(module)
            alpha = _module_alpha(module)
            enabled = _module_enabled(module, default=self.config.default_enabled)
            self._layer_info[str(name)] = {
                "dimensions": dims,
                "rank": rank,
                "alpha": alpha,
                "enabled": enabled,
                "module_type": _module_name_fallback(module),
            }

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------

    def recommend_placement(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PlacementRecommendation:
        self.statistics_tracker.recommendation_calls += 1
        rec = self._recommendation_for_layer(
            layer_name,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )
        return rec

    def _recommendation_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> PlacementRecommendation:
        policy = self.policy

        if isinstance(policy, AdaptivePlacementPolicy):
            score = policy._score_from_module(
                layer_name,
                rank=rank,
                alpha=alpha,
                in_features=in_features,
                out_features=out_features,
                layer=layer,
                context=context,
            )
            enabled = policy.placement_for_layer(
                layer_name,
                rank=rank,
                alpha=alpha,
                in_features=in_features,
                out_features=out_features,
                layer=layer,
                context=context,
            )
            reason = "adaptive placement"
        elif isinstance(policy, SparsePlacementPolicy):
            score = policy._score(
                layer_name,
                rank=rank,
                alpha=alpha,
                in_features=in_features,
                out_features=out_features,
                layer=layer,
                context=context,
            )
            enabled = policy.placement_for_layer(
                layer_name,
                rank=rank,
                alpha=alpha,
                in_features=in_features,
                out_features=out_features,
                layer=layer,
                context=context,
            )
            reason = "sparse placement"
        elif isinstance(policy, ODEAwarePlacementPolicy):
            score, _ = policy._extract_ode_signal(context)
            enabled = policy.placement_for_layer(layer_name, rank=rank, alpha=alpha, in_features=in_features, out_features=out_features, layer=layer, context=context)
            reason = "ODE-aware placement"
            self.statistics_tracker.ode_adjustments += 1
        elif isinstance(policy, TriggerAwarePlacementPolicy):
            score, _ = policy._extract_context_score(context)
            enabled = policy.placement_for_layer(layer_name, rank=rank, alpha=alpha, in_features=in_features, out_features=out_features, layer=layer, context=context)
            reason = "trigger-aware placement"
            self.statistics_tracker.trigger_adjustments += 1
        elif isinstance(policy, ScalingAwarePlacementPolicy):
            score = policy._score_from_alpha(alpha)
            enabled = policy.placement_for_layer(layer_name, rank=rank, alpha=alpha, in_features=in_features, out_features=out_features, layer=layer, context=context)
            reason = "scaling-aware placement"
        elif isinstance(policy, RankAwarePlacementPolicy):
            score = policy._score_from_rank(rank, in_features, out_features)
            enabled = policy.placement_for_layer(layer_name, rank=rank, alpha=alpha, in_features=in_features, out_features=out_features, layer=layer, context=context)
            reason = "rank-aware placement"
        elif isinstance(policy, LayerwisePlacementPolicy):
            score = 1.0 if policy.placement_for_layer(layer_name) else 0.0
            enabled = policy.placement_for_layer(layer_name)
            reason = "layerwise placement"
        else:
            score = 1.0 if self.config.default_enabled else 0.0
            enabled = bool(self.config.default_enabled)
            reason = "static placement"

        score = _clamp_float(float(score), 0.0, 1.0)
        cost = float(self.estimated_parameter_cost(layer_name, layer))
        return PlacementRecommendation(
            layer_name=layer_name,
            enabled=enabled,
            reason=reason,
            confidence=_clamp_float(max(0.5, score), 0.5, 1.0),
            estimated_cost=cost if cost > 0 else None,
            adaptation_strength=score,
            score=score,
            metadata={
                "policy": self.policy.policy_name,
                "rank": rank,
                "alpha": alpha,
                "in_features": in_features,
                "out_features": out_features,
            },
        )

    def recommend_placements(
        self,
        modules: Mapping[str, nn.Module],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, PlacementRecommendation]:
        if not isinstance(modules, Mapping):
            raise TypeError("modules must be a mapping.")
        self._capture_layer_info(modules)
        recommendations: Dict[str, PlacementRecommendation] = {}
        scores: Dict[str, float] = {}
        module_names = list(modules.keys())

        for name, module in modules.items():
            dims = _module_dimensions(module)
            rank = _module_rank(module)
            alpha = _module_alpha(module)
            rec = self.recommend_placement(
                name,
                rank=rank,
                alpha=alpha,
                in_features=dims[0] if dims else None,
                out_features=dims[1] if dims else None,
                layer=module,
                context=context,
            )
            recommendations[name] = rec
            scores[name] = rec.score if rec.score is not None else (1.0 if rec.enabled else 0.0)

        # Apply global budget if configured.
        active_indices: Optional[List[int]] = None
        if len(module_names) > 0:
            active_indices = self._select_active_indices(module_names, scores)
        if active_indices is not None:
            active_set = set(active_indices)
            for idx, name in enumerate(module_names):
                rec = recommendations[name]
                if idx in active_set:
                    rec.enabled = True
                else:
                    rec.enabled = False
        return recommendations

    def _select_active_indices(self, layer_names: Sequence[str], scores: Mapping[str, float]) -> List[int]:
        names = [str(name) for name in layer_names]
        n = len(names)
        if n == 0:
            return []

        min_active = 0 if self.config.allow_empty_placement else max(1, self.config.minimum_layers)
        max_active = self.config.maximum_layers if self.config.maximum_layers is not None else n
        min_active = min(min_active, n)
        max_active = min(max_active, n)

        density_min = int(math.ceil(self.config.minimum_density * n))
        density_max = int(math.floor(self.config.maximum_density * n))
        min_active = max(min_active, density_min)
        max_active = min(max_active, max(density_min, density_max, max_active))
        if max_active < min_active:
            max_active = min_active

        score_list = [float(scores.get(name, 0.0)) for name in names]
        order = _top_k_indices(score_list, n)

        k = max_active if self.config.default_enabled else min_active
        k = max(min_active, min(max_active, k))
        if k <= 0:
            return []

        return sorted(order[:k])

    # ------------------------------------------------------------------
    # ASSIGNMENT
    # ------------------------------------------------------------------

    def placement_for_layer(
        self,
        layer_name: str,
        *,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer: Optional[nn.Module] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        rec = self.recommend_placement(
            layer_name,
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            out_features=out_features,
            layer=layer,
            context=context,
        )
        self.active_assignments[layer_name] = bool(rec.enabled)
        return bool(rec.enabled)

    def assign_placements(self, placements_or_modules: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> Dict[str, bool]:
        if not isinstance(placements_or_modules, Mapping):
            raise TypeError("placements_or_modules must be a mapping.")

        normalized_input, has_modules = _coerce_layers_mapping(placements_or_modules)

        if has_modules:
            modules: Dict[str, nn.Module] = {k: v for k, v in normalized_input.items() if _is_module_like(v)}
            self._capture_layer_info(modules)
            recs = self.recommend_placements(modules, context=context)
            assignments = {name: bool(rec.enabled) for name, rec in recs.items()}
        else:
            assignments = self.validate_placement(normalized_input)

        self.active_assignments = assignments
        self._refresh_statistics()
        self._validate_current_state()
        return dict(assignments)

    def enable_layer(self, layer_name: str) -> None:
        self.active_assignments[str(layer_name)] = True
        self._refresh_statistics()

    def disable_layer(self, layer_name: str) -> None:
        self.active_assignments[str(layer_name)] = False
        self._refresh_statistics()

    # ------------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------------

    def global_placement_report(self) -> Dict[str, Any]:
        self._refresh_statistics()
        return {
            "active_layers": len(self.active_layers()),
            "inactive_layers": len(self.inactive_layers()),
            "placement_density": self.placement_density(),
            "placement_entropy": self.placement_entropy(),
            "placement_diversity_ratio": self.placement_diversity_ratio(),
            "placement_concentration": self.placement_concentration(),
            "placement_variance": self.placement_variance(),
            "placement_std": self.placement_std(),
            "placement_budget": self.placement_budget_report(),
            "placement_utilization":
                self.placement_utilization(),

            "placement_efficiency":
                self.placement_efficiency_score(),

            "placement_sparsity":
                self.placement_sparsity_score(),

            "placement_budget_utilization":
                self.placement_budget_utilization(),
        }

    def layerwise_placement_report(self) -> Dict[str, Dict[str, Any]]:
        report: Dict[str, Dict[str, Any]] = {}
        for name, enabled in self.active_assignments.items():
            info = self._layer_info.get(name, {})
            report[name] = {
                "enabled": bool(enabled),
                "rank": info.get("rank"),
                "alpha": info.get("alpha"),
                "dimensions": info.get("dimensions"),
                "module_type": info.get("module_type"),
                "parameter_cost": self.estimated_parameter_cost(name),
            }
        return report

    def search_space_metadata(self, layer_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        self.statistics_tracker.search_space_generation_calls += 1
        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        names = [str(n) for n in layer_names]
        meta = self.search_space.search_metadata(names)
        meta.update({
            "de_ready": bool(self.config.support_de_encoding),
            "joint_search_ready": bool(self.config.support_joint_search),
            "trigger_ready": True,
            "ode_ready": True,
        })
        return meta

    def generate_search_space(self, layer_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        names = [str(n) for n in layer_names]
        return {
            "global": self.search_space.candidate_masks(names),
            "candidate_count": len(self.search_space.candidate_masks(names)),
            "search_metadata": self.search_space_metadata(names),
        }

    def candidate_masks(self, layer_names: Optional[Sequence[str]] = None) -> List[List[int]]:
        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        return self.search_space.candidate_masks(layer_names)

    def candidate_placements(self, layer_names: Optional[Sequence[str]] = None) -> List[Dict[str, bool]]:
        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        return self.search_space.candidate_placements(layer_names)

    def search_space_cardinality(self, layer_names: Optional[Sequence[str]] = None) -> Union[int, float]:
        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        return self.search_space.search_space_cardinality(layer_names)

    def encoding(self) -> List[int]:
        self.statistics_tracker.de_encoding_calls += 1
        ordered = sorted(self.active_assignments.keys())
        return [1 if self.active_assignments[name] else 0 for name in ordered]

    def decode_encoding(self, vector: Sequence[int], layer_names: Sequence[str]) -> Dict[str, bool]:
        self.statistics_tracker.de_encoding_calls += 1
        if len(vector) != len(layer_names):
            raise ValueError("encoding size mismatch.")
        out: Dict[str, bool] = {}
        for name, bit in zip(layer_names, vector):
            out[str(name)] = _coerce_bool(bit, name="placement bit")
        return out

    def encoded_dimension(self) -> int:
        return len(self.active_assignments)

    def candidate_encodings(self, layer_names: Sequence[str]) -> List[List[int]]:
        self.statistics_tracker.de_encoding_calls += 1
        return self.search_space.candidate_masks(layer_names)

    def encode_for_de(self) -> List[int]:
        return self.encoding()

    def decode_from_de(self, vector: Sequence[int], layer_names: Sequence[str]) -> Dict[str, bool]:
        return self.decode_encoding(vector, layer_names)

    def candidate_encoding_metadata(self, layer_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        self.statistics_tracker.de_encoding_calls += 1
        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        names = [str(n) for n in layer_names]
        return {
            "de_search_dimensions": len(names),
            "de_candidate_count": len(self.search_space.candidate_masks(names)),
            "estimated_search_space_size": self.search_space_cardinality(names),
            "de_ready": bool(self.config.support_de_encoding),
            "joint_search_ready": bool(self.config.support_joint_search),
        }

    # ------------------------------------------------------------------
    # ABLATION
    # ------------------------------------------------------------------

    def ablation_report(
        self,
        placements: Optional[Sequence[Mapping[str, bool]]] = None,
        *,
        layer_names: Optional[Sequence[str]] = None,
    ) -> PlacementAblationReport:
        self.statistics_tracker.ablation_calls += 1

        if layer_names is None:
            layer_names = list(self.active_assignments.keys())
        names = [str(n) for n in layer_names]

        if placements is None:
            placements = self.candidate_placements(names)

        tested = [self.validate_placement(p) for p in placements]
        if not tested:
            raise ValueError("ablation_report requires at least one placement.")

        current = self.active_placement_map() if self.active_assignments else {name: self.config.default_enabled for name in names}
        active_layers = sum(1 for v in current.values() if v)
        density = active_layers / max(1, len(current))
        entropy = self.placement_entropy()
        diversity = self.placement_diversity_ratio()

        return PlacementAblationReport(
            tested_placements=tested,
            recommended_placement=dict(current),
            active_layers=active_layers,
            placement_density=density,
            placement_entropy=entropy,
            placement_diversity=diversity,
            estimated_adaptation_budget=float(active_layers),
            metadata={
                "policy": self.policy.policy_name,
                "candidate_count": len(tested),
                "layer_names": names,
            },
        )

    # ------------------------------------------------------------------
    # CONSTRAINTS
    # ------------------------------------------------------------------

    def constraint_report(self) -> Dict[str, Any]:
        self._refresh_statistics()
        violations = 0
        total = len(self.active_assignments)

        try:
            self._validate_current_state()
        except Exception:
            violations = 1

        return {
            "constraints_satisfied": violations == 0,
            "violations": violations,
            "minimum_layers": self.config.minimum_layers,
            "maximum_layers": self.config.maximum_layers,
            "minimum_density": self.config.minimum_density,
            "maximum_density": self.config.maximum_density,
            "allow_empty_placement": self.config.allow_empty_placement,
            "total_layers": total,
        }

    def _validate_current_state(self) -> None:
        self._refresh_statistics()
        self._validate_density_constraints(
            self.statistics_tracker.active_placements,
            self.statistics_tracker.layers_seen,
        )
        for name, value in self.active_assignments.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Invalid layer name in placement assignments.")
            _ = self.validate_assignment(value)

    def verify_integrity(self) -> bool:
        try:
            self._validate_current_state()
            self.statistics()
            self.metadata()
            return True
        except Exception:
            return False

    def validate_state(self) -> None:
        if not self.verify_integrity():
            raise RuntimeError("PlacementManager integrity check failed.")

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def statistics(self) -> PlacementStatistics:
        self._refresh_statistics()
        return self.statistics_tracker

    def reset_statistics(self) -> None:
        self.statistics_tracker = PlacementStatistics()

    def reset(self) -> None:
        self.active_assignments.clear()
        self._layer_info.clear()
        self.reset_statistics()

    # ------------------------------------------------------------------
    # METADATA / DIAGNOSTICS
    # ------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        self._refresh_statistics()
        return {
            "module": "PlacementManager",
            "policy": self.policy.policy_name,
            "configuration": self.config.to_dict(),
            "configuration_hash": self.configuration_hash(),
            "assignment_hash": self.assignment_hash(),
            "search_space_hash": self.search_space_hash(),
            "assignments": dict(self.active_assignments),
            "active_layers": self.active_layers(),
            "inactive_layers": self.inactive_layers(),
            "statistics": self.statistics().to_dict(),
            "diagnostics": self.diagnostics(),
            "constraint_report": self.constraint_report(),
            "search_space": self.search_space_metadata(),
            "placement_budget": self.placement_budget_report(),
            "parameter_efficiency": self.parameter_efficiency_report(),
            "global_placement_report": self.global_placement_report(),
            "layerwise_placement_report": self.layerwise_placement_report(),
            "de_ready": bool(self.config.support_de_encoding),
            "joint_search_ready": bool(self.config.support_joint_search),
            "trigger_ready": True,
            "ode_ready": True,
            "placement_search_ready": True,
            "ablation_ready": bool(self.config.support_ablations),
            "placement_utilization":
                self.placement_utilization(),

            "placement_efficiency":
                self.placement_efficiency_score(),

            "placement_sparsity":
                self.placement_sparsity_score(),

            "placement_budget_utilization":
                self.placement_budget_utilization(),

            "rank_coupling_ready": True,
            "scaling_coupling_ready": True,
            "trigger_coupling_ready": True,
            "ode_coupling_ready": True,

            "joint_rank_scaling_placement_ready": True,
            "joint_optimization_ready": True,
        }

    def diagnostics(self) -> Dict[str, Any]:
        self._refresh_statistics()
        total = self.statistics_tracker.layers_seen
        active = self.statistics_tracker.active_placements
        inactive = total - active
        return {
            "placement_density": self.placement_density(),
            "placement_entropy": self.placement_entropy(),
            "placement_diversity_ratio": self.placement_diversity_ratio(),
            "placement_concentration": self.placement_concentration(),
            "placement_variance": self.placement_variance(),
            "placement_std": self.placement_std(),
            "active_layers": active,
            "inactive_layers": inactive,
            "active_layer_ratio": self.active_layer_ratio(),
            "inactive_layer_ratio": self.inactive_layer_ratio(),
            "estimated_parameter_budget": self.estimated_parameter_budget(),
            "trigger_adjustments": self.statistics_tracker.trigger_adjustments,
            "ode_adjustments": self.statistics_tracker.ode_adjustments,
            "search_space_size": self.search_space_cardinality(),
            "de_encoding_calls": self.statistics_tracker.de_encoding_calls,
            "metadata_exports": self.statistics_tracker.metadata_exports,
            "recommendation_calls": self.statistics_tracker.recommendation_calls,
            "constraint_violations": self.statistics_tracker.constraint_violations,
            "placement_utilization":
                self.placement_utilization(),

            "placement_efficiency":
                self.placement_efficiency_score(),

            "placement_sparsity":
                self.placement_sparsity_score(),

            "adaptation_budget_usage":
                self.placement_budget_utilization(),

            "search_space_efficiency":
                self.placement_search_efficiency(),
        }

    # ------------------------------------------------------------------
    # FACTORY
    # ------------------------------------------------------------------

    @classmethod
    def from_adapter(cls, adapter: nn.Module, config: Optional[PlacementConfig] = None) -> "PlacementManager":
        if not isinstance(adapter, nn.Module):
            raise TypeError("adapter must be nn.Module.")
        manager = cls(config=config)
        assignments: Dict[str, bool] = {}
        modules: Dict[str, nn.Module] = {}

        if hasattr(adapter, "get_lora_layers"):
            try:
                raw = adapter.get_lora_layers()
            except Exception:
                raw = {}
            if isinstance(raw, Mapping):
                for name, layer in raw.items():
                    if _is_module_like(layer):
                        modules[str(name)] = layer
                        enabled = _module_enabled(layer, default=manager.config.default_enabled)
                        if hasattr(layer, "enabled"):
                            try:
                                enabled = bool(getattr(layer, "enabled"))
                            except Exception:
                                pass
                        assignments[str(name)] = bool(enabled)
                    else:
                        assignments[str(name)] = manager.config.default_enabled

        if hasattr(adapter, "adapted_modules_summary"):
            try:
                raw = adapter.adapted_modules_summary()
            except Exception:
                raw = None
            if isinstance(raw, Mapping):
                for name, info in raw.items():
                    if isinstance(info, Mapping) and "enabled" in info:
                        assignments[str(name)] = bool(info["enabled"])

        if modules:
            manager._capture_layer_info(modules)
        if assignments:
            manager.active_assignments = manager.validate_placement(assignments)
            manager._refresh_statistics()
        return manager

    @classmethod
    def from_model(cls, model: nn.Module, config: Optional[PlacementConfig] = None) -> "PlacementManager":
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module.")
        manager = cls(config=config)
        modules: Dict[str, nn.Module] = {}
        assignments: Dict[str, bool] = {}

        for name, module in model.named_modules():
            if _is_module_like(module):
                if isinstance(module, nn.Linear) or hasattr(module, "base_layer") or hasattr(module, "rank") or hasattr(module, "alpha") or hasattr(module, "enabled") or hasattr(module, "adaptation_alpha"):
                    modules[str(name)] = module
                    assignments[str(name)] = _module_enabled(module, default=manager.config.default_enabled)

        if modules:
            manager._capture_layer_info(modules)
            # If the model carries LoRA/placement information, honor it; otherwise keep the default policy.
            if manager.config.layerwise:
                manager.active_assignments = manager.validate_placement(assignments)
            else:
                recs = manager.recommend_placements(modules)
                manager.active_assignments = {name: bool(rec.enabled) for name, rec in recs.items()}
            manager._refresh_statistics()
        return manager

    # ------------------------------------------------------------------
    # REPRODUCIBILITY / EXPORT
    # ------------------------------------------------------------------

    def export_configuration(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        return {
            "placement_configuration": self.config.to_dict(),
            "statistics": self.statistics().to_dict(),
            "metadata": self.metadata(),
            "diagnostics": self.diagnostics(),
            "search_space": self.search_space_metadata(),
            "encoding": self.encoding(),
        }

    def export_assignments(self) -> Dict[str, bool]:
        return dict(self.active_assignments)

    def load_assignments(self, assignments: Mapping[str, Any]) -> None:
        if not isinstance(assignments, Mapping):
            raise TypeError("assignments must be a mapping.")
        normalized = self.validate_placement(assignments)
        self.active_assignments = normalized
        self._refresh_statistics()

    # ------------------------------------------------------------------
    # MODULE API
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError("PlacementManager is a management module and does not implement forward().")

    def extra_repr(self) -> str:
        self._refresh_statistics()
        return (
            f"policy={self.policy.policy_name}, "
            f"layers={self.statistics_tracker.layers_seen}, "
            f"active={self.statistics_tracker.active_placements}, "
            f"density={self.statistics_tracker.placement_density:.4f}"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PlacementConfig",
    "PlacementStatistics",
    "PlacementPolicyType",
    "PlacementRecommendation",
    "PlacementAblationReport",
    "PlacementSearchSpace",
    "BasePlacementPolicy",
    "StaticPlacementPolicy",
    "LayerwisePlacementPolicy",
    "RankAwarePlacementPolicy",
    "ScalingAwarePlacementPolicy",
    "TriggerAwarePlacementPolicy",
    "ODEAwarePlacementPolicy",
    "SparsePlacementPolicy",
    "AdaptivePlacementPolicy",
    "PlacementManager",
]
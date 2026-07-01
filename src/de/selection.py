from __future__ import annotations

"""Differential Evolution selection subsystem.

This module implements survivor selection for the joint LoRA rank / scaling /
placement search space used by the event-triggered continuous-time neuromorphic
learning framework.

Responsibilities:
- parent vs trial comparison
- greedy, elitist, diversity-preserving, and adaptive selection
- population validation and repair
- selection analytics and publication/export metadata
- deterministic reproducibility helpers
- reflection-based integration with population and individual abstractions
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import hashlib
import json
import math
import random
import statistics
from collections import Counter

import torch

if TYPE_CHECKING:
    from .population import DEIndividual, Population, PopulationSchema
else:  # pragma: no cover
    try:
        from .population import DEIndividual, Population, PopulationSchema
    except Exception:  # pragma: no cover
        DEIndividual = Any  # type: ignore
        Population = Any  # type: ignore
        PopulationSchema = Any  # type: ignore

try:
    from ..lora.rank_config import RankConfig, RankSearchSpace
except Exception:  # pragma: no cover
    RankConfig = Any  # type: ignore
    RankSearchSpace = Any  # type: ignore

try:
    from ..lora.scaling import ScalingConfig, ScalingSearchSpace
except Exception:  # pragma: no cover
    ScalingConfig = Any  # type: ignore
    ScalingSearchSpace = Any  # type: ignore

try:
    from ..lora.placement import PlacementConfig, PlacementSearchSpace
except Exception:  # pragma: no cover
    PlacementConfig = Any  # type: ignore
    PlacementSearchSpace = Any  # type: ignore


DEFAULT_FITNESS_EPS = 1e-12
DEFAULT_HISTORY_LIMIT = 256


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


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


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _coerce_bool(value: Any, *, name: str = "value") -> bool:
    b = _as_bool(value)
    if b is None:
        raise TypeError(f"{name} must be boolean-like.")
    return bool(b)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / len(values))


def _safe_median(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(statistics.median(values))


def _safe_variance(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) <= 1:
        return float(default)
    return float(statistics.pvariance(values))


def _safe_std(values: Sequence[float], default: float = 0.0) -> float:
    if len(values) <= 1:
        return float(default)
    return float(statistics.pstdev(values))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entropy_from_counts(counts: Sequence[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def _module_name_fallback(module: Any) -> str:
    return module.__class__.__name__


def _is_module_like(obj: Any) -> bool:
    return hasattr(obj, "named_modules") or hasattr(obj, "base_layer") or hasattr(obj, "forward")


def _coerce_mapping_or_empty(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _canonical_chromosome(values: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(values)


# =============================================================================
# REFLECTION / INFERENCE
# =============================================================================


def _extract_layer_names(source: Any = None, layer_names: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    if layer_names is not None:
        names = [str(name) for name in layer_names if str(name).strip()]
        if not names:
            raise ValueError("layer_names cannot be empty.")
        return tuple(names)

    if source is None:
        raise ValueError("layer_names could not be inferred.")

    for attr in ("adapted_layer_names",):
        if hasattr(source, attr):
            try:
                names = list(getattr(source, attr)())
                names = [str(name) for name in names if str(name).strip()]
                if names:
                    return tuple(names)
            except Exception:
                pass

    for attr in ("active_assignments", "layer_assignments"):
        if hasattr(source, attr):
            try:
                mapping = getattr(source, attr)
                if isinstance(mapping, Mapping):
                    names = [str(name) for name in mapping.keys() if str(name).strip()]
                    if names:
                        return tuple(names)
            except Exception:
                pass

    if hasattr(source, "named_modules"):
        try:
            discovered: List[str] = []
            for name, _module in source.named_modules():
                if name.strip():
                    discovered.append(str(name))
            if discovered:
                return tuple(discovered)
        except Exception:
            pass

    raise ValueError("Unable to infer layer names from the provided source.")


def _rank_candidates_from_config(rank_config: Any) -> Tuple[int, ...]:
    if rank_config is None:
        raise ValueError("rank_config is required.")

    if hasattr(rank_config, "candidate_ranks"):
        try:
            values = list(rank_config.candidate_ranks())
            values = [int(v) for v in values]
            if values:
                return tuple(sorted(set(values)))
        except Exception:
            pass

    if hasattr(rank_config, "allowed_ranks") and getattr(rank_config, "allowed_ranks") is not None:
        values = [int(v) for v in getattr(rank_config, "allowed_ranks")]
        if values:
            return tuple(sorted(set(values)))

    minimum = int(getattr(rank_config, "min_rank", getattr(rank_config, "minimum_rank", 1)))
    maximum = int(getattr(rank_config, "max_rank", getattr(rank_config, "maximum_rank", max(1, minimum))))
    allow_zero = bool(getattr(rank_config, "allow_zero_rank", False))
    enforce_pow2 = bool(getattr(rank_config, "enforce_power_of_two", False))

    values: List[int] = []
    if allow_zero:
        values.append(0)
    for rank in range(max(1, minimum), maximum + 1):
        if enforce_pow2 and (rank & (rank - 1)) != 0:
            continue
        values.append(rank)

    values = sorted(set(values))
    if not values:
        raise ValueError("No legal rank candidates available.")
    return tuple(values)


def _scaling_candidates_from_config(scaling_config: Any) -> Tuple[float, ...]:
    if scaling_config is None:
        raise ValueError("scaling_config is required.")

    if hasattr(scaling_config, "candidate_alphas"):
        try:
            values = list(scaling_config.candidate_alphas())
            values = [float(v) for v in values]
            if values:
                return tuple(sorted(set(values)))
        except Exception:
            pass

    if hasattr(scaling_config, "allowed_alphas") and getattr(scaling_config, "allowed_alphas") is not None:
        values = [float(v) for v in getattr(scaling_config, "allowed_alphas")]
        if values:
            return tuple(sorted(set(values)))

    minimum = float(getattr(scaling_config, "min_alpha", getattr(scaling_config, "minimum_alpha", 0.01)))
    maximum = float(getattr(scaling_config, "max_alpha", getattr(scaling_config, "maximum_alpha", max(minimum, 1.0))))
    step = float(getattr(scaling_config, "alpha_step", 0.25))
    allow_zero = bool(getattr(scaling_config, "allow_zero_alpha", False))
    power_of_two = bool(getattr(scaling_config, "power_of_two_scaling", False))

    values: List[float] = []
    if allow_zero:
        values.append(0.0)

    if power_of_two:
        v = 1.0
        while v < maximum + 1e-12:
            if v >= minimum - 1e-12:
                values.append(float(v))
            v *= 2.0
    else:
        v = minimum
        while v <= maximum + 1e-12:
            values.append(float(v))
            v += step

    values = sorted(set(float(v) for v in values if math.isfinite(float(v))))
    if not values:
        raise ValueError("No legal scaling candidates available.")
    return tuple(values)


def _placement_candidates_from_config(placement_config: Any, layer_names: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
    if placement_config is None:
        raise ValueError("placement_config is required.")

    if hasattr(placement_config, "candidate_masks"):
        try:
            masks = list(placement_config.candidate_masks(layer_names))
            normalized: List[Tuple[int, ...]] = []
            for mask in masks:
                normalized.append(tuple(1 if int(v) else 0 for v in mask))
            if normalized:
                return tuple(dict.fromkeys(normalized))
        except Exception:
            pass

    if hasattr(placement_config, "candidate_placements"):
        try:
            placements = list(placement_config.candidate_placements(layer_names))
            normalized: List[Tuple[int, ...]] = []
            for placement in placements:
                if isinstance(placement, Mapping):
                    mask = tuple(1 if _as_bool(placement.get(name, False)) else 0 for name in layer_names)
                    normalized.append(mask)
            if normalized:
                return tuple(dict.fromkeys(normalized))
        except Exception:
            pass

    minimum_layers = int(getattr(placement_config, "minimum_layers", 1))
    maximum_layers = getattr(placement_config, "maximum_layers", None)
    allow_empty = bool(getattr(placement_config, "allow_empty_placement", False))
    n = len(layer_names)
    if n <= 0:
        raise ValueError("layer_names cannot be empty for placement search space.")

    if maximum_layers is None:
        maximum_layers = n
    maximum_layers = min(int(maximum_layers), n)
    minimum_layers = min(max(0 if allow_empty else 1, int(minimum_layers)), n)

    masks: List[Tuple[int, ...]] = []
    if allow_empty:
        masks.append(tuple(0 for _ in range(n)))
    if maximum_layers >= n and bool(getattr(placement_config, "default_enabled", True)):
        masks.append(tuple(1 for _ in range(n)))

    for k in range(minimum_layers, maximum_layers + 1):
        if k <= 0:
            continue
        if k == n:
            masks.append(tuple(1 for _ in range(n)))
            continue
        mask = [0] * n
        for idx in range(k):
            mask[idx] = 1
        masks.append(tuple(mask))
        if n > 1:
            mask = [0] * n
            for idx in range(n - k, n):
                mask[idx] = 1
            masks.append(tuple(mask))

    if n > 1:
        mask = [0] * n
        for idx in range(0, n, 2):
            mask[idx] = 1
        masks.append(tuple(mask))
        mask = [0] * n
        for idx in range(1, n, 2):
            mask[idx] = 1
        masks.append(tuple(mask))

    normalized = tuple(dict.fromkeys(masks))
    if not normalized:
        raise ValueError("No legal placement candidates available.")
    return normalized


def _infer_schema(
    *,
    rank_config: Any = None,
    scaling_config: Any = None,
    placement_config: Any = None,
    source: Any = None,
    layer_names: Optional[Sequence[str]] = None,
) -> Optional["PopulationSchema"]:
    if rank_config is None or scaling_config is None or placement_config is None:
        return None
    if PopulationSchema is Any:  # pragma: no cover
        return None
    try:
        layers = _extract_layer_names(source=source, layer_names=layer_names)
        rank_candidates = _rank_candidates_from_config(rank_config)
        scaling_candidates = _scaling_candidates_from_config(scaling_config)
        placement_candidates = _placement_candidates_from_config(placement_config, layers)
        return PopulationSchema(
            layer_names=layers,
            rank_candidates=rank_candidates,
            scaling_candidates=scaling_candidates,
            placement_candidates=placement_candidates,
            minimum_layers=int(getattr(placement_config, "minimum_layers", 1)),
            maximum_layers=(
                int(getattr(placement_config, "maximum_layers"))
                if getattr(placement_config, "maximum_layers", None) is not None
                else None
            ),
            allow_empty_placement=bool(getattr(placement_config, "allow_empty_placement", False)),
        )
    except Exception:
        return None
    

@dataclass(frozen=True)
class PopulationCapabilities:
    schema: Any = None
    analytics: Mapping[str, float] = field(default_factory=dict)
    size: int = 0
    fitness_direction: Optional[bool] = None
    diversity: Optional[float] = None
    entropy: Optional[float] = None
    health: Optional[float] = None
    uniqueness_ratio: Optional[float] = None
    duplicate_ratio: Optional[float] = None
    effective_population_size: Optional[float] = None
    generation: Optional[int] = None


def _population_capabilities(population: Any) -> PopulationCapabilities:
    schema = _population_schema(population)

    analytics: Dict[str, float] = {}
    for attr in (
        "population_analytics",
        "analytics",
        "metrics",
        "statistics",
        "selection_analytics",
    ):
        if hasattr(population, attr):
            try:
                candidate = getattr(population, attr)
                if callable(candidate):
                    candidate = candidate()
                if isinstance(candidate, Mapping):
                    analytics.update(
                        {
                            str(k): float(v)
                            for k, v in candidate.items()
                            if isinstance(v, (int, float))
                            and math.isfinite(float(v))
                        }
                    )
            except Exception:
                pass

    def _fetch_numeric(*names: str) -> Optional[float]:
        for name in names:
            if hasattr(population, name):
                try:
                    value = getattr(population, name)
                    if callable(value):
                        value = value()
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        return float(value)
                except Exception:
                    pass

        for name in names:
            if name in analytics:
                return float(analytics[name])

        return None

    def _fetch_int(*names: str) -> Optional[int]:
        for name in names:
            if hasattr(population, name):
                try:
                    value = getattr(population, name)
                    if callable(value):
                        value = value()
                    if isinstance(value, int):
                        return int(value)
                except Exception:
                    pass
        return None

    size = _population_size(population)
    generation = _fetch_int("generation", "current_generation", "step")

    return PopulationCapabilities(
        schema=schema,
        analytics=analytics,
        size=size,
        fitness_direction=(
            getattr(population, "maximize", None)
            if hasattr(population, "maximize")
            else None
        ),
        diversity=_fetch_numeric("diversity", "population_diversity"),
        entropy=_fetch_numeric("entropy", "population_entropy"),
        health=_fetch_numeric("health", "population_health_score"),
        uniqueness_ratio=_fetch_numeric(
            "uniqueness_ratio",
            "population_uniqueness_ratio",
        ),
        duplicate_ratio=_fetch_numeric(
            "duplicate_ratio",
            "population_duplicate_ratio",
        ),
        effective_population_size=_fetch_numeric(
            "effective_population_size",
        ),
        generation=generation,
    )


@dataclass(frozen=True)
class PopulationInterface:
    schema: Any
    analytics: Mapping[str, float]
    individuals: Sequence[Any]
    generation: int
    size: int
    maximize: bool

    def metric(self, name: str, default: float = 0.0) -> float:
        value = self.analytics.get(name, default)
        try:
            return float(value)
        except Exception:
            return float(default)


def _population_interface(population: Any) -> PopulationInterface:
    caps = _population_capabilities(population)

    analytics = dict(caps.analytics)

    schema = caps.schema
    individuals = _population_individuals(population)

    generation = caps.generation if caps.generation is not None else 0

    maximize = (
        caps.fitness_direction
        if caps.fitness_direction is not None
        else True
    )

    return PopulationInterface(
        schema=schema,
        analytics=analytics,
        individuals=individuals,
        generation=generation,
        size=len(individuals),
        maximize=maximize,
    )

    # def _fetch_numeric(*names: str) -> Optional[float]:
    #     for name in names:
    #         if hasattr(population, name):
    #             try:
    #                 value = getattr(population, name)
    #                 if callable(value):
    #                     value = value()
    #                 if isinstance(value, (int, float)) and math.isfinite(float(value)):
    #                     return float(value)
    #             except Exception:
    #                 pass
    #     for name in names:
    #         if name in analytics:
    #             return float(analytics[name])
    #     return None

    # def _fetch_int(*names: str) -> Optional[int]:
    #     for name in names:
    #         if hasattr(population, name):
    #             try:
    #                 value = getattr(population, name)
    #                 if callable(value):
    #                     value = value()
    #                 if isinstance(value, int):
    #                     return int(value)
    #             except Exception:
    #                 pass
    #     return None

    # size = _population_size(population)
    # generation = _fetch_int("generation", "current_generation", "step")

    # return PopulationCapabilities(
    #     schema=schema,
    #     analytics=analytics,
    #     size=size,
    #     fitness_direction=getattr(population, "maximize", None) if hasattr(population, "maximize") else None,
    #     diversity=_fetch_numeric("diversity", "population_diversity"),
    #     entropy=_fetch_numeric("entropy", "population_entropy"),
    #     health=_fetch_numeric("health", "population_health_score"),
    #     uniqueness_ratio=_fetch_numeric("uniqueness_ratio", "population_uniqueness_ratio"),
    #     duplicate_ratio=_fetch_numeric("duplicate_ratio", "population_duplicate_ratio"),
    #     effective_population_size=_fetch_numeric("effective_population_size"),
    #     generation=generation,
    # )


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class SelectionConfig:
    elitism_rate: float = 0.10
    selection_pressure: float = 1.00
    minimum_selection_pressure: float = 0.25
    maximum_selection_pressure: float = 2.50
    enable_adaptive_selection: bool = True
    enable_diversity_preservation: bool = True
    enable_elitism: bool = True
    enable_constraint_validation: bool = True
    enable_population_repair: bool = True
    enable_statistics_tracking: bool = True
    enable_metadata_tracking: bool = True
    maximize: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_finite_numeric("elitism_rate", self.elitism_rate)
        _validate_finite_numeric("selection_pressure", self.selection_pressure)
        _validate_finite_numeric("minimum_selection_pressure", self.minimum_selection_pressure)
        _validate_finite_numeric("maximum_selection_pressure", self.maximum_selection_pressure)

        if not (0.0 <= float(self.elitism_rate) <= 1.0):
            raise ValueError("elitism_rate must be in [0, 1].")
        if self.minimum_selection_pressure <= 0:
            raise ValueError("minimum_selection_pressure must be positive.")
        if self.maximum_selection_pressure <= 0:
            raise ValueError("maximum_selection_pressure must be positive.")
        if self.minimum_selection_pressure > self.maximum_selection_pressure:
            raise ValueError("minimum_selection_pressure must be <= maximum_selection_pressure.")
        if not (self.minimum_selection_pressure <= self.selection_pressure <= self.maximum_selection_pressure):
            raise ValueError("selection_pressure must lie within the configured bounds.")

        _validate_bool("enable_adaptive_selection", self.enable_adaptive_selection)
        _validate_bool("enable_diversity_preservation", self.enable_diversity_preservation)
        _validate_bool("enable_elitism", self.enable_elitism)
        _validate_bool("enable_constraint_validation", self.enable_constraint_validation)
        _validate_bool("enable_population_repair", self.enable_population_repair)
        _validate_bool("enable_statistics_tracking", self.enable_statistics_tracking)
        _validate_bool("enable_metadata_tracking", self.enable_metadata_tracking)
        _validate_bool("maximize", self.maximize)
        if self.seed is not None:
            _validate_integer("seed", self.seed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionStatistics:
    selection_calls: int = 0
    comparisons: int = 0
    survivors: int = 0
    rejected_trials: int = 0
    accepted_trials: int = 0
    elite_survivors: int = 0
    constraint_repairs: int = 0
    population_repairs: int = 0
    adaptive_updates: int = 0
    diversity_preservations: int = 0
    metadata_exports: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelectionStrategyType(str, Enum):
    GREEDY = "greedy"
    ELITIST = "elitist"
    DIVERSITY_PRESERVING = "diversity_preserving"
    ADAPTIVE = "adaptive"


@dataclass
class SelectionAnalytics:
    duplicate_ratio: float = 0.0
    entropy: float = 0.0
    health: float = 0.0
    uniqueness_ratio: float = 0.0
    diversity_gain: float = 0.0
    survival_rate: float = 0.0
    turnover: float = 0.0
    replacement_ratio: float = 0.0
    quality_score: float = 0.0
    information_gain: float = 0.0
    exploration_score: float = 0.0
    exploitation_score: float = 0.0
    balance_score: float = 0.0
    difficulty_score: float = 0.0
    search_burden: float = 0.0
    effective_population_size: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================


def fitness_margin(parent_fitness: Optional[float], trial_fitness: Optional[float], *, scale: float = 1.0) -> float:
    values = [v for v in (parent_fitness, trial_fitness) if v is not None and math.isfinite(float(v))]
    base = max([1.0] + [abs(float(v)) for v in values])
    return float(DEFAULT_FITNESS_EPS * base * max(1.0, float(scale)))


def selection_margin(parent_fitness: Optional[float], trial_fitness: Optional[float], *, selection_pressure: float = 1.0) -> float:
    if selection_pressure <= 0:
        raise ValueError("selection_pressure must be positive.")
    values = [v for v in (parent_fitness, trial_fitness) if v is not None and math.isfinite(float(v))]
    base = max([1.0] + [abs(float(v)) for v in values])
    return float(DEFAULT_FITNESS_EPS * base / max(1.0, float(selection_pressure)))


def equal_fitness(
    parent_fitness: Optional[float],
    trial_fitness: Optional[float],
    *,
    maximize: bool = True,
    margin: Optional[float] = None,
    selection_pressure: float = 1.0,
) -> bool:
    if parent_fitness is None or trial_fitness is None:
        return parent_fitness is None and trial_fitness is None
    if margin is None:
        margin = selection_margin(parent_fitness, trial_fitness, selection_pressure=selection_pressure)
    return abs(float(parent_fitness) - float(trial_fitness)) <= float(margin)


def compare_fitness(
    parent_fitness: Optional[float],
    trial_fitness: Optional[float],
    *,
    maximize: bool = True,
    margin: Optional[float] = None,
    selection_pressure: float = 1.0,
) -> int:
    """Return 1 if trial is better, -1 if parent is better, 0 if equal."""
    if parent_fitness is None and trial_fitness is None:
        return 0
    if parent_fitness is None:
        return 1
    if trial_fitness is None:
        return -1

    pf = float(parent_fitness)
    tf = float(trial_fitness)
    if not math.isfinite(pf) and not math.isfinite(tf):
        return 0
    if not math.isfinite(pf):
        return 1
    if not math.isfinite(tf):
        return -1

    if margin is None:
        margin = selection_margin(pf, tf, selection_pressure=selection_pressure)

    diff = tf - pf
    if abs(diff) <= float(margin):
        return 0
    if maximize:
        return 1 if diff > 0 else -1
    return 1 if diff < 0 else -1


def compare_individuals(
    parent: Any,
    trial: Any,
    *,
    maximize: bool = True,
    margin: Optional[float] = None,
    selection_pressure: float = 1.0,
) -> int:
    return compare_fitness(
        _individual_fitness(parent),
        _individual_fitness(trial),
        maximize=maximize,
        margin=margin,
        selection_pressure=selection_pressure,
    )


def better_individual(
    parent: Any,
    trial: Any,
    *,
    maximize: bool = True,
    margin: Optional[float] = None,
    selection_pressure: float = 1.0,
) -> Any:
    cmp = compare_individuals(
        parent,
        trial,
        maximize=maximize,
        margin=margin,
        selection_pressure=selection_pressure,
    )
    if cmp > 0:
        return trial
    return parent


def _individual_fitness(individual: Any) -> Optional[float]:
    if individual is None:
        return None
    for attr in ("fitness", "score", "objective", "value"):
        if hasattr(individual, attr):
            try:
                value = getattr(individual, attr)
                if value is None:
                    continue
                if torch.is_tensor(value):
                    value = float(value.detach().mean().item())
                else:
                    value = float(value)
                if math.isfinite(value):
                    return value
            except Exception:
                pass
    metadata = getattr(individual, "metadata", None)
    if isinstance(metadata, Mapping):
        for attr in ("fitness", "score", "objective"):
            if attr in metadata:
                try:
                    value = float(metadata[attr])
                    if math.isfinite(value):
                        return value
                except Exception:
                    pass
    return None


def _individual_to_chromosome(individual: Any) -> Optional[Tuple[Any, ...]]:
    if individual is None:
        return None
    for attr in ("to_chromosome", "chromosome", "genome", "vector", "genes"):
        value = getattr(individual, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                pass
        if value is not None:
            try:
                return tuple(value)
            except Exception:
                pass
    if isinstance(individual, Sequence) and not isinstance(individual, (str, bytes)):
        try:
            return tuple(individual)
        except Exception:
            return None
    return None


def _chromosome_signature(individual: Any) -> str:
    chromosome = _individual_to_chromosome(individual)
    if chromosome is not None:
        return _stable_hash({"chromosome": list(chromosome)})
    fitness = _individual_fitness(individual)
    return _stable_hash({"fitness": fitness, "repr": repr(individual)})


def _novelty_distance(a: Any, b: Any) -> float:
    ca = _individual_to_chromosome(a)
    cb = _individual_to_chromosome(b)
    if ca is None or cb is None:
        return 0.0
    if len(ca) != len(cb):
        return 0.0
    try:
        return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(ca, cb))))
    except Exception:
        return 0.0


# =============================================================================
# POPULATION REFLECTION / BUILDERS
# =============================================================================


def _population_individuals(population: Any) -> List[Any]:
    if population is None:
        return []
    if hasattr(population, "individuals"):
        try:
            return list(population.individuals)
        except Exception:
            pass
    if isinstance(population, Sequence) and not isinstance(population, (str, bytes)):
        return list(population)
    try:
        return list(population)
    except Exception:
        pass
    raise TypeError("Unable to iterate over population.")


def _population_schema(population: Any) -> Optional["PopulationSchema"]:
    if population is None:
        return None
    for attr in ("schema", "population_schema"):
        if hasattr(population, attr):
            try:
                schema = getattr(population, attr)
                if schema is not None:
                    return schema
            except Exception:
                pass
    return None


def _population_size(population: Any) -> int:
    try:
        return len(_population_individuals(population))
    except Exception:
        return 0


def _population_analytics(population: Any) -> Dict[str, float]:
    """
    Retrieve analytics directly from the Population subsystem whenever
    available. Local reconstruction is used only as a compatibility
    fallback for legacy Population implementations.
    """

    interface = _population_interface(population)

    if interface.analytics:
        return dict(interface.analytics)

    individuals = list(interface.individuals)

    if not individuals:
        return {
            "duplicate_ratio": 0.0,
            "entropy": 0.0,
            "health": 0.0,
            "uniqueness_ratio": 0.0,
            "effective_population_size": 0.0,
        }

    signatures = [_chromosome_signature(ind) for ind in individuals]
    counts = Counter(signatures)

    duplicate_count = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    duplicate_ratio = duplicate_count / float(len(individuals))
    uniqueness_ratio = len(counts) / float(len(individuals))

    entropy = _entropy_from_counts(list(counts.values()))
    max_entropy = math.log2(max(2, len(individuals)))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    fitness_values = [
        value
        for value in (
            _individual_fitness(ind)
            for ind in individuals
        )
        if value is not None
        and math.isfinite(float(value))
    ]

    finite_ratio = len(fitness_values) / float(len(individuals))

    if fitness_values:
        spread = _safe_std(fitness_values)
        fitness_range = max(fitness_values) - min(fitness_values)

        if fitness_range <= 0:
            fitness_balance = 1.0
        else:
            mean_abs = max(
                1e-12,
                abs(_safe_mean(fitness_values)),
            )

            fitness_balance = (
                1.0
                / (1.0 + spread / mean_abs)
            )
    else:
        fitness_balance = 0.0

    health = _clamp_float(
        0.5 * finite_ratio
        + 0.25 * (1.0 - duplicate_ratio)
        + 0.25 * fitness_balance,
        0.0,
        1.0,
    )

    effective_population = (
        1.0
        / sum(
            (count / len(individuals)) ** 2
            for count in counts.values()
        )
    )

    return {
        "duplicate_ratio": float(
            _clamp_float(
                duplicate_ratio,
                0.0,
                1.0,
            )
        ),
        "entropy": float(
            _clamp_float(
                norm_entropy,
                0.0,
                1.0,
            )
        ),
        "health": float(health),
        "uniqueness_ratio": float(
            _clamp_float(
                uniqueness_ratio,
                0.0,
                1.0,
            )
        ),
        "effective_population_size": float(
            effective_population
        ),
        "fitness_balance": float(
            _clamp_float(
                fitness_balance,
                0.0,
                1.0,
            )
        ),
    }


def _build_individual(
    chromosome: Sequence[Any],
    *,
    template: Any = None,
    generation: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    fitness: Optional[float] = None,
    schema: Any = None,
) -> Any:
    candidates: List[Any] = []
    if template is not None:
        candidates.append(type(template))
    candidates.append(DEIndividual)

    for cls in candidates:
        if cls is None:
            continue
        for ctor_name in ("from_chromosome", "from_vector", "from_genome"):
            ctor = getattr(cls, ctor_name, None)
            if callable(ctor):
                try:
                    return ctor(chromosome=tuple(chromosome), schema=schema)
                except TypeError:
                    try:
                        return ctor(tuple(chromosome))
                    except Exception:
                        pass
                except Exception:
                    pass
        try:
            kwargs: Dict[str, Any] = {"chromosome": tuple(chromosome)}
            if schema is not None:
                kwargs["schema"] = schema
            if generation is not None:
                kwargs["generation"] = int(generation)
            if metadata is not None:
                kwargs["metadata"] = dict(metadata)
            if fitness is not None:
                kwargs["fitness"] = fitness
            return cls(**kwargs)
        except Exception:
            pass
        try:
            return cls(tuple(chromosome))
        except Exception:
            pass

    raise TypeError("Unable to construct DEIndividual from chromosome.")


def _build_population(individuals: Sequence[Any], *, schema: Any = None, template: Any = None) -> Any:
    population_candidates: List[Any] = []
    if template is not None:
        population_candidates.append(type(template))
    population_candidates.append(Population)

    for cls in population_candidates:
        if cls is None:
            continue
        for ctor_name in ("from_individuals", "from_population", "create"):
            ctor = getattr(cls, ctor_name, None)
            if callable(ctor):
                try:
                    if schema is not None:
                        return ctor(individuals=list(individuals), schema=schema)
                    return ctor(individuals=list(individuals))
                except TypeError:
                    try:
                        return ctor(list(individuals))
                    except Exception:
                        pass
                except Exception:
                    pass
        try:
            kwargs: Dict[str, Any] = {"individuals": list(individuals)}
            if schema is not None:
                kwargs["schema"] = schema
            return cls(**kwargs)
        except Exception:
            pass
        try:
            return cls(list(individuals))
        except Exception:
            pass

    # Return a minimal reflective proxy if Population cannot be constructed.
    class _PopulationProxy:
        def __init__(self, inds: Sequence[Any], sch: Any = None) -> None:
            self.individuals = list(inds)
            self.schema = sch

        def __iter__(self):
            return iter(self.individuals)

        def __len__(self):
            return len(self.individuals)

    return _PopulationProxy(individuals, schema)


# =============================================================================
# POPULATION METRICS / DIAGNOSTICS
# =============================================================================


def population_duplicate_ratio(population: Any) -> float:
    return float(_population_analytics(population).get("duplicate_ratio", 0.0))


def population_entropy(population: Any) -> float:
    return float(_population_analytics(population).get("entropy", 0.0))


def population_health(population: Any) -> float:
    return float(_population_analytics(population).get("health", 0.0))


def population_uniqueness_ratio(population: Any) -> float:
    return float(_population_analytics(population).get("uniqueness_ratio", 0.0))


def effective_population_size(population: Any) -> float:
    return float(_population_analytics(population).get("effective_population_size", 0.0))


def selection_diversity_gain(before_population: Any, after_population: Any) -> float:
    return float(population_uniqueness_ratio(after_population) - population_uniqueness_ratio(before_population))


def population_entropy_change(before_population: Any, after_population: Any) -> float:
    return float(population_entropy(after_population) - population_entropy(before_population))


def population_health_change(before_population: Any, after_population: Any) -> float:
    return float(population_health(after_population) - population_health(before_population))


def diversity_preservation_score(population: Any) -> float:
    analytics = _population_analytics(population)
    score = (
        0.35 * (1.0 - analytics.get("duplicate_ratio", 0.0))
        + 0.25 * analytics.get("entropy", 0.0)
        + 0.20 * analytics.get("uniqueness_ratio", 0.0)
        + 0.20 * analytics.get("health", 0.0)
    )
    return float(_clamp_float(score, 0.0, 1.0))


# =============================================================================
# SELECTION STRATEGIES
# =============================================================================


def greedy_selection(parent: Any, trial: Any, *, maximize: bool = True, selection_pressure: float = 1.0) -> Any:
    return better_individual(parent, trial, maximize=maximize, selection_pressure=selection_pressure)


def elite_individuals(population: Any, *, elite_rate: float = 0.10, maximize: bool = True) -> List[Any]:
    individuals = _population_individuals(population)
    if not individuals:
        return []
    if elite_rate <= 0:
        return []
    if elite_rate > 1:
        raise ValueError("elite_rate must be in [0, 1].")
    elite_count = max(1, int(round(len(individuals) * float(elite_rate))))
    ranked = sorted(
        individuals,
        key=lambda ind: (
            float(_individual_fitness(ind)) if _individual_fitness(ind) is not None and math.isfinite(float(_individual_fitness(ind))) else float("-inf"),
            _chromosome_signature(ind),
        ),
        reverse=maximize,
    )
    return ranked[:elite_count]


def elite_population(population: Any, *, elite_rate: float = 0.10, maximize: bool = True, schema: Any = None) -> Any:
    elites = elite_individuals(population, elite_rate=elite_rate, maximize=maximize)
    return _build_population(elites, schema=schema or _population_schema(population), template=population)


def elite_statistics(population: Any, *, elite_rate: float = 0.10, maximize: bool = True) -> Dict[str, Any]:
    elites = elite_individuals(population, elite_rate=elite_rate, maximize=maximize)
    fitness_values = [v for v in (_individual_fitness(ind) for ind in elites) if v is not None]
    return {
        "elite_count": len(elites),
        "elite_rate": float(elite_rate),
        "elite_fitness_mean": _safe_mean(fitness_values),
        "elite_fitness_median": _safe_median(fitness_values),
        "elite_fitness_variance": _safe_variance(fitness_values),
    }


def elite_preservation_score(before_population: Any, after_population: Any, *, elite_rate: float = 0.10, maximize: bool = True) -> float:
    before_elites = elite_individuals(before_population, elite_rate=elite_rate, maximize=maximize)
    after_signatures = {_chromosome_signature(ind) for ind in _population_individuals(after_population)}
    if not before_elites:
        return 0.0
    preserved = sum(1 for ind in before_elites if _chromosome_signature(ind) in after_signatures)
    return float(preserved / len(before_elites))


def _pairwise_survivors(
    parents: Sequence[Any],
    trials: Sequence[Any],
    *,
    maximize: bool = True,
    selection_pressure: float = 1.0,
) -> Tuple[List[Any], Dict[str, int]]:
    survivors: List[Any] = []
    accepted = 0
    rejected = 0
    comparisons = 0
    for idx in range(max(len(parents), len(trials))):
        parent = parents[idx] if idx < len(parents) else None
        trial = trials[idx] if idx < len(trials) else None
        if parent is None and trial is None:
            continue
        if parent is None:
            survivors.append(trial)
            accepted += 1
            continue
        if trial is None:
            survivors.append(parent)
            rejected += 1
            continue
        comparisons += 1
        winner = greedy_selection(parent, trial, maximize=maximize, selection_pressure=selection_pressure)
        survivors.append(winner)
        if winner is trial:
            accepted += 1
        else:
            rejected += 1
    return survivors, {"accepted": accepted, "rejected": rejected, "comparisons": comparisons}


def _candidate_pool(parents: Sequence[Any], trials: Sequence[Any], selected: Sequence[Any]) -> List[Any]:
    pool = list(parents) + list(trials) + list(selected)
    unique: List[Any] = []
    seen = set()
    for ind in pool:
        sig = _chromosome_signature(ind)
        if sig not in seen:
            unique.append(ind)
            seen.add(sig)
    return unique


def _semantic_diversity_score(individual: Any) -> float:
    chromosome = _individual_to_chromosome(individual)
    if chromosome is None or len(chromosome) == 0:
        return 0.0

    rank_part: List[Any] = []
    scaling_part: List[Any] = []
    placement_part: List[Any] = []

    # Try to split LoRA chromosome into thirds without assuming exact search-space shape.
    n = len(chromosome)
    third = max(1, n // 3)
    rank_part = list(chromosome[:third])
    scaling_part = list(chromosome[third : 2 * third])
    placement_part = list(chromosome[2 * third :])

    rank_entropy = _entropy_from_counts(list(Counter(rank_part).values()))
    scaling_entropy = _entropy_from_counts(list(Counter(scaling_part).values()))
    placement_entropy = _entropy_from_counts(list(Counter(placement_part).values()))

    rank_unique = len(set(rank_part)) / float(len(rank_part)) if rank_part else 0.0
    scaling_unique = len(set(scaling_part)) / float(len(scaling_part)) if scaling_part else 0.0
    placement_unique = len(set(placement_part)) / float(len(placement_part)) if placement_part else 0.0

    return float(
        _clamp_float(
            0.30 * rank_unique
            + 0.20 * scaling_unique
            + 0.20 * placement_unique
            + 0.15 * rank_entropy
            + 0.075 * scaling_entropy
            + 0.075 * placement_entropy,
            0.0,
            1.0,
        )
    )


def diversity_preserving_selection(
    parents: Sequence[Any],
    trials: Sequence[Any],
    *,
    maximize: bool = True,
    selection_pressure: float = 1.0,
    elite_rate: float = 0.10,
    target_size: Optional[int] = None,
    candidate_pool: Optional[Sequence[Any]] = None,
) -> List[Any]:
    base_survivors, _stats = _pairwise_survivors(parents, trials, maximize=maximize, selection_pressure=selection_pressure)
    if target_size is None:
        target_size = max(len(parents), len(trials), len(base_survivors))
    if target_size <= 0:
        return []

    pool = list(candidate_pool) if candidate_pool is not None else _candidate_pool(parents, trials, base_survivors)
    elite_count = 0 if elite_rate <= 0 else max(1, int(round(target_size * float(elite_rate))))

    elites = sorted(
        pool,
        key=lambda ind: (
            float(_individual_fitness(ind)) if _individual_fitness(ind) is not None and math.isfinite(float(_individual_fitness(ind))) else float("-inf"),
            _semantic_diversity_score(ind),
            _chromosome_signature(ind),
        ),
        reverse=maximize,
    )[:elite_count]

    selected: List[Any] = []
    selected_sigs: set[str] = set()

    def add_if_novel(ind: Any) -> None:
        sig = _chromosome_signature(ind)
        if sig not in selected_sigs:
            selected.append(ind)
            selected_sigs.add(sig)

    for ind in elites:
        add_if_novel(ind)

    ranked = sorted(
        pool,
        key=lambda ind: (
            float(_individual_fitness(ind)) if _individual_fitness(ind) is not None and math.isfinite(float(_individual_fitness(ind))) else float("-inf"),
            _semantic_diversity_score(ind),
            _chromosome_signature(ind),
        ),
        reverse=maximize,
    )

    while len(selected) < target_size and ranked:
        best_ind = None
        best_score = float("-inf")
        for ind in ranked:
            sig = _chromosome_signature(ind)
            if sig in selected_sigs:
                continue
            fitness = _individual_fitness(ind)
            fitness_score = float(fitness) if fitness is not None and math.isfinite(float(fitness)) else float("-inf")
            novelty = 0.0
            if selected:
                novelty = _safe_mean([_novelty_distance(ind, other) for other in selected])
            semantic = _semantic_diversity_score(ind)
            score = fitness_score + 0.10 * novelty + 0.15 * semantic if maximize else -fitness_score + 0.10 * novelty + 0.15 * semantic
            if score > best_score:
                best_score = score
                best_ind = ind
        if best_ind is None:
            break
        add_if_novel(best_ind)

    for ind in base_survivors:
        if len(selected) >= target_size:
            break
        add_if_novel(ind)

    return selected[:target_size]


def adaptive_selection(
    parents: Sequence[Any],
    trials: Sequence[Any],
    *,
    maximize: bool = True,
    selection_pressure: float = 1.0,
    minimum_pressure: float = 0.25,
    maximum_pressure: float = 2.50,
    diversity_analytics: Optional[Mapping[str, float]] = None,
    success_rate: float = 0.5,
    repair_frequency: float = 0.0,
    difficulty_score: float = 0.5,
    target_size: Optional[int] = None,
    enable_diversity_preservation: bool = True,
    elite_rate: float = 0.10,
    pressure_history: Optional[Sequence[float]] = None,
    population_history: Optional[
        Sequence[Mapping[str, float]]
    ] = None,
    ) -> Tuple[List[Any], float]:
    analytics = dict(diversity_analytics or {})
    duplicate_ratio = float(analytics.get("duplicate_ratio", 0.0))
    entropy = float(analytics.get("entropy", 0.0))
    health = float(analytics.get("health", 0.0))
    uniqueness = float(analytics.get("uniqueness_ratio", 0.0))
    previous_pressure = float(selection_pressure)

    history = list(pressure_history or [])
    pop_history = list(population_history or [])
    pressure_trend = 0.0
    if len(history) > 1:
        pressure_trend = float(history[-1] - history[0]) / float(len(history) - 1)

    diversity_trend = 0.0
    health_trend = 0.0
    if len(pop_history) > 1:
        first = pop_history[0]
        last = pop_history[-1]
        diversity_trend = float(last.get("uniqueness_ratio", 0.0) - first.get("uniqueness_ratio", 0.0))
        health_trend = float(last.get("health", 0.0) - first.get("health", 0.0))

    exploration_signal = (
        0.30 * (1.0 - duplicate_ratio)
        + 0.20 * entropy
        + 0.15 * uniqueness
        + 0.15 * max(0.0, diversity_trend)
        + 0.10 * max(0.0, health_trend)
    )

    exploitation_signal = (
        0.25 * health
        + 0.20 * success_rate
        + 0.15 * difficulty_score
        + 0.15 * (1.0 - repair_frequency)
        + 0.10 * max(0.0, -pressure_trend)
    )

    history = list(pressure_history or [])
    population_records = list(population_history or [])

    pressure_trend = 0.0
    if len(history) >= 2:
        pressure_trend = (
            history[-1] - history[0]
        ) / float(len(history) - 1)

    diversity_trend = 0.0
    health_trend = 0.0

    if len(population_records) >= 2:
        first = population_records[0]
        last = population_records[-1]

        diversity_trend = (
            last.get("uniqueness_ratio", 0.0)
            - first.get("uniqueness_ratio", 0.0)
        )

        health_trend = (
            last.get("health", 0.0)
            - first.get("health", 0.0)
        )

        history_score = (
            0.40 * max(0.0, diversity_trend)
            + 0.40 * max(0.0, health_trend)
            + 0.20 * max(0.0, -pressure_trend)
        )

    adaptive_score = (
        exploration_signal
        + exploitation_signal
        + 0.15 * history_score
        - 0.35 * duplicate_ratio
        - 0.10 * repair_frequency
    )

    new_pressure = previous_pressure * (0.70 + adaptive_score)
    new_pressure = _clamp_float(new_pressure, minimum_pressure, maximum_pressure)

    if enable_diversity_preservation and (
        duplicate_ratio > 0.10
        or entropy < 0.35
        or health < 0.40
        or diversity_trend < 0.0
    ):
        selected = diversity_preserving_selection(
            parents,
            trials,
            maximize=maximize,
            selection_pressure=new_pressure,
            elite_rate=elite_rate,
            target_size=target_size,
        )
    else:
        selected, _ = _pairwise_survivors(parents, trials, maximize=maximize, selection_pressure=new_pressure)
        if target_size is not None:
            selected = selected[:target_size]

    return selected, new_pressure


# =============================================================================
# SELECTION ENGINE
# =============================================================================


class SelectionEngine:
    """Primary public API for DE survivor selection."""

    def __init__(
        self,
        config: Optional[SelectionConfig] = None,
        *,
        population: Optional[Any] = None,
        schema: Optional[Any] = None,
        source: Any = None,
        layer_names: Optional[Sequence[str]] = None,
        rank_config: Any = None,
        scaling_config: Any = None,
        placement_config: Any = None,
    ) -> None:
        self.config = config or SelectionConfig()
        if not isinstance(self.config, SelectionConfig):
            raise TypeError("config must be SelectionConfig.")
        self._rng = random.Random(self.config.seed)
        self.statistics_tracker = SelectionStatistics()
        self._selection_history: List[Dict[str, Any]] = []
        self._pressure_history: List[float] = [float(self.config.selection_pressure)]
        self._population_history: List[Dict[str, Any]] = []
        self._last_report: Dict[str, Any] = {}

        self.population = population
        self.schema = schema or _population_schema(population)
        if self.schema is None:
            self.schema = _infer_schema(
                rank_config=rank_config,
                scaling_config=scaling_config,
                placement_config=placement_config,
                source=source if source is not None else population,
                layer_names=layer_names,
            )

        self.rank_config = rank_config
        self.scaling_config = scaling_config
        self.placement_config = placement_config

    # ------------------------------------------------------------------
    # INTERNAL UTILITIES
    # ------------------------------------------------------------------

    def _selection_pressure(self) -> float:
        return float(self._pressure_history[-1] if self._pressure_history else self.config.selection_pressure)

    def _record_history(self, report: Mapping[str, Any]) -> None:
        self._selection_history.append(dict(report))
        self._pressure_history.append(float(report.get("selection_pressure", self._selection_pressure())))
        if len(self._selection_history) > DEFAULT_HISTORY_LIMIT:
            self._selection_history = self._selection_history[-DEFAULT_HISTORY_LIMIT:]
        if len(self._pressure_history) > DEFAULT_HISTORY_LIMIT:
            self._pressure_history = self._pressure_history[-DEFAULT_HISTORY_LIMIT:]
        self._population_history.append(
            {
                "duplicate_ratio": float(report.get("duplicate_ratio", 0.0)),
                "entropy": float(report.get("entropy", 0.0)),
                "health": float(report.get("health", 0.0)),
                "quality_score": float(report.get("population_quality", 0.0)),
            }
        )
        if len(self._population_history) > DEFAULT_HISTORY_LIMIT:
            self._population_history = self._population_history[-DEFAULT_HISTORY_LIMIT:]

    def _target_size(self, parents: Sequence[Any], trials: Sequence[Any]) -> int:
        if len(parents) > 0:
            return len(parents)
        return len(trials)

    def _objective_maximize(self, maximize: Optional[bool]) -> bool:
        return self.config.maximize if maximize is None else bool(maximize)

    def _individual_summary(self, individual: Any) -> Dict[str, Any]:
        return {
            "fitness": _individual_fitness(individual),
            "signature": _chromosome_signature(individual),
            "has_chromosome": _individual_to_chromosome(individual) is not None,
        }

    def _fitness_values(self, population: Any) -> List[float]:
        return [v for v in (_individual_fitness(ind) for ind in _population_individuals(population)) if v is not None and math.isfinite(float(v))]

    def _population_quality_from_analytics(self, analytics: Mapping[str, float]) -> float:
        duplicate_ratio = float(analytics.get("duplicate_ratio", 0.0))
        entropy = float(analytics.get("entropy", 0.0))
        health = float(analytics.get("health", 0.0))
        uniqueness = float(analytics.get("uniqueness_ratio", 0.0))
        return float(_clamp_float(0.30 * (1.0 - duplicate_ratio) + 0.25 * entropy + 0.25 * health + 0.20 * uniqueness, 0.0, 1.0))

    def _population_analytics(self, population: Any) -> Dict[str, float]:
        return _population_analytics(population)

    def _adapt_pressure(self, parents: Any, trials: Any, selected: Any, *, repair_frequency: float = 0.0) -> float:
        if not self.config.enable_adaptive_selection:
            return self._selection_pressure()

        p_analytics = self._population_analytics(parents)
        s_analytics = self._population_analytics(selected)
        current_pressure = self._selection_pressure()
        success_rate = self.selection_success_rate(parents, trials, selected)
        survival_rate = self.population_survival_rate(parents, selected)
        difficulty_score = self.selection_difficulty_score(parents, trials, selected)
        adaptive_score = (
            0.30 * p_analytics.get("health", 0.0)
            + 0.15 * p_analytics.get("entropy", 0.0)
            + 0.10 * s_analytics.get("uniqueness_ratio", 0.0)
            + 0.15 * success_rate
            + 0.10 * survival_rate
            + 0.10 * difficulty_score
            - 0.25 * p_analytics.get("duplicate_ratio", 0.0)
            - 0.10 * repair_frequency
        )
        new_pressure = current_pressure * (0.75 + adaptive_score)
        new_pressure = _clamp_float(new_pressure, self.config.minimum_selection_pressure, self.config.maximum_selection_pressure)
        self.statistics_tracker.adaptive_updates += 1
        return float(new_pressure)

    # ------------------------------------------------------------------
    # COMPARISON API
    # ------------------------------------------------------------------

    def compare_individuals(self, parent: Any, trial: Any, *, maximize: Optional[bool] = None, margin: Optional[float] = None) -> int:
        return compare_individuals(parent, trial, maximize=self._objective_maximize(maximize), margin=margin, selection_pressure=self._selection_pressure())

    def compare_fitness(self, parent_fitness: Optional[float], trial_fitness: Optional[float], *, maximize: Optional[bool] = None, margin: Optional[float] = None) -> int:
        return compare_fitness(parent_fitness, trial_fitness, maximize=self._objective_maximize(maximize), margin=margin, selection_pressure=self._selection_pressure())

    def better_individual(self, parent: Any, trial: Any, *, maximize: Optional[bool] = None, margin: Optional[float] = None) -> Any:
        return better_individual(parent, trial, maximize=self._objective_maximize(maximize), margin=margin, selection_pressure=self._selection_pressure())

    def equal_fitness(self, parent_fitness: Optional[float], trial_fitness: Optional[float], *, maximize: Optional[bool] = None, margin: Optional[float] = None) -> bool:
        return equal_fitness(parent_fitness, trial_fitness, maximize=self._objective_maximize(maximize), margin=margin, selection_pressure=self._selection_pressure())

    def fitness_margin(self, parent_fitness: Optional[float], trial_fitness: Optional[float]) -> float:
        return fitness_margin(parent_fitness, trial_fitness, scale=self._selection_pressure())

    def selection_margin(self, parent_fitness: Optional[float], trial_fitness: Optional[float]) -> float:
        return selection_margin(parent_fitness, trial_fitness, selection_pressure=self._selection_pressure())

    # ------------------------------------------------------------------
    # POPULATION VALIDATION / REPAIR
    # ------------------------------------------------------------------

    def validate_population(self, population: Any) -> bool:
        individuals = _population_individuals(population)
        if not individuals:
            raise ValueError("population cannot be empty.")

        for ind in individuals:
            fitness = _individual_fitness(ind)
            if fitness is not None and not math.isfinite(float(fitness)):
                raise ValueError("population contains non-finite fitness values.")
            if self.schema is not None and hasattr(self.schema, "chromosome_length"):
                chromosome = _individual_to_chromosome(ind)
                if chromosome is not None and len(chromosome) != int(getattr(self.schema, "chromosome_length")):
                    raise ValueError("individual chromosome length mismatch.")
        return True

    def repair_invalid_individuals(self, population: Any, candidate_pool: Optional[Sequence[Any]] = None) -> Any:
        individuals = _population_individuals(population)
        repaired: List[Any] = []
        for ind in individuals:
            valid = True
            fitness = _individual_fitness(ind)
            if fitness is not None and not math.isfinite(float(fitness)):
                valid = False
            if self.schema is not None and hasattr(self.schema, "chromosome_length"):
                chromosome = _individual_to_chromosome(ind)
                if chromosome is not None and len(chromosome) != int(getattr(self.schema, "chromosome_length")):
                    valid = False
            if valid:
                repaired.append(ind)

        if len(repaired) == len(individuals):
            return _build_population(repaired, schema=self.schema, template=population)

        self.statistics_tracker.constraint_repairs += len(individuals) - len(repaired)
        if not self.config.enable_population_repair:
            raise ValueError("Invalid individuals detected and repair is disabled.")

        pool = list(candidate_pool or []) + list(individuals)
        pool = [ind for ind in pool if _individual_fitness(ind) is None or math.isfinite(float(_individual_fitness(ind)))]
        if self.schema is not None and hasattr(self.schema, "chromosome_length"):
            chrom_len = int(getattr(self.schema, "chromosome_length"))
            pool = [ind for ind in pool if _individual_to_chromosome(ind) is None or len(_individual_to_chromosome(ind)) == chrom_len]

        needed = len(individuals) - len(repaired)
        seen = {_chromosome_signature(ind) for ind in repaired}
        for ind in sorted(pool, key=lambda x: (_individual_fitness(x) if _individual_fitness(x) is not None else float("-inf"), _chromosome_signature(x)), reverse=self.config.maximize):
            if needed <= 0:
                break
            sig = _chromosome_signature(ind)
            if sig in seen:
                continue
            repaired.append(ind)
            seen.add(sig)
            needed -= 1

        if len(repaired) < len(individuals):
            raise ValueError("Unable to repair population to full size.")
        self.statistics_tracker.population_repairs += 1
        return _build_population(repaired, schema=self.schema, template=population)

    def repair_duplicates(self, population: Any, candidate_pool: Optional[Sequence[Any]] = None) -> Any:
        individuals = _population_individuals(population)
        if not individuals:
            raise ValueError("population cannot be empty.")
        seen = set()
        unique: List[Any] = []
        duplicates = 0
        for ind in individuals:
            sig = _chromosome_signature(ind)
            if sig in seen:
                duplicates += 1
                continue
            seen.add(sig)
            unique.append(ind)

        if duplicates == 0:
            return _build_population(unique, schema=self.schema, template=population)

        self.statistics_tracker.constraint_repairs += duplicates
        if not self.config.enable_population_repair:
            raise ValueError("Duplicate individuals detected and repair is disabled.")

        pool = list(candidate_pool or []) + list(individuals)
        pool = sorted(pool, key=lambda x: (_individual_fitness(x) if _individual_fitness(x) is not None else float("-inf"), _chromosome_signature(x)), reverse=self.config.maximize)
        for ind in pool:
            if len(unique) >= len(individuals):
                break
            sig = _chromosome_signature(ind)
            if sig in seen:
                continue
            unique.append(ind)
            seen.add(sig)

        if len(unique) < len(individuals):
            raise ValueError("Unable to repair duplicate population to full size.")
        self.statistics_tracker.population_repairs += 1
        return _build_population(unique, schema=self.schema, template=population)

    def repair_population(self, population: Any, candidate_pool: Optional[Sequence[Any]] = None) -> Any:
        repaired = population
        if self.config.enable_constraint_validation:
            try:
                self.validate_population(repaired)
            except Exception:
                if self.config.enable_population_repair:
                    repaired = self.repair_invalid_individuals(repaired, candidate_pool=candidate_pool)
                else:
                    raise
        if self.config.enable_diversity_preservation:
            try:
                repaired = self.repair_duplicates(repaired, candidate_pool=candidate_pool)
            except Exception:
                if not self.config.enable_population_repair:
                    raise
        if self.config.enable_constraint_validation:
            self.validate_population(repaired)
        return repaired

    def validate_selection(self, population: Any) -> bool:
        return self.validate_population(population)

    # ------------------------------------------------------------------
    # BASIC SURVIVOR SELECTION
    # ------------------------------------------------------------------

    def select_survivor(self, parent: Any, trial: Any, *, maximize: Optional[bool] = None, margin: Optional[float] = None) -> Any:
        self.statistics_tracker.comparisons += 1
        winner = self.better_individual(parent, trial, maximize=maximize, margin=margin)
        if winner is trial:
            self.statistics_tracker.accepted_trials += 1
        else:
            self.statistics_tracker.rejected_trials += 1
        self.statistics_tracker.survivors += 1
        return winner

    def greedy_selection(self, parents: Sequence[Any], trials: Sequence[Any], *, maximize: Optional[bool] = None, selection_pressure: Optional[float] = None) -> List[Any]:
        maximize = self._objective_maximize(maximize)
        pressure = self._selection_pressure() if selection_pressure is None else float(selection_pressure)
        survivors, stats = _pairwise_survivors(parents, trials, maximize=maximize, selection_pressure=pressure)
        self.statistics_tracker.comparisons += stats["comparisons"]
        self.statistics_tracker.accepted_trials += stats["accepted"]
        self.statistics_tracker.rejected_trials += stats["rejected"]
        self.statistics_tracker.survivors += len(survivors)
        return survivors

    def _adaptive_elite_rate(self, parent_population: Any, selected_population: Any) -> float:
        parent_caps = _population_capabilities(parent_population)
        selected_caps = _population_capabilities(selected_population)

        diversity = selected_caps.uniqueness_ratio
        if diversity is None:
            diversity = population_uniqueness_ratio(selected_population)

        health = selected_caps.health
        if health is None:
            health = population_health(selected_population)

        generation = selected_caps.generation or parent_caps.generation or 0
        pressure = self._selection_pressure()

        early_phase = 1.0 / (1.0 + max(0, generation))
        convergence_bias = 1.0 - _clamp_float(diversity, 0.0, 1.0)
        health_bias = _clamp_float(health, 0.0, 1.0)

        elite_rate = self.config.elitism_rate
        elite_rate *= 0.75 + 0.25 * health_bias
        elite_rate *= 0.75 + 0.25 * convergence_bias
        elite_rate *= 0.85 + 0.15 * early_phase
        elite_rate *= 0.90 + 0.10 * (pressure / self.config.maximum_selection_pressure)

        return _clamp_float(elite_rate, 0.01 if self.config.enable_elitism else 0.0, 1.0)


    def elitist_selection(
        self,
        parents: Sequence[Any],
        trials: Sequence[Any],
        *,
        maximize: Optional[bool] = None,
        elite_rate: Optional[float] = None,
        target_size: Optional[int] = None,
    ) -> List[Any]:
        maximize = self._objective_maximize(maximize)
        target_size = self._target_size(parents, trials) if target_size is None else int(target_size)
        if target_size <= 0:
            return []

        greedy = self.greedy_selection(parents, trials, maximize=maximize)
        pool = _candidate_pool(parents, trials, greedy)

        if elite_rate is None:
            elite_rate = self._adaptive_elite_rate(parents, greedy)
        elite_rate = float(elite_rate)
        if not (0.0 <= elite_rate <= 1.0):
            raise ValueError("elite_rate must be in [0, 1].")

        elites = elite_individuals(pool, elite_rate=elite_rate, maximize=maximize)
        self.statistics_tracker.elite_survivors += len(elites)

        survivors: List[Any] = []
        seen: set[str] = set()

        for ind in elites:
            sig = _chromosome_signature(ind)
            if sig not in seen:
                survivors.append(ind)
                seen.add(sig)

        for ind in greedy:
            if len(survivors) >= target_size:
                break
            sig = _chromosome_signature(ind)
            if sig in seen:
                continue
            survivors.append(ind)
            seen.add(sig)

        if len(survivors) < target_size:
            for ind in pool:
                if len(survivors) >= target_size:
                    break
                sig = _chromosome_signature(ind)
                if sig in seen:
                    continue
                survivors.append(ind)
                seen.add(sig)

        return survivors[:target_size]

    def diversity_preserving_selection(self, parents: Sequence[Any], trials: Sequence[Any], *, maximize: Optional[bool] = None, elite_rate: Optional[float] = None, target_size: Optional[int] = None) -> List[Any]:
        maximize = self._objective_maximize(maximize)
        elite_rate = self.config.elitism_rate if elite_rate is None else float(elite_rate)
        target_size = self._target_size(parents, trials) if target_size is None else int(target_size)
        selected = diversity_preserving_selection(
            parents,
            trials,
            maximize=maximize,
            selection_pressure=self._selection_pressure(),
            elite_rate=elite_rate,
            target_size=target_size,
        )
        self.statistics_tracker.diversity_preservations += 1
        return selected

    def adaptive_selection(self, parents: Sequence[Any], trials: Sequence[Any], *, maximize: Optional[bool] = None, target_size: Optional[int] = None) -> Tuple[List[Any], float]:
        maximize = self._objective_maximize(maximize)
        target_size = self._target_size(parents, trials) if target_size is None else int(target_size)
        parent_pop = _build_population(parents, schema=self.schema)
        trial_pop = _build_population(trials, schema=self.schema)
        selected_pre, pressure = adaptive_selection(
            parents,
            trials,
            maximize=maximize,
            selection_pressure=self._selection_pressure(),
            minimum_pressure=self.config.minimum_selection_pressure,
            maximum_pressure=self.config.maximum_selection_pressure,
            diversity_analytics=self._population_analytics(parent_pop),
            success_rate=self.selection_success_rate(parent_pop, trial_pop, None),
            repair_frequency=0.0,
            difficulty_score=self.selection_difficulty_score(parent_pop, trial_pop, None),
            target_size=target_size,
            enable_diversity_preservation=self.config.enable_diversity_preservation,
            elite_rate=self.config.elitism_rate,
            pressure_history=self._pressure_history,
        )
        self.statistics_tracker.adaptive_updates += 1
        return selected_pre, pressure

    # ------------------------------------------------------------------
    # HIGH-LEVEL SELECTION PIPELINE
    # ------------------------------------------------------------------

    def select(
        self,
        parent_population: Any,
        trial_population: Any,
        *,
        strategy: Optional[str] = None,
        maximize: Optional[bool] = None,
    ) -> Any:
        parents = _population_individuals(parent_population)
        trials = _population_individuals(trial_population)
        selected = self.select_population(parent_population, trial_population, strategy=strategy, maximize=maximize)
        self.population = selected
        return selected

    def select_population(
        self,
        parent_population: Any,
        trial_population: Any,
        *,
        strategy: Optional[str] = None,
        maximize: Optional[bool] = None,
    ) -> Any:
        parents = _population_individuals(parent_population)
        trials = _population_individuals(trial_population)
        target_size = self._target_size(parents, trials)
        if target_size <= 0:
            raise ValueError("parent_population and trial_population cannot both be empty.")

        maximize = self._objective_maximize(maximize)
        strategy_name = (strategy or (SelectionStrategyType.ADAPTIVE.value if self.config.enable_adaptive_selection else SelectionStrategyType.GREEDY.value)).lower().strip()
        self.statistics_tracker.selection_calls += 1

        if strategy_name == SelectionStrategyType.GREEDY.value:
            survivors = self.greedy_selection(parents, trials, maximize=maximize)
            pressure = self._selection_pressure()
        elif strategy_name == SelectionStrategyType.ELITIST.value:
            survivors = self.elitist_selection(parents, trials, maximize=maximize, elite_rate=self.config.elitism_rate, target_size=target_size)
            pressure = self._selection_pressure()
        elif strategy_name == SelectionStrategyType.DIVERSITY_PRESERVING.value:
            survivors = self.diversity_preserving_selection(parents, trials, maximize=maximize, elite_rate=self.config.elitism_rate, target_size=target_size)
            pressure = self._selection_pressure()
        elif strategy_name == SelectionStrategyType.ADAPTIVE.value:
            survivors, pressure = self.adaptive_selection(parents, trials, maximize=maximize, target_size=target_size)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy_name}")

        candidate_pool = _candidate_pool(parents, trials, survivors)
        survivor_pop = _build_population(survivors, schema=self.schema, template=parent_population)
        if self.config.enable_population_repair:
            survivor_pop = self.repair_population(survivor_pop, candidate_pool=candidate_pool)
        elif self.config.enable_constraint_validation:
            self.validate_population(survivor_pop)

        if self.config.enable_elitism and strategy_name != SelectionStrategyType.ELITIST.value:
            elite_count = max(1, int(round(target_size * self.config.elitism_rate))) if self.config.elitism_rate > 0 else 0
            if elite_count > 0:
                elites = elite_individuals(parent_population, elite_rate=self.config.elitism_rate, maximize=maximize)
                current = _population_individuals(survivor_pop)
                current_sigs = {_chromosome_signature(ind) for ind in current}
                elite_added = 0
                for ind in elites:
                    if elite_added >= elite_count:
                        break
                    sig = _chromosome_signature(ind)
                    if sig not in current_sigs:
                        current.insert(0, ind)
                        current_sigs.add(sig)
                        elite_added += 1
                if elite_added > 0:
                    current = current[:target_size]
                    survivor_pop = _build_population(current, schema=self.schema, template=parent_population)
                    self.statistics_tracker.elite_survivors += elite_added

        if self.config.enable_diversity_preservation:
            survivor_pop = self.repair_duplicates(survivor_pop, candidate_pool=candidate_pool)

        if self.config.enable_constraint_validation:
            self.validate_population(survivor_pop)

        self.config.selection_pressure = float(_clamp_float(pressure, self.config.minimum_selection_pressure, self.config.maximum_selection_pressure))
        self._pressure_history[-1] = self.config.selection_pressure

        report = self.selection_report(parent_population, trial_population, survivor_pop, strategy=strategy_name)
        self._record_history(report)
        self._last_report = report
        return survivor_pop

    def population_update(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        report = self.selection_report(parent_population, trial_population, selected_population)
        self._record_history(report)
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # SELECTION ANALYTICS
    # ------------------------------------------------------------------

    def selection_success_rate(self, parent_population: Any, trial_population: Any, selected_population: Optional[Any]) -> float:
        parents = _population_individuals(parent_population)
        trials = _population_individuals(trial_population)
        if selected_population is None:
            if not parents:
                return 0.0
            paired = min(len(parents), len(trials))
            if paired == 0:
                return 0.0
            accepted = 0
            for i in range(paired):
                if compare_individuals(parents[i], trials[i], maximize=self.config.maximize, selection_pressure=self._selection_pressure()) > 0:
                    accepted += 1
            return accepted / float(paired)
        selected = _population_individuals(selected_population)
        if not trials:
            return 0.0
        accepted = sum(1 for ind in selected if _chromosome_signature(ind) in {_chromosome_signature(t) for t in trials})
        return accepted / float(max(1, len(trials)))

    def population_survival_rate(self, parent_population: Any, selected_population: Any) -> float:
        parents = _population_individuals(parent_population)
        selected = _population_individuals(selected_population)
        if not parents:
            return 0.0
        parent_sigs = {_chromosome_signature(ind) for ind in parents}
        surviving = sum(1 for ind in selected if _chromosome_signature(ind) in parent_sigs)
        return surviving / float(len(parents))

    def population_turnover(self, parent_population: Any, selected_population: Any) -> float:
        return float(1.0 - self.population_survival_rate(parent_population, selected_population))

    def replacement_ratio(self, parent_population: Any, selected_population: Any) -> float:
        return float(self.population_turnover(parent_population, selected_population))

    def population_improvement(self, parent_population: Any, selected_population: Any) -> float:
        parents = [v for v in (_individual_fitness(ind) for ind in _population_individuals(parent_population)) if v is not None and math.isfinite(float(v))]
        selected = [v for v in (_individual_fitness(ind) for ind in _population_individuals(selected_population)) if v is not None and math.isfinite(float(v))]
        if not parents or not selected:
            return 0.0
        if self.config.maximize:
            return float(_safe_mean(selected) - _safe_mean(parents))
        return float(_safe_mean(parents) - _safe_mean(selected))

    def population_quality_score(self, population: Any) -> float:
        return self._population_quality_from_analytics(self._population_analytics(population))

    def selection_health_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        parent_a = self._population_analytics(parent_population)
        selected_a = self._population_analytics(selected_population)
        return float(_clamp_float(0.40 * selected_a.get("health", 0.0) + 0.30 * (1.0 - selected_a.get("duplicate_ratio", 0.0)) + 0.15 * selected_a.get("uniqueness_ratio", 0.0) + 0.15 * parent_a.get("health", 0.0), 0.0, 1.0))

    def selection_stability_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        turnover = self.population_turnover(parent_population, selected_population)
        duplicate_ratio = population_duplicate_ratio(selected_population)
        return float(_clamp_float(1.0 - 0.5 * turnover - 0.5 * duplicate_ratio, 0.0, 1.0))

    def selection_readiness_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        analytics = self._population_analytics(selected_population)
        return float(_clamp_float(0.35 * analytics.get("health", 0.0) + 0.35 * analytics.get("entropy", 0.0) + 0.30 * analytics.get("uniqueness_ratio", 0.0), 0.0, 1.0))

    def selection_efficiency_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        improvement = self.population_improvement(parent_population, selected_population)
        burden = self.selection_search_burden(parent_population, trial_population, selected_population)
        return float(_clamp_float(0.5 * (1.0 / (1.0 + abs(burden))) + 0.5 * (1.0 if improvement >= 0 else 0.0), 0.0, 1.0))

    def selection_difficulty_score(self, parent_population: Any, trial_population: Any, selected_population: Optional[Any]) -> float:
        parent_a = self._population_analytics(parent_population)
        trial_a = self._population_analytics(trial_population)
        duplicate_pressure = 0.5 * (parent_a.get("duplicate_ratio", 0.0) + trial_a.get("duplicate_ratio", 0.0))
        health_gap = abs(parent_a.get("health", 0.0) - trial_a.get("health", 0.0))
        success_rate = self.selection_success_rate(parent_population, trial_population, selected_population)
        difficulty = 0.45 * duplicate_pressure + 0.25 * health_gap + 0.30 * (1.0 - success_rate)
        return float(_clamp_float(difficulty, 0.0, 1.0))

    def selection_search_burden(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        parent_size = _population_size(parent_population)
        trial_size = _population_size(trial_population)
        selected_size = _population_size(selected_population)
        return float(parent_size + trial_size + selected_size)

    def effective_population_size(self, population: Any) -> float:
        return effective_population_size(population)

    def selection_information_gain(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        before = self._population_analytics(parent_population)
        after = self._population_analytics(selected_population)
        return float(max(0.0, after.get("entropy", 0.0) - before.get("entropy", 0.0)) + max(0.0, after.get("uniqueness_ratio", 0.0) - before.get("uniqueness_ratio", 0.0)))

    def selection_exploration_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        after = self._population_analytics(selected_population)
        info_gain = self.selection_information_gain(parent_population, trial_population, selected_population)
        return float(_clamp_float(0.40 * after.get("entropy", 0.0) + 0.30 * after.get("uniqueness_ratio", 0.0) + 0.20 * (1.0 - after.get("duplicate_ratio", 0.0)) + 0.10 * min(1.0, info_gain), 0.0, 1.0))

    def selection_exploitation_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        pressure = self._selection_pressure()
        success_rate = self.selection_success_rate(parent_population, trial_population, selected_population)
        improvement = self.population_improvement(parent_population, selected_population)
        return float(_clamp_float(0.35 * (pressure / self.config.maximum_selection_pressure) + 0.35 * success_rate + 0.30 * (1.0 if improvement >= 0 else 0.0), 0.0, 1.0))

    def selection_balance_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        exploration = self.selection_exploration_score(parent_population, trial_population, selected_population)
        exploitation = self.selection_exploitation_score(parent_population, trial_population, selected_population)
        return float(_clamp_float(1.0 - abs(exploration - exploitation), 0.0, 1.0))

    def selection_learning_score(self, parent_population: Any, trial_population: Any, selected_population: Any) -> float:
        pressure_trend = self.selection_pressure_trend()
        improvement = self.population_improvement(parent_population, selected_population)
        balance = self.selection_balance_score(parent_population, trial_population, selected_population)
        return float(_clamp_float(0.40 * balance + 0.35 * (1.0 if improvement >= 0 else 0.0) + 0.25 * (1.0 - abs(pressure_trend)), 0.0, 1.0))

    def selection_pressure_history(self) -> List[float]:
        return list(self._pressure_history)

    def selection_pressure_trend(self) -> float:
        values = self._pressure_history
        if len(values) <= 1:
            return 0.0
        return float(values[-1] - values[0]) / float(len(values) - 1)

    def population_trend(self) -> Dict[str, float]:
        if len(self._population_history) <= 1:
            return {"duplicate_ratio": 0.0, "entropy": 0.0, "health": 0.0, "quality_score": 0.0}
        first = self._population_history[0]
        last = self._population_history[-1]
        return {
            "duplicate_ratio": float(last.get("duplicate_ratio", 0.0) - first.get("duplicate_ratio", 0.0)),
            "entropy": float(last.get("entropy", 0.0) - first.get("entropy", 0.0)),
            "health": float(last.get("health", 0.0) - first.get("health", 0.0)),
            "quality_score": float(last.get("quality_score", 0.0) - first.get("quality_score", 0.0)),
        }

    def stagnation_score(self) -> float:
        if len(self._selection_history) < 2:
            return 0.0
        recent = self._selection_history[-min(8, len(self._selection_history)) :]
        improvements = [float(item.get("population_improvement", 0.0)) for item in recent]
        diversity = [float(item.get("diversity_gain", 0.0)) for item in recent]
        score = 1.0 - _clamp_float(0.5 * abs(_safe_mean(improvements)) + 0.5 * abs(_safe_mean(diversity)), 0.0, 1.0)
        return float(_clamp_float(score, 0.0, 1.0))

    def convergence_velocity(self) -> float:
        if len(self._selection_history) < 2:
            return 0.0
        recent = self._selection_history[-min(8, len(self._selection_history)) :]
        improvements = [float(item.get("population_improvement", 0.0)) for item in recent]
        return float(_safe_mean(improvements))

    def selection_health_score_from_latest(self) -> float:
        if not self._last_report:
            return 0.0
        return float(self._last_report.get("selection_health_score", 0.0))

    # ------------------------------------------------------------------
    # REPRODUCIBILITY
    # ------------------------------------------------------------------

    def selection_hash(self) -> str:
        payload = {
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict() if hasattr(self.schema, "to_dict") else repr(self.schema),
            "statistics": self.statistics_tracker.to_dict(),
            "history": self._selection_history,
            "pressure_history": self._pressure_history,
        }
        return _stable_hash(payload)

    def selection_signature(self) -> str:
        return self.selection_hash()

    def selection_fingerprint(self) -> Dict[str, Any]:
        return {
            "hash": self.selection_hash(),
            "signature": self.selection_signature(),
            "config": self.config.to_dict(),
            "statistics": self.statistics_tracker.to_dict(),
            "history_length": len(self._selection_history),
            "pressure_history_length": len(self._pressure_history),
        }

    # ------------------------------------------------------------------
    # METADATA / REPORTING
    # ------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        return {
            "strategy": self._last_report.get("strategy", None),
            "selection_pressure": self._selection_pressure(),
            "accepted_trials": self.statistics_tracker.accepted_trials,
            "rejected_trials": self.statistics_tracker.rejected_trials,
            "population_quality": self._last_report.get("population_quality", 0.0),
            "diversity_gain": self._last_report.get("diversity_gain", 0.0),
            "selection_hash": self.selection_hash(),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "statistics": self.statistics_tracker.to_dict(),
            "selection_pressure_history": self.selection_pressure_history(),
            "population_trend": self.population_trend(),
            "stagnation_score": self.stagnation_score(),
            "convergence_velocity": self.convergence_velocity(),
            "last_report": dict(self._last_report),
        }

    def selection_statistics_report(self) -> Dict[str, Any]:
        return self.statistics_tracker.to_dict()

    def experiment_metadata(self) -> Dict[str, Any]:
        return {
            "selection_hash": self.selection_hash(),
            "signature": self.selection_signature(),
            "configuration": self.config.to_dict(),
            "schema": self.schema.to_dict() if hasattr(self.schema, "to_dict") else None,
            "statistics": self.statistics_tracker.to_dict(),
        }

    def experiment_signature(self) -> str:
        return _stable_hash(self.experiment_metadata())

    def export_configuration(self) -> Dict[str, Any]:
        return {
            "selection_config": self.config.to_dict(),
            "schema": self.schema.to_dict() if hasattr(self.schema, "to_dict") else None,
            "experiment_signature": self.experiment_signature(),
        }

    # ------------------------------------------------------------------
    # PUBLICATION / BENCHMARK EXPORT
    # ------------------------------------------------------------------

    def publication_metrics(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        return {
            "selection_success_rate": self.selection_success_rate(parent_population, trial_population, selected_population),
            "population_survival_rate": self.population_survival_rate(parent_population, selected_population),
            "population_turnover": self.population_turnover(parent_population, selected_population),
            "replacement_ratio": self.replacement_ratio(parent_population, selected_population),
            "population_improvement": self.population_improvement(parent_population, selected_population),
            "population_quality_score": self.population_quality_score(selected_population),
            "selection_health_score": self.selection_health_score(parent_population, trial_population, selected_population),
            "selection_stability_score": self.selection_stability_score(parent_population, trial_population, selected_population),
            "selection_readiness_score": self.selection_readiness_score(parent_population, trial_population, selected_population),
            "selection_efficiency_score": self.selection_efficiency_score(parent_population, trial_population, selected_population),
            "selection_exploration_score": self.selection_exploration_score(parent_population, trial_population, selected_population),
            "selection_exploitation_score": self.selection_exploitation_score(parent_population, trial_population, selected_population),
            "selection_balance_score": self.selection_balance_score(parent_population, trial_population, selected_population),
        }

    def benchmark_metadata(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        return {
            "selection_pressure": self._selection_pressure(),
            "duplicate_ratio": population_duplicate_ratio(selected_population),
            "entropy": population_entropy(selected_population),
            "health": population_health(selected_population),
            "uniqueness_ratio": population_uniqueness_ratio(selected_population),
            "effective_population_size": self.effective_population_size(selected_population),
            "fitness_margin": self.fitness_margin(None, None),
            "selection_hash": self.selection_hash(),
        }

    def ablation_metadata(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        return {
            "greedy": self.greedy_selection(_population_individuals(parent_population), _population_individuals(trial_population), maximize=self.config.maximize),
            "elitist_statistics": elite_statistics(selected_population, elite_rate=self.config.elitism_rate, maximize=self.config.maximize),
            "diversity_preservation_score": diversity_preservation_score(selected_population),
            "selection_difficulty_score": self.selection_difficulty_score(parent_population, trial_population, selected_population),
            "pressure_history": self.selection_pressure_history(),
        }

    def paper_metrics(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        return {
            "elite_preservation_score": elite_preservation_score(parent_population, selected_population, elite_rate=self.config.elitism_rate, maximize=self.config.maximize),
            "selection_information_gain": self.selection_information_gain(parent_population, trial_population, selected_population),
            "selection_exploration_score": self.selection_exploration_score(parent_population, trial_population, selected_population),
            "selection_exploitation_score": self.selection_exploitation_score(parent_population, trial_population, selected_population),
            "selection_balance_score": self.selection_balance_score(parent_population, trial_population, selected_population),
            "population_quality_score": self.population_quality_score(selected_population),
            "population_duplicate_ratio": population_duplicate_ratio(selected_population),
            "population_entropy": population_entropy(selected_population),
            "population_health": population_health(selected_population),
        }

    def publication_export(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        return {
            "publication_metrics": self.publication_metrics(parent_population, trial_population, selected_population),
            "benchmark_metadata": self.benchmark_metadata(parent_population, trial_population, selected_population),
            "ablation_metadata": self.ablation_metadata(parent_population, trial_population, selected_population),
            "paper_metrics": self.paper_metrics(parent_population, trial_population, selected_population),
            "selection_signature": self.selection_signature(),
        }

    def reproducibility_bundle(self, parent_population: Any, trial_population: Any, selected_population: Any) -> Dict[str, Any]:
        return {
            "selection_hash": self.selection_hash(),
            "experiment_signature": self.experiment_signature(),
            "configuration": self.export_configuration(),
            "statistics": self.selection_statistics_report(),
            "publication_export": self.publication_export(parent_population, trial_population, selected_population),
            "diagnostics": self.diagnostics(),
        }

    def selection_report(self, parent_population: Any, trial_population: Any, selected_population: Any, *, strategy: Optional[str] = None) -> Dict[str, Any]:
        parent_analytics = self._population_analytics(parent_population)
        selected_analytics = self._population_analytics(selected_population)
        diversity_gain = selection_diversity_gain(parent_population, selected_population)
        entropy_change = population_entropy_change(parent_population, selected_population)
        health_change = population_health_change(parent_population, selected_population)
        quality_score = self.population_quality_score(selected_population)
        report = {
            "strategy": strategy,
            "selection_pressure": self._selection_pressure(),
            "accepted_trials": self.statistics_tracker.accepted_trials,
            "rejected_trials": self.statistics_tracker.rejected_trials,
            "population_quality": quality_score,
            "diversity_gain": diversity_gain,
            "entropy_change": entropy_change,
            "health_change": health_change,
            "selection_hash": self.selection_hash(),
            "parent_analytics": parent_analytics,
            "selected_analytics": selected_analytics,
            "selection_success_rate": self.selection_success_rate(parent_population, trial_population, selected_population),
            "population_survival_rate": self.population_survival_rate(parent_population, selected_population),
            "population_turnover": self.population_turnover(parent_population, selected_population),
            "population_improvement": self.population_improvement(parent_population, selected_population),
            "selection_health_score": self.selection_health_score(parent_population, trial_population, selected_population),
            "selection_stability_score": self.selection_stability_score(parent_population, trial_population, selected_population),
            "selection_readiness_score": self.selection_readiness_score(parent_population, trial_population, selected_population),
            "selection_efficiency_score": self.selection_efficiency_score(parent_population, trial_population, selected_population),
            "selection_difficulty_score": self.selection_difficulty_score(parent_population, trial_population, selected_population),
            "selection_search_burden": self.selection_search_burden(parent_population, trial_population, selected_population),
            "effective_population_size": self.effective_population_size(selected_population),
            "selection_information_gain": self.selection_information_gain(parent_population, trial_population, selected_population),
            "selection_exploration_score": self.selection_exploration_score(parent_population, trial_population, selected_population),
            "selection_exploitation_score": self.selection_exploitation_score(parent_population, trial_population, selected_population),
            "selection_balance_score": self.selection_balance_score(parent_population, trial_population, selected_population),
            "selection_learning_score": self.selection_learning_score(parent_population, trial_population, selected_population),
        }
        return report

    def selection_statistics_report(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        return self.statistics_tracker.to_dict()


# =============================================================================
# MODULE-LEVEL PUBLIC EXPORTS
# =============================================================================


__all__ = [
    "SelectionConfig",
    "SelectionStatistics",
    "SelectionAnalytics",
    "SelectionStrategyType",
    "SelectionEngine",
    "compare_individuals",
    "compare_fitness",
    "better_individual",
    "equal_fitness",
    "fitness_margin",
    "selection_margin",
    "greedy_selection",
    "elitist_selection",
    "diversity_preserving_selection",
    "adaptive_selection",
    "elite_individuals",
    "elite_population",
    "elite_statistics",
    "elite_preservation_score",
    "population_duplicate_ratio",
    "selection_diversity_gain",
    "population_entropy_change",
    "population_health_change",
    "diversity_preservation_score",
    "population_entropy",
    "population_health",
    "population_uniqueness_ratio",
    "effective_population_size",
]
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import hashlib
import json
import math
import random
import statistics

if TYPE_CHECKING:
    from .population import (
        DEIndividual,
        Population,
        PopulationSchema,
    )

try:
    from .rank_config import RankConfig, RankSearchSpace
except Exception:  # pragma: no cover
    RankConfig = Any  # type: ignore
    RankSearchSpace = Any  # type: ignore

try:
    from .scaling import ScalingConfig, ScalingSearchSpace
except Exception:  # pragma: no cover
    ScalingConfig = Any  # type: ignore
    ScalingSearchSpace = Any  # type: ignore

try:
    from .placement import PlacementConfig, PlacementSearchSpace
except Exception:  # pragma: no cover
    PlacementConfig = Any  # type: ignore
    PlacementSearchSpace = Any  # type: ignore


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


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _coerce_bool(value: Any, *, name: str = "assignment") -> bool:
    b = _as_bool(value)
    if b is None:
        raise TypeError(f"{name} must be boolean-like.")
    return bool(b)


def _canonical_int_tuple(values: Iterable[Any]) -> Tuple[int, ...]:
    out: List[int] = []
    for value in values:
        _validate_integer("rank_encoding", value)
        out.append(int(value))
    return tuple(out)


def _canonical_float_tuple(values: Iterable[Any]) -> Tuple[float, ...]:
    out: List[float] = []
    for value in values:
        _validate_finite_numeric("scaling_encoding", value)
        out.append(float(value))
    return tuple(out)


def _canonical_mask_tuple(values: Iterable[Any]) -> Tuple[int, ...]:
    out: List[int] = []
    for value in values:
        bit = _as_bool(value)
        if bit is None:
            raise TypeError("placement_encoding must be boolean-like.")
        out.append(1 if bit else 0)
    return tuple(out)


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


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _module_name_fallback(module: Any) -> str:
    return module.__class__.__name__

def _is_module_like(obj: Any) -> bool:
    return (
        hasattr(obj, "named_modules")
        or hasattr(obj, "base_layer")
        or hasattr(obj, "forward")
    )

# def _is_module_like(obj: Any) -> bool:
#     return isinstance(obj, Any) or hasattr(obj, "named_modules") or hasattr(obj, "base_layer")


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hamming_distance(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    if len(a) == 0:
        return 0.0
    diff = 0
    for x, y in zip(a, b):
        diff += 0 if x == y else 1
    return float(diff / len(a))


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have equal length.")
    if len(a) == 0:
        return 0.0
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return [idx for idx, _ in indexed[:k]]


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    denom = mx - mn
    if abs(denom) < 1e-12:
        return [0.5 for _ in scores]
    return [float((s - mn) / denom) for s in scores]


# ============================================================================
# SCHEMA INFERENCE
# ============================================================================

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

    for attr in ("get_lora_layers",):
        if hasattr(source, attr):
            try:
                layers = getattr(source, attr)()
                if isinstance(layers, Mapping):
                    names = [str(name) for name in layers.keys() if str(name).strip()]
                    if names:
                        return tuple(names)
            except Exception:
                pass

    if hasattr(source, "named_modules"):
        try:
            discovered = []
            for name, _module in source.named_modules():
                if name.strip():
                    discovered.append(str(name))
            if discovered:
                return tuple(discovered)
        except Exception:
            pass

    raise ValueError("Unable to infer layer names from the provided source.")


def _rank_candidates_from_config(rank_config: Any) -> Tuple[int, ...]:
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
    if hasattr(placement_config, "candidate_masks"):
        try:
            masks = list(placement_config.candidate_masks(layer_names))
            normalized = []
            for mask in masks:
                normalized.append(tuple(1 if int(v) else 0 for v in mask))
            if normalized:
                return tuple(dict.fromkeys(normalized))
        except Exception:
            pass

    if hasattr(placement_config, "candidate_placements"):
        try:
            placements = list(placement_config.candidate_placements(layer_names))
            normalized = []
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
        idxs = list(range(k))
        mask = [0] * n
        for idx in idxs:
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
    rank_config: Any,
    scaling_config: Any,
    placement_config: Any,
    source: Any = None,
    layer_names: Optional[Sequence[str]] = None,
) -> "PopulationSchema":
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


# ============================================================================
# CONFIGURATION
# ============================================================================

class MutationStrategyType(str, Enum):
    RAND_1 = "de/rand/1"
    BEST_1 = "de/best/1"
    CURRENT_TO_BEST_1 = "de/current-to-best/1"
    RAND_2 = "de/rand/2"
    BEST_2 = "de/best/2"


@dataclass
class MutationConfig:
    mutation_factor: float = 0.5
    minimum_mutation_factor: float = 0.1
    maximum_mutation_factor: float = 1.0

    enable_adaptive_mutation: bool = True

    rank_mutation_probability: float = 1.0
    scaling_mutation_probability: float = 1.0
    placement_mutation_probability: float = 1.0

    enable_diversity_guided_mutation: bool = True
    enable_constraint_preservation: bool = True

    enable_statistics_tracking: bool = True
    enable_metadata_tracking: bool = True

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_finite_numeric("mutation_factor", self.mutation_factor)
        _validate_finite_numeric("minimum_mutation_factor", self.minimum_mutation_factor)
        _validate_finite_numeric("maximum_mutation_factor", self.maximum_mutation_factor)
        _validate_finite_numeric("rank_mutation_probability", self.rank_mutation_probability)
        _validate_finite_numeric("scaling_mutation_probability", self.scaling_mutation_probability)
        _validate_finite_numeric("placement_mutation_probability", self.placement_mutation_probability)

        if self.seed is not None:
            _validate_integer("seed", self.seed)

        _validate_bool("enable_adaptive_mutation", self.enable_adaptive_mutation)
        _validate_bool("enable_diversity_guided_mutation", self.enable_diversity_guided_mutation)
        _validate_bool("enable_constraint_preservation", self.enable_constraint_preservation)
        _validate_bool("enable_statistics_tracking", self.enable_statistics_tracking)
        _validate_bool("enable_metadata_tracking", self.enable_metadata_tracking)

        if self.minimum_mutation_factor <= 0:
            raise ValueError("minimum_mutation_factor must be positive.")
        if self.maximum_mutation_factor <= 0:
            raise ValueError("maximum_mutation_factor must be positive.")
        if self.minimum_mutation_factor > self.maximum_mutation_factor:
            raise ValueError("minimum_mutation_factor must be <= maximum_mutation_factor.")
        if self.mutation_factor < self.minimum_mutation_factor or self.mutation_factor > self.maximum_mutation_factor:
            raise ValueError("mutation_factor must lie within [minimum_mutation_factor, maximum_mutation_factor].")

        for name, value in (
            ("rank_mutation_probability", self.rank_mutation_probability),
            ("scaling_mutation_probability", self.scaling_mutation_probability),
            ("placement_mutation_probability", self.placement_mutation_probability),
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")

        if (
            self.rank_mutation_probability == 0.0
            and self.scaling_mutation_probability == 0.0
            and self.placement_mutation_probability == 0.0
        ):
            raise ValueError("At least one mutation probability must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MutationStatistics:
    mutation_calls: int = 0
    mutants_generated: int = 0
    rank_mutations: int = 0
    scaling_mutations: int = 0
    placement_mutations: int = 0
    rejected_mutants: int = 0
    constraint_repairs: int = 0
    adaptive_mutations: int = 0
    diversity_guided_mutations: int = 0
    metadata_exports: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MUTATION ENGINE
# ============================================================================

class MutationEngine:
    """
    Differential Evolution mutation engine for mixed discrete chromosomes.

    Chromosome layout:
        [rank segment] + [scaling segment] + [placement segment]
    """

    def __init__(
        self,
        config: Optional[MutationConfig] = None,
        *,
        schema: Optional[PopulationSchema] = None,
        population: Optional[Population] = None,
        rank_config: Any = None,
        scaling_config: Any = None,
        placement_config: Any = None,
        source: Any = None,
        layer_names: Optional[Sequence[str]] = None,
    ) -> None:
        if config is None:
            config = MutationConfig()
        if not isinstance(config, MutationConfig):
            raise TypeError("config must be MutationConfig.")

        self.config = config
        self.statistics_tracker = MutationStatistics()
        self._rng = random.Random(config.seed)

        if population is not None:
            if not isinstance(population, Population):
                raise TypeError("population must be Population.")
            self.schema = population.schema
        elif schema is not None:
            if not isinstance(schema, PopulationSchema):
                raise TypeError("schema must be PopulationSchema.")
            self.schema = schema
        elif rank_config is not None and scaling_config is not None and placement_config is not None:
            self.schema = _infer_schema(
                rank_config=rank_config,
                scaling_config=scaling_config,
                placement_config=placement_config,
                source=source,
                layer_names=layer_names,
            )
        else:
            raise ValueError(
                "MutationEngine requires either population, schema, or rank/scaling/placement configs."
            )

        self._rank_candidates = tuple(self.schema.rank_candidates)
        self._scaling_candidates = tuple(self.schema.scaling_candidates)
        self._placement_candidates = tuple(self.schema.placement_candidates)
        self._layer_count = int(self.schema.layer_count)

    # ------------------------------------------------------------------
    # INTERNAL UTILITIES
    # ------------------------------------------------------------------

    @property
    def chromosome_length(self) -> int:
        return int(self.schema.chromosome_length)

    def _split_chromosome(self, chromosome: Sequence[Union[int, float]]) -> Tuple[List[int], List[float], List[int]]:
        if not isinstance(chromosome, Sequence):
            raise TypeError("chromosome must be a sequence.")
        expected = self.chromosome_length
        if len(chromosome) != expected:
            raise ValueError(f"chromosome length mismatch. Expected {expected}, got {len(chromosome)}.")

        n = self._layer_count
        rank_segment = _canonical_int_tuple(chromosome[:n])
        scaling_segment = _canonical_float_tuple(chromosome[n:2 * n])
        placement_segment = _canonical_mask_tuple(chromosome[2 * n:3 * n])
        return list(rank_segment), list(scaling_segment), list(placement_segment)

    def _merge_chromosome(self, rank_segment: Sequence[int], scaling_segment: Sequence[float], placement_segment: Sequence[int]) -> List[Union[int, float]]:
        if len(rank_segment) != self._layer_count or len(scaling_segment) != self._layer_count or len(placement_segment) != self._layer_count:
            raise ValueError("segment length mismatch.")
        return [*map(int, rank_segment), *map(float, scaling_segment), *map(int, placement_segment)]

    def _chromosome_from_individual(self, individual: DEIndividual) -> List[Union[int, float]]:
        if not isinstance(individual, DEIndividual):
            raise TypeError("individual must be DEIndividual.")
        return individual.to_chromosome()

    def _individual_from_chromosome(
        self,
        chromosome: Sequence[Union[int, float]],
        *,
        generation: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        fitness: Optional[float] = None,
        identifier: Optional[str] = None,
    ) -> DEIndividual:
        rank_segment, scaling_segment, placement_segment = self._split_chromosome(chromosome)
        return DEIndividual(
            rank_encoding=tuple(rank_segment),
            scaling_encoding=tuple(scaling_segment),
            placement_encoding=tuple(placement_segment),
            fitness=fitness,
            metadata=dict(metadata or {}),
            generation=int(generation),
            identifier=identifier,
        )

    def _population_individuals(self, population: Population) -> List[DEIndividual]:
        if not isinstance(population, Population):
            raise TypeError("population must be Population.")
        return list(population.individuals)

    def _best_individual(self, population: Population) -> DEIndividual:
        individuals = self._population_individuals(population)
        if not individuals:
            raise ValueError("population has no individuals.")

        def key(ind: DEIndividual) -> Tuple[float, str]:
            fitness = float(ind.fitness) if ind.fitness is not None and math.isfinite(float(ind.fitness)) else float("-inf")
            return (fitness, ind.signature())

        return max(individuals, key=key)

    def _sorted_fitness_individuals(self, population: Population) -> List[DEIndividual]:
        individuals = self._population_individuals(population)

        def key(ind: DEIndividual) -> Tuple[float, str]:
            fitness = float(ind.fitness) if ind.fitness is not None and math.isfinite(float(ind.fitness)) else float("-inf")
            return (fitness, ind.signature())

        return sorted(individuals, key=key, reverse=True)

    def _target_index(self, population: Population, target: DEIndividual) -> Optional[int]:
        for idx, ind in enumerate(self._population_individuals(population)):
            if ind.signature() == target.signature():
                return idx
        return None

    def _select_indices(self, population: Population, exclude: Sequence[int], count: int) -> List[int]:
        individuals = self._population_individuals(population)
        n = len(individuals)
        candidates = [i for i in range(n) if i not in set(exclude)]
        if len(candidates) < count:
            raise RuntimeError(
                f"Not enough distinct donors for mutation. Required {count}, available {len(candidates)}."
            )
        return self._rng.sample(candidates, count)

    def _select_donors(
        self,
        population: Population,
        target: Optional[DEIndividual],
        count: int,
    ) -> List[DEIndividual]:
        individuals = self._population_individuals(population)
        if not individuals:
            raise ValueError("population has no individuals.")

        exclude: List[int] = []
        if target is not None:
            target_idx = self._target_index(population, target)
            if target_idx is not None:
                exclude.append(target_idx)

        indices = self._select_indices(population, exclude=exclude, count=count)
        return [individuals[i] for i in indices]

    def _select_donors_diversity_guided(
        self,
        population: Population,
        target: Optional[DEIndividual],
        count: int,
    ) -> List[DEIndividual]:
        individuals = self._population_individuals(population)
        if not individuals:
            raise ValueError("population has no individuals.")

        target_vec = None
        if target is not None:
            target_vec = [float(v) for v in target.to_chromosome()]

        scored: List[Tuple[int, float]] = []
        for idx, ind in enumerate(individuals):
            if target is not None and ind.signature() == target.signature():
                continue
            vec = [float(v) for v in ind.to_chromosome()]
            if target_vec is None:
                score = 0.0
            else:
                score = _euclidean_distance(vec, target_vec)
            if ind.fitness is not None and math.isfinite(float(ind.fitness)):
                score += 0.01 * float(ind.fitness)
            scored.append((idx, score))

        if len(scored) < count:
            raise RuntimeError(
                f"Not enough distinct donors for diversity-guided mutation. Required {count}, available {len(scored)}."
            )

        scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
        top_pool = scored[: max(count, len(scored) // 2 or count)]
        chosen = []
        pool = top_pool if len(top_pool) >= count else scored
        if len(pool) < count:
            raise RuntimeError("Unable to select diversity-guided donors.")
        used = set()
        while len(chosen) < count:
            idx, _score = self._rng.choice(pool)
            if idx in used:
                continue
            used.add(idx)
            chosen.append(individuals[idx])
        return chosen

    def _mutation_factor(self, population: Optional[Population] = None, target: Optional[DEIndividual] = None) -> float:
        factor = float(self.config.mutation_factor)
        if not self.config.enable_adaptive_mutation or population is None:
            return self._clip_mutation_factor(factor)

        self.statistics_tracker.adaptive_mutations += 1

        diversity = 0.5
        health = 0.5
        difficulty = 1.0
        uniqueness = 0.5
        convergence_risk = 0.5

        if hasattr(population, "population_diversity"):
            try:
                report = population.population_diversity()
                if isinstance(report, Mapping):
                    diversity = float(report.get("uniqueness_ratio", report.get("diversity_health_score", diversity)))
                    uniqueness = float(report.get("uniqueness_ratio", uniqueness))
            except Exception:
                pass

        if hasattr(population, "population_health_score"):
            try:
                health = float(population.population_health_score())
            except Exception:
                pass

        if hasattr(population, "optimization_difficulty_score"):
            try:
                difficulty = float(population.optimization_difficulty_score())
            except Exception:
                pass

        if hasattr(population, "premature_convergence_risk"):
            try:
                convergence_risk = float(population.premature_convergence_risk())
            except Exception:
                pass

        diversity = max(0.0, min(1.0, float(diversity)))
        health = max(0.0, min(1.0, float(health)))
        uniqueness = max(0.0, min(1.0, float(uniqueness)))
        convergence_risk = max(0.0, min(1.0, float(convergence_risk)))

        exploration_pressure = (1.0 - diversity) * 0.35 + convergence_risk * 0.35 + (1.0 - uniqueness) * 0.15
        stability_pressure = health * 0.20
        difficulty_pressure = 0.0
        if math.isfinite(difficulty):
            difficulty_pressure = min(1.0, max(0.0, math.log10(max(1.0, difficulty + 1.0)) / 3.0))

        if target is not None and getattr(target, "fitness", None) is None:
            exploration_pressure += 0.05

        factor = factor * (1.0 + exploration_pressure + difficulty_pressure - stability_pressure)
        if self.config.enable_diversity_guided_mutation:
            factor *= 1.05 + 0.10 * (1.0 - diversity)

        return self._clip_mutation_factor(factor)

    def _clip_mutation_factor(self, value: float) -> float:
        return float(
            max(
                self.config.minimum_mutation_factor,
                min(self.config.maximum_mutation_factor, float(value)),
            )
        )

    def _segment_probability_mask(self) -> Tuple[bool, bool, bool]:
        return (
            self.config.rank_mutation_probability > 0.0,
            self.config.scaling_mutation_probability > 0.0,
            self.config.placement_mutation_probability > 0.0,
        )

    def _choose_segment_activation(self, probability: float) -> bool:
        probability = max(0.0, min(1.0, float(probability)))
        return self._rng.random() < probability

    def _strategy_name(self, strategy: Union[str, MutationStrategyType]) -> str:
        if isinstance(strategy, MutationStrategyType):
            return strategy.value
        return str(strategy).strip().lower()

    def _record_mutation_statistics(self, rank_changed: bool, scaling_changed: bool, placement_changed: bool, repaired: bool = False) -> None:
        if rank_changed:
            self.statistics_tracker.rank_mutations += 1
        if scaling_changed:
            self.statistics_tracker.scaling_mutations += 1
        if placement_changed:
            self.statistics_tracker.placement_mutations += 1
        if repaired:
            self.statistics_tracker.constraint_repairs += 1

    # ------------------------------------------------------------------
    # MUTATION STRATEGIES
    # ------------------------------------------------------------------

    def _mutate_numeric_segment_rand_1(
        self,
        target_segment: Sequence[Union[int, float]],
        donor_a: Sequence[Union[int, float]],
        donor_b: Sequence[Union[int, float]],
        donor_c: Sequence[Union[int, float]],
        factor: float,
    ) -> List[float]:
        if len(target_segment) != len(donor_a) or len(donor_a) != len(donor_b) or len(donor_b) != len(donor_c):
            raise ValueError("segment length mismatch.")
        return [
            float(a) + factor * (float(b) - float(c))
            for a, b, c in zip(donor_a, donor_b, donor_c)
        ]

    def _mutate_numeric_segment_best_1(
        self,
        best_segment: Sequence[Union[int, float]],
        donor_a: Sequence[Union[int, float]],
        donor_b: Sequence[Union[int, float]],
        factor: float,
    ) -> List[float]:
        if len(best_segment) != len(donor_a) or len(donor_a) != len(donor_b):
            raise ValueError("segment length mismatch.")
        return [
            float(best) + factor * (float(a) - float(b))
            for best, a, b in zip(best_segment, donor_a, donor_b)
        ]

    def _mutate_numeric_segment_current_to_best_1(
        self,
        target_segment: Sequence[Union[int, float]],
        best_segment: Sequence[Union[int, float]],
        donor_a: Sequence[Union[int, float]],
        donor_b: Sequence[Union[int, float]],
        factor: float,
    ) -> List[float]:
        if len(target_segment) != len(best_segment) or len(best_segment) != len(donor_a) or len(donor_a) != len(donor_b):
            raise ValueError("segment length mismatch.")
        return [
            float(target) + factor * (float(best) - float(target)) + factor * (float(a) - float(b))
            for target, best, a, b in zip(target_segment, best_segment, donor_a, donor_b)
        ]

    def _mutate_numeric_segment_rand_2(
        self,
        donor_a: Sequence[Union[int, float]],
        donor_b: Sequence[Union[int, float]],
        donor_c: Sequence[Union[int, float]],
        donor_d: Sequence[Union[int, float]],
        donor_e: Sequence[Union[int, float]],
        factor: float,
    ) -> List[float]:
        if len(donor_a) != len(donor_b) or len(donor_b) != len(donor_c) or len(donor_c) != len(donor_d) or len(donor_d) != len(donor_e):
            raise ValueError("segment length mismatch.")
        return [
            float(a) + factor * (float(b) - float(c)) + factor * (float(d) - float(e))
            for a, b, c, d, e in zip(donor_a, donor_b, donor_c, donor_d, donor_e)
        ]

    def _mutate_numeric_segment_best_2(
        self,
        best_segment: Sequence[Union[int, float]],
        donor_a: Sequence[Union[int, float]],
        donor_b: Sequence[Union[int, float]],
        donor_c: Sequence[Union[int, float]],
        donor_d: Sequence[Union[int, float]],
        factor: float,
    ) -> List[float]:
        if len(best_segment) != len(donor_a) or len(donor_a) != len(donor_b) or len(donor_b) != len(donor_c) or len(donor_c) != len(donor_d):
            raise ValueError("segment length mismatch.")
        return [
            float(best) + factor * (float(a) - float(b)) + factor * (float(c) - float(d))
            for best, a, b, c, d in zip(best_segment, donor_a, donor_b, donor_c, donor_d)
        ]

    def mutate_rank_segment(
        self,
        target_segment: Sequence[int],
        candidate_segment: Sequence[float],
    ) -> List[int]:
        if len(target_segment) != self._layer_count or len(candidate_segment) != self._layer_count:
            raise ValueError("rank segment length mismatch.")
        if not self._rank_candidates:
            raise ValueError("rank candidate space is empty.")
        repaired: List[int] = []
        for value in candidate_segment:
            value_f = float(value)
            repaired.append(min(self._rank_candidates, key=lambda x: (abs(float(x) - value_f), x)))
        return repaired

    def mutate_scaling_segment(
        self,
        target_segment: Sequence[float],
        candidate_segment: Sequence[float],
    ) -> List[float]:
        if len(target_segment) != self._layer_count or len(candidate_segment) != self._layer_count:
            raise ValueError("scaling segment length mismatch.")
        if not self._scaling_candidates:
            raise ValueError("scaling candidate space is empty.")
        repaired: List[float] = []
        for value in candidate_segment:
            value_f = float(value)
            repaired.append(min(self._scaling_candidates, key=lambda x: (abs(float(x) - value_f), float(x))))
        return repaired

    def mutate_placement_segment(
        self,
        target_segment: Sequence[int],
        candidate_segment: Sequence[float],
    ) -> List[int]:
        if len(target_segment) != self._layer_count or len(candidate_segment) != self._layer_count:
            raise ValueError("placement segment length mismatch.")
        if not self._placement_candidates:
            raise ValueError("placement candidate space is empty.")

        raw_mask = [1 if float(v) >= 0.5 else 0 for v in candidate_segment]
        best_mask = None
        best_score = None
        for mask in self._placement_candidates:
            hamming = _hamming_distance(mask, raw_mask)
            density_diff = abs(sum(mask) - sum(raw_mask)) / max(1, self._layer_count)
            score = (hamming, density_diff, tuple(mask))
            if best_score is None or score < best_score:
                best_score = score
                best_mask = list(mask)
        if best_mask is None:
            raise RuntimeError("Unable to repair placement segment.")
        return best_mask

    def repair_rank_segment(self, segment: Sequence[Union[int, float]]) -> List[int]:
        if len(segment) != self._layer_count:
            raise ValueError("rank segment length mismatch.")
        repaired = [min(self._rank_candidates, key=lambda x: (abs(float(x) - float(v)), x)) for v in segment]
        return [int(v) for v in repaired]

    def repair_scaling_segment(self, segment: Sequence[Union[int, float]]) -> List[float]:
        if len(segment) != self._layer_count:
            raise ValueError("scaling segment length mismatch.")
        repaired = [min(self._scaling_candidates, key=lambda x: (abs(float(x) - float(v)), float(x))) for v in segment]
        return [float(v) for v in repaired]

    def repair_placement_segment(self, segment: Sequence[Union[int, float]]) -> List[int]:
        if len(segment) != self._layer_count:
            raise ValueError("placement segment length mismatch.")
        candidate = self.mutate_placement_segment([1 if _as_bool(v) else 0 for v in segment], [float(v) for v in segment])
        return [int(v) for v in candidate]

    def repair_mutant(self, chromosome: Sequence[Union[int, float]]) -> List[Union[int, float]]:
        rank_segment, scaling_segment, placement_segment = self._split_chromosome(chromosome)
        repaired_rank = self.repair_rank_segment(rank_segment)
        repaired_scaling = self.repair_scaling_segment(scaling_segment)
        repaired_placement = self.repair_placement_segment(placement_segment)
        return self._merge_chromosome(repaired_rank, repaired_scaling, repaired_placement)

    def constraint_preserving_mutation(self, chromosome: Sequence[Union[int, float]]) -> List[Union[int, float]]:
     
        repaired = self.repair_mutant(chromosome)
       
        self.validate_mutant(repaired)
        
        return repaired

    def _apply_segment_strategy(
        self,
        strategy: str,
        target: DEIndividual,
        population: Population,
        factor: float,
        best: Optional[DEIndividual] = None,
        diversity_guided: bool = False,
    ) -> List[Union[int, float]]:
        target_vec = self._chromosome_from_individual(target)
        target_rank, target_scaling, target_placement = self._split_chromosome(target_vec)

        if diversity_guided:
            if best is None:
                best = self._best_individual(population)
            donors_needed = {
                MutationStrategyType.RAND_1.value: 3,
                MutationStrategyType.BEST_1.value: 2,
                MutationStrategyType.CURRENT_TO_BEST_1.value: 2,
                MutationStrategyType.RAND_2.value: 5,
                MutationStrategyType.BEST_2.value: 4,
            }.get(strategy)
            if donors_needed is None:
                raise ValueError(f"Unsupported mutation strategy: {strategy}")
            donors = self._select_donors_diversity_guided(population, target, donors_needed)
            self.statistics_tracker.diversity_guided_mutations += 1
        else:
            donors_needed = {
                MutationStrategyType.RAND_1.value: 3,
                MutationStrategyType.BEST_1.value: 2,
                MutationStrategyType.CURRENT_TO_BEST_1.value: 2,
                MutationStrategyType.RAND_2.value: 5,
                MutationStrategyType.BEST_2.value: 4,
            }.get(strategy)
            if donors_needed is None:
                raise ValueError(f"Unsupported mutation strategy: {strategy}")
            donors = self._select_donors(population, target, donors_needed)

        donor_vectors = [self._chromosome_from_individual(ind) for ind in donors]
        donor_ranks, donor_scalings, donor_placements = [self._split_chromosome(vec) for vec in donor_vectors]

        if strategy == MutationStrategyType.RAND_1.value:
            rank_candidate = self._mutate_numeric_segment_rand_1(target_rank, donor_ranks[0], donor_ranks[1], donor_ranks[2], factor)
            scaling_candidate = self._mutate_numeric_segment_rand_1(target_scaling, donor_scalings[0], donor_scalings[1], donor_scalings[2], factor)
            placement_candidate = self._mutate_numeric_segment_rand_1(target_placement, donor_placements[0], donor_placements[1], donor_placements[2], factor)

        elif strategy == MutationStrategyType.BEST_1.value:
            if best is None:
                best = self._best_individual(population)
            best_vec = self._split_chromosome(self._chromosome_from_individual(best))
            rank_candidate = self._mutate_numeric_segment_best_1(best_vec[0], donor_ranks[0], donor_ranks[1], factor)
            scaling_candidate = self._mutate_numeric_segment_best_1(best_vec[1], donor_scalings[0], donor_scalings[1], factor)
            placement_candidate = self._mutate_numeric_segment_best_1(best_vec[2], donor_placements[0], donor_placements[1], factor)

        elif strategy == MutationStrategyType.CURRENT_TO_BEST_1.value:
            if best is None:
                best = self._best_individual(population)
            best_vec = self._split_chromosome(self._chromosome_from_individual(best))
            rank_candidate = self._mutate_numeric_segment_current_to_best_1(target_rank, best_vec[0], donor_ranks[0], donor_ranks[1], factor)
            scaling_candidate = self._mutate_numeric_segment_current_to_best_1(target_scaling, best_vec[1], donor_scalings[0], donor_scalings[1], factor)
            placement_candidate = self._mutate_numeric_segment_current_to_best_1(target_placement, best_vec[2], donor_placements[0], donor_placements[1], factor)

        elif strategy == MutationStrategyType.RAND_2.value:
            rank_candidate = self._mutate_numeric_segment_rand_2(donor_ranks[0], donor_ranks[1], donor_ranks[2], donor_ranks[3], donor_ranks[4], factor)
            scaling_candidate = self._mutate_numeric_segment_rand_2(donor_scalings[0], donor_scalings[1], donor_scalings[2], donor_scalings[3], donor_scalings[4], factor)
            placement_candidate = self._mutate_numeric_segment_rand_2(donor_placements[0], donor_placements[1], donor_placements[2], donor_placements[3], donor_placements[4], factor)

        elif strategy == MutationStrategyType.BEST_2.value:
            if best is None:
                best = self._best_individual(population)
            best_vec = self._split_chromosome(self._chromosome_from_individual(best))
            rank_candidate = self._mutate_numeric_segment_best_2(best_vec[0], donor_ranks[0], donor_ranks[1], donor_ranks[2], donor_ranks[3], factor)
            scaling_candidate = self._mutate_numeric_segment_best_2(best_vec[1], donor_scalings[0], donor_scalings[1], donor_scalings[2], donor_scalings[3], factor)
            placement_candidate = self._mutate_numeric_segment_best_2(best_vec[2], donor_placements[0], donor_placements[1], donor_placements[2], donor_placements[3], factor)

        else:
            raise ValueError(f"Unsupported mutation strategy: {strategy}")

        mutated_rank = self.mutate_rank_segment(target_rank, rank_candidate) if self._choose_segment_activation(self.config.rank_mutation_probability) else list(target_rank)
        mutated_scaling = self.mutate_scaling_segment(target_scaling, scaling_candidate) if self._choose_segment_activation(self.config.scaling_mutation_probability) else list(target_scaling)
        mutated_placement = self.mutate_placement_segment(target_placement, placement_candidate) if self._choose_segment_activation(self.config.placement_mutation_probability) else list(target_placement)

        rank_changed = mutated_rank != target_rank
        scaling_changed = mutated_scaling != target_scaling
        placement_changed = mutated_placement != target_placement
        self._record_mutation_statistics(rank_changed, scaling_changed, placement_changed)

        return self._merge_chromosome(mutated_rank, mutated_scaling, mutated_placement)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def generate_mutant(
        self,
        target: DEIndividual,
        population: Population,
        *,
        strategy: Union[str, MutationStrategyType] = MutationStrategyType.RAND_1,
        best: Optional[DEIndividual] = None,
        mutation_factor: Optional[float] = None,
        diversity_guided: Optional[bool] = None,
        generation: Optional[int] = None,
    ) -> List[Union[int, float]]:
        if not isinstance(target, DEIndividual):
            raise TypeError("target must be DEIndividual.")
        if not isinstance(population, Population):
            raise TypeError("population must be Population.")

        strategy_name = self._strategy_name(strategy)
        factor = self._mutation_factor(population=population, target=target) if mutation_factor is None else self._clip_mutation_factor(float(mutation_factor))
        if diversity_guided is None:
            diversity_guided = self.config.enable_diversity_guided_mutation

        mutant = self._apply_segment_strategy(
            strategy=strategy_name,
            target=target,
            population=population,
            factor=factor,
            best=best,
            diversity_guided=diversity_guided,
        )

        if self.config.enable_constraint_preservation:
            try:
                original_mutant = list(mutant)

                mutant = self.constraint_preserving_mutation(mutant)

                if mutant != original_mutant:
                    self.statistics_tracker.constraint_repairs += 1
            except Exception as exc:
                self.statistics_tracker.rejected_mutants += 1
                raise RuntimeError("Mutation produced an unrecoverable illegal chromosome.") from exc
        else:
            self.validate_mutant(mutant)

        return mutant

    def mutate(
        self,
        target: DEIndividual,
        population: Population,
        *,
        strategy: Union[str, MutationStrategyType] = MutationStrategyType.RAND_1,
        best: Optional[DEIndividual] = None,
        mutation_factor: Optional[float] = None,
        diversity_guided: Optional[bool] = None,
        generation: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DEIndividual:
        self.statistics_tracker.mutation_calls += 1
        strategy_name = self._strategy_name(strategy)
        factor = self._mutation_factor(population=population, target=target) if mutation_factor is None else self._clip_mutation_factor(float(mutation_factor))
        if diversity_guided is None:
            diversity_guided = self.config.enable_diversity_guided_mutation

        mutant_chromosome = self.generate_mutant(
            target,
            population,
            strategy=strategy_name,
            best=best,
            mutation_factor=factor,
            diversity_guided=diversity_guided,
            generation=generation,
        )

        target_generation = int(target.generation + 1 if generation is None else generation)
        payload_metadata = {
            "strategy": strategy_name,
            "mutation_factor": factor,
            "target_signature": target.signature(),
            "population_signature": population.population_signature() if hasattr(population, "population_signature") else None,
            "diversity_guided": bool(diversity_guided),
        }
        if metadata is not None:
            payload_metadata.update({str(k): v for k, v in dict(metadata).items()})
        if self.config.enable_metadata_tracking:
            payload_metadata["mutation_hash"] = _stable_hash({
                "target": target.signature(),
                "mutant": mutant_chromosome,
                "strategy": strategy_name,
                "factor": factor,
                "generation": target_generation,
            })

        mutant = self._individual_from_chromosome(
            mutant_chromosome,
            generation=target_generation,
            metadata=payload_metadata,
            fitness=None,
        )
        self.statistics_tracker.mutants_generated += 1
        self.validate_mutant(mutant_chromosome)
        return mutant

    def mutate_population(
        self,
        population: Population,
        *,
        strategy: Union[str, MutationStrategyType] = MutationStrategyType.RAND_1,
        mutation_factor: Optional[float] = None,
        diversity_guided: Optional[bool] = None,
    ) -> List[DEIndividual]:
        if not isinstance(population, Population):
            raise TypeError("population must be Population.")
        mutants: List[DEIndividual] = []
        best = self._best_individual(population)

        for target in population.individuals:
            mutant = self.mutate(
                target,
                population,
                strategy=strategy,
                best=best,
                mutation_factor=mutation_factor,
                diversity_guided=diversity_guided,
            )
            mutants.append(mutant)

        return mutants

    def diversity_guided_mutation(
        self,
        target: DEIndividual,
        population: Population,
        *,
        strategy: Union[str, MutationStrategyType] = MutationStrategyType.CURRENT_TO_BEST_1,
        best: Optional[DEIndividual] = None,
    ) -> DEIndividual:
        # self.statistics_tracker.diversity_guided_mutations += 1
        return self.mutate(
            target,
            population,
            strategy=strategy,
            best=best,
            diversity_guided=True,
        )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_mutant(self, mutant: Sequence[Union[int, float]] | DEIndividual) -> bool:
        if isinstance(mutant, DEIndividual):
            chromosome = mutant.to_chromosome()
        else:
            chromosome = list(mutant)

        if len(chromosome) != self.chromosome_length:
            raise ValueError(
                f"mutant chromosome length mismatch. Expected {self.chromosome_length}, got {len(chromosome)}."
            )

        rank_segment, scaling_segment, placement_segment = self._split_chromosome(chromosome)

        if len(rank_segment) != self._layer_count or len(scaling_segment) != self._layer_count or len(placement_segment) != self._layer_count:
            raise ValueError("mutant segment dimensionality mismatch.")

        rank_set = set(self._rank_candidates)
        scaling_set = set(self._scaling_candidates)
        placement_set = set(self._placement_candidates)

        if any(rank not in rank_set for rank in rank_segment):
            raise ValueError("mutant rank segment contains illegal values.")
        if any(alpha not in scaling_set for alpha in scaling_segment):
            raise ValueError("mutant scaling segment contains illegal values.")
        if any(bit not in (0, 1) for bit in placement_segment):
            raise ValueError("mutant placement segment must be binary.")
        if tuple(placement_segment) not in placement_set:
            raise ValueError("mutant placement segment is not a legal placement mask.")

        active = sum(placement_segment)
        if active < self.schema.minimum_layers:
            raise ValueError("mutant placement active layer count below minimum_layers.")
        if self.schema.maximum_layers is not None and active > self.schema.maximum_layers:
            raise ValueError("mutant placement active layer count exceeds maximum_layers.")

        # if active < self.schema.minimum_layers:
        #     raise ValueError(
        #         "mutant placement active layer count below schema minimum."
        #     )

        # if (
        #     self.schema.maximum_layers is not None
        #     and active > self.schema.maximum_layers
        # ):
        #     raise ValueError(
        #         "mutant placement active layer count exceeds schema maximum."
        #     )

        return True

    def validate_population_mutation(self, mutants: Sequence[DEIndividual]) -> bool:
        if not isinstance(mutants, Sequence):
            raise TypeError("mutants must be a sequence.")
        for mutant in mutants:
            self.validate_mutant(mutant)
        return True

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------

    def mutation_magnitude(self, target: Sequence[Union[int, float]], mutant: Sequence[Union[int, float]]) -> float:
        if len(target) != len(mutant):
            raise ValueError("target and mutant must have equal length.")
        rank_len = self._layer_count
        rank_a = [float(v) for v in target[:rank_len]]
        rank_b = [float(v) for v in mutant[:rank_len]]
        scaling_a = [float(v) for v in target[rank_len:2 * rank_len]]
        scaling_b = [float(v) for v in mutant[rank_len:2 * rank_len]]
        placement_a = [float(v) for v in target[2 * rank_len:3 * rank_len]]
        placement_b = [float(v) for v in mutant[2 * rank_len:3 * rank_len]]

        rank_norm = max(1.0, float(max(self._rank_candidates) - min(self._rank_candidates))) if self._rank_candidates else 1.0
        scaling_norm = max(1.0, float(max(self._scaling_candidates) - min(self._scaling_candidates))) if self._scaling_candidates else 1.0
        placement_norm = 1.0

        rank_delta = _safe_mean([abs(a - b) / rank_norm for a, b in zip(rank_a, rank_b)])
        scaling_delta = _safe_mean([abs(a - b) / scaling_norm for a, b in zip(scaling_a, scaling_b)])
        placement_delta = _hamming_distance(placement_a, placement_b) / placement_norm

        return float((rank_delta + scaling_delta + placement_delta) / 3.0)

    def segment_mutation_statistics(
        self,
        target: Sequence[Union[int, float]],
        mutant: Sequence[Union[int, float]],
    ) -> Dict[str, Any]:
        if len(target) != len(mutant):
            raise ValueError("target and mutant must have equal length.")
        rank_len = self._layer_count
        target_rank, target_scaling, target_placement = self._split_chromosome(target)
        mutant_rank, mutant_scaling, mutant_placement = self._split_chromosome(mutant)

        rank_changes = sum(1 for a, b in zip(target_rank, mutant_rank) if a != b)
        scaling_changes = sum(1 for a, b in zip(target_scaling, mutant_scaling) if float(a) != float(b))
        placement_changes = sum(1 for a, b in zip(target_placement, mutant_placement) if a != b)

        return {
            "rank_changes": rank_changes,
            "scaling_changes": scaling_changes,
            "placement_changes": placement_changes,
            "rank_change_ratio": rank_changes / max(1, rank_len),
            "scaling_change_ratio": scaling_changes / max(1, rank_len),
            "placement_change_ratio": placement_changes / max(1, rank_len),
            "mutation_magnitude": self.mutation_magnitude(target, mutant),
        }

    def mutation_diversity_gain(
        self,
        population: Population,
        mutants: Sequence[DEIndividual],
    ) -> float:
        if not isinstance(population, Population):
            raise TypeError("population must be Population.")
        if not isinstance(mutants, Sequence):
            raise TypeError("mutants must be a sequence.")

        original = [ind.to_chromosome() for ind in population.individuals]
        mutated = [m.to_chromosome() if isinstance(m, DEIndividual) else list(m) for m in mutants]

        if len(original) <= 1 or len(mutated) <= 1:
            return 0.0

        def avg_pairwise(vectors: Sequence[Sequence[Union[int, float]]]) -> float:
            if len(vectors) <= 1:
                return 0.0
            total = 0.0
            count = 0
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    total += _euclidean_distance([float(x) for x in vectors[i]], [float(x) for x in vectors[j]])
                    count += 1
            return total / max(1, count)

        original_dispersion = avg_pairwise(original)
        mutated_dispersion = avg_pairwise(mutated)
        if original_dispersion <= 1e-12:
            return float(mutated_dispersion)
        return float((mutated_dispersion - original_dispersion) / original_dispersion)

    def mutation_efficiency_score(self, population: Optional[Population] = None) -> float:
        generated = max(1, self.statistics_tracker.mutants_generated)
        rejected = self.statistics_tracker.rejected_mutants
        repair = self.statistics_tracker.constraint_repairs

        base = 1.0 - (rejected / max(1, generated + rejected))
        repair_bonus = min(0.25, repair / max(1, generated) * 0.1)

        if population is not None:
            diversity = 0.5
            health = 0.5
            if hasattr(population, "population_health_score"):
                try:
                    health = float(population.population_health_score())
                except Exception:
                    pass
            if hasattr(population, "population_diversity"):
                try:
                    report = population.population_diversity()
                    if isinstance(report, Mapping):
                        diversity = float(report.get("uniqueness_ratio", report.get("diversity_health_score", diversity)))
                except Exception:
                    pass
            base = base * (0.5 + 0.25 * diversity + 0.25 * health)

        return float(max(0.0, min(1.0, base + repair_bonus)))

    def exploration_score(self, population: Optional[Population] = None) -> float:
        if population is None:
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        0.5 + 0.5 * self.diversity_preservation_score(),
                    ),
                )
            )
        diversity = 0.5
        if hasattr(population, "population_diversity"):
            try:
                report = population.population_diversity()
                if isinstance(report, Mapping):
                    diversity = float(report.get("uniqueness_ratio", report.get("diversity_health_score", diversity)))
            except Exception:
                pass
        return float(max(0.0, min(1.0, 0.5 + 0.5 * diversity)))

    def exploitation_score(self, population: Optional[Population] = None) -> float:
        return float(1.0 - self.exploration_score(population=population))

    def mutation_health_score(self, population: Optional[Population] = None) -> float:
        efficiency = self.mutation_efficiency_score(population=population)
        stability = self.mutation_stability_score(population=population)
        readiness = self.mutation_readiness_score(population=population)
        return float(max(0.0, min(1.0, 0.4 * efficiency + 0.3 * stability + 0.3 * readiness)))

    def mutation_stability_score(self, population: Optional[Population] = None) -> float:
        if population is None:
            return float(max(0.0, min(1.0, 1.0 - self.adaptive_mutation_score())))
        health = 0.5
        if hasattr(population, "population_health_score"):
            try:
                health = float(population.population_health_score())
            except Exception:
                pass
        return float(max(0.0, min(1.0, health)))

    def mutation_readiness_score(self, population: Optional[Population] = None) -> float:
        difficulty = self.mutation_difficulty_score(population=population)
        health = self.mutation_stability_score(population=population)
        return float(max(0.0, min(1.0, 0.5 * health + 0.5 * (1.0 - min(1.0, difficulty / 10.0)))))

    def diversity_preservation_score(self, population: Optional[Population] = None) -> float:
        if population is None:
            return 0.5
        diversity = 0.5
        risk = 0.5
        if hasattr(population, "population_diversity"):
            try:
                report = population.population_diversity()
                if isinstance(report, Mapping):
                    diversity = float(report.get("uniqueness_ratio", report.get("diversity_health_score", diversity)))
            except Exception:
                pass
        if hasattr(population, "premature_convergence_risk"):
            try:
                risk = float(population.premature_convergence_risk())
            except Exception:
                pass
        return float(max(0.0, min(1.0, 0.5 * diversity + 0.5 * (1.0 - risk))))

    def adaptive_mutation_score(self, population: Optional[Population] = None) -> float:
        if population is None:
            return 0.5 if self.config.enable_adaptive_mutation else 0.0
        diversity = self.diversity_preservation_score(population=population)
        difficulty = self.mutation_difficulty_score(population=population)
        health = self.mutation_stability_score(population=population)
        score = 0.4 * (1.0 - diversity) + 0.3 * min(1.0, difficulty / 10.0) + 0.3 * (1.0 - health)
        return float(max(0.0, min(1.0, score)))

    def mutation_difficulty_score(self, population: Optional[Population] = None) -> float:
        if population is not None:
            if hasattr(population, "optimization_difficulty_score"):
                try:
                    return float(population.optimization_difficulty_score())
                except Exception:
                    pass
            if hasattr(population, "search_space_complexity"):
                try:
                    return float(population.search_space_complexity())
                except Exception:
                    pass
        return float(self.mutation_search_burden())

    def effective_mutation_dimensions(self) -> int:
        dims = 0
        if self.config.rank_mutation_probability > 0.0:
            dims += self._layer_count
        if self.config.scaling_mutation_probability > 0.0:
            dims += self._layer_count
        if self.config.placement_mutation_probability > 0.0:
            dims += self._layer_count
        return int(dims)

    def mutation_search_burden(self) -> float:
        rank_count = max(1, len(self._rank_candidates))
        scaling_count = max(1, len(self._scaling_candidates))
        placement_count = max(1, len(self._placement_candidates))
        layer_count = max(1, self._layer_count)
        cardinality = float(layer_count * math.log2(rank_count) + layer_count * math.log2(scaling_count) + math.log2(placement_count))
        return float(cardinality * max(1, self.effective_mutation_dimensions()))

    # ------------------------------------------------------------------
    # REPRODUCIBILITY
    # ------------------------------------------------------------------

    def mutation_hash(self) -> str:
        payload = {
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "statistics": self.statistics_tracker.to_dict(),
        }
        return _stable_hash(payload)

    def mutation_signature(self) -> str:
        return f"{self.chromosome_length}:{self.mutation_hash()}"

    def mutation_fingerprint(self) -> Dict[str, Any]:
        return {
            "mutation_hash": self.mutation_hash(),
            "mutation_signature": self.mutation_signature(),
            "chromosome_length": self.chromosome_length,
            "effective_mutation_dimensions": self.effective_mutation_dimensions(),
        }

    # ------------------------------------------------------------------
    # METADATA / DIAGNOSTICS
    # ------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        return {
            "module": "MutationEngine",
            "configuration": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "strategy_support": [s.value for s in MutationStrategyType],
            "mutation_hash": self.mutation_hash(),
            "mutation_signature": self.mutation_signature(),
            "mutation_factor": self.config.mutation_factor,
            "min_mutation_factor": self.config.minimum_mutation_factor,
            "max_mutation_factor": self.config.maximum_mutation_factor,
            "mutants_generated": self.statistics_tracker.mutants_generated,
            "repair_count": self.statistics_tracker.constraint_repairs,
            "diversity_gain": None,
            "mutation_efficiency": self.mutation_efficiency_score(),
            "statistics": self.statistics().to_dict() if self.config.enable_statistics_tracking else self.statistics_tracker.to_dict(),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "strategy": [s.value for s in MutationStrategyType],
            "mutation_factor": self.config.mutation_factor,
            "mutants_generated": self.statistics_tracker.mutants_generated,
            "repair_count": self.statistics_tracker.constraint_repairs,
            "diversity_gain": None,
            "mutation_efficiency": self.mutation_efficiency_score(),
            "mutation_hash": self.mutation_hash(),
            "mutation_health_score": self.mutation_health_score(),
            "mutation_stability_score": self.mutation_stability_score(),
            "mutation_readiness_score": self.mutation_readiness_score(),
            "diversity_preservation_score": self.diversity_preservation_score(),
            "adaptive_mutation_score": self.adaptive_mutation_score(),
            "mutation_difficulty_score": self.mutation_difficulty_score(),
            "effective_mutation_dimensions": self.effective_mutation_dimensions(),
            "mutation_search_burden": self.mutation_search_burden(),
        }

    def mutation_report(self, population: Optional[Population] = None, mutants: Optional[Sequence[DEIndividual]] = None) -> Dict[str, Any]:
        diversity_gain = None
        if population is not None and mutants is not None:
            try:
                diversity_gain = self.mutation_diversity_gain(population, mutants)
            except Exception:
                diversity_gain = None
        return {
            "strategy": [s.value for s in MutationStrategyType],
            "mutation_factor": self.config.mutation_factor,
            "mutants_generated": self.statistics_tracker.mutants_generated,
            "repair_count": self.statistics_tracker.constraint_repairs,
            "diversity_gain": diversity_gain,
            "mutation_efficiency": self.mutation_efficiency_score(population=population),
            "mutation_hash": self.mutation_hash(),
            "mutation_health_score": self.mutation_health_score(population=population),
            "mutation_stability_score": self.mutation_stability_score(population=population),
            "mutation_readiness_score": self.mutation_readiness_score(population=population),
            "diversity_preservation_score": self.diversity_preservation_score(population=population),
            "adaptive_mutation_score": self.adaptive_mutation_score(population=population),
            "mutation_difficulty_score": self.mutation_difficulty_score(population=population),
            "effective_mutation_dimensions": self.effective_mutation_dimensions(),
            "mutation_search_burden": self.mutation_search_burden(),
        }

    def mutation_statistics_report(self) -> Dict[str, Any]:
        return {
            "mutation_calls": self.statistics_tracker.mutation_calls,
            "mutants_generated": self.statistics_tracker.mutants_generated,
            "rank_mutations": self.statistics_tracker.rank_mutations,
            "scaling_mutations": self.statistics_tracker.scaling_mutations,
            "placement_mutations": self.statistics_tracker.placement_mutations,
            "rejected_mutants": self.statistics_tracker.rejected_mutants,
            "constraint_repairs": self.statistics_tracker.constraint_repairs,
            "adaptive_mutations": self.statistics_tracker.adaptive_mutations,
            "diversity_guided_mutations": self.statistics_tracker.diversity_guided_mutations,
            "metadata_exports": self.statistics_tracker.metadata_exports,
        }

    # ------------------------------------------------------------------
    # EXPERIMENT TRACKING
    # ------------------------------------------------------------------

    def experiment_metadata(self) -> Dict[str, Any]:
        return {
            "mutation_configuration": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "mutation_statistics": self.statistics().to_dict(),
            "mutation_hash": self.mutation_hash(),
            "mutation_signature": self.mutation_signature(),
            "search_burden": self.mutation_search_burden(),
            "effective_mutation_dimensions": self.effective_mutation_dimensions(),
            "strategy_support": [s.value for s in MutationStrategyType],
        }

    def experiment_signature(self) -> str:
        return _stable_hash(self.experiment_metadata())

    def export_configuration(self) -> Dict[str, Any]:
        return {
            "mutation_configuration": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "mutation_statistics": self.statistics().to_dict(),
            "metadata": self.metadata(),
            "diagnostics": self.diagnostics(),
            "mutation_report": self.mutation_report(),
            "experiment_metadata": self.experiment_metadata(),
            "experiment_signature": self.experiment_signature(),
            "fingerprint": self.mutation_fingerprint(),
        }

    # ------------------------------------------------------------------
    # STATISTICS / STATE
    # ------------------------------------------------------------------

    def statistics(self) -> MutationStatistics:
        return self.statistics_tracker

    def verify_integrity(self) -> bool:
        try:
            _ = self.mutation_hash()
            _ = self.mutation_signature()
            return True
        except Exception:
            return False

    def validate_state(self) -> None:
        self.validate_mutant(
            self._merge_chromosome(
                [self._rank_candidates[0]] * self._layer_count,
                [self._scaling_candidates[0]] * self._layer_count,
                list(self._placement_candidates[0]),
            )
        )

        if not self.verify_integrity():
            raise RuntimeError(
                "MutationEngine integrity check failed."
            )

    def reset_statistics(self) -> None:
        self.statistics_tracker = MutationStatistics()

    def clone(self) -> "MutationEngine":
        return MutationEngine(
            config=replace(self.config),
            schema=self.schema,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MutationStrategyType",
    "MutationConfig",
    "MutationStatistics",
    "MutationEngine",
]
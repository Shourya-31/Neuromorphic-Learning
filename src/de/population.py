from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import json
import math
import random
import statistics

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


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _coerce_mapping_or_empty(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _extract_layer_names(source: Any = None, layer_names: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    if layer_names is not None:
        names = [str(name) for name in layer_names if str(name).strip()]
        if not names:
            raise ValueError("layer_names cannot be empty.")
        return tuple(names)

    if source is None:
        raise ValueError("layer_names could not be inferred.")

    # Common reflective hooks from the existing repository.
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
            for name, module in source.named_modules():
                if name.strip():
                    discovered.append(str(name))
            if discovered:
                return tuple(discovered)
        except Exception:
            pass

    raise ValueError("Unable to infer layer names from the provided source.")


def _normalize_rank_config(rank_config: Any) -> Any:
    if rank_config is None:
        raise ValueError("rank_config is required.")
    if isinstance(rank_config, RankConfig):
        return rank_config
    return rank_config


def _normalize_scaling_config(scaling_config: Any) -> Any:
    if scaling_config is None:
        raise ValueError("scaling_config is required.")
    if isinstance(scaling_config, ScalingConfig):
        return scaling_config
    return scaling_config


def _normalize_placement_config(placement_config: Any) -> Any:
    if placement_config is None:
        raise ValueError("placement_config is required.")
    if isinstance(placement_config, PlacementConfig):
        return placement_config
    return placement_config


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

    # Heuristic set of diverse candidate masks.
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

    # Spread-out masks.
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


def _placement_density(mask: Sequence[int]) -> float:
    if not mask:
        return 0.0
    return float(sum(1 for v in mask if int(v) != 0) / len(mask))


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

@dataclass
class PopulationConfig:
    population_size: int = 32
    seed: Optional[int] = None
    allow_duplicate_individuals: bool = False
    enable_diversity_checks: bool = True
    enable_population_hashing: bool = True
    enable_metadata_tracking: bool = True
    enable_statistics_tracking: bool = True
    rank_weight: float = 1.0
    scaling_weight: float = 1.0
    placement_weight: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_integer("population_size", self.population_size)

        if self.seed is not None:
            _validate_integer("seed", self.seed)

        _validate_bool("allow_duplicate_individuals", self.allow_duplicate_individuals)
        _validate_bool("enable_diversity_checks", self.enable_diversity_checks)
        _validate_bool("enable_population_hashing", self.enable_population_hashing)
        _validate_bool("enable_metadata_tracking", self.enable_metadata_tracking)
        _validate_bool("enable_statistics_tracking", self.enable_statistics_tracking)

        _validate_numeric("rank_weight", self.rank_weight)
        _validate_numeric("scaling_weight", self.scaling_weight)
        _validate_numeric("placement_weight", self.placement_weight)

        if self.rank_weight < 0 or self.scaling_weight < 0 or self.placement_weight < 0:
            raise ValueError("Population weights must be non-negative.")

        if (self.rank_weight + self.scaling_weight + self.placement_weight) <= 0:
            raise ValueError("At least one population weight must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# STATISTICS
# ============================================================================

@dataclass
class PopulationStatistics:
    initialization_calls: int = 0
    population_creations: int = 0
    individual_creations: int = 0
    duplicate_individuals: int = 0
    rejected_individuals: int = 0
    validation_failures: int = 0
    encoding_operations: int = 0
    decoding_operations: int = 0
    metadata_exports: int = 0
    hash_operations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SCHEMA
# ============================================================================

@dataclass(frozen=True)
class PopulationSchema:
    layer_names: Tuple[str, ...]
    rank_candidates: Tuple[int, ...]
    scaling_candidates: Tuple[float, ...]
    placement_candidates: Tuple[Tuple[int, ...], ...]
    minimum_layers: int
    maximum_layers: Optional[int]
    allow_empty_placement: bool

    def __post_init__(self) -> None:
        if not self.layer_names:
            raise ValueError("layer_names cannot be empty.")
        if not self.rank_candidates:
            raise ValueError("rank_candidates cannot be empty.")
        if not self.scaling_candidates:
            raise ValueError("scaling_candidates cannot be empty.")
        if not self.placement_candidates:
            raise ValueError("placement_candidates cannot be empty.")

        if self.minimum_layers < 0:
            raise ValueError("minimum_layers must be non-negative.")
        if self.maximum_layers is not None and self.maximum_layers < 0:
            raise ValueError("maximum_layers must be non-negative when provided.")
        if self.maximum_layers is not None and self.maximum_layers < self.minimum_layers:
            raise ValueError("maximum_layers must be >= minimum_layers.")
        if not self.allow_empty_placement and self.minimum_layers <= 0:
            raise ValueError("minimum_layers must be positive when empty placement is disabled.")

        object.__setattr__(self, "layer_names", tuple(str(x) for x in self.layer_names))
        object.__setattr__(self, "rank_candidates", tuple(sorted(set(int(v) for v in self.rank_candidates))))
        object.__setattr__(self, "scaling_candidates", tuple(sorted(set(float(v) for v in self.scaling_candidates))))
        normalized_masks = []
        for mask in self.placement_candidates:
            mask = tuple(1 if int(v) else 0 for v in mask)
            if len(mask) != len(self.layer_names):
                raise ValueError("Placement mask length mismatch.")
            normalized_masks.append(mask)
        object.__setattr__(self, "placement_candidates", tuple(dict.fromkeys(normalized_masks)))

    @property
    def layer_count(self) -> int:
        return len(self.layer_names)

    @property
    def rank_dimension(self) -> int:
        return self.layer_count

    @property
    def scaling_dimension(self) -> int:
        return self.layer_count

    @property
    def placement_dimension(self) -> int:
        return self.layer_count

    @property
    def effective_search_dimensions(self) -> int:
        return self.rank_dimension + self.scaling_dimension + self.placement_dimension

    @property
    def chromosome_length(self) -> int:
        return self.effective_search_dimensions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_names": list(self.layer_names),
            "rank_candidates": list(self.rank_candidates),
            "scaling_candidates": list(self.scaling_candidates),
            "placement_candidates": [list(mask) for mask in self.placement_candidates],
            "minimum_layers": self.minimum_layers,
            "maximum_layers": self.maximum_layers,
            "allow_empty_placement": self.allow_empty_placement,
        }


# ============================================================================
# INDIVIDUAL
# ============================================================================

@dataclass(frozen=True)
class DEIndividual:
    rank_encoding: Tuple[int, ...]
    scaling_encoding: Tuple[float, ...]
    placement_encoding: Tuple[int, ...]
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    identifier: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank_encoding", tuple(int(v) for v in self.rank_encoding))
        object.__setattr__(self, "scaling_encoding", tuple(float(v) for v in self.scaling_encoding))
        object.__setattr__(self, "placement_encoding", tuple(1 if int(v) else 0 for v in self.placement_encoding))

        if self.fitness is not None:
            _validate_finite_numeric("fitness", self.fitness)

        _validate_non_negative_integer("generation", self.generation)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        if self.identifier is None:
            object.__setattr__(self, "identifier", self._compute_identifier())
        else:
            if not isinstance(self.identifier, str) or not self.identifier.strip():
                raise ValueError("identifier must be a non-empty string.")
            object.__setattr__(self, "identifier", self.identifier)

        object.__setattr__(self, "metadata", {str(k): v for k, v in dict(self.metadata).items()})

    def _payload(self, include_fitness: bool = True) -> Dict[str, Any]:
        payload = {
            "rank_encoding": list(self.rank_encoding),
            "scaling_encoding": list(self.scaling_encoding),
            "placement_encoding": list(self.placement_encoding),
            "generation": int(self.generation),
            "metadata": self.metadata,
        }
        if include_fitness:
            payload["fitness"] = self.fitness
        return payload

    def _compute_identifier(self) -> str:
        return _stable_hash(self._payload(include_fitness=True))

    def signature(self) -> str:
        return self.identifier or self._compute_identifier()

    def individual_hash(self) -> str:
        return self.signature()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank_encoding": list(self.rank_encoding),
            "scaling_encoding": list(self.scaling_encoding),
            "placement_encoding": list(self.placement_encoding),
            "fitness": self.fitness,
            "metadata": dict(self.metadata),
            "generation": int(self.generation),
            "identifier": self.signature(),
        }

    def to_chromosome(self) -> List[Union[int, float]]:
        return [*self.rank_encoding, *self.scaling_encoding, *self.placement_encoding]

    def with_fitness(self, fitness: Optional[float]) -> "DEIndividual":
        return replace(self, fitness=fitness)

    def with_metadata(self, **metadata: Any) -> "DEIndividual":
        merged = dict(self.metadata)
        merged.update(metadata)
        return replace(self, metadata=merged)

    def clone(self, *, generation: Optional[int] = None) -> "DEIndividual":
        return replace(self, generation=self.generation if generation is None else int(generation))


# ============================================================================
# POPULATION CONTAINER
# ============================================================================

class Population:
    def __init__(
        self,
        config: PopulationConfig,
        schema: PopulationSchema,
        individuals: Optional[Sequence[DEIndividual]] = None,
    ) -> None:
        if not isinstance(config, PopulationConfig):
            raise TypeError("config must be PopulationConfig.")
        if not isinstance(schema, PopulationSchema):
            raise TypeError("schema must be PopulationSchema.")

        self.config = config
        self.schema = schema
        self.statistics_tracker = PopulationStatistics()
        self.individuals: List[DEIndividual] = []

        if individuals is not None:
            self.individuals = [self._coerce_individual(ind) for ind in individuals]

        self._refresh_statistics()
        self.validate_population()

    # ------------------------------------------------------------------
    # Internal coercion
    # ------------------------------------------------------------------

    def _coerce_individual(self, individual: Any) -> DEIndividual:
        if isinstance(individual, DEIndividual):
            return individual

        if isinstance(individual, Mapping):
            if "rank_encoding" not in individual or "scaling_encoding" not in individual or "placement_encoding" not in individual:
                raise ValueError("Individual mapping must contain rank_encoding, scaling_encoding, and placement_encoding.")
            return DEIndividual(
                rank_encoding=tuple(individual["rank_encoding"]),
                scaling_encoding=tuple(individual["scaling_encoding"]),
                placement_encoding=tuple(individual["placement_encoding"]),
                fitness=individual.get("fitness"),
                metadata=_coerce_mapping_or_empty(individual.get("metadata")),
                generation=int(individual.get("generation", 0)),
                identifier=individual.get("identifier"),
            )

        raise TypeError("Unsupported individual representation.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_individual(self, individual: DEIndividual) -> None:
        if not isinstance(individual, DEIndividual):
            raise TypeError("individual must be DEIndividual.")

        expected = self.schema.layer_count
        if len(individual.rank_encoding) != expected:
            raise ValueError(f"rank_encoding length mismatch. Expected {expected}, got {len(individual.rank_encoding)}.")
        if len(individual.scaling_encoding) != expected:
            raise ValueError(f"scaling_encoding length mismatch. Expected {expected}, got {len(individual.scaling_encoding)}.")
        if len(individual.placement_encoding) != expected:
            raise ValueError(f"placement_encoding length mismatch. Expected {expected}, got {len(individual.placement_encoding)}.")

        rank_set = set(self.schema.rank_candidates)
        scaling_set = set(self.schema.scaling_candidates)
        placement_set = set(self.schema.placement_candidates)

        if any(rank not in rank_set for rank in individual.rank_encoding):
            raise ValueError("rank_encoding contains values outside the legal rank search space.")

        if any(alpha not in scaling_set for alpha in individual.scaling_encoding):
            raise ValueError("scaling_encoding contains values outside the legal scaling search space.")

        if any(bit not in (0, 1) for bit in individual.placement_encoding):
            raise ValueError("placement_encoding must be binary.")

        if tuple(individual.placement_encoding) not in placement_set:
            raise ValueError("placement_encoding is not a legal placement mask.")

        active = sum(individual.placement_encoding)
        density = active / max(1, len(individual.placement_encoding))
        if active < self.schema.minimum_layers:
            raise ValueError("placement active layer count is below minimum_layers.")
        if self.schema.maximum_layers is not None and active > self.schema.maximum_layers:
            raise ValueError("placement active layer count exceeds maximum_layers.")
        if density < 0.0 or density > 1.0:
            raise ValueError("placement density out of range.")
        if individual.fitness is not None:
            _validate_finite_numeric("fitness", individual.fitness)
        _validate_non_negative_integer("generation", individual.generation)

    def validate_encodings(self, encodings: Sequence[Sequence[Union[int, float]]]) -> None:
        if not isinstance(encodings, Sequence):
            raise TypeError("encodings must be a sequence.")
        for vector in encodings:
            self._decode_chromosome(vector)  # raises descriptive errors

    def validate_population(self) -> bool:
        self.statistics_tracker.validation_failures = 0
        if len(self.individuals) != self.config.population_size:
            raise ValueError(
                f"Population size mismatch. Expected {self.config.population_size}, got {len(self.individuals)}."
            )

        seen = set()
        duplicates = 0
        for ind in self.individuals:
            try:
                self.validate_individual(ind)
            except Exception:
                self.statistics_tracker.validation_failures += 1
                raise
            sig = ind.signature()
            if sig in seen:
                duplicates += 1
                if not self.config.allow_duplicate_individuals:
                    raise ValueError(f"Duplicate individual detected: {sig}")
            seen.add(sig)

        self.statistics_tracker.duplicate_individuals = duplicates
        return True

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode_individual(self, individual: DEIndividual) -> List[Union[int, float]]:
        self.statistics_tracker.encoding_operations += 1
        self.validate_individual(individual)
        return individual.to_chromosome()

    def _decode_chromosome(self, vector: Sequence[Union[int, float]]) -> DEIndividual:
        if not isinstance(vector, Sequence):
            raise TypeError("chromosome must be a sequence.")
        expected = self.schema.chromosome_length
        if len(vector) != expected:
            raise ValueError(f"encoding length mismatch. Expected {expected}, got {len(vector)}.")

        n = self.schema.layer_count
        rank_vec = _canonical_int_tuple(vector[:n])
        scaling_vec = _canonical_float_tuple(vector[n:2*n])
        placement_vec = _canonical_mask_tuple(vector[2*n:3*n])

        return DEIndividual(
            rank_encoding=rank_vec,
            scaling_encoding=scaling_vec,
            placement_encoding=placement_vec,
            fitness=None,
            metadata={},
            generation=0,
        )

    def decode_individual(
        self,
        vector: Sequence[Union[int, float]],
        *,
        generation: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        fitness: Optional[float] = None,
    ) -> DEIndividual:
        self.statistics_tracker.decoding_operations += 1
        ind = self._decode_chromosome(vector)
        return replace(
            ind,
            generation=int(generation),
            fitness=fitness,
            metadata=_coerce_mapping_or_empty(metadata),
        )

    def encode_population(self) -> List[List[Union[int, float]]]:
        return [self.encode_individual(ind) for ind in self.individuals]

    def decode_population(self, vectors: Sequence[Sequence[Union[int, float]]]) -> List[DEIndividual]:
        if len(vectors) != self.config.population_size:
            raise ValueError(f"Population vector count mismatch. Expected {self.config.population_size}, got {len(vectors)}.")
        return [self.decode_individual(vec) for vec in vectors]

    # ------------------------------------------------------------------
    # Diversity analysis
    # ------------------------------------------------------------------

    def _flatten_numeric(self, individual: DEIndividual) -> List[float]:
        return [float(v) for v in individual.rank_encoding] + [float(v) for v in individual.scaling_encoding] + [float(v) for v in individual.placement_encoding]

    def population_uniqueness_ratio(self) -> float:
        if not self.individuals:
            return 0.0
        unique = len({ind.signature() for ind in self.individuals})
        return float(unique / len(self.individuals))

    def duplicate_ratio(self) -> float:
        return float(1.0 - self.population_uniqueness_ratio())

    def population_entropy(self) -> float:
        if not self.individuals:
            return 0.0
        counts: Dict[str, int] = {}
        for ind in self.individuals:
            counts[ind.signature()] = counts.get(ind.signature(), 0) + 1
        return _entropy_from_counts(list(counts.values()))

    def population_variance(self) -> float:
        if len(self.individuals) <= 1:
            return 0.0
        flattened = [self._flatten_numeric(ind) for ind in self.individuals]
        dims = len(flattened[0])
        variances = []
        for j in range(dims):
            col = [vec[j] for vec in flattened]
            variances.append(_safe_variance(col))
        return _safe_mean(variances)

    def encoding_dispersion(self) -> float:
        if len(self.individuals) <= 1:
            return 0.0
        vectors = [ind.to_chromosome() for ind in self.individuals]
        total = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total += _euclidean_distance(
                    [float(v) for v in vectors[i]],
                    [float(v) for v in vectors[j]],
                )
                count += 1
        return float(total / max(1, count))

    def population_diversity(self) -> Dict[str, float]:

        return {
            "population_entropy": self.population_entropy(),
            "population_variance": self.population_variance(),
            "uniqueness_ratio": self.population_uniqueness_ratio(),
            "duplicate_ratio": self.duplicate_ratio(),
            "encoding_dispersion": self.encoding_dispersion(),
        }
    
    def average_pairwise_distance(self) -> float:
        if len(self.individuals) <= 1:
            return 0.0

        vectors = [self._flatten_numeric(ind) for ind in self.individuals]

        distances = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distances.append(
                    _euclidean_distance(vectors[i], vectors[j])
                )

        return _safe_mean(distances)


    def population_concentration(self) -> float:
        return float(1.0 - self.population_uniqueness_ratio())


    def diversity_health_score(self) -> float:
        entropy = self.population_entropy()
        uniqueness = self.population_uniqueness_ratio()
        variance = self.population_variance()

        entropy_score = min(1.0, entropy / max(1.0, self.search_space_entropy()))
        variance_score = math.tanh(variance)

        score = (
            0.4 * entropy_score +
            0.4 * uniqueness +
            0.2 * variance_score
        )

        return float(max(0.0, min(1.0, score)))


    def premature_convergence_risk(self) -> float:
        return float(
            max(
                0.0,
                min(
                    1.0,
                    1.0 - self.diversity_health_score()
                )
            )
        )


    def exploration_score(self) -> float:
        return self.diversity_health_score()


    def exploitation_score(self) -> float:
        return float(1.0 - self.exploration_score())


    def population_health_score(self) -> float:
        diversity = self.diversity_health_score()

        uniqueness = self.population_uniqueness_ratio()

        dispersion = math.tanh(
            self.encoding_dispersion()
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    (
                        0.4 * diversity +
                        0.3 * uniqueness +
                        0.3 * dispersion
                    )
                )
            )
        )
    
    def convergence_readiness(self) -> float:
        return float(
            (
                self.population_health_score()
                +
                (1.0 - self.premature_convergence_risk())
            ) / 2.0
        )


    def stagnation_risk(self) -> float:
        return float(
            max(
                0.0,
                min(
                    1.0,
                    1.0 - self.encoding_dispersion() /
                    max(1.0, self.search_space_entropy())
                )
            )
        )

    def diversity_report(self) -> Dict[str, Any]:
        return {
            "population_entropy": self.population_entropy(),
            "population_variance": self.population_variance(),
            "uniqueness_ratio": self.population_uniqueness_ratio(),
            "duplicate_ratio": self.duplicate_ratio(),
            "encoding_dispersion": self.encoding_dispersion(),
            "average_pairwise_distance": self.average_pairwise_distance(),
            "population_concentration": self.population_concentration(),
            "diversity_health_score": self.diversity_health_score(),
            "premature_convergence_risk": self.premature_convergence_risk(),
            "exploration_score": self.exploration_score(),
            "exploitation_score": self.exploitation_score(),
            "population_health_score": self.population_health_score(),
            "convergence_readiness": self.convergence_readiness(),
            "stagnation_risk": self.stagnation_risk(),
        }

    # ------------------------------------------------------------------
    # Search-space analytics
    # ------------------------------------------------------------------

    def rank_dimensions(self) -> int:
        return self.schema.rank_dimension

    def scaling_dimensions(self) -> int:
        return self.schema.scaling_dimension

    def placement_dimensions(self) -> int:
        return self.schema.placement_dimension

    def effective_search_dimensions(self) -> int:
        return self.schema.effective_search_dimensions

    def estimated_search_space_size(self) -> Union[int, float]:
        rank_count = len(self.schema.rank_candidates)
        scaling_count = len(self.schema.scaling_candidates)
        placement_count = len(self.schema.placement_candidates)
        layer_count = self.schema.layer_count
        try:
            total = (rank_count ** layer_count) * (scaling_count ** layer_count) * placement_count
            return int(total)
        except OverflowError:
            return float("inf")

    def search_space_entropy(self) -> float:
        rank_count = max(1, len(self.schema.rank_candidates))
        scaling_count = max(1, len(self.schema.scaling_candidates))
        placement_count = max(1, len(self.schema.placement_candidates))
        layer_count = max(1, self.schema.layer_count)
        return float(
            layer_count * math.log2(rank_count)
            + layer_count * math.log2(scaling_count)
            + math.log2(placement_count)
        )

    def search_space_complexity(self) -> float:
        return float(self.effective_search_dimensions() * self.search_space_entropy())
    
    def rank_contribution_ratio(self) -> float:
        total = self.effective_search_dimensions()
        return float(self.rank_dimensions() / total)


    def scaling_contribution_ratio(self) -> float:
        total = self.effective_search_dimensions()
        return float(self.scaling_dimensions() / total)


    def placement_contribution_ratio(self) -> float:
        total = self.effective_search_dimensions()
        return float(self.placement_dimensions() / total)


    def optimization_balance_score(self) -> float:
        values = [
            self.rank_contribution_ratio(),
            self.scaling_contribution_ratio(),
            self.placement_contribution_ratio(),
        ]

        target = 1.0 / len(values)

        deviation = _safe_mean(
            [abs(v - target) for v in values]
        )

        return float(max(0.0, 1.0 - deviation))


    def effective_candidate_count(self) -> int:
        return int(
            (
                len(self.schema.rank_candidates) ** self.schema.layer_count
            )
            *
            (
                len(self.schema.scaling_candidates) ** self.schema.layer_count
            )
            *
            len(self.schema.placement_candidates)
        )


    def optimization_difficulty_score(self) -> float:
        complexity = self.search_space_complexity()

        return float(
            math.log10(
                max(
                    10.0,
                    complexity + 10.0
                )
            )
        )
    
    def search_space_utilization(self) -> float:
        total = max(
            1,
            self.effective_candidate_count()
        )

        return float(
            min(
                1.0,
                len(self.individuals) / total
            )
        )


    def constraint_reduction_factor(self) -> float:
        active_masks = len(self.schema.placement_candidates)

        theoretical_masks = 2 ** self.schema.layer_count

        return float(
            active_masks /
            max(1, theoretical_masks)
        )


    # ------------------------------------------------------------------
    # Hash / reproducibility
    # ------------------------------------------------------------------

    def individual_hash(self, individual: DEIndividual) -> str:
        self.statistics_tracker.hash_operations += 1
        return individual.signature()

    def population_hash(self) -> str:
        self.statistics_tracker.hash_operations += 1
        payload = {
            "config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "individuals": [ind.to_dict() for ind in self.individuals],
        }
        return _stable_hash(payload)

    def population_signature(self) -> str:
        return f"{len(self.individuals)}:{self.population_hash()}"

    def population_fingerprint(self) -> Dict[str, Any]:
        return {
            "population_hash": self.population_hash(),
            "population_signature": self.population_signature(),
            "population_size": len(self.individuals),
            "unique_individuals": len({ind.signature() for ind in self.individuals}),
        }

    # ------------------------------------------------------------------
    # Reports / metadata
    # ------------------------------------------------------------------

    def search_space_report(self) -> Dict[str, Any]:
        return {
            "rank_dimensions": self.rank_dimensions(),
            "scaling_dimensions": self.scaling_dimensions(),
            "placement_dimensions": self.placement_dimensions(),
            "effective_search_dimensions": self.effective_search_dimensions(),

            "rank_candidate_count": len(self.schema.rank_candidates),
            "scaling_candidate_count": len(self.schema.scaling_candidates),
            "placement_candidate_count": len(self.schema.placement_candidates),

            "optimization_balance_score": self.optimization_balance_score(),

            "effective_candidate_count": self.effective_candidate_count(),

            "estimated_search_space_size": self.estimated_search_space_size(),
            "search_space_entropy": self.search_space_entropy(),
            "search_space_complexity": self.search_space_complexity(),

            "constraint_reduction_factor": self.constraint_reduction_factor(),

            "search_space_utilization": self.search_space_utilization(),

            "optimization_difficulty_score": self.optimization_difficulty_score(),
        }

    def diversity_report_full(self) -> Dict[str, Any]:
        return self.diversity_report()

    def metadata(self) -> Dict[str, Any]:
        self.statistics_tracker.metadata_exports += 1
        return {
            "population_size": len(self.individuals),
            "search_dimensions": self.effective_search_dimensions(),
            "rank_dimensions": self.rank_dimensions(),
            "scaling_dimensions": self.scaling_dimensions(),
            "placement_dimensions": self.placement_dimensions(),
            "search_space_size": self.estimated_search_space_size(),
            "population_diversity": self.population_diversity(),
            "population_entropy": self.population_entropy(),
            "uniqueness_ratio": self.population_uniqueness_ratio(),
            "population_hash": self.population_hash() if self.config.enable_population_hashing else None,
            "population_signature": self.population_signature(),
            "population_fingerprint": self.population_fingerprint() if self.config.enable_population_hashing else None,
            "schema": self.schema.to_dict(),
            "configuration": self.config.to_dict(),
            "statistics": self.statistics().to_dict() if self.config.enable_statistics_tracking else self.statistics_tracker.to_dict(),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
        "population_size": len(self.individuals),

        "duplicate_ratio": self.duplicate_ratio(),

        "population_variance": self.population_variance(),

        "encoding_dispersion": self.encoding_dispersion(),

        "population_health_score": self.population_health_score(),

        "premature_convergence_risk": self.premature_convergence_risk(),

        "exploration_score": self.exploration_score(),

        "exploitation_score": self.exploitation_score(),

        "validation_failures": self.statistics_tracker.validation_failures,

        "population_hash": (
            self.population_hash()
            if self.config.enable_population_hashing
            else None
        ),

        "search_space_size": self.estimated_search_space_size(),

        "search_space_entropy": self.search_space_entropy(),

        "search_space_complexity": self.search_space_complexity(),

        "optimization_difficulty_score": self.optimization_difficulty_score(),

        "evolution_health": self.evolution_health_report(),
}

    def population_report(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata() if self.config.enable_metadata_tracking else {},
            "diagnostics": self.diagnostics(),
            "diversity": self.diversity_report(),
            "search_space": self.search_space_report(),
        }
    
    def mutation_readiness(self) -> float:
        return self.population_health_score()


    def crossover_readiness(self) -> float:
        return self.population_health_score()


    def selection_readiness(self) -> float:
        return self.population_health_score()


    def optimizer_readiness(self) -> float:
        return float(
            (
                self.population_health_score()
                + self.optimization_balance_score()
            ) / 2.0
        )


    def evolution_health_report(self) -> Dict[str, float]:
        return {
            "mutation_readiness": self.mutation_readiness(),
            "crossover_readiness": self.crossover_readiness(),
            "selection_readiness": self.selection_readiness(),
            "optimizer_readiness": self.optimizer_readiness(),
            "population_health_score": self.population_health_score(),
            "premature_convergence_risk": self.premature_convergence_risk(),
        }

    def experiment_metadata(self) -> Dict[str, Any]:
        return {
            "population_size": self.config.population_size,
            "seed": self.config.seed,
            "allow_duplicate_individuals": self.config.allow_duplicate_individuals,
            "enable_diversity_checks": self.config.enable_diversity_checks,
            "enable_population_hashing": self.config.enable_population_hashing,
            "enable_metadata_tracking": self.config.enable_metadata_tracking,
            "enable_statistics_tracking": self.config.enable_statistics_tracking,
            "rank_weight": self.config.rank_weight,
            "scaling_weight": self.config.scaling_weight,
            "placement_weight": self.config.placement_weight,
            "population_signature": self.population_signature(),
            "schema": self.schema.to_dict(),
        }

    def experiment_signature(self) -> str:
        return _stable_hash(self.experiment_metadata())

    def export_configuration(self) -> Dict[str, Any]:
        return {
            "population_config": self.config.to_dict(),
            "schema": self.schema.to_dict(),
            "population_metadata": self.metadata(),
            "diagnostics": self.diagnostics(),
            "search_space_report": self.search_space_report(),
            "diversity_report": self.diversity_report(),
            "experiment_metadata": self.experiment_metadata(),
            "experiment_signature": self.experiment_signature(),
            "individuals": [ind.to_dict() for ind in self.individuals],
            "encodings": self.encode_population(),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _refresh_statistics(self) -> None:
        self.statistics_tracker.population_creations += 1 if self.individuals else 0
        self.statistics_tracker.individual_creations = len(self.individuals)

    def statistics(self) -> PopulationStatistics:
        return self.statistics_tracker

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clone(self) -> "Population":
        return Population(self.config, self.schema, [ind.clone() for ind in self.individuals])

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __repr__(self) -> str:
        return f"Population(size={len(self.individuals)}, hash={self.population_hash() if self.config.enable_population_hashing else 'disabled'})"


# ============================================================================
# FACTORY
# ============================================================================

class PopulationFactory:
    @staticmethod
    def _rng(seed: Optional[int]) -> random.Random:
        return random.Random(seed)

    @staticmethod
    def _make_individual(
        rank_encoding: Sequence[int],
        scaling_encoding: Sequence[float],
        placement_encoding: Sequence[int],
        *,
        generation: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        fitness: Optional[float] = None,
        identifier: Optional[str] = None,
    ) -> DEIndividual:
        return DEIndividual(
            rank_encoding=tuple(int(v) for v in rank_encoding),
            scaling_encoding=tuple(float(v) for v in scaling_encoding),
            placement_encoding=tuple(1 if int(v) else 0 for v in placement_encoding),
            fitness=fitness,
            metadata=_coerce_mapping_or_empty(metadata),
            generation=generation,
            identifier=identifier,
        )

    @staticmethod
    def _decode_vector(
        schema: PopulationSchema,
        vector: Sequence[Union[int, float]],
        *,
        generation: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        fitness: Optional[float] = None,
        identifier: Optional[str] = None,
    ) -> DEIndividual:
        if not isinstance(vector, Sequence):
            raise TypeError("chromosome must be a sequence.")
        expected = schema.chromosome_length
        if len(vector) != expected:
            raise ValueError(f"encoding length mismatch. Expected {expected}, got {len(vector)}.")

        n = schema.layer_count
        rank_vec = _canonical_int_tuple(vector[:n])
        scaling_vec = _canonical_float_tuple(vector[n:2*n])
        placement_vec = _canonical_mask_tuple(vector[2*n:3*n])

        return DEIndividual(
            rank_encoding=rank_vec,
            scaling_encoding=scaling_vec,
            placement_encoding=placement_vec,
            fitness=fitness,
            metadata=_coerce_mapping_or_empty(metadata),
            generation=int(generation),
            identifier=identifier,
        )

    @classmethod
    def create_population_from_encodings(
        cls,
        *,
        encodings: Sequence[Sequence[Union[int, float]]],
        rank_config: Any,
        scaling_config: Any,
        placement_config: Any,
        population_config: Optional[PopulationConfig] = None,
        source: Any = None,
        layer_names: Optional[Sequence[str]] = None,
        generation: int = 0,
    ) -> Population:
        if population_config is None:
            population_config = PopulationConfig(population_size=len(encodings))
        schema = _infer_schema(
            rank_config=rank_config,
            scaling_config=scaling_config,
            placement_config=placement_config,
            source=source,
            layer_names=layer_names,
        )
        individuals = [cls._decode_vector(schema, vec, generation=generation) for vec in encodings]
        population = Population(population_config, schema, individuals)
        population.statistics_tracker.initialization_calls += 1
        return population

    @classmethod
    def create_population_from_metadata(
        cls,
        *,
        metadata: Mapping[str, Any],
        rank_config: Any,
        scaling_config: Any,
        placement_config: Any,
        population_config: Optional[PopulationConfig] = None,
        source: Any = None,
        layer_names: Optional[Sequence[str]] = None,
    ) -> Population:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        raw = dict(metadata)
        if population_config is None:
            pcfg = raw.get("population_config", {})
            if isinstance(pcfg, Mapping):
                population_config = PopulationConfig(
                    population_size=int(pcfg.get("population_size", len(raw.get("individuals", raw.get("encodings", []))) or 1)),
                    seed=pcfg.get("seed"),
                    allow_duplicate_individuals=bool(pcfg.get("allow_duplicate_individuals", False)),
                    enable_diversity_checks=bool(pcfg.get("enable_diversity_checks", True)),
                    enable_population_hashing=bool(pcfg.get("enable_population_hashing", True)),
                    enable_metadata_tracking=bool(pcfg.get("enable_metadata_tracking", True)),
                    enable_statistics_tracking=bool(pcfg.get("enable_statistics_tracking", True)),
                    rank_weight=float(pcfg.get("rank_weight", 1.0)),
                    scaling_weight=float(pcfg.get("scaling_weight", 1.0)),
                    placement_weight=float(pcfg.get("placement_weight", 1.0)),
                )
            else:
                population_config = PopulationConfig(population_size=len(raw.get("individuals", raw.get("encodings", []))) or 1)

        schema_raw = raw.get("schema", {})
        if isinstance(schema_raw, Mapping) and schema_raw:
            schema = PopulationSchema(
                layer_names=tuple(schema_raw["layer_names"]),
                rank_candidates=tuple(schema_raw["rank_candidates"]),
                scaling_candidates=tuple(schema_raw["scaling_candidates"]),
                placement_candidates=tuple(tuple(m) for m in schema_raw["placement_candidates"]),
                minimum_layers=int(schema_raw.get("minimum_layers", 1)),
                maximum_layers=schema_raw.get("maximum_layers"),
                allow_empty_placement=bool(schema_raw.get("allow_empty_placement", False)),
            )
        else:
            schema = _infer_schema(
                rank_config=rank_config,
                scaling_config=scaling_config,
                placement_config=placement_config,
                source=source,
                layer_names=layer_names,
            )

        individuals_data = raw.get("individuals")
        if individuals_data is None:
            individuals_data = raw.get("encodings")

        if individuals_data is None:
            raise ValueError("metadata does not contain individuals or encodings.")

        individuals: List[DEIndividual] = []
        for item in individuals_data:
            if isinstance(item, Mapping) and "rank_encoding" in item:
                individuals.append(
                    DEIndividual(
                        rank_encoding=tuple(item["rank_encoding"]),
                        scaling_encoding=tuple(item["scaling_encoding"]),
                        placement_encoding=tuple(item["placement_encoding"]),
                        fitness=item.get("fitness"),
                        metadata=_coerce_mapping_or_empty(item.get("metadata")),
                        generation=int(item.get("generation", 0)),
                        identifier=item.get("identifier"),
                    )
                )
            else:
                individuals.append(cls._decode_vector(schema, item))  # type: ignore[arg-type]

        population = Population(population_config, schema, individuals)
        population.statistics_tracker.initialization_calls += 1
        return population

    @classmethod
    def create_seeded_population(
        cls,
        *,
        rank_config: Any,
        scaling_config: Any,
        placement_config: Any,
        population_config: Optional[PopulationConfig] = None,
        source: Any = None,
        layer_names: Optional[Sequence[str]] = None,
        initialization_strategy: str = "random",
        seed: Optional[int] = None,
    ) -> Population:
        if population_config is None:
            population_config = PopulationConfig(population_size=32, seed=seed)
        else:
            if seed is not None:
                population_config = replace(population_config, seed=seed)
        return cls.create_random_population(
            rank_config=rank_config,
            scaling_config=scaling_config,
            placement_config=placement_config,
            population_config=population_config,
            source=source,
            layer_names=layer_names,
            initialization_strategy=initialization_strategy,
        )

    @classmethod
    def create_random_population(
        cls,
        *,
        rank_config: Any,
        scaling_config: Any,
        placement_config: Any,
        population_config: Optional[PopulationConfig] = None,
        source: Any = None,
        layer_names: Optional[Sequence[str]] = None,
        initialization_strategy: str = "random",
        generation: int = 0,
    ) -> Population:
        if population_config is None:
            population_config = PopulationConfig()

        if not isinstance(population_config, PopulationConfig):
            raise TypeError("population_config must be PopulationConfig.")

        schema = _infer_schema(
            rank_config=rank_config,
            scaling_config=scaling_config,
            placement_config=placement_config,
            source=source,
            layer_names=layer_names,
        )

        rng = cls._rng(population_config.seed)
        population_config = replace(population_config)
        size = population_config.population_size

        individuals = cls._initialize_individuals(
            size=size,
            schema=schema,
            rng=rng,
            strategy=initialization_strategy.lower(),
            generation=generation,
            config=population_config,
        )
        population = Population(population_config, schema, individuals)
        population.statistics_tracker.initialization_calls += 1
        return population

    @classmethod
    def clone_population(cls, population: Population) -> Population:
        if not isinstance(population, Population):
            raise TypeError("population must be Population.")
        return population.clone()

    # ------------------------------------------------------------------
    # Initialization strategies
    # ------------------------------------------------------------------

    @classmethod
    def _initialize_individuals(
        cls,
        *,
        size: int,
        schema: PopulationSchema,
        rng: random.Random,
        strategy: str,
        generation: int,
        config: PopulationConfig,
    ) -> List[DEIndividual]:
        rank_candidates = list(schema.rank_candidates)
        scaling_candidates = list(schema.scaling_candidates)
        placement_candidates = list(schema.placement_candidates)

        if strategy not in {"random", "uniform", "stratified", "diversity_aware"}:
            raise ValueError(
                "initialization_strategy must be one of: random, uniform, stratified, diversity_aware."
            )

        individuals: List[DEIndividual] = []
        seen = set()

        def build_from_indices(rank_idx: int, scale_idx: int, place_idx: int, idx: int) -> DEIndividual:
            rank_encoding = [
                rank_candidates[(rank_idx + offset) % len(rank_candidates)]
                for offset in range(schema.layer_count)
            ]
            scaling_encoding = [
                scaling_candidates[(scale_idx + 2 * offset) % len(scaling_candidates)]
                for offset in range(schema.layer_count)
            ]
            mask = placement_candidates[place_idx % len(placement_candidates)]
            ind = cls._make_individual(
                rank_encoding=rank_encoding,
                scaling_encoding=scaling_encoding,
                placement_encoding=mask,
                generation=generation,
                metadata={
                    "initialization_strategy": strategy,
                    "population_index": idx,
                    "seed": config.seed,
                },
            )
            return ind

        if strategy == "random":
            for idx in range(size):
                for attempt in range(256):
                    ind = cls._make_individual(
                        rank_encoding=[rng.choice(rank_candidates) for _ in range(schema.layer_count)],
                        scaling_encoding=[rng.choice(scaling_candidates) for _ in range(schema.layer_count)],
                        placement_encoding=rng.choice(placement_candidates),
                        generation=generation,
                        metadata={
                            "initialization_strategy": strategy,
                            "population_index": idx,
                            "seed": config.seed,
                        },
                    )
                    sig = ind.signature()
                    if config.allow_duplicate_individuals or sig not in seen:
                        individuals.append(ind)
                        seen.add(sig)
                        break
                else:
                    raise RuntimeError("Unable to create a unique random individual within retry budget.")

        elif strategy == "uniform":
            for idx in range(size):
                rank_idx = int(round((idx / max(1, size - 1)) * (len(rank_candidates) - 1))) if size > 1 else len(rank_candidates) // 2
                scale_idx = int(round((idx / max(1, size - 1)) * (len(scaling_candidates) - 1))) if size > 1 else len(scaling_candidates) // 2
                place_idx = int(round((idx / max(1, size - 1)) * (len(placement_candidates) - 1))) if size > 1 else len(placement_candidates) // 2
                ind = build_from_indices(rank_idx, scale_idx, place_idx, idx)
                sig = ind.signature()
                if config.allow_duplicate_individuals or sig not in seen:
                    individuals.append(ind)
                    seen.add(sig)
                else:
                    # deterministic fallback perturbation
                    rank_idx = (rank_idx + idx + 1) % len(rank_candidates)
                    scale_idx = (scale_idx + idx + 1) % len(scaling_candidates)
                    place_idx = (place_idx + idx + 1) % len(placement_candidates)
                    ind = build_from_indices(rank_idx, scale_idx, place_idx, idx)
                    sig = ind.signature()
                    if not config.allow_duplicate_individuals and sig in seen:
                        raise RuntimeError("Uniform initialization produced duplicate individuals.")
                    individuals.append(ind)
                    seen.add(sig)

        elif strategy == "stratified":
            rank_cycle = cls._cycled_indices(len(rank_candidates), size, rng)
            scale_cycle = cls._cycled_indices(len(scaling_candidates), size, rng)
            place_cycle = cls._cycled_indices(len(placement_candidates), size, rng)
            for idx in range(size):
                ind = build_from_indices(rank_cycle[idx], scale_cycle[idx], place_cycle[idx], idx)
                sig = ind.signature()
                if config.allow_duplicate_individuals or sig not in seen:
                    individuals.append(ind)
                    seen.add(sig)
                else:
                    # shift all indices until unique
                    for shift in range(1, len(rank_candidates) + len(scaling_candidates) + len(placement_candidates) + 1):
                        ind = build_from_indices(
                            rank_cycle[idx] + shift,
                            scale_cycle[idx] + 2 * shift,
                            place_cycle[idx] + 3 * shift,
                            idx,
                        )
                        sig = ind.signature()
                        if config.allow_duplicate_individuals or sig not in seen:
                            individuals.append(ind)
                            seen.add(sig)
                            break
                    else:
                        raise RuntimeError("Stratified initialization could not produce a unique individual.")

        else:  # diversity_aware
            pool = cls._candidate_product_pool(rank_candidates, scaling_candidates, placement_candidates, schema.layer_count, rng)
            if not pool:
                raise RuntimeError("Diversity-aware initialization has no candidate pool.")
            for idx in range(size):
                best = None
                best_score = -1.0
                proposals = pool if len(pool) <= 64 else rng.sample(pool, 64)
                for proposal in proposals:
                    ind = cls._proposal_to_individual(proposal, idx, generation, config, strategy, seen, allow_duplicates=config.allow_duplicate_individuals)
                    score = cls._proposal_diversity_score(ind, individuals)
                    if score > best_score:
                        best_score = score
                        best = ind
                if best is None:
                    raise RuntimeError("Failed to generate diversity-aware individual.")
                sig = best.signature()
                if not config.allow_duplicate_individuals and sig in seen:
                    # deterministic fallback
                    for proposal in pool:
                        ind = cls._proposal_to_individual(proposal, idx, generation, config, strategy, seen, allow_duplicates=False)
                        sig = ind.signature()
                        if sig not in seen:
                            best = ind
                            break
                individuals.append(best)
                seen.add(best.signature())

        if len(individuals) != size:
            raise RuntimeError("Population initialization failed to generate the requested population size.")

        return individuals

    @staticmethod
    def _cycled_indices(cardinality: int, size: int, rng: random.Random) -> List[int]:
        if cardinality <= 0:
            raise ValueError("cardinality must be positive.")
        order = list(range(cardinality))
        rng.shuffle(order)
        result = []
        for i in range(size):
            result.append(order[i % cardinality])
        return result

    @staticmethod
    def _candidate_product_pool(
        rank_candidates: Sequence[int],
        scaling_candidates: Sequence[float],
        placement_candidates: Sequence[Tuple[int, ...]],
        layer_count: int,
        rng: random.Random,
    ) -> List[Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[int, ...]]]:
        pool: List[Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[int, ...]]] = []
        max_candidates = 256
        # deterministic sampled combinations rather than full Cartesian explosion
        for _ in range(max_candidates):
            rank_vec = tuple(rng.choice(rank_candidates) for _ in range(layer_count))
            scaling_vec = tuple(rng.choice(scaling_candidates) for _ in range(layer_count))
            placement_vec = tuple(rng.choice(placement_candidates))
            pool.append((rank_vec, scaling_vec, placement_vec))
        # deduplicate while preserving order
        unique = []
        seen = set()
        for item in pool:
            key = json.dumps({
                "r": item[0],
                "s": item[1],
                "p": item[2],
            }, sort_keys=True, default=str)
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return unique

    @staticmethod
    def _proposal_to_individual(
        proposal: Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[int, ...]],
        idx: int,
        generation: int,
        config: PopulationConfig,
        strategy: str,
        seen: set,
        *,
        allow_duplicates: bool,
    ) -> DEIndividual:
        ind = PopulationFactory._make_individual(
            rank_encoding=proposal[0],
            scaling_encoding=proposal[1],
            placement_encoding=proposal[2],
            generation=generation,
            metadata={
                "initialization_strategy": strategy,
                "population_index": idx,
                "seed": config.seed,
            },
        )
        if not allow_duplicates and ind.signature() in seen:
            return ind
        return ind

    @staticmethod
    def _proposal_diversity_score(individual: DEIndividual, existing: Sequence[DEIndividual]) -> float:
        if not existing:
            return 1.0
        vec = individual.to_chromosome()
        distances = []
        for other in existing:
            distances.append(_euclidean_distance([float(x) for x in vec], [float(x) for x in other.to_chromosome()]))
        return float(_safe_mean(distances))

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    @classmethod
    def create_seeded_from_source(
        cls,
        *,
        source: Any,
        rank_config: Any,
        scaling_config: Any,
        placement_config: Any,
        population_config: Optional[PopulationConfig] = None,
        layer_names: Optional[Sequence[str]] = None,
        initialization_strategy: str = "random",
        seed: Optional[int] = None,
    ) -> Population:
        if population_config is None:
            population_config = PopulationConfig(seed=seed)
        elif seed is not None:
            population_config = replace(population_config, seed=seed)
        return cls.create_random_population(
            rank_config=rank_config,
            scaling_config=scaling_config,
            placement_config=placement_config,
            population_config=population_config,
            source=source,
            layer_names=layer_names,
            initialization_strategy=initialization_strategy,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PopulationConfig",
    "PopulationStatistics",
    "PopulationSchema",
    "DEIndividual",
    "Population",
    "PopulationFactory",
]
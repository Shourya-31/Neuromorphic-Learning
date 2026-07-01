from __future__ import annotations

from dataclasses import asdict, dataclass, field

from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

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
    from .population import (
        PopulationSchema,
        DEIndividual,
        Population,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from .mutation import (
        MutationEngine,
        MutationConfig,
    )
except Exception:  # pragma: no cover
    MutationEngine = Any  # type: ignore
    MutationConfig = Any  # type: ignore

try:
    from ..lora.rank_config import (
        RankConfig,
        RankSearchSpace,
    )
except Exception:  # pragma: no cover
    RankConfig = Any  # type: ignore
    RankSearchSpace = Any  # type: ignore

try:
    from ..lora.scaling import (
        ScalingConfig,
        ScalingSearchSpace,
    )
except Exception:  # pragma: no cover
    ScalingConfig = Any  # type: ignore
    ScalingSearchSpace = Any  # type: ignore

try:
    from ..lora.placement import (
        PlacementConfig,
        PlacementSearchSpace,
    )
except Exception:  # pragma: no cover
    PlacementConfig = Any  # type: ignore
    PlacementSearchSpace = Any  # type: ignore


# ============================================================================
# VALIDATION HELPERS
# ============================================================================


def _validate_integer(
    name: str,
    value: Any,
) -> None:

    if not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )


def _validate_positive_integer(
    name: str,
    value: int,
) -> None:

    _validate_integer(
        name,
        value,
    )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )


def _validate_non_negative_integer(
    name: str,
    value: int,
) -> None:

    _validate_integer(
        name,
        value,
    )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative."
        )


def _validate_numeric(
    name: str,
    value: Any,
) -> None:

    if not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )


def _validate_finite_numeric(
    name: str,
    value: Any,
) -> None:

    _validate_numeric(
        name,
        value,
    )

    if not math.isfinite(
        float(value)
    ):
        raise ValueError(
            f"{name} must be finite."
        )


def _validate_probability(
    name: str,
    value: float,
) -> None:

    _validate_finite_numeric(
        name,
        value,
    )

    if not (
        0.0 <= float(value) <= 1.0
    ):
        raise ValueError(
            f"{name} must satisfy 0 <= value <= 1."
        )


def _validate_bool(
    name: str,
    value: Any,
) -> None:

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be a boolean."
        )


def _safe_mean(
    values: Sequence[float],
    default: float = 0.0,
) -> float:

    if not values:
        return float(default)

    return float(
        sum(values) / len(values)
    )


def _safe_median(
    values: Sequence[float],
    default: float = 0.0,
) -> float:

    if not values:
        return float(default)

    return float(
        statistics.median(values)
    )


def _safe_variance(
    values: Sequence[float],
    default: float = 0.0,
) -> float:

    if len(values) <= 1:
        return float(default)

    return float(
        statistics.pvariance(values)
    )


def _safe_std(
    values: Sequence[float],
    default: float = 0.0,
) -> float:

    if len(values) <= 1:
        return float(default)

    return float(
        statistics.pstdev(values)
    )


def _stable_hash(
    obj: Any,
) -> str:

    payload = json.dumps(
        obj,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _safe_getattr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    try:
        return getattr(
            obj,
            name,
            default,
        )
    except Exception:
        return default


def _hamming_distance(
    a: Sequence[Any],
    b: Sequence[Any],
) -> float:

    if len(a) != len(b):
        raise ValueError(
            "Vectors must have equal length."
        )

    if len(a) == 0:
        return 0.0

    diff = 0

    for x, y in zip(a, b):
        diff += 0 if x == y else 1

    return float(
        diff / len(a)
    )


def _euclidean_distance(
    a: Sequence[float],
    b: Sequence[float],
) -> float:

    if len(a) != len(b):
        raise ValueError(
            "Vectors must have equal length."
        )

    if len(a) == 0:
        return 0.0

    return float(
        math.sqrt(
            sum(
                (
                    float(x)
                    -
                    float(y)
                ) ** 2
                for x, y in zip(a, b)
            )
        )
    )


# ============================================================================
# CROSSOVER STRATEGY
# ============================================================================


class CrossoverStrategy(
    str,
    Enum,
):

    BINOMIAL = "binomial"

    EXPONENTIAL = "exponential"

    SEGMENT = "segment"

    DIVERSITY_GUIDED = (
        "diversity_guided"
    )

    CONSTRAINT_PRESERVING = (
        "constraint_preserving"
    )


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class CrossoverConfig:

    crossover_rate: float = 0.90

    minimum_crossover_rate: float = 0.05

    maximum_crossover_rate: float = 1.00

    enable_adaptive_crossover: bool = True

    rank_crossover_probability: float = 1.00

    scaling_crossover_probability: float = 1.00

    placement_crossover_probability: float = 1.00

    enable_diversity_guided_crossover: bool = True

    enable_constraint_preservation: bool = True

    enable_statistics_tracking: bool = True

    enable_metadata_tracking: bool = True

    seed: Optional[int] = None

    def __post_init__(
        self,
    ) -> None:

        _validate_probability(
            "crossover_rate",
            self.crossover_rate,
        )

        _validate_probability(
            "minimum_crossover_rate",
            self.minimum_crossover_rate,
        )

        _validate_probability(
            "maximum_crossover_rate",
            self.maximum_crossover_rate,
        )

        if (
            self.minimum_crossover_rate
            >
            self.maximum_crossover_rate
        ):
            raise ValueError(
                "minimum_crossover_rate "
                "must be <= "
                "maximum_crossover_rate."
            )

        if not (
            self.minimum_crossover_rate
            <=
            self.crossover_rate
            <=
            self.maximum_crossover_rate
        ):
            raise ValueError(
                "crossover_rate "
                "outside legal bounds."
            )

        _validate_probability(
            "rank_crossover_probability",
            self.rank_crossover_probability,
        )

        _validate_probability(
            "scaling_crossover_probability",
            self.scaling_crossover_probability,
        )

        _validate_probability(
            "placement_crossover_probability",
            self.placement_crossover_probability,
        )

        _validate_bool(
            "enable_adaptive_crossover",
            self.enable_adaptive_crossover,
        )

        _validate_bool(
            "enable_diversity_guided_crossover",
            self.enable_diversity_guided_crossover,
        )

        _validate_bool(
            "enable_constraint_preservation",
            self.enable_constraint_preservation,
        )

        _validate_bool(
            "enable_statistics_tracking",
            self.enable_statistics_tracking,
        )

        _validate_bool(
            "enable_metadata_tracking",
            self.enable_metadata_tracking,
        )


# ============================================================================
# STATISTICS
# ============================================================================


@dataclass
class CrossoverStatistics:

    binomial_uses: int = 0

    exponential_uses: int = 0

    segment_uses: int = 0

    diversity_guided_uses: int = 0

    constraint_preserving_uses: int = 0

    crossover_calls: int = 0

    trials_generated: int = 0

    rank_crossovers: int = 0

    scaling_crossovers: int = 0

    placement_crossovers: int = 0

    rejected_trials: int = 0

    constraint_repairs: int = 0

    adaptive_crossovers: int = 0

    diversity_guided_crossovers: int = 0

    metadata_exports: int = 0

    successful_trials: int = 0

    failed_trials: int = 0

    cumulative_improvement: float = 0.0

    best_trial_improvement: float = 0.0

    worst_trial_improvement: float = 0.0

    adaptive_rate_updates: int = 0

    minimum_adaptive_rate: float = float("inf")

    maximum_adaptive_rate: float = 0.0

    average_adaptive_rate: float = 0.0

    cumulative_adaptive_rate: float = 0.0

    recent_population_diversity: List[float] = field(default_factory=list)

    recent_trial_improvements: List[float] = field(default_factory=list)

    recent_success_flags: List[int] = field(default_factory=list)

    history_limit: int = 64

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ============================================================================
# SEGMENT INFORMATION
# ============================================================================


@dataclass(frozen=True)
class ChromosomeSegments:

    rank_start: int
    rank_end: int

    scaling_start: int
    scaling_end: int

    placement_start: int
    placement_end: int

    chromosome_length: int

    @property
    def rank_slice(
        self,
    ) -> slice:

        return slice(
            self.rank_start,
            self.rank_end,
        )

    @property
    def scaling_slice(
        self,
    ) -> slice:

        return slice(
            self.scaling_start,
            self.scaling_end,
        )

    @property
    def placement_slice(
        self,
    ) -> slice:

        return slice(
            self.placement_start,
            self.placement_end,
        )
    
    # ============================================================================
# SCHEMA INSPECTION
# ============================================================================


class ChromosomeSchemaInspector:
    """
    Extracts chromosome segmentation
    information from PopulationSchema.

    Layout:

        [rank segment]
        +
        [scaling segment]
        +
        [placement segment]
    """

    def __init__(
        self,
        schema: PopulationSchema,
    ) -> None:

        if schema is None:
            raise ValueError(
                "schema cannot be None."
            )

        self.schema = schema

        self._segments = (
            self._infer_segments()
        )

    @property
    def segments(
        self,
    ) -> ChromosomeSegments:

        return self._segments

    def _infer_segments(
        self,
    ) -> ChromosomeSegments:

        segment_metadata = getattr(
            self.schema,
            "segment_metadata",
            None,
        )

        if segment_metadata is not None:

            required = {
                "rank",
                "scaling",
                "placement",
            }

            available = {
                str(k)
                for k in segment_metadata.keys()
            }

            missing = (
                required
                -
                available
            )

            if missing:
                raise ValueError(
                    f"Missing segment definitions: "
                    f"{sorted(missing)}"
                )

            rank_meta = segment_metadata[
                "rank"
            ]

            scaling_meta = segment_metadata[
                "scaling"
            ]

            placement_meta = segment_metadata[
                "placement"
            ]

            chromosome_length = int(
                getattr(
                    self.schema,
                    "chromosome_length",
                )
            )

            return ChromosomeSegments(
                rank_start=int(
                    rank_meta["start"]
                ),
                rank_end=int(
                    rank_meta["end"]
                ),

                scaling_start=int(
                    scaling_meta["start"]
                ),
                scaling_end=int(
                    scaling_meta["end"]
                ),

                placement_start=int(
                    placement_meta["start"]
                ),
                placement_end=int(
                    placement_meta["end"]
                ),

                chromosome_length=
                chromosome_length,
            )

        rank_slice = getattr(
            self.schema,
            "rank_slice",
            None,
        )

        scaling_slice = getattr(
            self.schema,
            "scaling_slice",
            None,
        )

        placement_slice = getattr(
            self.schema,
            "placement_slice",
            None,
        )

        if (
            rank_slice is not None
            and
            scaling_slice is not None
            and
            placement_slice is not None
        ):

            chromosome_length = int(
                getattr(
                    self.schema,
                    "chromosome_length",
                )
            )

            return ChromosomeSegments(
                rank_start=int(
                    rank_slice.start
                ),
                rank_end=int(
                    rank_slice.stop
                ),

                scaling_start=int(
                    scaling_slice.start
                ),
                scaling_end=int(
                    scaling_slice.stop
                ),

                placement_start=int(
                    placement_slice.start
                ),
                placement_end=int(
                    placement_slice.stop
                ),

                chromosome_length=
                chromosome_length,
            )

        raise RuntimeError(
            "Unable to infer chromosome "
            "segmentation from schema. "
            "PopulationSchema must expose "
            "segment_metadata or segment "
            "slices."
        )


# ============================================================================
# SEARCH SPACE REFLECTION
# ============================================================================


class SearchSpaceReflection:

    def __init__(
        self,
        schema: PopulationSchema,
    ) -> None:

        self.schema = schema

    def _discover_rank_space(
        self,
    ):

        candidates = []

        for attr in (
            "rank_search_space",
            "rank_space",
            "rank_config",
        ):

            obj = getattr(
                self.schema,
                attr,
                None,
            )

            if obj is None:
                continue

            for candidate_attr in (
                "allowed_ranks",
                "candidate_ranks",
                "ranks",
            ):

                values = getattr(
                    obj,
                    candidate_attr,
                    None,
                )

                if values is not None:

                    candidates.extend(
                        int(v)
                        for v in values
                    )

        return tuple(
            sorted(
                set(candidates)
            )
        )

    def _discover_scaling_space(
        self,
    ):

        candidates = []

        for attr in (
            "scaling_search_space",
            "scaling_space",
            "scaling_config",
        ):

            obj = getattr(
                self.schema,
                attr,
                None,
            )

            if obj is None:
                continue

            for candidate_attr in (
                "allowed_alphas",
                "candidate_alphas",
                "alphas",
            ):

                values = getattr(
                    obj,
                    candidate_attr,
                    None,
                )

                if values is not None:

                    candidates.extend(
                        float(v)
                        for v in values
                    )

        return tuple(
            sorted(
                set(candidates)
            )
        )

    def _discover_placement_space(
        self,
    ):

        for attr in (
            "placement_search_space",
            "placement_space",
            "placement_config",
        ):

            obj = getattr(
                self.schema,
                attr,
                None,
            )

            if obj is None:
                continue

            for candidate_attr in (
                "candidate_masks",
                "placement_masks",
                "allowed_masks",
            ):

                values = getattr(
                    obj,
                    candidate_attr,
                    None,
                )

                if values is not None:

                    return tuple(
                        tuple(
                            int(v)
                            for v in mask
                        )
                        for mask in values
                    )

        return tuple()

    @property
    def rank_candidates(
        self,
    ):

        discovered = (
            self._discover_rank_space()
        )

        if discovered:
            return discovered

        values = getattr(
            self.schema,
            "rank_candidates",
            None,
        )

        if values is None:
            raise RuntimeError(
                "Unable to discover "
                "rank search space."
            )

        return tuple(
            int(v)
            for v in values
        )

    @property
    def scaling_candidates(
        self,
    ):

        discovered = (
            self._discover_scaling_space()
        )

        if discovered:
            return discovered

        values = getattr(
            self.schema,
            "scaling_candidates",
            None,
        )

        if values is None:
            raise RuntimeError(
                "Unable to discover "
                "scaling search space."
            )

        return tuple(
            float(v)
            for v in values
        )

    @property
    def placement_candidates(
        self,
    ):

        discovered = (
            self._discover_placement_space()
        )

        if discovered:
            return discovered

        values = getattr(
            self.schema,
            "placement_candidates",
            None,
        )

        if values is None:
            raise RuntimeError(
                "Unable to discover "
                "placement search space."
            )

        return tuple(
            tuple(
                int(v)
                for v in mask
            )
            for mask in values
        )

    @property
    def minimum_layers(
        self,
    ):

        for attr in (
            "placement_search_space",
            "placement_space",
            "placement_config",
        ):

            obj = getattr(
                self.schema,
                attr,
                None,
            )

            if (
                obj is not None
                and
                hasattr(
                    obj,
                    "minimum_layers",
                )
            ):
                return int(
                    obj.minimum_layers
                )

        return int(
            getattr(
                self.schema,
                "minimum_layers",
                1,
            )
        )

    @property
    def maximum_layers(
        self,
    ):

        for attr in (
            "placement_search_space",
            "placement_space",
            "placement_config",
        ):

            obj = getattr(
                self.schema,
                attr,
                None,
            )

            if (
                obj is not None
                and
                hasattr(
                    obj,
                    "maximum_layers",
                )
            ):
                return int(
                    obj.maximum_layers
                )

        return getattr(
            self.schema,
            "maximum_layers",
            None,
        )

    @property
    def allow_empty_placement(
        self,
    ):

        for attr in (
            "placement_search_space",
            "placement_space",
            "placement_config",
        ):

            obj = getattr(
                self.schema,
                attr,
                None,
            )

            if (
                obj is not None
                and
                hasattr(
                    obj,
                    "allow_empty_placement",
                )
            ):
                return bool(
                    obj.allow_empty_placement
                )

        return bool(
            getattr(
                self.schema,
                "allow_empty_placement",
                False,
            )
        )


# ============================================================================
# CHROMOSOME UTILITIES
# ============================================================================


class ChromosomeUtilities:

    @staticmethod
    def split(
        chromosome: Sequence[
            Union[int, float]
        ],
        segments: ChromosomeSegments,
    ) -> Tuple[
        List[int],
        List[float],
        List[int],
    ]:

        if (
            len(chromosome)
            !=
            segments.chromosome_length
        ):
            raise ValueError(
                "Chromosome length mismatch."
            )

        rank_segment = [
            int(v)
            for v in chromosome[
                segments.rank_slice
            ]
        ]

        scaling_segment = [
            float(v)
            for v in chromosome[
                segments.scaling_slice
            ]
        ]

        placement_segment = [
            int(v)
            for v in chromosome[
                segments.placement_slice
            ]
        ]

        return (
            rank_segment,
            scaling_segment,
            placement_segment,
        )

    @staticmethod
    def merge(
        rank_segment: Sequence[int],
        scaling_segment: Sequence[
            float
        ],
        placement_segment: Sequence[
            int
        ],
    ) -> List[
        Union[int, float]
    ]:

        return (
            list(rank_segment)
            +
            list(scaling_segment)
            +
            list(placement_segment)
        )

    @staticmethod
    def chromosome_distance(
        a: Sequence[
            Union[int, float]
        ],
        b: Sequence[
            Union[int, float]
        ],
    ) -> float:

        return _euclidean_distance(
            [
                float(v)
                for v in a
            ],
            [
                float(v)
                for v in b
            ],
        )

    @staticmethod
    def diversity(
        chromosomes: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
    ) -> float:

        if (
            len(chromosomes)
            <= 1
        ):
            return 0.0

        distances = []

        for i in range(
            len(chromosomes)
        ):
            for j in range(
                i + 1,
                len(chromosomes),
            ):
                distances.append(
                    ChromosomeUtilities
                    .chromosome_distance(
                        chromosomes[i],
                        chromosomes[j],
                    )
                )

        return _safe_mean(
            distances
        )


# ============================================================================
# CROSSOVER ANALYTICS
# ============================================================================


@dataclass
class CrossoverAnalytics:

    diversity_gain: float = 0.0

    crossover_magnitude: float = 0.0

    exploration_score: float = 0.0

    exploitation_score: float = 0.0

    crossover_efficiency: float = 0.0

    crossover_health: float = 0.0

    crossover_stability: float = 0.0

    crossover_readiness: float = 0.0

    adaptive_score: float = 0.0

    search_burden: float = 0.0

    effective_dimensions: int = 0

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ============================================================================
# CROSSOVER BASE ENGINE
# ============================================================================


class CrossoverEngine:

    """
    Main DE crossover engine.

    Responsibilities:

    - crossover()
    - crossover_population()
    - generate_trial()

    - validation
    - repair

    - analytics
    - metadata
    """

    def __init__(
        self,
        schema: PopulationSchema,
        config: Optional[
            CrossoverConfig
        ] = None,
    ) -> None:

        if schema is None:
            raise ValueError(
                "schema cannot be None."
            )

        self.schema = schema

        self.config = (
            config
            or
            CrossoverConfig()
        )

        self.statistics_tracker = (
            CrossoverStatistics()
        )

        self.random_state = (
            random.Random(
                self.config.seed
            )
        )

        self.inspector = (
            ChromosomeSchemaInspector(
                schema
            )
        )

        self.search_space = (
            SearchSpaceReflection(
                schema
            )
        )

    def _individual_to_chromosome(
        self,
        individual: Any,
    ) -> List[Union[int, float]]:

        for attr in (
            "to_chromosome",
            "chromosome",
            "genome",
            "vector",
            "genes",
        ):

            value = getattr(
                individual,
                attr,
                None,
            )

            if callable(value):
                value = value()

            if value is not None:
                try:
                    return list(value)
                except Exception:
                    pass

        if isinstance(
            individual,
            Sequence,
        ) and not isinstance(
            individual,
            (str, bytes),
        ):
            return list(individual)

        raise TypeError(
            "Unable to extract chromosome "
            "from individual."
        )


    def _build_individual(
        self,
        chromosome: Sequence[Union[int, float]],
        *,
        template: Any = None,
        generation: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:

        candidates = []

        if template is not None:
            candidates.append(type(template))
        candidates.append(DEIndividual)

        for cls in candidates:
            if cls is None:
                continue

            for ctor_name in (
                "from_chromosome",
                "from_vector",
                "from_genome",
            ):
                ctor = getattr(cls, ctor_name, None)
                if callable(ctor):
                    try:
                        return ctor(
                            chromosome=tuple(chromosome),
                            schema=self.schema,
                        )
                    except TypeError:
                        try:
                            return ctor(tuple(chromosome))
                        except Exception:
                            pass
                    except Exception:
                        pass

            try:
                if generation is not None or metadata is not None:
                    kwargs = {
                        "chromosome": tuple(chromosome),
                        "schema": self.schema,
                    }
                    if generation is not None:
                        kwargs["generation"] = int(generation)
                    if metadata is not None:
                        kwargs["metadata"] = dict(metadata)
                    return cls(**kwargs)
            except Exception:
                pass

            try:
                return cls(
                    tuple(chromosome),
                )
            except Exception:
                pass

        raise TypeError(
            "Unable to construct DEIndividual "
            "from chromosome."
        )


    def _population_individuals(
        self,
        population: Any,
    ) -> List[Any]:

        if population is None:
            return []

        if hasattr(population, "individuals"):
            try:
                return list(population.individuals)
            except Exception:
                pass

        if isinstance(
            population,
            Sequence,
        ) and not isinstance(
            population,
            (str, bytes),
        ):
            return list(population)

        try:
            return list(population)
        except Exception:
            pass

        raise TypeError(
            "Unable to iterate over population."
        )


    def _population_analytics(
        self,
        population: Any,
    ) -> Dict[str, float]:

        metrics = {
            "diversity": 0.5,
            "health": 0.5,
            "risk": 0.5,
            "uniqueness": 0.5,
            "difficulty": 0.5,
        }

        if population is None:
            return metrics

        method_map = {
            "diversity": (
                "population_diversity",
                "diversity",
            ),
            "health": (
                "population_health_score",
                "health_score",
            ),
            "risk": (
                "premature_convergence_risk",
                "convergence_risk",
            ),
            "uniqueness": (
                "uniqueness_ratio",
                "unique_ratio",
            ),
            "difficulty": (
                "optimization_difficulty_score",
                "difficulty_score",
                "search_space_complexity",
            ),
        }

        for key, names in method_map.items():
            for name in names:
                value = getattr(population, name, None)
                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue

                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metrics[key] = float(value)
                    break

        return {
            key: max(0.0, min(1.0, float(value)))
            for key, value in metrics.items()
        }

    @property
    def segments(
        self,
    ) -> ChromosomeSegments:

        return (
            self.inspector.segments
        )

    @property
    def chromosome_length(
        self,
    ) -> int:

        return (
            self.segments
            .chromosome_length
        )

    def _clip_crossover_rate(
        self,
        value: float,
    ) -> float:

        return max(
            self.config
            .minimum_crossover_rate,
            min(
                self.config
                .maximum_crossover_rate,
                float(value),
            ),
        )

    def adaptive_crossover_rate(
        self,
        population: Optional[Population] = None,
    ) -> float:

        if not self.config.enable_adaptive_crossover:
            return self.config.crossover_rate

        base_rate = float(self.config.crossover_rate)
        adapted_rate = base_rate

        analytics = self._population_analytics(population)

        success_rate = (
            self.statistics_tracker.successful_trials
            /
            max(
                1,
                self.statistics_tracker.successful_trials
                + self.statistics_tracker.failed_trials,
            )
        )

        average_improvement = (
            self.statistics_tracker.cumulative_improvement
            /
            max(
                1,
                self.statistics_tracker.successful_trials
                + self.statistics_tracker.failed_trials,
            )
        )

        repair_frequency = (
            self.statistics_tracker.constraint_repairs
            /
            max(
                1,
                self.statistics_tracker.trials_generated,
            )
        )

        population_component = (
            analytics["risk"] * 0.35
            +
            (1.0 - analytics["diversity"]) * 0.35
            +
            (1.0 - analytics["health"]) * 0.15
            +
            analytics["difficulty"] * 0.15
        )

        performance_component = (
            (1.0 - success_rate) * 0.45
            -
            max(0.0, average_improvement) * 0.20
            +
            repair_frequency * 0.35
        )

        adapted_rate *= 1.0 + population_component + performance_component
        adapted_rate = self._clip_crossover_rate(adapted_rate)

        if abs(adapted_rate - base_rate) > 1e-9:
            self.statistics_tracker.adaptive_crossovers += 1

        self.statistics_tracker.adaptive_rate_updates += 1
        self.statistics_tracker.minimum_adaptive_rate = min(
            self.statistics_tracker.minimum_adaptive_rate,
            adapted_rate,
        )
        self.statistics_tracker.maximum_adaptive_rate = max(
            self.statistics_tracker.maximum_adaptive_rate,
            adapted_rate,
        )
        self.statistics_tracker.cumulative_adaptive_rate += adapted_rate
        self.statistics_tracker.average_adaptive_rate = (
            self.statistics_tracker.cumulative_adaptive_rate
            /
            max(
                1,
                self.statistics_tracker.adaptive_rate_updates,
            )
        )

        return adapted_rate

# NEXT SECTION:
# - binomial_crossover()
# - exponential_crossover()
# - segment crossover
# - diversity guided crossover
# - repair system
# - validation system
# - metadata/export/reporting

# ============================================================================
# BINOMIAL CROSSOVER
# ============================================================================

    def binomial_crossover(
        self,
        target: Sequence[
            Union[int, float]
        ],
        mutant: Sequence[
            Union[int, float]
        ],
        crossover_rate: float,
    ) -> List[
        Union[int, float]
    ]:

        if len(target) != len(mutant):
            raise ValueError(
                "Target and mutant lengths differ."
            )

        n = len(target)

        if n == 0:
            return []

        j_rand = (
            self.random_state
            .randrange(n)
        )

        trial = []

        for j in range(n):

            if (
                self.random_state.random()
                <
                crossover_rate
            ) or (
                j == j_rand
            ):
                trial.append(
                    mutant[j]
                )
            else:
                trial.append(
                    target[j]
                )

        return trial


# ============================================================================
# EXPONENTIAL CROSSOVER
# ============================================================================

    def exponential_crossover(
        self,
        target: Sequence[
            Union[int, float]
        ],
        mutant: Sequence[
            Union[int, float]
        ],
        crossover_rate: float,
    ) -> List[
        Union[int, float]
    ]:

        if len(target) != len(mutant):
            raise ValueError(
                "Target and mutant lengths differ."
            )

        n = len(target)

        if n == 0:
            return []

        start = (
            self.random_state
            .randrange(n)
        )

        trial = list(target)

        index = start

        inherited = 0

        while (
            inherited < n
        ):

            trial[index] = (
                mutant[index]
            )

            inherited += 1

            if (
                self.random_state.random()
                >= crossover_rate
            ):
                break

            index = (
                index + 1
            ) % n

        return trial


# ============================================================================
# SEGMENT CROSSOVER
# ============================================================================

    def segment_crossover(
        self,
        target: Sequence[
            Union[int, float]
        ],
        mutant: Sequence[
            Union[int, float]
        ],
    ) -> List[
        Union[int, float]
    ]:

        (
            target_rank,
            target_scaling,
            target_placement,
        ) = ChromosomeUtilities.split(
            target,
            self.segments,
        )

        (
            mutant_rank,
            mutant_scaling,
            mutant_placement,
        ) = ChromosomeUtilities.split(
            mutant,
            self.segments,
        )

        rank_trial = (
            target_rank.copy()
        )

        scaling_trial = (
            target_scaling.copy()
        )

        placement_trial = (
            target_placement.copy()
        )

        if (
            self.random_state.random()
            <
            self.config
            .rank_crossover_probability
        ):

            rank_trial = (
                mutant_rank.copy()
            )

            self.statistics_tracker \
                .rank_crossovers += 1

        if (
            self.random_state.random()
            <
            self.config
            .scaling_crossover_probability
        ):

            scaling_trial = (
                mutant_scaling.copy()
            )

            self.statistics_tracker \
                .scaling_crossovers += 1

        if (
            self.random_state.random()
            <
            self.config
            .placement_crossover_probability
        ):

            placement_trial = (
                mutant_placement.copy()
            )

            self.statistics_tracker \
                .placement_crossovers += 1

        return (
            ChromosomeUtilities
            .merge(
                rank_trial,
                scaling_trial,
                placement_trial,
            )
        )


# ============================================================================
# DIVERSITY GUIDED CROSSOVER
# ============================================================================

    def diversity_guided_crossover(
        self,
        target: Sequence[Union[int, float]],
        mutant: Sequence[Union[int, float]],
        population: Population,
    ) -> List[Union[int, float]]:

        self.statistics_tracker.diversity_guided_crossovers += 1

        analytics = self._population_analytics(population)
        effective_rate = self.adaptive_crossover_rate(population)

        exploration_pressure = (
            (1.0 - analytics["diversity"]) * 0.30
            +
            analytics["risk"] * 0.25
            +
            (1.0 - analytics["uniqueness"]) * 0.20
            +
            analytics["difficulty"] * 0.15
            +
            (1.0 - analytics["health"]) * 0.10
        )

        exploitation_pressure = (
            analytics["health"] * 0.40
            +
            analytics["diversity"] * 0.30
            +
            analytics["uniqueness"] * 0.20
            +
            (1.0 - analytics["risk"]) * 0.10
        )

        adjusted_rate = self._clip_crossover_rate(
            effective_rate
            +
            (exploration_pressure - exploitation_pressure) * 0.25
        )

        if exploration_pressure >= exploitation_pressure + 0.15:
            return self.exponential_crossover(
                target,
                mutant,
                adjusted_rate,
            )

        if exploitation_pressure >= exploration_pressure + 0.15:
            return self.binomial_crossover(
                target,
                mutant,
                adjusted_rate,
            )

        bin_trial = self.binomial_crossover(
            target,
            mutant,
            adjusted_rate,
        )
        exp_trial = self.exponential_crossover(
            target,
            mutant,
            adjusted_rate,
        )

        bin_score = self.crossover_magnitude(target, bin_trial)
        exp_score = self.crossover_magnitude(target, exp_trial)

        return bin_trial if bin_score >= exp_score else exp_trial


# ============================================================================
# SEGMENT VALIDATION
# ============================================================================

    def validate_rank_segment(
        self,
        segment: Sequence[int],
    ) -> bool:

        legal = set(
            self.search_space
            .rank_candidates
        )

        for value in segment:

            if int(value) not in legal:
                raise ValueError(
                    f"Illegal rank value: "
                    f"{value}"
                )

        return True

    def validate_scaling_segment(
        self,
        segment: Sequence[
            float
        ],
    ) -> bool:

        legal = set(
            self.search_space
            .scaling_candidates
        )

        for value in segment:

            if float(value) not in legal:
                raise ValueError(
                    f"Illegal scaling value: "
                    f"{value}"
                )

        return True

    def validate_placement_segment(
        self,
        segment: Sequence[int],
    ) -> bool:

        mask = tuple(
            int(v)
            for v in segment
        )

        legal_masks = set(
            self.search_space
            .placement_candidates
        )

        if mask not in legal_masks:
            raise ValueError(
                "Illegal placement mask."
            )

        active_layers = sum(mask)

        if (
            active_layers
            <
            self.search_space
            .minimum_layers
        ):
            raise ValueError(
                "Placement violates "
                "minimum_layers."
            )

        maximum = (
            self.search_space
            .maximum_layers
        )

        if (
            maximum is not None
            and
            active_layers > maximum
        ):
            raise ValueError(
                "Placement violates "
                "maximum_layers."
            )

        return True


# ============================================================================
# TRIAL VALIDATION
# ============================================================================

    def validate_trial(
        self,
        chromosome: Sequence[
            Union[int, float]
        ],
    ) -> bool:

        if (
            len(chromosome)
            !=
            self.chromosome_length
        ):
            raise ValueError(
                "Chromosome length mismatch."
            )

        (
            rank_segment,
            scaling_segment,
            placement_segment,
        ) = ChromosomeUtilities.split(
            chromosome,
            self.segments,
        )

        self.validate_rank_segment(
            rank_segment
        )

        self.validate_scaling_segment(
            scaling_segment
        )

        self.validate_placement_segment(
            placement_segment
        )

        return True


# ============================================================================
# REPAIR SYSTEM
# ============================================================================

    def repair_rank_segment(
        self,
        segment: Sequence[int],
    ) -> List[int]:

        legal_ranks = sorted(
            set(
                int(v)
                for v in (
                    self.search_space
                    .rank_candidates
                )
            )
        )

        if not legal_ranks:
            raise RuntimeError(
                "Rank search space is empty."
            )

        original_budget = sum(
            int(v)
            for v in segment
        )

        repaired = []

        for value in segment:

            value = int(value)

            nearest = min(
                legal_ranks,
                key=lambda r: (
                    abs(r - value),
                    r,
                ),
            )

            repaired.append(
                int(nearest)
            )

        repaired_budget = sum(
            repaired
        )

        budget_error = (
            original_budget
            -
            repaired_budget
        )

        if budget_error == 0:
            return repaired

        tolerance = max(
            legal_ranks
        )

        iterations = 0
        max_iterations = (
            len(repaired) * 10
        )

        while (
            abs(budget_error)
            > tolerance
            and
            iterations
            <
            max_iterations
        ):

            iterations += 1

            if budget_error > 0:

                candidates = []

                for idx, rank in enumerate(
                    repaired
                ):

                    larger = [
                        r
                        for r in legal_ranks
                        if r > rank
                    ]

                    if not larger:
                        continue

                    next_rank = min(
                        larger
                    )

                    delta = (
                        next_rank
                        -
                        rank
                    )

                    candidates.append(
                        (
                            abs(
                                budget_error
                                -
                                delta
                            ),
                            idx,
                            next_rank,
                            delta,
                        )
                    )

                if not candidates:
                    break

                (
                    _,
                    idx,
                    new_rank,
                    delta,
                ) = min(
                    candidates
                )

                repaired[idx] = (
                    new_rank
                )

                budget_error -= (
                    delta
                )

            else:

                candidates = []

                for idx, rank in enumerate(
                    repaired
                ):

                    smaller = [
                        r
                        for r in legal_ranks
                        if r < rank
                    ]

                    if not smaller:
                        continue

                    next_rank = max(
                        smaller
                    )

                    delta = (
                        rank
                        -
                        next_rank
                    )

                    candidates.append(
                        (
                            abs(
                                budget_error
                                +
                                delta
                            ),
                            idx,
                            next_rank,
                            delta,
                        )
                    )

                if not candidates:
                    break

                (
                    _,
                    idx,
                    new_rank,
                    delta,
                ) = min(
                    candidates
                )

                repaired[idx] = (
                    new_rank
                )

                budget_error += (
                    delta
                )

        return [
            int(v)
            for v in repaired
        ]

    def repair_scaling_segment(
        self,
        segment: Sequence[
            float
        ],
    ) -> List[float]:

        legal_alphas = sorted(
            set(
                float(v)
                for v in (
                    self.search_space
                    .scaling_candidates
                )
            )
        )

        if not legal_alphas:
            raise RuntimeError(
                "Scaling search space "
                "is empty."
            )

        original_mean = (
            _safe_mean(
                [
                    float(v)
                    for v in segment
                ]
            )
        )

        original_std = (
            _safe_std(
                [
                    float(v)
                    for v in segment
                ]
            )
        )

        repaired = []

        for alpha in segment:

            alpha = float(alpha)

            nearest = min(
                legal_alphas,
                key=lambda x: (
                    abs(
                        x - alpha
                    ),
                    x,
                ),
            )

            repaired.append(
                float(nearest)
            )

        repaired_mean = (
            _safe_mean(
                repaired
            )
        )

        mean_error = (
            original_mean
            -
            repaired_mean
        )

        if abs(mean_error) > 1e-8:

            ordering = sorted(
                range(
                    len(repaired)
                ),
                key=lambda i:
                abs(
                    repaired[i]
                    -
                    original_mean
                ),
                reverse=True,
            )

            for idx in ordering:

                current = (
                    repaired[idx]
                )

                candidates = []

                for candidate in (
                    legal_alphas
                ):

                    trial = (
                        repaired.copy()
                    )

                    trial[idx] = (
                        candidate
                    )

                    trial_mean = (
                        _safe_mean(
                            trial
                        )
                    )

                    candidates.append(
                        (
                            abs(
                                original_mean
                                -
                                trial_mean
                            ),
                            candidate,
                        )
                    )

                if candidates:

                    best = min(
                        candidates
                    )[1]

                    repaired[idx] = (
                        float(best)
                    )

        repaired_std = (
            _safe_std(
                repaired
            )
        )

        if (
            original_std > 0
            and
            repaired_std > 0
        ):

            scale_ratio = (
                repaired_std
                /
                original_std
            )

            if (
                scale_ratio < 0.50
                or
                scale_ratio > 1.50
            ):

                repaired.sort(
                    key=lambda x: abs(
                        x
                        -
                        original_mean
                    )
                )

        return [
            float(v)
            for v in repaired
        ]

# NEXT SECTION:
# - repair_placement_segment()
# - repair_trial()
# - generate_trial()
# - crossover()
# - crossover_population()
# - analytics
# - metadata/export/report/reporting
# - reproducibility hashes
# - experiment tracking

# ============================================================================
# PLACEMENT REPAIR
# ============================================================================

    def repair_placement_segment(
        self,
        segment: Sequence[int],
    ) -> List[int]:

        candidate_masks = (
            self.search_space
            .placement_candidates
        )

        if not candidate_masks:
            raise RuntimeError(
                "No legal placement masks "
                "available."
            )

        proposed_mask = tuple(
            int(v)
            for v in segment
        )

        proposed_density = (
            sum(proposed_mask)
            /
            max(
                1,
                len(proposed_mask),
            )
        )

        proposed_active = sum(
            proposed_mask
        )

        best_mask = None

        best_score = None

        for mask in candidate_masks:

            mask = tuple(
                int(v)
                for v in mask
            )

            mask_active = sum(
                mask
            )

            mask_density = (
                mask_active
                /
                max(
                    1,
                    len(mask),
                )
            )

            hamming_component = (
                _hamming_distance(
                    proposed_mask,
                    mask,
                )
            )

            density_component = abs(
                proposed_density
                -
                mask_density
            )

            activity_component = (
                abs(
                    proposed_active
                    -
                    mask_active
                )
                /
                max(
                    1,
                    len(mask),
                )
            )

            score = (
                0.55
                * hamming_component
                +
                0.25
                * density_component
                +
                0.20
                * activity_component
            )

            if (
                best_score is None
                or
                score < best_score
            ):

                best_score = score

                best_mask = mask

        if best_mask is None:
            raise RuntimeError(
                "Unable to repair "
                "placement segment."
            )

        active_layers = sum(
            best_mask
        )

        minimum_layers = (
            self.search_space
            .minimum_layers
        )

        maximum_layers = (
            self.search_space
            .maximum_layers
        )

        if (
            active_layers
            <
            minimum_layers
        ):
            raise ValueError(
                "Repaired placement "
                "violates minimum_layers."
            )

        if (
            maximum_layers
            is not None
            and
            active_layers
            >
            maximum_layers
        ):
            raise ValueError(
                "Repaired placement "
                "violates maximum_layers."
            )

        return [
            int(v)
            for v in best_mask
        ]

    def repair_trial(
        self,
        chromosome: Sequence[
            Union[int, float]
        ],
    ) -> List[
        Union[int, float]
    ]:

        original = list(
            chromosome
        )

        (
            rank_segment,
            scaling_segment,
            placement_segment,
        ) = ChromosomeUtilities.split(
            chromosome,
            self.segments,
        )

        repaired_rank = (
            self.repair_rank_segment(
                rank_segment
            )
        )

        repaired_scaling = (
            self.repair_scaling_segment(
                scaling_segment
            )
        )

        repaired_placement = (
            self.repair_placement_segment(
                placement_segment
            )
        )

        repaired = (
            ChromosomeUtilities.merge(
                repaired_rank,
                repaired_scaling,
                repaired_placement,
            )
        )

        if repaired != original:

            self.statistics_tracker \
                .constraint_repairs += 1

        self.validate_trial(
            repaired
        )

        return repaired


# ============================================================================
# CONSTRAINT PRESERVING CROSSOVER
# ============================================================================

    def constraint_preserving_crossover(
        self,
        target: Sequence[
            Union[int, float]
        ],
        mutant: Sequence[
            Union[int, float]
        ],
        population: Optional[
            Population
        ] = None,
    ) -> List[
        Union[int, float]
    ]:

        trial = (
            self.binomial_crossover(
                target,
                mutant,
                self.adaptive_crossover_rate(
                    population
                ),
            )
        )

        try:

            self.validate_trial(
                trial
            )

            return trial

        except Exception:

            repaired = (
                self.repair_trial(
                    trial
                )
            )

            self.validate_trial(
                repaired
            )

            return repaired


# ============================================================================
# TRIAL GENERATION
# ============================================================================

    def generate_trial(
        self,
        target: Sequence[Union[int, float]],
        mutant: Sequence[Union[int, float]],
        strategy: CrossoverStrategy = CrossoverStrategy.BINOMIAL,
        population: Optional[Population] = None,
    ) -> List[Union[int, float]]:

        self.statistics_tracker.crossover_calls += 1

        effective_rate = self.adaptive_crossover_rate(population)

        if strategy == CrossoverStrategy.BINOMIAL:
            self.statistics_tracker.binomial_uses += 1
            trial = self.binomial_crossover(target, mutant, effective_rate)

        elif strategy == CrossoverStrategy.EXPONENTIAL:
            self.statistics_tracker.exponential_uses += 1
            trial = self.exponential_crossover(target, mutant, effective_rate)

        elif strategy == CrossoverStrategy.SEGMENT:
            self.statistics_tracker.segment_uses += 1
            trial = self.segment_crossover(target, mutant)

        elif strategy == CrossoverStrategy.DIVERSITY_GUIDED:
            self.statistics_tracker.diversity_guided_uses += 1
            if population is None:
                raise ValueError(
                    "Population required for diversity-guided crossover."
                )
            trial = self.diversity_guided_crossover(
                target,
                mutant,
                population,
            )

        elif strategy == CrossoverStrategy.CONSTRAINT_PRESERVING:
            self.statistics_tracker.constraint_preserving_uses += 1
            trial = self.constraint_preserving_crossover(
                target,
                mutant,
                population,
            )

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        if self.config.enable_constraint_preservation:
            trial = self.repair_trial(trial)

        self.validate_trial(trial)
        self.statistics_tracker.trials_generated += 1
        return trial


# ============================================================================
# INDIVIDUAL CROSSOVER
# ============================================================================

    def crossover(
        self,
        target: DEIndividual,
        mutant: DEIndividual,
        strategy: CrossoverStrategy = CrossoverStrategy.BINOMIAL,
        population: Optional[Population] = None,
    ) -> DEIndividual:

        target_vector = self._individual_to_chromosome(target)
        mutant_vector = self._individual_to_chromosome(mutant)

        trial_vector = self.generate_trial(
            target_vector,
            mutant_vector,
            strategy,
            population,
        )

        return self._build_individual(
            trial_vector,
            template=target,
            generation=getattr(target, "generation", None),
            metadata=getattr(target, "metadata", None),
        )


# ============================================================================
# POPULATION CROSSOVER
# ============================================================================

    def crossover_population(
        self,
        population: Population,
        mutants: Sequence[DEIndividual],
        strategy: CrossoverStrategy = CrossoverStrategy.BINOMIAL,
    ) -> List[DEIndividual]:

        targets = self._population_individuals(population)

        if len(targets) != len(mutants):
            raise ValueError("Population size mismatch.")

        trials = []

        for target, mutant in zip(targets, mutants):
            trials.append(
                self.crossover(
                    target,
                    mutant,
                    strategy,
                    population,
                )
            )

        return trials


# ============================================================================
# CROSSOVER MAGNITUDE
# ============================================================================

    def crossover_magnitude(
        self,
        target: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        if len(target) != len(trial):
            raise ValueError(
                "Length mismatch."
            )

        differences = []

        for (
            a,
            b,
        ) in zip(
            target,
            trial,
        ):

            differences.append(
                abs(
                    float(a)
                    -
                    float(b)
                )
            )

        return _safe_mean(
            differences
        )


# ============================================================================
# DIVERSITY GAIN
# ============================================================================

    def crossover_diversity_gain(
        self,
        original_population: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
        trial_population: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
    ) -> float:

        before = (
            ChromosomeUtilities
            .diversity(
                original_population
            )
        )

        after = (
            ChromosomeUtilities
            .diversity(
                trial_population
            )
        )

        return (
            after - before
        )


# ============================================================================
# SEGMENT ANALYTICS
# ============================================================================

    def segment_crossover_statistics(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        total = max(
            1,
            self.statistics_tracker
            .trials_generated,
        )

        return {
            "rank_crossovers":
            self.statistics_tracker
            .rank_crossovers,

            "scaling_crossovers":
            self.statistics_tracker
            .scaling_crossovers,

            "placement_crossovers":
            self.statistics_tracker
            .placement_crossovers,

            "rank_ratio":
            self.statistics_tracker
            .rank_crossovers
            / total,

            "scaling_ratio":
            self.statistics_tracker
            .scaling_crossovers
            / total,

            "placement_ratio":
            self.statistics_tracker
            .placement_crossovers
            / total,
        }
    
    def inheritance_ratio(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        if len(parent) != len(trial):
            raise ValueError(
                "Length mismatch."
            )

        inherited = 0

        for a, b in zip(
            parent,
            trial,
        ):

            if a != b:
                inherited += 1

        return (
            inherited
            /
            max(
                1,
                len(parent),
            )
        )


    def inheritance_entropy(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        ratio = (
            self.inheritance_ratio(
                parent,
                trial,
            )
        )

        if (
            ratio <= 0.0
            or
            ratio >= 1.0
        ):
            return 0.0

        return -(
            ratio
            *
            math.log2(ratio)
            +
            (
                1.0
                -
                ratio
            )
            *
            math.log2(
                1.0
                -
                ratio
            )
        )


    def segment_disruption_rate(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> Dict[
        str,
        float,
    ]:

        (
            parent_rank,
            parent_scaling,
            parent_placement,
        ) = ChromosomeUtilities.split(
            parent,
            self.segments,
        )

        (
            trial_rank,
            trial_scaling,
            trial_placement,
        ) = ChromosomeUtilities.split(
            trial,
            self.segments,
        )

        def _ratio(
            a,
            b,
        ):

            changed = sum(
                1
                for x, y in zip(a, b)
                if x != y
            )

            return (
                changed
                /
                max(
                    1,
                    len(a),
                )
            )

        return {
            "rank":
            _ratio(
                parent_rank,
                trial_rank,
            ),

            "scaling":
            _ratio(
                parent_scaling,
                trial_scaling,
            ),

            "placement":
            _ratio(
                parent_placement,
                trial_placement,
            ),
        }


    def chromosome_information_gain(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        disruption = (
            self.segment_disruption_rate(
                parent,
                trial,
            )
        )

        return _safe_mean(
            [
                disruption["rank"],
                disruption["scaling"],
                disruption["placement"],
            ]
        )


    def trial_parent_distance(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        return (
            ChromosomeUtilities
            .chromosome_distance(
                parent,
                trial,
            )
        )


    def crossover_disruption_score(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        entropy = (
            self.inheritance_entropy(
                parent,
                trial,
            )
        )

        gain = (
            self.chromosome_information_gain(
                parent,
                trial,
            )
        )

        distance = (
            self.trial_parent_distance(
                parent,
                trial,
            )
        )

        return (
            0.40 * entropy
            +
            0.30 * gain
            +
            0.30 * distance
        )


    def population_exploration_effect(
        self,
        parents: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
        trials: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
    ) -> float:

        if (
            len(parents)
            ==
            0
        ):
            return 0.0

        scores = []

        for parent, trial in zip(
            parents,
            trials,
        ):

            scores.append(
                self.crossover_disruption_score(
                    parent,
                    trial,
                )
            )

        return _safe_mean(
            scores
        )
    
    def information_preservation_score(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        disruption = (
            self.chromosome_information_gain(
                parent,
                trial,
            )
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - disruption,
            ),
        )


    def segment_utilization_score(
        self,
    ) -> Dict[
        str,
        float,
    ]:

        total = max(
            1,
            self.statistics_tracker
            .trials_generated,
        )

        return {
            "rank":
            self.statistics_tracker
            .rank_crossovers
            / total,

            "scaling":
            self.statistics_tracker
            .scaling_crossovers
            / total,

            "placement":
            self.statistics_tracker
            .placement_crossovers
            / total,
        }


    def trial_innovation_score(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        distance = (
            self.trial_parent_distance(
                parent,
                trial,
            )
        )

        entropy = (
            self.inheritance_entropy(
                parent,
                trial,
            )
        )

        return (
            0.5 * distance
            +
            0.5 * entropy
        )


    def inheritance_stability_score(
        self,
        parent: Sequence[
            Union[int, float]
        ],
        trial: Sequence[
            Union[int, float]
        ],
    ) -> float:

        return (
            self.information_preservation_score(
                parent,
                trial,
            )
        )


    def population_diversity_contribution(
        self,
        parent_population: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
        trial_population: Sequence[
            Sequence[
                Union[int, float]
            ]
        ],
    ) -> float:

        return (
            self.crossover_diversity_gain(
                parent_population,
                trial_population,
            )
        )


    def crossover_effectiveness_index(
        self,
    ) -> float:

        efficiency = (
            self.crossover_efficiency_score()
        )

        health = (
            self.crossover_health_score()
        )

        stability = (
            self.crossover_stability_score()
        )

        exploration = (
            self.exploration_score()
        )

        return _safe_mean(
            [
                efficiency,
                health,
                stability,
                exploration,
            ]
        )


    def publication_metrics(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return {
            "effectiveness_index":
            self.crossover_effectiveness_index(),

            "health_score":
            self.crossover_health_score(),

            "stability_score":
            self.crossover_stability_score(),

            "readiness_score":
            self.crossover_readiness_score(),

            "adaptive_score":
            self.adaptive_crossover_score(),

            "exploration_score":
            self.exploration_score(),

            "exploitation_score":
            self.exploitation_score(),

            "diversity_preservation":
            self.diversity_preservation_score(),

            "effective_dimensions":
            self.effective_crossover_dimensions(),

            "search_burden":
            self.crossover_search_burden(),
        }
    
    def benchmark_metadata(
        self,
    ) -> Dict[str, Any]:

        return {
            "chromosome_length":
            self.chromosome_length,

            "effective_dimensions":
            self.effective_crossover_dimensions(),

            "difficulty_score":
            self.crossover_difficulty_score(),

            "search_burden":
            self.crossover_search_burden(),

            "crossover_rate":
            self.config.crossover_rate,
        }


    def ablation_metadata(
        self,
    ) -> Dict[str, Any]:

        return {
            "adaptive_crossover":
            self.config.enable_adaptive_crossover,

            "diversity_guided":
            self.config.enable_diversity_guided_crossover,

            "constraint_preservation":
            self.config.enable_constraint_preservation,

            "rank_probability":
            self.config.rank_crossover_probability,

            "scaling_probability":
            self.config.scaling_crossover_probability,

            "placement_probability":
            self.config.placement_crossover_probability,
        }


    def reproducibility_bundle(
        self,
    ) -> Dict[str, Any]:

        return {
            "config":
            asdict(
                self.config
            ),

            "fingerprint":
            self.crossover_fingerprint(),

            "statistics":
            self.statistics_tracker
            .to_dict(),

            "hash":
            self.crossover_hash(),

            "signature":
            self.crossover_signature(),
        }


    def paper_metrics(
        self,
    ) -> Dict[str, Any]:

        return {
            "health_score":
            self.crossover_health_score(),

            "stability_score":
            self.crossover_stability_score(),

            "readiness_score":
            self.crossover_readiness_score(),

            "effectiveness_index":
            self.crossover_effectiveness_index(),

            "adaptive_learning":
            self.adaptive_learning_score(),

            "exploration_score":
            self.exploration_score(),

            "exploitation_score":
            self.exploitation_score(),

            "diversity_preservation":
            self.diversity_preservation_score(),
        }


    def publication_export(
        self,
    ) -> Dict[str, Any]:

        return {
            "paper_metrics":
            self.paper_metrics(),

            "benchmark":
            self.benchmark_metadata(),

            "ablation":
            self.ablation_metadata(),

            "reproducibility":
            self.reproducibility_bundle(),

            "strategy":
            self.strategy_selection_metadata(),

            "trend":
            self.trend_metadata(),
        }
    
    def crossover_success_tracker(
        self,
        parent_fitness: float,
        trial_fitness: float,
    ) -> Dict[str, float]:

        improvement = (
            trial_fitness
            -
            parent_fitness
        )

        success = (
            1.0
            if improvement > 0.0
            else 0.0
        )

        relative_improvement = (
            improvement
            /
            max(
                abs(parent_fitness),
                1e-12,
            )
        )

        return {
            "success": success,
            "improvement": improvement,
            "relative_improvement":
            relative_improvement,
        }
    

    def register_trial_outcome(
        self,
        parent_fitness: float,
        trial_fitness: float,
    ) -> None:

        improvement = (
            trial_fitness
            -
            parent_fitness
        )

        success = (
            1
            if improvement > 0.0
            else 0
        )

        if success:

            self.statistics_tracker \
                .successful_trials += 1

        else:

            self.statistics_tracker \
                .failed_trials += 1

        self.statistics_tracker \
            .cumulative_improvement += (
                improvement
            )

        self.statistics_tracker \
            .best_trial_improvement = max(
                self.statistics_tracker
                .best_trial_improvement,
                improvement,
            )

        self.statistics_tracker \
            .worst_trial_improvement = min(
                self.statistics_tracker
                .worst_trial_improvement,
                improvement,
            )

        self.statistics_tracker \
            .recent_trial_improvements.append(
                float(improvement)
            )

        self.statistics_tracker \
            .recent_success_flags.append(
                int(success)
            )

        limit = int(
            getattr(
                self.statistics_tracker,
                "history_limit",
                64,
            )
        )

        if limit > 0:

            self.statistics_tracker \
                .recent_trial_improvements = (
                    self.statistics_tracker
                    .recent_trial_improvements[
                        -limit:
                    ]
                )

            self.statistics_tracker \
                .recent_success_flags = (
                    self.statistics_tracker
                    .recent_success_flags[
                        -limit:
                    ]
                )


    def crossover_success_rate(
        self,
        evaluations: Sequence[
            Mapping[str, float]
        ],
    ) -> float:

        if not evaluations:
            return 0.0

        return _safe_mean(
            [
                float(
                    item.get(
                        "success",
                        0.0,
                    )
                )
                for item in evaluations
            ]
        )


    def average_crossover_improvement(
        self,
        evaluations: Sequence[
            Mapping[str, float]
        ],
    ) -> float:

        if not evaluations:
            return 0.0

        return _safe_mean(
            [
                float(
                    item.get(
                        "improvement",
                        0.0,
                    )
                )
                for item in evaluations
            ]
        )


    def adaptive_strategy_recommendation(
        self,
        evaluations: Sequence[
            Mapping[str, float]
        ],
    ) -> str:

        success_rate = (
            self.crossover_success_rate(
                evaluations
            )
        )

        average_gain = (
            self.average_crossover_improvement(
                evaluations
            )
        )

        if (
            success_rate >= 0.70
            and
            average_gain > 0
        ):
            return "exploit"

        if (
            success_rate <= 0.30
        ):
            return "explore"

        return "balanced"


    def optimizer_feedback_metadata(
        self,
        evaluations: Sequence[
            Mapping[str, float]
        ],
    ) -> Dict[str, Any]:

        return {
            "success_rate":
            self.crossover_success_rate(
                evaluations
            ),

            "average_improvement":
            self.average_crossover_improvement(
                evaluations
            ),

            "recommended_policy":
            self.adaptive_strategy_recommendation(
                evaluations
            ),
        }
    
    def _trend_slope(
        self,
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:
            return 0.0

        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = _safe_mean(values)

        numerator = 0.0
        denominator = 0.0

        for idx, value in enumerate(values):
            dx = float(idx) - x_mean
            dy = float(value) - y_mean
            numerator += dx * dy
            denominator += dx * dx

        if denominator == 0.0:
            return 0.0

        return float(numerator / denominator)


    def record_population_snapshot(
        self,
        population: Population,
    ) -> None:

        diversity = 0.0

        try:
            diversity = float(
                population
                .population_diversity()
            )
        except Exception:
            pass

        self.statistics_tracker \
            .recent_population_diversity.append(
                float(diversity)
            )

        limit = int(
            getattr(
                self.statistics_tracker,
                "history_limit",
                64,
            )
        )

        if limit > 0:

            self.statistics_tracker \
                .recent_population_diversity = (
                    self.statistics_tracker
                    .recent_population_diversity[
                        -limit:
                    ]
                )


    def diversity_trend(
        self,
        window: int = 8,
    ) -> float:

        _validate_positive_integer(
            "window",
            window,
        )

        values = (
            self.statistics_tracker
            .recent_population_diversity[
                -window:
            ]
        )

        return self._trend_slope(
            values
        )


    def improvement_trend(
        self,
        window: int = 8,
    ) -> float:

        _validate_positive_integer(
            "window",
            window,
        )

        values = (
            self.statistics_tracker
            .recent_trial_improvements[
                -window:
            ]
        )

        return self._trend_slope(
            values
        )


    def stagnation_score(
        self,
        window: int = 8,
    ) -> float:

        diversity_slope = (
            self.diversity_trend(
                window=window
            )
        )

        improvement_slope = (
            self.improvement_trend(
                window=window
            )
        )

        repair_ratio = (
            self.statistics_tracker
            .constraint_repairs
            /
            max(
                1,
                self.statistics_tracker
                .trials_generated,
            )
        )

        score = (
            0.45 * max(
                0.0,
                -diversity_slope,
            )
            +
            0.45 * max(
                0.0,
                -improvement_slope,
            )
            +
            0.10 * repair_ratio
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )


    def convergence_velocity(
        self,
        window: int = 8,
    ) -> float:

        diversity_slope = (
            self.diversity_trend(
                window=window
            )
        )

        improvement_slope = (
            self.improvement_trend(
                window=window
            )
        )

        return float(
            improvement_slope
            -
            diversity_slope
        )


    def trend_metadata(
        self,
        window: int = 8,
    ) -> Dict[str, Any]:

        return {
            "diversity_trend":
            self.diversity_trend(
                window=window
            ),

            "improvement_trend":
            self.improvement_trend(
                window=window
            ),

            "stagnation_score":
            self.stagnation_score(
                window=window
            ),

            "convergence_velocity":
            self.convergence_velocity(
                window=window
            ),
        }

# NEXT SECTION:
# - crossover_efficiency_score()
# - exploration_score()
# - exploitation_score()
# - health/stability/readiness metrics
# - crossover_hash()
# - metadata()
# - diagnostics()
# - experiment_metadata()
# - export_configuration()
# - __all__

# ============================================================================
# CROSSOVER EFFICIENCY
# ============================================================================

    def crossover_efficiency_score(
        self,
    ) -> float:

        generated = (
            self.statistics_tracker
            .trials_generated
        )

        if generated == 0:
            return 0.0

        successful = (
            generated
            -
            self.statistics_tracker
            .rejected_trials
        )

        repair_penalty = (
            self.statistics_tracker
            .constraint_repairs
            /
            generated
        )

        score = (
            successful
            /
            generated
        ) * (
            1.0
            -
            repair_penalty
            * 0.25
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )


# ============================================================================
# EXPLORATION / EXPLOITATION
# ============================================================================

    def exploration_score(
        self,
    ) -> float:

        segment_stats = (
            self.segment_crossover_statistics()
        )

        exploration = (
            segment_stats[
                "rank_ratio"
            ]
            +
            segment_stats[
                "scaling_ratio"
            ]
            +
            segment_stats[
                "placement_ratio"
            ]
        ) / 3.0

        return max(
            0.0,
            min(
                1.0,
                exploration,
            ),
        )

    def exploitation_score(
        self,
    ) -> float:

        return (
            1.0
            -
            self.exploration_score()
        )


# ============================================================================
# ADVANCED RESEARCH ANALYTICS
# ============================================================================

    def diversity_preservation_score(
        self,
    ) -> float:

        efficiency = (
            self.crossover_efficiency_score()
        )

        repair_penalty = (
            self.statistics_tracker
            .constraint_repairs
            /
            max(
                1,
                self.statistics_tracker
                .trials_generated,
            )
        )

        score = (
            efficiency
            -
            repair_penalty
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    def adaptive_crossover_score(
        self,
    ) -> float:

        if (
            not self.config
            .enable_adaptive_crossover
        ):
            return 0.0

        adaptive_calls = (
            self.statistics_tracker
            .adaptive_crossovers
        )

        generated = max(
            1,
            self.statistics_tracker
            .trials_generated,
        )

        return min(
            1.0,
            adaptive_calls
            /
            generated,
        )
    
    def adaptive_learning_score(
        self,
    ) -> float:

        success_rate = (
            self.statistics_tracker
            .successful_trials
            /
            max(
                1,
                self.statistics_tracker
                .successful_trials
                +
                self.statistics_tracker
                .failed_trials,
            )
        )

        adaptation_activity = (
            self.statistics_tracker
            .adaptive_rate_updates
            /
            max(
                1,
                self.statistics_tracker
                .trials_generated,
            )
        )

        return _safe_mean(
            [
                success_rate,
                adaptation_activity,
            ]
        )

    def crossover_health_score(
        self,
    ) -> float:

        values = [
            self.crossover_efficiency_score(),
            self.diversity_preservation_score(),
            self.adaptive_crossover_score(),
        ]

        return _safe_mean(
            values
        )

    def crossover_stability_score(
        self,
    ) -> float:

        rejected_ratio = (
            self.statistics_tracker
            .rejected_trials
            /
            max(
                1,
                self.statistics_tracker
                .trials_generated,
            )
        )

        repair_ratio = (
            self.statistics_tracker
            .constraint_repairs
            /
            max(
                1,
                self.statistics_tracker
                .trials_generated,
            )
        )

        score = (
            1.0
            -
            (
                rejected_ratio
                +
                repair_ratio
            )
            / 2.0
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    def crossover_readiness_score(
        self,
    ) -> float:

        return _safe_mean(
            [
                self.crossover_health_score(),
                self.crossover_stability_score(),
            ]
        )

    def crossover_difficulty_score(
        self,
    ) -> float:

        dimension_score = (
            self.chromosome_length
        )

        search_cardinality = (
            len(
                self.search_space
                .rank_candidates
            )
            *
            len(
                self.search_space
                .scaling_candidates
            )
            *
            max(
                1,
                len(
                    self.search_space
                    .placement_candidates
                )
            )
        )

        return (
            math.log1p(
                search_cardinality
            )
            *
            dimension_score
        )

    def effective_crossover_dimensions(
        self,
    ) -> int:

        dimensions = 0

        if (
            self.config
            .rank_crossover_probability
            > 0
        ):
            dimensions += (
                self.segments
                .rank_end
                -
                self.segments
                .rank_start
            )

        if (
            self.config
            .scaling_crossover_probability
            > 0
        ):
            dimensions += (
                self.segments
                .scaling_end
                -
                self.segments
                .scaling_start
            )

        if (
            self.config
            .placement_crossover_probability
            > 0
        ):
            dimensions += (
                self.segments
                .placement_end
                -
                self.segments
                .placement_start
            )

        return dimensions

    def crossover_search_burden(
        self,
    ) -> float:

        return (
            self.crossover_difficulty_score()
            *
            self.effective_crossover_dimensions()
        )


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

    def crossover_hash(
        self,
    ) -> str:

        payload = {
            "config":
            asdict(
                self.config
            ),

            "statistics":
            self.statistics_tracker
            .to_dict(),

            "chromosome_length":
            self.chromosome_length,
        }

        return _stable_hash(
            payload
        )

    def crossover_signature(
        self,
    ) -> str:

        return (
            "CROSSOVER::"
            +
            self.crossover_hash()[
                :16
            ]
        )

    def crossover_fingerprint(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return {
            "signature":
            self.crossover_signature(),

            "hash":
            self.crossover_hash(),

            "dimensions":
            self.effective_crossover_dimensions(),
        }


# ============================================================================
# REPORTING
# ============================================================================

    def metadata(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        self.statistics_tracker \
            .metadata_exports += 1

        return {
            "strategy":
            "crossover_engine",

            "crossover_rate":
            self.config
            .crossover_rate,

            "trials_generated":
            self.statistics_tracker
            .trials_generated,

            "repair_count":
            self.statistics_tracker
            .constraint_repairs,

            "publication_export":
            self.publication_export(),

            "diversity_gain":
            self.diversity_preservation_score(),

            "crossover_efficiency":
            self.crossover_efficiency_score(),

            "crossover_hash":
            self.crossover_hash(),
        }

    def diagnostics(
        self,
    ) -> Dict[
        str,
        Any,
    ]:
        

        success_rate = (
            self.statistics_tracker
            .successful_trials
            /
            max(
                1,
                self.statistics_tracker
                .successful_trials
                +
                self.statistics_tracker
                .failed_trials,
            )
        )

        return {
            "health":
            self.crossover_health_score(),

            "stability":
            self.crossover_stability_score(),

            "readiness":
            self.crossover_readiness_score(),

            "adaptive":
            self.adaptive_crossover_score(),

            "benchmark":
            self.benchmark_metadata(),

            "ablation":
            self.ablation_metadata(),

            "publication":
            self.paper_metrics(),

            "difficulty":
            self.crossover_difficulty_score(),

            "success_rate":
            success_rate,

            "cumulative_improvement":
            self.statistics_tracker
            .cumulative_improvement,

            "best_trial_improvement":
            self.statistics_tracker
            .best_trial_improvement,

            "worst_trial_improvement":
            self.statistics_tracker
            .worst_trial_improvement,

            "adaptive_rate_updates":
            self.statistics_tracker
            .adaptive_rate_updates,

            "average_adaptive_rate":
            self.statistics_tracker
            .average_adaptive_rate,

            "minimum_adaptive_rate":
            self.statistics_tracker
            .minimum_adaptive_rate,

            "maximum_adaptive_rate":
            self.statistics_tracker
            .maximum_adaptive_rate,

            "trend_metadata":
            self.trend_metadata(),

            "effectiveness_index":
            self.crossover_effectiveness_index(),
        }

    def crossover_report(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        report = {}

        report.update(
            self.publication_metrics()
        )

        report.update(
            self.metadata()
        )

        report.update(
            self.diagnostics()
        )

        report.update(
            self.segment_crossover_statistics()
        )

        report.update(
            self.publication_export()
        )

        return report

    def crossover_statistics_report(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return (
            self.statistics_tracker
            .to_dict()
        )


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

    def experiment_metadata(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return {
            "configuration":
            asdict(
                self.config
            ),

            "statistics":
            self.statistics_tracker
            .to_dict(),

            "publication_metrics":
            self.publication_metrics(),

            "signature":
            self.crossover_signature(),

            "fingerprint":
            self.crossover_fingerprint(),

            "benchmark_metadata":
            self.benchmark_metadata(),

            "ablation_metadata":
            self.ablation_metadata(),

            "paper_metrics":
            self.paper_metrics(),

            "reproducibility_bundle":
            self.reproducibility_bundle(),
        }

    def experiment_signature(
        self,
    ) -> str:

        payload = (
            self.experiment_metadata()
        )

        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
            ).encode(
                "utf-8"
            )
        ).hexdigest()

    def export_configuration(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        configuration = {
            "config":
            asdict(
                self.config
            ),

            "metadata":
            self.metadata(),

            "diagnostics":
            self.diagnostics(),

            "statistics":
            self.statistics_tracker
            .to_dict(),
        }

        configuration[
            "benchmark_metadata"
        ] = (
            self.benchmark_metadata()
        )

        configuration[
            "ablation_metadata"
        ] = (
            self.ablation_metadata()
        )

        configuration[
            "paper_metrics"
        ] = (
            self.paper_metrics()
        )

        return configuration


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CrossoverStrategy",
    "CrossoverConfig",
    "CrossoverStatistics",
    "ChromosomeSegments",
    "ChromosomeSchemaInspector",
    "SearchSpaceReflection",
    "ChromosomeUtilities",
    "CrossoverAnalytics",
    "CrossoverEngine",
]
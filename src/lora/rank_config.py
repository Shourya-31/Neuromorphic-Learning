from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import statistics

import torch
import torch.nn as nn


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


def _is_power_of_two(
    value: int,
) -> bool:

    if value <= 0:
        return False

    return (
        value &
        (value - 1)
    ) == 0


def _canonical_rank_list(
    ranks: Iterable[int],
) -> List[int]:

    output = []

    for rank in ranks:

        _validate_integer(
            "rank",
            rank,
        )

        output.append(
            int(rank)
        )

    return sorted(
        set(output)
    )


def _estimate_lora_parameters(
    rank: int,
    in_features: int,
    out_features: int,
) -> int:
    """
    LoRA parameter count.

    A:
        [rank, in_features]

    B:
        [out_features, rank]

    total:
        rank * in_features
        +
        out_features * rank
    """

    return int(
        rank *
        (
            in_features
            +
            out_features
        )
    )


def _clamp_rank(
    rank: int,
    minimum: int,
    maximum: int,
) -> int:

    return max(
        minimum,
        min(
            maximum,
            rank,
        ),
    )


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class RankConfig:
    """
    Global rank-management configuration.

    Future compatibility:

    - placement.py
    - scaling.py
    - differential evolution
    - rank sweeps
    - rank ablations
    """

    default_rank: int = 4

    min_rank: int = 1

    max_rank: int = 64

    allowed_ranks: Optional[
        List[int]
    ] = None

    layerwise: bool = False

    proportional: bool = False

    proportional_factor: float = 0.10

    symmetric: bool = True

    enforce_power_of_two: bool = False

    allow_zero_rank: bool = False

    strict: bool = True

    verbose: bool = False

    def __post_init__(
        self,
    ) -> None:

        _validate_integer(
            "default_rank",
            self.default_rank,
        )

        _validate_integer(
            "min_rank",
            self.min_rank,
        )

        _validate_integer(
            "max_rank",
            self.max_rank,
        )

        if self.allow_zero_rank:

            if self.default_rank < 0:
                raise ValueError(
                    "default_rank must be >= 0."
                )

            if self.min_rank < 0:
                raise ValueError(
                    "min_rank must be >= 0."
                )

            if self.max_rank < 0:
                raise ValueError(
                    "max_rank must be >= 0."
                )

        else:

            if self.default_rank <= 0:
                raise ValueError(
                    "default_rank must be positive."
                )

            if self.min_rank <= 0:
                raise ValueError(
                    "min_rank must be positive."
                )

            if self.max_rank <= 0:
                raise ValueError(
                    "max_rank must be positive."
                )

        if self.min_rank > self.max_rank:
            raise ValueError(
                "min_rank must be <= max_rank."
            )

        _validate_numeric(
            "proportional_factor",
            self.proportional_factor,
        )

        if self.proportional_factor <= 0:
            raise ValueError(
                "proportional_factor must be positive."
            )

        if (
            self.layerwise
            and
            self.proportional
        ):
            raise ValueError(
                "layerwise and proportional cannot both be enabled."
            )

        if self.allowed_ranks is not None:

            if not isinstance(
                self.allowed_ranks,
                list,
            ):
                raise TypeError(
                    "allowed_ranks must be a list."
                )

            normalized: List[int] = []

            for rank in self.allowed_ranks:

                _validate_integer(
                    "allowed_rank",
                    rank,
                )

                if rank < 0:
                    raise ValueError(
                        "allowed ranks cannot be negative."
                    )

                if rank == 0:

                    if not self.allow_zero_rank:
                        raise ValueError(
                            "rank 0 requires allow_zero_rank=True."
                        )

                else:

                    if rank < self.min_rank:
                        raise ValueError(
                            "allowed rank below min_rank."
                        )

                    if rank > self.max_rank:
                        raise ValueError(
                            "allowed rank above max_rank."
                        )

                    if (
                        self.enforce_power_of_two
                        and
                        not _is_power_of_two(rank)
                    ):
                        raise ValueError(
                            "allowed rank violates power-of-two constraint."
                        )

                normalized.append(
                    int(rank)
                )

            normalized = (
                _canonical_rank_list(
                    normalized
                )
            )

            if len(normalized) == 0:
                raise ValueError(
                    "allowed_ranks cannot be empty."
                )

            self.allowed_ranks = normalized

            if (
                self.strict
                and
                self.default_rank
                not in self.allowed_ranks
            ):
                raise ValueError(
                    "default_rank must exist in allowed_ranks."
                )

        if (
            self.enforce_power_of_two
            and
            self.default_rank > 0
            and
            not _is_power_of_two(
                self.default_rank
            )
        ):
            raise ValueError(
                "default_rank violates power-of-two constraint."
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ============================================================================
# STATISTICS
# ============================================================================


@dataclass
class RankStatistics:

    layers_seen: int = 0

    layers_assigned: int = 0

    distinct_ranks_used: int = 0

    minimum_rank_observed: int = 0

    maximum_rank_observed: int = 0

    mean_rank: float = 0.0

    median_rank: float = 0.0

    total_rank_budget: int = 0

    estimated_adapted_parameters: int = 0

    estimated_rank_search_candidates: int = 0

    rank_constraint_violations: int = 0

    recommendation_calls: int = 0

    search_space_generation_calls: int = 0

    ablation_calls: int = 0

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ============================================================================
# POLICY TYPES
# ============================================================================


class RankPolicyType(
    str,
    Enum,
):

    FIXED = "fixed"

    LAYERWISE = "layerwise"

    PROPORTIONAL = "proportional"


# ============================================================================
# RECOMMENDATION
# ============================================================================


@dataclass
class RankRecommendation:

    layer_name: str

    rank: int

    reason: str

    estimated_parameter_cost: int

    confidence: float = 1.0

    metadata: Optional[
        Dict[str, Any]
    ] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ============================================================================
# ABLATION REPORT
# ============================================================================


@dataclass
class RankAblationReport:

    tested_ranks: List[int]

    parameter_costs: Dict[
        int,
        int,
    ]

    recommended_rank: int

    minimum_rank: int

    maximum_rank: int

    rank_budget: int

    metadata: Dict[
        str,
        Any,
    ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# ============================================================================
# SEARCH SPACE
# ============================================================================


class RankSearchSpace:
    """
    Generates legal rank candidates
    under repository constraints.

    Compatible with:

    - rank sweeps
    - DE search
    - ablations
    - parameter-efficiency studies
    """

    def __init__(
        self,
        config: RankConfig,
    ) -> None:

        if not isinstance(
            config,
            RankConfig,
        ):
            raise TypeError(
                "config must be RankConfig."
            )

        self.config = config

    def candidate_ranks(
        self,
    ) -> List[int]:

        if self.config.allowed_ranks is not None:
            return list(
                self.config.allowed_ranks
            )

        candidates: List[int] = []

        if self.config.allow_zero_rank:
            candidates.append(0)

        for rank in range(
            max(
                1,
                self.config.min_rank,
            ),
            self.config.max_rank + 1,
        ):

            if (
                self.config.enforce_power_of_two
                and
                not _is_power_of_two(rank)
            ):
                continue

            candidates.append(rank)

        candidates = (
            _canonical_rank_list(
                candidates
            )
        )

        if (
            self.config.strict
            and
            not candidates
        ):
            raise RuntimeError(
                "No legal candidate ranks available."
            )

        return candidates

    def filter_by_budget(
        self,
        candidates: Sequence[int],
        max_budget: int,
    ) -> List[int]: 
        _validate_non_negative_integer(
            "max_budget",
            max_budget,
        )

        valid = []

        for rank in candidates:

            if rank <= max_budget:
                valid.append(
                    int(rank)
                )

        return valid

    def layerwise_candidates(
        self,
        layer_names: Sequence[str],
    ) -> Dict[str, List[int]]:

        output = {}

        candidates = (
            self.candidate_ranks()
        )

        for layer_name in layer_names:

            output[
                str(layer_name)
            ] = list(
                candidates
            )

        return output

    def search_metadata(
        self,
    ) -> Dict[str, Any]:

        candidates = (
            self.candidate_ranks()
        )

        return {
            "candidate_count":
                len(candidates),

            "minimum_rank":
                min(candidates)
                if candidates
                else None,

            "maximum_rank":
                max(candidates)
                if candidates
                else None,

            "candidates":
                candidates,

            "power_of_two":
                self.config.enforce_power_of_two,
        }


# ============================================================================
# POLICIES
# ============================================================================


class BaseRankPolicy:
    """
    Base rank policy.
    """

    def __init__(
        self,
        config: RankConfig,
    ) -> None:

        if not isinstance(
            config,
            RankConfig,
        ):
            raise TypeError(
                "config must be RankConfig."
            )

        self.config = config

    @property
    def policy_name(
        self,
    ) -> str:

        raise RuntimeError(
            "Policy name unavailable."
        )

    def rank_for_layer(
        self,
        layer_name: str,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> int:

        raise RuntimeError(
            "Policy does not implement "
            "rank_for_layer."
        )


class FixedRankPolicy(
    BaseRankPolicy,
):

    @property
    def policy_name(
        self,
    ) -> str:

        return (
            RankPolicyType.FIXED.value
        )

    def rank_for_layer(
        self,
        layer_name: str,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> int:

        return int(
            self.config.default_rank
        )


class LayerwiseRankPolicy(
    BaseRankPolicy,
):

    def __init__(
        self,
        config: RankConfig,
        assignments: Optional[
            Dict[str, int]
        ] = None,
    ) -> None:

        super().__init__(
            config
        )

        self.assignments = (
            assignments or {}
        )

    @property
    def policy_name(
        self,
    ) -> str:

        return (
            RankPolicyType.LAYERWISE.value
        )

    def rank_for_layer(
        self,
        layer_name: str,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> int:

        if (
            layer_name
            in self.assignments
        ):
            return int(
                self.assignments[
                    layer_name
                ]
            )

        return int(
            self.config.default_rank
        )


class ProportionalRankPolicy(
    BaseRankPolicy,
):

    @property
    def policy_name(
        self,
    ) -> str:

        return (
            RankPolicyType.PROPORTIONAL.value
        )

    def rank_for_layer(
        self,
        layer_name: str,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> int:

        if (
            in_features is None
            or
            out_features is None
        ):
            return int(
                self.config.default_rank
            )

        rank = int(
            round(
                min(
                    in_features,
                    out_features,
                )
                *
                self.config.proportional_factor
            )
        )

        rank = _clamp_rank(
            rank=max(
                1,
                rank,
            ),
            minimum=self.config.min_rank,
            maximum=self.config.max_rank,
        )

        return rank


# ============================================================================
# MANAGER
# ============================================================================


class RankManager:
    """
    Central rank-management API.

    Responsibilities
    ----------------

    - rank assignment
    - rank validation
    - rank recommendation
    - rank search generation
    - rank budget accounting
    - DE encoding support
    - metadata export
    """

    def __init__(
        self,
        config: Optional[
            RankConfig
        ] = None,
        *,
        layer_assignments: Optional[
            Dict[str, int]
        ] = None,
    ) -> None:

        if config is None:
            config = RankConfig()

        if not isinstance(
            config,
            RankConfig,
        ):
            raise TypeError(
                "config must be RankConfig."
            )

        self.config = config

        self.statistics_tracker = (
            RankStatistics()
        )

        self.active_assignments: Dict[
            str,
            int,
        ] = {}

        self.search_space = (
            RankSearchSpace(
                config
            )
        )

        self.layer_assignments = (
            layer_assignments
            or {}
        )

        self.policy = (
            self._build_policy()
        )

    # ------------------------------------------------------------------
    # POLICY
    # ------------------------------------------------------------------

    def _build_policy(
        self,
    ) -> BaseRankPolicy:

        if self.config.layerwise:

            return LayerwiseRankPolicy(
                self.config,
                self.layer_assignments,
            )

        if self.config.proportional:

            return ProportionalRankPolicy(
                self.config
            )

        return FixedRankPolicy(
            self.config
        )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate_rank(
        self,
        rank: int,
    ) -> int:

        _validate_integer(
            "rank",
            rank,
        )

        if rank == 0:

            if not self.config.allow_zero_rank:
                raise ValueError(
                    "rank 0 is disabled by allow_zero_rank=False."
                )

            return 0

        if rank < 0:
            raise ValueError(
                "rank must be non-negative."
            )

        if rank < self.config.min_rank:
            raise ValueError(
                "rank below min_rank."
            )

        if rank > self.config.max_rank:
            raise ValueError(
                "rank above max_rank."
            )

        if (
            self.config.enforce_power_of_two
            and not _is_power_of_two(rank)
        ):
            raise ValueError(
                "rank violates power-of-two constraint."
            )

        return int(rank)

    def normalize_rank(
        self,
        rank: int,
    ) -> int:

        _validate_integer(
            "rank",
            rank,
        )

        if rank == 0 and self.config.allow_zero_rank:
            return 0

        if rank < 0 and self.config.allow_zero_rank:
            return 0

        rank = _clamp_rank(
            rank,
            self.config.min_rank,
            self.config.max_rank,
        )

        if (
            self.config.enforce_power_of_two
            and rank > 0
            and not _is_power_of_two(rank)
        ):

            candidates = self.search_space.candidate_ranks()

            if not candidates:
                raise RuntimeError(
                    "No legal ranks exist."
                )

            rank = min(
                candidates,
                key=lambda value: abs(value - rank),
            )

        return int(rank)

    # ------------------------------------------------------------------
    # DIMENSIONS
    # ------------------------------------------------------------------

    def module_dimensions(
        self,
        module: nn.Module,
    ) -> Tuple[int, int]:

        if not isinstance(
            module,
            nn.Module,
        ):
            raise TypeError(
                "module must be nn.Module."
            )

        if hasattr(
            module,
            "in_features",
        ) and hasattr(
            module,
            "out_features",
        ):
            return (
                int(
                    module.in_features
                ),
                int(
                    module.out_features
                ),
            )

        if hasattr(
            module,
            "base_layer",
        ):

            base = (
                module.base_layer
            )

            if hasattr(
                base,
                "in_features",
            ) and hasattr(
                base,
                "out_features",
            ):
                return (
                    int(
                        base.in_features
                    ),
                    int(
                        base.out_features
                    ),
                )

        raise ValueError(
            "Unable to infer module "
            "dimensions."
        )

    # ------------------------------------------------------------------
    # RANK ASSIGNMENT
    # ------------------------------------------------------------------

    def rank_for_layer(
        self,
        layer_name: str,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> int:

        rank = (
            self.policy.rank_for_layer(
                layer_name,
                in_features=in_features,
                out_features=out_features,
            )
        )

        rank = self.normalize_rank(
            rank
        )

        self.active_assignments[
            layer_name
        ] = rank

        return rank
    

    def rank_for_module(
        self,
        module_name: str,
        module: nn.Module,
    ) -> int:

        in_features, out_features = (
            self.module_dimensions(
                module
            )
        )

        return self.rank_for_layer(
            module_name,
            in_features=in_features,
            out_features=out_features,
        )

    def assign_ranks(
        self,
        modules: Mapping[
            str,
            nn.Module,
        ],
    ) -> Dict[str, int]:
        """
        Assign ranks to an entire module collection.
        """

        if not isinstance(
            modules,
            Mapping,
        ):
            raise TypeError(
                "modules must be a mapping."
            )

        assignments: Dict[str, int] = {}
        dimensions: Dict[str, Tuple[int, int]] = {}
        skipped: List[str] = []

        for module_name, module in modules.items():
            try:
                dimensions[module_name] = self.module_dimensions(module)
                rank = self.rank_for_module(
                    module_name,
                    module,
                )
                assignments[module_name] = rank
            except Exception as exc:
                if self.config.strict:
                    raise RuntimeError(
                        f"Failed to assign rank for module '{module_name}'."
                    ) from exc
                skipped.append(module_name)

        self.active_assignments = dict(assignments)
        self._update_assignment_statistics(dimensions=dimensions)
        self.statistics_tracker.rank_constraint_violations = len(skipped)

        return dict(assignments)

    # ------------------------------------------------------------------
    # SEARCH SPACE
    # ------------------------------------------------------------------

    def generate_search_space(
        self,
        layer_names: Optional[
            Sequence[str]
        ] = None,
    ) -> Dict[str, Any]:

        self.statistics_tracker\
            .search_space_generation_calls += 1

        candidates = (
            self.search_space
            .candidate_ranks()
        )

        if layer_names is None:

            return {
                "global":
                    candidates,

                "candidate_count":
                    len(candidates),
            }

        return (
            self.search_space
            .layerwise_candidates(
                layer_names
            )
        )

    def candidate_ranks(
        self,
    ) -> List[int]:

        return (
            self.search_space
            .candidate_ranks()
        )

    # ------------------------------------------------------------------
    # PARAMETER COST
    # ------------------------------------------------------------------

    def estimated_parameter_cost(
        self,
        rank: int,
        in_features: int,
        out_features: int,
    ) -> int:

        rank = self.validate_rank(
            rank
        )

        return (
            _estimate_lora_parameters(
                rank,
                in_features,
                out_features,
            )
        )

    def assignment_parameter_cost(
        self,
        assignments: Mapping[
            str,
            int,
        ],
        dimensions: Mapping[
            str,
            Tuple[int, int]
        ],
    ) -> int:

        total = 0

        for (
            layer_name,
            rank,
        ) in assignments.items():

            if (
                layer_name
                not in dimensions
            ):
                continue

            in_features, out_features = (
                dimensions[
                    layer_name
                ]
            )

            total += (
                self.estimated_parameter_cost(
                    rank,
                    in_features,
                    out_features,
                )
            )

        return int(total)

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------

    def recommend_rank(
        self,
        layer_name: str,
        *,
        in_features: int,
        out_features: int,
    ) -> RankRecommendation:

        self.statistics_tracker\
            .recommendation_calls += 1

        recommended_rank = (
            self.rank_for_layer(
                layer_name,
                in_features=in_features,
                out_features=out_features,
            )
        )

        parameter_cost = (
            self.estimated_parameter_cost(
                recommended_rank,
                in_features,
                out_features,
            )
        )

        if self.config.proportional:

            reason = (
                "proportional policy"
            )

        elif self.config.layerwise:

            reason = (
                "layerwise policy"
            )

        else:

            reason = (
                "fixed policy"
            )

        return RankRecommendation(
            layer_name=
                layer_name,

            rank=
                recommended_rank,

            reason=
                reason,

            estimated_parameter_cost=
                parameter_cost,

            confidence=
                1.0,

            metadata={
                "in_features":
                    in_features,

                "out_features":
                    out_features,

                "policy":
                    self.policy.policy_name,
            },
        )

    def recommend_ranks(
        self,
        modules: Mapping[
            str,
            nn.Module,
        ],
    ) -> Dict[
        str,
        RankRecommendation,
    ]:

        recommendations = {}

        for (
            module_name,
            module,
        ) in modules.items():

            try:

                in_features, out_features = (
                    self.module_dimensions(
                        module
                    )
                )

            except Exception:

                continue

            recommendations[
                module_name
            ] = self.recommend_rank(
                module_name,
                in_features=
                    in_features,

                out_features=
                    out_features,
            )

        return recommendations

    # ------------------------------------------------------------------
    # ABLATION
    # ------------------------------------------------------------------

    def ablation_report(
        self,
        ranks: Optional[
            Sequence[int]
        ] = None,
        *,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        layer_name: str = "layer",
    ) -> RankAblationReport:

        self.statistics_tracker.ablation_calls += 1

        if ranks is None:
            ranks = self.candidate_ranks()

        ranks = _canonical_rank_list(ranks)

        if not ranks:
            raise ValueError(
                "ablation_report requires at least one legal rank."
            )

        parameter_costs: Dict[int, int] = {}

        for rank in ranks:
            if (
                in_features is not None
                and out_features is not None
            ):
                parameter_costs[rank] = self.estimated_parameter_cost(
                    rank,
                    in_features,
                    out_features,
                )
            else:
                parameter_costs[rank] = int(rank)

        recommended_rank = self.normalize_rank(
            self.config.default_rank
        )

        return RankAblationReport(
            tested_ranks=list(ranks),
            parameter_costs=parameter_costs,
            recommended_rank=recommended_rank,
            minimum_rank=min(ranks),
            maximum_rank=max(ranks),
            rank_budget=sum(ranks),
            metadata={
                "policy": self.policy.policy_name,
                "candidate_count": len(ranks),
                "layer_name": layer_name,
                "in_features": in_features,
                "out_features": out_features,
            },
        )

    # ------------------------------------------------------------------
    # BUDGET
    # ------------------------------------------------------------------

    def total_rank_budget(
        self,
    ) -> int:

        return int(
            sum(
                self.active_assignments
                .values()
            )
        )

    def rank_budget_report(
        self,
    ) -> Dict[str, Any]:

        ranks = list(self.active_assignments.values())

        if not ranks:
            return {
                "total_rank_budget": 0,
                "average_rank": 0.0,
                "minimum_rank": 0,
                "maximum_rank": 0,
                "distinct_ranks": 0,
                "budget_utilization": 0.0,
                "estimated_adapted_parameters": 0,
                "search_space_candidate_count": len(self.candidate_ranks()),
            }

        total_budget = sum(ranks)
        max_budget = max(1, len(ranks) * self.config.max_rank)

        return {
            "total_rank_budget": total_budget,
            "average_rank": sum(ranks) / len(ranks),
            "minimum_rank": min(ranks),
            "maximum_rank": max(ranks),
            "distinct_ranks": len(set(ranks)),
            "budget_utilization": total_budget / max_budget,
            "estimated_adapted_parameters": self.statistics_tracker.estimated_adapted_parameters,
            "search_space_candidate_count": len(self.candidate_ranks()),
        }

    # ------------------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------------------

    def encoding(
        self,
    ) -> List[int]:
        """
        DE-compatible vector
        representation.
        """

        ordered_names = sorted(
            self.active_assignments
            .keys()
        )

        return [
            self.active_assignments[
                name
            ]
            for name in ordered_names
        ]

    def decode_encoding(
        self,
        vector: Sequence[int],
        layer_names: Sequence[str],
    ) -> Dict[str, int]:

        if len(vector) != len(
            layer_names
        ):
            raise ValueError(
                "encoding size mismatch."
            )

        assignments = {}

        for (
            layer_name,
            rank,
        ) in zip(
            layer_names,
            vector,
        ):
            assignments[
                layer_name
            ] = self.normalize_rank(
                int(rank)
            )

        return assignments

    def encoded_dimension(
        self,
    ) -> int:

        return len(
            self.active_assignments
        )

    def candidate_encodings(
        self,
        layer_names: Sequence[str],
    ) -> List[List[int]]:

        candidates = (
            self.candidate_ranks()
        )

        encodings = []

        for rank in candidates:

            encodings.append(
                [
                    rank
                    for _ in layer_names
                ]
            )

        return encodings
    
    # ------------------------------------------------------------------
    # STATISTICS UPDATE
    # ------------------------------------------------------------------

    def _update_assignment_statistics(
        self,
        dimensions: Optional[
            Mapping[str, Tuple[int, int]]
        ] = None,
    ) -> None:

        ranks = list(self.active_assignments.values())

        self.statistics_tracker.layers_seen = len(ranks)
        self.statistics_tracker.layers_assigned = len(ranks)
        self.statistics_tracker.estimated_rank_search_candidates = len(
            self.candidate_ranks()
        )

        if len(ranks) == 0:
            self.statistics_tracker.distinct_ranks_used = 0
            self.statistics_tracker.minimum_rank_observed = 0
            self.statistics_tracker.maximum_rank_observed = 0
            self.statistics_tracker.mean_rank = 0.0
            self.statistics_tracker.median_rank = 0.0
            self.statistics_tracker.total_rank_budget = 0
            self.statistics_tracker.estimated_adapted_parameters = 0
            return

        self.statistics_tracker.distinct_ranks_used = len(set(ranks))
        self.statistics_tracker.minimum_rank_observed = min(ranks)
        self.statistics_tracker.maximum_rank_observed = max(ranks)
        self.statistics_tracker.mean_rank = float(statistics.mean(ranks))
        self.statistics_tracker.median_rank = float(statistics.median(ranks))
        self.statistics_tracker.total_rank_budget = int(sum(ranks))

        if dimensions is not None:
            self.statistics_tracker.estimated_adapted_parameters = (
                self.assignment_parameter_cost(
                    self.active_assignments,
                    dimensions,
                )
            )
        else:
            self.statistics_tracker.estimated_adapted_parameters = 0

    # ------------------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------------------

    def rank_distribution(
        self,
    ) -> Dict[int, int]:

        distribution: Dict[
            int,
            int,
        ] = {}

        for rank in (
            self.active_assignments
            .values()
        ):

            distribution[
                rank
            ] = (
                distribution.get(
                    rank,
                    0,
                )
                + 1
            )

        return distribution

    def rank_summary(
        self,
    ) -> str:

        if not (
            self.active_assignments
        ):
            return (
                "No rank assignments."
            )

        ranks = sorted(
            self.active_assignments
            .values()
        )

        return (
            f"layers="
            f"{len(ranks)}, "
            f"min="
            f"{min(ranks)}, "
            f"max="
            f"{max(ranks)}, "
            f"mean="
            f"{statistics.mean(ranks):.2f}"
        )

    # ------------------------------------------------------------------
    # CONSTRAINT CHECKS
    # ------------------------------------------------------------------

    def check_constraints(
        self,
        assignments: Optional[
            Mapping[str, int]
        ] = None,
    ) -> bool:

        if assignments is None:

            assignments = (
                self.active_assignments
            )

        violations = 0

        for rank in (
            assignments.values()
        ):

            try:

                self.validate_rank(
                    rank
                )

            except Exception:

                violations += 1

        self.statistics_tracker\
            .rank_constraint_violations = (
                violations
            )

        return violations == 0

    def constraint_report(
        self,
    ) -> Dict[str, Any]:

        valid = (
            self.check_constraints()
        )

        return {
            "constraints_satisfied":
                valid,

            "violations":
                self.statistics_tracker
                .rank_constraint_violations,

            "power_of_two":
                self.config
                .enforce_power_of_two,

            "minimum_rank":
                self.config.min_rank,

            "maximum_rank":
                self.config.max_rank,
        }

    # ------------------------------------------------------------------
    # MODULE SUMMARIES
    # ------------------------------------------------------------------

    def module_dimension_summary(
        self,
        modules: Mapping[
            str,
            nn.Module,
        ],
    ) -> Dict[
        str,
        Dict[str, int]
    ]:

        summary = {}

        for (
            module_name,
            module,
        ) in modules.items():

            try:

                in_features, out_features = (
                    self.module_dimensions(
                        module
                    )
                )

            except Exception:

                continue

            summary[
                module_name
            ] = {
                "in_features":
                    in_features,

                "out_features":
                    out_features,

                "maximum_rank":
                    min(
                        in_features,
                        out_features,
                    ),
            }

        return summary

    # ------------------------------------------------------------------
    # SEARCH EXPORT
    # ------------------------------------------------------------------

    def search_space_metadata(
        self,
    ) -> Dict[str, Any]:

        candidates = (
            self.candidate_ranks()
        )

        return {
            "candidate_count":
                len(candidates),

            "candidate_ranks":
                candidates,

            "search_ready":
                True,

            "de_ready":
                True,

            "ablation_ready":
                True,

            "encoded_dimension":
                self.encoded_dimension(),
        }

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def metadata(
        self,
    ) -> Dict[str, Any]:

        self._update_assignment_statistics()

        return {
            "module": "RankManager",
            "policy": self.policy.policy_name,
            "policy_flags": {
                "layerwise": self.config.layerwise,
                "proportional": self.config.proportional,
                "symmetric": self.config.symmetric,
                "strict": self.config.strict,
                "allow_zero_rank": self.config.allow_zero_rank,
                "enforce_power_of_two": self.config.enforce_power_of_two,
            },
            "configuration": self.config.to_dict(),
            "active_assignments": dict(self.active_assignments),
            "candidate_ranks": self.candidate_ranks(),
            "rank_distribution": self.rank_distribution(),
            "total_rank_budget": self.total_rank_budget(),
            "rank_budget_report": self.rank_budget_report(),
            "search_space": self.search_space_metadata(),
            "encoding": self.encoding_metadata(),
            "statistics": self.statistics().to_dict(),
            "constraint_report": self.constraint_report(),
            "rank_search_ready": True,
            "ablation_ready": True,
            "de_optimization_ready": True,
            "placement_ready": True,
            "scaling_ready": True,
        }

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def diagnostics(
        self,
    ) -> Dict[str, Any]:

        ranks = list(self.active_assignments.values())

        if not ranks:
            return {
                "num_layers": 0,
                "rank_distribution": {},
                "budget_usage": 0,
                "candidate_count": len(self.candidate_ranks()),
                "estimated_adapted_parameters": 0,
                "estimated_rank_search_candidates": len(self.candidate_ranks()),
                "constraints_satisfied": True,
                "policy": self.policy.policy_name,
            }

        budget_report = self.rank_budget_report()

        return {
            "num_layers": len(ranks),
            "minimum_rank": min(ranks),
            "maximum_rank": max(ranks),
            "mean_rank": float(statistics.mean(ranks)),
            "median_rank": float(statistics.median(ranks)),
            "rank_distribution": self.rank_distribution(),
            "budget_usage": self.total_rank_budget(),
            "candidate_count": len(self.candidate_ranks()),
            "estimated_adapted_parameters": self.statistics_tracker.estimated_adapted_parameters,
            "estimated_rank_search_candidates": self.statistics_tracker.estimated_rank_search_candidates,
            "constraints_satisfied": self.check_constraints(),
            "budget_utilization": budget_report["budget_utilization"],
            "policy": self.policy.policy_name,
        }

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def statistics(
        self,
    ) -> RankStatistics:

        self._update_assignment_statistics()

        return (
            self.statistics_tracker
        )

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def export_configuration(
        self,
    ) -> Dict[str, Any]:

        return {
            "rank_configuration":
                self.config.to_dict(),

            "statistics":
                self.statistics()
                .to_dict(),

            "metadata":
                self.metadata(),

            "diagnostics":
                self.diagnostics(),

            "rank_budget_report":
                self.rank_budget_report(),

            "search_space":
                self.search_space_metadata(),

            "encoding":
                self.encoding(),
        }

    # ------------------------------------------------------------------
    # REPRODUCIBILITY
    # ------------------------------------------------------------------

    def export_assignments(
        self,
    ) -> Dict[str, int]:

        return dict(
            self.active_assignments
        )

    def load_assignments(
        self,
        assignments: Mapping[
            str,
            int,
        ],
    ) -> None:

        if not isinstance(
            assignments,
            Mapping,
        ):
            raise TypeError(
                "assignments must be a mapping."
            )

        normalized: Dict[str, int] = {}

        for layer_name, rank in assignments.items():
            rank_value = int(rank)

            if self.config.strict:
                rank_value = self.validate_rank(rank_value)
            else:
                rank_value = self.normalize_rank(rank_value)

            normalized[str(layer_name)] = rank_value

        self.active_assignments = normalized
        self._update_assignment_statistics()

    # ------------------------------------------------------------------
    # FACTORY
    # ------------------------------------------------------------------

    @classmethod
    def from_adapter(
        cls,
        adapter: nn.Module,
        config: Optional[
            RankConfig
        ] = None,
    ) -> "RankManager":

        if not isinstance(
            adapter,
            nn.Module,
        ):
            raise TypeError(
                "adapter must be nn.Module."
            )

        manager = cls(
            config=config
        )

        assignments: Dict[str, int] = {}

        if hasattr(adapter, "rank_configuration"):
            try:
                raw = adapter.rank_configuration()
            except Exception:
                raw = None

            if isinstance(raw, Mapping):
                assignments.update(
                    {
                        str(k): int(v)
                        for k, v in raw.items()
                    }
                )

        if not assignments and hasattr(adapter, "get_lora_layers"):
            try:
                layers = adapter.get_lora_layers()
            except Exception:
                layers = {}

            if isinstance(layers, Mapping):
                for name, layer in layers.items():
                    if hasattr(layer, "rank"):
                        assignments[str(name)] = int(layer.rank)
                    elif hasattr(layer, "adaptation_rank"):
                        assignments[str(name)] = int(layer.adaptation_rank)

        if assignments:
            manager.load_assignments(assignments)

        return manager

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        config: Optional[
            RankConfig
        ] = None,
    ) -> "RankManager":

        if not isinstance(
            model,
            nn.Module,
        ):
            raise TypeError(
                "model must be nn.Module."
            )

        manager = cls(
            config=config
        )

        modules: Dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or hasattr(module, "base_layer"):
                modules[name] = module

        if modules:
            try:
                manager.assign_ranks(modules)
            except Exception:
                if manager.config.strict:
                    raise

        return manager

    # @classmethod
    # def from_model(
    #     cls,
    #     model: nn.Module,
    #     config: Optional[
    #         RankConfig
    #     ] = None,
    # ) -> "RankManager":

    #     if not isinstance(
    #         model,
    #         nn.Module,
    #     ):
    #         raise TypeError(
    #             "model must be nn.Module."
    #         )

    #     return cls(
    #         config=config
    #     )

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    def verify_integrity(
        self,
    ) -> bool:

        try:

            self.check_constraints()

            self.statistics()

            self.metadata()

            return True

        except Exception:

            return False

    def validate_state(
        self,
    ) -> None:

        if not self.verify_integrity():

            raise RuntimeError(
                "RankManager integrity "
                "check failed."
            )

    # --------------------------------------------------------------
    # RESET
    # --------------------------------------------------------------

    @torch.no_grad()
    def reset_statistics(
        self,
    ) -> None:

        self.statistics_tracker = (
            RankStatistics()
        )

    @torch.no_grad()
    def reset(
        self,
    ) -> None:

        self.active_assignments.clear()

        self.reset_statistics()

    # --------------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------------

    def recommendation_summary(
        self,
        recommendations: Mapping[
            str,
            RankRecommendation,
        ],
    ) -> Dict[str, Any]:

        ranks = [
            rec.rank
            for rec
            in recommendations.values()
        ]

        if not ranks:

            return {}

        return {
            "num_layers":
                len(ranks),

            "average_rank":
                sum(ranks) / len(ranks),

            "minimum_rank":
                min(ranks),

            "maximum_rank":
                max(ranks),
        }

    def encoding_metadata(
        self,
    ) -> Dict[str, Any]:

        return {
            "dimension":
                self.encoded_dimension(),

            "candidate_ranks":
                self.candidate_ranks(),

            "search_space_size":
                len(
                    self.candidate_ranks()
                ),

            "encoding_ready":
                True,

            "de_ready":
                True,
        }

    # --------------------------------------------------------------
    # STRING REPRESENTATION
    # --------------------------------------------------------------

    def extra_repr(
        self,
    ) -> str:

        return (
            f"policy="
            f"{self.policy.policy_name}, "
            f"assignments="
            f"{len(self.active_assignments)}, "
            f"rank_budget="
            f"{self.total_rank_budget()}"
        )
    
# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "RankConfig",
    "RankStatistics",
    "RankPolicyType",
    "RankRecommendation",
    "RankAblationReport",
    "RankSearchSpace",
    "FixedRankPolicy",
    "LayerwiseRankPolicy",
    "ProportionalRankPolicy",
    "RankManager",
]
        

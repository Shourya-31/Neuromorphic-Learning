from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import torch
import torch.nn as nn
from torch import Tensor

from .lora_layer import (
    LoRAConfig,
    LoRALayer,
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class AdapterConfig:
    """
    Configuration for LoRA adaptation orchestration.

    This configuration controls:

    - automatic module discovery
    - LoRA injection
    - module filtering
    - parameter freezing
    - experiment instrumentation

    Future compatibility:

    - rank_config.py
    - placement.py
    - scaling.py
    - DE optimization
    """

    rank: int = 4

    alpha: float = 1.0

    dropout: float = 0.0

    target_modules: Optional[
        List[str]
    ] = None

    exclude_modules: Optional[
        List[str]
    ] = None

    enabled: bool = True

    auto_inject: bool = True

    freeze_base_model: bool = True

    verbose: bool = False

    def __post_init__(
        self,
    ) -> None:

        if not isinstance(
            self.rank,
            int,
        ):
            raise TypeError(
                "rank must be an integer."
            )

        if self.rank <= 0:
            raise ValueError(
                "rank must be positive."
            )

        if not isinstance(
            self.alpha,
            (int, float),
        ):
            raise TypeError(
                "alpha must be numeric."
            )

        if self.alpha <= 0:
            raise ValueError(
                "alpha must be positive."
            )

        if not isinstance(
            self.dropout,
            (int, float),
        ):
            raise TypeError(
                "dropout must be numeric."
            )

        if not (
            0.0 <= self.dropout < 1.0
        ):
            raise ValueError(
                "dropout must satisfy "
                "0 <= dropout < 1."
            )

        if (
            self.target_modules
            is not None
        ):
            if not isinstance(
                self.target_modules,
                list,
            ):
                raise TypeError(
                    "target_modules "
                    "must be a list."
                )

            for item in self.target_modules:

                if not isinstance(
                    item,
                    str,
                ):
                    raise TypeError(
                        "target_modules "
                        "must contain strings."
                    )

        if (
            self.exclude_modules
            is not None
        ):
            if not isinstance(
                self.exclude_modules,
                list,
            ):
                raise TypeError(
                    "exclude_modules "
                    "must be a list."
                )

            for item in self.exclude_modules:

                if not isinstance(
                    item,
                    str,
                ):
                    raise TypeError(
                        "exclude_modules "
                        "must contain strings."
                    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(self)


# ============================================================================
# STATISTICS
# ============================================================================


@dataclass
class AdapterStatistics:
    """
    Runtime adapter statistics.

    Used for:

    - PEFT studies
    - rank ablations
    - placement studies
    - parameter efficiency reports
    - DE optimization logging
    """

    num_modules_scanned: int = 0

    num_modules_adapted: int = 0

    num_linear_layers: int = 0

    num_lora_layers: int = 0

    total_parameters: int = 0

    trainable_parameters: int = 0

    adapted_parameters: int = 0

    parameter_ratio: float = 0.0

    injection_calls: int = 0

    enable_calls: int = 0

    disable_calls: int = 0

    merge_calls: int = 0

    unmerge_calls: int = 0

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(self)


# ============================================================================
# LORA ADAPTER
# ============================================================================


class LoRAAdapter(
    nn.Module,
):
    """
    Central LoRA orchestration layer.

    Responsibilities
    ----------------

    1. Discover linear layers

    2. Inject LoRA wrappers

    3. Track adapted modules

    4. Freeze base parameters

    5. Export metadata

    6. Support experiments

    7. Support future DE search

    Notes
    -----

    Does not implement LoRA itself.

    Uses:

        LoRALayer

    as the adaptation primitive.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdapterConfig,
    ) -> None:

        super().__init__()

        self.model = model

        self.config = config

        self._validate_model()

        self._validate_config()

        self.lora_layers: Dict[
            str,
            LoRALayer,
        ] = {}

        self.adapted_modules: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self.injection_complete = False

        self.statistics_tracker = (
            AdapterStatistics()
        )

        self._adapted_module_names: Set[
            str
        ] = set()

        if self.config.auto_inject:
            self.inject()

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_model(
        self,
    ) -> None:

        if not isinstance(
            self.model,
            nn.Module,
        ):
            raise TypeError(
                "model must be "
                "an nn.Module."
            )

        parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )

        if parameter_count <= 0:
            raise ValueError(
                "Model contains "
                "no parameters."
            )

    def _validate_config(
        self,
    ) -> None:

        if not isinstance(
            self.config,
            AdapterConfig,
        ):
            raise TypeError(
                "config must be "
                "AdapterConfig."
            )

    def _validate_injection(
        self,
    ) -> None:

        if not self.injection_complete:
            raise RuntimeError(
                "LoRA injection has "
                "not been completed."
            )

        if len(self.lora_layers) == 0:
            raise RuntimeError(
                "No LoRA layers found."
            )

    # ------------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------------

    def discover_linear_layers(
        self,
    ) -> Dict[str, nn.Linear]:
        """
        Discover all eligible
        nn.Linear layers.

        Uses named_modules()
        for efficient traversal.
        """

        discovered = {}

        scanned = 0

        linear_count = 0

        for (
            module_name,
            module,
        ) in self.model.named_modules():

            scanned += 1

            if isinstance(
                module,
                nn.Linear,
            ):
                linear_count += 1

                discovered[
                    module_name
                ] = module

        self.statistics_tracker.num_modules_scanned = (
            scanned
        )

        self.statistics_tracker.num_linear_layers = (
            linear_count
        )

        return discovered

    # ------------------------------------------------------------------
    # FILTERING
    # ------------------------------------------------------------------

    def should_adapt_module(
        self,
        module_name: str,
    ) -> bool:
        """
        Determine whether a module
        should receive LoRA adaptation.

        Supports:

        - target_modules
        - exclude_modules
        """

        if not isinstance(
            module_name,
            str,
        ):
            raise TypeError(
                "module_name must be a string."
            )

        module_name = (
            module_name.strip()
        )

        if len(module_name) == 0:
            return False

        if (
            self.config.exclude_modules
            is not None
        ):
            for pattern in (
                self.config.exclude_modules
            ):
                if pattern in module_name:
                    return False

        if (
            self.config.target_modules
            is None
        ):
            return True

        for pattern in (
            self.config.target_modules
        ):
            if pattern in module_name:
                return True

        

        return False

    # ------------------------------------------------------------------
    # MODULE REPLACEMENT
    # ------------------------------------------------------------------

    def _replace_module(
        self,
        module_path: str,
        new_module: nn.Module,
    ) -> None:
        """
        Replace a nested module.

        Supports:

        - standard attributes
        - nn.Sequential
        - nn.ModuleList
        - nn.ModuleDict
        """

        if not isinstance(
            module_path,
            str,
        ):
            raise TypeError(
                "module_path must be string."
            )

        if len(
            module_path.strip()
        ) == 0:
            raise ValueError(
                "module_path cannot be empty."
            )

        path_parts = (
            module_path.split(".")
        )

        parent = self.model

        for part in path_parts[:-1]:

            if isinstance(
                parent,
                (
                    nn.Sequential,
                    nn.ModuleList,
                )
            ):
                parent = parent[
                    int(part)
                ]

            elif isinstance(
                parent,
                nn.ModuleDict,
            ):
                parent = parent[
                    part
                ]

            else:

                if not hasattr(
                    parent,
                    part,
                ):
                    raise RuntimeError(
                        f"Unable to locate "
                        f"parent module "
                        f"'{module_path}'."
                    )

                parent = getattr(
                    parent,
                    part,
                )

        leaf = path_parts[-1]

        if isinstance(
            parent,
            (
                nn.Sequential,
                nn.ModuleList,
            )
        ):
            parent[
                int(leaf)
            ] = new_module

        elif isinstance(
            parent,
            nn.ModuleDict,
        ):
            parent[
                leaf
            ] = new_module

        else:

            if not hasattr(
                parent,
                leaf,
            ):
                raise RuntimeError(
                    f"Module "
                    f"'{module_path}' "
                    f"does not exist."
                )

            setattr(
                parent,
                leaf,
                new_module,
            )

    # ------------------------------------------------------------------
    # FREEZE SUPPORT
    # ------------------------------------------------------------------

    def _freeze_base_model(
        self,
    ) -> None:
        """
        Freeze all non-LoRA
        parameters.

        Called after injection.
        """

        for (
            _,
            parameter,
        ) in self.model.named_parameters():

            parameter.requires_grad = False

        for layer in (
            self.lora_layers.values()
        ):

            for parameter in (
                layer.lora_parameters()
            ):

                parameter.requires_grad = True

    # ------------------------------------------------------------------
    # INJECTION
    # ------------------------------------------------------------------

    def inject(
        self,
    ) -> nn.Module:
        """
        Inject LoRA layers into all
        eligible linear modules.

        Preserves:

        - module hierarchy
        - pretrained weights
        - parameter values
        """

        self.statistics_tracker.injection_calls += 1

        if self.injection_complete:
            return self.model

        discovered_layers = (
            self.discover_linear_layers()
        )

        lora_config = LoRAConfig(
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            enabled=self.config.enabled,
        )

        adapted_count = 0

        for (
            module_name,
            linear_layer,
        ) in discovered_layers.items():

            if not self.should_adapt_module(
                module_name
            ):
                continue

            # if isinstance(
            #     linear_layer,
            #     LoRALayer,
            # ):
            #     continue

            lora_layer = LoRALayer(
                base_layer=linear_layer,
                config=lora_config,
            )

            self._replace_module(
                module_name,
                lora_layer,
            )

            self.lora_layers[
                module_name
            ] = lora_layer

            self.adapted_modules[
                module_name
            ] = {
                "module_name":
                    module_name,

                "layer":
                    lora_layer,

                "rank":
                    lora_layer.rank,

                "parameter_count":
                    lora_layer.num_lora_parameters(),

                "base_parameters":
                    lora_layer.num_base_parameters(),

                "alpha":
                    lora_layer.alpha,
            }

            self._adapted_module_names.add(
                module_name
            )

            adapted_count += 1

        self.statistics_tracker.num_modules_adapted = (
            adapted_count
        )

        self.statistics_tracker.num_lora_layers = (
            len(self.lora_layers)
        )

        self.injection_complete = True

        if (
            self.config.freeze_base_model
        ):
            self._freeze_base_model()

        self._refresh_statistics()

        return self.model

    # ------------------------------------------------------------------
    # ENABLE
    # ------------------------------------------------------------------

    def enable(
        self,
    ) -> None:

        self._validate_injection()

        self.statistics_tracker.enable_calls += 1

        for layer in (
            self.lora_layers.values()
        ):
            layer.enable()

    # ------------------------------------------------------------------
    # DISABLE
    # ------------------------------------------------------------------

    def disable(
        self,
    ) -> None:

        self._validate_injection()

        self.statistics_tracker.disable_calls += 1

        for layer in (
            self.lora_layers.values()
        ):
            layer.disable()

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    def merge(
        self,
    ) -> None:

        self._validate_injection()

        self.statistics_tracker.merge_calls += 1

        for layer in (
            self.lora_layers.values()
        ):

            if layer.can_merge():
                layer.merge()

    # ------------------------------------------------------------------
    # UNMERGE
    # ------------------------------------------------------------------

    def unmerge(
        self,
    ) -> None:

        self._validate_injection()

        self.statistics_tracker.unmerge_calls += 1

        for layer in (
            self.lora_layers.values()
        ):

            if layer.can_unmerge():
                layer.unmerge()

    # ------------------------------------------------------------------
    # PARAMETER ACCOUNTING
    # ------------------------------------------------------------------

    def total_parameters(
        self,
    ) -> int:

        return int(
            sum(
                parameter.numel()
                for parameter in
                self.model.parameters()
            )
        )

    def trainable_parameters(
        self,
    ) -> int:
        
        return int(
            sum(
                parameter.numel()
                for parameter in
                self.model.parameters()
                if parameter.requires_grad
            )
        )

    def adapted_parameters(
        self,
    ) -> int:

        total = 0

        for layer in (
            self.lora_layers.values()
        ):
            total += (
                layer.num_lora_parameters()
            )

        return int(total)

    def parameter_ratio(
        self,
    ) -> float:

        base_parameters = 0

        for layer in (
            self.lora_layers.values()
        ):
            base_parameters += (
                layer.num_base_parameters()
            )

        adapted = (
            self.adapted_parameters()
        )

        if base_parameters <= 0:
            return 0.0

        return float(
            adapted / base_parameters
        )

    # ------------------------------------------------------------------
    # INTERNAL STATISTICS
    # ------------------------------------------------------------------

    def _refresh_statistics(
        self,
    ) -> None:

        self.statistics_tracker.total_parameters = (
            self.total_parameters()
        )

        self.statistics_tracker.trainable_parameters = (
            self.trainable_parameters()
        )

        self.statistics_tracker.adapted_parameters = (
            self.adapted_parameters()
        )

        self.statistics_tracker.parameter_ratio = (
            self.parameter_ratio()
        )

        self.statistics_tracker.num_lora_layers = (
            len(self.lora_layers)
        )

    # ------------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------------

    def adapted_model(
        self,
    ) -> nn.Module:

        if (
            not self.injection_complete
        ):
            raise RuntimeError(
                "Adapter has not been "
                "injected."
            )

        return self.model

    def get_lora_layers(
        self,
    ) -> Dict[str, LoRALayer]:

        return dict(
            self.lora_layers
        )
    
    def adapted_modules_summary(
        self,
    ) -> Dict[str, Any]:

        return {
            name: {
                k: v
                for (
                    k,
                    v,
                )
                in info.items()
                if k != "layer"
            }
            for (
                name,
                info,
            )
            in self.adapted_modules.items()
        }

    def adapted_layer_names(
        self,
    ) -> List[str]:

        return sorted(
            self.lora_layers.keys()
        )

    def num_adapted_layers(
        self,
    ) -> int:

        return len(
            self.lora_layers
        )

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def metadata(
        self,
    ) -> Dict[str, Any]:

        self._refresh_statistics()

        return {
            "num_modules_scanned":
                self.statistics_tracker
                .num_modules_scanned,

            "num_modules_adapted":
                self.statistics_tracker
                .num_modules_adapted,

            "num_linear_layers":
                self.statistics_tracker
                .num_linear_layers,

            "num_lora_layers":
                self.statistics_tracker
                .num_lora_layers,

            "injection_calls":
                self.statistics_tracker
                .injection_calls,

            "enable_calls":
                self.statistics_tracker
                .enable_calls,

            "disable_calls":
                self.statistics_tracker
                .disable_calls,

            "merge_calls":
                self.statistics_tracker
                .merge_calls,

            "unmerge_calls":
                self.statistics_tracker
                .unmerge_calls,
            "module":
                "LoRAAdapter",

            "injection_complete":
                self.injection_complete,
        

            "rank":
                self.config.rank,

            "alpha":
                self.config.alpha,

            "dropout":
                self.config.dropout,

            "freeze_base_model":
                self.config.freeze_base_model,

            "enabled":
                self.config.enabled,

            "number_of_adapted_layers":
                self.num_adapted_layers(),

            "adapted_layer_names":
                self.adapted_layer_names(),

            "adapted_modules":
                self.adapted_modules_summary(),

            "target_modules":
                self.config.target_modules,

            "exclude_modules":
                self.config.exclude_modules,

            "parameter_statistics": {
                "total_parameters":
                    self.total_parameters(),

                "trainable_parameters":
                    self.trainable_parameters(),

                "adapted_parameters":
                    self.adapted_parameters(),

                "parameter_ratio":
                    self.parameter_ratio(),
            },

            "memory_statistics":
                self.memory_report(),

            "placement_statistics":
                self.placement_report(),

            "rank_statistics":
                self.rank_report(),

            "experiment_export_ready":
                True,

            "placement_ready":
                True,

            "rank_search_ready":
                True,

            "de_optimization_ready":
                True,

            "scaling_studies_ready":
                True,

            "ablation_ready":
                True,

            "num_modules_scanned":
                self.statistics_tracker
                .num_modules_scanned,

            "num_modules_adapted":
                self.statistics_tracker
                .num_modules_adapted,

            "num_linear_layers":
                self.statistics_tracker
                .num_linear_layers,

            "num_lora_layers":
                self.statistics_tracker
                .num_lora_layers,

            "injection_calls":
                self.statistics_tracker
                .injection_calls,

            "enable_calls":
                self.statistics_tracker
                .enable_calls,

            "disable_calls":
                self.statistics_tracker
                .disable_calls,

            "merge_calls":
                self.statistics_tracker
                .merge_calls,

            "unmerge_calls":
                self.statistics_tracker
                .unmerge_calls,
        }
    
    def export_configuration(
        self,
    ) -> Dict[str, Any]:
        """
        Export complete adapter
        configuration for:

        - experiments
        - hyperparameter sweeps
        - DE optimization
        - reproducibility
        """

        self._refresh_statistics()

        return {
            "adapter_config":
                self.config.to_dict(),

            "statistics":
                self.statistics_tracker.to_dict(),

            "adapted_layers":
                self.adapted_layer_names(),

            "rank_configuration":
                self.rank_configuration(),

            "placement_configuration":
                self.placement_configuration(),

            "scaling_configuration":
                self.scaling_configuration(),

            "memory_report":
                self.memory_report(),

            "rank_report":
                self.rank_report(),

            "placement_report":
                self.placement_report(),
        }

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def statistics(
        self,
    ) -> AdapterStatistics:

        self._refresh_statistics()

        return self.statistics_tracker

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def diagnostics(
        self,
    ) -> Dict[str, Any]:

        self._validate_injection()

        diagnostics = {}

        adaptation_strength = []

        weight_norms = []

        a_norms = []

        b_norms = []

        for (
            module_name,
            layer,
        ) in self.lora_layers.items():

            layer_diag = (
                layer.diagnostics()
            )

            diagnostics[
                module_name
            ] = layer_diag

            adaptation_strength.append(
                layer_diag[
                    "adaptation_strength"
                ]
            )

            weight_norms.append(
                layer_diag[
                    "lora_weight_norm"
                ]
            )

            a_norms.append(
                layer_diag[
                    "lora_A_norm"
                ]
            )

            b_norms.append(
                layer_diag[
                    "lora_B_norm"
                ]
            )

        diagnostics[
            "aggregate"
        ] = {
            "num_layers":
                len(self.lora_layers),

            "mean_adaptation_strength":
                (
                    sum(
                        adaptation_strength
                    )
                    /
                    max(
                        1,
                        len(
                            adaptation_strength
                        )
                    )
                ),

            "mean_lora_weight_norm":
                (
                    sum(weight_norms)
                    /
                    max(
                        1,
                        len(weight_norms)
                    )
                ),

            "mean_lora_A_norm":
                (
                    sum(a_norms)
                    /
                    max(
                        1,
                        len(a_norms)
                    )
                ),

            "mean_lora_B_norm":
                (
                    sum(b_norms)
                    /
                    max(
                        1,
                        len(b_norms)
                    )
                ),
        }

        return diagnostics

    # ------------------------------------------------------------------
    # CHECKPOINT EXPORT
    # ------------------------------------------------------------------

    def lora_state_dict(
        self,
    ) -> Dict[str, Any]:

        self._validate_injection()

        weights = {}

        for (
            module_name,
            layer,
        ) in self.lora_layers.items():

            layer_state = (
                layer.lora_state_dict()
            )

            weights[
                f"{module_name}.lora_A"
            ] = layer_state[
                "lora_A"
            ]

            weights[
                f"{module_name}.lora_B"
            ] = layer_state[
                "lora_B"
            ]

        return {
            "weights":
                weights,

            "metadata": {
                "rank":
                    self.config.rank,

                "alpha":
                    self.config.alpha,

                "dropout":
                    self.config.dropout,

                "adapted_layers":
                    self.adapted_layer_names(),

                "placement_configuration":
                    self.placement_configuration(),

                "rank_configuration":
                    self.rank_configuration(),

                "scaling_configuration":
                    self.scaling_configuration(),
            },
        }

    # ------------------------------------------------------------------
    # CHECKPOINT LOAD
    # ------------------------------------------------------------------

    @torch.no_grad()
    def load_lora_state_dict(
        self,
        state_dict: Dict[
            str,
            Any,
        ],
        strict: bool = True,
    ) -> None:

        self._validate_injection()

        if not isinstance(
            state_dict,
            dict,
        ):
            raise TypeError(
                "state_dict must "
                "be a dictionary."
            )

        if (
            "weights"
            in state_dict
        ):
            state_dict = (
                state_dict[
                    "weights"
                ]
            )

        missing = []

        for (
            module_name,
            layer,
        ) in self.lora_layers.items():

            key_a = (
                f"{module_name}.lora_A"
            )

            key_b = (
                f"{module_name}.lora_B"
            )

            if key_a not in state_dict:
                missing.append(
                    key_a
                )
                continue

            if key_b not in state_dict:
                missing.append(
                    key_b
                )
                continue

            state = {
                "lora_A":
                    state_dict[
                        key_a
                    ].to(
                        layer.lora_A.device
                    ),
                "lora_B":
                    state_dict[
                        key_b
                    ].to(
                        layer.lora_B.device
                    ),
            }

            layer.load_lora_state_dict(
                state
            )

        if strict and missing:

            raise KeyError(
                "Missing LoRA keys: "
                + ", ".join(
                    missing
                )
            )

    # ------------------------------------------------------------------
    # EXPERIMENT SUPPORT
    # ------------------------------------------------------------------

    def rank_configuration(
        self,
    ) -> Dict[str, int]:

        return {
            name: layer.rank
            for (
                name,
                layer,
            ) in self.lora_layers.items()
        }
    
    def rank_report(
        self,
    ) -> Dict[str, Any]:
        """
        Rank-analysis report.

        Used by:

        - rank_config.py
        - rank ablations
        - DE optimization
        """

        self._validate_injection()

        rank_counts = {}

        total_adapted_parameters = 0

        for layer in (
            self.lora_layers.values()
        ):

            rank = layer.rank

            rank_counts[rank] = (
                rank_counts.get(
                    rank,
                    0,
                )
                + 1
            )

            total_adapted_parameters += (
                layer.num_lora_parameters()
            )

        return {
            "num_adapted_layers":
                len(
                    self.lora_layers
                ),

            "rank_distribution":
                rank_counts,

            "total_adapted_parameters":
                total_adapted_parameters,

            "average_rank":
                (
                    sum(
                        r * c
                        for (
                            r,
                            c,
                        ) in rank_counts.items()
                    )
                    /
                    max(
                        1,
                        sum(
                            rank_counts.values()
                        )
                    )
                ),
        }

    def placement_configuration(
        self,
    ) -> List[str]:

        return self.adapted_layer_names()
    
    def placement_report(
        self,
) -> Dict[str, Any]:
        """
        Placement-analysis report.

        Used by:

        - placement.py
        - placement ablations
        - DE placement search
        """

        self._validate_injection()

        report = {}

        for (
            name,
            layer,
        ) in self.lora_layers.items():

            report[name] = {
                "rank":
                    layer.rank,

                "alpha":
                    layer.alpha,

                "in_features":
                    layer.base_layer.in_features,

                "out_features":
                    layer.base_layer.out_features,

                "base_parameters":
                    layer.num_base_parameters(),

                "lora_parameters":
                    layer.num_lora_parameters(),

                "parameter_ratio":
                    layer.parameter_ratio(),
            }

        return report

    def scaling_configuration(
        self,
    ) -> Dict[str, float]:

        return {
            name: float(
                layer.alpha / layer.rank
            )
            for (
                name,
                layer,
            ) in self.lora_layers.items()
        }

    def efficiency_report(
        self,
    ) -> Dict[str, Any]:

        self._validate_injection()

        reports = {}

        for (
            name,
            layer,
        ) in self.lora_layers.items():

            reports[
                name
            ] = (
                layer.efficiency_report()
            )

        reports[
            "aggregate"
        ] = {
            "adapted_layers":
                self.num_adapted_layers(),

            "total_parameters":
                self.total_parameters(),

            "trainable_parameters":
                self.trainable_parameters(),

            "adapted_parameters":
                self.adapted_parameters(),

            "parameter_ratio":
                self.parameter_ratio(),
        }

        return reports
    
    def memory_report(
        self,
    ) -> Dict[str, Any]:
        """
        Memory-efficiency report.

        Used for:

        - paper tables
        - PEFT studies
        - memory ablations
        """

        self._validate_injection()

        base_parameters = 0

        lora_parameters = 0

        for layer in (
            self.lora_layers.values()
        ):

            base_parameters += (
                layer.num_base_parameters()
            )

            lora_parameters += (
                layer.num_lora_parameters()
            )

        base_memory_mb = (
            base_parameters * 4
        ) / (1024 ** 2)

        lora_memory_mb = (
            lora_parameters * 4
        ) / (1024 ** 2)

        memory_reduction_ratio = 0.0

        if base_memory_mb > 0:

            memory_reduction_ratio = (
                1.0
                -
                (
                    lora_memory_mb
                    /
                    base_memory_mb
                )
            )

        return {
            "base_parameters":
                base_parameters,

            "lora_parameters":
                lora_parameters,

            "base_memory_mb":
                base_memory_mb,

            "lora_memory_mb":
                lora_memory_mb,

            "memory_reduction_ratio":
                memory_reduction_ratio,
        }

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify_integrity(
        self,
    ) -> bool:

        if (
            not self.injection_complete
        ):
            return False

        if len(
            self.lora_layers
        ) == 0:
            return False

        for layer in (
            self.lora_layers.values()
        ):

            if not (
                layer.verify_integrity()
            ):
                return False

        return True

    def validate_state(
        self,
    ) -> None:

        self._validate_injection()

        if not self.verify_integrity():

            raise RuntimeError(
                "Adapter integrity "
                "check failed."
            )

        for (
            module_name,
            layer,
        ) in self.lora_layers.items():

            try:
                layer.validate_state()

            except Exception as exc:

                raise RuntimeError(
                    f"Validation failed "
                    f"for layer "
                    f"'{module_name}'."
                ) from exc

    # ------------------------------------------------------------------
    # RESET SUPPORT
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_statistics(
        self,
    ) -> None:

        self.statistics_tracker = (
            AdapterStatistics()
        )

        self.statistics_tracker.num_lora_layers = (
            len(
                self.lora_layers
            )
        )

        self.statistics_tracker.num_modules_adapted = (
            len(
                self.lora_layers
            )
        )

        for layer in (
            self.lora_layers.values()
        ):
            layer.reset_statistics()

    @torch.no_grad()
    def reset(
        self,
    ) -> None:

        self._validate_injection()

        for layer in (
            self.lora_layers.values()
        ):
            layer.reset()

        self.reset_statistics()

    # ------------------------------------------------------------------
    # MODULE API
    # ------------------------------------------------------------------

    def forward(
        self,
        *args,
        **kwargs,
    ):

        return self.model(
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # STRING REPRESENTATION
    # ------------------------------------------------------------------

    def extra_repr(
        self,
    ) -> str:

        return (
            f"adapted_layers="
            f"{len(self.lora_layers)}, "
            f"rank="
            f"{self.config.rank}, "
            f"alpha="
            f"{self.config.alpha:.4f}, "
            f"freeze_base_model="
            f"{self.config.freeze_base_model}, "
            f"injection_complete="
            f"{self.injection_complete}"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AdapterConfig",
    "AdapterStatistics",
    "LoRAAdapter",
]
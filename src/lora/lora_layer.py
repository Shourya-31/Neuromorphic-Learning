from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class LoRAConfig:
    """
    Configuration for Low-Rank Adaptation.

    Paper formulation:

        W' = W + αBA

    where

        A ∈ R^(r × d)
        B ∈ R^(k × r)

    and r << min(d, k)
    """

    rank: int = 4

    alpha: float = 1.0

    dropout: float = 0.0

    use_bias: bool = True

    merge_weights: bool = False

    enabled: bool = True

    def __post_init__(self) -> None:

        if not isinstance(self.rank, int):
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

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(self)


# ============================================================================
# STATISTICS
# ============================================================================


@dataclass
class LoRAStatistics:

    forward_calls: int

    merge_count: int

    unmerge_count: int

    enable_count: int

    disable_count: int

    merged: bool

    enabled: bool

    rank: int

    alpha: float

    base_parameters: int

    lora_parameters: int

    parameter_ratio: float

    memory_bytes: int

    lora_memory_bytes: int

    memory_reduction_ratio: float

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(self)


# ============================================================================
# LORA LAYER
# ============================================================================


class LoRALayer(nn.Module):
    """
    Research-grade LoRA wrapper.

    Generic adaptation layer for:

        nn.Linear

    without assumptions about the
    surrounding architecture.

    Designed for future integration with:

        src/ode/vector_field.py

    Supports:

    - merge / unmerge
    - enable / disable
    - parameter accounting
    - metadata export
    - rank studies
    - DE optimization
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig,
    ) -> None:
        super().__init__()

        self._validate_linear_layer(
            base_layer
        )

        if not isinstance(
            config,
            LoRAConfig,
        ):
            raise TypeError(
                "config must be LoRAConfig."
            )

        self.config = config

        self.base_layer = base_layer

        self.in_features = int(
            base_layer.in_features
        )

        self.out_features = int(
            base_layer.out_features
        )

        self.rank = int(
            config.rank
        )

        max_rank = min(
            self.in_features,
            self.out_features,
        )

        if self.rank > max_rank:
            raise ValueError(
                f"rank ({self.rank}) exceeds "
                f"maximum allowable rank "
                f"({max_rank})."
            )

        self.alpha = float(
            config.alpha
        )

        self.scaling = (
            self.alpha /
            float(self.rank)
        )

        self.enabled = bool(
            config.enabled
        )

        self.merged = False

        self.forward_calls = 0
        self.merge_count = 0
        self.unmerge_count = 0

        self.enable_count = 0
        self.disable_count = 0

        self._cached_merge_delta: Optional[Tensor] = None

        self.register_buffer(
            "_merge_checksum",
            torch.zeros(
                (),
                dtype=torch.float64,
            ),
            persistent=False,
        )

        self._freeze_base_layer()

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.empty(
                self.out_features,
                self.rank,
            )
        )

        if config.dropout > 0.0:

            self.dropout = nn.Dropout(
                p=float(
                    config.dropout
                )
            )

        else:

            self.dropout = nn.Identity()

        self.register_buffer(
            "_base_weight_snapshot",
            self.base_layer.weight.detach().clone(),
            persistent=False,
        )


        self.reset_lora_parameters()

        if config.merge_weights:
            self.merge()

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_linear_layer(
        self,
        layer: nn.Module,
    ) -> None:

        if not isinstance(
            layer,
            nn.Linear,
        ):
            raise TypeError(
                "base_layer must "
                "be nn.Linear."
            )

        if layer.in_features <= 0:
            raise ValueError(
                "Linear layer has "
                "invalid in_features."
            )

        if layer.out_features <= 0:
            raise ValueError(
                "Linear layer has "
                "invalid out_features."
            )

        if layer.weight.numel() == 0:
            raise ValueError(
                "Linear layer weight "
                "must be non-empty."
            )

        if not torch.isfinite(
            layer.weight
        ).all():
            raise RuntimeError(
                "Linear layer weight "
                "contains non-finite values."
            )

    def _validate_input(
        self,
        x: Tensor,
    ) -> None:

        if not isinstance(
            x,
            Tensor,
        ):
            raise TypeError(
                "Input must be Tensor."
            )

        if x.numel() == 0:
            raise ValueError(
                "Input must be non-empty."
            )

        if x.shape[-1] != (
            self.in_features
        ):
            raise ValueError(
                f"Expected last dimension "
                f"{self.in_features}, "
                f"got {x.shape[-1]}."
            )

        if not torch.isfinite(
            x
        ).all():
            raise RuntimeError(
                "Input contains "
                "non-finite values."
            )

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def _freeze_base_layer(
        self,
    ) -> None:

        for parameter in (
            self.base_layer.parameters()
        ):
            parameter.requires_grad = False

    @torch.no_grad()
    def reset_lora_parameters(
        self,
    ) -> None:
        """
        Standard LoRA initialization.

        A ~ Kaiming Uniform
        B = 0

        Guarantees:

            output_before_training
            ==
            original_output
        """

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B,
        )

    # ------------------------------------------------------------------
    # PARAMETER ACCOUNTING
    # ------------------------------------------------------------------

    def num_base_parameters(
        self,
    ) -> int:

        return int(
            sum(
                p.numel()
                for p in
                self.base_layer.parameters()
            )
        )

    def num_lora_parameters(
        self,
    ) -> int:

        return int(
            self.lora_A.numel()
            +
            self.lora_B.numel()
        )

    def parameter_ratio(
        self,
    ) -> float:

        base = (
            self.num_base_parameters()
        )

        if base <= 0:
            return 0.0

        return float(
            self.num_lora_parameters()
            /
            base
        )

    # ------------------------------------------------------------------
    # ENABLE / DISABLE
    # ------------------------------------------------------------------

    def enable(
        self,
    ) -> None:
        
        self.enable_count += 1

        self.enabled = True

    def disable(
        self,
    ) -> None:
        
        self.disable_count += 1

        self.enabled = False

    # ------------------------------------------------------------------
    # WEIGHT UPDATE
    # ------------------------------------------------------------------

    def delta_weight(
        self,
    ) -> Tensor:
        """
        Computes:

            α/r * BA
        """

        delta = torch.matmul(
            self.lora_B,
            self.lora_A,
        )

        delta = (
            self.scaling
            *
            delta
        )

        return delta.to(
            device=
            self.base_layer.weight.device,
            dtype=
            self.base_layer.weight.dtype,
        )

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def merge(
        self,
    ) -> None:

        if self.merged:
            return

        delta = self.delta_weight()

        self.base_layer.weight.add_(
            delta
        )

        self._cached_merge_delta = (
            delta.detach().clone()
        )

        self._merge_checksum.fill_(
            float(
                torch.norm(
                    delta
                ).detach().cpu()
            )
        )

        self.merge_count += 1

        self.merged = True


    @torch.no_grad()
    def unmerge(
        self,
    ) -> None:

        if not self.merged:
            return

        if self._cached_merge_delta is None:
            raise RuntimeError(
                "Missing cached merge delta."
            )

        self.base_layer.weight.sub_(
            self._cached_merge_delta
        )

        self._cached_merge_delta = None

        self.unmerge_count += 1

        self.merged = False

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def _compute_lora_output(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Computes:

            (BA)x

        using efficient
        vectorized operations.
        """

        x = self.dropout(x)

        lora_hidden = F.linear(
            x,
            self.lora_A,
            bias=None,
        )

        lora_output = F.linear(
            lora_hidden,
            self.lora_B,
            bias=None,
        )

        return lora_output

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        self._validate_input(
            x
        )
        if (
            self.training
            and
            self.merged
        ):
            raise RuntimeError(
                "Cannot train merged "
                "LoRA layer."
            )

        self.forward_calls += 1

        base_output = (
            self.base_layer(x)
        )

        if not self.enabled:
            return base_output

        if self.merged:
            return base_output

        lora_output = (
            self._compute_lora_output(
                x
            )
        )

        return (
            base_output
            +
            self.scaling
            *
            lora_output
        )

    # ------------------------------------------------------------------
    # STATE QUERIES
    # ------------------------------------------------------------------

    @property
    def is_enabled(
        self,
    ) -> bool:

        return self.enabled

    @property
    def is_merged(
        self,
    ) -> bool:

        return self.merged

    @property
    def adaptation_rank(
        self,
    ) -> int:

        return self.rank

    @property
    def adaptation_alpha(
        self,
    ) -> float:

        return self.alpha

    @property
    def adaptation_scaling(
        self,
    ) -> float:

        return self.scaling

    # ------------------------------------------------------------------
    # TRAINABLE PARAMETERS
    # ------------------------------------------------------------------

    def trainable_parameters(
        self,
    ) -> int:
        """
        Number of trainable LoRA
        parameters.

        Useful for PEFT tables.
        """

        return int(
            sum(
                parameter.numel()
                for parameter in
                self.parameters()
                if parameter.requires_grad
            )
        )

    def frozen_parameters(
        self,
    ) -> int:

        return int(
            sum(
                parameter.numel()
                for parameter in
                self.parameters()
                if not parameter.requires_grad
            )
        )
    
    def parameter_memory_bytes(
        self,
    ) -> int:

        total = 0

        for parameter in self.parameters():

            total += (
                parameter.numel()
                * parameter.element_size()
            )

        return int(total)

    def lora_memory_bytes(
        self,
    ) -> int:

        total = (
            self.lora_A.numel()
            * self.lora_A.element_size()
        )

        total += (
            self.lora_B.numel()
            * self.lora_B.element_size()
        )

        return int(total)

    def memory_reduction_ratio(
        self,
    ) -> float:

        base = (
            self.num_base_parameters()
        )

        lora = (
            self.num_lora_parameters()
        )

        if lora == 0:
            return float("inf")

        return float(base / lora)

    # ------------------------------------------------------------------
    # RESEARCH METADATA
    # ------------------------------------------------------------------

    def metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Export metadata for:

        - experiment manager
        - metrics subsystem
        - visualization subsystem
        - future DE optimization
        """

        return {
            "module":
                "LoRALayer",

            "rank":
                self.rank,

            "alpha":
                self.alpha,

            "scaling":
                self.scaling,

            "merged":
                self.merged,

            "enabled":
                self.enabled,

            "dropout":
                self.config.dropout,

            "base_parameters":
                self.num_base_parameters(),

            "lora_parameters":
                self.num_lora_parameters(),

            "parameter_ratio":
                self.parameter_ratio(),

            "trainable_parameters":
                self.trainable_parameters(),

            "frozen_parameters":
                self.frozen_parameters(),

            "in_features":
                self.in_features,

            "out_features":
                self.out_features,

           "forward_calls":
                self.forward_calls,

            "merge_count":
                self.merge_count,

            "unmerge_count":
                self.unmerge_count,

            "enable_count":
                self.enable_count,

            "disable_count":
                self.disable_count,

            "merge_weights":
                self.config.merge_weights,
        }

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    def statistics(
    self,
) -> LoRAStatistics:

        return LoRAStatistics(
            forward_calls=
                self.forward_calls,

            merge_count=
                self.merge_count,

            unmerge_count=
                self.unmerge_count,

            enable_count=
                self.enable_count,

            disable_count=
                self.disable_count,

            merged=
                self.merged,

            enabled=
                self.enabled,

            rank=
                self.rank,

            alpha=
                self.alpha,

            base_parameters=
                self.num_base_parameters(),

            lora_parameters=
                self.num_lora_parameters(),

            parameter_ratio=
                self.parameter_ratio(),

            memory_bytes=
                self.parameter_memory_bytes(),

            lora_memory_bytes=
                self.lora_memory_bytes(),

            memory_reduction_ratio=
                self.memory_reduction_ratio(),
        )

    # ------------------------------------------------------------------
    # EXPORT HELPERS
    # ------------------------------------------------------------------

    def state_summary(
        self,
    ) -> Dict[str, Any]:
        """
        Lightweight runtime summary.
        """

        return {
            "enabled":
                self.enabled,

            "merged":
                self.merged,

            "rank":
                self.rank,

            "alpha":
                self.alpha,

            "scaling":
                self.scaling,
        }

    def parameter_summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "base_parameters":
                self.num_base_parameters(),

            "lora_parameters":
                self.num_lora_parameters(),

            "parameter_ratio":
                self.parameter_ratio(),

            "trainable_parameters":
                self.trainable_parameters(),

            "frozen_parameters":
                self.frozen_parameters(),
        }

    # ------------------------------------------------------------------
    # VALIDATION UTILITIES
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify_integrity(
        self,
    ) -> bool:
        """
        Research utility.

        Checks internal tensors
        for consistency.
        """

        tensors = [
            self.base_layer.weight,
            self.lora_A,
            self.lora_B,
        ]

        if (
            self.base_layer.bias
            is not None
        ):
            tensors.append(
                self.base_layer.bias
            )

        for tensor in tensors:

            if tensor.numel() == 0:
                return False

            if not torch.isfinite(
                tensor
            ).all():
                return False

        return True

    def validate_state(
        self,
    ) -> None:

        if self.rank <= 0:
            raise RuntimeError(
                "Invalid rank."
            )

        if self.scaling <= 0:
            raise RuntimeError(
                "Invalid scaling."
            )

        if (
            self.lora_A.shape
            !=
            (
                self.rank,
                self.in_features,
            )
        ):
            raise RuntimeError(
                "LoRA A shape corruption."
            )

        if (
            self.lora_B.shape
            !=
            (
                self.out_features,
                self.rank,
            )
        ):
            raise RuntimeError(
                "LoRA B shape corruption."
            )

        if not self.verify_integrity():
            raise RuntimeError(
                "LoRA integrity check failed."
            )

    # ------------------------------------------------------------------
    # DEVICE HELPERS
    # ------------------------------------------------------------------

    @property
    def device(
        self,
    ) -> torch.device:

        return (
            self.lora_A.device
        )

    @property
    def dtype(
        self,
    ) -> torch.dtype:

        return (
            self.lora_A.dtype
        )

    def same_device_as_base(
        self,
    ) -> bool:

        return (
            self.base_layer.weight.device
            ==
            self.lora_A.device
            ==
            self.lora_B.device
        )

    def same_dtype_as_base(
        self,
    ) -> bool:

        return (
            self.base_layer.weight.dtype
            ==
            self.lora_A.dtype
            ==
            self.lora_B.dtype
        )
    

    # ------------------------------------------------------------------
    # SERIALIZATION HELPERS
    # ------------------------------------------------------------------

    def export_configuration(
        self,
    ) -> Dict[str, Any]:
        """
        Export LoRA configuration.

        Useful for:

        - checkpoints
        - experiment tracking
        - hyperparameter sweeps
        - DE optimization
        """

        return {
            "rank":
                self.rank,

            "alpha":
                self.alpha,

            "dropout":
                self.config.dropout,

            "use_bias":
                self.config.use_bias,

            "merge_weights":
                self.config.merge_weights,

            "enabled":
                self.enabled,
        }

    def export_state(
        self,
    ) -> Dict[str, Any]:
        """
        Export complete runtime state.
        """

        return {
            "configuration":
                self.export_configuration(),

            "statistics":
                self.statistics().to_dict(),

            "metadata":
                self.metadata(),
        }
    
    @torch.no_grad()
    def restore_base_weight(
        self,
    ) -> None:

        if self.merged:
            raise RuntimeError(
                "Cannot restore base weight while merged."
            )
        
        if self.base_layer.weight.shape != self._base_weight_snapshot.shape:
            raise RuntimeError(
                "Base weight snapshot mismatch."
            )

        self.base_layer.weight.copy_(
            self._base_weight_snapshot
        )

    # ------------------------------------------------------------------
    # RESET HELPERS
    # ------------------------------------------------------------------

    def reset_statistics(
        self,
    ) -> None:
        """
        Reset runtime counters.
        """

        self.forward_calls = 0
        self.merge_count = 0
        self.unmerge_count = 0

        self.enable_count = 0
        self.disable_count = 0

    @torch.no_grad()
    def reset(
        self,
    ) -> None:
        """
        Complete reset.

        Useful for:

        - rank ablations
        - hyperparameter sweeps
        - reproducibility studies
        """

        if self.merged:
            self.unmerge()

        self.reset_lora_parameters()

        self.reset_statistics()

        self._cached_merge_delta = None

        self._merge_checksum.zero_()

        self.enabled = bool(
            self.config.enabled
        )

    # ------------------------------------------------------------------
    # RESEARCH DIAGNOSTICS
    # ------------------------------------------------------------------

    @torch.no_grad()
    def lora_weight_norm(
        self,
    ) -> float:
        """
        ||BA||
        """

        delta = (
            self.lora_B
            @
            self.lora_A
        )

        return float(
            torch.norm(
                delta
            ).detach().cpu()
        )

    @torch.no_grad()
    def lora_a_norm(
        self,
    ) -> float:

        return float(
            torch.norm(
                self.lora_A
            ).detach().cpu()
        )

    @torch.no_grad()
    def lora_b_norm(
        self,
    ) -> float:

        return float(
            torch.norm(
                self.lora_B
            ).detach().cpu()
        )

    @torch.no_grad()
    def adaptation_strength(
        self,
    ) -> float:
        """
        Effective adaptation magnitude.

        Useful for:

        - scaling studies
        - adaptation analysis
        """

        return float(
            self.scaling
            *
            self.lora_weight_norm()
        )

    @torch.no_grad()
    def diagnostics(
        self,
    ) -> Dict[str, float]:

        return {
            "lora_A_norm":
                self.lora_a_norm(),

            "lora_B_norm":
                self.lora_b_norm(),

            "lora_weight_norm":
                self.lora_weight_norm(),

            "adaptation_strength":
                self.adaptation_strength(),

            "merge_checksum":
                float(
                    self._merge_checksum.item()
                ),
        }

    # ------------------------------------------------------------------
    # MERGE VALIDATION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def can_merge(
        self,
    ) -> bool:
        """
        Check whether merge is safe.
        """

        if self.merged:
            return False

        if not self.verify_integrity():
            return False

        return True

    @torch.no_grad()
    def can_unmerge(
        self,
    ) -> bool:
        """
        Check whether unmerge is safe.
        """

        return self.merged

    # ------------------------------------------------------------------
    # PARAMETER EFFICIENCY REPORT
    # ------------------------------------------------------------------

    def efficiency_report(
        self,
    ) -> Dict[str, Any]:
        """
        Paper-oriented efficiency report.

        Corresponds to:

        Parameter Reduction Ratio
        Trainable Parameter Count
        Adaptation Efficiency
        """

        base = (
            self.num_base_parameters()
        )

        lora = (
            self.num_lora_parameters()
        )

        reduction_ratio = (
            float(base / lora)
            if lora > 0
            else float("inf")
        )

        return {
            "base_parameters":
                base,

            "lora_parameters":
                lora,

            "parameter_ratio":
                self.parameter_ratio(),

            "parameter_reduction_ratio":
                reduction_ratio,

            "trainable_parameters":
                self.trainable_parameters(),

            "rank":
                self.rank,

            "alpha":
                self.alpha,
        }
    
    def lora_state_dict(
        self,
    ) -> Dict[str, Tensor]:

        return {
            "lora_A":
                self.lora_A.detach().cpu(),

            "lora_B":
                self.lora_B.detach().cpu(),
        }

    @torch.no_grad()
    def load_lora_state_dict(
        self,
        state: Dict[str, Tensor],
    ) -> None:

        if "lora_A" not in state:
            raise KeyError(
                "Missing lora_A."
            )

        if "lora_B" not in state:
            raise KeyError(
                "Missing lora_B."
            )

        self.lora_A.copy_(
            state["lora_A"].to(
                device=self.lora_A.device,
                dtype=self.lora_A.dtype,
            )
        )

        self.lora_B.copy_(
            state["lora_B"].to(
                device=self.lora_B.device,
                dtype=self.lora_B.dtype,
            )
        )


    # ------------------------------------------------------------------
    # STRING REPRESENTATION
    # ------------------------------------------------------------------

    def extra_repr(
        self,
    ) -> str:

        return (
            f"rank={self.rank}, "
            f"alpha={self.alpha:.4f}, "
            f"scaling={self.scaling:.6f}, "
            f"enabled={self.enabled}, "
            f"merged={self.merged}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}"
        )

    # ------------------------------------------------------------------
    # PYTORCH STATE HOOKS
    # ------------------------------------------------------------------

    # def train(
    #     self,
    #     mode: bool = True,
    # ):
    #     super().train(mode)
    #     return self

    # def eval(
    #     self,
    # ):
    #     super().eval()
    #     return self

    # ------------------------------------------------------------------
    # FACTORY
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> "LoRALayer":
        """
        Convenience constructor.
        """

        config = LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        return cls(
            base_layer=layer,
            config=config,
        )


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "LoRAConfig",
    "LoRAStatistics",
    "LoRALayer",
]
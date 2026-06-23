"""
High-level Neural ODE refinement block for event-triggered neuromorphic learning.

Architecture
------------
Input
  ↓
Input Projection
  ↓
Vector Field
  ↓
ODE Solver
  ↓
Output Projection
  ↓
Residual Refinement
  ↓
Refined Output

This module intentionally delegates:

- continuous dynamics          -> vector_field.py
- numerical integration        -> numerical_methods.py
- solver orchestration         -> ode_solver.py

The ODE block serves as the bridge between model representations and the
continuous-time refinement subsystem.

Paper alignment
---------------
h0 = Projection(x)

dh/dt = f_theta(h, t)

hT = ODESolver.integrate(...)

y = OutputProjection(hT)

y = x + y    (optional residual refinement)

Designed for:
- event-triggered refinement
- solver benchmarking
- ablation studies
- stability analysis
- metrics integration
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn

from .ode_solver import (
    AdaptiveRK4Solver,
    EulerSolver,
    IntegrationResult,
    RK2Solver,
    RK4Solver,
)

from .vector_field import (
    BaseVectorField,
    VectorFieldConfig,
    VectorFieldFactory,
)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ODEBlockConfig:
    """
    Configuration for Neural ODE refinement block.
    """

    hidden_dim: int
    ode_hidden_dim: int

    integration_time: float = 1.0

    solver_type: str = "rk4"
    adaptive: bool = False

    dt: float = 0.05
    min_dt: float = 1e-4
    max_dt: float = 0.1

    store_trajectory: bool = False

    residual_connection: bool = True
    projection_bias: bool = True

    vector_field_hidden_dim: int = 128
    vector_field_layers: int = 2

    use_time: bool = True
    residual_vector_field: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.ode_hidden_dim <= 0:
            raise ValueError("ode_hidden_dim must be positive.")

        if self.integration_time <= 0:
            raise ValueError("integration_time must be positive.")

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        if self.min_dt <= 0:
            raise ValueError("min_dt must be positive.")

        if self.max_dt <= 0:
            raise ValueError("max_dt must be positive.")

        if self.max_dt < self.min_dt:
            raise ValueError(
                "max_dt must be greater than or equal to min_dt."
            )

        if self.vector_field_hidden_dim <= 0:
            raise ValueError(
                "vector_field_hidden_dim must be positive."
            )

        if self.vector_field_layers <= 0:
            raise ValueError(
                "vector_field_layers must be positive."
            )

        solver = self.solver_type.lower()

        if solver not in {"euler", "rk2", "rk4"}:
            raise ValueError(
                f"Unsupported solver_type '{self.solver_type}'. "
                "Supported: euler, rk2, rk4."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# BLOCK STATISTICS
# ============================================================================


@dataclass
class ODEBlockStatistics:
    """
    Running averages accumulated across ODE refinements.

    Paper-aligned metrics:
    - Solver Step Count
    - Function Evaluations
    - ODE Computational Cost
    - Latent State Smoothness
    """

    forward_calls: int = 0
    ode_activations: int = 0

    average_integration_time: float = 0.0
    average_steps: float = 0.0
    average_function_evaluations: float = 0.0

    average_state_norm: float = 0.0
    average_step_norm: float = 0.0

    average_state_change: float = 0.0

    def update(
        self,
        *,
        integration_time: float,
        steps: int,
        function_evaluations: int,
        state_norm: float,
        step_norm: float,
        state_change: float,
    ) -> None:

        self.forward_calls += 1
        self.ode_activations += 1

        n = float(self.forward_calls)

        self.average_integration_time += (
            integration_time -
            self.average_integration_time
        ) / n

        self.average_steps += (
            steps -
            self.average_steps
        ) / n

        self.average_function_evaluations += (
            function_evaluations -
            self.average_function_evaluations
        ) / n

        self.average_state_norm += (
            state_norm -
            self.average_state_norm
        ) / n

        self.average_step_norm += (
            step_norm -
            self.average_step_norm
        ) / n

        self.average_state_change += (
            state_change -
            self.average_state_change
        ) / n

    def reset(self) -> None:
        self.forward_calls = 0
        self.ode_activations = 0

        self.average_integration_time = 0.0
        self.average_steps = 0.0
        self.average_function_evaluations = 0.0

        self.average_state_norm = 0.0
        self.average_step_norm = 0.0

        self.average_state_change = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# ODE BLOCK
# ============================================================================


class ODEBlock(nn.Module):
    """
    High-level Neural ODE refinement module.

    Responsibilities
    ----------------
    1. latent projection
    2. vector field creation
    3. solver orchestration
    4. latent refinement
    5. diagnostics exposure
    """

    def __init__(
        self,
        config: ODEBlockConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Linear(
            config.hidden_dim,
            config.ode_hidden_dim,
            bias=config.projection_bias,
        )

        self.output_projection = nn.Linear(
            config.ode_hidden_dim,
            config.hidden_dim,
            bias=config.projection_bias,
        )

        self.vector_field = self._create_vector_field()

        self.solver = self._create_solver()

        self.statistics = ODEBlockStatistics()

    # ---------------------------------------------------------------------
    # Creation helpers
    # ---------------------------------------------------------------------

    def _create_vector_field(self) -> BaseVectorField:

        field_config = VectorFieldConfig(
            input_dim=self.config.ode_hidden_dim,
            hidden_dim=self.config.vector_field_hidden_dim,
            output_dim=self.config.ode_hidden_dim,
            num_layers=self.config.vector_field_layers,
            use_time=self.config.use_time,
            residual=self.config.residual_vector_field,
            name=(
                "residual"
                if self.config.residual_vector_field
                else "mlp"
            ),
        )

        return VectorFieldFactory.from_config(
            field_config
        )

    def _create_solver(self):

        solver_name = self.config.solver_type.lower()

        if self.config.adaptive:

            if solver_name != "rk4":
                raise ValueError(
                    "Adaptive integration currently "
                    "requires solver_type='rk4'."
                )

            return AdaptiveRK4Solver(
                dt=self.config.dt,
                min_dt=self.config.min_dt,
                max_dt=self.config.max_dt,
                store_trajectory=self.config.store_trajectory,
            )

        if solver_name == "euler":
            return EulerSolver(
                dt=self.config.dt,
                store_trajectory=self.config.store_trajectory,
            )

        if solver_name == "rk2":
            return RK2Solver(
                dt=self.config.dt,
                store_trajectory=self.config.store_trajectory,
            )

        if solver_name == "rk4":
            return RK4Solver(
                dt=self.config.dt,
                store_trajectory=self.config.store_trajectory,
            )

        raise ValueError(
            f"Unsupported solver type: {solver_name}"
        )

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def _validate_input(
        self,
        x: Tensor,
    ) -> None:

        if not isinstance(x, Tensor):
            raise TypeError(
                "Input must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "Expected input shape "
                "[batch_size, hidden_dim]."
            )

        if x.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.config.hidden_dim}, "
                f"got {x.shape[-1]}."
            )

        if x.numel() == 0:
            raise ValueError(
                "Input tensor must be non-empty."
            )

        if not torch.isfinite(x).all():
            raise FloatingPointError(
                "Input contains non-finite values."
            )

    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------

    def _update_statistics(
        self,
        result: IntegrationResult,
    ) -> None:

        solver_stats = result.statistics
        diagnostics = self.solver.get_diagnostics()

        self.statistics.update(
            integration_time=solver_stats.integration_time,
            steps=solver_stats.num_steps,
            function_evaluations=(
                solver_stats.function_evaluations
            ),
            state_norm=diagnostics.mean_state_norm,
            step_norm=diagnostics.mean_step_norm,
            state_change=diagnostics.average_state_change,
        )

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:

        solver_metadata = {}

        if hasattr(self.solver, "metadata"):
            solver_metadata = self.solver.metadata()

        return {
            "module": "ODEBlock",

            "configuration":
                self.config.to_dict(),

            "vector_field":
                self.vector_field.field_stats(),

            "solver":
                solver_metadata,

            "statistics":
                self.statistics.to_dict(),

            "metrics_ready": True,
            "stability_ready": True,
            "ode_block_ready": True,
        }

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------

    def reset_statistics(self) -> None:

        self.statistics.reset()

        if hasattr(self.solver, "reset_statistics"):
            self.solver.reset_statistics()

        if hasattr(self.vector_field, "reset_stats"):
            self.vector_field.reset_stats()

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_diagnostics: bool = False,
    ) -> Union[
        Tensor,
        Dict[str, Any],
    ]:

        self._validate_input(x)

        h0 = self.input_projection(x)

        if h0.shape[-1] != self.config.ode_hidden_dim:
            raise RuntimeError(
                "Input projection produced "
                "invalid latent dimension."
            )

        if not torch.isfinite(h0).all():
            raise FloatingPointError(
                "Projected latent state contains "
                "non-finite values."
            )

        result = self.solver.integrate(
            vector_field=self.vector_field,
            initial_state=h0,
            t0=0.0,
            t1=self.config.integration_time,
        )

        hT = result.final_state

        if hT.shape != h0.shape:
            raise RuntimeError(
                "ODE solver returned incompatible "
                "latent shape."
            )

        if not torch.isfinite(hT).all():
            raise FloatingPointError(
                "ODE solver returned non-finite values."
            )

        y = self.output_projection(hT)

        if y.shape != x.shape:
            raise RuntimeError(
                "Output projection shape mismatch."
            )

        if self.config.residual_connection:
            y = x + y

        if not torch.isfinite(y).all():
            raise FloatingPointError(
                "ODE block produced non-finite output."
            )

        self._update_statistics(result)

        if not return_diagnostics:
            return y

        return {
            "output": y,

            "integration_result":
                result,

            "solver_statistics":
                result.statistics.to_dict(),

            "solver_diagnostics":
                self.solver.get_diagnostics().to_dict(),

            "block_statistics":
                self.statistics.to_dict(),
        }


__all__ = [
    "ODEBlockConfig",
    "ODEBlockStatistics",
    "ODEBlock",
]
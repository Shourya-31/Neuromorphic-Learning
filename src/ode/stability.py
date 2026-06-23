"""
Stability analysis framework for Neural ODE refinement.

This module provides research-oriented diagnostics for evaluating the
continuous-time dynamics used within the event-triggered neuromorphic
learning architecture.

The implementation intentionally does NOT:

- implement vector fields
- implement numerical integration
- implement ODE solvers
- modify latent trajectories

Instead, it analyzes already-existing components from:

    src.ode.vector_field
    src.ode.ode_solver
    src.ode.ode_block

Supported analyses
------------------
1. Jacobian diagnostics
2. Lipschitz estimation
3. Trajectory stability
4. Perturbation robustness
5. Solver stability diagnostics

Methodology Alignment
---------------------
Continuous dynamics:

    dh/dt = f_theta(h,t)

Jacobian:

    J_f = ∂f_theta / ∂h

Lipschitz estimate:

    ||f(h+ε)-f(h)|| / ||ε||

Trajectory stability:

    robustness of latent evolution

Designed for:
    - paper experiments
    - ablation studies
    - solver comparisons
    - visualization pipelines
    - research reporting
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import torch

from torch import Tensor

from .vector_field import BaseVectorField
from .ode_solver import IntegrationResult
from .ode_solver import SolverStatistics
from .ode_block import ODEBlock


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class StabilityConfig:
    """
    Configuration for stability analysis.

    Controls computational cost of Jacobian,
    Lipschitz and trajectory diagnostics.
    """

    jacobian_samples: int = 8

    finite_difference_eps: float = 1e-4

    trajectory_tolerance: float = 1e-3

    lipschitz_samples: int = 32

    perturbation_scale: float = 1e-3

    max_batch_size: int = 128

    enable_jacobian: bool = True

    enable_lipschitz: bool = True

    enable_trajectory: bool = True

    enable_solver_metrics: bool = True

    def __post_init__(self) -> None:

        if self.jacobian_samples <= 0:
            raise ValueError(
                "jacobian_samples must be positive."
            )

        if self.finite_difference_eps <= 0:
            raise ValueError(
                "finite_difference_eps must be positive."
            )

        if self.trajectory_tolerance <= 0:
            raise ValueError(
                "trajectory_tolerance must be positive."
            )

        if self.lipschitz_samples <= 0:
            raise ValueError(
                "lipschitz_samples must be positive."
            )

        if self.perturbation_scale <= 0:
            raise ValueError(
                "perturbation_scale must be positive."
            )

        if self.max_batch_size <= 0:
            raise ValueError(
                "max_batch_size must be positive."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# JACOBIAN METRICS
# ============================================================================


@dataclass
class JacobianMetrics:
    """
    Jacobian stability diagnostics.

    Captures local sensitivity of the
    continuous-time vector field.
    """

    mean_jacobian_norm: float = 0.0

    max_jacobian_norm: float = 0.0

    min_jacobian_norm: float = 0.0

    spectral_radius_estimate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# LIPSCHITZ METRICS
# ============================================================================


@dataclass
class LipschitzMetrics:
    """
    Empirical Lipschitz estimates.

    Used to evaluate local robustness
    of the latent dynamics.
    """

    mean_lipschitz: float = 0.0

    max_lipschitz: float = 0.0

    min_lipschitz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# TRAJECTORY METRICS
# ============================================================================


@dataclass
class TrajectoryMetrics:
    """
    Stability metrics computed from an
    already-integrated trajectory.
    """

    trajectory_length: float = 0.0

    average_state_change: float = 0.0

    max_state_change: float = 0.0

    trajectory_variance: float = 0.0

    stable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SOLVER STABILITY METRICS
# ============================================================================


@dataclass
class SolverStabilityMetrics:
    """
    Stability diagnostics extracted from
    solver metadata.

    Supports both fixed-step and
    adaptive solvers.
    """

    accepted_steps: int = 0

    rejected_steps: int = 0

    acceptance_rate: float = 0.0

    mean_step_size: float = 0.0

    step_size_variance: float = 0.0

    mean_error: float = 0.0

    max_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# STABILITY REPORT
# ============================================================================


@dataclass
class StabilityReport:
    """
    Unified research report produced
    by StabilityAnalyzer.
    """

    jacobian_metrics: Optional[
        JacobianMetrics
    ] = None

    lipschitz_metrics: Optional[
        LipschitzMetrics
    ] = None

    trajectory_metrics: Optional[
        TrajectoryMetrics
    ] = None

    solver_metrics: Optional[
        SolverStabilityMetrics
    ] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "jacobian_metrics":
                None
                if self.jacobian_metrics is None
                else self.jacobian_metrics.to_dict(),

            "lipschitz_metrics":
                None
                if self.lipschitz_metrics is None
                else self.lipschitz_metrics.to_dict(),

            "trajectory_metrics":
                None
                if self.trajectory_metrics is None
                else self.trajectory_metrics.to_dict(),

            "solver_metrics":
                None
                if self.solver_metrics is None
                else self.solver_metrics.to_dict(),
        }


# ============================================================================
# STABILITY ANALYZER
# ============================================================================


class StabilityAnalyzer:
    """
    Primary stability analysis framework.

    Provides:

    - Jacobian analysis
    - Lipschitz analysis
    - Trajectory diagnostics
    - Solver diagnostics
    - ODE block integration

    This class never modifies model
    parameters or trajectories.
    """

    def __init__(
        self,
        config: StabilityConfig,
    ) -> None:

        self.config = config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tensor(
        self,
        tensor: Tensor,
        name: str = "tensor",
    ) -> None:

        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{name} must be a Tensor."
            )

        if tensor.numel() == 0:
            raise ValueError(
                f"{name} must be non-empty."
            )

        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"{name} contains non-finite values."
            )

    def _validate_vector_field(
        self,
        vector_field: BaseVectorField,
    ) -> None:
        

        if not isinstance(
            vector_field,
            BaseVectorField,
        ):
            raise TypeError(
                "vector_field must inherit "
                "from BaseVectorField."
            )

    def _validate_integration_result(
        self,
        result: IntegrationResult,
    ) -> None:

        if not isinstance(
            result,
            IntegrationResult,
        ):
            raise TypeError(
                "Expected IntegrationResult."
            )

        self._validate_tensor(
            result.final_state,
            "final_state",
        )

    # ------------------------------------------------------------------
    # Jacobian Estimation
    # ------------------------------------------------------------------

    def estimate_jacobian_norm(
        self,
        vector_field: BaseVectorField,
        state: Tensor,
        t: Union[float, Tensor] = 0.0,
    ) -> float:

        self._validate_vector_field(
            vector_field
        )

        self._validate_tensor(
            state,
            "state",
        )

        state = (
            state.detach()
            .clone()
            .requires_grad_(True)
        )

        output = vector_field(
            state,
            t,
        )

        if output.ndim > 1:
            output = output.sum(
                dim=tuple(
                    range(
                        1,
                        output.ndim
                    )
                )
            )

        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=state,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        norm = torch.norm(
            gradients
        )

        return float(
            norm.detach().cpu()
        )

    # ------------------------------------------------------------------
    # Jacobian Analysis
    # ------------------------------------------------------------------

    def analyze_jacobian(
        self,
        vector_field: BaseVectorField,
        state: Tensor,
        t: Union[float, Tensor] = 0.0,
    ) -> JacobianMetrics:
        """
        Compute Jacobian stability metrics.

        Uses multiple perturbation samples
        around a reference latent state.
        """

        self._validate_vector_field(
            vector_field
        )

        self._validate_tensor(
            state,
            "state",
        )

        norms = []

        spectral_estimates = []

        for _ in range(
            self.config.jacobian_samples
        ):

            noise = (
                torch.randn_like(state)
                * self.config.perturbation_scale
            )

            sample = state + noise

            jac_norm = (
                self.estimate_jacobian_norm(
                    vector_field,
                    sample,
                    t,
                )
            )

            norms.append(jac_norm)

            spectral_estimates.append(
                jac_norm
            )

        if not norms:

            return JacobianMetrics()

        return JacobianMetrics(
            mean_jacobian_norm=float(
                sum(norms) / len(norms)
            ),

            max_jacobian_norm=float(
                max(norms)
            ),

            min_jacobian_norm=float(
                min(norms)
            ),

            spectral_radius_estimate=float(
                max(spectral_estimates)
            ),
        )

    # ------------------------------------------------------------------
    # Lipschitz Estimation
    # ------------------------------------------------------------------

    def estimate_lipschitz_constant(
        self,
        vector_field: BaseVectorField,
        state: Tensor,
        t: Union[float, Tensor] = 0.0,
    ) -> LipschitzMetrics:
        """
        Empirical Lipschitz analysis.

        Estimates:

            ||f(h+e)-f(h)|| / ||e||

        over multiple perturbation
        samples.
        """

        self._validate_vector_field(
            vector_field
        )

        self._validate_tensor(
            state,
            "state",
        )

        with torch.no_grad():

            reference = vector_field(
                state,
                t,
            )

            values = []

            for _ in range(
                self.config.lipschitz_samples
            ):

                epsilon = (
                    torch.randn_like(state)
                    * self.config.perturbation_scale
                )

                perturbed_state = (
                    state + epsilon
                )

                perturbed_output = (
                    vector_field(
                        perturbed_state,
                        t,
                    )
                )

                numerator = (
                    torch.norm(
                        perturbed_output
                        - reference
                    )
                )

                denominator = (
                    torch.norm(
                        epsilon
                    )
                    + 1e-12
                )

                value = (
                    numerator
                    / denominator
                )

                values.append(
                    float(
                        value
                        .detach()
                        .cpu()
                    )
                )

        if not values:

            return LipschitzMetrics()

        return LipschitzMetrics(
            mean_lipschitz=float(
                sum(values)
                / len(values)
            ),

            max_lipschitz=float(
                max(values)
            ),

            min_lipschitz=float(
                min(values)
            ),
        )

    # ------------------------------------------------------------------
    # Trajectory Analysis
    # ------------------------------------------------------------------

    def analyze_trajectory(
        self,
        result: IntegrationResult,
    ) -> TrajectoryMetrics:
        """
        Analyze already integrated
        latent trajectories.

        Uses solver-produced trajectory.

        Never reintegrates.
        """

        self._validate_integration_result(
            result
        )

        trajectory = result.trajectory

        if trajectory is None:

            return TrajectoryMetrics(
                stable=True
            )

        if trajectory.shape[0] < 2:

            return TrajectoryMetrics(
                stable=True
            )

        diffs = (
            trajectory[1:]
            - trajectory[:-1]
        )

        step_norms = torch.norm(
            diffs.reshape(
                diffs.shape[0],
                -1,
            ),
            dim=1,
        )

        trajectory_length = float(
            step_norms.sum()
            .detach()
            .cpu()
        )

        avg_change = float(
            step_norms.mean()
            .detach()
            .cpu()
        )

        max_change = float(
            step_norms.max()
            .detach()
            .cpu()
        )

        variance = float(
            trajectory.var()
            .detach()
            .cpu()
        )

        stable = (
            variance
            <
            self.config.trajectory_tolerance
        )

        return TrajectoryMetrics(
            trajectory_length=
                trajectory_length,

            average_state_change=
                avg_change,

            max_state_change=
                max_change,

            trajectory_variance=
                variance,

            stable=
                stable,
        )

    # ------------------------------------------------------------------
    # Solver Analysis
    # ------------------------------------------------------------------

    def analyze_solver(
        self,
        result: IntegrationResult,
    ) -> SolverStabilityMetrics:
        """
        Analyze solver behavior.

        Supports:

        - fixed-step solvers
        - adaptive RK4 solver

        Uses existing metadata.
        """

        self._validate_integration_result(
            result
        )

        stats: SolverStatistics = (
            result.statistics
        )

        metadata = result.metadata



        diagnostics = metadata.get(
            "diagnostics",
            {},
        )

        adaptive_stats = metadata.get(
            "adaptive_statistics",
            {},
        )

        accepted_steps = int(
            stats.accepted_steps
        )

        rejected_steps = int(
            stats.rejected_steps
        )

        total_steps = max(
            1,
            stats.num_steps,
        )

        acceptance_rate = (
            accepted_steps
            / total_steps
        )

        mean_step_size = float(
            stats.average_dt
        )

        dt_history = adaptive_stats.get(
            "dt_history",
            None,
        )

        if (
            dt_history is not None
            and len(dt_history) > 1
        ):
            step_size_variance = float(
                torch.tensor(
                    dt_history,
                    dtype=torch.float32,
                ).var(
                    unbiased=False
                ).item()
            )
        else:
            step_size_variance = 0.0

        mean_error = float(
            adaptive_stats.get(
                "average_error",
                0.0,
            )
        )

        max_error = float(
            adaptive_stats.get(
                "maximum_error",
                0.0,
            )
        )

        return SolverStabilityMetrics(
            accepted_steps=
                accepted_steps,

            rejected_steps=
                rejected_steps,

            acceptance_rate=
                acceptance_rate,

            mean_step_size=
                mean_step_size,

            step_size_variance=
                step_size_variance,

            mean_error=
                mean_error,

            max_error=
                max_error,
        )

    # ------------------------------------------------------------------
    # Full Stability Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        vector_field: BaseVectorField,
        integration_result: IntegrationResult,
        reference_state: Optional[
            Tensor
        ] = None,
        t: Union[
            float,
            Tensor,
        ] = 0.0,
    ) -> StabilityReport:
        """
        Primary research API.

        Executes selected analyses
        according to configuration.

        Workflow
        --------

        Jacobian Analysis
              ↓

        Lipschitz Analysis
              ↓

        Trajectory Analysis
              ↓

        Solver Analysis
              ↓

        StabilityReport
        """

        self._validate_vector_field(
            vector_field
        )

        self._validate_integration_result(
            integration_result
        )

        if reference_state is None:

            reference_state = (
                integration_result
                .final_state
            )

        self._validate_tensor(
            reference_state,
            "reference_state",
        )

        jacobian_metrics = None

        lipschitz_metrics = None

        trajectory_metrics = None

        solver_metrics = None

        if self.config.enable_jacobian:

            jacobian_metrics = (
                self.analyze_jacobian(
                    vector_field=
                        vector_field,

                    state=
                        reference_state,

                    t=t,
                )
            )

        if self.config.enable_lipschitz:

            lipschitz_metrics = (
                self
                .estimate_lipschitz_constant(
                    vector_field=
                        vector_field,

                    state=
                        reference_state,

                    t=t,
                )
            )

        if self.config.enable_trajectory:

            trajectory_metrics = (
                self
                .analyze_trajectory(
                    integration_result
                )
            )

        if self.config.enable_solver_metrics:

            solver_metrics = (
                self.analyze_solver(
                    integration_result
                )
            )

        return StabilityReport(
            jacobian_metrics=
                jacobian_metrics,

            lipschitz_metrics=
                lipschitz_metrics,

            trajectory_metrics=
                trajectory_metrics,

            solver_metrics=
                solver_metrics,
        )

    # ------------------------------------------------------------------
    # ODE Block Integration
    # ------------------------------------------------------------------

    def analyze_block(
        self,
        ode_block: ODEBlock,
        integration_result: IntegrationResult,
        reference_state: Tensor,
    ) -> StabilityReport:
        """
        Convenience helper for
        direct ODEBlock analysis.

        Automatically extracts
        the vector field.
        """

        if not isinstance(
            ode_block,
            ODEBlock,
        ):
            raise TypeError(
                "Expected ODEBlock."
            )

        vector_field = (
            ode_block.vector_field
        )

        return self.analyze(
            vector_field=
                vector_field,

            integration_result=
                integration_result,

            reference_state=
                reference_state,
        )

    # ------------------------------------------------------------------
    # Robustness Analysis
    # ------------------------------------------------------------------

    def perturbation_robustness(
        self,
        vector_field: BaseVectorField,
        state: Tensor,
        t: Union[
            float,
            Tensor,
        ] = 0.0,
        samples: Optional[
            int
        ] = None,
    ) -> Dict[str, float]:
        """
        Additional research utility.

        Measures sensitivity of
        latent dynamics to small
        perturbations.

        Useful for:

        - robustness studies
        - ablation experiments
        - supplementary material
        """

        self._validate_vector_field(
            vector_field
        )

        self._validate_tensor(
            state,
            "state",
        )

        sample_count = (
            samples
            if samples is not None
            else self.config
            .lipschitz_samples
        )

        with torch.no_grad():

            reference = (
                vector_field(
                    state,
                    t,
                )
            )

            deviations = []

            for _ in range(
                sample_count
            ):

                noise = (
                    torch.randn_like(
                        state
                    )
                    *
                    self.config
                    .perturbation_scale
                )

                output = vector_field(
                    state + noise,
                    t,
                )

                deviation = (
                    torch.norm(
                        output
                        - reference
                    )
                )

                deviations.append(
                    float(
                        deviation
                        .detach()
                        .cpu()
                    )
                )

        if len(deviations) == 0:

            return {
                "mean_deviation":
                    0.0,

                "max_deviation":
                    0.0,

                "min_deviation":
                    0.0,
            }

        return {
            "mean_deviation":
                float(
                    sum(deviations)
                    /
                    len(deviations)
                ),

            "max_deviation":
                float(
                    max(deviations)
                ),

            "min_deviation":
                float(
                    min(deviations)
                ),
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(
        self,
    ) -> Dict[str, Any]:

        return {
            "module":
                "StabilityAnalyzer",

            "configuration":
                self.config.to_dict(),

            "jacobian_enabled":
                self.config.enable_jacobian,

            "lipschitz_enabled":
                self.config.enable_lipschitz,

            "trajectory_enabled":
                self.config.enable_trajectory,

            "solver_enabled":
                self.config.enable_solver_metrics,
        }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "StabilityConfig",

    "JacobianMetrics",

    "LipschitzMetrics",

    "TrajectoryMetrics",

    "SolverStabilityMetrics",

    "StabilityReport",

    "StabilityAnalyzer",
]
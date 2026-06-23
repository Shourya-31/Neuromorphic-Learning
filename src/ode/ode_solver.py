from __future__ import annotations
import math

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from .numerical_methods import (
    NumericalMethodBase,
    EulerStep,
    RK2Step,
    RK4Step,
    AdaptiveStepController,
)

VectorField = Callable[[Tensor, Any], Tensor]


@dataclass
class SolverStatistics:
    solver_name: str
    num_steps: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    function_evaluations: int = 0
    integration_time: float = 0.0
    final_dt: float = 0.0
    average_dt: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrationResult:
    final_state: Tensor
    trajectory: Optional[Tensor]
    times: Optional[Tensor]
    statistics: SolverStatistics
    metadata: Dict[str, Any]

@dataclass
class SolverDiagnostics:
    mean_state_norm: float = 0.0
    max_state_norm: float = 0.0

    mean_derivative_norm: float = 0.0
    max_derivative_norm: float = 0.0

    mean_step_norm: float = 0.0
    max_step_norm: float = 0.0

    average_state_change: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ODESolverBase(ABC):
    def __init__(self, method: NumericalMethodBase, store_trajectory: bool = False):
        self.method = method
        self.store_trajectory = store_trajectory
        self._stats = SolverStatistics(self.__class__.__name__)
        self._diagnostics = SolverDiagnostics()

    def get_diagnostics(self) -> SolverDiagnostics:
        return self._diagnostics

    @property
    def solver_name(self) -> str:
        return self.__class__.__name__

    def reset_statistics(self) -> None:
        self._stats = SolverStatistics(self.solver_name)
        self._diagnostics = SolverDiagnostics()

    def get_statistics(self) -> SolverStatistics:
        return self._stats

    def metadata(self) -> Dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "adaptive": getattr(self.method, "supports_adaptive", False),
            "store_trajectory": self.store_trajectory,
            "method": self.method.method_info(),
            "integration": self.method.integration_metadata(),

            "diagnostics": self._diagnostics.to_dict(),

            "metrics_ready": True,
            "stability_ready": True,
            "ode_block_ready": True,
        }

    @abstractmethod
    def integrate(self, vector_field: VectorField, initial_state: Tensor,
                  t0: float, t1: float, **kwargs: Any) -> IntegrationResult:
        ...


class FixedStepSolver(ODESolverBase):
    def __init__(self, method: NumericalMethodBase, dt: float,
                 store_trajectory: bool = False, max_steps: int = 1_000_000):
        super().__init__(method, store_trajectory)
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.dt = float(dt)
        self.max_steps = max_steps

    def _validate(self, state: Tensor, shape) -> None:
        if state.shape != shape:
            raise RuntimeError("State shape corruption detected.")
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite state encountered.")

    def integrate(self, vector_field, initial_state, t0, t1, **kwargs):
        if not math.isfinite(t0):
            raise ValueError("t0 must be finite.")

        if not math.isfinite(t1):
            raise ValueError("t1 must be finite.")

        if t1 <= t0:
            raise ValueError("t1 must be greater than t0.")

        self.reset_statistics()

        if initial_state.numel() == 0:
            raise ValueError(
                "initial_state must be non-empty."
            )

        if not torch.isfinite(initial_state).all():
            raise FloatingPointError(
                "initial_state contains non-finite values."
            )

        start = time.perf_counter()

        state = initial_state
        shape = state.shape
        t = float(t0)

        traj = [state.detach().clone()] if self.store_trajectory else None
        times = [t] if self.store_trajectory else None

        dt_sum = 0.0
        state_norms = []
        derivative_norms = []
        step_norms = []
        state_changes = []

        while t < t1:
            if self._stats.num_steps >= self.max_steps:
                raise RuntimeError("Maximum step count exceeded.")

            dt = min(self.dt, t1 - t)

            previous_state = state.detach().clone()

            state = self.method.step(
                vector_field,
                state,
                t,
                dt,
            )
        
            diag = getattr(self.method, "diagnostics", None)

            if diag is not None:
                state_norms.append(diag.state_norm)
                derivative_norms.append(diag.derivative_norm)
                step_norms.append(diag.step_norm)

            self._validate(state, shape)

            t += dt
            dt_sum += dt

            self._stats.num_steps += 1
            self._stats.accepted_steps += 1
            self._stats.function_evaluations += self.method.function_evaluations

            state_changes.append(
                float(
                    torch.norm(
                        state - previous_state
                    ).detach().cpu()
                )
            )

            if self.store_trajectory:
                traj.append(state.detach().clone())
                times.append(t)

        self._stats.integration_time = time.perf_counter() - start
        self._stats.final_dt = dt
        self._stats.average_dt = dt_sum / max(1, self._stats.num_steps)

        if state_norms:
            self._diagnostics.mean_state_norm = (
                sum(state_norms) / len(state_norms)
            )
            self._diagnostics.max_state_norm = max(state_norms)

        if derivative_norms:
            self._diagnostics.mean_derivative_norm = (
                sum(derivative_norms) / len(derivative_norms)
            )
            self._diagnostics.max_derivative_norm = max(derivative_norms)

        if step_norms:
            self._diagnostics.mean_step_norm = (
                sum(step_norms) / len(step_norms)
            )
            self._diagnostics.max_step_norm = max(step_norms)

        if state_changes:
            self._diagnostics.average_state_change = (
                sum(state_changes) / len(state_changes)
            )

        return IntegrationResult(
            final_state=state,
            trajectory=torch.stack(traj) if traj is not None else None,
            times=torch.tensor(times, dtype=state.dtype, device=state.device) if times is not None else None,
            statistics=self._stats,
            metadata=self.metadata(),
        )


class EulerSolver(FixedStepSolver):
    def __init__(self, dt: float, store_trajectory: bool = False):
        super().__init__(EulerStep(), dt, store_trajectory)

class RK2Solver(FixedStepSolver):
    """
    Fixed-step midpoint RK2 solver.
    """

    def __init__(
        self,
        dt: float,
        store_trajectory: bool = False,
        ):
            super().__init__(
                RK2Step(),
                dt,
                store_trajectory,
            )

class RK4Solver(FixedStepSolver):
    def __init__(self, dt: float, store_trajectory: bool = False):
        super().__init__(RK4Step(), dt, store_trajectory)


class AdaptiveRK4Solver(ODESolverBase):
    def __init__(
        self,
        dt: float = 1e-2,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        store_trajectory: bool = False,
        max_steps: int = 1_000_000,
    ):
        super().__init__(RK4Step(), store_trajectory)
        self.rk4 = RK4Step()
        self.rk2 = RK2Step()
        self.error_history = []
        self.dt_history = []
        self.controller = AdaptiveStepController(
            rtol=rtol, atol=atol, min_dt=min_dt, max_dt=max_dt
        )
        self.dt = dt
        self.max_steps = max_steps

    def integrate(self, vector_field, initial_state, t0, t1, **kwargs):
        self.reset_statistics()
        self.error_history.clear()
        self.dt_history.clear()

        if not math.isfinite(t0):
            raise ValueError("t0 must be finite.")

        if not math.isfinite(t1):
            raise ValueError("t1 must be finite.")

        if t1 <= t0:
            raise ValueError("t1 must be greater than t0.")

        if initial_state.numel() == 0:
            raise ValueError(
                "initial_state must be non-empty."
            )

        if not torch.isfinite(initial_state).all():
            raise FloatingPointError(
                "initial_state contains non-finite values."
            )

        start = time.perf_counter()

        state = initial_state
        shape = state.shape
        t = float(t0)
        dt = self.dt
        dt_sum = 0.0

        traj = [state.detach().clone()] if self.store_trajectory else None
        times = [t] if self.store_trajectory else None
        state_norms = []
        derivative_norms = []
        step_norms = []
        state_changes = []

        while t < t1:
            if self._stats.num_steps >= self.max_steps:
                raise RuntimeError("Maximum step count exceeded.")

            dt = min(dt, t1 - t)

            rk4_state = self.rk4.step(vector_field, state, t, dt)
            rk2_state = self.rk2.step(vector_field, state, t, dt)

            diag = self.rk4.diagnostics

            if diag is not None:
                state_norms.append(diag.state_norm)
                derivative_norms.append(diag.derivative_norm)
                step_norms.append(diag.step_norm)

            # self._diagnostics.mean_state_norm

            error = self.controller.compute_error(rk4_state, rk2_state)
            self.error_history.append(
                float(error.detach().cpu())
            )
            accepted = self.controller.accept_step(error)

            self._stats.num_steps += 1
            self._stats.function_evaluations += (
                self.rk4.function_evaluations + self.rk2.function_evaluations
            )

            if accepted:
                previous_state = state.detach().clone()
                if not torch.isfinite(rk4_state).all():
                    raise FloatingPointError("Non-finite state encountered.")
                if rk4_state.shape != shape:
                    raise RuntimeError("State shape corruption detected.")

                state = rk4_state

                state_changes.append(
                    float(
                        torch.norm(
                            state - previous_state
                        ).detach().cpu()
                    )
                )

                t += dt
                self.dt_history.append(dt)
                dt_sum += dt
                self._stats.accepted_steps += 1

                if self.store_trajectory:
                    traj.append(state.detach().clone())
                    times.append(t)
            else:
                self._stats.rejected_steps += 1

            dt = float(self.controller.suggest_dt(dt, error))

            if dt < self.controller.min_dt:
                raise RuntimeError("Adaptive step size collapsed below min_dt.")

        self._stats.integration_time = time.perf_counter() - start
        self._stats.final_dt = dt
        self._stats.average_dt = dt_sum / max(1, self._stats.accepted_steps)

        if state_norms:
            self._diagnostics.mean_state_norm = (
                sum(state_norms) / len(state_norms)
            )
            self._diagnostics.max_state_norm = max(state_norms)

        if derivative_norms:
            self._diagnostics.mean_derivative_norm = (
                sum(derivative_norms) /
                len(derivative_norms)
            )
            self._diagnostics.max_derivative_norm = (
                max(derivative_norms)
            )

        if step_norms:
            self._diagnostics.mean_step_norm = (
                sum(step_norms) /
                len(step_norms)
            )
            self._diagnostics.max_step_norm = (
                max(step_norms)
            )

        if state_changes:
            self._diagnostics.average_state_change = (
                sum(state_changes) /
                len(state_changes)
            )
        adaptive_stats = {
            "acceptance_rate":
                self._stats.accepted_steps /
                max(1, self._stats.num_steps),

            "rejection_rate":
                self._stats.rejected_steps /
                max(1, self._stats.num_steps),

            "average_error":
                sum(self.error_history) /
                max(1, len(self.error_history)),

            "maximum_error":
                max(self.error_history)
                if self.error_history else 0.0,

            "minimum_dt_used":
                min(self.dt_history)
                if self.dt_history else 0.0,

            "maximum_dt_used":
                max(self.dt_history)
                if self.dt_history else 0.0,

            "average_dt_used":
                sum(self.dt_history) /
                max(1, len(self.dt_history)),
        }
        return IntegrationResult(
            final_state=state,
            trajectory=torch.stack(traj) if traj is not None else None,
            times=torch.tensor(times, dtype=state.dtype, device=state.device) if times is not None else None,
            statistics=self._stats,
            metadata={
                **self.metadata(),
                "adaptive_controller": {
                    "rtol": self.controller.rtol,
                    "atol": self.controller.atol,
                    "min_dt": self.controller.min_dt,
                    "max_dt": self.controller.max_dt,
                },
                "adaptive_statistics": adaptive_stats,
            },
        )


__all__ = [
    "SolverStatistics",
    "SolverDiagnostics",
    "IntegrationResult",
    "ODESolverBase",
    "FixedStepSolver",
    "EulerSolver",
    "RK2Solver",
    "RK4Solver",
    "AdaptiveRK4Solver",
]
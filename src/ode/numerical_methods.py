r"""Numerical integration primitives for event-triggered Neural ODE refinement.

This module implements the low-level numerical step methods supporting the paper's
continuous-time refinement stage. It directly operationalizes:

- Equation (13): Euler step
    h_{k+1} = h_k + \Delta t f_\theta(h_k, t_k)
- Equation (14): general numerical approximation
    h(t_{k+1}) = h(t_k) + \int_{t_k}^{t_{k+1}} f_\theta(h(t), t) dt

The implementations below provide reusable local integration steps only. Solver
orchestration, full ODE loops, and higher-level integration pipelines are handled
elsewhere in the architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor


TimeLike = Union[float, int, Tensor]
VectorField = Callable[[Tensor, TimeLike], Tensor]


@dataclass
class StepDiagnostics:
    """
    Lightweight runtime diagnostics for numerical integration.

    Used by:
        - ode_metrics.py
        - stability.py
        - solver instrumentation

    Does NOT affect integration.
    """

    state_norm: float
    derivative_norm: float
    step_norm: float


    # def _compute_step_diagnostics(
    #     h: Tensor,
    #     dh: Tensor,
    #     h_next: Tensor,
    # ) -> StepDiagnostics:
    #     return StepDiagnostics(
    #         state_norm=float(torch.norm(h).detach().cpu()),
    #         derivative_norm=float(torch.norm(dh).detach().cpu()),
    #         step_norm=float(torch.norm(h_next - h).detach().cpu()),
    #     )
    

    # def integration_metadata(self) -> Dict[str, Any]:
    #     return {
    #         "method": self.method_name,
    #         "order": self.order,
    #         "supports_adaptive": self.supports_adaptive,
    #         "n_function_evals": self.n_function_evals,
    #     }


def _as_finite_tensor(value: Any, *, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} must be finite.")
    return tensor

def _compute_step_diagnostics(
        h: Tensor,
        dh: Tensor,
        h_next: Tensor,
    ) -> StepDiagnostics:
        return StepDiagnostics(
            state_norm=float(torch.norm(h).detach().cpu()),
            derivative_norm=float(torch.norm(dh).detach().cpu()),
            step_norm=float(torch.norm(h_next - h).detach().cpu()),
        )

def _validate_state(h: Tensor) -> None:
    if not isinstance(h, Tensor):
        raise TypeError("h must be a torch.Tensor.")
    if h.numel() == 0:
        raise ValueError("h must be non-empty.")
    if not torch.isfinite(h).all():
        raise RuntimeError("h must be finite.")


def _validate_callable(func: Any) -> None:
    if not callable(func):
        raise TypeError("func must be callable.")


def _validate_dt(dt: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    dt_tensor = _as_finite_tensor(dt, device=device, dtype=dtype, name="dt")
    if torch.any(dt_tensor <= 0):
        raise ValueError("dt must be strictly positive.")
    return dt_tensor


def _validate_time_like(t: TimeLike, *, device: torch.device, dtype: torch.dtype) -> TimeLike:
    if isinstance(t, Tensor):
        if not torch.isfinite(t).all():
            raise RuntimeError("t must be finite.")
        return t.to(device=device, dtype=dtype)
    if isinstance(t, (float, int)):
        t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
        if not torch.isfinite(t_tensor).all():
            raise RuntimeError("t must be finite.")
        return t_tensor
    return t


def _add_time(t: TimeLike, increment: Tensor) -> TimeLike:
    if isinstance(t, Tensor):
        return t + increment
    if isinstance(t, (float, int)):
        return type(t)(torch.as_tensor(t).item() + increment.item())  # preserve scalar type
    # Best-effort fallback for broadcastable tensor-like values.
    return t + increment


def _validate_step_output(output: Tensor, reference: Tensor, method_name: str) -> None:
    if not isinstance(output, Tensor):
        raise TypeError(f"{method_name} must return a torch.Tensor.")
    if output.shape != reference.shape:
        raise RuntimeError(
            f"{method_name} must preserve shape. Expected {tuple(reference.shape)}, got {tuple(output.shape)}."
        )
    if output.dtype != reference.dtype:
        raise RuntimeError(f"{method_name} must preserve dtype.")
    if output.device != reference.device:
        raise RuntimeError(f"{method_name} must preserve device.")
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{method_name} produced non-finite values.")


class NumericalMethodBase(ABC):
    """Abstract base class for local ODE integration steps.

    The subclasses in this module implement Equation (13), Equation (14), and
    standard higher-order one-step methods. This class is intentionally limited
    to local step computation; full solver orchestration is implemented elsewhere.
    """

    method_name: str
    order: int
    supports_adaptive: bool
    paper_equation: str
    recommended_use: str
    stability_category: str
    n_function_evals: int

    @property
    def diagnostics(self) -> Optional[StepDiagnostics]:
        return self._last_diagnostics

    def __init__(
        self,
        method_name: str,
        order: int,
        supports_adaptive: bool,
        *,
        paper_equation: str,
        recommended_use: str,
        stability_category: str,
        n_function_evals: int,
    ) -> None:

        self.method_name = method_name
        self.order = order
        self.supports_adaptive = supports_adaptive

        self.paper_equation = paper_equation
        self.recommended_use = recommended_use
        self.stability_category = stability_category
        self.n_function_evals = n_function_evals

        self._last_diagnostics: Optional[StepDiagnostics] = None

    @abstractmethod
    def step(self, func: VectorField, h: Tensor, t: TimeLike, dt: Any) -> Tensor:
        """Advance the latent state by one numerical integration step."""

    def extra_repr(self) -> str:
        return (
            f"name={self.method_name}, order={self.order}, "
            f"supports_adaptive={self.supports_adaptive}"
        )

    def method_info(self) -> Dict[str, Any]:
        return {
            "name": self.method_name,
            "order": self.order,
            "accuracy_order": self.order,
            "supports_adaptive": self.supports_adaptive,
            "paper_equation": self.paper_equation,
            "recommended_use": self.recommended_use,
            "stability_category": self.stability_category,
            "n_function_evals": self.n_function_evals,
        }
    
    def integration_metadata(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "order": self.order,
            "supports_adaptive": self.supports_adaptive,
            "n_function_evals": self.n_function_evals,
            "paper_equation": self.paper_equation,
            "stability_category": self.stability_category,
        }
    
    @property
    def function_evaluations(self) -> int:
        return self.n_function_evals

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.__class__.__name__}({self.extra_repr()})"


class EulerStep(NumericalMethodBase):
    r"""First-order Euler integrator implementing Equation (13).

    h_{k+1} = h_k + \Delta t f_\theta(h_k, t_k)
    """

    def __init__(self) -> None:
       super().__init__(
            method_name="euler",
            order=1,
            supports_adaptive=False,
            paper_equation="Equation (13)",
            recommended_use="Fast baseline",
            stability_category="Low",
            n_function_evals=1,
        )

    def step(self, func: VectorField, h: Tensor, t: TimeLike, dt: Any) -> Tensor:
        _validate_callable(func)
        _validate_state(h)
        dt_tensor = _validate_dt(dt, device=h.device, dtype=h.dtype)
        t_val = _validate_time_like(t, device=h.device, dtype=h.dtype)

        dh = func(h, t_val)
        _validate_step_output(dh, h, "EulerStep derivative")

        h_next = h + dt_tensor * dh
        _validate_step_output(h_next, h, "EulerStep output")
        self._last_diagnostics = _compute_step_diagnostics(
            h,
            dh,
            h_next,
        )

        return h_next


class RK2Step(NumericalMethodBase):
    """Second-order Runge-Kutta midpoint integrator.

    This is a standard higher-order approximation for Equation (14), using a
    midpoint evaluation to improve accuracy over Euler stepping.
    """

    def __init__(self) -> None:
        super().__init__(
            method_name="rk2",
            order=2,
            supports_adaptive=False,
            paper_equation="Equation (14)",
            recommended_use="Intermediate accuracy",
            stability_category="Medium",
            n_function_evals=2,
        )

    def step(self, func: VectorField, h: Tensor, t: TimeLike, dt: Any) -> Tensor:
        _validate_callable(func)
        _validate_state(h)
        dt_tensor = _validate_dt(dt, device=h.device, dtype=h.dtype)
        t_val = _validate_time_like(t, device=h.device, dtype=h.dtype)

        half_dt = dt_tensor * 0.5

        k1 = func(h, t_val)
        _validate_step_output(k1, h, "RK2Step k1")

        h_mid = h + half_dt * k1
        _validate_step_output(h_mid, h, "RK2Step midpoint state")

        t_mid = _add_time(t_val, half_dt)
        k2 = func(h_mid, t_mid)
        _validate_step_output(k2, h, "RK2Step k2")

        h_next = h + dt_tensor * k2
        _validate_step_output(h_next, h, "RK2Step output")
        self._last_diagnostics = _compute_step_diagnostics(
            h,
            k2,
            h_next,
        )
        return h_next


class RK4Step(NumericalMethodBase):
    """Classical fourth-order Runge-Kutta integrator.

    RK4 is the strongest fixed-step method included here and is suitable for the
    local integration backbone used by later solver orchestration. It is a
    standard higher-order method consistent with the general approximation in
    Equation (14).
    """

    def __init__(self) -> None:
        super().__init__(
            method_name="rk4",
            order=4,
            supports_adaptive=True,
            paper_equation="Equation (14)",
            recommended_use="Primary research solver",
            stability_category="High",
            n_function_evals=4,
        )

    def step(self, func: VectorField, h: Tensor, t: TimeLike, dt: Any) -> Tensor:
        _validate_callable(func)
        _validate_state(h)
        dt_tensor = _validate_dt(dt, device=h.device, dtype=h.dtype)
        t_val = _validate_time_like(t, device=h.device, dtype=h.dtype)

        half_dt = dt_tensor * 0.5

        k1 = func(h, t_val)
        _validate_step_output(k1, h, "RK4Step k1")

        h2 = h + half_dt * k1
        _validate_step_output(h2, h, "RK4Step h2")
        t2 = _add_time(t_val, half_dt)
        k2 = func(h2, t2)
        _validate_step_output(k2, h, "RK4Step k2")

        h3 = h + half_dt * k2
        _validate_step_output(h3, h, "RK4Step h3")
        k3 = func(h3, t2)
        _validate_step_output(k3, h, "RK4Step k3")

        h4 = h + dt_tensor * k3
        _validate_step_output(h4, h, "RK4Step h4")
        t4 = _add_time(t_val, dt_tensor)
        k4 = func(h4, t4)
        _validate_step_output(k4, h, "RK4Step k4")

        h_next = h + (dt_tensor / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _validate_step_output(h_next, h, "RK4Step output")
        effective_derivative = (
            k1 +
            2.0 * k2 +
            2.0 * k3 +
            k4
        ) / 6.0

        self._last_diagnostics = _compute_step_diagnostics(
            h,
            effective_derivative,
            h_next,
        )
        return h_next


@dataclass
class AdaptiveStepController:
    """Lightweight local step controller for future adaptive solvers.

    The controller estimates local error from the discrepancy between RK4 and
    RK2 states. It does not perform recursive integration or solver orchestration;
    it only assesses a step and suggests a next step size.
    """

    rtol: float = 1e-3
    atol: float = 1e-6
    min_dt: float = 1e-6
    max_dt: float = 1.0
    growth_factor: float = 2.0
    shrink_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative.")
        if self.atol < 0:
            raise ValueError("atol must be non-negative.")
        if self.min_dt <= 0:
            raise ValueError("min_dt must be > 0.")
        if self.max_dt <= self.min_dt:
            raise ValueError("max_dt must be greater than min_dt.")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.")
        if not (0.0 < self.shrink_factor < 1.0):
            raise ValueError("shrink_factor must satisfy 0 < shrink_factor < 1.")

    def compute_error(self, rk4_state: Tensor, rk2_state: Tensor) -> Tensor:
        _validate_state(rk4_state)
        _validate_state(rk2_state)
        if rk4_state.shape != rk2_state.shape:
            raise ValueError("rk4_state and rk2_state must have the same shape.")

        diff = torch.abs(rk4_state - rk2_state)
        scale = self.atol + self.rtol * torch.maximum(torch.abs(rk4_state), torch.abs(rk2_state))
        normalized = diff / scale.clamp_min(torch.finfo(rk4_state.dtype).tiny)
        error = torch.max(normalized)
        if not torch.isfinite(error):
            raise RuntimeError("Adaptive error estimate is non-finite.")
        return error

    def error_metadata(
    self,
    error: Tensor,
    ) -> Dict[str, Any]:

        error_value = float(error.detach().cpu())

        return {
            "error": error_value,
            "accepted": error_value <= 1.0,
            "rtol": self.rtol,
            "atol": self.atol,
        }

    def accept_step(self, error: Tensor) -> bool:
        error_tensor = torch.as_tensor(error)
        if not torch.isfinite(error_tensor).all():
            raise RuntimeError("Error estimate must be finite.")
        return bool(error_tensor <= 1.0)

    def suggest_dt(self, current_dt: Any, error: Tensor) -> Tensor:
        current_dt_tensor = torch.as_tensor(current_dt, dtype=torch.float64)
        if not torch.isfinite(current_dt_tensor).all():
            raise RuntimeError("current_dt must be finite.")
        if torch.any(current_dt_tensor <= 0):
            raise ValueError("current_dt must be strictly positive.")

        error_tensor = torch.as_tensor(error, dtype=torch.float64)
        if not torch.isfinite(error_tensor).all():
            raise RuntimeError("Error estimate must be finite.")

        if error_tensor <= 1.0:
            proposed = current_dt_tensor * self.growth_factor
        else:
            proposed = current_dt_tensor * self.shrink_factor / torch.clamp(error_tensor, min=1.0)

        proposed = torch.clamp(proposed, min=self.min_dt, max=self.max_dt)
        return proposed

    def extra_repr(self) -> str:
        return (
            f"rtol={self.rtol}, atol={self.atol}, min_dt={self.min_dt}, max_dt={self.max_dt}, "
            f"growth_factor={self.growth_factor}, shrink_factor={self.shrink_factor}"
        )


class NumericalMethodFactory:
    """Factory for numerical step methods."""

    _REGISTRY: Dict[str, Callable[[], NumericalMethodBase]] = {
        "euler": EulerStep,
        "rk2": RK2Step,
        "rk4": RK4Step,
    }

    @classmethod
    def create(cls, method_name: str) -> NumericalMethodBase:
        if not isinstance(method_name, str) or not method_name.strip():
            raise ValueError("method_name must be a non-empty string.")
        key = method_name.strip().lower()
        if key not in cls._REGISTRY:
            raise ValueError(f"Unknown numerical method: {method_name}")
        return cls._REGISTRY[key]()


__all__ = [
    "StepDiagnostics",
    "NumericalMethodBase",
    "EulerStep",
    "RK2Step",
    "RK4Step",
    "AdaptiveStepController",
    "NumericalMethodFactory",
]

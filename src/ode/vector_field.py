"""Neural ODE vector field for event-triggered neuromorphic refinement.

This module implements the continuous-time learnable dynamics

    dh(t) / dt = f_theta(h(t), t)

for the Neural ODE refinement stage of the paper pipeline. In the full
architecture, refinement is activated only after trigger/bypass routing
decides that a latent trajectory is temporally complex enough to require
continuous-time stabilization. Numerical integration is intentionally *not*
implemented here; solvers and ODE blocks live in separate modules.

Paper-faithful structure:
    f_theta(h(t), t) = W_2 sigma(W_1 h(t) + b_1) + b_2 + g(t)

This module therefore implements an explicit decomposition into:
- state-dependent dynamics from h(t)
- explicit time-dependent dynamics from t

Design goals:
- clean, import-safe, research-grade PyTorch implementation
- stable MLP-based vector fields with optional time conditioning
- residual vector field option for ODE-friendly latent refinement
- extensibility for future solver, stability, and LoRA integration
- lightweight runtime observability hooks for metrics
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor, nn


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def _validate_float_in_range(name: str, value: float, low: float, high: float) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}.")
    if not (low <= float(value) < high):
        raise ValueError(f"{name} must be in [{low}, {high}), got {value!r}.")


def _resolve_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key == "silu":
        return nn.SiLU()
    if key == "elu":
        return nn.ELU()
    if key == "softplus":
        return nn.Softplus()
    raise ValueError(
        f"Unsupported activation {name!r}. Supported: tanh, gelu, relu, silu, elu, softplus."
    )


def _maybe_apply_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    if enabled and isinstance(module, nn.Linear):
        return nn.utils.spectral_norm(module)
    return module


def _safe_clamp(x: Tensor, enabled: bool, clamp_value: float) -> Tensor:
    if not enabled:
        return x
    return torch.clamp(x, min=-clamp_value, max=clamp_value)


def _broadcast_time_to_shape(
    t: Optional[Any],
    target_shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Broadcast scalar/batched time to the leading shape of h."""
    if len(target_shape) == 0:
        if t is None:
            return torch.zeros((), device=device, dtype=dtype)
        t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
        if t_tensor.ndim > 0 and t_tensor.numel() != 1:
            raise ValueError(
                "Time tensor is not broadcastable to a scalar state input."
            )
        return t_tensor.reshape(())

    if t is None:
        return torch.zeros(*target_shape, device=device, dtype=dtype)

    t_tensor = torch.as_tensor(t, device=device, dtype=dtype)

    if t_tensor.ndim == 0:
        return t_tensor.expand(*target_shape)

    if tuple(t_tensor.shape) == tuple(target_shape):
        return t_tensor

    if t_tensor.ndim < len(target_shape):
        pad = (1,) * (len(target_shape) - t_tensor.ndim)
        t_tensor = t_tensor.reshape(*pad, *t_tensor.shape)

    try:
        return torch.broadcast_to(t_tensor, target_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Time tensor with shape {tuple(t_tensor.shape)} is not broadcastable "
            f"to target leading shape {tuple(target_shape)}."
        ) from exc


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str,
    dropout: float,
    layer_norm: bool,
    bias: bool,
    spectral_norm: bool,
    final_scale: float,
) -> nn.Sequential:
    """Construct a stable MLP used for state or time branches."""
    _validate_positive_int("input_dim", input_dim)
    _validate_positive_int("output_dim", output_dim)
    _validate_positive_int("hidden_dim", hidden_dim)
    _validate_positive_int("num_layers", num_layers)
    _validate_float_in_range("dropout", dropout, 0.0, 1.0)

    act_factory = lambda: _resolve_activation(activation)
    layers: list[nn.Module] = []

    if num_layers == 1:
        out = nn.Linear(input_dim, output_dim, bias=bias)
        out = _maybe_apply_spectral_norm(out, spectral_norm)
        layers.append(out)
        mlp = nn.Sequential(*layers)
        _initialize_mlp(mlp, final_scale=final_scale)
        return mlp

    first = nn.Linear(input_dim, hidden_dim, bias=bias)
    first = _maybe_apply_spectral_norm(first, spectral_norm)
    layers.append(first)
    if layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(act_factory())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        lin = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        lin = _maybe_apply_spectral_norm(lin, spectral_norm)
        layers.append(lin)
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_factory())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    out = nn.Linear(hidden_dim, output_dim, bias=bias)
    out = _maybe_apply_spectral_norm(out, spectral_norm)
    layers.append(out)

    mlp = nn.Sequential(*layers)
    _initialize_mlp(mlp, final_scale=final_scale)
    return mlp


def _initialize_mlp(module: nn.Module, final_scale: float) -> None:
    """Stable Xavier initialization with a conservative final layer scale."""
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return

    for lin in linear_layers[:-1]:
        nn.init.xavier_uniform_(lin.weight)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)

    final = linear_layers[-1]
    nn.init.xavier_uniform_(final.weight)
    final.weight.data.mul_(float(final_scale))
    if final.bias is not None:
        nn.init.zeros_(final.bias)


@dataclass
class VectorFieldConfig:
    """Configuration for continuous-time vector fields."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    time_embedding_dim: int = 0
    num_layers: int = 2
    activation: str = "tanh"
    dropout: float = 0.0
    residual: bool = False
    use_time: bool = True
    layer_norm: bool = False
    bias: bool = True
    clamp_output: bool = False
    output_clamp_value: float = 10.0
    spectral_norm: bool = True
    init_scale: float = 1.0
    name: str = "mlp"
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_int("input_dim", self.input_dim)
        _validate_positive_int("hidden_dim", self.hidden_dim)
        _validate_positive_int("output_dim", self.output_dim)
        _validate_non_negative_int("time_embedding_dim", self.time_embedding_dim)
        _validate_positive_int("num_layers", self.num_layers)
        _validate_float_in_range("dropout", self.dropout, 0.0, 1.0)

        if float(self.output_clamp_value) <= 0:
            raise ValueError(
                f"output_clamp_value must be positive, got {self.output_clamp_value!r}."
            )
        if float(self.init_scale) <= 0:
            raise ValueError(f"init_scale must be positive, got {self.init_scale!r}.")
        if float(self.residual_scale) <= 0:
            raise ValueError(
                f"residual_scale must be positive, got {self.residual_scale!r}."
            )

        self.name = self.name.strip().lower()

        valid_names = {"mlp", "residual"}
        if self.name not in valid_names:
            raise ValueError(
                f"Unsupported vector field name {self.name!r}. Supported: {sorted(valid_names)}."
            )

        if self.residual and self.input_dim != self.output_dim:
            raise ValueError(
                "Residual vector fields require input_dim == output_dim, "
                f"got input_dim={self.input_dim}, output_dim={self.output_dim}."
            )


class BaseVectorField(nn.Module, ABC):
    """Base class for continuous-time dynamics f_theta(h, t).

    This class provides shared validation, stable initialization hooks, and
    runtime observability utilities that are useful for ODE solvers, ODE blocks,
    and later stability diagnostics. The actual numerical integration is
    implemented elsewhere.
    """

    def __init__(self, config: VectorFieldConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("_forward_calls", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("_output_norm_sum", torch.zeros((), dtype=torch.float32), persistent=False)

    @property
    def field_name(self) -> str:
        return self.config.name

    @property
    def model_name(self) -> str:
        return self.field_name

    @property
    def forward_call_count(self) -> int:
        return int(self._forward_calls.item())

    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"name={cfg.name!r}, input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, "
            f"output_dim={cfg.output_dim}, time_embedding_dim={cfg.time_embedding_dim}, "
            f"num_layers={cfg.num_layers}, activation={cfg.activation!r}, "
            f"use_time={cfg.use_time}, residual={cfg.residual}, dropout={cfg.dropout}, "
            f"layer_norm={cfg.layer_norm}, spectral_norm={cfg.spectral_norm}, "
            f"clamp_output={cfg.clamp_output}, init_scale={cfg.init_scale}"
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return self.parameter_count()

    def trainable_parameters(self) -> int:
        return self.trainable_parameter_count()

    def avg_output_norm(self) -> float:
        calls = max(1, self.forward_call_count)
        return float((self._output_norm_sum / calls).item())

    def jacobian_ready(self) -> bool:
        """Indicate that the field is compatible with external Jacobian analysis."""
        return True

    def field_stats(self) -> Dict[str, Any]:
        """Lightweight runtime metadata for downstream ODE metrics."""
        return {
            "model_name": self.model_name,
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "output_dim": self.config.output_dim,
            "time_embedding_dim": self.config.time_embedding_dim,
            "use_time": self.config.use_time,
            "residual": self.config.residual,
            "spectral_norm": self.config.spectral_norm,
            "stability_mode": "spectral_norm" if self.config.spectral_norm else "none",
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.trainable_parameter_count(),
            "forward_calls": self.forward_call_count,
            "avg_output_norm": self.avg_output_norm(),
            "jacobian_ready": self.jacobian_ready(),
        }

    def reset_stats(self) -> None:
        self._forward_calls.zero_()
        self._output_norm_sum.zero_()

    def validate_input(self, h: Tensor) -> None:
        if not isinstance(h, Tensor):
            raise TypeError(f"h must be a torch.Tensor, got {type(h)!r}.")
        if h.ndim < 1:
            raise ValueError(
                f"h must have shape [..., input_dim], got tensor with shape {tuple(h.shape)}."
            )
        if h.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected h.shape[-1] == input_dim == {self.config.input_dim}, got {h.shape[-1]}."
            )
        if not h.is_floating_point():
            raise ValueError("h must be a floating-point tensor.")

    def _update_stats(self, output: Tensor) -> None:
        with torch.no_grad():
            self._forward_calls.add_(1)
            if output.ndim == 0:
                norm = output.abs()
            else:
                norm = output.detach().reshape(-1, output.shape[-1]).norm(dim=-1).mean()
            self._output_norm_sum.add_(norm.to(self._output_norm_sum.dtype))

    def supports_external_stability_checks(self) -> bool:
        return True

    def jacobian_norm_estimate(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError(
            "Jacobian diagnostics are intentionally not implemented in the vector field. "
            "Use src/ode/stability.py for stability analysis."
        )

    @abstractmethod
    def forward(self, h: Tensor, t: Optional[Any] = None) -> Tensor:
        """Compute dh/dt for latent state h at time t."""


class TimeEmbedding(nn.Module):
    """Lightweight continuous-time embedding.

    This module maps scalar or batched time values into a fixed-dimensional
    representation suitable for conditioning the vector field.

    If `time_embedding_dim == 0`, the module returns an empty tensor with the
    correct leading shape, so it can be used transparently in concatenation
    logic or as a no-op branch.
    """

    def __init__(
        self,
        time_embedding_dim: int,
        *,
        max_period: float = 1000.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _validate_non_negative_int("time_embedding_dim", time_embedding_dim)
        if max_period <= 1.0:
            raise ValueError(f"max_period must be > 1.0, got {max_period!r}.")

        self.time_embedding_dim = time_embedding_dim
        self.max_period = float(max_period)

        if time_embedding_dim == 0:
            self.proj = None
            self.register_buffer("_frequencies", torch.empty(0), persistent=False)
            return

        num_bands = max(1, math.ceil(time_embedding_dim / 2))
        frequencies = torch.exp(
            torch.linspace(
                0.0,
                math.log(self.max_period),
                steps=num_bands,
                dtype=torch.float32,
            )
        )
        self.register_buffer("_frequencies", frequencies, persistent=False)
        self.proj = nn.Linear(2 * num_bands, time_embedding_dim, bias=bias)
        self._init_projection()

    def _init_projection(self) -> None:
        if self.proj is None:
            return
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def extra_repr(self) -> str:
        return f"time_embedding_dim={self.time_embedding_dim}, max_period={self.max_period}"

    def forward(
        self,
        t: Optional[Any],
        *,
        reference_shape: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        if self.time_embedding_dim == 0:
            if reference_shape is None:
                if t is None:
                    return torch.empty(0)
                t_tensor = torch.as_tensor(t)
                reference_shape = tuple(t_tensor.shape) if t_tensor.ndim > 0 else ()
            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32
            return torch.empty(*reference_shape, 0, device=device, dtype=dtype)

        if reference_shape is None:
            if t is None:
                raise ValueError(
                    "reference_shape must be provided when t is None for time embedding."
                )
            t_tensor = torch.as_tensor(t)
            reference_shape = tuple(t_tensor.shape) if t_tensor.ndim > 0 else ()

        if device is None:
            device = self._frequencies.device
        if dtype is None:
            dtype = torch.float32

        t_broadcast = _broadcast_time_to_shape(
            t, reference_shape, device=device, dtype=dtype
        )
        t_flat = t_broadcast.reshape(-1, 1)

        freqs = self._frequencies.to(device=device, dtype=dtype).reshape(1, -1)
        angles = t_flat * freqs
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.proj is None:
            emb = features
        else:
            emb = self.proj(features)

        return emb.reshape(*reference_shape, self.time_embedding_dim)


class MLPVectorField(BaseVectorField):
    """Paper-faithful continuous-time vector field with explicit time decomposition.

    The implementation mirrors the methodology equation structurally:

        f_theta(h(t), t) = state_term(h(t)) + time_term(t)

    where:
    - state_term is modeled by an MLP over the latent state h(t)
    - time_term is modeled by an explicit time branch g(t)

    This explicit decomposition is preferred over concatenating h and t into a
    single monolithic network because it directly matches the paper's continuous-
    time refinement formulation and keeps the learnable dynamics easy to inspect.
    """

    def __init__(self, config: VectorFieldConfig) -> None:
        super().__init__(config)

        self.time_embedding: Optional[TimeEmbedding]
        self.time_input_dim: int

        if config.use_time:
            if config.time_embedding_dim > 0:
                self.time_embedding = TimeEmbedding(config.time_embedding_dim, bias=config.bias)
                self.time_input_dim = config.time_embedding_dim
            else:
                self.time_embedding = None
                self.time_input_dim = 1
        else:
            self.time_embedding = None
            self.time_input_dim = 0

        self.state_net = _build_mlp(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            bias=config.bias,
            spectral_norm=config.spectral_norm,
            final_scale=0.05 * float(config.init_scale),
        )

        if config.use_time:
            time_hidden_dim = max(8, min(config.hidden_dim, max(8, self.time_input_dim * 2)))
            time_num_layers = 1 if self.time_input_dim == 1 else min(2, config.num_layers)
            self.time_net = _build_mlp(
                input_dim=self.time_input_dim,
                output_dim=config.output_dim,
                hidden_dim=time_hidden_dim,
                num_layers=time_num_layers,
                activation=config.activation,
                dropout=0.0,
                layer_norm=False,
                bias=config.bias,
                spectral_norm=config.spectral_norm,
                final_scale=0.05 * float(config.init_scale),
            )
        else:
            self.time_net = None

    def _state_term(self, h: Tensor) -> Tensor:
        return self.state_net(h.reshape(-1, h.shape[-1]))

    def _time_term(self, h: Tensor, t: Optional[Any]) -> Tensor:
        if not self.config.use_time:
            return torch.zeros(
                h.shape[:-1] + (self.config.output_dim,),
                device=h.device,
                dtype=h.dtype,
            )

        leading_shape = h.shape[:-1]
        if self.time_embedding is not None:
            time_feat = self.time_embedding(
                t,
                reference_shape=leading_shape,
                device=h.device,
                dtype=h.dtype,
            )
            flat_time = time_feat.reshape(-1, time_feat.shape[-1])
        else:
            t_broadcast = _broadcast_time_to_shape(
                t, leading_shape, device=h.device, dtype=h.dtype
            )
            flat_time = t_broadcast.reshape(-1, 1)

        if self.time_net is None:
            return torch.zeros(
                h.shape[:-1] + (self.config.output_dim,),
                device=h.device,
                dtype=h.dtype,
            )

        return self.time_net(flat_time)

    def _compute_field(self, h: Tensor, t: Optional[Any]) -> Tensor:
        self.validate_input(h)
        leading_shape = h.shape[:-1]
        flat_state = self._state_term(h)

        if self.config.use_time:
            flat_time = self._time_term(h, t)
        else:
            flat_time = torch.zeros(
                flat_state.shape[0],
                self.config.output_dim,
                device=h.device,
                dtype=h.dtype,
            )

        field = flat_state + flat_time
        field = _safe_clamp(field, self.config.clamp_output, float(self.config.output_clamp_value))
        return field.reshape(*leading_shape, self.config.output_dim)

    def forward(self, h: Tensor, t: Optional[Any] = None) -> Tensor:
        output = self._compute_field(h, t)
        self._update_stats(output)
        return output


class ResidualVectorField(MLPVectorField):
    """Residual-stabilized vector field that still returns an ODE derivative.

    This variant does *not* add h directly into the derivative. Instead, it
    scales the paper-faithful vector field by a learnable-design-friendly fixed
    residual factor:

        dh/dt = alpha * f_theta(h, t)

    This keeps the output derivative-like while providing a conservative control
    knob for stability-oriented refinement.
    """

    def __init__(self, config: VectorFieldConfig) -> None:
        if config.input_dim != config.output_dim:
            raise ValueError(
                "ResidualVectorField requires input_dim == output_dim, "
                f"got input_dim={config.input_dim}, output_dim={config.output_dim}."
            )
        cfg = VectorFieldConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            time_embedding_dim=config.time_embedding_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            dropout=config.dropout,
            residual=True,
            use_time=config.use_time,
            layer_norm=config.layer_norm,
            bias=config.bias,
            clamp_output=config.clamp_output,
            output_clamp_value=config.output_clamp_value,
            spectral_norm=config.spectral_norm,
            init_scale=config.init_scale,
            name="residual",
            residual_scale=config.residual_scale,
        )
        super().__init__(cfg)
        self.residual_scale = float(cfg.residual_scale)

    def forward(self, h: Tensor, t: Optional[Any] = None) -> Tensor:
        output = self._compute_field(h, t)
        output = self.residual_scale * output
        output = _safe_clamp(output, self.config.clamp_output, float(self.config.output_clamp_value))
        self._update_stats(output)
        return output


class VectorFieldFactory:
    """Factory for continuous-time vector field variants."""

    @staticmethod
    def from_config(config: VectorFieldConfig) -> BaseVectorField:
        name = config.name.strip().lower()
        if config.residual or name == "residual":
            return ResidualVectorField(config)
        if name == "mlp":
            return MLPVectorField(config)
        raise ValueError(
            f"Unknown vector field name {config.name!r}. Supported: 'mlp', 'residual'."
        )

    @staticmethod
    def from_kwargs(**kwargs: Any) -> BaseVectorField:
        config = VectorFieldConfig(**kwargs)
        return VectorFieldFactory.from_config(config)


__all__ = [
    "VectorFieldConfig",
    "BaseVectorField",
    "TimeEmbedding",
    "MLPVectorField",
    "ResidualVectorField",
    "VectorFieldFactory",
]
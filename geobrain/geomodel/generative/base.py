# pyright: reportPrivateImportUsage=false
"""
Minimal local base classes for generative simulators.

Rather than routing GAN/VAE/Diffusion through a global
``Simulator.create('vae', ...)`` registry built on ``..base.GenerativeSimulator``
and ``..config.GenerativeConfig``, this package uses direct class
imports. This file provides the smallest
viable replacement: a plain ``GenerativeSimulator`` (no abstract methods,
no registry hook, no global factory) and a tiny ``GenerativeConfig``
dataclass with exactly the fields the ported simulators consume.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import (  # type: ignore[attr-defined]
    device as TorchDevice,
    dtype as TorchDType,
    float32 as _f32,
    float64 as _f64,
)

from ..capabilities import GeomodelCapabilityReport
from ..resources import GeomodelResourceEstimate


@dataclass(frozen=True, slots=True)
class GenerativeConfig:
    """
    Configuration for generative simulators (GAN / VAE / Diffusion).

    Args:
        shape: Output grid dimensions.
        latent_dim: Latent space dimensionality (mostly used by GANs).
        n_realizations: Number of samples to generate.
        checkpoint: Path to a model checkpoint file (informational).
        seed: Random seed for reproducibility.
        device: Compute device. Default ``'cpu'`` for reproducibility across
            machines; pass ``'cuda'`` or ``'auto'`` to opt into GPU.
        dtype: Data type ('float32' or 'float64').
        use_amp: Whether to use automatic mixed precision (CUDA only).
    """

    shape: Tuple[int, ...] = (64, 64, 64)
    latent_dim: int = 512
    n_realizations: int = 1
    seed: Optional[int] = None
    device: str = "cpu"
    dtype: str = "float32"
    use_amp: bool = False
    output_device: Literal["compute", "cpu"] = "compute"
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not shape or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape):
            raise GeoBrainError(
                "shape must contain positive exact integers",
                object_name=type(self).__name__, field="shape",
                expected="non-empty tuple of positive ints", actual=shape,
            )
        for name in ("latent_dim", "n_realizations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GeoBrainError(
                    f"{name} must be a positive exact integer",
                    object_name=type(self).__name__, field=name,
                    expected="positive int", actual=value,
                )
        if self.dtype not in ("float32", "float64"):
            raise GeoBrainError(
                "generative dtype must be float32 or float64",
                object_name=type(self).__name__, field="dtype",
                expected="float32 or float64", actual=self.dtype,
            )
        if self.device not in ("cpu", "cuda", "mps"):
            raise GeoBrainError(
                "generative device must be explicit",
                object_name=type(self).__name__, field="device",
                expected="cpu, cuda, or mps", actual=self.device,
            )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise GeoBrainError(
                "requested CUDA device is unavailable",
                object_name=type(self).__name__, field="device",
                expected="available cuda", actual=self.device,
            )
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise GeoBrainError(
                "requested MPS device is unavailable",
                object_name=type(self).__name__, field="device",
                expected="available mps", actual=self.device,
            )
        if self.use_amp and (self.device != "cuda" or self.dtype != "float32"):
            raise GeoBrainError(
                "automatic mixed precision requires CUDA float32 compute",
                object_name=type(self).__name__, field="use_amp",
                expected="device='cuda' and dtype='float32'", actual=(self.device, self.dtype),
            )
        if self.output_device not in ("compute", "cpu"):
            raise GeoBrainError(
                "output_device is unsupported",
                object_name=type(self).__name__, field="output_device",
                expected="compute or cpu", actual=self.output_device,
            )
        if self.budget_bytes is not None and (isinstance(self.budget_bytes, bool) or not isinstance(self.budget_bytes, int) or self.budget_bytes < 1):
            raise GeoBrainError(
                "budget_bytes must be a positive exact integer or None",
                object_name=type(self).__name__, field="budget_bytes",
                expected="positive int or None", actual=self.budget_bytes,
            )
        object.__setattr__(self, "shape", shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def torch_device(self) -> TorchDevice:
        return TorchDevice(self.device)

    @property
    def torch_dtype(self) -> TorchDType:
        return _f64 if self.dtype == "float64" else _f32


class SoftFieldDecoder(nn.Module):
    """Differentiable decoder ``z → soft field`` for latent-space inversion.

    Wraps a simulator's ``decode`` callable and applies a softmax over the class
    channel. Unlike the simulators' ``decode``/``simulate`` methods, which are
    ``@torch.no_grad`` and end in ``argmax`` (zero-gradient, label output); this
    has NO ``no_grad`` and NO ``argmax``, so gradients flow ``z → logits →
    softmax`` and the result is a soft (one-hot-like) facies field that a
    physics ``PropertyTransform`` can map to elastic/EM properties. This is the
    seam that makes ``geobrain.nn.LatentReparameterization(decoder=...)``
    actually differentiable through a learned geomodel prior.

    The decoder follows the latent's device (the wrapped ``decode`` is expected
    to load/move the underlying model onto ``z.device``); its weights are held
    by the simulator and intentionally *not* registered here, latent inversion
    optimises ``z``, not the (frozen) generator.
    """

    def __init__(
        self,
        decode: "Callable[[torch.Tensor], torch.Tensor]",
        *,
        softmax_dim: int = 1,
    ) -> None:
        super().__init__()
        self._decode = decode
        self._softmax_dim = softmax_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._decode(z), dim=self._softmax_dim)


#: Backward-compatible private alias (legacy spelling).
_SoftFieldDecoder = SoftFieldDecoder


class GenerativeSimulator:
    """
    Minimal base for neural generative simulators.

    Provides device management, checkpoint hooks, and latent sampling. Unlike
    a global ``GenerativeSimulator`` base, this class is not abstract and
    has no factory-registry coupling, subclasses just override
    ``_load_checkpoint`` and ``_simulate`` as needed.

    Args:
        model: Pre-built generator/decoder module.
        checkpoint_path: Path to a saved state-dict (.pth).
        latent_dim: Latent space dimensionality.
        transform: Optional output transform applied after generation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        latent_dim: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        conditioning_data: Optional[torch.Tensor] = None,
    ):
        self._model = model
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        self.transform = transform
        self.conditioning_data = conditioning_data
        # Per-call RNG. ``simulate(seed=...)`` builds a local ``torch.Generator``
        # here (see ``_set_seed``) and ``_simulate`` threads it into every
        # ``torch.randn`` draw, so seeding is reproducible WITHOUT mutating the
        # global RNG: matching the injected-generator contract used by the
        # samplers and nn modules. ``None`` means "use the default global RNG".
        self._generator: Optional[torch.Generator] = None
        self._dtype: TorchDType = _f32

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def simulate(
        self,
        config: Optional[GenerativeConfig] = None,
        **overrides,
    ) -> torch.Tensor:
        """Run generation with optional config overrides."""
        if config is None:
            config = GenerativeConfig()
        if overrides:
            allowed = {item.name for item in fields(GenerativeConfig)}
            unknown = sorted(set(overrides) - allowed)
            if unknown:
                raise GeoBrainError(
                    "unknown generative configuration override",
                    object_name=type(self).__name__, field="overrides",
                    expected=sorted(allowed), actual=unknown,
                )
            config = replace(config, **overrides)
        self._set_seed(config)
        result = self._simulate(config)
        if config.output_device == "cpu":
            result = result.to("cpu")
        if self.transform is not None:
            if isinstance(result, dict):
                result = {
                    k: self.transform(v) if isinstance(v, torch.Tensor) else v
                    for k, v in result.items()
                }
            else:
                result = self.transform(result)
        return result

    def __call__(
        self,
        config: Optional[GenerativeConfig] = None,
        **overrides,
    ) -> torch.Tensor:
        return self.simulate(config, **overrides)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _simulate(self, config: GenerativeConfig):  # pragma: no cover - override
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _simulate()."
        )

    def _load_checkpoint(self, device: TorchDevice) -> None:  # pragma: no cover
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load_checkpoint()."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self, config: GenerativeConfig) -> None:
        """Build a per-call ``torch.Generator`` from ``config.seed``.

        Instead of mutating the global RNG via ``torch.manual_seed`` (which
        would disturb every other torch consumer), this seeds a local
        ``Generator`` on the config's device that ``_simulate`` threads into its
        ``torch.randn`` draws. Same seed → identical output, with the global RNG
        left untouched. ``seed=None`` resets ``self._generator`` so draws fall
        back to the default global RNG.
        """
        seed = getattr(config, "seed", None)
        self._dtype = config.torch_dtype
        if seed is None:
            self._generator = None
            return
        device = config.torch_device
        self._generator = torch.Generator(device=device).manual_seed(int(seed))

    def _ensure_model_loaded(self, device: TorchDevice) -> Any:
        if self._model is None:
            if self.checkpoint_path is None:
                raise GeoBrainError(
                    f"{self.__class__.__name__} has no model. Pass 'model' or "
                    f"'checkpoint_path' to the constructor."
                )
            self._load_checkpoint(device)
        assert self._model is not None
        self._model = self._model.to(device=device, dtype=self._dtype).eval()
        return self._model

    def _sample_latent(self, n: int, device: TorchDevice) -> torch.Tensor:
        if self.latent_dim is None:
            raise GeoBrainError(
                f"{self.__class__.__name__} requires latent_dim to be set."
            )
        return torch.randn(
            n, self.latent_dim, device=device, dtype=self._dtype,
            generator=self._generator,
        )

    def set_conditioning(self, data) -> "GenerativeSimulator":
        self.conditioning_data = data
        return self

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @classmethod
    def capabilities(cls) -> GeomodelCapabilityReport:
        return GeomodelCapabilityReport(
            cls.__name__, "geomodel", "experimental", cls.__name__,
            (3,), ("categorical",), (None,), "m",
            ("float32", "float64"), ("cpu", "cuda", "mps"), ("torch",),
            ("model dependent",),
            "fixed verified checkpoint, seed, dtype, and device",
            "soft decoder only; label generation is non-differentiable",
            ("model-card provider",), True, True, (),
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "geobrain.geomodel.input/1.0",
            "type": "object",
            "properties": {
                "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                "n_realizations": {"type": "integer", "minimum": 1},
                "seed": {"type": ["integer", "null"]},
                "device": {"enum": ["cpu", "cuda", "mps"]},
                "dtype": {"enum": ["float32", "float64"]},
                "output_device": {"enum": ["compute", "cpu"]},
            },
            "required": ["shape", "n_realizations", "device", "dtype"],
            "additionalProperties": False,
        }

    def estimate_resources(self, config: GenerativeConfig) -> GeomodelResourceEstimate:
        itemsize = 8 if config.dtype == "float64" else 4
        output = int(torch.tensor(config.shape).prod().item()) * config.n_realizations * itemsize
        latent = config.latent_dim * config.n_realizations * itemsize
        components = (("latent", latent), ("output", output))
        return GeomodelResourceEstimate(
            latent + output, components, 0, 0, 0, None, 0, 0, 1,
            ("model-parameter memory excluded", "forward output lower bound"),
        )

    @property
    def decoder(self) -> Optional[nn.Module]:
        if self._model is None:
            return None
        dec = getattr(self._model, "decoder", None)
        if isinstance(dec, nn.Module):
            return dec
        return self._model

    def differentiable_decoder(self) -> nn.Module:
        """Return an ``nn.Module`` whose ``forward(z)`` emits a DIFFERENTIABLE
        soft field, softmax class probabilities, with **no** ``torch.no_grad``
        and **no** ``argmax``, for gradient-based latent-space inversion
        (``geobrain.nn.LatentReparameterization``). The simulators' ``decode`` /
        ``simulate`` methods are deliberately non-differentiable (they argmax to
        discrete facies labels); this is the explicit autograd path. Subclasses
        with a continuous decode override this; the generic base cannot.
        """
        raise NotImplementedError(
            f"{type(self).__name__} provides no differentiable decoder "
            "(check `is_differentiable` first)."
        )

    @property
    def is_differentiable(self) -> bool:
        """True iff the simulator provides a differentiable decoder: i.e. a
        gradient-flowing ``z → soft field`` map via :meth:`differentiable_decoder`."""
        return (
            type(self).differentiable_decoder
            is not GenerativeSimulator.differentiable_decoder
        )

    def as_reparameterization(
        self,
        outputs,
        *,
        latent_field: str = "latent",
        latent_shape=None,
        transforms=None,
    ):
        """This simulator's frozen decoder as a differentiable
        :class:`~geobrain.nn.LatentReparameterization`: one line from a
        pretrained geomodel generator to an invertible prior::

            problem = InverseProblem(
                forward=physics_op @ sim.as_reparameterization(
                    outputs={"sand": 0, "shale": 1}, latent_shape=(64, 8, 8)),
                observed=..., likelihood=..., prior=...,
            )

        Wraps :meth:`differentiable_decoder` (the gradient-flowing
        ``z -> soft field`` map), so it requires :attr:`is_differentiable`.
        All arguments follow :class:`~geobrain.nn.LatentReparameterization`.
        """
        from geobrain.core.errors import GeoBrainError

        if not self.is_differentiable:
            raise GeoBrainError(
                f"{type(self).__name__} provides no differentiable decoder, "
                "so it cannot enter a gradient-based inversion "
                "(check `is_differentiable` first)",
                object_name=type(self).__name__,
                field="differentiable_decoder",
                expected="subclass overriding differentiable_decoder()",
                actual="base implementation",
            )
        from geobrain.nn import LatentReparameterization

        return LatentReparameterization(
            self.differentiable_decoder(),
            outputs,
            latent_field=latent_field,
            latent_shape=latent_shape,
            transforms=transforms,
        )

    @property
    def supports_conditioning(self) -> bool:
        return False


__all__ = ["GenerativeConfig", "GenerativeSimulator", "SoftFieldDecoder"]

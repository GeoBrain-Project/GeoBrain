"""Lazy CUDA extension loading for the experimental Wave native backend.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import torch


NativeExtensionName = Literal[
    "probe", "acoustic2d", "acoustic3d", "elastic2d", "elastic3d"
]

_EXTENSION_NAMES: dict[NativeExtensionName, tuple[str, str]] = {
    "probe": ("geobrain_wave_native_probe", "_probe.cu"),
    "acoustic2d": ("geobrain_wave_native_acoustic2d", "acoustic2d.cu"),
    "acoustic3d": ("geobrain_wave_native_acoustic3d", "acoustic3d.cu"),
    "elastic2d": ("geobrain_wave_native_elastic2d", "elastic2d.cu"),
    "elastic3d": ("geobrain_wave_native_elastic3d", "elastic3d.cu"),
}
_BUILD_ENVIRONMENT = (
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "TORCH_CUDA_ARCH_LIST",
)


def is_available() -> bool:
    """Return whether PyTorch reports an actual CUDA execution device."""
    return cast(bool, torch.cuda.is_available())


def is_cuda_tensor(tensor: torch.Tensor) -> bool:
    """Return whether one live tensor is placed on a CUDA device."""
    return cast(bool, tensor.device.type == "cuda")


@contextmanager
def _build_environment() -> Iterator[None]:
    """Apply and always restore the isolated extension-build environment."""
    saved = {name: os.environ.get(name) for name in _BUILD_ENVIRONMENT}
    try:
        for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
            os.environ.pop(name, None)
        if "TORCH_EXTENSIONS_DIR" not in os.environ:
            scratch = os.environ.get("SCRATCH")
            if scratch:
                os.environ["TORCH_EXTENSIONS_DIR"] = str(
                    Path(scratch) / "torch_extensions"
                )
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@lru_cache(maxsize=None)
def load_native_extension(name: NativeExtensionName) -> Any:
    """Build or fetch one cached native extension by stable logical name."""
    from torch.utils.cpp_extension import load

    extension_name, source_name = _EXTENSION_NAMES[name]
    with _build_environment():
        source = Path(__file__).resolve().parent / source_name
        return load(name=extension_name, sources=[str(source)], verbose=False)


class _ProbeFunction(torch.autograd.Function):  # type: ignore[misc]
    """Autograd wrapper for the small CUDA build/adjoint probe."""

    @staticmethod
    def forward(
        context: Any, x: torch.Tensor, scale: float
    ) -> torch.Tensor:
        context.save_for_backward(x)
        context.scale = float(scale)
        return load_native_extension("probe").forward(x, float(scale))

    @staticmethod
    def backward(
        context: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (x,) = context.saved_tensors
        result = load_native_extension("probe").backward(
            x, gradient.contiguous(), context.scale
        )
        return result, None


def probe(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Run the native build probe through its hand-written CUDA adjoint."""
    return _ProbeFunction.apply(x, scale)


__all__ = [
    "NativeExtensionName",
    "is_available",
    "is_cuda_tensor",
    "load_native_extension",
    "probe",
]

"""
cuDSS backend for the sparse implicit-VJP bridge.

:class:`CuDSSSolver` implements the
:class:`~geobrain.core.adjoint.SparseFactorSolver` contract with NVIDIA's
cuDSS direct sparse LU via a thin ctypes binding against the
``nvidia-cudss-cu12`` wheel (optional extra, the roadmap's
optional-dependency policy). It is the factor-REUSING GPU direct solver:
analysis + factorization happen once per matrix; every subsequent RHS
(polarizations, adjoints, Newton steps) is a triangular solve::

    from geobrain.core.linalg import sparse_solver
    from geobrain.core.linalg.backend_cudss import CuDSSSolver

    with sparse_solver(CuDSSSolver()):
        ...  # sparse EM/Helmholtz forwards factor once on the GPU per system

Scope (v1, correctness-first): CUDSS_MTYPE_GENERAL LU (no symmetry
exploitation), float32/float64/complex64/complex128 values, int32 indices.
``solve(adjoint=True)`` lazily factors ``A^H`` the first time it is needed
(cuDSS exposes no transpose-solve on an existing factorization), still one
factorization per matrix per direction, amortized over all its solves.

Availability: :func:`cudss_available`. When the wheel or a CUDA device is
missing, construction raises a structured error naming the install step;
the pure-torch default path is untouched.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import ctypes
import glob
import os
from typing import Any

import torch

from ..errors import GeoBrainError
from .seam import SparseFactorSolver

__all__ = ["CuDSSSolver", "cudss_available"]

# --- cuDSS ABI constants -----------------------------------------------------
# The phase encoding and the cudssMatrixCreateCsr arity CHANGED between the
# 0.5 era (GLIBC 2.17: the newest line that runs on CentOS-7 nodes) and 0.8
# (needs GLIBC 2.27). Both are supported; the loaded library's version is
# probed via cudssGetProperty and the right table picked at load time.
_ABI = {
    "old": {  # <= 0.5.x (cudss.h: ANALYSIS=1, FACTORIZATION=2, SOLVE=8)
        "analysis": 1, "factorization": 2, "solve": 8, "csr_has_offset_type": False,
    },
    "new": {  # >= 0.6.x (bitmask phases; CreateCsr grew offsetType)
        "analysis": (1 << 0) | (1 << 1),
        "factorization": 1 << 2,
        "solve": (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9),
        "csr_has_offset_type": True,
    },
}
_MTYPE_GENERAL = 0
_MVIEW_FULL = 0
_BASE_ZERO = 0
_LAYOUT_COL_MAJOR = 0
_STATUS_SUCCESS = 0
# CUDA library_types.h ordinals
_CUDA_DTYPE = {
    torch.float32: 0,     # CUDA_R_32F
    torch.float64: 1,     # CUDA_R_64F
    torch.complex64: 4,   # CUDA_C_32F
    torch.complex128: 5,  # CUDA_C_64F
}
_CUDA_R_32I = 10


def _find_libcudss() -> str | None:
    """Locate ``libcudss.so`` from the wheel install or ``CUDSS_LIB``.

    De-facto LINUX-ONLY: the ``nvidia-cudss-cu12`` wheel ships a shared
    object (``libcudss.so``) and no Windows/macOS binary, so on those
    platforms this returns ``None`` and the backend degrades to its
    structured unavailable error.
    """
    env = os.environ.get("CUDSS_LIB")
    if env and os.path.exists(env):
        return env
    try:
        import nvidia  # the wheel's namespace package

        for root in nvidia.__path__:
            hits = glob.glob(os.path.join(root, "**", "libcudss.so*"), recursive=True)
            if hits:
                return sorted(hits)[0]
    except Exception:
        pass
    return None


_LIB: ctypes.CDLL | None = None
_HANDLE: ctypes.c_void_p | None = None
_ABI_TABLE: dict | None = None


def _detect_abi(lib: ctypes.CDLL) -> dict:
    """Pick the ABI table from the loaded library's MAJOR.MINOR."""
    try:
        major, minor = ctypes.c_int(0), ctypes.c_int(0)
        # libraryPropertyType: MAJOR_VERSION=0, MINOR_VERSION=1
        lib.cudssGetProperty(0, ctypes.byref(major))
        lib.cudssGetProperty(1, ctypes.byref(minor))
        return _ABI["new"] if (major.value, minor.value) >= (0, 6) else _ABI["old"]
    except Exception:
        return _ABI["old"]


def _load() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = _find_libcudss()
        if path is None:
            raise GeoBrainError(
                "cuDSS library not found, install the optional extra with "
                "`pip install nvidia-cudss-cu12` (or point CUDSS_LIB at "
                "libcudss.so)",
                object_name="CuDSSSolver",
                field="libcudss",
                expected="nvidia-cudss-cu12 wheel or CUDSS_LIB",
                actual=None,
            )
        # torch must be CUDA-initialized first so cuBLAS/cudart sonames the
        # library links against are already resolvable in-process.
        _LIB = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        global _ABI_TABLE
        _ABI_TABLE = _detect_abi(_LIB)
    return _LIB


def cudss_available() -> bool:
    """True iff the wheel's library is locatable (does not touch the GPU)."""
    return _find_libcudss() is not None


def _check(status: int, what: str) -> None:
    if status != _STATUS_SUCCESS:
        raise GeoBrainError(
            f"cuDSS {what} failed",
            object_name="CuDSSSolver",
            field=what,
            expected="CUDSS_STATUS_SUCCESS (0)",
            actual=int(status),
        )


def _global_handle(lib: ctypes.CDLL) -> ctypes.c_void_p:
    global _HANDLE
    if _HANDLE is None:
        h = ctypes.c_void_p()
        _check(lib.cudssCreate(ctypes.byref(h)), "cudssCreate")
        _HANDLE = h
    return _HANDLE


class _Factorization:
    """One direction's (A or A^H) analysis+factorization, kept alive.

    Owns the device CSR tensors (their ``data_ptr`` is registered with
    cuDSS), the cudss matrix object and the config/data pair. ``solve``
    re-executes only the SOLVE phase against the stored objects.
    """

    def __init__(self, lib, handle, crow, col, vals, n: int) -> None:
        self._lib = lib
        self._handle = handle
        # Keep the tensors alive for the lifetime of the factorization.
        self._crow, self._col, self._vals = crow, col, vals
        self.n = n
        self.dtype = vals.dtype
        self.device = vals.device

        self._config = ctypes.c_void_p()
        _check(lib.cudssConfigCreate(ctypes.byref(self._config)), "cudssConfigCreate")
        self._data = ctypes.c_void_p()
        _check(lib.cudssDataCreate(handle, ctypes.byref(self._data)), "cudssDataCreate")

        self._A = ctypes.c_void_p()
        abi = _ABI_TABLE or _ABI["old"]
        csr_args = [
            ctypes.byref(self._A),
            ctypes.c_int64(n), ctypes.c_int64(n),
            ctypes.c_int64(vals.numel()),
            ctypes.c_void_p(crow.data_ptr()), None,
            ctypes.c_void_p(col.data_ptr()),
            ctypes.c_void_p(vals.data_ptr()),
            ctypes.c_int(_CUDA_R_32I),           # indexType (0.5) / offsetType (0.8)
        ]
        if abi["csr_has_offset_type"]:
            csr_args.append(ctypes.c_int(_CUDA_R_32I))   # indexType (0.8 extra)
        csr_args += [
            ctypes.c_int(_CUDA_DTYPE[vals.dtype]),
            ctypes.c_int(_MTYPE_GENERAL), ctypes.c_int(_MVIEW_FULL),
            ctypes.c_int(_BASE_ZERO),
        ]
        _check(lib.cudssMatrixCreateCsr(*csr_args), "cudssMatrixCreateCsr")
        # Analysis and factorization take no RHS; cuDSS accepts NULL there.
        _check(
            lib.cudssExecute(handle, ctypes.c_int(abi["analysis"]),
                             self._config, self._data, self._A, None, None),
            "cudssExecute(analysis)",
        )
        _check(
            lib.cudssExecute(handle, ctypes.c_int(abi["factorization"]),
                             self._config, self._data, self._A, None, None),
            "cudssExecute(factorization)",
        )

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        n = self.n
        one_d = b.ndim == 1
        B = b.unsqueeze(1) if one_d else b
        k = B.shape[1]
        # Column-major (n, k) buffers: a (k, n) row-major block IS the
        # column-major layout cuDSS expects (ld = n).
        b_cm = B.to(self.device, self.dtype).T.contiguous()
        x_cm = torch.empty_like(b_cm)
        lib, handle = self._lib, self._handle
        mb = ctypes.c_void_p()
        mx = ctypes.c_void_p()
        _check(
            lib.cudssMatrixCreateDn(
                ctypes.byref(mb), ctypes.c_int64(n), ctypes.c_int64(k),
                ctypes.c_int64(n), ctypes.c_void_p(b_cm.data_ptr()),
                ctypes.c_int(_CUDA_DTYPE[self.dtype]),
                ctypes.c_int(_LAYOUT_COL_MAJOR),
            ),
            "cudssMatrixCreateDn(b)",
        )
        _check(
            lib.cudssMatrixCreateDn(
                ctypes.byref(mx), ctypes.c_int64(n), ctypes.c_int64(k),
                ctypes.c_int64(n), ctypes.c_void_p(x_cm.data_ptr()),
                ctypes.c_int(_CUDA_DTYPE[self.dtype]),
                ctypes.c_int(_LAYOUT_COL_MAJOR),
            ),
            "cudssMatrixCreateDn(x)",
        )
        try:
            _check(
                lib.cudssExecute(
                    handle, ctypes.c_int((_ABI_TABLE or _ABI["old"])["solve"]),
                    self._config, self._data, self._A, mx, mb),
                "cudssExecute(solve)",
            )
            torch.cuda.synchronize(self.device)
        finally:
            lib.cudssMatrixDestroy(mb)
            lib.cudssMatrixDestroy(mx)
        x = x_cm.T  # back to row-major (n, k) view
        return x[:, 0].clone() if one_d else x.clone()

    def __del__(self) -> None:  # best-effort native cleanup
        lib = getattr(self, "_lib", None)
        if lib is None:
            return
        try:
            if getattr(self, "_A", None):
                lib.cudssMatrixDestroy(self._A)
            if getattr(self, "_data", None):
                lib.cudssDataDestroy(self._handle, self._data)
            if getattr(self, "_config", None):
                lib.cudssConfigDestroy(self._config)
        except Exception:
            pass


class _CuDSSHandle:
    """Seam handle: forward factorization eager, adjoint lazy."""

    __slots__ = ("fwd", "adj", "_indices", "_values", "_shape", "_solver")

    def __init__(self, fwd, indices, values, shape, solver) -> None:
        self.fwd = fwd
        self.adj = None
        self._indices = indices
        # Detach at storage: the handle may outlive the forward call, and a
        # graph-carrying values tensor would pin the whole autograd graph in
        # memory for the handle lifetime (gradients flow per-solve through
        # the bridge, never through this cached copy).
        self._values = values.detach()
        self._shape = shape
        self._solver = solver

    def adjoint_factorization(self):
        if self.adj is None:
            idx = self._indices
            swapped = torch.stack([idx[1], idx[0]])
            self.adj = self._solver._factor_coo(
                swapped, self._values.conj(), self._shape
            )
        return self.adj


class CuDSSSolver(SparseFactorSolver):
    """GPU direct sparse LU (NVIDIA cuDSS) behind the sparse-adjoint seam.

    Factor once, solve many, the property the measured cupy-QR route lacks. ``adjoint=True`` solves ``A^H x = b`` via a
    lazily-built second factorization of ``A^H``.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise GeoBrainError(
                "CuDSSSolver requires a CUDA device",
                object_name="CuDSSSolver",
                field="cuda",
                expected="torch.cuda.is_available()",
                actual=False,
            )
        self._lib = _load()
        self._handle = _global_handle(self._lib)

    # -- SparseFactorSolver contract --------------------------------------
    def factor(self, indices: torch.Tensor, values: torch.Tensor, shape) -> Any:
        if not bool(torch.isfinite(values.detach().abs()).all()):
            raise GeoBrainError(
                "CuDSSSolver.factor requires finite matrix values",
                object_name="CuDSSSolver",
                field="values",
                expected="all-finite",
                actual="NaN/Inf present",
            )
        fwd = self._factor_coo(indices, values, shape)
        return _CuDSSHandle(fwd, indices, values, shape, self)

    def solve(self, handle: Any, b: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        fac = handle.adjoint_factorization() if adjoint else handle.fwd
        x = fac.solve(b)
        return x.to(b.device, b.dtype)

    # -- internals ---------------------------------------------------------
    def _factor_coo(self, indices, values, shape) -> _Factorization:
        n = int(shape[0])
        if int(shape[1]) != n:
            raise GeoBrainError(
                "CuDSSSolver requires a square matrix",
                object_name="CuDSSSolver",
                field="shape",
                expected="(n, n)",
                actual=tuple(shape),
            )
        dtype = values.dtype
        if dtype not in _CUDA_DTYPE:
            raise GeoBrainError(
                "CuDSSSolver supports float32/float64/complex64/complex128",
                object_name="CuDSSSolver",
                field="values.dtype",
                expected=sorted(str(d) for d in _CUDA_DTYPE),
                actual=str(dtype),
            )
        dev = torch.device("cuda")
        coo = torch.sparse_coo_tensor(
            indices.to(dev), values.detach().to(dev), (n, n)
        ).coalesce()
        csr = coo.to_sparse_csr()
        if csr.values().numel() > 2**31 - 1 or n > 2**31 - 1:
            raise GeoBrainError(
                "cuDSS backend uses 32-bit indexing; matrix exceeds int32 range",
                object_name="CuDSSSolver", field="A",
                expected="n and nnz < 2^31",
                actual=f"n={n}, nnz={csr.values().numel()}",
            )
        crow = csr.crow_indices().to(torch.int32).contiguous()
        col = csr.col_indices().to(torch.int32).contiguous()
        vals = csr.values().contiguous()
        return _Factorization(self._lib, self._handle, crow, col, vals, n)

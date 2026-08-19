"""Explicit-lifetime assembly and factor cache for inductive EM.

There is intentionally no module-global cache.  A caller creates one scope,
uses it for a single forward/batch, and closes it; closing clears every tensor
and numeric factor so autograd state cannot leak into a later forward.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from geobrain.physics.em.errors import EMContractError

from .contracts import AssemblyCacheKey, EMExecutionDiagnostics, FactorCacheKey


T = TypeVar("T")


class EMExecutionCache:
    """One forward-local cache with explicit ownership and diagnostics."""

    __slots__ = ("_assemblies", "_closed", "_factors", "diagnostics")

    def __init__(self) -> None:
        self._assemblies: dict[AssemblyCacheKey, object] = {}
        self._factors: dict[FactorCacheKey, object] = {}
        self._closed = False
        self.diagnostics = EMExecutionDiagnostics()

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise EMContractError(
                "EMExecutionCache is closed",
                details={"state": "closed"},
                object_name=type(self).__name__,
                field="lifetime",
                expected="open cache",
                actual="closed cache",
            )

    @staticmethod
    def _require_key(key: object, expected: type) -> None:
        if not isinstance(key, expected):
            raise EMContractError(
                f"cache key must be {expected.__name__}",
                details={"received_type": type(key).__qualname__},
                object_name="EMExecutionCache",
                field="key",
                expected=expected.__name__,
                actual=type(key).__qualname__,
            )

    def get_or_assemble(
        self,
        key: AssemblyCacheKey,
        factory: Callable[[], T],
    ) -> T:
        """Return the scoped assembly for ``key``, creating it exactly once."""
        self._require_open()
        self._require_key(key, AssemblyCacheKey)
        if key in self._assemblies:
            self.diagnostics.assembly_cache_hits += 1
            return cast(T, self._assemblies[key])
        value = factory()
        self._assemblies[key] = value
        self.diagnostics.assembly_count += 1
        return value

    def bind_assembly(self, key: AssemblyCacheKey, value: T) -> T:
        """Bind a preassembled value without an untyped closure at the call site."""
        self._require_open()
        self._require_key(key, AssemblyCacheKey)
        if key in self._assemblies:
            self.diagnostics.assembly_cache_hits += 1
            return cast(T, self._assemblies[key])
        self._assemblies[key] = value
        self.diagnostics.assembly_count += 1
        return value

    def reuse_assembly(self, key: AssemblyCacheKey) -> object:
        """Record and return one intentional hit on an already-bound assembly."""
        self._require_open()
        self._require_key(key, AssemblyCacheKey)
        try:
            value = self._assemblies[key]
        except KeyError as exc:
            raise EMContractError(
                "cannot reuse an unregistered inductive assembly",
                details={"formulation_version": key.formulation_version},
                object_name=type(self).__name__,
                field="assembly",
                expected="bind_assembly called for this key",
                actual="missing",
            ) from exc
        self.diagnostics.assembly_cache_hits += 1
        return value

    def require_assembly(self, key: AssemblyCacheKey) -> object:
        """Return an already-bound matrix or fail instead of guessing one."""
        self._require_open()
        self._require_key(key, AssemblyCacheKey)
        try:
            return self._assemblies[key]
        except KeyError as exc:
            raise EMContractError(
                "factor solve requires a registered assembly",
                details={"formulation_version": key.formulation_version},
                object_name=type(self).__name__,
                field="assembly",
                expected="get_or_assemble called for this key",
                actual="missing",
            ) from exc

    def get_or_factor(
        self,
        key: FactorCacheKey,
        factory: Callable[[], T],
    ) -> T:
        """Return the scoped factor for ``key``, creating it exactly once."""
        self._require_open()
        self._require_key(key, FactorCacheKey)
        if key in self._factors:
            self.diagnostics.factor_cache_hits += 1
            return cast(T, self._factors[key])
        value = factory()
        self._factors[key] = value
        self.diagnostics.factorization_count += 1
        return value

    def close(self) -> None:
        """Release all cached objects and permanently close this scope."""
        if not self._closed:
            self._assemblies.clear()
            self._factors.clear()
            self._closed = True

    def __enter__(self) -> "EMExecutionCache":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["EMExecutionCache"]

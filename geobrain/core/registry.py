"""
Generic registry for extensible types.

Each family of pluggable types gets its own ``Registry[T]`` instance. Lookups are
by string name; the same name cannot be registered twice unless
``overwrite=True`` (reserved for tests).

The platform instantiates this ``Registry`` for **rock-physics models**
(``physics/rock/models/_registry.py``) and **AVO models**
(``physics/wave/reflectivity/registry.py``); **Bayesian samplers** use the same
name-dispatch idea via a class-level dict (``Sampler.register`` in
``bayes/base.py``). Forward operators are **not** auto-registered; they are
constructed directly (``Acoustic2D(...)``, ``op_b @ op_a``), so there is no
``operator_registry``; pick a real registry (rock models) for the recipe::

    from geobrain.physics.rock.models import register, get_model

    @register("VRH")          # decorator registers the class under "VRH"
    class VoigtReussHill(EffectiveModel):
        ...

    cls = get_model("VRH")    # look it up by name

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

from .errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → object mapping. Same name twice → error."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, *, overwrite: bool = False) -> Callable[[T], T]:
        """
        Decorator that registers the decorated value under ``name``.

        Args:
            name: Non-empty string key to register under.
            overwrite: Allow replacing an existing entry (reserve for tests).

        Returns:
            A decorator that records its argument and returns it unchanged.
        """
        if not isinstance(name, str) or not name:
            raise RegistryError(
                "registry name must be a non-empty string",
                object_name=f"Registry[{self._kind}]",
                field="name",
                actual=name,
            )

        def _decorator(value: T) -> T:
            if name in self._entries and not overwrite:
                raise RegistryError(
                    "duplicate registration",
                    object_name=f"Registry[{self._kind}]",
                    field=name,
                    expected="unique name",
                    actual=self._entries[name],
                )
            self._entries[name] = value
            return value

        return _decorator

    def get(self, name: str) -> T:
        """Return the entry registered under ``name``; raise if absent."""
        if name not in self._entries:
            raise RegistryError(
                "no entry with that name",
                object_name=f"Registry[{self._kind}]",
                field=name,
                expected=f"one of {sorted(self._entries)}",
            )
        return self._entries[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

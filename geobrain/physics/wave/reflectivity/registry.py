"""
Reflectivity-model registry with AVO aliases.

A thin wrapper around :class:`geobrain.core.registry.Registry` that adds alias
support (``"aki"`` → ``"AkiRichards"``) and a name-based instantiation helper. The
canonical-name set lives in the platform registry so AVO models are discoverable
alongside the rest of GeoBrain's extensibility surface; the alias layer is local to
reflectivity family because the seismic literature uses several short names per model.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from geobrain.core.registry import Registry

from .base import AVOModel

_registry: Registry[type[AVOModel]] = Registry("avo_models")
_aliases: dict[str, str] = {}
_metadata: dict[str, dict[str, Any]] = {}


def register_avo(
    name: str, *, aliases: list[str] | None = None, **metadata: Any
) -> Callable[[type[AVOModel]], type[AVOModel]]:
    """
    Decorator: register an :class:`AVOModel` subclass under ``name``, with optional
    aliases and free-form metadata.

    Re-registration under the same canonical name overwrites the entry (so module
    reloads in test sessions don't blow up). Aliases must be unique across the
    registry.
    """

    def decorator(cls: type[AVOModel]) -> type[AVOModel]:
        # overwrite=True is the public seam Registry already exposes for
        # reload idempotency: no reason to reach into the private _entries
        # dict for it.
        _registry.register(name, overwrite=True)(cls)
        _metadata[name] = dict(metadata)
        for alias in aliases or []:
            _aliases[alias] = name
        return cls

    return decorator


def _resolve(name: str) -> str:
    return _aliases.get(name, name)


def get_avo(name: str) -> type[AVOModel]:
    """Look up an AVO model class by canonical name or alias."""
    return cast(type[AVOModel], _registry.get(_resolve(name)))


def create_avo(name: str, **kwargs: Any) -> AVOModel:
    """Instantiate an AVO model by canonical name or alias."""
    return get_avo(name)(**kwargs)


def list_avo() -> list[str]:
    """Canonical (non-alias) names of registered AVO models."""
    return list(_registry.names())


def list_avo_aliases() -> dict[str, str]:
    """Alias → canonical-name map."""
    return dict(_aliases)


__all__ = [
    "create_avo",
    "get_avo",
    "list_avo",
    "list_avo_aliases",
    "register_avo",
]

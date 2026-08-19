"""Structured optional-provider discovery without import-path mutation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType

from ..errors import GeomodelCapabilityError, GeomodelContractError

__all__ = ["LDMC_PROVIDER", "OptionalProvider", "require_provider"]


@dataclass(frozen=True, slots=True)
class OptionalProvider:
    """One optional generative-model backend and how to obtain it.

    Attributes:
        name: provider display name.
        import_name: importable module name probed at use time.
        install_hint: the exact install command surfaced in errors.
    """

    name: str
    import_name: str
    install_hint: str

    def __post_init__(self) -> None:
        if not all(isinstance(item, str) and item.strip() for item in (self.name, self.import_name, self.install_hint)):
            raise GeomodelContractError(
                "optional-provider fields must be non-empty strings",
                object_name=type(self).__name__, field="name/import_name/install_hint",
                expected="non-empty strings", actual=(self.name, self.import_name, self.install_hint),
            )


LDMC_PROVIDER = OptionalProvider("ldmc", "LDMC", "install geobrain[generative-ldmc]")


def require_provider(provider: OptionalProvider) -> ModuleType:
    """Import an optional generative-model provider or fail with the fix.

    Args:
        provider: the :class:`OptionalProvider` record naming the module
            and its install hint (e.g. :data:`LDMC_PROVIDER`).

    Returns:
        The imported provider module.

    Raises:
        GeomodelCapabilityError: if the provider is not installed, the
            error's ``hint`` carries the exact install command.
        GeomodelContractError: if ``provider`` is not an
            :class:`OptionalProvider`, or the import itself fails.
    """
    if not isinstance(provider, OptionalProvider):
        raise GeomodelContractError(
            "provider must be OptionalProvider",
            object_name="require_provider", field="provider",
            expected="OptionalProvider", actual=type(provider).__name__,
        )
    if importlib.util.find_spec(provider.import_name) is None:
        raise GeomodelCapabilityError(
            f"optional provider {provider.name!r} is unavailable",
            object_name="require_provider",
            field="optional_dependency",
            expected=provider.import_name,
            actual="not installed",
            hint=provider.install_hint,
        )
    try:
        return importlib.import_module(provider.import_name)
    except ImportError as exc:
        raise GeomodelCapabilityError(
            f"optional provider {provider.name!r} could not be imported",
            object_name="require_provider",
            field="optional_dependency",
            expected=provider.import_name,
            actual=str(exc),
            hint=provider.install_hint,
        ) from exc

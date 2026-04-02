"""Simple registries for backends and tools."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Type


class Registry:
    """Name-to-object registry with decorator-based registration support."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str | None = None) -> Callable[[Any], Any]:
        """Register object under explicit key or object.name."""

        def _decorator(obj: Any) -> Any:
            registry_key = key or getattr(obj, "name", obj.__name__)
            if registry_key in self._items:
                raise ValueError(f"{self.name} already contains key '{registry_key}'")
            self._items[registry_key] = obj
            return obj

        return _decorator

    def get(self, key: str) -> Any:
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def items(self) -> Dict[str, Any]:
        return dict(self._items)

    def keys(self) -> Iterable[str]:
        return self._items.keys()


# Global registries used by the execution and agent layers.
BACKEND_REGISTRY = Registry("backends")
TOOL_REGISTRY = Registry("tools")

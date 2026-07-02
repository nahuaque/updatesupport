"""Lightweight plugin registry for domain-specific extensions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any


@dataclass(frozen=True)
class UpdateSupportPlugin:
    """Domain extension descriptor.

    Plugins can expose metric factories, Q preset factories, report profiles,
    and compilers without adding domain-specific vocabulary to the core package.
    The values are intentionally generic callables so plugins can evolve their
    own public APIs while core provides discovery and namespacing.
    """

    name: str
    version: str | None = None
    description: str = ""
    metrics: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    q_presets: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    report_profiles: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    compilers: Mapping[str, Callable[..., Any]] = field(default_factory=dict)


class PluginRegistry:
    """Mutable registry for explicit and entry-point-discovered plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, UpdateSupportPlugin] = {}

    def register(self, plugin: UpdateSupportPlugin) -> UpdateSupportPlugin:
        if not plugin.name:
            raise ValueError("plugin name must be non-empty")
        self._plugins[plugin.name] = plugin
        return plugin

    def unregister(self, name: str) -> None:
        self._plugins.pop(name, None)

    def get(self, name: str) -> UpdateSupportPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(f"updatesupport plugin is not registered: {name!r}") from exc

    def list(self) -> tuple[UpdateSupportPlugin, ...]:
        return tuple(self._plugins[name] for name in sorted(self._plugins))

    def discover(self, *, group: str = "updatesupport.plugins") -> tuple[
        UpdateSupportPlugin, ...
    ]:
        discovered = []
        for entry_point in _entry_points(group):
            plugin = _load_plugin(entry_point)
            self.register(plugin)
            discovered.append(plugin)
        return tuple(discovered)


_REGISTRY = PluginRegistry()


def register_plugin(plugin: UpdateSupportPlugin) -> UpdateSupportPlugin:
    """Register a plugin explicitly."""

    return _REGISTRY.register(plugin)


def unregister_plugin(name: str) -> None:
    """Remove a plugin from the process-local registry if present."""

    _REGISTRY.unregister(name)


def get_plugin(name: str) -> UpdateSupportPlugin:
    """Return a registered plugin by name."""

    return _REGISTRY.get(name)


def list_plugins() -> tuple[UpdateSupportPlugin, ...]:
    """Return registered plugins sorted by name."""

    return _REGISTRY.list()


def discover_plugins(*, group: str = "updatesupport.plugins") -> tuple[
    UpdateSupportPlugin, ...
]:
    """Discover plugins declared through Python package entry points."""

    return _REGISTRY.discover(group=group)


def plugin_metric(plugin_name: str, metric_name: str) -> Callable[..., Any]:
    """Return a metric factory from a registered plugin."""

    return _registry_lookup(plugin_name, "metrics", metric_name)


def plugin_q_preset(plugin_name: str, preset_name: str) -> Callable[..., Any]:
    """Return a Q preset factory from a registered plugin."""

    return _registry_lookup(plugin_name, "q_presets", preset_name)


def plugin_report_profile(plugin_name: str, profile_name: str) -> Callable[..., Any]:
    """Return a report-profile callable from a registered plugin."""

    return _registry_lookup(plugin_name, "report_profiles", profile_name)


def plugin_compiler(plugin_name: str, compiler_name: str) -> Callable[..., Any]:
    """Return a compiler callable from a registered plugin."""

    return _registry_lookup(plugin_name, "compilers", compiler_name)


def _registry_lookup(
    plugin_name: str,
    collection_name: str,
    item_name: str,
) -> Callable[..., Any]:
    plugin = get_plugin(plugin_name)
    collection = getattr(plugin, collection_name)
    try:
        return collection[item_name]
    except KeyError as exc:
        raise KeyError(
            f"updatesupport plugin {plugin_name!r} has no "
            f"{collection_name[:-1]} named {item_name!r}"
        ) from exc


def _entry_points(group: str) -> tuple[EntryPoint, ...]:
    selected = entry_points()
    if hasattr(selected, "select"):
        return tuple(selected.select(group=group))
    return tuple(selected.get(group, ()))


def _load_plugin(entry_point: EntryPoint) -> UpdateSupportPlugin:
    loaded = entry_point.load()
    value = loaded() if callable(loaded) and not isinstance(loaded, UpdateSupportPlugin) else loaded
    if not isinstance(value, UpdateSupportPlugin):
        raise TypeError(
            f"entry point {entry_point.name!r} did not load an UpdateSupportPlugin"
        )
    return value

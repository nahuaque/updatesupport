"""Lightweight plugin registry for domain-specific extensions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any


@dataclass(frozen=True)
class PluginMetadata:
    """Optional package metadata for plugin discovery and documentation."""

    package: str | None = None
    homepage: str | None = None
    domain: str | None = None
    tags: tuple[str, ...] = ()
    min_updatesupport_version: str | None = None

    def __post_init__(self) -> None:
        if self.tags is None:
            tags = ()
        elif isinstance(self.tags, str):
            tags = (self.tags,)
        else:
            try:
                tags = tuple(self.tags)
            except TypeError:
                tags = (self.tags,)
        object.__setattr__(self, "tags", tags)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly plugin metadata."""

        return {
            "package": self.package,
            "homepage": self.homepage,
            "domain": self.domain,
            "tags": list(self.tags),
            "min_updatesupport_version": self.min_updatesupport_version,
        }


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
    metadata: PluginMetadata = field(default_factory=PluginMetadata)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the exposed plugin surfaces."""

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata.as_dict(),
            "metrics": sorted(self.metrics),
            "q_presets": sorted(self.q_presets),
            "report_profiles": sorted(self.report_profiles),
            "compilers": sorted(self.compilers),
        }


@dataclass(frozen=True)
class PluginValidationIssue:
    """Validation issue for an updatesupport plugin descriptor."""

    severity: str
    code: str
    message: str
    surface: str | None = None
    key: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-friendly issue payload."""

        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "surface": self.surface,
            "key": self.key,
        }


@dataclass(frozen=True)
class PluginValidationReport:
    """Validation report for a plugin descriptor."""

    plugin_name: str
    issues: tuple[PluginValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[PluginValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[PluginValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly validation report."""

        return {
            "plugin_name": self.plugin_name,
            "ok": self.ok,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def raise_for_errors(self) -> None:
        """Raise ``ValueError`` if validation found blocking errors."""

        if self.ok:
            return
        messages = "; ".join(issue.message for issue in self.errors)
        raise ValueError(
            f"invalid updatesupport plugin {self.plugin_name!r}: {messages}"
        )


class PluginRegistry:
    """Mutable registry for explicit and entry-point-discovered plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, UpdateSupportPlugin] = {}

    def register(
        self,
        plugin: UpdateSupportPlugin,
        *,
        replace: bool = False,
        validate: bool = True,
    ) -> UpdateSupportPlugin:
        if validate:
            assert_valid_plugin(plugin)
        elif not plugin.name:
            raise ValueError("plugin name must be non-empty")
        existing = self._plugins.get(plugin.name)
        if existing is not None and existing is not plugin and not replace:
            raise ValueError(
                f"updatesupport plugin name is already registered: {plugin.name!r}"
            )
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

    def discover(
        self,
        *,
        group: str = "updatesupport.plugins",
        replace: bool = False,
        validate: bool = True,
    ) -> tuple[UpdateSupportPlugin, ...]:
        discovered = []
        for entry_point in _entry_points(group):
            plugin = _load_plugin(entry_point)
            self.register(plugin, replace=replace, validate=validate)
            discovered.append(plugin)
        return tuple(discovered)


_REGISTRY = PluginRegistry()


def register_plugin(
    plugin: UpdateSupportPlugin,
    *,
    replace: bool = False,
    validate: bool = True,
) -> UpdateSupportPlugin:
    """Register a plugin explicitly."""

    return _REGISTRY.register(plugin, replace=replace, validate=validate)


def unregister_plugin(name: str) -> None:
    """Remove a plugin from the process-local registry if present."""

    _REGISTRY.unregister(name)


def get_plugin(name: str) -> UpdateSupportPlugin:
    """Return a registered plugin by name."""

    return _REGISTRY.get(name)


def list_plugins() -> tuple[UpdateSupportPlugin, ...]:
    """Return registered plugins sorted by name."""

    return _REGISTRY.list()


def discover_plugins(
    *,
    group: str = "updatesupport.plugins",
    replace: bool = False,
    validate: bool = True,
) -> tuple[UpdateSupportPlugin, ...]:
    """Discover plugins declared through Python package entry points."""

    return _REGISTRY.discover(group=group, replace=replace, validate=validate)


def validate_plugin(plugin: UpdateSupportPlugin) -> PluginValidationReport:
    """Validate a plugin descriptor without mutating the registry."""

    issues: list[PluginValidationIssue] = []
    raw_name = getattr(plugin, "name", None)
    plugin_name = raw_name if isinstance(raw_name, str) else "<invalid>"

    if not isinstance(plugin, UpdateSupportPlugin):
        issues.append(
            PluginValidationIssue(
                severity="error",
                code="plugin-type",
                message="plugin must be an UpdateSupportPlugin",
            )
        )
        return PluginValidationReport(plugin_name=plugin_name, issues=tuple(issues))

    if not isinstance(plugin.name, str) or not plugin.name.strip():
        issues.append(
            PluginValidationIssue(
                severity="error",
                code="plugin-name",
                message="plugin name must be a non-empty string",
            )
        )
    elif plugin.name != plugin.name.strip() or not _is_plugin_name(plugin.name):
        issues.append(
            PluginValidationIssue(
                severity="error",
                code="plugin-name",
                message=(
                    "plugin name must contain only ASCII letters, numbers, '.', "
                    "'_', or '-'"
                ),
            )
        )

    _validate_optional_string(
        issues,
        plugin.version,
        code="plugin-version",
        message="plugin version must be a string or None",
    )
    if not isinstance(plugin.description, str):
        issues.append(
            PluginValidationIssue(
                severity="error",
                code="plugin-description",
                message="plugin description must be a string",
            )
        )

    if not isinstance(plugin.metadata, PluginMetadata):
        issues.append(
            PluginValidationIssue(
                severity="error",
                code="plugin-metadata",
                message="plugin metadata must be PluginMetadata",
            )
        )
    else:
        _validate_metadata(issues, plugin.metadata)

    for surface in ("metrics", "q_presets", "report_profiles", "compilers"):
        _validate_surface(issues, plugin, surface)

    return PluginValidationReport(plugin_name=plugin_name, issues=tuple(issues))


def assert_valid_plugin(plugin: UpdateSupportPlugin) -> PluginValidationReport:
    """Validate a plugin descriptor and raise on blocking errors."""

    report = validate_plugin(plugin)
    report.raise_for_errors()
    return report


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
    value = (
        loaded()
        if callable(loaded) and not isinstance(loaded, UpdateSupportPlugin)
        else loaded
    )
    if not isinstance(value, UpdateSupportPlugin):
        raise TypeError(
            f"entry point {entry_point.name!r} did not load an UpdateSupportPlugin"
        )
    return value


def _is_plugin_name(name: str) -> bool:
    return all(char.isascii() and (char.isalnum() or char in "._-") for char in name)


def _validate_optional_string(
    issues: list[PluginValidationIssue],
    value: object,
    *,
    code: str,
    message: str,
) -> None:
    if value is not None and not isinstance(value, str):
        issues.append(
            PluginValidationIssue(severity="error", code=code, message=message)
        )


def _validate_metadata(
    issues: list[PluginValidationIssue],
    metadata: PluginMetadata,
) -> None:
    for field_name in (
        "package",
        "homepage",
        "domain",
        "min_updatesupport_version",
    ):
        value = getattr(metadata, field_name)
        _validate_optional_string(
            issues,
            value,
            code=f"metadata-{field_name}",
            message=f"metadata {field_name} must be a string or None",
        )
    for tag in metadata.tags:
        if not isinstance(tag, str) or not tag:
            issues.append(
                PluginValidationIssue(
                    severity="error",
                    code="metadata-tags",
                    message="metadata tags must be non-empty strings",
                )
            )
            break


def _validate_surface(
    issues: list[PluginValidationIssue],
    plugin: UpdateSupportPlugin,
    surface: str,
) -> None:
    collection = getattr(plugin, surface)
    if not isinstance(collection, Mapping):
        issues.append(
            PluginValidationIssue(
                severity="error",
                code="surface-type",
                message=f"plugin {surface} must be a mapping",
                surface=surface,
            )
        )
        return

    for key, value in collection.items():
        if not isinstance(key, str) or not key:
            issues.append(
                PluginValidationIssue(
                    severity="error",
                    code="surface-key",
                    message=f"plugin {surface} keys must be non-empty strings",
                    surface=surface,
                    key=str(key),
                )
            )
        if not callable(value):
            issues.append(
                PluginValidationIssue(
                    severity="error",
                    code="surface-callable",
                    message=f"plugin {surface} value {key!r} must be callable",
                    surface=surface,
                    key=str(key),
                )
            )

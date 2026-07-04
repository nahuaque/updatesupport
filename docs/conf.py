"""Sphinx configuration for updatesupport."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "updatesupport"
author = "updatesupport contributors"
copyright = "2026, updatesupport contributors"

try:
    release = package_version("updatesupport")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "updatesupport"
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0f766e",
        "color-api-name": "#0f766e",
        "font-stack": (
            "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "
            "Segoe UI, sans-serif"
        ),
        "font-stack--monospace": (
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
        ),
    },
    "dark_css_variables": {
        "color-brand-primary": "#2dd4bf",
        "color-brand-content": "#5eead4",
        "color-api-name": "#5eead4",
    },
}

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

nitpicky = False

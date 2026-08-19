"""Sphinx configuration for the GeoBrain documentation.

The look is the GeoBrain house style: ``sphinx_book_theme`` with the
project's own palette in ``_static/geobrain.css``. What is deliberately
NOT here is machinery: no generated API stubs, no gallery pipeline, no
translation catalogues. Every page in this site is hand-written and every
code block in it is meant to run against the version of ``geobrain`` sitting
beside it, which is why the package is put on the path below rather than
being imported from an install.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Staged assets.
#
# Pictures the site shows are kept in one place and copied in at build time
# rather than duplicated into _static, because a duplicated binary drifts:
# the logo under _static had already fallen a version behind the one in
# assets/ and nothing in the build could notice.
FIGURE_DIR = Path(__file__).resolve().parent / "_figures"


def _stage(source: Path) -> str:
    """Copy one asset into the staging directory, returning its doc-relative path."""
    FIGURE_DIR.mkdir(exist_ok=True)
    target = FIGURE_DIR / source.name
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        target.write_bytes(source.read_bytes())
    return f"{FIGURE_DIR.name}/{target.name}"


project = "GeoBrain"
author = "Mingliang Liu"
copyright = "2026, Mingliang Liu"  # noqa: A001
# Read the version rather than importing geobrain: the docs must build even
# where torch is not installed. Matching on the assignment, not on the first
# quote in the file, because the file opens with a docstring.
_VERSION_FILE = (REPO_ROOT / "geobrain" / "_version.py").read_text(encoding="utf-8")
_match = re.search(r'^__version__\s*=\s*"([^"]+)"', _VERSION_FILE, re.M)
if _match is None:
    raise RuntimeError(f"no __version__ assignment in {REPO_ROOT / 'geobrain/_version.py'}")
release = _match.group(1)
version = release

# No autodoc, so no napoleon; no cross-reference roles, so no intersphinx.
# Dropping intersphinx also means the docs build with no network, which is
# what someone who has just cloned the repository has.
# `githubpages` writes a .nojekyll file into the output. Without it GitHub
# Pages runs the HTML through Jekyll, which drops every directory whose name
# starts with an underscore: _static, _images, _sources. The site would come
# back styleless and with no figures, and nothing in the build would warn.
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
]
myst_heading_anchors = 3

# One entry only: an empty theme-switcher.html that overrides the theme's,
# because this site has no dark palette to switch to.
templates_path: list[str] = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "GeoBrain: Differentiable Subsurface Modeling"
html_logo = _stage(REPO_ROOT / "assets" / "geobrain_logo.png")
html_static_path = ["_static"]
html_css_files = ["geobrain.css"]
# Loaded in the head, ahead of the theme's deferred script, so a stale
# `mode` in localStorage or a dark desktop cannot put this light-only
# stylesheet on a dark page. See the tail of _static/geobrain.css.
html_js_files = ["force-light.js"]
# Light only. The stylesheet defines one palette (a warm paper ground and
# the GeoBrain greens); it has no dark variant, so following the system
# preference leaves the theme dark and the custom colours light. Offering
# a switch that breaks the page is worse than not offering one, so the
# toggle is removed from the navbar as well.
html_context = {"default_mode": "light"}

html_theme_options = {
    "repository_url": "https://github.com/GeoBrain-Project/GeoBrain",
    "use_repository_button": True,
    "use_source_button": False,
    "use_issues_button": True,
    "show_navbar_depth": 1,
    "home_page_in_toc": True,
    "navbar_persistent": [],
    "navbar_end": ["navbar-icon-links"],
}

# Copy-button: strip the prompt from a pasted REPL line, keep everything else.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Figures.
#
# Every figure in this site was produced by a script in examples/, and none
# of them is drawn for the documentation. Sphinx will not read outside its
# source directory, so the gallery output is staged into docs/_figures/
# before the build starts and referenced from there as /_figures/<name>.png.
# The staging directory is generated and ignored by git; the originals under
# examples/*/out/ are the copies that are kept.
def _stage_figures(_app=None) -> list[str]:
    """Mirror examples/*/out/ into docs/_figures/, returning what was staged."""
    FIGURE_DIR.mkdir(exist_ok=True)
    staged = []
    for source in sorted((REPO_ROOT / "examples").glob("*/out/*")):
        if source.suffix not in {".png", ".gif"}:
            continue
        target = FIGURE_DIR / source.name
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            target.write_bytes(source.read_bytes())
        staged.append(target.name)
    return staged


def setup(app):
    app.connect("builder-inited", _stage_figures)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

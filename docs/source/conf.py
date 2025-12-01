# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
# Full options: https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# import sbss
# -- Path setup --------------------------------------------------------------
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path.absolute()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Scalable BSS Toolkit"
project_copyright = "National Institute of Advanced Industrial Science And Technology (AIST)"
author = "National Institute of Advanced Industrial Science And Technology (AIST)"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

language = "en"

# -- HTML output -------------------------------------------------
html_theme = "shibuya"
html_show_sourcelink = False
html_show_sphinx = False

html_static_path = ["_static"]
html_theme_options = {
    "accent_color": "blue",
    "nav_links": [
        {"title": "User Guide", "url": "user_guide/index"},
        {
            "title": "API Reference",
            "url": "api_reference/index",
            "children": [
                {
                    "title": "Common Utilities",
                    "url": "api_reference/common",
                    "summary": "sbss.common",
                },
                {
                    "title": "Neural FCA",
                    "url": "api_reference/nfca",
                    "summary": "sbss.nfca",
                },
            ],
        },
        {"title": "Recipes", "url": "recipes/index"},
    ],
    "github_url": "https://github.com/b-sigpro/sbss",
    "globaltoc_expand_depth": 1,
}

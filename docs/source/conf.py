# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../../src"))

from datetime import datetime

from sphinx_pyproject import SphinxConfig

config = SphinxConfig(pyproject_file="../../pyproject.toml")


project = config.name
copyright = f"{datetime.today().year}, {config.author}"

author = config.author
release = config.version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["build"]

source_suffix = [".rst", ".md"]

# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "torchvision": ("https://pytorch.org/vision/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/master/en", None),
    "diffusers": ("https://huggingface.co/docs/diffusers/main/en", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Tell myst-parser to assign header anchors for h1-h3.
myst_heading_anchors = 3

# By default, sort documented members by type within classes and modules.
autodoc_member_order = "groupwise"

python_use_unqualified_type_names = True

# Include default values when documenting parameter types.
typehints_defaults = "comma"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "https://github.com/organizations/py-img-gen/settings/profile"

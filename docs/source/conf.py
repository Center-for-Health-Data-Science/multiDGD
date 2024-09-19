import os
import sys
#sys.path.insert(0, os.path.abspath('../..'))

# Get the absolute path to the package.
path = os.path.abspath("../../src")

# Insert the path into the PATH.
sys.path.insert(0, path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'multiDGD'
copyright = '2024, Viktoria Schuster'
author = 'Viktoria Schuster'
#release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinxemoji.sphinxemoji',
    'sphinx_copybutton',
    'sphinxcontrib.bibtex'
]
autosummary_generate = True 

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    #"undoc-members": True,
    #"private-members": True
}

bibtex_bibfiles = ['references.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    #'analytics_anonymize_ip': False,
    #'logo_only': False,
    #'display_version': True,
    #'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    #'vcs_pageview_mode': '',
    #'style_nav_header_background': 'white',
    # Toc options
    #'collapse_navigation': True,
    #'sticky_navigation': True,
    #'navigation_depth': 4,
    #'includehidden': True,
    #'titles_only': False
}

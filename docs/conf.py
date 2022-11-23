# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from visbrain.config import CONFIG
import mne

# -- Project information -----------------------------------------------------

project = 'reg-connectivity'
copyright = '2022, Sara Sommariva Elisabetta Vallarino'
author = 'Sara Sommariva Elisabetta Vallarino'

# The full version, including alpha/beta/rc tags
release = '0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_gallery.gen_gallery', 
	      "sphinx.ext.napoleon",
              "pyvista.ext.plot_directive"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'

html_theme_options = {
    'navbar_sidebarrel': False,
    'navbar_pagenav': False,
    'globaltoc_includehidden': False,
    'navbar_links': [
#        ("API", "api"),
	("Single simulation", "auto_single_simulation/index"),
	("Paper results", "auto_paper_results/index"),
	("Run on cluster", "auto_paper_cluster/index"),
        ("GitHub", "https://github.com/sarasommariva/regconnectivity", True)
    ],
    'bootswatch_theme': "cerulean"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

examples_dirs = ['../examples_single_simulation', '../examples_paper_results', '../examples_paper_cluster']
gallery_dirs = ['auto_single_simulation', 'auto_paper_results', 'auto_paper_cluster', ]

sphinx_gallery_conf = {
     'examples_dirs': examples_dirs,   # path to your example scripts
     'gallery_dirs': gallery_dirs,  # path to where to save gallery generated output
     'image_scrapers': ('matplotlib', 'pyvista'), # figures extension to be embedded
     'default_thumb_file': os.path.join('_static', 'X_lamX.png'),
     #'within_subsection_order': FileNameSortKey
}

try:
    #mlab = mne.utils._import_mlab()
    find_mayavi_figures = True
    # Do not pop up any mayavi windows while running the
    # examples. These are very annoying since they steal the focus.
    mlab.options.offscreen = True
except Exception:
    find_mayavi_figures = False

#CONFIG['MPL_RENDER'] = True # Embed visbrain figures in the documentation

## Introduction

These notes provide an overview of the documentation generation process for Pywr-DRB.

## Jupyter-Books

Consider the following simplified depiction of the Pywr-DRB code base (note that `...` indicate other folders/files which are being ignored for the sake of this instruction):


```Example code structure
Pywr-DRB/
	.github/workflows/
		deploy.yml
	docs/ 
		_config.yml
		_toc.yml
		intro.md
		api/
			...
			post.rst
	pywrdrb/
		...
		post/
			__init__.py
			get_results.py
```


Inside the `Pywr-DRB/pywrdrb/post/get_results.py` script we have a function called `get_pywrdrb_results()` with it's own complete docstring. 

The `Pywr-DRB/pywrdrb/post/__init__.py` script is used to import the `get_pywrdrb_results()` function using a relative import (relative from the location of this `__init__.py` file), and simply contains:

```python
from .get_results import get_pywrdrb_results
```


We then tell autodocs to generate the documentation summary when building the JupyterBook site. This instruction is written in the `Pywr-DRB/docs/api/post.py` file. This file contains the following:

```python
pywrdrb.post      # Title for the page
================  # Specifies this is a header (must be as long as title) 

.. autosummary::          # This "directive" tells autodoc to make a table with function docs
   :toctree: generated/   # Tells autodoc to put new pages in ./generated/ folder
   :nosignatures:         # Only include the function name in the table

   pywrdrb.post.get_pywrdrb_results   # Include this function in the table
```

## Table of contents

The `_toc.yml` file defines the structure of the website. This structure resembles the following:

```yml
format: jb-book
# =============== Landing Page =================
root: intro.md
parts:
  - chapters:
    # =============== References =================
    - file: api/api.md
      sections:
        - title: Post-Processing
          file: api/post.rst
```

### Configuration file
The `docs/_config.py` file serves as the primary configuration file that dictates how the book is built and displayed. It provides a centralized place to define settings and parameters that control various aspects of the book.

Below is a look at the contents.  We shouldn't need to modify this anytime in the near future.  However, it is good to be familiar with it's contents and role. 

```yml
# Book settings
# Learn more at https://jupyterbook.org/customize/config.html

title: Pywr-DRB Integrated Water Resource Management Model
author: The Reed Group at Cornell CEE
logo: logo.png

# Force re-execution of notebooks on each build.
# See https://jupyterbook.org/content/execute.html
execute:
  execute_notebooks: force

# Enable autogeneration of API docs
sphinx:
  extra_extensions:
  - 'sphinx.ext.autodoc'
  - 'sphinx.ext.autosummary'
  - 'sphinx.ext.napoleon'
  config:
    autosummary_generate: True

# Define the name of the latex output file for PDF builds
latex:
  latex_documents:
    targetname: book.tex

# Add a bibtex file so that we can create citations
bibtex_bibfiles:
  - references.bib

# Information about where the book exists on the web
repository:
  url: https://github.com/Pywr-DRB/Pywr-DRB  # Online location of your book
  path_to_book: docs  # Optional path to your book, relative to the repository root

# Add GitHub buttons to your book
# See https://jupyterbook.org/customize/config.html#add-a-link-to-your-repository
html:
  use_issues_button: true
  use_repository_button: true
```

***
## ReStructuredText + Sphinx + autodocs

[reStructuredText](https://docutils.sourceforge.io/rst.html) (reST; `file.rst`) is a plain text markup language used primarily for technical documentation. These files are able to be read with Sphinx which generates the documentation for Pywr-DRB. 

Key features of reST files include:
- Headings: Defined by underline characters (=, -, ~, etc.).
- Lists: Support for both ordered and unordered lists.
- Directives: Special commands for including content like code blocks, tables, or images (e.g., .. code-block:: python).
- Roles: Inline markup for references and links (e.g., :ref:`target` ).

Here is a [primer/intro page](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) about working with `.rst` files.

Inside of the `.rst` files we include autodoc code, which [the Sphinx autodoc extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc) uses to produce the documentation pages. These different formatting options and commands are explained below, but it will be helpful to refer to the [sphinx documentation](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) for many issues. This can be finicky. 

Notice the Warning on the `sphinx.ext.autodoc` site:

> **Warning**
> autodoc imports the modules to be documented. If any modules have side effects on import, these will be executed by autodoc when sphinx-build is run.
> 
> If you document scripts (as opposed to library modules), make sure their main routine is protected by a if __name__ == '__main__' condition.


### Directives 

Directives in reStructuredText files are special instructions that provide additional information or functionality within the document. They start with a `..` and are followed by the directive name and `::`.  Below the directive, you provide optional arguments and/or content.

Here is an example of a `code-block` directive:

```rst
.. code-block:: python

   def hello_world():
       print("Hello, World!")

```

The autodoc extension in Sphinx uses specific directives to automatically generate documentation from docstrings in the source code. 

Specifically, we use the `.. autosummary::` and `.. automodule::` directives, described below. 

#### `.. atuosummary::`
The `.. autosummary` directive in Sphinx is used to generate summary tables for modules, classes, functions, and methods. It is particularly useful for creating organized documentation with minimal effort, as it can automatically extract information from docstrings.

**Options:**
- `:toctree:`
	- Specifies the directory where the auto-generated .rst files should be placed. This is necessary for creating links to the detailed documentation of each item.
	- For the Pywr-DRB documentation, we make a new folder called "generated/" where all of the documentation will be stored. I.e., you should specify: `:toctree: generated/`
- `:nosignatures:`
	- If present, function and method signatures will not be shown in the summary table.
- `:recursive:`
	- If specified, recursively document all modules.

**Example:**
```rst
STARFIT Reservoir Operations
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   
    pywrdrb.parameters.STARFITReservoirRelease
```

#### `.. automodule:: <pywrdrb.module>`

This directive is used to automatically document a module in Sphinx. The argument `pywrdrb.module` specifies the module to be documented.

Options:
- `:members:`
	- This option includes all public members (functions, classes, variables, etc.) of the module in the documentation. It ensures that all the elements defined within `pywrdrb.module` are documented.
- `:undoc-members:`
	- This option includes members that do not have a docstring. By default, Sphinx only includes members with docstrings, but this option ensures that all members, documented or not, are included.
- `:show-inheritance:`
	- This option includes a list of base classes for each class in the module documentation. It shows the inheritance hierarchy of classes, which can be helpful for understanding the class structure and relationships.


***
## Troubleshooting build errors

### API build errors:

- `Extension error (sphinx.ext.autosummary): Handler <function process_generate_options at 0x7f9eaf6a8b80> for event 'builder-inited' threw an exception (exception: no module named pywrdrb.pre)`
	- Make sure the module has the correct `__init__.py` file 
	- Make sure all of the packages used in this module are installed in the virtual environment used by GitActions. 

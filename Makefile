## ----------------------------------------------------------------------
## Smart Invoice
## A simple Makefile that helps managing the project.
## ----------------------------------------------------------------------

# | Variables
SRC_DIR = .
NOTEBOOKS_DIR = notebooks
TESTS_DIR = unit_tests

# | Actions
help:     		## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

test:	  		## Run tests
	@python -m unittest discover $(TESTS_DIR) -p *_test.py

notebook:		## Convert all notebooks in "notebooks" dir into Markdown.
	@jupytext --to ipynb $(NOTEBOOKS_DIR)/*.md

markdown:		## Convert all Markdown file in "notebooks" dir into Jupyter Notebooks.
	@jupytext --to md $(NOTEBOOKS_DIR)/*.ipynb

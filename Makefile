#
# Installation
#

.PHONY: setup
setup:
	pip install -U uv

.PHONY: install
install:
	uv sync --all-extras

.PHONY: install-doc
install-doc:
	uv sync --extra doc

#
# linter/formatter/typecheck
#

.PHONY: lint
lint: install
	uv run ruff check --output-format=github .

.PHONY: format
format: install
	uv run ruff format --check --diff .

.PHONY: typecheck
typecheck: install
	uv run mypy --cache-dir=/dev/null .

.PHONY: test
test: install
	uv run pytest -vsx --log-cli-level=INFO

.PHONY: html
html: install-doc
	uv run sphinx-build -M html docs/source docs/build

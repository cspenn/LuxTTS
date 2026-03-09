#!/bin/bash
set -e
echo "Running ruff check..."
uv run ruff check zipvoice/
echo "Running ruff format check..."
uv run ruff format --check zipvoice/
echo "Running mypy..."
uv run mypy zipvoice/
echo "Running pytest..."
uv run pytest tests/
echo "Running deptry..."
uv run deptry zipvoice/
echo "All checks passed!"

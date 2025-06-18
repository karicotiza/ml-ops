# ml-hub

Hub for saving models, tracking their metrics and automatic inference.

## Code quality

The code of the ml-hub passes the following checks without any errors
and warnings:

* `uv run pytest --cov=ml_hub --doctest-modules`
* `uv run ruff check ./ml_hub/ ./tests/`
* `uv run flake8 ./ml_hub/ ./tests/`
* `uv run black --diff --check ./ml_hub/ ./tests/`
* `uv run pyright ./ml_hub/ ./tests/`
* `uv run mypy ./ml_hub/ ./tests/`

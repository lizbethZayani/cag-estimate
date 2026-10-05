"""Tests for the ``cag-estimate`` console script entry point."""

import importlib
from unittest.mock import patch

import cag_estimate


def test_main_runs_uvicorn_on_the_fastapi_app():
    # Importing the ``cag_estimate.main`` submodule elsewhere shadows the
    # ``main`` function attribute; reload restores the package namespace.
    importlib.reload(cag_estimate)
    with patch("uvicorn.run") as run:
        cag_estimate.main()

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == "cag_estimate.main:app"
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000

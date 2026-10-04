def main() -> None:
    """Console script entry point: serve the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run("cag_estimate.main:app", host="0.0.0.0", port=8000)

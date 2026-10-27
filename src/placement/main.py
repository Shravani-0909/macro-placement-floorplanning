"""Application entry point for running the placement API server."""

import uvicorn

from .api import app

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run("src.placement.api:app", host="0.0.0.0", port=8000, reload=True)

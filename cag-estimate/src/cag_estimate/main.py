"""
FastAPI application for CAG (Context-Augmented Generation) project estimation.

This API uses Context-Augmented Generation to generate detailed project estimations
based on meeting transcriptions. The system analyzes meeting notes and generates
task breakdowns with hours, costs, and resource requirements.

Access the interactive API documentation at /docs (Swagger UI) or /redoc (ReDoc).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cag_estimate.config import get_settings
from cag_estimate.logging_config import configure_logging
from cag_estimate.routers import estimations, sessions


# Health check response model
class HealthStatus(BaseModel):
    """Health check response."""

    status: str
    service: str


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging once at startup from ``Settings.log_level``."""
    configure_logging(get_settings().log_level)
    yield


# Create FastAPI app with detailed documentation
app = FastAPI(
    lifespan=lifespan,
    title="CAG Estimate API",
    description="Context-Augmented Generation for Project Estimation - Generates detailed project estimations based on meeting transcriptions",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(estimations.router)
app.include_router(sessions.router)


@app.get("/health", response_model=HealthStatus, status_code=200)
async def health_check() -> HealthStatus:
    """
    Health check endpoint to verify service status.

    Returns:
        HealthStatus with status "ok" if service is running
    """
    return HealthStatus(status="ok", service="CAG Estimate API")


@app.get("/", tags=["info"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "CAG Estimate API",
        "version": "0.1.0",
        "description": "Context-Augmented Generation for Project Estimation",
        "docs": "/docs",
        "health": "/health",
    }

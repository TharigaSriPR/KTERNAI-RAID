"""
Main FastAPI Application Entry Point.
Initializes the HTTP service for OAuth 2.0 flows, authentication, and email synchronization.

Canonical Authentication Prefix:
    /api/v1/auth
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ALLOWED_ORIGINS
from src.api.routes_auth import router as auth_router
from src.database.db_manager import DatabaseManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database tables are initialized on startup."""
    db = DatabaseManager()
    db.init_db()
    yield


app = FastAPI(
    title="KTern.AI RAID Identification Agent API",
    description="Multi-user OAuth 2.0 and Email Intelligence Platform Backend",
    version="2.0.0",
    lifespan=lifespan
)

# Enforce strictly configured trusted CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "X-Session-Token", "Content-Type", "Accept"],
)

# Register routers
app.include_router(auth_router)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "healthy", "service": "raid-agent-api", "version": "2.0.0"}

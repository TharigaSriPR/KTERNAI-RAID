"""
FastAPI Authentication Routes.
Exposes endpoints for user registration, login, logout, session rotation, and current user retrieval.

Canonical authentication API prefix:
    /api/v1/auth

Exposed routes:
    POST /api/v1/auth/register       - Register a new user account
    POST /api/v1/auth/login          - Authenticate credentials and create active session token
    POST /api/v1/auth/logout         - Invalidate active session token
    POST /api/v1/auth/rotate-session - Rotate existing valid session token
    GET  /api/v1/auth/me             - Retrieve current authenticated user profile
"""

import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Request, Response, status
from pydantic import BaseModel, Field, field_validator

from src.auth.session_manager import SessionManager, AuthenticationError, RateLimitExceededError
from src.database.db_manager import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# Global session manager instance
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager


EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., description="Unique email address")
    password: str = Field(..., min_length=8, description="Plaintext password")
    full_name: Optional[str] = Field(None, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format.")
        return clean


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., description="Registered username or email")
    password: str = Field(..., description="Account password")


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    role: str = "user"


class LoginResponse(BaseModel):
    session_token: str
    user: UserResponse
    expires_at: str


class RotateSessionResponse(BaseModel):
    session_token: str
    user_id: str
    username: str
    expires_at: str


class MessageResponse(BaseModel):
    message: str
    success: bool = True


# ---------------------------------------------------------------------------
# Dependency: Extract & Validate Authenticated User
# ---------------------------------------------------------------------------

def get_current_token(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token")
) -> str:
    """
    Extracts raw bearer token from:
    1. Authorization header (Bearer <token>)
    2. X-Session-Token header
    3. session_token HttpOnly cookie (for browser redirects such as OAuth callback)
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif x_session_token:
        token = x_session_token.strip()
    elif request.cookies.get("session_token"):
        token = request.cookies.get("session_token").strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Missing Bearer header, X-Session-Token header, or session cookie."
        )
    return token


def get_current_user(
    token: str = Depends(get_current_token),
    sm: SessionManager = Depends(get_session_manager)
) -> dict:
    """Validates session token and returns authenticated user dictionary."""
    user = sm.validate_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked session token."
        )
    return user


# ---------------------------------------------------------------------------
# Route Implementations
# ---------------------------------------------------------------------------

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    sm: SessionManager = Depends(get_session_manager)
):
    """Registers a new user account."""
    try:
        user = sm.register_user(
            username=payload.username,
            email=payload.email,
            plain_password=payload.password,
            full_name=payload.full_name
        )
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Registration error: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed.")


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    sm: SessionManager = Depends(get_session_manager)
):
    """Authenticates credentials, respects rate limits, issues a hashed session, and sets session cookie."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    user_agent = request.headers.get("User-Agent", "Unknown")

    try:
        session_info = sm.authenticate(
            username_or_email=payload.username_or_email,
            plain_password=payload.password,
            ip_address=client_ip,
            user_agent=user_agent
        )
        secure_cookie = request.url.scheme == "https"
        response.set_cookie(
            key="session_token",
            value=session_info["session_token"],
            httponly=True,
            samesite="lax",
            secure=secure_cookie,
            path="/"
        )
        return {
            "session_token": session_info["session_token"],
            "user": {
                "id": session_info["user_id"],
                "username": session_info["username"],
                "email": session_info["email"],
                "full_name": session_info.get("full_name"),
                "role": session_info.get("role", "user")
            },
            "expires_at": session_info["expires_at"]
        }
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc)
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc)
        )


@router.post("/rotate-session", response_model=RotateSessionResponse)
def rotate_session(
    request: Request,
    response: Response,
    token: str = Depends(get_current_token),
    sm: SessionManager = Depends(get_session_manager)
):
    """Atomically rotates the session token, invalidating the old token and updating session cookie."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    user_agent = request.headers.get("User-Agent", "Unknown")

    try:
        result = sm.rotate_session(token, ip_address=client_ip, user_agent=user_agent)
        secure_cookie = request.url.scheme == "https"
        response.set_cookie(
            key="session_token",
            value=result["session_token"],
            httponly=True,
            samesite="lax",
            secure=secure_cookie,
            path="/"
        )
        return result
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


@router.post("/logout", response_model=MessageResponse)
def logout(
    response: Response,
    token: str = Depends(get_current_token),
    sm: SessionManager = Depends(get_session_manager)
):
    """Invalidates the caller's session token and clears the session cookie."""
    success = sm.logout(token)
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Session invalidated successfully." if success else "Session already inactive.", "success": success}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the profile of the currently authenticated user."""
    return {
        "id": current_user["user_id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user.get("full_name"),
        "role": current_user.get("role", "user")
    }

"""
Session management and user authentication layer.
Enforces:
- SHA-256 hashed session token storage (bearer tokens never stored in plaintext).
- Server-side session verification with 24-hour expiration and activity tracking.
- Session rotation and logout invalidation.
- Login rate limiting and progressive account lockout (5 attempts / 15-minute lock).
"""

import uuid
import secrets
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

from src.database.db_manager import DatabaseManager
from src.auth.password import PasswordSecurity
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication credentials or session tokens are invalid."""
    pass


class RateLimitExceededError(AuthenticationError):
    """Raised when consecutive failed logins trigger account or IP lockout."""
    def __init__(self, message: str, locked_until: Optional[str] = None):
        super().__init__(message)
        self.locked_until = locked_until


class SessionManager:
    """
    Coordinates password verification, session lifecycle, and rate limiting.
    """

    def __init__(self, db: Optional[DatabaseManager] = None, session_ttl_hours: int = 24):
        self.db = db or DatabaseManager()
        self.password_security = PasswordSecurity()
        self.session_ttl = timedelta(hours=session_ttl_hours)

    @staticmethod
    def hash_token(raw_token: str) -> str:
        """Computes SHA-256 digest of a raw bearer token."""
        return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    def register_user(
        self,
        username: str,
        email: str,
        plain_password: str,
        full_name: Optional[str] = None,
        role: str = "user"
    ) -> Dict[str, Any]:
        """Registers a new application user account."""
        clean_username = username.strip()
        clean_email = email.strip().lower()

        if len(clean_username) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        if "@" not in clean_email or "." not in clean_email:
            raise ValueError("A valid email address is required.")
        if len(plain_password) < 8:
            raise ValueError("Password must be at least 8 characters long.")

        # Check existing user
        if self.db.get_user_by_username_or_email(clean_username) or self.db.get_user_by_username_or_email(clean_email):
            raise ValueError("A user with this username or email already exists.")

        password_hash, salt, algorithm = self.password_security.hash_password(plain_password)
        user_id = f"user-{uuid.uuid4().hex[:12]}"

        user_data = {
            "id": user_id,
            "username": clean_username,
            "email": clean_email,
            "password_hash": password_hash,
            "salt": salt,
            "algorithm": algorithm,
            "full_name": full_name or clean_username,
            "role": role
        }
        self.db.create_user(user_data)
        logger.info("Successfully registered new user: %s (id=%s)", clean_username, user_id)

        return {
            "id": user_id,
            "username": clean_username,
            "email": clean_email,
            "full_name": user_data["full_name"],
            "role": role
        }

    def authenticate(
        self,
        username_or_email: str,
        plain_password: str,
        ip_address: str = "127.0.0.1",
        user_agent: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Authenticates user with rate limiting and returns a new session.
        """
        clean_id = username_or_email.strip()

        # 1. Rate limiting check
        lockout = self.db.get_login_lockout(clean_id, ip_address)
        if lockout:
            locked_until = lockout.get("locked_until")
            logger.warning("Rejected login attempt for locked account/IP: %s from %s", clean_id, ip_address)
            raise RateLimitExceededError(
                "Too many failed login attempts. Please try again later.",
                locked_until=locked_until
            )

        # 2. Lookup user
        user = self.db.get_user_by_username_or_email(clean_id)
        if not user:
            # Mitigation for timing side-channels: execute dummy constant-time verification
            self.password_security.verify_dummy(plain_password)
            res = self.db.record_failed_login(clean_id, ip_address)
            if res.get("is_locked"):
                raise RateLimitExceededError(
                    "Too many failed login attempts. Please try again later.",
                    locked_until=res.get("locked_until")
                )
            raise AuthenticationError("Invalid username/email or password.")

        # 3. Verify password
        is_valid = self.password_security.verify_password(
            plain_password=plain_password,
            password_hash=user["password_hash"],
            salt=user.get("salt", ""),
            algorithm=user.get("algorithm", "argon2id")
        )
        if not is_valid:
            res = self.db.record_failed_login(clean_id, ip_address)
            if res.get("is_locked"):
                raise RateLimitExceededError(
                    "Too many failed login attempts. Please try again later.",
                    locked_until=res.get("locked_until")
                )
            raise AuthenticationError("Invalid username/email or password.")

        # 4. Successful login: reset failed attempts
        self.db.reset_failed_logins(clean_id, ip_address)

        # 5. Generate raw bearer token and store SHA-256 hash
        raw_token = secrets.token_urlsafe(32)
        token_hash = self.hash_token(raw_token)
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        expires_at = (now + self.session_ttl).isoformat()

        session_record = {
            "id": session_id,
            "user_id": user["id"],
            "session_token_hash": token_hash,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "expires_at": expires_at
        }
        self.db.create_session(session_record)
        logger.info("User authenticated: %s (session_id=%s)", user["username"], session_id)

        return {
            "session_token": raw_token,
            "user_id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user.get("full_name"),
            "role": user.get("role", "user"),
            "expires_at": expires_at
        }

    def validate_session(self, raw_token: str) -> Optional[Dict[str, Any]]:
        """
        Validates raw bearer token by checking its SHA-256 hash in SQLite.
        Returns user information if valid; None if expired or invalid.
        """
        if not raw_token or not isinstance(raw_token, str):
            return None

        token_hash = self.hash_token(raw_token)
        session = self.db.get_session_by_hash(token_hash)
        if not session:
            return None

        # Check expiration
        try:
            expires_dt = datetime.fromisoformat(session["expires_at"].replace("Z", "+00:00"))
            if expires_dt < datetime.now(timezone.utc):
                self.db.invalidate_session(token_hash)
                logger.info("Session %s has expired and was invalidated.", session["id"])
                return None
        except Exception:
            return None

        # Update last active time
        self.db.update_session_activity(session["id"])

        return {
            "session_id": session["id"],
            "user_id": session["user_id"],
            "username": session["username"],
            "email": session["email"],
            "full_name": session.get("full_name"),
            "role": session.get("role", "user"),
            "expires_at": session["expires_at"]
        }

    def rotate_session(
        self,
        raw_old_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rotates an existing active session: invalidates old token and issues a fresh one.
        """
        old_hash = self.hash_token(raw_old_token)
        new_raw_token = secrets.token_urlsafe(32)
        new_hash = self.hash_token(new_raw_token)
        new_session_id = f"sess-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        new_expires_at = (now + self.session_ttl).isoformat()

        user_id = self.db.rotate_session(
            old_token_hash=old_hash,
            new_session_id=new_session_id,
            new_token_hash=new_hash,
            new_expires_at=new_expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        if not user_id:
            raise AuthenticationError("Invalid or inactive session token for rotation.")

        user = self.db.get_user_by_id(user_id)
        logger.info("Session rotated for user_id=%s (new_session_id=%s)", user_id, new_session_id)

        return {
            "session_token": new_raw_token,
            "user_id": user_id,
            "username": user["username"] if user else "",
            "expires_at": new_expires_at
        }

    def logout(self, raw_token: str) -> bool:
        """Invalidates a session token."""
        if not raw_token:
            return False
        token_hash = self.hash_token(raw_token)
        return self.db.invalidate_session(token_hash)

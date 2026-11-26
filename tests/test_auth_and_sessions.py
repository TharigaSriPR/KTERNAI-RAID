"""
Automated Unit and Integration Tests for Phase 1: Authentication and Session Security.
Covers:
- Argon2id password hashing and explicit error on missing dependency (no silent fallback).
- Explicitly configured PBKDF2-HMAC-SHA256 support and hash identification.
- User enumeration resistance (constant-time rejection & uniform error messages).
- Multi-device session isolation (rotating/logging out Device A does not invalidate Device B).
- Secure session management with SHA-256 hashed session tokens.
- Session expiration, activity tracking, rotation, and logout invalidation.
- Login rate limiting, progressive account lockout, and distributed IP brute-force protection.
- FastAPI authentication API endpoints (register, login, me, rotate-session, logout) and CORS headers.
"""

import pytest
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from src.database.db_manager import DatabaseManager
from src.auth import password as password_module
from src.auth.password import PasswordSecurity
from src.auth.session_manager import SessionManager, AuthenticationError, RateLimitExceededError
from src.api.main import app
from config import CORS_ALLOWED_ORIGINS


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite DatabaseManager instance in temporary storage."""
    db_file = tmp_path / "test_auth.db"
    db = DatabaseManager(db_file)
    db.init_db()
    return db


@pytest.fixture
def session_mgr(temp_db):
    """Provides a SessionManager wired to the isolated temporary database."""
    return SessionManager(db=temp_db, session_ttl_hours=24)


# ---------------------------------------------------------------------------
# 1. Password Security & Hashing Tests (Argon2id + PBKDF2)
# ---------------------------------------------------------------------------

def test_argon2id_hashing_and_verification():
    ps = PasswordSecurity(default_algorithm="argon2id")
    pwd = "SecureP@ssword123!"

    pwd_hash, salt, algo = ps.hash_password(pwd)
    assert pwd_hash is not None
    assert pwd_hash.startswith("$argon2id$")
    assert algo == "argon2id"

    # Correct password succeeds
    assert ps.verify_password(pwd, pwd_hash, salt=salt, algorithm=algo) is True

    # Incorrect password fails in constant time
    assert ps.verify_password("WrongPassword!", pwd_hash, salt=salt, algorithm=algo) is False

    # Empty inputs handled safely
    assert ps.verify_password("", pwd_hash, salt=salt, algorithm=algo) is False
    assert ps.verify_password(pwd, "", salt=salt, algorithm=algo) is False


def test_argon2id_missing_dependency_fails_explicitly(monkeypatch):
    """Verifies that Argon2id never silently falls back and fails clearly if unavailable."""
    monkeypatch.setattr(password_module, "ARGON2_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="Argon2id hashing is configured as the default"):
        PasswordSecurity(default_algorithm="argon2id")

    ps = PasswordSecurity(default_algorithm="pbkdf2_sha256")
    with pytest.raises(RuntimeError, match="Argon2id hashing requested, but argon2-cffi is not installed"):
        ps.hash_password("ValidPassword123!", algorithm="argon2id")


def test_pbkdf2_explicit_configuration():
    """Verifies that PBKDF2 works when explicitly configured and is identifiable by format."""
    ps = PasswordSecurity(default_algorithm="pbkdf2_sha256")
    pwd = "CompliancePassword2026$"

    pwd_hash, salt, algo = ps.hash_password(pwd, algorithm="pbkdf2_sha256")
    assert algo == "pbkdf2_sha256"
    assert pwd_hash.startswith("$pbkdf2-sha256$600000$")
    assert len(salt) == 64  # 32 bytes hex

    assert ps.verify_password(pwd, pwd_hash, salt=salt, algorithm=algo) is True
    assert ps.verify_password("WrongPassword", pwd_hash, salt=salt, algorithm=algo) is False


def test_password_invalid_input():
    ps = PasswordSecurity()
    with pytest.raises(ValueError):
        ps.hash_password("")


# ---------------------------------------------------------------------------
# 2. User Enumeration Resistance & Timing Mitigation
# ---------------------------------------------------------------------------

def test_user_enumeration_resistance(session_mgr):
    """
    Verifies that invalid username and invalid password return the EXACT same error message,
    eliminating user-enumeration vulnerabilities.
    """
    session_mgr.register_user("real_user", "real@kaartech.com", "CorrectPassword123!")

    # Nonexistent user
    with pytest.raises(AuthenticationError) as exc_nonexistent:
        session_mgr.authenticate("nonexistent_user", "AnyPassword123!")

    # Existing user with wrong password
    with pytest.raises(AuthenticationError) as exc_wrong_pwd:
        session_mgr.authenticate("real_user", "WrongPassword123!")

    # Both errors must be identical
    assert str(exc_nonexistent.value) == str(exc_wrong_pwd.value)
    assert str(exc_nonexistent.value) == "Invalid username/email or password."


# ---------------------------------------------------------------------------
# 3. User Registration & Session Hashing Tests
# ---------------------------------------------------------------------------

def test_user_registration(session_mgr):
    user = session_mgr.register_user(
        username="alice_pm",
        email="alice@kaartech.com",
        plain_password="StrongPassword123!",
        full_name="Alice Smith"
    )
    assert user["id"].startswith("user-")
    assert user["username"] == "alice_pm"
    assert user["email"] == "alice@kaartech.com"

    # Duplicate username or email rejected
    with pytest.raises(ValueError, match="already exists"):
        session_mgr.register_user("alice_pm", "other@kaartech.com", "Password123!")

    with pytest.raises(ValueError, match="already exists"):
        session_mgr.register_user("other_user", "alice@kaartech.com", "Password123!")


def test_hashed_session_token_storage(session_mgr, temp_db):
    session_mgr.register_user("bob_lead", "bob@kaartech.com", "SecureBobPassword1!")
    auth_res = session_mgr.authenticate("bob_lead", "SecureBobPassword1!")

    raw_token = auth_res["session_token"]
    assert len(raw_token) >= 32

    # Query the database directly to inspect the stored record
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_token_hash, is_active FROM user_sessions WHERE user_id = ?", (auth_res["user_id"],))
        rows = cursor.fetchall()
        assert len(rows) == 1
        stored_hash = rows[0]["session_token_hash"]
        is_active = rows[0]["is_active"]

    # Verify SHA-256 hash matches
    expected_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    assert stored_hash == expected_hash
    assert is_active == 1

    # Verify the raw bearer token is NEVER stored in SQLite
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM user_sessions WHERE session_token_hash = ?", (raw_token,))
        assert cursor.fetchone() is None


def test_session_validation(session_mgr):
    session_mgr.register_user("charlie_dev", "charlie@kaartech.com", "CharliePassword123!")
    auth = session_mgr.authenticate("charlie@kaartech.com", "CharliePassword123!")
    raw_token = auth["session_token"]

    # Valid token resolves user
    user = session_mgr.validate_session(raw_token)
    assert user is not None
    assert user["user_id"] == auth["user_id"]
    assert user["username"] == "charlie_dev"

    # Tampered or bogus token rejected
    assert session_mgr.validate_session("invalid-token-12345") is None
    assert session_mgr.validate_session("") is None


def test_session_expiration(session_mgr, temp_db):
    session_mgr.register_user("david_qa", "david@kaartech.com", "DavidPassword123!")
    auth = session_mgr.authenticate("david_qa", "DavidPassword123!")
    raw_token = auth["session_token"]
    token_hash = session_mgr.hash_token(raw_token)

    # Manually expire the session in the database
    past_date = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE user_sessions SET expires_at = ? WHERE session_token_hash = ?", (past_date, token_hash))
        conn.commit()

    # Expired token returns None and gets marked inactive
    assert session_mgr.validate_session(raw_token) is None

    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_active FROM user_sessions WHERE session_token_hash = ?", (token_hash,))
        assert cursor.fetchone()["is_active"] == 0


def test_multi_device_session_isolation(session_mgr):
    """
    Verifies that rotating or logging out Device A does NOT invalidate Device B's session.
    """
    session_mgr.register_user("multi_dev", "multi@kaartech.com", "MultiPassword123!")

    # Device A login
    auth_a = session_mgr.authenticate("multi_dev", "MultiPassword123!", ip_address="192.168.1.10", user_agent="Laptop")
    token_a = auth_a["session_token"]

    # Device B login
    auth_b = session_mgr.authenticate("multi_dev", "MultiPassword123!", ip_address="192.168.1.20", user_agent="Mobile")
    token_b = auth_b["session_token"]

    assert token_a != token_b
    assert session_mgr.validate_session(token_a) is not None
    assert session_mgr.validate_session(token_b) is not None

    # Rotate Device A
    rotation_a = session_mgr.rotate_session(token_a)
    token_a_new = rotation_a["session_token"]

    # Old Device A token invalidated
    assert session_mgr.validate_session(token_a) is None
    # New Device A token valid
    assert session_mgr.validate_session(token_a_new) is not None
    # Device B session MUST remain active and valid!
    assert session_mgr.validate_session(token_b) is not None

    # Logout Device A
    session_mgr.logout(token_a_new)
    assert session_mgr.validate_session(token_a_new) is None
    # Device B STILL remains active!
    assert session_mgr.validate_session(token_b) is not None


# ---------------------------------------------------------------------------
# 4. Login Rate Limiting & Account Lockout Tests
# ---------------------------------------------------------------------------

def test_login_rate_limiting_lockout(session_mgr):
    session_mgr.register_user("grace_lead", "grace@kaartech.com", "GraceCorrectPassword1!")
    target = "grace_lead"
    ip_attacker = "192.168.1.100"
    ip_legit = "192.168.1.200"

    # 4 failed attempts from attacker IP raise generic error
    for _ in range(4):
        with pytest.raises(AuthenticationError, match="Invalid username/email or password"):
            session_mgr.authenticate(target, "WrongPassword", ip_address=ip_attacker)

    # 5th failed attempt triggers lockout
    with pytest.raises(RateLimitExceededError, match="Too many failed login attempts"):
        session_mgr.authenticate(target, "WrongPassword", ip_address=ip_attacker)

    # Attacker IP remains locked out
    with pytest.raises(RateLimitExceededError, match="Too many failed login attempts"):
        session_mgr.authenticate(target, "GraceCorrectPassword1!", ip_address=ip_attacker)

    # Legitimate user from a DIFFERENT IP can still authenticate without DoS lockout!
    auth_legit = session_mgr.authenticate(target, "GraceCorrectPassword1!", ip_address=ip_legit)
    assert auth_legit["session_token"] is not None


def test_global_ip_rate_limiting(session_mgr):
    """Verifies that an IP attempting credential stuffing across multiple accounts is blocked globally."""
    ip_scanner = "10.0.0.99"
    for i in range(15):
        try:
            session_mgr.authenticate(f"user_{i}", "SomePassword123!", ip_address=ip_scanner)
        except AuthenticationError:
            pass

    # The 16th attempt from this IP is blocked regardless of username
    with pytest.raises(RateLimitExceededError, match="Too many failed login attempts"):
        session_mgr.authenticate("any_other_user", "AnyPassword!", ip_address=ip_scanner)


def test_successful_login_resets_failed_attempts(session_mgr, temp_db):
    session_mgr.register_user("harry_eng", "harry@kaartech.com", "HarryPassword123!")
    ip = "192.168.1.101"

    # 2 failed attempts
    for _ in range(2):
        with pytest.raises(AuthenticationError):
            session_mgr.authenticate("harry_eng", "Wrong123!", ip_address=ip)

    # Successful login
    auth = session_mgr.authenticate("harry_eng", "HarryPassword123!", ip_address=ip)
    assert auth["session_token"] is not None

    # Verify attempt records for that identifier+IP were cleared
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM login_attempts WHERE identifier = ? AND ip_address = ?", ("harry_eng", ip))
        assert cursor.fetchone() is None


# ---------------------------------------------------------------------------
# 5. FastAPI Authentication API Endpoint & CORS Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(temp_db):
    """Provides a TestClient for FastAPI with SessionManager pointing to temp_db."""
    sm = SessionManager(db=temp_db)
    from src.api.routes_auth import get_session_manager
    app.dependency_overrides[get_session_manager] = lambda: sm
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_fastapi_auth_lifecycle(api_client):
    # 1. Register
    reg_resp = api_client.post("/api/v1/auth/register", json={
        "username": "api_user",
        "email": "api_user@kaartech.com",
        "password": "ApiUserSecure123!",
        "full_name": "API Test User"
    })
    assert reg_resp.status_code == 201
    user_data = reg_resp.json()
    assert user_data["username"] == "api_user"

    # 2. Login
    login_resp = api_client.post("/api/v1/auth/login", json={
        "username_or_email": "api_user",
        "password": "ApiUserSecure123!"
    })
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    token = login_data["session_token"]
    assert token is not None

    # 3. GET /me with Bearer token
    me_resp = api_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "api_user"

    # 4. GET /me with X-Session-Token header
    me_resp2 = api_client.get("/api/v1/auth/me", headers={"X-Session-Token": token})
    assert me_resp2.status_code == 200
    assert me_resp2.json()["email"] == "api_user@kaartech.com"

    # 5. Rotate Session
    rotate_resp = api_client.post("/api/v1/auth/rotate-session", headers={"Authorization": f"Bearer {token}"})
    assert rotate_resp.status_code == 200
    new_token = rotate_resp.json()["session_token"]
    assert new_token != token

    # 6. Verify old token is 401 Unauthorized
    old_me = api_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert old_me.status_code == 401

    # 7. Verify new token is 200 OK
    new_me = api_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {new_token}"})
    assert new_me.status_code == 200

    # 8. Logout
    logout_resp = api_client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {new_token}"})
    assert logout_resp.status_code == 200

    # 9. Verify token is now 401 Unauthorized
    logged_out_me = api_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {new_token}"})
    assert logged_out_me.status_code == 401


def test_cors_origin_headers(api_client):
    """Verifies that CORS origins are restricted to configured values and wildcard is not allowed."""
    # Trusted origin
    trusted_origin = CORS_ALLOWED_ORIGINS[0]
    res_trusted = api_client.options("/health", headers={
        "Origin": trusted_origin,
        "Access-Control-Request-Method": "GET"
    })
    assert res_trusted.headers.get("access-control-allow-origin") == trusted_origin

    # Untrusted origin
    res_untrusted = api_client.options("/health", headers={
        "Origin": "http://evil-attacker.com",
        "Access-Control-Request-Method": "GET"
    })
    assert res_untrusted.headers.get("access-control-allow-origin") is None

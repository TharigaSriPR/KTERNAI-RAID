"""
Password hashing and verification module.
Enforces Argon2id (RFC 9106) as the mandatory primary memory-hard password hasher.
Fails explicitly if Argon2id is requested but argon2-cffi is unavailable (no silent fallback).
Supports NIST-compliant PBKDF2-HMAC-SHA256 (600,000 iterations) only when explicitly configured.
Includes constant-time dummy verification to defeat username enumeration timing attacks.
"""

import os
import secrets
import hashlib
import hmac
from typing import Tuple, Optional

try:
    from argon2 import PasswordHasher, Type
    from argon2.exceptions import VerifyMismatchError, InvalidHashError
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False
    PasswordHasher = None
    Type = None
    VerifyMismatchError = Exception
    InvalidHashError = Exception


class PasswordSecurity:
    """
    Handles cryptographic password hashing and constant-time verification.
    """

    def __init__(self, default_algorithm: str = "argon2id"):
        self.default_algorithm = default_algorithm.lower()
        if self.default_algorithm == "argon2id":
            if not ARGON2_AVAILABLE:
                raise RuntimeError(
                    "Argon2id hashing is configured as the default, but 'argon2-cffi' is not installed. "
                    "Install it via: pip install argon2-cffi"
                )
            self._argon2_hasher = PasswordHasher(
                time_cost=3,
                memory_cost=65536,  # 64 MB
                parallelism=4,
                hash_len=32,
                type=Type.ID
            )
        elif self.default_algorithm == "pbkdf2_sha256":
            self._argon2_hasher = None
        else:
            raise ValueError(f"Unsupported default algorithm: {default_algorithm}. Use 'argon2id' or 'pbkdf2_sha256'.")

        # Generate a dummy Argon2id hash for constant-time rejection on nonexistent users
        if ARGON2_AVAILABLE and self._argon2_hasher is not None:
            self._dummy_hash = self._argon2_hasher.hash("dummy_constant_time_password_sample")
        else:
            self._dummy_hash = "$pbkdf2-sha256$600000$dummy$"

    def hash_password(self, plain_password: str, algorithm: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Hashes a plaintext password using Argon2id or explicitly requested PBKDF2-HMAC-SHA256.
        Returns: (password_hash, salt, algorithm)
        No silent fallback: raises RuntimeError if Argon2id is requested and argon2-cffi is missing.
        """
        if not plain_password or not isinstance(plain_password, str):
            raise ValueError("Password must be a non-empty string.")

        algo = (algorithm or self.default_algorithm).lower()

        if algo == "argon2id":
            if not ARGON2_AVAILABLE or self._argon2_hasher is None:
                raise RuntimeError("Argon2id hashing requested, but argon2-cffi is not installed.")
            encoded_hash = self._argon2_hasher.hash(plain_password)
            salt = secrets.token_hex(16)
            return encoded_hash, salt, "argon2id"

        elif algo == "pbkdf2_sha256":
            salt = secrets.token_hex(32)
            derived = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                bytes.fromhex(salt),
                iterations=600000,
                dklen=32
            )
            encoded_hash = f"$pbkdf2-sha256$600000${salt}${derived.hex()}"
            return encoded_hash, salt, "pbkdf2_sha256"

        else:
            raise ValueError(f"Unsupported algorithm '{algo}'. Supported: 'argon2id', 'pbkdf2_sha256'")

    def verify_password(
        self,
        plain_password: str,
        password_hash: str,
        salt: str = "",
        algorithm: Optional[str] = None
    ) -> bool:
        """
        Verifies plaintext password against stored hash in constant time.
        Detects algorithm from hash header or explicit argument.
        """
        if not plain_password or not password_hash:
            return False

        try:
            # Detect algorithm from hash prefix if available
            is_argon2 = password_hash.startswith("$argon2") or algorithm == "argon2id"
            is_pbkdf2 = password_hash.startswith("$pbkdf2") or algorithm == "pbkdf2_sha256"

            if is_argon2:
                if not ARGON2_AVAILABLE or self._argon2_hasher is None:
                    raise RuntimeError("Cannot verify Argon2id hash: argon2-cffi is not installed.")
                try:
                    return self._argon2_hasher.verify(password_hash, plain_password)
                except (VerifyMismatchError, InvalidHashError):
                    return False
                except Exception:
                    return False

            elif is_pbkdf2:
                # Extract salt and derived hash from stored format if present
                if password_hash.startswith("$pbkdf2-sha256$"):
                    parts = password_hash.split("$")
                    if len(parts) >= 5:
                        iterations = int(parts[2])
                        stored_salt = bytes.fromhex(parts[3])
                        expected_dk = parts[4]
                        derived = hashlib.pbkdf2_hmac(
                            "sha256",
                            plain_password.encode("utf-8"),
                            stored_salt,
                            iterations=iterations,
                            dklen=32
                        )
                        return hmac.compare_digest(derived.hex(), expected_dk)

                # Fallback to legacy separated salt + hash
                if not salt:
                    return False
                derived = hashlib.pbkdf2_hmac(
                    "sha256",
                    plain_password.encode("utf-8"),
                    bytes.fromhex(salt),
                    iterations=600000,
                    dklen=32
                )
                return hmac.compare_digest(derived.hex(), password_hash)

            else:
                return False

        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            return False

    def verify_dummy(self, plain_password: str):
        """
        Performs a constant-time dummy verification on nonexistent users to eliminate
        response-time discrepancies between valid and invalid usernames (timing attack mitigation).
        """
        try:
            if ARGON2_AVAILABLE and self._argon2_hasher is not None and self._dummy_hash.startswith("$argon2"):
                try:
                    self._argon2_hasher.verify(self._dummy_hash, plain_password)
                except Exception:
                    pass
            else:
                hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), b"dummy_salt_bytes_32_bytes_long!", 600000, 32)
        except Exception:
            pass

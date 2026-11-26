"""
Database Migrations module for the RAID Identification Agent.
Implements safe, transactional, repeatable schema versioning and tenant isolation.
Preserves existing prototype data by associating legacy records with a locked 'default_user' account.
Backs up SQLite database prior to running migrations.
"""

import sqlite3
import shutil
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIGRATION_VERSION = 7
MIGRATION_DESCRIPTION = "Data folder file synchronisation and tracking"

SEED_SYSTEM_USER = """
INSERT OR IGNORE INTO users (id, username, email, password_hash, salt, algorithm, full_name, role)
VALUES (
    'default_user',
    'system_default',
    'default_user@system.local',
    'LOCKED_SYSTEM_ACCOUNT_DO_NOT_LOGIN',
    'NONE',
    'argon2id',
    'System Legacy Account',
    'admin'
);
"""

TABLES_TO_MIGRATE = [
    "raw_signals",
    "raid_drafts",
    "approved_raid",
    "predictions",
    "deduplication_matches",
    "audit_history"
]


def backup_database(db_path: Path) -> Optional[Path]:
    """Creates a timestamped or .bak copy of the database before applying schema changes."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None
    backup_path = db_path.with_suffix(".db.bak")
    try:
        shutil.copy2(db_path, backup_path)
        logger.info("Database backup created at: %s", backup_path)
        return backup_path
    except Exception as exc:
        logger.warning("Failed to create database backup at %s: %s", backup_path, exc)
        return None


def run_migrations(db_path: Path) -> bool:
    """
    Executes repeatable, idempotent schema migrations.
    Returns True if migrations were applied, False if already up to date.
    """
    if not db_path.exists():
        return False

    backup_database(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    applied_any = False

    try:
        with conn:
            cursor = conn.cursor()

            # 1. Initialize schema_migrations tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    description TEXT NOT NULL
                );
            """)

            cursor.execute("SELECT version FROM schema_migrations")
            applied_versions = {row["version"] for row in cursor.fetchall()}

            # ---------------------------------------------------------------
            # Version 2: Multi-tenant user_id isolation and composite indexes
            # ---------------------------------------------------------------
            if 2 not in applied_versions:
                logger.info("Applying schema migration version 2: Multi-tenant user_id isolation and composite indexes")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        algorithm TEXT NOT NULL DEFAULT 'argon2id',
                        full_name TEXT,
                        role TEXT DEFAULT 'user' CHECK(role IN ('user', 'admin')),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute(SEED_SYSTEM_USER)

                for table in TABLES_TO_MIGRATE:
                    cursor.execute(f"PRAGMA table_info({table})")
                    existing_cols = [row["name"] for row in cursor.fetchall()]

                    if existing_cols and "user_id" not in existing_cols:
                        logger.info("Adding user_id column to table '%s'", table)
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT 'default_user';")

                    if existing_cols:
                        cursor.execute(f"UPDATE {table} SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';")

                cursor.execute("""
                    SELECT user_id, source_type, source_id, COUNT(*) as cnt
                    FROM raw_signals
                    WHERE source_id IS NOT NULL AND source_id != ''
                    GROUP BY user_id, source_type, source_id
                    HAVING cnt > 1;
                """)
                duplicates = cursor.fetchall()
                for dup in duplicates:
                    cursor.execute("""
                        SELECT rowid FROM raw_signals
                        WHERE user_id = ? AND source_type = ? AND source_id = ?
                        ORDER BY rowid ASC
                    """, (dup["user_id"], dup["source_type"], dup["source_id"]))
                    rowids = [r["rowid"] for r in cursor.fetchall()]
                    for rid in rowids[1:]:
                        cursor.execute(
                            "UPDATE raw_signals SET source_id = source_id || '-dup-' || ? WHERE rowid = ?",
                            (rid, rid)
                        )

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_signals_user ON raw_signals(user_id);")
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_raw_signals_user_source ON raw_signals(user_id, source_type, source_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raid_drafts_user ON raid_drafts(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raid_drafts_user_status ON raid_drafts(user_id, status);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_approved_raid_user ON approved_raid(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_dedupe_user ON deduplication_matches(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_history_user ON audit_history(user_id);")

                cursor.execute("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (2, 'Multi-tenant user_id isolation and composite indexes');
                """)
                applied_any = True

            # ---------------------------------------------------------------
            # Version 3: Gmail OAuth 2.0 tables: user_gmail_tokens & oauth_states
            # ---------------------------------------------------------------
            if 3 not in applied_versions:
                logger.info("Applying schema migration version 3: Gmail OAuth 2.0 tables and indexes")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_gmail_tokens (
                        id TEXT PRIMARY KEY,
                        user_id TEXT UNIQUE NOT NULL,
                        google_email TEXT NOT NULL,
                        encrypted_access_token TEXT NOT NULL,
                        encrypted_refresh_token TEXT,
                        token_expiry DATETIME,
                        granted_scopes TEXT,
                        connected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        connection_status TEXT DEFAULT 'connected' CHECK(connection_status IN ('connected', 'revoked', 'disconnected', 'expired')),
                        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_gmail_tokens_user ON user_gmail_tokens(user_id);")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS oauth_states (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state TEXT UNIQUE NOT NULL,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        code_verifier TEXT,
                        redirect_uri TEXT NOT NULL,
                        expires_at DATETIME NOT NULL,
                        is_used INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_states_state ON oauth_states(state);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_states_user ON oauth_states(user_id);")

                cursor.execute("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (3, 'Gmail OAuth 2.0 tables and indexes');
                """)
                applied_any = True

            # ---------------------------------------------------------------
            # Version 4: Gmail sync checkpoints and history audit tables
            # ---------------------------------------------------------------
            if 4 not in applied_versions:
                logger.info("Applying schema migration version 4: Gmail sync checkpoints and history audit tables")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS gmail_sync_checkpoints (
                        id TEXT PRIMARY KEY,
                        user_id TEXT UNIQUE NOT NULL,
                        last_history_id TEXT,
                        last_sync_at DATETIME,
                        sync_status TEXT DEFAULT 'idle',
                        is_syncing INTEGER DEFAULT 0,
                        sync_lock_at DATETIME,
                        messages_synced_total INTEGER DEFAULT 0,
                        last_error TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_gmail_sync_checkpoints_user ON gmail_sync_checkpoints(user_id);")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS gmail_sync_history (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        sync_mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        messages_fetched INTEGER DEFAULT 0,
                        messages_processed INTEGER DEFAULT 0,
                        duplicates_skipped INTEGER DEFAULT 0,
                        messages_failed INTEGER DEFAULT 0,
                        drafts_created INTEGER DEFAULT 0,
                        error_message TEXT,
                        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME,
                        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_gmail_sync_history_user ON gmail_sync_history(user_id, started_at);")

                cursor.execute("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (4, 'Gmail sync checkpoints and history audit tables');
                """)
                applied_any = True

            # ---------------------------------------------------------------
            # Version 5: Classification failure tracking columns & evidence status
            # ---------------------------------------------------------------
            if 5 not in applied_versions:
                logger.info("Applying schema migration version 5: Classification failure tracking columns and evidence status")

                # 1. Update raw_signals table
                cursor.execute("PRAGMA table_info(raw_signals)")
                existing_signal_cols = [row["name"] for row in cursor.fetchall()]

                signal_cols_to_add = [
                    ("classification_status", "TEXT DEFAULT 'PENDING'"),
                    ("classification_failure_reason", "TEXT"),
                    ("classification_error_category", "TEXT"),
                    ("classification_attempts", "INTEGER DEFAULT 0"),
                    ("last_attempted_at", "DATETIME"),
                    ("last_classified_at", "DATETIME")
                ]

                for col_name, col_def in signal_cols_to_add:
                    if col_name not in existing_signal_cols:
                        logger.info("Adding column '%s' to raw_signals", col_name)
                        cursor.execute(f"ALTER TABLE raw_signals ADD COLUMN {col_name} {col_def};")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_signals_status ON raw_signals(classification_status);")

                # 2. Update raid_drafts table
                cursor.execute("PRAGMA table_info(raid_drafts)")
                existing_draft_cols = [row["name"] for row in cursor.fetchall()]

                if "evidence_status" not in existing_draft_cols:
                    logger.info("Adding column 'evidence_status' to raid_drafts")
                    cursor.execute("ALTER TABLE raid_drafts ADD COLUMN evidence_status TEXT DEFAULT 'verified';")

                cursor.execute("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (5, 'Classification failure tracking columns and evidence status');
                """)
                applied_any = True

            # ---------------------------------------------------------------
            # Version 6: LLM provider provenance & mock protection
            # ---------------------------------------------------------------
            if 6 not in applied_versions:
                logger.info("Applying schema migration version 6: LLM provider provenance and mock protection")

                # 1. Update raw_signals table
                cursor.execute("PRAGMA table_info(raw_signals)")
                existing_signal_cols = [row["name"] for row in cursor.fetchall()]

                signal_cols_to_add = [
                    ("llm_provider", "TEXT"),
                    ("llm_model", "TEXT")
                ]

                for col_name, col_def in signal_cols_to_add:
                    if col_name not in existing_signal_cols:
                        logger.info("Adding column '%s' to raw_signals", col_name)
                        cursor.execute(f"ALTER TABLE raw_signals ADD COLUMN {col_name} {col_def};")

                # 2. Update raid_drafts table
                cursor.execute("PRAGMA table_info(raid_drafts)")
                existing_draft_cols = [row["name"] for row in cursor.fetchall()]

                draft_cols_to_add = [
                    ("llm_provider", "TEXT DEFAULT 'unknown'"),
                    ("llm_model", "TEXT DEFAULT 'unknown'")
                ]

                for col_name, col_def in draft_cols_to_add:
                    if col_name not in existing_draft_cols:
                        logger.info("Adding column '%s' to raid_drafts", col_name)
                        cursor.execute(f"ALTER TABLE raid_drafts ADD COLUMN {col_name} {col_def};")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_raid_drafts_provider ON raid_drafts(user_id, llm_provider);")

                # 3. Mark the 6 confirmed mock retried signals and drafts
                mock_signals = (
                    'sig-5777789c', 'sig-e7a77ee4', 'sig-88a6d32b',
                    'sig-f657316b', 'sig-183b939b', 'sig-754cb9b4'
                )
                placeholders_sig = ",".join("?" for _ in mock_signals)
                cursor.execute(f"""
                    UPDATE raw_signals
                    SET llm_provider = 'mock', llm_model = 'mock-llm'
                    WHERE id IN ({placeholders_sig})
                """, mock_signals)

                mock_drafts = (
                    'raid-d999b559', 'raid-a9600e18', 'raid-badf03fe',
                    'raid-a682a6dc', 'raid-cb803eb0', 'raid-de08c8ae'
                )
                placeholders_draft = ",".join("?" for _ in mock_drafts)
                cursor.execute(f"""
                    UPDATE raid_drafts
                    SET llm_provider = 'mock', llm_model = 'mock-llm'
                    WHERE id IN ({placeholders_draft})
                """, mock_drafts)

                # 4. Mark predictive engine drafts
                cursor.execute("""
                    UPDATE raid_drafts
                    SET llm_provider = 'rule_based_foresight', llm_model = 'predictive_engine'
                    WHERE detection_source = 'predictive_engine'
                      AND (llm_provider IS NULL OR llm_provider = '' OR llm_provider = 'unknown')
                """)

                # 5. Mark uncertain historical records as legacy_unknown (do NOT assume Gemini)
                cursor.execute("""
                    UPDATE raid_drafts
                    SET llm_provider = 'legacy_unknown', llm_model = 'unknown'
                    WHERE (llm_provider IS NULL OR llm_provider = '' OR llm_provider = 'unknown')
                """)

                cursor.execute("""
                    UPDATE raw_signals
                    SET llm_provider = 'legacy_unknown', llm_model = 'unknown'
                    WHERE classification_status IN ('COMPLETED', 'NOT_RAID', 'FAILED')
                      AND (llm_provider IS NULL OR llm_provider = '' OR llm_provider = 'unknown')
                """)

                cursor.execute("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (6, 'LLM provider provenance and mock protection');
                """)
                applied_any = True

            # ---------------------------------------------------------------
            # Version 7: Data folder file synchronisation and tracking
            # ---------------------------------------------------------------
            if 7 not in applied_versions:
                logger.info("Applying schema migration version 7: Data folder file synchronisation and tracking")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS source_file_sync (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        last_modified TEXT NOT NULL,
                        record_count INTEGER DEFAULT 0,
                        signals_created INTEGER DEFAULT 0,
                        signals_classified INTEGER DEFAULT 0,
                        drafts_created INTEGER DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'DISCOVERED',
                        error_message TEXT,
                        user_id TEXT DEFAULT 'default_user',
                        discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processed_at DATETIME
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_file_sync_user ON source_file_sync(user_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_file_sync_hash ON source_file_sync(content_hash);")
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_source_file_sync_path_user ON source_file_sync(file_path, user_id);")

                cursor.execute("""
                    INSERT INTO schema_migrations (version, description)
                    VALUES (7, 'Data folder file synchronisation and tracking');
                """)
                applied_any = True

            if applied_any:
                logger.info("Successfully applied schema migrations up to version %d.", MIGRATION_VERSION)
            else:
                logger.debug("Database %s is already up to date (version %d).", db_path.name, MIGRATION_VERSION)

            return applied_any

    except Exception as exc:
        logger.error("Failed to apply migration version %d: %s", MIGRATION_VERSION, exc, exc_info=True)
        raise
    finally:
        conn.close()

"""
Data models and SQLite schema definitions for the RAID Identification Agent.
Implements the Suggested RAID item template, Audit History, Predictions,
and Semantic Deduplication records from PRD/Brief.
"""

# SQLite Schema Definition DDL
CREATE_RAW_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS raw_signals (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,         -- 'email', 'chat', 'meeting_notes', 'project_plan', 'defect'
    source_id TEXT,                    -- external ID (message ID, task ID, defect ID)
    author TEXT,
    timestamp DATETIME NOT NULL,
    subject TEXT,
    content TEXT NOT NULL,
    workstream TEXT,
    phase TEXT,
    is_noise INTEGER DEFAULT 0,        -- 1 if filtered as noise
    noise_reason TEXT,
    user_id TEXT DEFAULT 'default_user',
    classification_status TEXT DEFAULT 'PENDING',
    classification_failure_reason TEXT,
    classification_error_category TEXT,
    classification_attempts INTEGER DEFAULT 0,
    last_attempted_at DATETIME,
    last_classified_at DATETIME,
    llm_provider TEXT,
    llm_model TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_raw_signals_source ON raw_signals(source_type, source_id);
"""

CREATE_RAID_DRAFTS_TABLE = """
CREATE TABLE IF NOT EXISTS raid_drafts (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('Risk', 'Assumption', 'Issue', 'Decision', 'Dependency')),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    workstream TEXT,
    phase TEXT,
    probability TEXT,                  -- For Risks (Low, Medium, High)
    impact TEXT,                       -- For Risks (Low, Medium, High)
    severity TEXT,                     -- For Issues (Low, Medium, High, Critical)
    scoring_reason TEXT,               -- One-line explanation of score
    likely_owner TEXT,
    due_date TEXT,
    mitigation_action TEXT,            -- For Risks & Issues
    confidence_score REAL NOT NULL,    -- 0.0 to 1.0
    evidence_text TEXT NOT NULL,       -- Exact source quote / reference
    evidence_source_id TEXT NOT NULL,  -- Links to raw_signals.id
    evidence_source_type TEXT NOT NULL,-- 'email', 'chat', 'meeting_notes', etc.
    detection_source TEXT NOT NULL,    -- 'llm_extraction' or 'predictive_engine'
    duplicate_of_id TEXT,              -- If flagged as duplicate
    status TEXT DEFAULT 'Draft' CHECK(status IN ('Draft', 'Pending_Review', 'Accepted', 'Approved', 'Modified', 'Rejected')),
    user_id TEXT DEFAULT 'default_user',
    evidence_status TEXT DEFAULT 'verified',
    llm_provider TEXT DEFAULT 'unknown',
    llm_model TEXT DEFAULT 'unknown',
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(evidence_source_id) REFERENCES raw_signals(id)
);
"""

CREATE_APPROVED_RAID_TABLE = """
CREATE TABLE IF NOT EXISTS approved_raid (
    id TEXT PRIMARY KEY,
    draft_id TEXT,
    type TEXT NOT NULL CHECK(type IN ('Risk', 'Assumption', 'Issue', 'Decision', 'Dependency')),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    workstream TEXT,
    phase TEXT,
    severity_or_impact TEXT,
    owner TEXT,
    due_date TEXT,
    mitigation_action TEXT,
    status TEXT DEFAULT 'Open' CHECK(status IN ('Open', 'In Progress', 'Closed')),
    evidence_links TEXT,
    approved_by TEXT NOT NULL,
    user_id TEXT DEFAULT 'default_user',
    approved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(draft_id) REFERENCES raid_drafts(id)
);
"""

CREATE_AUDIT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS audit_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    item_table TEXT NOT NULL,          -- 'raid_drafts' or 'approved_raid'
    action TEXT NOT NULL,              -- 'PROPOSED', 'ACCEPTED', 'EDITED', 'REJECTED', 'CLOSED'
    proposed_data TEXT,                -- JSON snapshot of agent proposal
    final_data TEXT,                   -- JSON snapshot after human decision
    action_by TEXT NOT NULL,           -- 'Agent' or PM user name
    action_notes TEXT,
    user_id TEXT DEFAULT 'default_user',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    prediction_type TEXT NOT NULL,     -- 'TASK_OVERDUE', 'UPCOMING_DELAY', 'DEFECT_CLUSTER', 'ASSUMPTION_CONTRADICTED'
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    suggested_raid_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT,
    evidence TEXT NOT NULL,
    source_signal_ids TEXT NOT NULL,   -- JSON array of source signal IDs
    related_raid_ids TEXT,             -- JSON array of related RAID item IDs
    workstream TEXT,
    phase TEXT,
    status TEXT DEFAULT 'DETECTED',
    user_id TEXT DEFAULT 'default_user',
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,                     -- JSON dictionary of evaluation parameters
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
"""

CREATE_DEDUPLICATION_MATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS deduplication_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    new_raid_id TEXT,
    matched_raid_id TEXT,
    similarity_score REAL NOT NULL,
    match_status TEXT NOT NULL,        -- 'LIKELY_DUPLICATE', 'POSSIBLE_DUPLICATE', 'RELATED_NOT_DUPLICATE', 'NO_MATCH'
    match_reason TEXT NOT NULL,
    same_category INTEGER NOT NULL DEFAULT 0,
    same_workstream INTEGER NOT NULL DEFAULT 0,
    requires_human_review INTEGER NOT NULL DEFAULT 0,
    user_id TEXT DEFAULT 'default_user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dedupe_new_raid ON deduplication_matches(new_raid_id);
"""

# Phase 1 Authentication & User Management DDL
CREATE_USERS_TABLE = """
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
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""

CREATE_USER_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS user_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_token_hash TEXT UNIQUE NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    last_active_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sessions_hash ON user_sessions(session_token_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id);
"""

CREATE_LOGIN_ATTEMPTS_TABLE = """
CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,
    ip_address TEXT NOT NULL,
    attempt_count INTEGER DEFAULT 1,
    last_attempt_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    locked_until DATETIME
);
CREATE INDEX IF NOT EXISTS idx_login_attempts_lookup ON login_attempts(identifier, ip_address);
"""

# Phase 3 Gmail OAuth 2.0 & Token Storage DDL
CREATE_USER_GMAIL_TOKENS_TABLE = """
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
CREATE INDEX IF NOT EXISTS idx_user_gmail_tokens_user ON user_gmail_tokens(user_id);
"""

CREATE_OAUTH_STATES_TABLE = """
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
CREATE INDEX IF NOT EXISTS idx_oauth_states_state ON oauth_states(state);
CREATE INDEX IF NOT EXISTS idx_oauth_states_user ON oauth_states(user_id);
"""

# Phase 4 Gmail Sync Checkpoints and History Audit DDL
CREATE_GMAIL_SYNC_CHECKPOINTS_TABLE = """
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
CREATE INDEX IF NOT EXISTS idx_gmail_sync_checkpoints_user ON gmail_sync_checkpoints(user_id);
"""

CREATE_GMAIL_SYNC_HISTORY_TABLE = """
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
CREATE INDEX IF NOT EXISTS idx_gmail_sync_history_user ON gmail_sync_history(user_id, started_at);
"""

CREATE_SOURCE_FILE_SYNC_TABLE = """
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
CREATE INDEX IF NOT EXISTS idx_source_file_sync_user ON source_file_sync(user_id);
CREATE INDEX IF NOT EXISTS idx_source_file_sync_hash ON source_file_sync(content_hash);
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_file_sync_path_user ON source_file_sync(file_path, user_id);
"""

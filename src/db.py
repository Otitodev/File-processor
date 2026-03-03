"""
SQLite persistence layer for processed report summaries.

Database file (reports.db) is created next to app.py on first startup.
No external dependencies — uses Python's built-in sqlite3 module.

Schema
------
runs    — one row per processed file (claimant, filename, timestamp, count)
reports — one row per ReportSummary, foreign-keyed to runs
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import List

from .report_analyzer import ReportSummary

# DATA_DIR lets Railway (or any host) point the DB at a persistent volume.
# Locally it defaults to the project root, so behaviour is unchanged.
DB_PATH = os.path.join(os.environ.get("DATA_DIR", "."), "reports.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    claimant     TEXT    NOT NULL,
    filename     TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    report_count INTEGER NOT NULL,
    session_id   TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    report_index    INTEGER NOT NULL,
    title           TEXT    NOT NULL,
    start_page      INTEGER NOT NULL,
    end_page        INTEGER NOT NULL,
    summary         TEXT    NOT NULL,
    source_filename TEXT    NOT NULL DEFAULT '',
    report_type     TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS skipped_reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    filename   TEXT    NOT NULL,
    reason     TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    start_page INTEGER NOT NULL,
    end_page   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_prompts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with sqlite3.connect(db_path) as con:
        con.executescript(_DDL)
        # Add session_id column if it doesn't exist (migration for existing DBs)
        try:
            con.execute("ALTER TABLE runs ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass  # column already exists
        # Backfill legacy rows so every run has a unique non-empty session_id
        con.execute("UPDATE runs SET session_id = 'legacy_' || id WHERE session_id = ''")
        # Add source_filename column to reports (Feature 7)
        try:
            con.execute("ALTER TABLE reports ADD COLUMN source_filename TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass
        # Add report_type column to reports (Feature 9)
        try:
            con.execute("ALTER TABLE reports ADD COLUMN report_type TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass


def save_run(
    claimant_name: str,
    source_filename: str,
    summaries: List[ReportSummary],
    db_path: str = DB_PATH,
    session_id: str = "",
) -> int:
    """
    Persist a completed run and all its report summaries.

    Returns the new run_id (integer primary key).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        cur = con.execute(
            "INSERT INTO runs (claimant, filename, created_at, report_count, session_id) VALUES (?, ?, ?, ?, ?)",
            (claimant_name, source_filename, created_at, len(summaries), session_id),
        )
        run_id = cur.lastrowid
        con.executemany(
            "INSERT INTO reports (run_id, report_index, title, start_page, end_page, summary,"
            " source_filename, report_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (run_id, s.report_index, s.title, s.start_page, s.end_page, s.summary,
                 s.source_filename, s.report_type)
                for s in summaries
            ],
        )
    return run_id


def list_runs(db_path: str = DB_PATH) -> List[dict]:
    """
    Return all runs, newest first.

    Each dict has keys: id, claimant, filename, created_at, report_count.
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, claimant, filename, created_at, report_count"
            " FROM runs ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_run_summaries(run_id: int, db_path: str = DB_PATH) -> List[ReportSummary]:
    """Return all ReportSummary objects for a given run_id, ordered by report_index."""
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT report_index, title, start_page, end_page, summary"
            " FROM reports WHERE run_id = ? ORDER BY report_index",
            (run_id,),
        ).fetchall()
    return [ReportSummary(**dict(r)) for r in rows]


def delete_run(run_id: int, db_path: str = DB_PATH) -> None:
    """Delete a run and all its associated reports."""
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM runs WHERE id = ?", (run_id,))


def list_sessions(db_path: str = DB_PATH) -> List[dict]:
    """Return one dict per session, newest first.

    Keys: session_id, claimant, filenames (comma-joined), created_at, report_count.
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT
                session_id,
                claimant,
                GROUP_CONCAT(filename, ', ') AS filenames,
                MIN(created_at)             AS created_at,
                SUM(report_count)           AS report_count
            FROM runs
            GROUP BY session_id
            ORDER BY MIN(id) DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_session_summaries(session_id: str, db_path: str = DB_PATH) -> List[ReportSummary]:
    """Return all ReportSummary objects for every run in a session, ordered by file then report."""
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT r.report_index, r.title, r.start_page, r.end_page, r.summary,
                   COALESCE(NULLIF(r.source_filename, ''), runs.filename) AS source_filename,
                   r.report_type
            FROM reports r
            JOIN runs ON r.run_id = runs.id
            WHERE runs.session_id = ?
            ORDER BY runs.id, r.report_index
        """, (session_id,)).fetchall()
    return [ReportSummary(**dict(r)) for r in rows]


def delete_session(session_id: str, db_path: str = DB_PATH) -> None:
    """Delete all runs (and their reports) belonging to a session."""
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM runs WHERE session_id = ?", (session_id,))
        con.execute("DELETE FROM skipped_reports WHERE session_id = ?", (session_id,))


# ---------------------------------------------------------------------------
# Skipped reports (Feature 6)
# ---------------------------------------------------------------------------


def save_skipped(session_id: str, filename: str, skipped: list, db_path: str = DB_PATH) -> None:
    """
    Persist skipped report entries for a session + file.

    Each item in `skipped` is a dict with keys: reason, title, start_page, end_page.
    """
    if not skipped:
        return
    with sqlite3.connect(db_path) as con:
        con.executemany(
            "INSERT INTO skipped_reports (session_id, filename, reason, title, start_page, end_page)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (session_id, filename, s["reason"], s["title"], s["start_page"], s["end_page"])
                for s in skipped
            ],
        )


def get_session_skipped(session_id: str, db_path: str = DB_PATH) -> list:
    """Return all skipped report rows for a session, ordered by id."""
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT filename, reason, title, start_page, end_page"
            " FROM skipped_reports WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Saved prompts
# ---------------------------------------------------------------------------


def save_prompt(name: str, text: str, db_path: str = DB_PATH) -> None:
    """Insert or update a saved prompt by name (upsert)."""
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO saved_prompts (name, text, created_at) VALUES (?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET text = excluded.text",
            (name, text, created_at),
        )


def list_prompts(db_path: str = DB_PATH) -> List[dict]:
    """Return all saved prompts, oldest first. Each dict has: id, name, text, created_at."""
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, name, text, created_at FROM saved_prompts ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_prompt(prompt_id: int, db_path: str = DB_PATH) -> None:
    """Delete a saved prompt by its id."""
    with sqlite3.connect(db_path) as con:
        con.execute("DELETE FROM saved_prompts WHERE id = ?", (prompt_id,))


# ---------------------------------------------------------------------------
# App settings (key/value store)
# ---------------------------------------------------------------------------


def get_setting(key: str, default: str = "", db_path: str = DB_PATH) -> str:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else default


def set_setting(key: str, value: str, db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

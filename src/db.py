"""
SQLite persistence layer for processed report summaries.

Database file (reports.db) is created next to app.py on first startup.
No external dependencies — uses Python's built-in sqlite3 module.

Schema
------
runs    — one row per processed file (claimant, filename, timestamp, count)
reports — one row per ReportSummary, foreign-keyed to runs
"""

import sqlite3
from datetime import datetime, timezone
from typing import List

from .report_analyzer import ReportSummary

DB_PATH = "reports.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    claimant     TEXT    NOT NULL,
    filename     TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    report_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    report_index INTEGER NOT NULL,
    title        TEXT    NOT NULL,
    start_page   INTEGER NOT NULL,
    end_page     INTEGER NOT NULL,
    summary      TEXT    NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with sqlite3.connect(db_path) as con:
        con.executescript(_DDL)


def save_run(
    claimant_name: str,
    source_filename: str,
    summaries: List[ReportSummary],
    db_path: str = DB_PATH,
) -> int:
    """
    Persist a completed run and all its report summaries.

    Returns the new run_id (integer primary key).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        cur = con.execute(
            "INSERT INTO runs (claimant, filename, created_at, report_count) VALUES (?, ?, ?, ?)",
            (claimant_name, source_filename, created_at, len(summaries)),
        )
        run_id = cur.lastrowid
        con.executemany(
            "INSERT INTO reports (run_id, report_index, title, start_page, end_page, summary)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (run_id, s.report_index, s.title, s.start_page, s.end_page, s.summary)
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

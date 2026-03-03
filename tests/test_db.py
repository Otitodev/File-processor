"""
Tests for src/db.py

Uses a temporary SQLite file per test so each test runs in isolation.
No mocking required — sqlite3 is pure Python.
"""

import sqlite3
import pytest

from src.db import init_db, save_run, list_runs, get_run_summaries, delete_run
from src.report_analyzer import ReportSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_summaries(n: int = 2) -> list:
    return [
        ReportSummary(
            report_index=i + 1,
            title=f"Report {i + 1}",
            start_page=i * 10 + 1,
            end_page=(i + 1) * 10,
            summary=f"Summary of report {i + 1}.",
        )
        for i in range(n)
    ]


@pytest.fixture
def db(tmp_path):
    """Fresh database file, tables already created."""
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_runs_table(self, db):
        with sqlite3.connect(db) as con:
            names = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "runs" in names

    def test_creates_reports_table(self, db):
        with sqlite3.connect(db) as con:
            names = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "reports" in names

    def test_idempotent(self, db):
        """Calling init_db twice on the same file must not raise."""
        init_db(db)


# ---------------------------------------------------------------------------
# save_run
# ---------------------------------------------------------------------------

class TestSaveRun:
    def test_returns_positive_integer_id(self, db):
        run_id = save_run("Jane Smith", "report.pdf", make_summaries(), db)
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_saves_claimant_and_filename(self, db):
        save_run("Jane Smith", "report.pdf", make_summaries(1), db)
        runs = list_runs(db)
        assert runs[0]["claimant"] == "Jane Smith"
        assert runs[0]["filename"] == "report.pdf"

    def test_saves_correct_report_count(self, db):
        save_run("Jane Smith", "report.pdf", make_summaries(3), db)
        assert list_runs(db)[0]["report_count"] == 3

    def test_empty_summaries_allowed(self, db):
        run_id = save_run("Jane Smith", "empty.pdf", [], db)
        assert run_id > 0
        assert list_runs(db)[0]["report_count"] == 0

    def test_multiple_runs_saved(self, db):
        save_run("Alice", "a.pdf", make_summaries(1), db)
        save_run("Bob", "b.pdf", make_summaries(2), db)
        assert len(list_runs(db)) == 2

    def test_reports_inserted(self, db):
        save_run("Jane Smith", "report.pdf", make_summaries(2), db)
        with sqlite3.connect(db) as con:
            count = con.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_empty_db_returns_empty_list(self, db):
        assert list_runs(db) == []

    def test_returns_all_expected_keys(self, db):
        save_run("Jane", "f.pdf", make_summaries(1), db)
        run = list_runs(db)[0]
        assert set(run.keys()) == {"id", "claimant", "filename", "created_at", "report_count"}

    def test_newest_first_ordering(self, db):
        save_run("Alice", "a.pdf", make_summaries(1), db)
        save_run("Bob", "b.pdf", make_summaries(1), db)
        runs = list_runs(db)
        assert runs[0]["claimant"] == "Bob"
        assert runs[1]["claimant"] == "Alice"

    def test_created_at_is_iso_string(self, db):
        save_run("Jane", "f.pdf", make_summaries(1), db)
        created_at = list_runs(db)[0]["created_at"]
        # ISO 8601 strings contain 'T'
        assert "T" in created_at


# ---------------------------------------------------------------------------
# get_run_summaries
# ---------------------------------------------------------------------------

class TestGetRunSummaries:
    def test_returns_report_summary_instances(self, db):
        run_id = save_run("Jane", "r.pdf", make_summaries(1), db)
        result = get_run_summaries(run_id, db)
        assert all(isinstance(s, ReportSummary) for s in result)

    def test_ordered_by_report_index(self, db):
        run_id = save_run("Jane", "r.pdf", make_summaries(3), db)
        result = get_run_summaries(run_id, db)
        assert [s.report_index for s in result] == [1, 2, 3]

    def test_content_round_trips(self, db):
        original = make_summaries(1)[0]
        run_id = save_run("Jane", "r.pdf", [original], db)
        result = get_run_summaries(run_id, db)[0]
        assert result.title == original.title
        assert result.summary == original.summary
        assert result.start_page == original.start_page
        assert result.end_page == original.end_page
        assert result.report_index == original.report_index

    def test_empty_run_returns_empty_list(self, db):
        run_id = save_run("Jane", "r.pdf", [], db)
        assert get_run_summaries(run_id, db) == []

    def test_unknown_run_id_returns_empty_list(self, db):
        assert get_run_summaries(9999, db) == []


# ---------------------------------------------------------------------------
# delete_run
# ---------------------------------------------------------------------------

class TestDeleteRun:
    def test_removes_run_from_list(self, db):
        run_id = save_run("Jane", "r.pdf", make_summaries(1), db)
        delete_run(run_id, db)
        assert list_runs(db) == []

    def test_cascades_to_reports(self, db):
        run_id = save_run("Jane", "r.pdf", make_summaries(3), db)
        delete_run(run_id, db)
        with sqlite3.connect(db) as con:
            count = con.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        assert count == 0

    def test_only_deletes_target_run(self, db):
        run_id1 = save_run("Alice", "a.pdf", make_summaries(1), db)
        run_id2 = save_run("Bob", "b.pdf", make_summaries(2), db)
        delete_run(run_id1, db)
        runs = list_runs(db)
        assert len(runs) == 1
        assert runs[0]["claimant"] == "Bob"

    def test_nonexistent_id_is_noop(self, db):
        """Deleting a run that does not exist must not raise."""
        delete_run(9999, db)

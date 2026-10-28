"""Shared pytest fixtures.

Two kinds of test live under tests/:

* unit tests       — pure Python, no database. Always run.
* `db` tests       — need Postgres with the Phase 1 tables and the Phase 3
                     `sdn_names` index (DATABASE_URL in .env). Skipped, not
                     failed, when the database is unreachable or not built.
* `slow` tests     — `db` tests that block thousands of names (≈1–3 min).
                     Excluded by default; run with `pytest -m slow`.

A *fixture* is a function pytest calls for you and passes, by parameter name,
into any test that asks for it. `scope="session"` means it is created once for
the whole run (one DB connection, one Blocker) instead of once per test.
"""

from __future__ import annotations

import os

import pytest

try:  # .env is optional for unit tests
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def pytest_collection_modifyitems(config, items):
    """Every test in a module that uses the `conn` fixture is a db test."""
    for item in items:
        if "conn" in getattr(item, "fixturenames", ()) or "blocker" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.db)


@pytest.fixture(scope="session")
def conn():
    """One autocommit connection for the run, or skip every db test."""
    psycopg = pytest.importorskip("psycopg")
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set — db tests skipped")
    try:
        c = psycopg.connect(url, connect_timeout=3, autocommit=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres unreachable ({type(e).__name__}) — db tests skipped")
    with c.cursor() as cur:
        cur.execute("SELECT to_regclass('sdn_names') IS NOT NULL AND to_regclass('sdn_entries') IS NOT NULL")
        if not cur.fetchone()[0]:
            c.close()
            pytest.skip("sdn_names / sdn_entries missing — run load_sdn and build_name_index first")
    yield c
    c.close()


@pytest.fixture(scope="session")
def blocker(conn):
    """The production Blocker at its default settings (k=10, threshold 0.3)."""
    from phase3_blocking.blocking import Blocker

    return Blocker(conn)


@pytest.fixture(scope="session")
def idf(blocker):
    """IDF table computed over the real list (same object the Blocker uses)."""
    return blocker.idf


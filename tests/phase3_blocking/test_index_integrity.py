"""Step 0 against the real database: is sdn_names exactly what build_name_index
would build from today's sdn_entries / sdn_aliases?

The strongest check is the first one: rebuild every row in Python with the
production build_rows() and compare with the table, row for row. If someone
edits normalize() or the legal-form tables and forgets to rebuild the index,
queries are normalised one way and the index another — and every generator
silently loses recall. This test catches that.

Needs Postgres (marked `db` automatically through the `conn` fixture).
"""

from __future__ import annotations

from collections import Counter

import pytest

from common.reference_tables import MIN_ACRONYM_LENGTH
from phase3_blocking.blocking import ACRONYM_LIMIT
from phase3_blocking.build_name_index import build_rows

# 23 Sep 2026 sdn.xml, as loaded in Phase 1. Update when the list is reloaded.
EXPECTED = {
    ("aircraft", "alias"): 2, ("aircraft", "primary"): 342,
    ("entity", "alias"): 13_415, ("entity", "primary"): 9_975,
    ("individual", "alias"): 10_916, ("individual", "primary"): 7_534,
    ("vessel", "alias"): 289, ("vessel", "primary"): 1_540,
}


def q(conn, sql, *params):
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return cur.fetchall()


def test_pg_trgm_installed(conn):
    assert q(conn, "SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")


def test_row_counts_by_type_and_kind(conn):
    got = {(t, k): n for t, k, n in q(conn, "SELECT entry_type, name_kind, count(*) FROM sdn_names GROUP BY 1, 2")}
    assert got == EXPECTED
    assert sum(got.values()) == 44_013


def test_every_entry_has_exactly_one_primary_row(conn):
    [(entries,)] = q(conn, "SELECT count(*) FROM sdn_entries")
    [(primaries, distinct)] = q(conn, "SELECT count(*), count(DISTINCT uid) FROM sdn_names WHERE name_kind = 'primary'")
    assert entries == primaries == distinct


def test_every_alias_is_indexed(conn):
    [(aliases,)] = q(conn, "SELECT count(*) FROM sdn_aliases")
    [(rows,)] = q(conn, "SELECT count(*) FROM sdn_names WHERE name_kind = 'alias'")
    assert rows == aliases


@pytest.mark.parametrize("index, method, column", [
    ("sdn_names_trgm", "gin", "name_norm gin_trgm_ops"),
    ("sdn_names_phon", "gin", "phonetic_keys"),
    ("sdn_names_acro", "gin", "acronyms"),
    ("sdn_names_uid", "btree", "uid"),
    ("sdn_names_type", "btree", "entry_type"),
])
def test_indexes_exist(conn, index, method, column):
    [(defn,)] = q(conn, "SELECT indexdef FROM pg_indexes WHERE tablename = 'sdn_names' AND indexname = %s", index)
    assert f"USING {method} ({column})" in defn


def test_table_equals_a_fresh_build(conn):
    """Every one of the 44,013 rows, every column, recomputed with today's code."""
    with conn.cursor() as cur:
        fresh = Counter(
            (uid, et, kind, weak, disp, norm, stem, tuple(keys), tuple(acr))
            for uid, et, kind, weak, disp, norm, stem, keys, acr in build_rows(cur))
        cur.execute("SELECT uid, entry_type, name_kind, is_weak, display, name_norm, stem, phonetic_keys, acronyms "
                    "FROM sdn_names")
        stored = Counter(
            (uid, et, kind, weak, disp, norm, stem, tuple(keys), tuple(acr))
            for uid, et, kind, weak, disp, norm, stem, keys, acr in cur.fetchall())
    only_fresh, only_stored = fresh - stored, stored - fresh
    assert not only_fresh and not only_stored, (
        f"sdn_names is stale — re-run build_name_index. e.g. code builds {list(only_fresh)[:2]} "
        f"but table has {list(only_stored)[:2]}")


def test_column_invariants(conn):
    bad = q(conn, """
        SELECT uid, display FROM sdn_names WHERE
              name_norm = ''
           OR (entry_type = 'individual' AND (stem IS NOT NULL OR acronyms <> '{}'))
           OR (entry_type <> 'individual' AND (stem IS NULL OR stem = ''))
           OR name_norm !~ '^[A-Z0-9]+( [A-Z0-9]+)*$'
        LIMIT 5""")
    assert bad == []


def test_individual_names_in_natural_order(conn):
    """Only five individual rows still contain ', ' — OFAC names with two commas
    or a 'JR' suffix (see test_index_columns.test_natural_order_multi_comma)."""
    rows = q(conn, "SELECT uid FROM sdn_names WHERE entry_type = 'individual' AND display LIKE '%%, %%' ORDER BY uid")
    assert [r[0] for r in rows] == [7223, 7866, 10851, 10853, 52352]


def test_no_acronym_below_minimum(conn):
    [(shortest,)] = q(conn, "SELECT min(length(a)) FROM (SELECT unnest(acronyms) a FROM sdn_names) t")
    assert shortest == MIN_ACRONYM_LENGTH == 3


def test_acronym_group_sizes(conn):
    """The ambiguous-acronym design assumes the largest tie group fits ACRONYM_LIMIT,
    counted in ROWS — the SQL LIMIT counts rows, not entries. With the 3-character
    minimum the largest is FYY7, 62 rows / 62 entries. (Before it, AS had 124 rows
    over 100 entries and lost 21.) If a list reload pushes a group past the limit,
    this fails before blocking silently truncates a tie group."""
    rows = q(conn, """
        SELECT a, count(*), count(DISTINCT uid) FROM
          (SELECT unnest(acronyms) a, uid FROM sdn_names WHERE entry_type <> 'individual') t
        GROUP BY a ORDER BY 2 DESC LIMIT 1""")
    assert rows == [("FYY7", 62, 62)]
    assert rows[0][1] < ACRONYM_LIMIT

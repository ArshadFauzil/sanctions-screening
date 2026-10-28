"""Helpers shared by the Phase 3 tests: shortlist lookups and a fake DB connection.

FakeConnection
--------------
`Blocker` talks to Postgres through exactly four statements:

1. `SET pg_trgm.similarity_threshold = …`       (constructor)
2. `SELECT stem FROM sdn_names …`               (constructor: IDF table)
3. `GENERATE_SQL`  -> rows of (uid, generator)  (block: candidate generation)
4. `NAMES_SQL`     -> every name of those uids  (block: rescoring input)

The fake answers each one from Python lists, so the ranking, merging, acronym
and tie-group logic in `Blocker.block()` can be tested exactly, with no
database. This is called a *test double*: an object that stands in for a real
dependency so the code under test can be exercised in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NAME_COLS = ("uid", "entry_type", "name_kind", "is_weak", "display", "name_norm", "stem")


def shortlisted(cands, uid: int) -> bool:
    return any(c.uid == uid for c in cands)


def rank_of(cands, uid: int):
    for i, c in enumerate(cands, 1):
        if c.uid == uid:
            return i
    return None


def name_row(uid, display, *, entry_type="individual", name_kind="primary", is_weak=False, stem=None):
    """One sdn_names row, built the same way build_name_index.build_rows() builds it."""
    from common.names import normalize, strip_legal_forms

    norm = normalize(display)
    if stem is None and entry_type != "individual":
        stem = strip_legal_forms(display)
    return {"uid": uid, "entry_type": entry_type, "name_kind": name_kind, "is_weak": is_weak,
            "display": display, "name_norm": norm, "stem": stem}


@dataclass
class _Col:
    name: str


@dataclass
class FakeConnection:
    """generated: list of (uid, generator) the GENERATE_SQL call returns.
    names: list of name_row() dicts; NAMES_SQL returns those whose uid AND
    entry_type match the query parameters, as the real SQL does.
    stems: the entity primary stems the IDF table is built from."""

    generated: list[tuple[int, str]] = field(default_factory=list)
    names: list[dict] = field(default_factory=list)
    stems: list[str | None] = field(default_factory=list)
    executed: list[tuple[str, object]] = field(default_factory=list)

    def cursor(self):
        return _FakeCursor(self)


class _FakeCursor:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self._rows: list = []
        self.description = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        s = " ".join(sql.split())
        if s.startswith("SET pg_trgm.similarity_threshold"):
            self._rows = []
        elif s.startswith("SELECT stem FROM sdn_names"):
            self._rows = [(x,) for x in self.conn.stems]
        elif "UNION ALL" in s:  # GENERATE_SQL
            self._rows = list(self.conn.generated)
        elif s.startswith("SELECT uid, entry_type, name_kind"):  # NAMES_SQL
            uids, types = set(params["uids"]), set(params["types"])
            rows = [r for r in self.conn.names if r["uid"] in uids and r["entry_type"] in types]
            self.description = [_Col(c) for c in NAME_COLS]
            self._rows = [tuple(r[c] for c in NAME_COLS) for r in rows]
        else:  # pragma: no cover - would mean Blocker issued SQL we do not model
            raise AssertionError(f"unexpected SQL: {s[:80]}")

    def fetchall(self):
        return list(self._rows)

    def sql_calls(self):  # pragma: no cover
        return self.conn.executed


def generate_params(conn: FakeConnection) -> dict:
    """The parameters of the last GENERATE_SQL call."""
    for sql, params in reversed(conn.executed):
        if "UNION ALL" in sql:
            return params
    raise AssertionError("GENERATE_SQL was never executed")

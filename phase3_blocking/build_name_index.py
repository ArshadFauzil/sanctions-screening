#!/usr/bin/env python3
"""Build the blocking index: one row per SDN name, primary AND alias (Phase 3).

    uv run python -m phase3_blocking.build_name_index

Creates table sdn_names (~44,000 rows) plus three indexes, from the tables
load_sdn.py already filled. Safe to re-run: drops and rebuilds.

What each column is for
-----------------------
display        Natural order for individuals ("Vadim Yevgenyevich Dengin"),
               because customer records are written that way. OFAC's
               "DENGIN, Vadim Yevgenyevich" scores 0.15 lower in Jaro-Winkler
               against the same customer record purely from word order.
name_norm      normalize(display). The trigram (GIN) index sits on this.
stem           Entities only: every legal form stripped (strip_legal_forms).
               Rescoring compares stems so "LTD" vs "LLC" is not a difference.
phonetic_keys  Metaphone of each token of 3+ characters. MOHAMMED, MOHAMED and
               MUHAMMAD all become MHMT. Short tokens (AL, BIN) are skipped:
               they would match thousands of rows and blocking would drown.
acronyms       Entities only: initials of the name under several stripping
               rules, so a customer onboarded as "SCD" or "C4GC" can be found.
               No similarity measure can find an acronym; only exact lookup.
               Only acronyms of MIN_ACRONYM_LENGTH (3)+ characters are stored.
"""

from __future__ import annotations

import argparse
import os
import time

import jellyfish
import psycopg
from dotenv import load_dotenv

from common.names import normalize, strip_legal_forms, strip_legal_suffix
from common.reference_tables import MIN_ACRONYM_LENGTH

DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
DROP TABLE IF EXISTS sdn_names;
CREATE TABLE sdn_names (
    id            BIGSERIAL PRIMARY KEY,
    uid           INTEGER NOT NULL REFERENCES sdn_entries(uid) ON DELETE CASCADE,
    entry_type    TEXT    NOT NULL,
    name_kind     TEXT    NOT NULL CHECK (name_kind IN ('primary', 'alias')),
    is_weak       BOOLEAN NOT NULL DEFAULT FALSE,
    display       TEXT    NOT NULL,
    name_norm     TEXT    NOT NULL,
    stem          TEXT,
    phonetic_keys TEXT[]  NOT NULL DEFAULT '{}',
    acronyms      TEXT[]  NOT NULL DEFAULT '{}'
);
"""

INDEXES = """
CREATE INDEX sdn_names_trgm ON sdn_names USING gin (name_norm gin_trgm_ops);
CREATE INDEX sdn_names_phon ON sdn_names USING gin (phonetic_keys);
CREATE INDEX sdn_names_acro ON sdn_names USING gin (acronyms);
CREATE INDEX sdn_names_uid  ON sdn_names (uid);
CREATE INDEX sdn_names_type ON sdn_names (entry_type);
ANALYZE sdn_names;
"""


def natural_order(name: str, is_individual: bool) -> str:
    """'LAST, First Middle' -> 'First Middle LAST' for individuals only.

    Entity names can legitimately contain a comma ("CHINA 49 GROUP CO.,
    LIMITED") and must never be reordered.
    """
    if is_individual and ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def phonetic_keys(norm: str) -> list[str]:
    keys = {jellyfish.metaphone(t) for t in norm.split() if len(t) >= 3 and t.isalpha()}
    return sorted(k for k in keys if k)


def initials(stem: str) -> str:
    return "".join(t[0] for t in stem.split())


def acronyms(name: str) -> list[str]:
    """Initials under three stripping rules; a customer might use any of them.

    CHINA 49 GROUP CO., LIMITED ->  C4GCL (nothing stripped)
                                    C4GC  (last legal form stripped)
                                    C4G   (all legal forms stripped)

    Variants shorter than MIN_ACRONYM_LENGTH (3) are not stored: ALPHA OMEGA
    gets no acronym at all, because "AO" would tie with dozens of entities.
    """
    variants = {initials(normalize(name)), initials(strip_legal_suffix(name)),
                initials(strip_legal_forms(name))}
    return sorted(v for v in variants if len(v) >= MIN_ACRONYM_LENGTH)


def build_rows(cur) -> list[tuple]:
    rows = []
    cur.execute("SELECT uid, entry_type, first_name, last_name, primary_name FROM sdn_entries")
    entries = cur.fetchall()
    types = {uid: et for uid, et, *_ in entries}
    for uid, et, first, last, primary in entries:
        disp = f"{first} {last}" if (et == "individual" and first) else (last or primary)
        rows.append((uid, et, "primary", False, disp))
    cur.execute("SELECT uid, alias_name, is_weak FROM sdn_aliases")
    for uid, alias, weak in cur.fetchall():
        et = types[uid]
        rows.append((uid, et, "alias", weak, natural_order(alias, et == "individual")))

    out = []
    for uid, et, kind, weak, disp in rows:
        norm = normalize(disp)
        if not norm:
            continue
        is_ind = et == "individual"
        out.append((uid, et, kind, weak, disp, norm,
                    None if is_ind else strip_legal_forms(disp),
                    phonetic_keys(norm),
                    [] if is_ind else acronyms(disp)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()
    load_dotenv()
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        ap.error("--database-url not given and $DATABASE_URL is not set")

    t0 = time.time()
    with psycopg.connect(url, connect_timeout=10) as conn, conn.cursor() as cur:
        rows = build_rows(cur)
        cur.execute(DDL)
        with cur.copy("COPY sdn_names (uid, entry_type, name_kind, is_weak, display, name_norm, "
                      "stem, phonetic_keys, acronyms) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
        cur.execute(INDEXES)
        cur.execute("SELECT name_kind, count(*) FROM sdn_names GROUP BY 1 ORDER BY 1")
        counts = dict(cur.fetchall())
    print(f"sdn_names built: {sum(counts.values()):,} rows {counts} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

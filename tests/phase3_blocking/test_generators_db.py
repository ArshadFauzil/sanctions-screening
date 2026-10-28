"""Step 1 against real Postgres — each generator ISOLATED, then together.

GENERATE_SQL is three parenthesised SELECTs joined by UNION ALL. The helper
`branch()` runs one of them on its own, so a test can say "the trigram
generator alone finds this" or "the phonetic generator alone returns only rows
sharing a key" — properties invisible once the three are merged.
"""

from __future__ import annotations

import jellyfish
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from common.names import normalize
from phase3_blocking.blocking import ACRONYM_LIMIT, GENERATE_SQL, SEARCH_TYPES, Blocker
from phase3_blocking.build_name_index import phonetic_keys
from tests.phase3_blocking.test_index_columns import _trigram_similarity
from tests.phase3_blocking.helpers import shortlisted

BRANCHES = dict(zip(("trigram", "phonetic", "acronym"), (s.strip() for s in GENERATE_SQL.split("UNION ALL"))))


def params(name, subject_type, lim=50, alim=ACRONYM_LIMIT):
    norm = normalize(name)
    qacr, qinit = Blocker._query_acronym_inputs(norm, subject_type)
    keys = [k for k in {jellyfish.metaphone(t) for t in norm.split() if len(t) >= 3 and t.isalpha()} if k]
    return {"q": norm, "types": SEARCH_TYPES[subject_type], "keys": keys, "qacr": qacr, "qinit": qinit,
            "lim": lim, "alim": alim}


def branch(conn, which, name, subject_type, *, threshold=0.3, **kw):
    """Rows (uid, display, entry_type, similarity, phonetic_keys, acronyms, name_norm) one generator returns."""
    sql = BRANCHES[which].strip("()")
    sql = sql.replace("SELECT uid, 'trigram' AS gen", "SELECT uid, display, entry_type, similarity(name_norm, %(q)s), "
                      "phonetic_keys, acronyms, name_norm")
    sql = sql.replace(f"SELECT uid, '{which}'", "SELECT uid, display, entry_type, similarity(name_norm, %(q)s), "
                      "phonetic_keys, acronyms, name_norm")
    with conn.cursor() as cur:
        cur.execute(f"SET pg_trgm.similarity_threshold = {float(threshold)}")
        try:
            cur.execute(sql, params(name, subject_type, **kw))
            return cur.fetchall()
        finally:
            cur.execute("SET pg_trgm.similarity_threshold = 0.3")   # the shared Blocker's setting


TRIGRAM_QUERIES = [
    ("Vadim Yevgendevich Dengin", "individual"), ("Hassen Mofakhami Shahrestani", "individual"),
    ("Mohammed Hassan", "individual"), ("Andrew Al-Libi", "individual"),
    ("Sina Composite Delijan", "entity"), ("China 49 Group", "entity"), ("Mar Azul", "entity"),
    ("Alpha General Trading", "entity"),
]


# --------------------------------------------------------------------------
# pg_trgm itself
# --------------------------------------------------------------------------

def test_pg_trgm_documented_numbers(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT similarity('HASSEN', 'HASSAN'), similarity(%s, %s)",
                    (normalize("Yazbeck Ahmad Mohamad"), normalize("Ahmad Mohamad Yazbeck")))
        hassen, yazbeck = cur.fetchone()
    assert hassen == pytest.approx(0.40)
    assert yazbeck == pytest.approx(1.0)        # trigrams are a bag: word order is free


@given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=10),
       st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=10))
@settings(max_examples=200, deadline=None)
def test_python_trigram_model_matches_pg_trgm(conn, a, b):
    """The single-token trigram model used in test_index_columns agrees with Postgres."""
    with conn.cursor() as cur:
        cur.execute("SELECT similarity(%s, %s)", (a, b))
        assert cur.fetchone()[0] == pytest.approx(_trigram_similarity(a, b), abs=1e-6)


# --------------------------------------------------------------------------
# trigram generator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, st_", TRIGRAM_QUERIES)
def test_trigram_rows_respect_threshold_order_limit_and_type(conn, name, st_):
    rows = branch(conn, "trigram", name, st_)
    sims = [r[3] for r in rows]
    assert all(s >= 0.3 - 1e-6 for s in sims)
    assert sims == sorted(sims, reverse=True)
    assert len(rows) <= 50
    assert {r[2] for r in rows} <= set(SEARCH_TYPES[st_])


@pytest.mark.parametrize("name, st_", TRIGRAM_QUERIES)
@pytest.mark.parametrize("lo, hi", [(0.3, 0.4), (0.3, 0.5), (0.4, 0.6), (0.5, 0.8)])
def test_higher_threshold_returns_a_subset(conn, name, st_, lo, hi):
    loose = {r[0:2] for r in branch(conn, "trigram", name, st_, threshold=lo, lim=10_000)}
    tight = {r[0:2] for r in branch(conn, "trigram", name, st_, threshold=hi, lim=10_000)}
    assert tight <= loose


@pytest.mark.parametrize("lim", [1, 5, 20])
def test_trigram_limit_counts_rows(conn, lim):
    rows = branch(conn, "trigram", "Mohammed Hassan", "individual", lim=lim)
    assert len(rows) == lim


def test_dengin_trigram_rows(conn):
    """blocking.md: 5 rows pass at 0.3, the true entry first at 0.821."""
    rows = branch(conn, "trigram", "Vadim Yevgendevich Dengin", "individual")
    assert len(rows) == 5
    assert rows[0][0] == 36778 and rows[0][3] == pytest.approx(0.821, abs=5e-4)


def test_hassen_trigram_rows(conn):
    """blocking.md: 3 rows pass, all belonging to the true entry."""
    rows = branch(conn, "trigram", "Hassen Mofakhami Shahrestani", "individual")
    assert len(rows) == 3 and {r[0] for r in rows} == {44962}


@pytest.mark.parametrize("variant", [
    "Vadim Yevgenyevich Dengin",       # exact
    "Dengin Vadim Yevgenyevich",       # word order
    "Vadim Yevgenyevich Dengim",       # typo, last letter
    "Wadim Yevgenyevich Dengin",       # typo, first letter
    "Vadim Yevgenievich Dengin",       # transliteration
    "Vadim Dengin",                    # dropped patronymic
    "VADIM-YEVGENYEVICH DENGIN",       # punctuation
])
def test_trigram_alone_finds_dengin_variants(conn, variant):
    assert 36778 in {r[0] for r in branch(conn, "trigram", variant, "individual")}


# --------------------------------------------------------------------------
# phonetic generator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, st_", TRIGRAM_QUERIES)
def test_phonetic_rows_share_a_key_and_respect_type(conn, name, st_):
    qkeys = set(phonetic_keys(normalize(name)))
    rows = branch(conn, "phonetic", name, st_)
    assert len(rows) <= 50
    for r in rows:
        assert qkeys & set(r[4])
        assert r[2] in SEARCH_TYPES[st_]
    sims = [r[3] for r in rows]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.parametrize("name", ["AL", "AL EL", "BA DE", "M. A.", "C4G 49"])
def test_no_keys_means_no_phonetic_rows(conn, name):
    assert branch(conn, "phonetic", name, "individual") == []


@pytest.mark.parametrize("spelling", ["MOHAMMED", "MOHAMED", "MUHAMMAD"])
def test_phonetic_connects_mohammed_spellings(conn, spelling):
    """All three share key MHMT: each finds names written with the others."""
    rows = branch(conn, "phonetic", f"{spelling} Yazbeck", "individual", lim=10_000)
    found_tokens = {t for r in rows for t in r[6].split()}
    assert {"MOHAMMED", "MOHAMED", "MUHAMMAD", "MOHAMAD"} & found_tokens - {spelling}


def test_phonetic_finds_what_trigram_cannot(conn):
    """A query whose true row is below the trigram threshold but shares a key.
    'Hasan Mufakami Shahrestany': respelled throughout."""
    q = "Hasan Mufakami Shahrestany"
    tri = {r[0] for r in branch(conn, "trigram", q, "individual")}
    pho = {r[0] for r in branch(conn, "phonetic", q, "individual")}
    assert 44962 in pho
    assert 44962 in tri or 44962 in pho     # at least one generator; record which
    print(f"\ntrigram finds it: {44962 in tri}; phonetic finds it: {44962 in pho}")


# --------------------------------------------------------------------------
# acronym generator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("acr", ["C4G", "C4GC", "C4GCL"])
def test_china_49_group_every_acronym_variant(conn, acr):
    rows = branch(conn, "acronym", acr, "entity")
    assert {r[0] for r in rows} == {33271}


def test_scd_acronym(conn):
    rows = branch(conn, "acronym", "SCD", "entity")
    assert 36004 in {r[0] for r in rows}
    assert all("SCD" in r[5] for r in rows)


def test_long_unique_acronym(conn):
    """blocking.md: the 13-letter acronym an early length cap missed."""
    assert {r[0] for r in branch(conn, "acronym", "GGRASPRDRLDCV", "entity")} == {12196}


def test_acronym_rows_ordered_by_uid_and_limited(conn):
    rows = branch(conn, "acronym", "AHF", "entity")      # 26 rows over 6 entries
    uids = [r[0] for r in rows]
    assert uids == sorted(uids)
    assert len(rows) <= ACRONYM_LIMIT


def test_full_name_query_finds_entity_listed_under_its_acronym(conn):
    """The other direction: the customer gives the full name, the list has the
    acronym ('BNC'). name_norm = ANY(expansions)."""
    rows = branch(conn, "acronym", "Banco Nacional Cubano", "entity")
    assert 306 in {r[0] for r in rows}


@pytest.mark.parametrize("acr", ["AS", "TS", "AR"])
def test_acronym_branch_ignores_two_letter_queries(conn, acr):
    assert branch(conn, "acronym", acr, "entity") == []


def test_acronym_generator_never_runs_for_individuals(conn):
    assert branch(conn, "acronym", "SCD", "individual") == []


def test_acronym_branch_needs_single_token_or_expansion(conn):
    """A multi-token query is never itself treated as an acronym."""
    rows = branch(conn, "acronym", "SCD LLC", "entity")
    assert 36004 not in {r[0] for r in rows}   # the gap logged as a finding


# --------------------------------------------------------------------------
# type filter, through the whole Blocker
# --------------------------------------------------------------------------

def test_entity_query_reaches_vessels(blocker):
    """A company named after a listed ship is a real collision the adjudicator must see."""
    out = blocker.block("Mar Azul", "entity")
    assert any(c.uid == 4238 and c.entry_type == "vessel" for c in out)


def test_individual_query_never_returns_non_individuals(blocker):
    for name in ["Mar Azul", "China 49 Group", "Sina Composite Delijan Company", "SCD"]:
        assert all(c.entry_type == "individual" for c in blocker.block(name, "individual"))


def test_entity_query_never_returns_individuals(blocker):
    for name in ["Vadim Yevgenyevich Dengin", "Hassan Mofakhami Shahrestani", "Talha Al-Libi"]:
        assert all(c.entry_type != "individual" for c in blocker.block(name, "entity"))


# --------------------------------------------------------------------------
# the generators together
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, st_", TRIGRAM_QUERIES)
def test_union_equals_the_branches(conn, blocker, name, st_):
    """Every shortlisted entry was produced by the generators it claims."""
    found = {g: {r[0] for r in branch(conn, g, name, st_)} for g in BRANCHES}
    for c in blocker.block(name, st_):
        for g in c.generators:
            assert c.uid in found[g]
        for g in set(BRANCHES) - set(c.generators):
            assert c.uid not in found[g]


@pytest.mark.parametrize("name, st_", TRIGRAM_QUERIES)
def test_blocking_is_deterministic(blocker, name, st_):
    a = [c.to_dict() for c in blocker.block(name, st_)]
    b = [c.to_dict() for c in blocker.block(name, st_)]
    assert a == b


@pytest.mark.parametrize("variant", [str.lower, str.upper, str.title,
                                     lambda s: s.replace(" ", "  "),
                                     lambda s: s.replace(" ", "-"),
                                     lambda s: f"  {s}. "])
@pytest.mark.parametrize("name, st_", TRIGRAM_QUERIES)
def test_normalisation_invariance(blocker, name, st_, variant):
    """Case, spacing, hyphens and punctuation must not change the shortlist."""
    assert ([c.to_dict() for c in blocker.block(variant(name), st_)]
            == [c.to_dict() for c in blocker.block(name, st_)])


@pytest.mark.parametrize("k", [1, 3, 5, 10, 25])
def test_k_is_honoured_on_real_data(conn, k):
    b = Blocker(conn, k=k)
    try:
        out = b.block("Mohammed Hassan", "individual")
        assert len(out) == k
        assert not any(c.ambiguous for c in out)
    finally:
        with conn.cursor() as cur:
            cur.execute("SET pg_trgm.similarity_threshold = 0.3")


SINGLE_TOKEN_ALIAS = pytest.mark.xfail(strict=True, reason=(
    "FINDING: uid 8187 has the one-word weak alias 'HASSAN'. Symmetric Monge-Elkan scores "
    "it 1.0 in the alias->query direction, so ANY query containing Hassan scores >= ~0.9 "
    "against it (0.948 here) and it outranks the true entry (0.945). 674 individual "
    "names are a single token, 670 of them weak aliases."))


@pytest.mark.parametrize("variant", [
    "Hassan Mofakhami Shahrestani", "Hassen Mofakhami Shahrestani", "Hasan Mofakhami Shahrestani",
    "Hassan Mofakhmi Shahrashtani", "Shahrestani Hassan Mofakhami", "H. Mofakhami Shahrestani",
    pytest.param("Hassan Shahrestani", marks=SINGLE_TOKEN_ALIAS),
    "Hassan Mofakhami", "HASSAN MOFAKHAMI-SHAHRESTANI",
])
def test_every_variant_of_one_listed_name_ranks_it_first(blocker, variant):
    """One real entry (uid 44962) under transliteration, order, initial, drop and
    punctuation variants: ranked first."""
    out = blocker.block(variant, "individual")
    assert out and out[0].uid == 44962


@pytest.mark.parametrize("variant", [
    "Hassan Mofakhami Shahrestani", "Hassen Mofakhami Shahrestani", "Hasan Mofakhami Shahrestani",
    "Hassan Mofakhmi Shahrashtani", "Shahrestani Hassan Mofakhami", "H. Mofakhami Shahrestani",
    "Hassan Shahrestani", "Hassan Mofakhami", "HASSAN MOFAKHAMI-SHAHRESTANI",
])
def test_every_variant_of_one_listed_name_is_shortlisted(blocker, variant):
    assert 44962 in {c.uid for c in blocker.block(variant, "individual")}


def test_single_token_weak_alias_inflation(blocker):
    """Characterisation of the finding above: a two-word query that merely CONTAINS
    a one-word weak alias scores ~0.9 against it."""
    [c] = [c for c in blocker.block("Hassan Smith", "individual") if c.uid == 8187]
    assert c.matched_name == "HASSAN" and c.is_weak and c.score > 0.85

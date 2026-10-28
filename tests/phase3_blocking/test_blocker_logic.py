"""Blocker.block() logic with a fake database (see helpers.FakeConnection).

What is tested here is everything block() does AROUND the SQL: which
parameters each generator receives, how the three generators' results are
merged, best-of-names rescoring, the acronym score floor, ranking and ties,
the k cut-off, and ambiguous acronym tie groups. The SQL itself is tested
against real Postgres in test_generators_db.py.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from common.names import normalize
from common.reference_tables import MIN_ACRONYM_LENGTH
from phase3_blocking.blocking import ACRONYM_LIMIT, ACRONYM_SCORE, SEARCH_TYPES, Blocker, Candidate
from phase3_blocking.build_name_index import phonetic_keys
from tests.phase3_blocking.helpers import FakeConnection, generate_params, name_row

STEMS = ["ALPHA", "SINA BANK", "MELLAT BANK", "PROPITIOUS FOREVER TRADING", None, ""]


def make(generated=(), names=(), **kw):
    conn = FakeConnection(generated=list(generated), names=list(names), stems=STEMS)
    return Blocker(conn, **kw), conn


# --------------------------------------------------------------------------
# constructor
# --------------------------------------------------------------------------

def test_constructor_sets_threshold_and_builds_idf():
    b, conn = make(trigram_threshold=0.45)
    assert "similarity_threshold = 0.45" in conn.executed[0][0]
    assert b.idf.n == 4                      # None and "" stems are skipped
    assert b.k == 10 and b.limit == 50 and b.acronym_limit == ACRONYM_LIMIT


def test_threshold_cannot_inject_sql():
    """The threshold is interpolated into SQL text, so it must be forced to a float."""
    with pytest.raises(ValueError):
        make(trigram_threshold="0.3; DROP TABLE sdn_names")


# --------------------------------------------------------------------------
# query side: what each generator is asked for
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   ", "--.,'", "\t\n", "()"])
def test_empty_query_returns_nothing_and_runs_no_sql(name):
    b, conn = make()
    n_before = len(conn.executed)
    assert b.block(name, "individual") == []
    assert len(conn.executed) == n_before


def test_unknown_subject_type_raises():
    b, _ = make()
    with pytest.raises(KeyError):
        b.block("MAR AZUL", "vessel")


@pytest.mark.parametrize("subject_type", ["individual", "entity"])
def test_generator_parameters(subject_type):
    b, conn = make(generator_limit=17, acronym_limit=33)
    b.block("Mohammed al-Hassan", subject_type)
    p = generate_params(conn)
    assert p["q"] == "MOHAMMED AL HASSAN"
    assert p["types"] == SEARCH_TYPES[subject_type]
    assert sorted(p["keys"]) == ["HSN", "MHMT"]           # AL has no key
    assert p["lim"] == 17 and p["alim"] == 33


def test_search_types():
    """Entity queries also search vessels and aircraft; individuals only individuals."""
    assert SEARCH_TYPES == {"individual": ["individual"], "entity": ["entity", "vessel", "aircraft"]}


@given(st.text(max_size=60))
@settings(max_examples=300)
def test_query_keys_equal_index_keys(raw):
    """The query's phonetic keys are computed inline in block(); the index's by
    phonetic_keys(). If the two formulas ever diverged, the phonetic generator
    would silently stop matching. Same set for any input."""
    b, conn = make()
    b.block(raw, "individual")
    if normalize(raw):
        assert sorted(generate_params(conn)["keys"]) == phonetic_keys(normalize(raw))


# --------------------------------------------------------------------------
# acronym inputs: (acronym the query might BE, acronyms the query might EXPAND to)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, as_acronym, expansions", [
    ("SCD", ["SCD"], []),
    ("ABC", ["ABC"], []),                                 # 3 letters: the minimum
    ("AB", [], []),                                       # 2 letters: never looked up (MIN_ACRONYM_LENGTH)
    ("AS", [], []),
    ("A", [], []),                                        # 1 letter: never
    ("C4G", ["C4G"], []),                                 # digits allowed
    ("GGRASPRDRLDCV", ["GGRASPRDRLDCV"], []),             # 13 letters: the old cap of 8 missed it
    ("ROSNEFT", ["ROSNEFT"], []),                         # any single token is TRIED as an acronym
    ("SINA COMPOSITE DELIJAN COMPANY", [], ["SCD", "SCDC"]),
    ("CHINA 49 GROUP CO., LIMITED", [], ["C4G", "C4GC", "C4GCL"]),
    ("ALPHA OMEGA", [], []),                              # would expand to AO: too short
    ("ALPHA OMEGA SIGMA", [], ["AOS"]),
    ("OOO STROYINVEST", [], []),                          # OS / S: too short
    ("M&S", [], ["MAS"]),                                 # '&' -> AND (see finding in README)
])
def test_query_acronym_inputs(raw, as_acronym, expansions):
    assert Blocker._query_acronym_inputs(normalize(raw), "entity") == (as_acronym, expansions)


@pytest.mark.parametrize("raw", ["SCD", "SINA COMPOSITE DELIJAN", "AB"])
def test_individuals_never_use_acronyms(raw):
    assert Blocker._query_acronym_inputs(normalize(raw), "individual") == ([], [])


@pytest.mark.xfail(strict=True, reason=(
    "FINDING: an acronym followed by a legal form ('SCD LLC', 'C4G LTD') is two tokens, "
    "so it is not tried as an acronym. Perturbation-pair matrix: acronym + legal_suffix_swap "
    "is found 9/106 times. Fix: also try strip_legal_forms(query) as an acronym."))
@pytest.mark.parametrize("raw, acr", [("SCD LLC", "SCD"), ("C4G LTD", "C4G"), ("OOO SCD", "SCD")])
def test_acronym_with_legal_form_is_tried_as_acronym(raw, acr):
    as_acronym, _ = Blocker._query_acronym_inputs(normalize(raw), "entity")
    assert acr in as_acronym


@given(st.lists(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=10),
                min_size=1, max_size=6))
@settings(max_examples=300)
def test_acronym_input_properties(toks):
    norm = " ".join(toks)
    as_acr, exp = Blocker._query_acronym_inputs(norm, "entity")
    assert as_acr == ([norm] if len(toks) == 1 and len(norm) >= MIN_ACRONYM_LENGTH else [])
    assert exp == sorted(set(exp)) and all(len(e) >= MIN_ACRONYM_LENGTH for e in exp)
    if len(toks) == 1:
        assert exp == []


# --------------------------------------------------------------------------
# merging, rescoring, ranking
# --------------------------------------------------------------------------

def test_no_generated_candidates_skips_rescoring():
    b, conn = make(generated=[])
    assert b.block("Vadim Dengin", "individual") == []
    assert not any(sql.lstrip().startswith("SELECT uid, entry_type") for sql, _ in conn.executed)


def test_best_name_of_an_entry_wins_and_carries_its_metadata():
    """Generation may find an alias; rescoring looks at EVERY name of the entry."""
    names = [
        name_row(1, "Hassan Mofakhmi Shahrashtani", name_kind="alias", is_weak=True),
        name_row(1, "Hassan Mofakhami Shahrestani"),
    ]
    b, _ = make(generated=[(1, "trigram")], names=names)
    [c] = b.block("Hassen Mofakhami Shahrestani", "individual")
    assert c.matched_name == "Hassan Mofakhami Shahrestani"
    assert c.name_kind == "primary" and c.is_weak is False
    assert c.score == pytest.approx(0.978, abs=5e-4)


def test_weak_alias_kept_when_it_is_the_best_name():
    names = [name_row(1, "Completely Different"), name_row(1, "Abu Hajir", name_kind="alias", is_weak=True)]
    b, _ = make(generated=[(1, "phonetic")], names=names)
    [c] = b.block("Abu Hajir", "individual")
    assert (c.name_kind, c.is_weak, c.score) == ("alias", True, 1.0)


def test_generators_are_merged_per_entry_and_sorted():
    names = [name_row(1, "Vadim Dengin")]
    gen = [(1, "trigram"), (1, "phonetic"), (1, "trigram")]   # trigram may return two rows of one uid
    b, _ = make(generated=gen, names=names)
    [c] = b.block("Vadim Dengin", "individual")
    assert c.generators == ["phonetic", "trigram"]


def test_ranking_by_score_then_uid():
    names = [name_row(5, "Vadim Dengin"), name_row(3, "Vadim Dengin"), name_row(9, "Vadim Dengyn"),
             name_row(1, "Ivan Petrov")]
    b, _ = make(generated=[(u, "trigram") for u in (5, 3, 9, 1)], names=names)
    out = b.block("Vadim Dengin", "individual")
    assert [c.uid for c in out] == [3, 5, 9, 1]          # exact ties broken by the smaller uid
    assert [c.score for c in out] == sorted((c.score for c in out), reverse=True)


@pytest.mark.parametrize("k", [1, 2, 3, 10])
def test_k_cut(k):
    names = [name_row(u, f"Vadim Dengin{'X' * u}") for u in range(1, 6)]
    b, _ = make(generated=[(u, "trigram") for u in range(1, 6)], names=names, k=k)
    out = b.block("Vadim Dengin", "individual")
    assert len(out) == min(k, 5)
    assert out[0].uid == 1


def test_individual_rescoring_ignores_stem():
    row = name_row(1, "Vadim Dengin")
    row["stem"] = "SOMETHING ELSE"
    b, _ = make(generated=[(1, "trigram")], names=[row])
    assert b.block("Vadim Dengin", "individual")[0].score == 1.0


def test_entity_rescoring_uses_stems_on_both_sides():
    names = [name_row(1, "ALPHA GENERAL TRADING L.L.C.", entry_type="entity")]
    b, _ = make(generated=[(1, "trigram")], names=names)
    assert b.block("OOO Alpha General Trading FZE", "entity")[0].score == 1.0


def test_entity_rescoring_falls_back_to_name_norm_without_stem():
    row = name_row(7, "MAR AZUL", entry_type="vessel")
    row["stem"] = None
    b, _ = make(generated=[(7, "trigram")], names=[row])
    [c] = b.block("Mar Azul", "entity")
    assert (c.entry_type, c.score) == ("vessel", 1.0)


# --------------------------------------------------------------------------
# acronym score floor and tie groups
# --------------------------------------------------------------------------

def test_acronym_hit_gets_floor_score():
    names = [name_row(1, "SINA COMPOSITE DELIJAN COMPANY", entry_type="entity")]
    b, _ = make(generated=[(1, "acronym")], names=names)
    [c] = b.block("SCD", "entity")
    assert c.score == ACRONYM_SCORE == 0.90
    assert c.ambiguous is False                           # the only one: not ambiguous


def test_acronym_floor_never_lowers_a_better_score():
    names = [name_row(1, "SCD", entry_type="entity")]
    b, _ = make(generated=[(1, "acronym"), (1, "trigram")], names=names)
    assert b.block("SCD", "entity")[0].score == 1.0


def test_floor_only_for_the_acronym_generator():
    names = [name_row(1, "SINA COMPOSITE DELIJAN COMPANY", entry_type="entity")]
    b, _ = make(generated=[(1, "trigram")], names=names)
    assert b.block("SCD", "entity")[0].score == 0.0


def test_ambiguous_tie_group_is_returned_whole_beyond_k():
    """5 entities share the acronym; k=2. All 5 come back, flagged; the 2
    non-acronym candidates beyond k are NOT appended."""
    ent = lambda u, n: name_row(u, n, entry_type="entity")  # noqa: E731
    names = [ent(u, f"ALPHA SIERRA CHARLIE {chr(64 + u)}") for u in range(1, 6)] + [ent(10, "ASC"), ent(11, "BRAVO")]
    gen = [(u, "acronym") for u in range(1, 6)] + [(10, "trigram"), (11, "trigram")]
    b, _ = make(generated=gen, names=names, k=2)
    out = b.block("ASC", "entity")
    uids = [c.uid for c in out]
    assert uids[:2] == [10, 1]                    # exact name 1.0 first, then the acronym ties by uid
    assert set(uids) == {10, 1, 2, 3, 4, 5}       # 11 (trigram only, beyond k) dropped
    assert len(uids) == len(set(uids))            # no duplicates
    assert all(c.ambiguous for c in out if "acronym" in c.generators)
    assert not any(c.ambiguous for c in out if "acronym" not in c.generators)


def test_ambiguity_counts_acronym_hits_beyond_k_too():
    ent = lambda u, n: name_row(u, n, entry_type="entity")  # noqa: E731
    names = [ent(1, "ASC"), ent(2, "ALPHA SIERRA CHARLIE")]
    b, _ = make(generated=[(1, "trigram"), (1, "acronym"), (2, "acronym")], names=names, k=1)
    out = b.block("ASC", "entity")
    assert [c.uid for c in out] == [1, 2] and all(c.ambiguous for c in out)


def test_candidate_to_dict():
    c = Candidate(uid=1, score=0.123456, entry_type="entity", matched_name="X", name_kind="primary",
                  is_weak=False, generators=["acronym"], ambiguous=True)
    d = c.to_dict()
    assert d["score"] == 0.1235
    assert set(d) == {"uid", "score", "entry_type", "matched_name", "name_kind", "is_weak", "generators", "ambiguous"}


# --------------------------------------------------------------------------
# invariants over random fake result sets
# --------------------------------------------------------------------------

GEN = st.sampled_from(["trigram", "phonetic", "acronym"])
NAME = st.sampled_from(["ALPHA SIERRA", "ASC", "ALPHA", "SIERRA ALPHA", "SINA BANK", "ASX TRADING LLC", "BRAVO"])


@given(st.lists(st.tuples(st.integers(1, 15), GEN), min_size=0, max_size=30),
       st.lists(st.tuples(st.integers(1, 15), NAME), min_size=0, max_size=40),
       st.integers(1, 6), st.sampled_from(["ASC", "ALPHA SIERRA", "SINA"]))
@settings(max_examples=400)
def test_block_invariants(gen, rows, k, query):
    names = [name_row(u, n, entry_type="entity") for u, n in rows]
    b, _ = make(generated=gen, names=names, k=k)
    out = b.block(query, "entity")
    uids = [c.uid for c in out]
    with_names = {u for u, _ in rows}
    generated = {u for u, _ in gen}
    acr = {u for u, g in gen if g == "acronym"} & with_names
    assert len(uids) == len(set(uids))                         # unique
    assert set(uids) <= generated & with_names                 # only generated entries with a name
    assert len(out) <= k + len(acr)                            # beyond k only via tie groups
    head = out[:k]
    assert [(-c.score, c.uid) for c in head] == sorted((-c.score, c.uid) for c in head)
    for c in out:
        assert 0.0 <= c.score <= 1.0
        if "acronym" in c.generators:
            assert c.score >= ACRONYM_SCORE
        assert c.ambiguous == ("acronym" in c.generators and len(acr) > 1)
    if len(acr) > 1:
        assert acr <= set(uids)                                # the whole tie group is returned

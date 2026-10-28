"""Step 2 — rescoring: token_sim, Monge-Elkan (individuals), IDF + entity_score.

Two kinds of test:

* example tests  — the worked examples in blocking.md, recomputed exactly;
* property tests — `hypothesis` generates hundreds of random inputs and checks
  a rule that must hold for ALL of them (symmetry, bounds, word-order
  invariance…). When one fails, hypothesis shrinks it to the smallest
  counter-example, which is usually the bug report you need.

No database (a small synthetic IDF table stands in for the list; the
real-list figures are checked in test_documented_cases.py).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from rapidfuzz.distance import JaroWinkler

from common.names import normalize
from phase3_blocking.blocking import (
    IDF,
    TOKEN_MATCH_FLOOR,
    _monge_elkan,
    _weighted_me,
    entity_score,
    individual_score,
    token_sim,
)

TOKEN = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=12)
TOKENS = st.lists(TOKEN, min_size=1, max_size=6)
EPS = 1e-9


# --------------------------------------------------------------------------
# token_sim
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    ("HASSAN", "HASSAN", 1.0),
    ("A", "AYMAN", 0.9),          # initial consistent with the full token
    ("AYMAN", "A", 0.9),          # ...in either direction
    ("A", "A", 1.0),
    ("A", "B", 0.0),              # inconsistent initials
    ("B", "AYMAN", 0.0),
])
def test_token_sim_initials_rule(a, b, expected):
    assert token_sim(a, b) == expected


def test_initials_rule_beats_raw_jaro_winkler():
    """blocking.md: JW scores 'A' vs 'AYMAN' ~0.76 purely on length."""
    assert JaroWinkler.similarity("A", "AYMAN") < 0.8 < token_sim("A", "AYMAN")


def test_token_sim_hassen_hassan():
    assert token_sim("HASSEN", "HASSAN") == pytest.approx(0.933, abs=5e-4)


@given(TOKEN, TOKEN)
@settings(max_examples=500)
def test_token_sim_bounds_and_symmetry(a, b):
    s = token_sim(a, b)
    assert 0.0 <= s <= 1.0
    assert s == pytest.approx(token_sim(b, a))


@given(TOKEN)
def test_token_sim_identity(a):
    assert token_sim(a, a) == 1.0


@given(TOKEN, TOKEN)
def test_token_sim_only_identity_scores_one(a, b):
    assume(a != b)
    assert token_sim(a, b) < 1.0


# --------------------------------------------------------------------------
# Monge-Elkan / individual_score
# --------------------------------------------------------------------------

def test_monge_elkan_empty():
    assert _monge_elkan([], ["A"]) == 0.0
    assert _monge_elkan(["A"], []) == 0.0
    assert individual_score([], ["HASSAN"]) == 0.0


def test_documented_hassen_example_restores_order():
    """blocking.md: whole-string JW ranks the 3-difference alias ABOVE the
    1-difference primary; token-wise Monge-Elkan restores the right order."""
    q = normalize("Hassen Mofakhami Shahrestani")
    primary = normalize("Hassan Mofakhami Shahrestani")
    alias = normalize("Hassan Mofakhmi Shahrashtani")
    assert JaroWinkler.similarity(q, primary) == pytest.approx(0.912, abs=5e-4)
    assert JaroWinkler.similarity(q, alias) == pytest.approx(0.925, abs=5e-4)
    assert JaroWinkler.similarity(q, alias) > JaroWinkler.similarity(q, primary)   # the flaw
    p = individual_score(q.split(), primary.split())
    a = individual_score(q.split(), alias.split())
    assert p == pytest.approx(0.978, abs=5e-4)
    assert a == pytest.approx(0.947, abs=5e-4)   # earlier blocking.md: 0.946
    assert p > a                                  # the fix


def test_documented_word_order_example():
    """blocking.md: 'Yazbeck Ahmad Mohamad' vs 'Ahmad Mohamad Yazbeck'.
    Whole-string JW is low; Monge-Elkan is exactly 1."""
    q, t = normalize("Yazbeck Ahmad Mohamad"), normalize("Ahmad Mohamad Yazbeck")
    assert JaroWinkler.similarity(q, t) < 0.7     # measured 0.676
    assert individual_score(q.split(), t.split()) == 1.0


def test_natural_order_matters_for_whole_string_jw():
    """blocking.md: the reason sdn_names stores individuals in natural order."""
    q = normalize("Vadim Yevgendevich Dengin")
    ofac = JaroWinkler.similarity(q, normalize("DENGIN, Vadim Yevgenyevich"))
    natural = JaroWinkler.similarity(q, normalize("Vadim Yevgenyevich Dengin"))
    assert ofac == pytest.approx(0.802, abs=5e-4)
    assert natural == pytest.approx(0.959, abs=5e-4)   # earlier blocking.md: 0.955
    assert natural - ofac > 0.15


@pytest.mark.parametrize("query, target, low, high", [
    ("A. Mohammed Hassan", "Ayman Mohammed Hassan", 0.96, 0.97),   # initial: 0.9 on one of 3 tokens
    ("Mohammed Hassan", "Mohammed Ali Hassan", 0.88, 0.95),        # dropped middle name: mild penalty
    ("Mohammed Hassan", "Mohammed Hassan", 1.0, 1.0),
    ("Hassan Mohammed", "Mohammed Hassan", 1.0, 1.0),              # order-free
    ("Mohammed Hassan Hassan", "Mohammed Hassan", 1.0, 1.0),       # repeated token is not penalised
    ("John Smith", "Mohammed Hassan", 0.0, 0.6),
])
def test_individual_score_ranges(query, target, low, high):
    s = individual_score(normalize(query).split(), normalize(target).split())
    assert low - EPS <= s <= high + EPS


@given(TOKENS, TOKENS)
@settings(max_examples=500)
def test_individual_score_bounds_and_symmetry(q, t):
    s = individual_score(q, t)
    assert 0.0 <= s <= 1.0 + EPS
    assert s == pytest.approx(individual_score(t, q))


@given(TOKENS, st.randoms())
@settings(max_examples=300)
def test_individual_score_word_order_invariant(q, rnd):
    shuffled = q[:]
    rnd.shuffle(shuffled)
    assert individual_score(q, shuffled) == pytest.approx(1.0)


@given(TOKENS, TOKEN)
@settings(max_examples=300)
def test_extra_token_never_increases_identity_score(q, extra):
    """A target with an additional token scores no higher than the exact name."""
    assert individual_score(q, q + [extra]) <= 1.0 + EPS
    if all(token_sim(extra, x) < 1.0 for x in q):
        assert individual_score(q, q + [extra]) < 1.0


@given(TOKENS)
def test_individual_identity(q):
    assert individual_score(q, q) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# IDF
# --------------------------------------------------------------------------

STEMS = (["ALPHA TRADING", "BETA TRADING", "GAMMA TRADING", "DELTA TRADING", "PROPITIOUS FOREVER TRADING"]
         + ["SINA BANK", "SINA ENERGY", "MELLAT BANK"] + ["TRADING TRADING TRADING"])


@pytest.fixture(scope="module")
def small_idf():
    return IDF(STEMS)


def test_idf_formula(small_idf):
    n = len(STEMS)
    assert small_idf.n == n
    assert small_idf.df["TRADING"] == 6       # counted once per stem, even when repeated
    assert small_idf("TRADING") == pytest.approx(math.log((n + 1) / 7) + 1)
    assert small_idf("PROPITIOUS") == pytest.approx(math.log((n + 1) / 2) + 1)


def test_idf_unseen_token_is_maximal(small_idf):
    assert small_idf("NEVERSEEN") == pytest.approx(small_idf.max)
    assert all(small_idf(t) <= small_idf.max for t in ["TRADING", "SINA", "PROPITIOUS"])


def test_idf_rarer_is_heavier(small_idf):
    assert small_idf("PROPITIOUS") > small_idf("SINA") > small_idf("TRADING")


def test_idf_minimum_is_one():
    """A token in every stem has idf ln((N+1)/(N+1)) + 1 = 1, never 0: it still counts."""
    idf = IDF(["A X", "B X", "C X"])
    assert idf("X") == pytest.approx(1.0)


# --------------------------------------------------------------------------
# entity_score
# --------------------------------------------------------------------------

@pytest.mark.parametrize("q, t, low, high", [
    ("PROPITIOUS FOREVER TRADING", "PROPITIOUS FOREVER TRADING", 1.0, 1.0),
    ("FOREVER PROPITIOUS TRADING", "PROPITIOUS FOREVER TRADING", 1.0, 1.0),   # order-free
    ("PROPITOUS FOREVER TRADING", "PROPITIOUS FOREVER TRADING", 0.9, 1.0),    # typo in the rare token
    ("BISHOP TRADING", "PROPITIOUS FOREVER TRADING", 0.0, 0.35),              # shares only the generic token
    ("SINA", "SINA BANK", 0.55, 0.9),                                          # subset
])
def test_entity_score_ranges(small_idf, q, t, low, high):
    s = entity_score(q.split(), t.split(), small_idf)
    assert low - EPS <= s <= high + EPS


def test_sharing_rare_token_beats_sharing_generic_token(small_idf):
    rare = entity_score(["PROPITIOUS", "X"], ["PROPITIOUS", "Y"], small_idf)
    generic = entity_score(["TRADING", "X"], ["TRADING", "Y"], small_idf)
    assert rare > generic


def test_token_match_floor(small_idf):
    """Below the floor a token counts as UNMATCHED (contributes 0), not as partial."""
    a, b = "ABCDEFGH", "ABXYZQRS"
    assert token_sim(a, b) < TOKEN_MATCH_FLOOR
    assert _weighted_me([a], [b], small_idf) == 0.0
    c = "ABCDEFGX"
    assert token_sim(a, c) >= TOKEN_MATCH_FLOOR
    assert _weighted_me([a], [c], small_idf) == pytest.approx(token_sim(a, c))


def test_acronym_scores_zero_by_similarity(small_idf):
    """blocking.md: no similarity measure can find an acronym — hence the generator."""
    assert entity_score(["SCD"], ["SINA", "COMPOSITE", "DELIJAN"], small_idf) == 0.0


def test_entity_score_empty(small_idf):
    assert entity_score([], ["SINA"], small_idf) == 0.0
    assert entity_score(["SINA"], [], small_idf) == 0.0


@given(TOKENS, TOKENS)
@settings(max_examples=500)
def test_entity_score_bounds_and_symmetry(q, t):
    idf = IDF(STEMS)
    s = entity_score(q, t, idf)
    assert 0.0 <= s <= 1.0 + EPS
    assert s == pytest.approx(entity_score(t, q, idf))


@given(TOKENS, st.randoms())
@settings(max_examples=300)
def test_entity_score_identity_and_order(q, rnd):
    idf = IDF(STEMS)
    shuffled = q[:]
    rnd.shuffle(shuffled)
    assert entity_score(q, shuffled, idf) == pytest.approx(1.0)

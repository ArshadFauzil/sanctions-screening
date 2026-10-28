"""Every worked example in phase3_blocking/blocking.md, run end to end on the real list.

Where the document's number no longer matches the code (the code moved on
after the doc was written), the test asserts the MEASURED value and says so
in a comment; tests/README.md lists those doc corrections.
"""

from __future__ import annotations

import pytest
from rapidfuzz.distance import JaroWinkler

from common.names import normalize, strip_legal_forms
from phase3_blocking.blocking import ACRONYM_SCORE, entity_score
from tests.phase3_blocking.helpers import rank_of

DENGIN, SHAHRESTANI, YAZBECK = 36778, 44962, 31754
SCD_UID, CHINA49, GGRAS, MAROC, PROPITIOUS = 36004, 33271, 12196, 44473, 54143


# ---- individuals ----------------------------------------------------------

def test_dengin_typo(blocker):
    """'Vadim Yevgendevich Dengin' -> true entry first, found by both generators."""
    out = blocker.block("Vadim Yevgendevich Dengin", "individual")
    assert out[0].uid == DENGIN
    assert out[0].generators == ["phonetic", "trigram"]
    assert out[0].score == pytest.approx(0.989, abs=5e-3)   # blocking.md example JSON prints 0.984


def test_hassen_typo_ranks_primary_first(blocker):
    out = blocker.block("Hassen Mofakhami Shahrestani", "individual")
    assert out[0].uid == SHAHRESTANI
    assert out[0].name_kind == "primary"
    assert out[0].score == pytest.approx(0.978, abs=5e-4)


def test_yazbeck_word_order(blocker):
    out = blocker.block("Yazbeck Ahmad Mohamad", "individual")
    assert out[0].uid == YAZBECK and out[0].score == 1.0


def test_andrew_al_libi_hard_negative(blocker):
    """blocking.md: a negative built from an AL-LIBI entry surfaces the listed
    AL-LIBI individuals — correct, the adjudicator clears them. Scores 0.82-0.89
    (an earlier version of the doc printed 0.67-0.73, from an older scorer)."""
    out = blocker.block("Andrew Al-Libi", "individual")
    al_libi = {48700, 7150, 18600, 9599, 12888}
    assert al_libi <= {c.uid for c in out}
    scores = [c.score for c in out if c.uid in al_libi]
    assert 0.8 <= min(scores) and max(scores) <= 0.9


# ---- entities: acronyms ------------------------------------------------------

def test_scd_every_similarity_measure_fails(conn, idf):
    full = "SINA COMPOSITE DELIJAN COMPANY"
    assert JaroWinkler.similarity("SCD", full) == pytest.approx(0.730, abs=5e-4)
    assert entity_score(["SCD"], strip_legal_forms(full).split(), idf) == 0.0
    with conn.cursor() as cur:
        cur.execute("SELECT similarity('SCD', %s)", (full,))
        assert cur.fetchone()[0] < 0.3


def test_scd_found_by_acronym(blocker):
    out = blocker.block("SCD", "entity")
    c = out[rank_of(out, SCD_UID) - 1]
    assert c.score == ACRONYM_SCORE and "acronym" in c.generators and not c.ambiguous


@pytest.mark.parametrize("acr", ["C4GCL", "C4GC", "C4G"])
def test_china_49_group_acronym_variants(blocker, acr):
    out = blocker.block(acr, "entity")
    assert [c.uid for c in out] == [CHINA49]


def test_thirteen_letter_acronym(blocker):
    assert [c.uid for c in blocker.block("GGRASPRDRLDCV", "entity")] == [GGRAS]


@pytest.mark.parametrize("acr", ["AS", "AR", "TS", "SS", "AB", "EM"])
def test_two_letter_acronyms_are_not_looked_up(blocker, acr):
    """Domain-owner decision (Oct 2026): acronyms shorter than 3 characters are a
    long shot for a true match. AS was shared by 100 entities, TS by 41, AR by
    35 — a tie group the adjudicator could only escalate. Such a query now goes
    through trigram/phonetic only, like any other short name."""
    out = blocker.block(acr, "entity")
    assert not any("acronym" in c.generators or c.ambiguous for c in out)
    assert len(out) <= blocker.k


def test_largest_remaining_tie_group_is_returned_whole(blocker):
    """With 3+ characters the largest shared acronym is FYY7 (62 entries, 62 rows):
    returned whole, flagged, beyond k."""
    out = blocker.block("FYY7", "entity")
    assert sum(c.ambiguous for c in out) == 62


@pytest.mark.parametrize("acr", ["SCD", "AHF", "LLCA"])
def test_three_and_four_letter_acronyms_are_looked_up(blocker, acr):
    assert any("acronym" in c.generators for c in blocker.block(acr, "entity"))


# ---- entities: legal forms and IDF -----------------------------------------------

def test_maroc_pvt_ltd_fixed(blocker):
    """The Phase 3 bug: PVT LTD is two legal forms. Fixed by strip_legal_forms."""
    out = blocker.block("MAROC INTERNATIONAL PVT", "entity")
    assert out[0].uid == MAROC and out[0].score == 1.0


def test_bishop_trading_negative_scores_low(blocker, idf):
    """A benchmark negative sharing only the generic TRADING. An earlier blocking.md printed
    0.108 (computed before legal-form stripping); measured 0.228 — still far
    below the 0.5 operating floor."""
    s = entity_score(strip_legal_forms("Bishop Trading Ltd").split(),
                     strip_legal_forms("PROPITIOUS FOREVER TRADING CO LTD").split(), idf)
    assert s == pytest.approx(0.228, abs=5e-3)
    assert s < 0.5
    out = blocker.block("Bishop Trading Ltd", "entity")
    assert all(c.score < 0.5 for c in out if c.uid == PROPITIOUS)


@pytest.mark.parametrize("token, df, weight", [
    ("TRADING", 495, 4.00), ("INTERNATIONAL", 266, 4.62), ("GENERAL", 121, 5.40),
    ("SINA", 12, 7.64), ("PROPITIOUS", 1, 9.51),
    # An earlier blocking.md printed COMPANY 1,920 / 2.65 — counted over raw names. IDF is
    # built over STEMS, and COMPANY is a legal suffix, so trailing COMPANYs are
    # stripped first: 140 / 5.26 (doc corrected 2026-10-28).
    ("COMPANY", 140, 5.26),
])
def test_idf_table(idf, token, df, weight):
    assert idf.n == 9_975
    assert idf.df[token] == df
    assert idf(token) == pytest.approx(weight, abs=5e-3)


def test_general_is_rarer_than_trading(idf):
    """blocking.md: 'the list decides what is generic, not intuition'."""
    assert idf("GENERAL") > idf("TRADING")


# ---- vessels ---------------------------------------------------------------------

def test_vessel_collision(blocker):
    out = blocker.block("Mar Azul", "entity")
    assert out[0].uid == 4238 and out[0].entry_type == "vessel" and out[0].score == 1.0


def test_normalize_equivalence_classes_used_in_doc():
    assert normalize("AL-FAISAL") == normalize("AL FAISAL") == normalize("AL.FAISAL")

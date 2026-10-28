"""Step 0 — the columns build_name_index.py computes for every SDN name.

natural_order()  display column (individuals reordered, entities untouched)
phonetic_keys()  metaphone keys the phonetic generator looks up
acronyms()       initials the acronym generator looks up
build_rows()     how primary names and aliases become sdn_names rows

No database: build_rows() is driven by a fake cursor.
"""

from __future__ import annotations

import itertools

import jellyfish
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from common.names import normalize, strip_legal_forms
from common.reference_tables import MIN_ACRONYM_LENGTH, TRANSLITERATION_GROUPS
from phase3_blocking.build_name_index import acronyms, build_rows, initials, natural_order, phonetic_keys

# --------------------------------------------------------------------------
# natural_order
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, is_ind, expected", [
    ("DENGIN, Vadim Yevgenyevich", True, "Vadim Yevgenyevich DENGIN"),
    ("AL-ZAWAHIRI, Ayman", True, "Ayman AL-ZAWAHIRI"),
    ("DE LOS REYES, Feliciano", True, "Feliciano DE LOS REYES"),
    ("MADURO MOROS", True, "MADURO MOROS"),                   # no comma: unchanged
    ("CHINA 49 GROUP CO., LIMITED", False, "CHINA 49 GROUP CO., LIMITED"),  # entity: never reordered
    ("SMITH, John", False, "SMITH, John"),                    # entity with ", ": still never reordered
    ("SMITH,John", True, "SMITH,John"),                       # comma without space: not a name split
])
def test_natural_order(name, is_ind, expected):
    assert natural_order(name, is_ind) == expected


@pytest.mark.parametrize("name, expected", [
    # Characterisation: OFAC names with TWO commas are split only at the first.
    # Real rows on the 23 Sep 2026 list (uid 7223, 7866). Rescoring is
    # token-based and order-free, so the stray comma costs nothing there.
    ("KNOWLES, JR., Samuel", "JR., Samuel KNOWLES"),
    ("MARUF, Taha, Muhyi al-Din", "Taha, Muhyi al-Din MARUF"),
])
def test_natural_order_multi_comma(name, expected):
    assert natural_order(name, True) == expected


@given(st.text(min_size=1), st.text(min_size=1))
def test_natural_order_preserves_tokens(last, first):
    """Reordering must never add or lose characters."""
    name = f"{last}, {first}"
    assert sorted(natural_order(name, True)) == sorted(name.replace(", ", " ", 1))


# --------------------------------------------------------------------------
# phonetic_keys
# --------------------------------------------------------------------------

@pytest.mark.parametrize("word, key", [
    # the table in blocking.md
    ("MOHAMMED", "MHMT"), ("MOHAMED", "MHMT"), ("MUHAMMAD", "MHMT"),
    ("HASSAN", "HSN"), ("HASSEN", "HSN"), ("HASAN", "HSN"),
    ("SHAHRESTANI", "XRSTN"), ("SHAHRASHTANI", "XRXTN"),
])
def test_documented_metaphone_keys(word, key):
    assert jellyfish.metaphone(word) == key


def test_shahrestani_variants_do_not_share_a_key():
    """blocking.md: the phonetic generator misses this pair; trigram must cover it."""
    assert phonetic_keys("SHAHRESTANI") != phonetic_keys("SHAHRASHTANI")


@pytest.mark.parametrize("norm, expected", [
    ("AL EL BA", []),                            # every token shorter than 3: no keys at all
    # Characterisation: an earlier blocking.md said "AL and BIN" get no key, but BIN has 3
    # letters, so BIN, IBN, BEN, ABU and BINT DO get keys (BN, IBN, BN, AB, BNT).
    # Functionally harmless (LIMIT 50, best similarity first) but every "bin"
    # name hits a very common key. Logged as a finding in tests/README.md.
    ("AL BIN ABU", ["AB", "BN"]),
    ("MOHAMMED AL HASSAN", ["HSN", "MHMT"]),     # AL skipped; sorted
    ("MOHAMMED MOHAMED", ["MHMT"]),              # duplicates collapse
    ("BITBANK3 C4G", []),                        # tokens with digits are skipped
    ("", []),
])
def test_phonetic_keys(norm, expected):
    assert phonetic_keys(norm) == expected


@given(st.lists(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=12), max_size=6))
def test_phonetic_keys_properties(toks):
    norm = " ".join(toks)
    keys = phonetic_keys(norm)
    assert keys == sorted(set(keys))
    assert all(keys)
    assert len(keys) <= sum(1 for t in toks if len(t) >= 3 and t.isalpha())


def _group_pairs():
    for g in TRANSLITERATION_GROUPS:
        words = sorted(w for w in g if w.isalpha() and len(w) >= 3)
        yield from itertools.combinations(words, 2)


TRANSLIT_PAIRS = list(_group_pairs())

# Which transliteration siblings the phonetic generator can connect on its own.
# Measured; pinned so a jellyfish upgrade that changes metaphone is noticed.
# Pairs NOT in this set rely on the trigram generator (or the rescoring of a
# row another generator found).
PHONETIC_EQUIVALENT = {
    tuple(sorted(p)) for p in TRANSLIT_PAIRS
    if jellyfish.metaphone(p[0]) == jellyfish.metaphone(p[1])
}


@pytest.mark.parametrize("a, b", TRANSLIT_PAIRS, ids=lambda x: x)
def test_transliteration_pairs_phonetic_matrix(a, b):
    """Every pair of spellings in every TRANSLITERATION_GROUP (168 pairs): either
    they share a metaphone key (the phonetic generator connects them) or their
    trigram similarity reaches the 0.3 threshold (the trigram generator does).
    Pairs reachable by neither must be exactly the pinned list below."""
    same_key = jellyfish.metaphone(a) == jellyfish.metaphone(b)
    tri = _trigram_similarity(a, b)
    unreachable = not same_key and tri < 0.3
    pinned = (a, b) in KNOWN_UNREACHABLE_TOKEN_PAIRS
    detail = f"{a}/{b}: metaphone {jellyfish.metaphone(a)} vs {jellyfish.metaphone(b)}, trigram {tri:.2f}"
    if unreachable:
        assert pinned, "NEW unreachable pair — a regression: " + detail
    else:
        assert not pinned, "pair is now reachable — remove it from KNOWN_UNREACHABLE_TOKEN_PAIRS: " + detail


def _trigrams(w: str) -> set[str]:
    """pg_trgm's definition: lower-case, pad two spaces in front and one behind."""
    s = f"  {w.lower()} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _trigram_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / len(ta | tb)


def test_trigram_reimplementation_matches_documented_example():
    """blocking.md: HASSEN vs HASSAN share 4 of 10 trigrams = 0.40 (pg_trgm agrees,
    see test_generators_db)."""
    assert _trigram_similarity("HASSEN", "HASSAN") == pytest.approx(0.40)


# Single-TOKEN sibling pairs that neither share a metaphone key nor reach the
# trigram threshold (0.3) on their own. Measured, then pinned. On a whole name
# the other tokens usually carry the row past the threshold, so these are
# RISKS, not certain misses — test_generators_db substitutes them into real
# listed names to measure the actual effect. If a library upgrade or a code
# change makes a pair reachable, the matrix test fails and asks you to remove
# it from this list (the list may only shrink).
KNOWN_UNREACHABLE_TOKEN_PAIRS = {
    ('ALEXANDER', 'OLEKSANDR'), ('ALEXANDR', 'OLEKSANDR'), ('BEN', 'IBN'), ('BIN', 'IBN'), ('EBRAHIM', 'IBRAHEEM'),
    ('EVGENII', 'YEVGENY'), ('EVGENY', 'YEVGENIY'), ('HALID', 'KHALED'), ('IGOR', 'IHOR'), ('JOSEPH', 'YOUSEF'),
    ('JOSEPH', 'YOUSSEF'), ('JOSEPH', 'YUSSEF'), ('JOSEPH', 'YUSUF'), ('JSC', 'OAO'), ('JSC', 'OJSC'),
    ('JSC', 'PAO'), ('JSC', 'ZAO'), ('LLC', 'OOO'), ('MEHMET', 'MOHAMED'), ('MEHMET', 'MOHAMMAD'),
    ('MEHMET', 'MOHAMMED'), ('MEHMET', 'MOHD'), ('MEHMET', 'MUHAMED'), ('MEHMET', 'MUHAMMAD'), ('MICHAIL', 'MYKHAILO'),
    ('MOHAMMAD', 'MOHD'), ('MOHAMMED', 'MOHD'), ('MOHD', 'MUHAMED'), ('MOHD', 'MUHAMMAD'), ('OAO', 'OJSC'),
    ('OAO', 'PAO'), ('OAO', 'ZAO'), ('OJSC', 'PAO'), ('OJSC', 'ZAO'), ('OMAR', 'UMAR'),
    ('OMER', 'UMAR'), ('PAO', 'ZAO'), ('SAEED', 'SAYYID'), ('SAEID', 'SAYED'), ('SAID', 'SAYED'),
    ('SERGEI', 'SERHIY'), ('SERGEY', 'SERHIY'), ('SERGUEI', 'SERHIY'),
}


# --------------------------------------------------------------------------
# acronyms
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("CHINA 49 GROUP CO., LIMITED", ["C4G", "C4GC", "C4GCL"]),   # the documented example
    ("SINA COMPOSITE DELIJAN COMPANY", ["SCD", "SCDC"]),
    ("MAROC INTERNATIONAL PVT LTD", ["MIP", "MIPL"]),     # "MI" is below the 3-character minimum
    ("OOO STROYINVEST", []),                      # OS and S: both below the minimum
    ("ROSNEFT", []),                              # one token: initials 'R' too short
    ("ALPHA OMEGA", []),                          # AO: 2-letter acronyms are not stored
    ("ALPHA OMEGA SIGMA", ["AOS"]),
])
def test_acronyms(name, expected):
    assert acronyms(name) == expected


def test_minimum_acronym_length_is_three():
    """Domain-owner decision (Oct 2026): 2-letter acronyms are a long shot for a
    true match (AS alone was shared by 100 entities) and are not indexed."""
    assert MIN_ACRONYM_LENGTH == 3


def test_ampersand_contributes_an_A():
    """Characterisation: '&' is normalised to AND, so 'M&S TRADING' has acronym
    MAST/MAS, not MST/MS. A customer onboarded as 'MST' would not be found by the
    acronym generator. Logged as a finding in tests/README.md."""
    assert "MST" not in acronyms("M&S TRADING")
    assert "MAST" in acronyms("M&S TRADING")


@given(st.lists(st.sampled_from(["ALPHA", "OMEGA", "SINA", "49", "GROUP", "TRADING", "LLC", "LTD", "OOO", "CO"]),
                min_size=1, max_size=6))
@settings(max_examples=300)
def test_acronym_properties(toks):
    name = " ".join(toks)
    acr = acronyms(name)
    assert acr == sorted(set(acr))
    assert all(len(a) >= MIN_ACRONYM_LENGTH for a in acr)
    # the 'all legal forms stripped' variant is always present when long enough
    full = initials(strip_legal_forms(name))
    assert (full in acr) == (len(full) >= MIN_ACRONYM_LENGTH)
    assert initials(normalize(name)) in acr or len(initials(normalize(name))) < MIN_ACRONYM_LENGTH


# --------------------------------------------------------------------------
# build_rows (fake cursor)
# --------------------------------------------------------------------------

class _Cur:
    def __init__(self, entries, aliases):
        self.entries, self.aliases, self._rows = entries, aliases, []

    def execute(self, sql, params=None):
        self._rows = self.entries if "FROM sdn_entries" in sql else self.aliases

    def fetchall(self):
        return list(self._rows)


ENTRIES = [
    # uid, entry_type, first_name, last_name, primary_name
    (1, "individual", "Vadim Yevgenyevich", "DENGIN", "DENGIN, Vadim Yevgenyevich"),
    (2, "individual", None, "MONONYMOUS", "MONONYMOUS"),
    (3, "entity", None, "CHINA 49 GROUP CO., LIMITED", "CHINA 49 GROUP CO., LIMITED"),
    (4, "entity", None, None, "FALLBACK PRIMARY LLC"),
    (5, "vessel", None, "MAR AZUL", "MAR AZUL"),
    (6, "entity", None, "---", "---"),                  # normalises to empty: dropped
]
ALIASES = [
    (1, "DENGIN, Vadim", False),
    (1, "DENGIN, V.", True),
    (3, "C4G", False),
    (5, "AZUL, MAR", False),                            # vessel alias: NOT reordered
]


@pytest.fixture(scope="module")
def rows():
    out = build_rows(_Cur(ENTRIES, ALIASES))
    keys = ("uid", "entry_type", "name_kind", "is_weak", "display", "name_norm", "stem", "phonetic_keys", "acronyms")
    return [dict(zip(keys, r)) for r in out]


def test_build_rows_counts(rows):
    assert sum(r["name_kind"] == "primary" for r in rows) == 5     # uid 6 dropped
    assert sum(r["name_kind"] == "alias" for r in rows) == 4


def test_primary_display(rows):
    disp = {r["uid"]: r["display"] for r in rows if r["name_kind"] == "primary"}
    assert disp[1] == "Vadim Yevgenyevich DENGIN"      # first + last
    assert disp[2] == "MONONYMOUS"                      # no first name: last only
    assert disp[3] == "CHINA 49 GROUP CO., LIMITED"
    assert disp[4] == "FALLBACK PRIMARY LLC"            # no last_name: primary_name
    assert 6 not in disp


def test_alias_display_and_weak_flag(rows):
    al = [(r["uid"], r["display"], r["is_weak"]) for r in rows if r["name_kind"] == "alias"]
    assert (1, "Vadim DENGIN", False) in al
    assert (1, "V. DENGIN", True) in al
    assert (5, "AZUL, MAR", False) in al


def test_individual_rows_have_no_stem_or_acronyms(rows):
    for r in rows:
        if r["entry_type"] == "individual":
            assert r["stem"] is None and r["acronyms"] == []


def test_non_individual_rows_have_stem_and_acronyms(rows):
    for r in rows:
        if r["entry_type"] != "individual":
            assert r["stem"] == strip_legal_forms(r["display"])
            assert r["acronyms"] == acronyms(r["display"])


def test_every_row_columns_consistent(rows):
    for r in rows:
        assert r["name_norm"] == normalize(r["display"]) != ""
        assert r["phonetic_keys"] == phonetic_keys(r["name_norm"])

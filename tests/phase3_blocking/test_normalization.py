"""normalize(), strip_legal_suffix(), strip_legal_forms() — the input to every generator.

Blocking and the benchmark share these functions (common/names.py). If they
changed behaviour, the index and the query would be normalised differently
and every generator would quietly lose recall, so they are pinned here first.

No database.
"""

from __future__ import annotations

import itertools
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from common.names import (
    find_legal_suffix,
    normalize,
    strip_diacritics,
    strip_legal_forms,
    strip_legal_suffix,
)
from common.reference_tables import LEADING_LEGAL_FORMS, LEGAL_SUFFIXES

ALLOWED = set(string.ascii_uppercase + string.digits + " ")
TRAILING_FORMS = sorted({normalize(x) for x in LEGAL_SUFFIXES if normalize(x)})
LEADING_FORMS = sorted({normalize(x) for x in LEADING_LEGAL_FORMS})
BASES = ["ALPHA OMEGA", "SINA COMPOSITE DELIJAN", "STROYINVEST", "CHINA 49 GROUP", "MAROC INTERNATIONAL"]


# --------------------------------------------------------------------------
# normalize()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    # case
    ("hassan", "HASSAN"),
    ("HaSsAn", "HASSAN"),
    # diacritics: Spanish, French, German, Turkish, Polish, Vietnamese, Nordic
    ("MUÑOZ", "MUNOZ"),
    ("Hélène Côté", "HELENE COTE"),
    ("Müller", "MULLER"),
    ("Şahin Çelik", "SAHIN CELIK"),
    ("Łukasz", "UKASZ"),          # Ł has no NFKD decomposition: becomes a space (see test below)
    ("Nguyễn Văn", "NGUYEN VAN"),
    ("Søren", "S REN"),           # ø likewise — and it SPLITS the token
    ("Ångström", "ANGSTROM"),
    # punctuation becomes a token boundary
    ("AL-FAISAL", "AL FAISAL"),
    ("AL.FAISAL", "AL FAISAL"),
    ("AL FAISAL", "AL FAISAL"),
    ("O'Brien", "O BRIEN"),
    ("'ABD AL-RAHMAN", "ABD AL RAHMAN"),
    ("CHINA 49 GROUP CO., LIMITED", "CHINA 49 GROUP CO LIMITED"),
    ("A/S", "A S"),
    ("(PVT.) LTD.", "PVT LTD"),
    # ampersand becomes the word AND
    ("M&S", "M AND S"),
    ("Black & Decker", "BLACK AND DECKER"),
    # whitespace collapses
    ("  Vadim \t Dengin \n", "VADIM DENGIN"),
    # digits survive
    ("BITBANK3", "BITBANK3"),
    # empty / punctuation-only
    ("", ""),
    ("   ", ""),
    ("--.,'", ""),
])
def test_normalize_examples(raw, expected):
    assert normalize(raw) == expected


def test_normalize_does_not_merge_compound_spellings():
    """Documented in names.normalize: ALFAISAL vs AL FAISAL is NOT converged here."""
    assert normalize("ALFAISAL") != normalize("AL-FAISAL")


@pytest.mark.parametrize("char", ["Ł", "ł", "Ø", "ø", "Đ", "ß"])
def test_letters_without_nfkd_decomposition(char):
    """Characterisation: letters NFKD cannot split are either uppercased to ASCII
    (ß -> SS) or turned into a SPACE (Ł, Ø, Đ), which also splits the token:
    Søren -> 'S REN'. A recall risk for Polish, Nordic and Vietnamese names,
    pinned here so a future fix (a small fold table: Ł->L, Ø->O, Đ->D) is a
    deliberate, visible change."""
    out = normalize(f"X{char}Y")
    assert set(out) <= ALLOWED
    assert out in {"XY", "X Y", "XSSY"}


@given(st.text())
@settings(max_examples=500)
def test_normalize_output_alphabet(s):
    out = normalize(s)
    assert set(out) <= ALLOWED
    assert out == out.strip()
    assert "  " not in out


@given(st.text())
@settings(max_examples=500)
def test_normalize_is_idempotent(s):
    """The index stores normalize(name); the query is normalize(query). Applying it
    twice must change nothing, or index and query could disagree."""
    assert normalize(normalize(s)) == normalize(s)


@given(st.text(alphabet=st.characters(categories=["Lu", "Ll"]), min_size=1))
def test_normalize_is_case_insensitive(s):
    assert normalize(s.lower()) == normalize(s.upper()) or "ß" in s or "ﬀ" in s.lower()


def test_strip_diacritics_keeps_base_letters():
    assert strip_diacritics("ÁÉÍÓÚÑÜ") == "AEIOUNU"


# --------------------------------------------------------------------------
# strip_legal_suffix() — one trailing form (benchmark)  vs
# strip_legal_forms()  — every form, both ends, repeatedly (blocking)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("form", TRAILING_FORMS)
@pytest.mark.parametrize("base", BASES)
def test_every_trailing_form_is_stripped(base, form):
    """All LEGAL_SUFFIXES × several bases: the stem is the bare name."""
    assert strip_legal_forms(f"{base} {form}") == normalize(base)


@pytest.mark.parametrize("form", LEADING_FORMS)
@pytest.mark.parametrize("base", BASES)
def test_every_leading_form_is_stripped(base, form):
    """All LEADING_LEGAL_FORMS × several bases (Russian/CIS 'OOO STROYINVEST')."""
    assert strip_legal_forms(f"{form} {base}") == normalize(base)


@pytest.mark.parametrize("lead, trail", list(itertools.product(
    ["OOO", "JOINT STOCK COMPANY", "LIMITED LIABILITY COMPANY", "PJSC"],
    ["LLC", "LTD", "PVT LTD", "CO., LIMITED", "SDN BHD", "GROUP HOLDING"])))
@pytest.mark.parametrize("base", BASES)
def test_leading_and_trailing_forms_together(base, lead, trail):
    assert strip_legal_forms(f"{lead} {base} {trail}") == normalize(base)


@pytest.mark.parametrize("raw, stem", [
    ("MAROC INTERNATIONAL PVT LTD", "MAROC INTERNATIONAL"),     # the bug fixed in Phase 3
    ("CHINA 49 GROUP CO., LIMITED", "CHINA 49 GROUP"),
    ("OOO STROYINVEST OOO", "STROYINVEST"),                      # benchmark bug fixed in Phase 2
    ("Alpha General Trading L.L.C.", "ALPHA GENERAL TRADING"),
    ("ALPHA GENERAL TRADING FZE", "ALPHA GENERAL TRADING"),
    ("XYZ HOLDINGS LTD", "XYZ HOLDINGS"),                         # HOLDINGS is not a legal form
    ("ABC SDN BHD", "ABC"),
    ("ABC LLC LLC LLC", "ABC"),
])
def test_strip_legal_forms_documented_examples(raw, stem):
    assert strip_legal_forms(raw) == stem


@pytest.mark.parametrize("raw, stem", [
    ("LLC", "LLC"), ("TRUST", "TRUST"), ("FOUNDATION", "FOUNDATION"), ("SDN BHD", "SDN BHD"),
    # several forms and nothing else: stripping stops at the last remaining form
    ("LLC LTD", "LLC"), ("OOO LLC", "OOO"),
])
def test_name_made_only_of_legal_forms_is_never_emptied(raw, stem):
    """A firm literally named 'TRUST' must not be stripped to an empty string."""
    assert strip_legal_forms(raw) == stem


@pytest.mark.parametrize("raw, one, all_", [
    ("MAROC INTERNATIONAL PVT LTD", "MAROC INTERNATIONAL PVT", "MAROC INTERNATIONAL"),
    ("CHINA 49 GROUP CO., LIMITED", "CHINA 49 GROUP CO", "CHINA 49 GROUP"),
    ("OOO STROYINVEST", "STROYINVEST", "STROYINVEST"),   # OOO is also a trailing-form token... as a whole name end
])
def test_suffix_vs_forms(raw, one, all_):
    """strip_legal_suffix removes ONE trailing form (kept unchanged for benchmark
    stability); strip_legal_forms removes all of them."""
    if raw.startswith("OOO"):
        # strip_legal_suffix only looks at the END, so a leading OOO survives
        assert strip_legal_suffix(raw) == "OOO STROYINVEST"
    else:
        assert strip_legal_suffix(raw) == one
    assert strip_legal_forms(raw) == all_


def test_find_legal_suffix_prefers_longest():
    assert find_legal_suffix("ABC L.L.C.") == "L L C"
    assert find_legal_suffix("ABC SDN BHD") == "SDN BHD"
    assert find_legal_suffix("ABC") is None


@pytest.mark.parametrize("raw", ["ACCO", "TRUSTCO", "BASALT", "COSTCO", "ASCO", "INCA"])
def test_forms_only_stripped_as_whole_tokens(raw):
    """'CO' inside 'ACCO', 'AS' inside 'BASALT' must not be stripped."""
    assert strip_legal_forms(raw) == raw


@given(st.text(alphabet=st.sampled_from(list(string.ascii_uppercase) + [" "] * 3), max_size=40))
@settings(max_examples=500)
def test_strip_legal_forms_idempotent_and_nonempty(s):
    out = strip_legal_forms(s)
    assert strip_legal_forms(out) == out
    if normalize(s):
        assert out, "a non-empty name must never be stripped to nothing"
    # the stem's tokens are a contiguous run of the normalised name's tokens
    assert f" {out} " in f" {normalize(s)} " or out == normalize(s)


@given(st.lists(st.sampled_from(TRAILING_FORMS + LEADING_FORMS), min_size=0, max_size=4),
       st.lists(st.sampled_from(TRAILING_FORMS), min_size=0, max_size=4),
       st.sampled_from(BASES))
@settings(max_examples=500)
def test_any_stack_of_forms_reduces_to_base(lead, trail, base):
    """Any sequence of leading forms, then the base, then any sequence of trailing
    forms. Leading forms that are only valid at the end are excluded below."""
    lead = [f for f in lead if f in LEADING_FORMS]
    raw = " ".join([*lead, base, *trail])
    assert strip_legal_forms(raw) == normalize(base)

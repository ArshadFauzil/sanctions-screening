"""Name normalisation shared by the benchmark generator (section 2) and blocking (section 3).

Both stages must normalise identically or blocking recall is measured against
names that were never indexed the same way. Hence one module, imported by both.
"""

from __future__ import annotations

import re
import unicodedata

from reference_tables import GENERIC_CORPORATE_TOKENS, LEGAL_SUFFIXES

# Longest suffix first, so "L.L.C." is matched before "L".
_SUFFIX_PATTERNS = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")


def strip_diacritics(s: str) -> str:
    """NFKD splits 'ñ' into 'n' + combining tilde; drop the combining marks.

    Legacy banking systems frequently store names ASCII-folded, so the same
    person appears as MUNOZ in one record and MUÑOZ in another.
    """
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(name: str) -> str:
    """Uppercase, fold diacritics, drop punctuation, collapse whitespace.

    Punctuation removal is what makes AL-FAISAL, AL FAISAL and AL.FAISAL
    converge. ALFAISAL does not converge here — that is the hyphen/spacing
    perturbation's job to test, and an acronym/compaction index to fix.
    """
    s = strip_diacritics(name).upper().replace("&", " AND ")
    s = _NON_ALNUM.sub(" ", s)
    return _MULTISPACE.sub(" ", s).strip()


def find_legal_suffix(name: str) -> str | None:
    """Return the trailing legal-form token if present, else None."""
    norm = normalize(name)
    for suffix in _SUFFIX_PATTERNS:
        tok = normalize(suffix)
        if tok and (norm == tok or norm.endswith(" " + tok)):
            return tok
    return None


def strip_legal_suffix(name: str) -> str:
    """Canonical entity stem: the name with its legal form removed.

    ALPHA GENERAL TRADING FZE, Alpha General Trading L.L.C. and
    ALPHA GENERAL TRADING all reduce to ALPHA GENERAL TRADING, which is what
    blocking compares. Keep the suffix separately — it hints at jurisdiction.
    """
    norm = normalize(name)
    suffix = find_legal_suffix(name)
    if suffix:
        norm = norm[: -len(suffix)].strip()
    return norm


def tokens(name: str) -> list[str]:
    return normalize(name).split()


def distinctive_tokens(name: str) -> list[str]:
    """Tokens that actually carry identity: not generic, not a legal form, len > 1.

    'ALPHA GENERAL TRADING FZE' -> ['ALPHA']. This is the hand-written floor;
    section 3 replaces the judgement with IDF computed over the real list.
    """
    suffix = find_legal_suffix(name)
    out = []
    for tok in strip_legal_suffix(name).split():
        if tok in GENERIC_CORPORATE_TOKENS or tok == suffix or len(tok) < 2:
            continue
        out.append(tok)
    return out

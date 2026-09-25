"""Perturbation functions: turn a real SDN entry into a realistic customer-record query.

Design notes
------------
*Free ground truth.* Every query is derived from one list entry, so that entry
IS the label. No hand annotation anywhere in this project.

*One function per perturbation, in a registry.* Each is independently testable,
the per-perturbation recall breakdown falls out for free, and adding one never
touches the driver.

*Applicability.* Many perturbations do not apply to a given entry (no middle
name to drop, no hyphen to respace). Those return None. The driver must skip
them — emitting an unchanged name would manufacture a trivially findable
positive and silently inflate every recall figure in the project.

*Determinism, seeded per entry.* The RNG seed is derived from
(uid, perturbation), not from a single global seed. So re-running with a
different sample size, or adding an entry, does not change the perturbations
produced for any other entry. That property is what makes a failure
reproducible while you are debugging it.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field

from names import find_legal_suffix, normalize, strip_diacritics, strip_legal_suffix
from reference_tables import (
    DROPPABLE_PARTICLES,
    SUBSTITUTABLE_SUFFIXES,
    TRANSLITERATION_MAP,
)


@dataclass(frozen=True)
class EntryContext:
    """Everything a perturbation may need about the source SDN entry."""

    uid: int
    entry_type: str                      # 'individual' | 'entity'
    primary_name: str                    # OFAC form: "LAST, First Middle" for individuals
    first_name: str | None
    last_name: str | None
    city: str | None = None              # first address city, for branch qualifiers
    country: str | None = None


@dataclass(frozen=True)
class PerturbResult:
    """A perturbed query.

    drop_attributes lets a perturbation degrade the *attributes* rather than the
    name — the 'we have no DOB for this customer' case, which is the commonest
    real reason a sanctions hit cannot be cleared.
    """

    name: str
    drop_attributes: tuple[str, ...] = field(default=())
    note: str = ""


def seed_for(uid: int, perturbation: str) -> random.Random:
    """Stable per-(entry, perturbation) RNG.

    hashlib rather than hash(): Python randomises str hashing per process unless
    PYTHONHASHSEED is fixed, which would make runs irreproducible.
    """
    digest = hashlib.blake2b(f"{uid}|{perturbation}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def natural_order(ctx: EntryContext) -> str:
    """Customer records store 'Ayman Al-Zawahiri', not 'AL-ZAWAHIRI, Ayman'.

    Every individual query starts from this form, so the benchmark is not
    accidentally testing OFAC's own formatting convention.
    """
    if ctx.entry_type != "individual":
        return ctx.primary_name
    first = (ctx.first_name or "").strip()
    last = (ctx.last_name or "").strip()
    return f"{first} {last}".strip().title() if first else last.title()


def base_name(ctx: EntryContext) -> str:
    return natural_order(ctx) if ctx.entry_type == "individual" else ctx.primary_name


# --------------------------------------------------------------------------
# Shared perturbations
# --------------------------------------------------------------------------

def p_transliteration(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """Swap one token for an equivalent romanisation (Mohammed -> Muhammad).

    One token, not all: replacing every token at once produces a string no
    human would recognise, which tests nothing realistic.
    """
    name = base_name(ctx)
    parts = name.split()
    candidates = [i for i, tok in enumerate(parts) if normalize(tok) in TRANSLITERATION_MAP]
    if not candidates:
        return None
    i = rng.choice(candidates)
    alternatives = TRANSLITERATION_MAP[normalize(parts[i])]
    replacement = rng.choice(alternatives)
    parts[i] = replacement.title() if ctx.entry_type == "individual" else replacement
    out = " ".join(parts)
    return PerturbResult(out, note=f"{parts[i]} <- token {i}") if normalize(out) != normalize(name) else None


def p_typo(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """One character insert / delete / substitute / transpose — a keying error.

    First-character typos are kept deliberately: they are realistic and they are
    the worst case for Jaro-Winkler, which weights shared prefixes.
    """
    name = base_name(ctx)
    positions = [i for i, ch in enumerate(name) if ch.isalpha()]
    if len(positions) < 3:
        return None
    op = rng.choice(("insert", "delete", "substitute", "transpose"))
    i = rng.choice(positions)
    chars = list(name)
    if op == "insert":
        chars.insert(i, rng.choice(string.ascii_lowercase))
    elif op == "delete":
        chars.pop(i)
    elif op == "substitute":
        chars[i] = rng.choice([c for c in string.ascii_lowercase if c != chars[i].lower()])
    else:
        j = i + 1 if i + 1 < len(chars) else i - 1
        chars[i], chars[j] = chars[j], chars[i]
    out = "".join(chars)
    return PerturbResult(out, note=op) if normalize(out) != normalize(name) else None


def p_diacritics(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """ASCII-fold the name. Only applicable when it actually contains marks."""
    name = base_name(ctx)
    folded = strip_diacritics(name)
    return PerturbResult(folded, note="ascii-folded") if folded != name else None


# --------------------------------------------------------------------------
# Individual-only
# --------------------------------------------------------------------------

def p_name_order_inversion(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """Surname-first without a comma — common in Asian and some European records."""
    if ctx.entry_type != "individual" or not (ctx.first_name and ctx.last_name):
        return None
    return PerturbResult(f"{ctx.last_name} {ctx.first_name}".title(), note="surname first")


def p_token_drop(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """Drop a middle name or an Arabic patronymic particle (bin / ibn / al)."""
    if ctx.entry_type != "individual":
        return None
    parts = base_name(ctx).split()
    if len(parts) < 3:
        return None
    particles = [i for i, t in enumerate(parts) if normalize(t) in DROPPABLE_PARTICLES]
    i = rng.choice(particles) if particles else rng.randrange(1, len(parts) - 1)
    out = " ".join(parts[:i] + parts[i + 1:])
    return PerturbResult(out, note=f"dropped '{parts[i]}'")


def p_initialize(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """'Mohamed Hassan Ali' -> 'M. Hassan Ali'. Abbreviated onboarding records."""
    if ctx.entry_type != "individual":
        return None
    parts = base_name(ctx).split()
    if len(parts) < 2 or len(parts[0]) < 2:
        return None
    return PerturbResult(" ".join([parts[0][0] + "."] + parts[1:]), note="given name initialised")


def p_dob_degradation(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """Name unchanged; the DOB is degraded or missing.

    The point of this case is the adjudicator's hardest rule: absent is not
    conflict. A missing DOB means the hit cannot be cleared on DOB, so these
    queries should land in the review band, not be auto-cleared.
    """
    if ctx.entry_type != "individual":
        return None
    mode = rng.choice(("year_only", "absent"))
    return PerturbResult(base_name(ctx), drop_attributes=("dob",) if mode == "absent" else (),
                         note=mode)


# --------------------------------------------------------------------------
# Entity-only
# --------------------------------------------------------------------------

def p_legal_suffix_swap(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """FZE -> LLC, or dropped entirely. The commonest real corporate variance."""
    if ctx.entry_type != "entity":
        return None
    suffix = find_legal_suffix(ctx.primary_name)
    # Slice the suffix off the RAW name, not off strip_legal_suffix()'s normalized
    # output. Otherwise this perturbation silently also respaces hyphens and
    # expands "&", contaminating the per-perturbation recall breakdown with
    # changes that belong to hyphen_spacing and ampersand.
    raw_parts = ctx.primary_name.split()
    stem = ctx.primary_name
    if suffix:
        for k in (2, 1):
            if len(raw_parts) > k and normalize(" ".join(raw_parts[-k:])) == suffix:
                stem = " ".join(raw_parts[:-k])
                break
    stem = stem.strip()
    if not stem:
        return None
    if suffix and rng.random() < 0.5:
        return PerturbResult(stem, note=f"dropped suffix {suffix}")
    replacement = rng.choice([s for s in SUBSTITUTABLE_SUFFIXES if s != suffix])
    return PerturbResult(f"{stem} {replacement}", note=f"{suffix or 'none'} -> {replacement}")


def p_acronym(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """'General Petroleum Company' -> 'GPC'.

    Expected to be the worst case in the whole benchmark: character-level
    similarity and token-set similarity both fail on acronyms. If recall here is
    poor the fix is an explicit acronym index, not a better similarity function.
    """
    if ctx.entry_type != "entity":
        return None
    # All stem tokens, not only the distinctive ones: a bank abbreviating
    # "Alpha General Trading" writes AGT, not A.
    stem_tokens = strip_legal_suffix(ctx.primary_name).split()
    if len(stem_tokens) < 2:
        return None
    return PerturbResult("".join(t[0] for t in stem_tokens), note="acronym of stem tokens")


def p_ampersand(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """'Smith & Sons' <-> 'Smith and Sons'. Trivial, and extremely common."""
    if ctx.entry_type != "entity":
        return None
    name = ctx.primary_name
    if "&" in name:
        return PerturbResult(name.replace("&", "and"), note="& -> and")
    if " AND " in name.upper():
        idx = name.upper().index(" AND ")
        return PerturbResult(name[:idx] + " & " + name[idx + 5:], note="and -> &")
    return None


def p_article_drop(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    if ctx.entry_type != "entity" or not ctx.primary_name.upper().startswith("THE "):
        return None
    return PerturbResult(ctx.primary_name[4:], note="dropped leading article")


def p_hyphen_spacing(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """AL-FAISAL -> AL FAISAL -> ALFAISAL. The third form defeats tokenisation.

    Applies to both subject types: hyphenated surnames (AL-, EL-, double-barrelled
    European names) are as common as hyphenated company names.
    """
    name = base_name(ctx)
    if "-" not in name:
        return None
    return PerturbResult(name.replace("-", " " if rng.random() < 0.5 else ""), note="hyphen respaced")


def p_branch_qualifier(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """'Alpha Trading' -> 'Alpha Trading (Dubai Branch)'. Subsidiary records."""
    if ctx.entry_type != "entity" or not ctx.city:
        return None
    return PerturbResult(f"{ctx.primary_name} ({ctx.city.title()} Branch)", note="branch qualifier")


def p_attribute_degradation(ctx: EntryContext, rng: random.Random) -> PerturbResult | None:
    """Entity equivalent of a missing DOB: no jurisdiction or no registration number."""
    if ctx.entry_type != "entity":
        return None
    dropped = rng.choice((("jurisdiction",), ("registration_number",),
                          ("jurisdiction", "registration_number")))
    return PerturbResult(ctx.primary_name, drop_attributes=dropped, note="attributes withheld")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

INDIVIDUAL_PERTURBATIONS = {
    "transliteration": p_transliteration,
    "name_order_inversion": p_name_order_inversion,
    "token_drop": p_token_drop,
    "initialize": p_initialize,
    "typo": p_typo,
    "diacritics": p_diacritics,
    "hyphen_spacing": p_hyphen_spacing,
    "dob_degradation": p_dob_degradation,
}

ENTITY_PERTURBATIONS = {
    "legal_suffix_swap": p_legal_suffix_swap,
    "acronym": p_acronym,
    "ampersand": p_ampersand,
    "article_drop": p_article_drop,
    "hyphen_spacing": p_hyphen_spacing,
    "branch_qualifier": p_branch_qualifier,
    "transliteration": p_transliteration,
    "typo": p_typo,
    "diacritics": p_diacritics,
    "attribute_degradation": p_attribute_degradation,
}


def perturbations_for(entry_type: str) -> dict:
    return INDIVIDUAL_PERTURBATIONS if entry_type == "individual" else ENTITY_PERTURBATIONS


def apply_all(ctx: EntryContext) -> list[tuple[str, PerturbResult]]:
    """Every applicable perturbation for one entry. Skips inapplicable ones."""
    out = []
    for pname, fn in perturbations_for(ctx.entry_type).items():
        result = fn(ctx, seed_for(ctx.uid, pname))
        if result is not None and result.name.strip():
            out.append((pname, result))
    return out

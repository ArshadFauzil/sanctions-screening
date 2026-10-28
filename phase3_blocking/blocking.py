"""Blocking: shortlist the SDN entries worth adjudicating for one customer name.

    from blocking import Blocker
    with psycopg.connect(url) as conn:
        blocker = Blocker(conn)
        candidates = blocker.block("Hassen Mofakhami Shahrestani", "individual")

Two stages (see docs/blocking.md for the measured reasoning):

1. GENERATION — three cheap, index-backed SQL lookups over sdn_names, unioned.
   Tuned for recall: anything plausibly the same party gets in.
     trigram   pg_trgm similarity >= threshold   typos, transliteration, word order
     phonetic  metaphone keys overlap            spellings that sound alike
     acronym   exact initials match (entities)   "SCD" — nothing else can find it;
                                                 3+ characters only (MIN_ACRONYM_LENGTH)

2. RESCORING — precise similarity in Python, on every name (primary + all
   aliases) of every generated entry. One score per entry = its best name.
     individuals  symmetric Monge-Elkan over tokens (per-token Jaro-Winkler)
     entities     IDF-weighted, typo-tolerant token matching on legal-form-free stems

Blocking uses NAMES ONLY. Year of birth, location and place of registration are
the adjudicator's criteria; filtering on them here would make a customer with a
missing year of birth (a possible_match) impossible to shortlist.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import jellyfish
from rapidfuzz.distance import JaroWinkler

from common.names import normalize, strip_legal_forms, strip_legal_suffix
from common.reference_tables import MIN_ACRONYM_LENGTH

# Which index rows a query may be compared against. An entity query also
# searches vessels and aircraft: a company named after a listed ship is a real
# collision the adjudicator has to see (and clear on entry type).
SEARCH_TYPES = {
    "individual": ["individual"],
    "entity": ["entity", "vessel", "aircraft"],
}

ACRONYM_SCORE = 0.90      # an exact acronym hit has no similarity score of its own
# Ambiguous acronyms return the whole tie group (domain-owner decision): the
# name cannot discriminate, so the adjudicator narrows it on place of
# registration — and the outcome for an ambiguous name is possible_match.
# NOTE: the SQL LIMIT counts index ROWS, not entries (an entry can carry the same
# acronym on several aliases). With 3+ character acronyms the largest group on
# the 23 Sep 2026 list is FYY7, 62 rows / 62 entries, so 100 never truncates;
# tests/phase3_blocking/test_index_integrity.py fails if a reload changes that.
# (Before the 3-character minimum, "AS" had 124 rows and lost 21 of 100 entries.)
ACRONYM_LIMIT = 100
TOKEN_MATCH_FLOOR = 0.85  # entity tokens below this JW count as unmatched (noise)


@dataclass
class Candidate:
    uid: int
    score: float
    entry_type: str
    matched_name: str
    name_kind: str
    is_weak: bool
    generators: list[str] = field(default_factory=list)
    ambiguous: bool = False   # one of several entities sharing the query's acronym

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


# --------------------------------------------------------------------------
# Token similarity and the two rescoring functions
# --------------------------------------------------------------------------

def token_sim(a: str, b: str) -> float:
    """Jaro-Winkler between two tokens, with one special case: initials.

    'A' (from "A. Mohammed ...") vs 'AYMAN' is a perfectly consistent initial,
    but Jaro-Winkler scores it ~0.76 because the strings differ in length.
    """
    if a == b:
        return 1.0
    if len(a) == 1 or len(b) == 1:
        return 0.9 if a[0] == b[0] else 0.0
    return JaroWinkler.similarity(a, b)


def _monge_elkan(xs: list[str], ys: list[str]) -> float:
    """Average, over tokens in xs, of each token's best match in ys."""
    if not xs or not ys:
        return 0.0
    return sum(max(token_sim(x, y) for y in ys) for x in xs) / len(xs)


def individual_score(query_tokens: list[str], name_tokens: list[str]) -> float:
    """Symmetric Monge-Elkan.

    Whole-string Jaro-Winkler rates 'Hassan Mofakhami Shahrestani' (1 letter off
    the query 'Hassen ...') at 0.912, BELOW an alias with three differences
    (0.925): one early edit shifts every later character match. Scoring token by
    token gives 0.978 vs 0.946 — the right order — and ignores word order.
    Averaging both directions penalises tokens present on only one side.
    """
    return (_monge_elkan(query_tokens, name_tokens) + _monge_elkan(name_tokens, query_tokens)) / 2


class IDF:
    """How identifying an entity token is: ln((N+1)/(df+1)) + 1 over N entity stems.

    On this list COMPANY appears in ~1,900 names (idf 2.6) and PROPITIOUS in 1
    (9.5). Unseen tokens get the maximum: a word nobody on the list uses is,
    if anything, maximally distinctive.
    """

    def __init__(self, stems: list[str]):
        self.n = len(stems)
        self.df = Counter()
        for s in stems:
            self.df.update(set(s.split()))
        self.max = math.log(self.n + 1) + 1

    def __call__(self, tok: str) -> float:
        return math.log((self.n + 1) / (self.df.get(tok, 0) + 1)) + 1


def _weighted_me(xs: list[str], ys: list[str], idf: IDF) -> float:
    if not xs or not ys:
        return 0.0
    num = den = 0.0
    for x in xs:
        w = idf(x)
        best = max(token_sim(x, y) for y in ys)
        num += w * (best if best >= TOKEN_MATCH_FLOOR else 0.0)
        den += w
    return num / den


def entity_score(query_tokens: list[str], name_tokens: list[str], idf: IDF) -> float:
    """IDF-weighted, typo-tolerant token matching, both directions averaged.

    Each token counts in proportion to how rare it is on the list, so sharing
    TRADING and COMPANY is worth little and sharing PROPITIOUS is worth a lot.
    Tokens match on Jaro-Winkler >= 0.85 rather than exact equality, so a typo
    in the distinctive token ("APLHA") does not destroy the score.
    """
    return (_weighted_me(query_tokens, name_tokens, idf) + _weighted_me(name_tokens, query_tokens, idf)) / 2


# --------------------------------------------------------------------------
# Generation SQL
# --------------------------------------------------------------------------

GENERATE_SQL = """
(SELECT uid, 'trigram' AS gen FROM sdn_names
  WHERE name_norm %% %(q)s AND entry_type = ANY(%(types)s)
  ORDER BY similarity(name_norm, %(q)s) DESC LIMIT %(lim)s)
UNION ALL
(SELECT uid, 'phonetic' FROM sdn_names
  WHERE phonetic_keys && %(keys)s::text[] AND entry_type = ANY(%(types)s)
  ORDER BY similarity(name_norm, %(q)s) DESC LIMIT %(lim)s)
UNION ALL
(SELECT uid, 'acronym' FROM sdn_names
  WHERE entry_type = ANY(%(types)s)
    AND (acronyms && %(qacr)s::text[] OR name_norm = ANY(%(qinit)s::text[]))
  ORDER BY uid LIMIT %(alim)s)
"""

NAMES_SQL = """
SELECT uid, entry_type, name_kind, is_weak, display, name_norm, stem
FROM sdn_names WHERE uid = ANY(%(uids)s) AND entry_type = ANY(%(types)s)
"""


class Blocker:
    def __init__(self, conn, k: int = 10, trigram_threshold: float = 0.3, generator_limit: int = 50,
                 acronym_limit: int = ACRONYM_LIMIT):
        self.conn = conn
        self.k = k
        self.limit = generator_limit
        self.acronym_limit = acronym_limit
        with conn.cursor() as cur:
            # The "%" operator compares against this session setting.
            cur.execute(f"SET pg_trgm.similarity_threshold = {float(trigram_threshold)}")
            cur.execute("SELECT stem FROM sdn_names WHERE entry_type = 'entity' AND name_kind = 'primary'")
            self.idf = IDF([r[0] for r in cur.fetchall() if r[0]])

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _query_acronym_inputs(norm: str, subject_type: str) -> tuple[list[str], list[str]]:
        """(acronym the query might BE, acronyms the query might EXPAND to)."""
        if subject_type != "entity":
            return [], []
        toks = norm.split()
        # Any single-token query of MIN_ACRONYM_LENGTH (3)+ characters may be an
        # acronym; 2-letter ones (AS, TS) are too ambiguous to be worth a lookup.
        # No upper cap: an early version capped this at 8 and missed a unique
        # 13-letter acronym.
        as_acronym = [norm] if len(toks) == 1 and len(norm) >= MIN_ACRONYM_LENGTH else []
        expansions = {"".join(t[0] for t in s.split()) for s in
                      (norm, strip_legal_suffix(norm), strip_legal_forms(norm)) if len(s.split()) >= 2}
        return as_acronym, sorted(e for e in expansions if len(e) >= MIN_ACRONYM_LENGTH)

    def _score(self, q_tokens: list[str], subject_type: str, row: dict) -> float:
        if subject_type == "individual":
            return individual_score(q_tokens, row["name_norm"].split())
        tokens = (row["stem"] or row["name_norm"]).split()
        return entity_score(q_tokens, tokens, self.idf)

    # -- the public call ------------------------------------------------------
    def block(self, name: str, subject_type: str) -> list[Candidate]:
        norm = normalize(name)
        if not norm:
            return []
        types = SEARCH_TYPES[subject_type]
        keys = [k for k in {jellyfish.metaphone(t) for t in norm.split() if len(t) >= 3 and t.isalpha()} if k]
        qacr, qinit = self._query_acronym_inputs(norm, subject_type)

        gens: dict[int, set[str]] = defaultdict(set)
        with self.conn.cursor() as cur:
            cur.execute(GENERATE_SQL, {"q": norm, "types": types, "keys": keys,
                                       "qacr": qacr, "qinit": qinit, "lim": self.limit,
                                       "alim": self.acronym_limit})
            for uid, gen in cur.fetchall():
                gens[uid].add(gen)
            if not gens:
                return []
            # Rescore EVERY name of each generated entry, not just the row that
            # matched: a generator may find an alias while the primary name
            # scores higher.
            cur.execute(NAMES_SQL, {"uids": list(gens), "types": types})
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        q_tokens = norm.split() if subject_type == "individual" else strip_legal_forms(norm).split()
        best: dict[int, Candidate] = {}
        for row in rows:
            s = self._score(q_tokens, subject_type, row)
            if "acronym" in gens[row["uid"]]:
                s = max(s, ACRONYM_SCORE)
            cur_best = best.get(row["uid"])
            if cur_best is None or s > cur_best.score:
                best[row["uid"]] = Candidate(
                    uid=row["uid"], score=s, entry_type=row["entry_type"],
                    matched_name=row["display"], name_kind=row["name_kind"],
                    is_weak=row["is_weak"], generators=sorted(gens[row["uid"]]))
        ranked = sorted(best.values(), key=lambda c: (-c.score, c.uid))
        shortlist = ranked[: self.k]
        # Several entities share the query's acronym: the name alone cannot say
        # which, and cutting at k would decide by tie order (i.e. by luck).
        # Return the whole tie group, flagged, beyond k if necessary.
        acronym_hits = [c for c in ranked if "acronym" in c.generators]
        if len(acronym_hits) > 1:
            for c in acronym_hits:
                c.ambiguous = True
            shortlist += [c for c in acronym_hits if c not in shortlist]
        return shortlist

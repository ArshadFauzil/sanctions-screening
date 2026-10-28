"""Exhaustive sweeps over the real list. Marked `slow` (≈4 min at default sizes);
run with   uv run pytest -m slow

Sizes are controlled by environment variables:
    SSA_SAMPLE=150        entries per subject type for the perturbation sweeps
    SSA_EXHAUSTIVE=1      every name / every acronym instead of a sample (≈45 min)
    SSA_UPDATE_BASELINES=1  rewrite baselines/perturbation_pairs.json from this run

Pattern used here: a RATCHET. Each sweep runs thousands of queries in one test
and compares the misses with a pinned list of KNOWN misses:
  * a miss that is not in the list  -> FAIL (a regression)
  * a listed miss that now succeeds -> FAIL (an improvement: delete it from the
    list so it can never silently come back)
So the known-miss lists can only shrink, and each one is a documented gap.

Fresh entries: the sweeps sample entries that are NOT in benchmark/queries.jsonl,
so these results are independent of the dev/test numbers in blocking.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from common.names import normalize
from common.paths import BENCHMARK_DIR
from common.reference_tables import TRANSLITERATION_GROUPS
from phase2_benchmark import perturbations as P

pytestmark = pytest.mark.slow

SAMPLE = int(os.environ.get("SSA_SAMPLE", "150"))
EXHAUSTIVE = os.environ.get("SSA_EXHAUSTIVE") == "1"
UPDATE = os.environ.get("SSA_UPDATE_BASELINES") == "1"
BASELINE_FILE = Path(__file__).parent / "baselines" / "perturbation_pairs.json"

# Attribute-only perturbations leave the name unchanged; blocking uses names only.
NAME_ONLY_SKIP = {"dob_degradation", "attribute_degradation"}


def _h(x) -> bytes:
    return hashlib.blake2b(str(x).encode(), digest_size=8).digest()


def _st(entry_type: str) -> str:
    return "individual" if entry_type == "individual" else "entity"


def ratchet(name: str, misses: set, known: set):
    new, fixed = sorted(misses - known), sorted(known - misses)
    msg = []
    if new:
        msg.append(f"{name}: {len(new)} NEW misses (regression): {new[:15]}")
    if fixed:
        msg.append(f"{name}: {len(fixed)} known misses now FOUND — delete them from the known list: {fixed[:15]}")
    assert not msg, "\n".join(msg)


@pytest.fixture(scope="module")
def fresh_entries(conn):
    bench_path = BENCHMARK_DIR / "queries.jsonl"
    in_bench = set()
    if bench_path.exists():
        in_bench = {json.loads(line)["source_uid"] for line in bench_path.open(encoding="utf-8") if line.strip()}
    with conn.cursor() as cur:
        cur.execute("SELECT uid, entry_type, primary_name, first_name, last_name FROM sdn_entries "
                    "WHERE entry_type IN ('individual', 'entity') ORDER BY uid")
        rows = [r for r in cur.fetchall() if r[0] not in in_bench]
    rows.sort(key=lambda r: _h(r[0]))
    n = None if EXHAUSTIVE else SAMPLE
    ind = [r for r in rows if r[1] == "individual"][:n]
    ent = [r for r in rows if r[1] == "entity"][:n]
    return ind + ent


def ctx_of(r):
    return P.EntryContext(uid=r[0], entry_type=r[1], primary_name=r[2], first_name=r[3], last_name=r[4])


# ===========================================================================
# 1. every perturbation, alone, on fresh entries — per-query ratchet
# ===========================================================================

KNOWN_SINGLE_MISSES = {
    # (uid, perturbation): why
    (4257, "transliteration"),     # ABDALLAH -> 'ABD AL': a two-token sibling; AL matches thousands
    (18557, "token_drop"),         # 'Song Choe': common Korean name pair, true entry outside top 10
}


def test_every_single_perturbation(blocker, fresh_entries):
    misses, total = set(), 0
    for r in fresh_entries:
        ctx = ctx_of(r)
        for pname, fn in P.perturbations_for(r[1]).items():
            if pname in NAME_ONLY_SKIP:
                continue
            res = fn(ctx, P.seed_for(r[0], pname))
            if res is None or not res.name.strip():
                continue
            total += 1
            if not any(c.uid == r[0] for c in blocker.block(res.name, _st(r[1]))):
                misses.add((r[0], pname))
    print(f"\nsingle perturbations: {total - len(misses)}/{total} shortlisted")
    ratchet("single perturbations", misses, KNOWN_SINGLE_MISSES if not EXHAUSTIVE and SAMPLE == 150 else misses)


# ===========================================================================
# 2. every ORDERED PAIR of perturbations — aggregate ratchet per pair
# ===========================================================================

def _pair_results(blocker, fresh_entries):
    stats = defaultdict(lambda: [0, 0])
    for r in fresh_entries:
        ctx = ctx_of(r)
        reg = {k: v for k, v in P.perturbations_for(r[1]).items() if k not in NAME_ONLY_SKIP}
        for p1 in reg:
            first = reg[p1](ctx, P.seed_for(r[0], p1))
            if first is None:
                continue
            for p2 in reg:
                if p2 == p1:
                    continue
                # Apply p2 to p1's output. Individuals: the whole output as last_name,
                # so natural_order() returns it unchanged (title-cased).
                ctx2 = P.EntryContext(uid=r[0], entry_type=r[1], primary_name=first.name,
                                      first_name=None, last_name=first.name if r[1] == "individual" else None)
                second = reg[p2](ctx2, P.seed_for(r[0], f"{p1}+{p2}"))
                if second is None or not second.name.strip():
                    continue
                key = f"{_st(r[1])}:{p1}+{p2}"
                stats[key][1] += 1
                stats[key][0] += any(c.uid == r[0] for c in blocker.block(second.name, _st(r[1])))
    return {k: {"found": f, "n": n, "recall": round(f / n, 4)} for k, (f, n) in sorted(stats.items())}


def test_every_perturbation_pair(blocker, fresh_entries):
    """Two perturbations compounded (e.g. typo THEN token drop). Some pairs are
    beyond what a name-only blocker can do (an acronym with a typo in it), so
    the assertion is a ratchet on each pair's recall, not a fixed target."""
    got = _pair_results(blocker, fresh_entries)
    print("\nperturbation pairs below 1.0:")
    for k, v in got.items():
        if v["recall"] < 1:
            print(f"  {k:<48} {v['found']:>4}/{v['n']:<4} {v['recall']:.3f}")
    if UPDATE:
        BASELINE_FILE.write_text(json.dumps({"sample": SAMPLE, "pairs": got}, indent=2) + "\n")
        pytest.skip("baseline rewritten")
    base = json.loads(BASELINE_FILE.read_text())
    if EXHAUSTIVE or base["sample"] != SAMPLE:
        pytest.skip("baseline was recorded at a different sample size")
    worse = {k: (v["recall"], base["pairs"][k]["recall"]) for k, v in got.items()
             if k in base["pairs"] and v["recall"] < base["pairs"][k]["recall"]}
    better = {k: (v["recall"], base["pairs"][k]["recall"]) for k, v in got.items()
              if k in base["pairs"] and v["recall"] > base["pairs"][k]["recall"]}
    assert not worse, f"pair recall dropped (now, baseline): {worse}"
    assert not better, f"pair recall improved — rerun with SSA_UPDATE_BASELINES=1 to lock it in: {better}"
    assert set(got) == set(base["pairs"]), "pair set changed — a perturbation was added or removed"


# ===========================================================================
# 3. every transliteration sibling substituted into real listed names
# ===========================================================================

KNOWN_TRANSLIT_MISSES = {
    # (uid, from, to): why
    (6635, "MOHAMMED", "MUHAMMAD"),   # 'MUHAMMAD OMAR': two of the commonest tokens on the list
    (46233, "MEHMET", "MOHAMMAD"),    # MEHMET->MOHAMMED family: no shared key, low trigram, 2-token name
    (46233, "MEHMET", "MOHAMED"),
    (46233, "MEHMET", "MOHAMMED"),
    (46233, "MEHMET", "MUHAMMAD"),
    (52601, "ABDALLA", "ABD AL"),     # two-token sibling
    (4257, "ABDALLAH", "ABD AL"),
    (15095, "YOUSEF", "JOSEPH"),      # single-token name; JOSEPH shares no key or trigram with YOUSEF
    (25544, "IBN", "BEN"),            # 'EP BEN': a two-token name made of particles
    (25544, "IBN", "BIN"),
}


def test_every_transliteration_sibling_in_real_names(conn, blocker):
    """For every ordered pair (a -> b) in every TRANSLITERATION_GROUP, take up to
    3 listed primary names containing token a (or all, with SSA_EXHAUSTIVE=1),
    replace a with b, and require the entry to be shortlisted."""
    per_token = None if EXHAUSTIVE else 3
    misses, total = set(), 0
    with conn.cursor() as cur:
        for g in TRANSLITERATION_GROUPS:
            words = [w for w in g if w.replace(" ", "").isalpha()]
            for a in words:
                cur.execute("SELECT uid, entry_type, name_norm FROM sdn_names WHERE name_kind = 'primary' "
                            "AND (' ' || name_norm || ' ') LIKE %s ORDER BY uid LIMIT %s",
                            (f"% {a} %", per_token or 1_000_000))
                for uid, et, norm in cur.fetchall():
                    for b in words:
                        if b == a:
                            continue
                        total += 1
                        q = f" {norm} ".replace(f" {a} ", f" {b} ", 1).strip()
                        if not any(c.uid == uid for c in blocker.block(q, _st(et))):
                            misses.add((uid, a, b))
    print(f"\ntransliteration substitutions: {total - len(misses)}/{total} shortlisted")
    ratchet("transliteration", misses, KNOWN_TRANSLIT_MISSES if not EXHAUSTIVE else misses)


# ===========================================================================
# 4. every name finds itself
# ===========================================================================

def test_every_name_finds_its_own_entry_with_score_one(conn, blocker):
    """The most basic guarantee: query = an exact listed name (primary or alias)
    -> that entry is shortlisted, scored 1.0. 2,000 names by default; all 44,013
    with SSA_EXHAUSTIVE=1."""
    with conn.cursor() as cur:
        cur.execute("SELECT uid, entry_type, display FROM sdn_names")
        rows = cur.fetchall()
    rows.sort(key=lambda r: _h(f"{r[0]}|{r[2]}"))
    if not EXHAUSTIVE:
        rows = rows[:2_000]
    bad = []
    for uid, et, disp in rows:
        hit = [c for c in blocker.block(disp, _st(et)) if c.uid == uid]
        if not hit or hit[0].score < 1.0 - 1e-9:
            bad.append((uid, disp, hit[0].score if hit else None))
    assert not bad, f"{len(bad)} names do not retrieve their own entry at 1.0: {bad[:10]}"


# ===========================================================================
# 5. every acronym
# ===========================================================================

def _acronym_groups(conn):
    with conn.cursor() as cur:
        cur.execute("""SELECT a, array_agg(DISTINCT uid) FROM
                         (SELECT unnest(acronyms) a, uid FROM sdn_names WHERE entry_type <> 'individual') t
                       GROUP BY a""")
        return cur.fetchall()


KNOWN_ACRONYM_GROUP_FAILURES: set[str] = set()   # AS (100-row limit) left with the 3-character minimum


def test_every_shared_acronym_returns_its_whole_group_flagged(conn, blocker):
    """All 2,325 acronyms (3+ characters) shared by 2+ entries: every entry in
    the group is returned and flagged ambiguous."""
    bad = set()
    for acr, uids in _acronym_groups(conn):
        if len(uids) < 2:
            continue
        out = blocker.block(acr, "entity")
        flagged = {c.uid for c in out if c.ambiguous}
        if not set(uids) <= flagged:
            bad.add(acr)
    ratchet("shared acronyms", bad, KNOWN_ACRONYM_GROUP_FAILURES)


def test_every_unique_acronym_finds_its_entry(conn, blocker):
    """Acronyms owned by one entry: that entry is shortlisted at >= 0.90 and not
    flagged. 1,000 by default; all ~16,000 with SSA_EXHAUSTIVE=1."""
    groups = sorted((a, u[0]) for a, u in _acronym_groups(conn) if len(u) == 1)
    groups.sort(key=lambda g: _h(g[0]))
    if not EXHAUSTIVE:
        groups = groups[:1_000]
    bad = []
    for acr, uid in groups:
        hit = [c for c in blocker.block(acr, "entity") if c.uid == uid]
        if not hit or hit[0].score < 0.9 or hit[0].ambiguous:
            bad.append((acr, uid))
    assert not bad, f"{len(bad)} unique acronyms not found cleanly: {bad[:10]}"


# ===========================================================================
# 6. nicknames — the known limitation deferred to Phase 4
# ===========================================================================

NICKNAMES = {"JOSE": "Pepe", "FRANCISCO": "Paco", "JOHN": "Jack", "WILLIAM": "Bill",
             "ALEKSANDR": "Sasha", "ALEXANDER": "Sasha", "ROBERT": "Bob", "MIKHAIL": "Misha",
             "DMITRY": "Dima"}


@pytest.fixture(scope="module")
def nickname_cases(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT uid, first_name, last_name FROM sdn_entries "
                    "WHERE entry_type = 'individual' AND first_name IS NOT NULL ORDER BY uid")
        out = []
        for uid, first, last in cur.fetchall():
            ft, lt = normalize(first).split(), normalize(last or "").split()
            if ft and lt and ft[0] in NICKNAMES:
                out.append((uid, NICKNAMES[ft[0]], ft, last, lt))
    return out


def _recall(blocker, cases, make_query):
    found = [any(c.uid == uid for c in blocker.block(make_query(*rest), "individual")) for uid, *rest in cases]
    return sum(found) / len(found), len(found)


def test_nickname_plus_rest_of_name(blocker, nickname_cases):
    """Middle names / second surnames carry the entry: measured 376/376."""
    r, n = _recall(blocker, nickname_cases, lambda nick, ft, last, lt: " ".join([nick, *ft[1:], last]))
    assert r == 1.0, f"{r:.3f} of {n}"


def test_full_given_name_plus_one_surname(blocker, nickname_cases):
    """No nickname, just a short form of the record: measured 374/376."""
    r, n = _recall(blocker, nickname_cases, lambda nick, ft, last, lt: f"{ft[0]} {lt[0]}")
    assert r >= 0.99, f"{r:.3f} of {n}"


@pytest.mark.xfail(strict=True, reason="Known limitation (blocking.md): nickname + one surname, "
                   "measured 354/376 = 0.94. Phase 4 adds a nickname table to blocking.")
def test_nickname_plus_one_surname(blocker, nickname_cases):
    r, n = _recall(blocker, nickname_cases, lambda nick, ft, last, lt: f"{nick} {lt[0]}")
    assert r >= 0.98, f"{r:.3f} of {n}"


@pytest.mark.xfail(strict=True, reason="Documented misses in blocking.md 'Known limitation: nicknames'")
@pytest.mark.parametrize("query, uid", [
    ("Pepe Lopez", 24705), ("Pepe Rodriguez", 49765), ("Paco Gonzalez", 58038), ("Paco Flores", 46492),
])
def test_documented_nickname_misses(blocker, query, uid):
    assert any(c.uid == uid for c in blocker.block(query, "individual"))

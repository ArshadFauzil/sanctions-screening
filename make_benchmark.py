#!/usr/bin/env python3
"""Build the labelled benchmark (plan section 2). No hand annotation anywhere.

    uv add faker
    python make_benchmark.py --individuals 120 --entities 80 --negatives 3000
    python make_benchmark.py --show-features        # diagnostic: what did Phase 1 load?

Outputs benchmark/queries.jsonl and benchmark/summary.json.

Why the population is skewed
----------------------------
A real sanctions hit is on the order of 1 in 10,000 customers. Measuring
precision on 200 positives against 100 negatives describes a world that does
not exist. 200 positives among 3,000 negatives, with blocking filtering the
population before any LLM sees it, makes false-positives-per-true-hit a
*measured* number rather than an extrapolated one — and keeps the LLM call
count inside a free tier.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from names import distinctive_tokens, find_legal_suffix, normalize, strip_legal_suffix
from perturbations import EntryContext, apply_all
from reference_tables import GENERIC_CORPORATE_TOKENS, SUBSTITUTABLE_SUFFIXES

SEED = 42
INDIVIDUAL_ATTRS = ("dob", "nationality", "place_of_birth", "citizenship")

# Deterministic sampling: md5 of the uid gives a stable pseudo-random order that
# does not depend on physical row order and, unlike ORDER BY random(), returns
# the same sample on every run.
SAMPLE_SQL = """
SELECT e.uid, e.entry_type, e.primary_name, e.first_name, e.last_name,
       (SELECT a.city    FROM sdn_addresses a WHERE a.uid = e.uid AND a.city    IS NOT NULL LIMIT 1) AS city,
       (SELECT a.country FROM sdn_addresses a WHERE a.uid = e.uid AND a.country IS NOT NULL LIMIT 1) AS country
FROM sdn_entries e
WHERE e.entry_type = %(entry_type)s
  AND ( EXISTS (SELECT 1 FROM sdn_features  f WHERE f.uid = e.uid AND f.feature_type = ANY(%(attrs)s))
     OR EXISTS (SELECT 1 FROM sdn_addresses a WHERE a.uid = e.uid AND a.country IS NOT NULL) )
ORDER BY md5(e.uid::text)
LIMIT %(limit)s
"""

FEATURES_SQL = """
SELECT feature_type, value_text, value_date, value_country
FROM sdn_features WHERE uid = %s
"""


def fetch_sample(cur, entry_type: str, limit: int, attrs: tuple[str, ...]) -> list[dict]:
    cur.execute(SAMPLE_SQL, {"entry_type": entry_type, "attrs": list(attrs), "limit": limit})
    return cur.fetchall()


def fetch_attributes(cur, uid: int) -> dict:
    """Collapse the feature rows into the attribute dict a customer record would carry."""
    cur.execute(FEATURES_SQL, (uid,))
    attrs: dict[str, str] = {}
    for row in cur.fetchall():
        ft, val = row["feature_type"], row["value_text"]
        if val and ft not in attrs:
            attrs[ft] = val
    return attrs


def build_idf(cur) -> dict[str, int]:
    """Document frequency of every token across entity names.

    Needed here to build generic-token hard negatives, and again in section 3
    for IDF-weighted blocking — so compute once and persist. 'PETROLEUM' is
    generic on an energy-heavy list and distinctive elsewhere; only the real
    list can tell you which.
    """
    cur.execute("SELECT primary_name FROM sdn_entries WHERE entry_type = 'entity'")
    df: Counter[str] = Counter()
    for row in cur.fetchall():
        df.update(set(strip_legal_suffix(row["primary_name"]).split()))
    return dict(df)


# ---------------------------------------------------------------------------
# Positives
# ---------------------------------------------------------------------------

NAME_PRESERVING = {"dob_degradation", "attribute_degradation"}


def make_positives(cur, rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        ctx = EntryContext(
            uid=row["uid"], entry_type=row["entry_type"], primary_name=row["primary_name"],
            first_name=row["first_name"], last_name=row["last_name"],
            city=row["city"], country=row["country"],
        )
        true_attrs = fetch_attributes(cur, ctx.uid)
        if ctx.country and "jurisdiction" not in true_attrs:
            true_attrs["jurisdiction"] = ctx.country
        for pname, result in apply_all(ctx):
            attrs = {k: v for k, v in true_attrs.items() if k not in result.drop_attributes}
            if pname == "dob_degradation" and result.note == "year_only" and "dob" in attrs:
                attrs["dob"] = attrs["dob"][-4:] if len(attrs["dob"]) >= 4 else attrs["dob"]
            out.append({
                "query_id": f"P-{ctx.uid}-{pname}",
                "subject_type": ctx.entry_type,
                "query_name": result.name,
                "query_attributes": attrs,
                "is_true_match": True,
                "source_uid": ctx.uid,
                "perturbation_type": pname,
                "perturbation_note": result.note,
            })
    return out


# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------

def make_negatives(cur, fake, rng: random.Random, n: int, idf: dict[str, int]) -> list[dict]:
    """Hard negatives are the false positives this project exists to kill.

    Four kinds, each testing a different failure mode. Easy negatives (no
    collision at all) stop blocking precision from being flattered.
    """
    cur.execute("""
        SELECT uid, entry_type, primary_name, last_name,
               (SELECT a.country FROM sdn_addresses a WHERE a.uid = e.uid AND a.country IS NOT NULL LIMIT 1) AS country
        FROM sdn_entries e WHERE entry_type IN ('individual','entity','vessel')
    """)
    listed = cur.fetchall()
    individuals = [r for r in listed if r["entry_type"] == "individual" and r["last_name"]]
    entities = [r for r in listed if r["entry_type"] == "entity"]
    vessels = [r for r in listed if r["entry_type"] == "vessel"]

    n_hard = n // 2
    out: list[dict] = []

    def add(kind: str, subject_type: str, name: str, attrs: dict, collides: int | None):
        out.append({
            "query_id": f"N-{len(out):05d}",
            "subject_type": subject_type,
            "query_name": name,
            "query_attributes": attrs,
            "is_true_match": False,
            "source_uid": None,
            "negative_kind": kind,
            "collides_with_uid": collides,
        })

    for i in range(n_hard):
        kind = ("individual_surname", "entity_generic_tokens",
                "entity_same_distinctive", "entity_vessel_name")[i % 4]

        if kind == "individual_surname" and individuals:
            src = rng.choice(individuals)
            # Same surname, invented given name, DOB far away, different nationality.
            add(kind, "individual",
                f"{fake.first_name()} {src['last_name'].title()}",
                {"dob": fake.date_of_birth(minimum_age=20, maximum_age=55).isoformat(),
                 "nationality": "Canada"},
                src["uid"])

        elif kind == "entity_generic_tokens" and entities:
            src = rng.choice(entities)
            stem_tokens = strip_legal_suffix(src["primary_name"]).split()
            generic = [t for t in stem_tokens if t in GENERIC_CORPORATE_TOKENS]
            if len(generic) < 1:
                continue
            invented = fake.last_name().upper()
            suffix = find_legal_suffix(src["primary_name"]) or rng.choice(SUBSTITUTABLE_SUFFIXES)
            add(kind, "entity", " ".join([invented] + generic + [suffix]).title(),
                {"jurisdiction": "Canada", "registration_number": str(fake.random_number(7, True))},
                src["uid"])

        elif kind == "entity_same_distinctive" and entities:
            src = rng.choice(entities)
            if not distinctive_tokens(src["primary_name"]):
                continue
            add(kind, "entity", src["primary_name"].title(),
                {"jurisdiction": "Canada", "registration_number": str(fake.random_number(7, True))},
                src["uid"])

        elif kind == "entity_vessel_name" and vessels:
            src = rng.choice(vessels)
            add(kind, "entity", f"{src['primary_name'].title()} Shipping Ltd",
                {"jurisdiction": "Canada"}, src["uid"])

    while len(out) < n:
        if rng.random() < 0.6:
            add("easy_individual", "individual", fake.name(),
                {"dob": fake.date_of_birth(minimum_age=20, maximum_age=75).isoformat(),
                 "nationality": "Canada"}, None)
        else:
            add("easy_entity", "entity",
                f"{fake.last_name()} {rng.choice(('Consulting','Logistics','Dental','Roofing'))} Inc",
                {"jurisdiction": "Canada"}, None)
    return out


# ---------------------------------------------------------------------------
# Conformance — this replaces hand labelling, so it has to be strict
# ---------------------------------------------------------------------------

def check(cur, queries: list[dict]) -> dict:
    cur.execute("SELECT uid, primary_name FROM sdn_entries")
    listed_names = {normalize(r["primary_name"]) for r in cur.fetchall()}
    cur.execute("SELECT uid, alias_name FROM sdn_aliases")
    aliases_by_uid: dict[int, set[str]] = {}
    for r in cur.fetchall():
        aliases_by_uid.setdefault(r["uid"], set()).add(normalize(r["alias_name"]))
    cur.execute("SELECT uid, primary_name FROM sdn_entries")
    primary_by_uid = {r["uid"]: normalize(r["primary_name"]) for r in cur.fetchall()}

    problems = Counter()
    ids = set()
    for q in queries:
        if q["query_id"] in ids:
            problems["duplicate_query_id"] += 1
        ids.add(q["query_id"])
        norm = normalize(q["query_name"])

        if q["is_true_match"]:
            uid = q["source_uid"]
            # A positive whose name is unchanged is a trivially findable
            # positive that would inflate every recall figure — except for the
            # two perturbations that deliberately leave the name alone.
            if q["perturbation_type"] not in NAME_PRESERVING and norm == primary_by_uid.get(uid):
                problems["positive_name_unchanged"] += 1
            # Reproducing a known alias is not a bug, but it is much easier than
            # intended, so count it and decide whether to keep it.
            if norm in aliases_by_uid.get(uid, set()):
                problems["positive_equals_known_alias"] += 1
        else:
            # A negative that happens to equal a listed name is a mislabelled
            # true positive and would corrupt the precision figure.
            if norm in listed_names:
                problems["negative_matches_listed_name"] += 1
    return dict(problems)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--individuals", type=int, default=120)
    ap.add_argument("--entities", type=int, default=80)
    ap.add_argument("--negatives", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("benchmark"))
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--show-features", action="store_true",
                    help="print the feature_type distribution Phase 1 loaded, then exit")
    args = ap.parse_args()

    load_dotenv()
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        ap.error("--database-url not given and $DATABASE_URL is not set")

    rng = random.Random(SEED)
    from faker import Faker
    fake = Faker("en_CA")
    Faker.seed(SEED)

    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        if args.show_features:
            cur.execute("""
                SELECT e.entry_type, f.feature_type, count(*) AS n
                FROM sdn_features f JOIN sdn_entries e USING (uid)
                GROUP BY 1, 2 ORDER BY 1, 3 DESC
            """)
            for r in cur.fetchall():
                print(f"{r['entry_type']:<12} {r['feature_type']:<24} {r['n']}")
            return

        idf = build_idf(cur)
        inds = fetch_sample(cur, "individual", args.individuals, INDIVIDUAL_ATTRS)
        ents = fetch_sample(cur, "entity", args.entities, ("registration_number",))
        print(f"sampled {len(inds)} individuals, {len(ents)} entities")

        positives = make_positives(cur, inds + ents)
        negatives = make_negatives(cur, fake, rng, args.negatives, idf)
        queries = positives + negatives
        problems = check(cur, queries)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    by_pert = Counter(q["perturbation_type"] for q in positives)
    by_neg = Counter(q["negative_kind"] for q in negatives)
    summary = {
        "positives": len(positives), "negatives": len(negatives),
        "positives_by_perturbation": dict(by_pert), "negatives_by_kind": dict(by_neg),
        "idf_tokens": len(idf), "conformance_problems": problems, "seed": SEED,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if problems:
        print("\nCONFORMANCE PROBLEMS — investigate before trusting any metric.")

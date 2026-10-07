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
import re
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from names import distinctive_tokens, find_legal_suffix, normalize, strip_legal_suffix
from perturbations import EntryContext, apply_all, base_name
from reference_tables import GENERIC_CORPORATE_TOKENS, SUBSTITUTABLE_SUFFIXES
from split import assign_splits, split_report, verify as verify_split

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

# ---------------------------------------------------------------------------
# Match criteria (domain rule, set by the analyst who owns this project)
#
#   individual: name + city and country they are/were based in + year of birth
#   entity:     full name + place of registration
#
# A query carries EXACTLY these attributes and nothing else. Anything more is
# both too strict as a matching rule and a leakage risk: copying raw SDN feature
# rows put OFAC's own sanctions-programme text into "customer" records, which
# hands the adjudicator the answer.
# ---------------------------------------------------------------------------
ALLOWED_ATTRIBUTES = {
    "individual": frozenset({"year_of_birth", "city", "country"}),
    "entity": frozenset({"registration_country"}),
}

# Expected disposition of each query under the match rule. Three outcomes, set
# by the domain owner:
#   true_match      every criterion present and agreeing
#   possible_match  most likely the listed party, but something is ambiguous:
#                   a criterion missing, a country-only location, several listed
#                   DOBs, or a partial name. Escalated to an analyst.
#   false_positive  a criterion decisively disagrees
# A possible_match on a true entity counts as RETAINED in the eval; only a
# false_positive on a true entity is a missed hit.
OUTCOMES = ("true_match", "possible_match", "false_positive")

# Name perturbations that leave only a partial name, so the name criterion
# cannot be confirmed: an acronym is not the full entity name, and an initial
# is not the given name. (A dropped middle name is NOT here — customer records
# routinely omit middle names and first + last still match.)
PARTIAL_NAME_PERTURBATIONS = frozenset({"acronym", "initialize"})


def expected_outcome(entry_type: str, attrs: dict, n_birth_years: int, perturbation: str) -> str:
    """Disposition a correct adjudicator should reach for a TRUE entity (a positive)."""
    if perturbation in PARTIAL_NAME_PERTURBATIONS:
        return "possible_match"
    if entry_type == "individual":
        complete = ("year_of_birth" in attrs and "country" in attrs and "city" in attrs
                    and n_birth_years <= 1)
    else:
        complete = "registration_country" in attrs
    return "true_match" if complete else "possible_match"


ADDRESSES_SQL = """
SELECT city, country FROM sdn_addresses WHERE uid = %s ORDER BY id
"""

# Real Canadian cities for synthetic negatives. Faker's city() invents towns
# ("Johnberg"), which an LLM adjudicator may treat as noise or as suspicious.
CANADIAN_CITIES = (
    "Toronto", "Mississauga", "Brampton", "Hamilton", "Ottawa", "Markham", "Vaughan",
    "Montreal", "Laval", "Quebec City", "Gatineau", "Vancouver", "Surrey", "Burnaby",
    "Richmond", "Calgary", "Edmonton", "Winnipeg", "Regina", "Saskatoon", "Halifax",
    "London", "Kitchener", "Windsor", "Oshawa", "Victoria", "St. John's", "Moncton",
)

_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")

FEATURES_SQL = """
SELECT feature_type, value_text, value_date, value_country
FROM sdn_features WHERE uid = %s
"""


def fetch_sample(cur, entry_type: str, limit: int, attrs: tuple[str, ...]) -> list[dict]:
    cur.execute(SAMPLE_SQL, {"entry_type": entry_type, "attrs": list(attrs), "limit": limit})
    return cur.fetchall()


def customer_attributes(cur, entry_type: str, uid: int) -> dict:
    """The attributes a bank's customer record would hold for this person or firm.

    Individuals: year of birth (first year found in any listed DOB — OFAC DOBs
    are often "circa 1951" or "1946 to 1948", and only the year is needed), plus
    city and country from the first listed address that has a country. If the
    entry has no address, place of birth stands in: it is a place the person
    was based in, which is what the rule asks for.

    Entities: place of registration. The best source is the issuing country on
    a registration-type identifier (company number, registration ID, tax ID);
    failing that, the country of the first listed address.
    """
    cur.execute(FEATURES_SQL, (uid,))
    feats = cur.fetchall()
    attrs: dict = {}
    if entry_type == "individual":
        for f in feats:
            if f["feature_type"] == "dob" and f["value_text"]:
                m = _YEAR.search(f["value_text"])
                if m:
                    attrs["year_of_birth"] = int(m.group(1))
                    break
        cur.execute(ADDRESSES_SQL, (uid,))
        addrs = cur.fetchall()
        # Prefer an address with BOTH city and country: OFAC often lists a
        # country-only address first, and taking it would discard a city the
        # entry actually has.
        addr = (next((a for a in addrs if a["country"] and a["city"]), None)
                or next((a for a in addrs if a["country"]), None))
        if addr:
            if addr["city"]:
                attrs["city"] = addr["city"]
            attrs["country"] = addr["country"]
        else:
            pob = next((f["value_text"] for f in feats
                        if f["feature_type"] == "place_of_birth" and f["value_text"]), None)
            if pob:
                parts = [x.strip() for x in pob.split(",") if x.strip()]
                if len(parts) >= 2:
                    attrs["city"], attrs["country"] = parts[0], parts[-1]
                elif parts:
                    attrs["country"] = parts[0]
    else:
        reg = next((f["value_country"] for f in feats if f.get("value_country")), None)
        if not reg:
            cur.execute(ADDRESSES_SQL, (uid,))
            reg = next((a["country"] for a in cur.fetchall() if a["country"]), None)
        if reg:
            attrs["registration_country"] = reg
    return attrs


def distinct_birth_years(cur, uid: int) -> int:
    """How many different birth years OFAC lists. More than one = undetermined."""
    cur.execute(FEATURES_SQL, (uid,))
    years = {int(m.group(1)) for f in cur.fetchall()
             if f["feature_type"] == "dob" and f["value_text"]
             for m in _YEAR.finditer(f["value_text"])}
    return len(years)


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
        true_attrs = customer_attributes(cur, ctx.entry_type, ctx.uid)
        n_years = distinct_birth_years(cur, ctx.uid) if ctx.entry_type == "individual" else 0
        for pname, result in apply_all(ctx):
            attrs = {k: v for k, v in true_attrs.items() if k not in result.drop_attributes}
            # Some perturbations are canonicalised away by normalize(): "&" vs
            # "and", accents, hyphen-as-space. Those queries are still realistic
            # input, but they are INVARIANT TESTS (blocking must score 1.00 on
            # them by construction) rather than hard variance. Flag them so the
            # section-3 breakdown can separate the two, and so the conformance
            # check does not mistake them for a broken generator.
            base = base_name(ctx)
            out.append({
                "query_id": f"P-{ctx.uid}-{pname}",
                "subject_type": ctx.entry_type,
                "query_name": result.name,
                "query_attributes": attrs,
                "is_true_match": True,
                "source_uid": ctx.uid,
                "perturbation_type": pname,
                "perturbation_note": result.note,
                "expected_outcome": expected_outcome(ctx.entry_type, attrs, n_years, pname),
                "raw_name_unchanged": result.name == base,
                "normalization_equivalent": (result.name != base
                                             and normalize(result.name) == normalize(base)),
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
            "expected_outcome": "false_positive",
            "collides_with_uid": collides,
        })

    for i in range(n_hard):
        kind = ("individual_surname", "entity_generic_tokens",
                "entity_same_distinctive", "entity_vessel_name")[i % 4]

        if kind == "individual_surname" and individuals:
            src = rng.choice(individuals)
            # Same surname, different given name, Canadian-based. Fails the name
            # criterion despite the surname collision a fuzzy matcher keys on.
            add(kind, "individual",
                f"{fake.first_name()} {src['last_name'].title()}",
                {"year_of_birth": fake.date_of_birth(minimum_age=20, maximum_age=55).year,
                 "city": rng.choice(CANADIAN_CITIES), "country": "Canada"},
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
                {"registration_country": "Canada"}, src["uid"])

        elif kind == "entity_same_distinctive" and entities:
            src = rng.choice(entities)
            # Identical full name, different place of registration: passes the name
            # criterion and fails the registration criterion, so NOT a match.
            # Skip the (rare) listed entity already based in Canada, which would
            # make this a genuine match and corrupt the label.
            if not distinctive_tokens(src["primary_name"]):
                continue
            # Check EVERY country on the source — registration IDs and all
            # addresses. Checking only the first address let EMPRESA CUBANA DE
            # AVIACION through: its first address is Guyana, but it also has a
            # Canadian one, so a Canadian-registered namesake would be a TRUE match.
            cur.execute(FEATURES_SQL, (src["uid"],))
            countries = {f["value_country"] for f in cur.fetchall() if f.get("value_country")}
            cur.execute(ADDRESSES_SQL, (src["uid"],))
            countries |= {a["country"] for a in cur.fetchall() if a["country"]}
            if any(c.strip().lower() == "canada" for c in countries):
                continue
            add(kind, "entity", src["primary_name"].title(),
                {"registration_country": "Canada"}, src["uid"])

        elif kind == "entity_vessel_name" and vessels:
            src = rng.choice(vessels)
            add(kind, "entity", f"{src['primary_name'].title()} Shipping Ltd",
                {"registration_country": "Canada"}, src["uid"])

    while len(out) < n:
        if rng.random() < 0.6:
            add("easy_individual", "individual", fake.name(),
                {"year_of_birth": fake.date_of_birth(minimum_age=20, maximum_age=75).year,
                 "city": rng.choice(CANADIAN_CITIES), "country": "Canada"}, None)
        else:
            add("easy_entity", "entity",
                f"{fake.last_name()} {rng.choice(('Consulting','Logistics','Dental','Roofing'))} Inc",
                {"registration_country": "Canada"}, None)
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

    problems: Counter[str] = Counter()
    informational: Counter[str] = Counter()
    ids = set()
    for q in queries:
        if q["query_id"] in ids:
            problems["duplicate_query_id"] += 1
        ids.add(q["query_id"])
        norm = normalize(q["query_name"])

        # Leakage guard: a query may carry only the match-criteria attributes.
        extra = set(q["query_attributes"]) - ALLOWED_ATTRIBUTES[q["subject_type"]]
        if extra:
            problems["disallowed_attribute"] += 1
        oc = q.get("expected_outcome")
        if oc not in OUTCOMES:
            problems["missing_or_invalid_expected_outcome"] += 1
        elif q["is_true_match"] and oc == "false_positive":
            problems["positive_labelled_false_positive"] += 1
        elif not q["is_true_match"] and oc != "false_positive":
            problems["negative_not_labelled_false_positive"] += 1
        if q["is_true_match"] and not q["query_attributes"]:
            informational["positive_without_any_attribute"] += 1

        if q["is_true_match"]:
            uid = q["source_uid"]
            # A positive whose name is unchanged is a trivially findable
            # positive that would inflate every recall figure — except for the
            # two perturbations that deliberately leave the name alone.
            # A positive whose RAW name is unchanged is a trivially findable
            # positive that would inflate every recall figure — a real bug,
            # unless the perturbation deliberately preserves the name and
            # degrades attributes instead.
            if q["perturbation_type"] not in NAME_PRESERVING and q.get("raw_name_unchanged"):
                problems["positive_raw_name_unchanged"] += 1
            # Changed the string but not its normalised form: informational, not
            # a bug. Counted so the number is visible rather than silent.
            if q.get("normalization_equivalent"):
                informational["positive_normalization_equivalent"] += 1
            # Reproducing a known alias is not a bug, but it is much easier than
            # intended, so count it and decide whether to keep it.
            if norm in aliases_by_uid.get(uid, set()):
                informational["positive_equals_known_alias"] += 1
        else:
            # A negative that happens to equal a listed name is normally a
            # mislabelled true positive and would corrupt the precision figure.
            # The exception is entity_same_distinctive, whose whole point is a
            # company with the SAME name as a listed entity but a different
            # place of registration — the hardest corporate false
            # positive there is, and deliberately constructed that way.
            if norm in listed_names and q.get("negative_kind") != "entity_same_distinctive":
                problems["negative_matches_listed_name"] += 1
    return {"problems": dict(problems), "informational": dict(informational)}


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

    # connect_timeout so an unreachable host errors in 10s instead of hanging,
    # which is indistinguishable from "the script produced no output".
    with psycopg.connect(url, connect_timeout=10) as conn, conn.cursor(row_factory=dict_row) as cur:
        if args.show_features:
            # A diagnostic must never be silent. Report the connection target and
            # the row counts BEFORE the distribution, so an empty database says so
            # instead of printing nothing and exiting 0.
            print(f"connected to db={conn.info.dbname} host={conn.info.host}:{conn.info.port} "
                  f"user={conn.info.user}")
            counts = {}
            for table in ("sdn_entries", "sdn_aliases", "sdn_features", "sdn_addresses"):
                cur.execute(f"SELECT count(*) AS n FROM {table}")
                counts[table] = cur.fetchone()["n"]
            print("\nrow counts (expected after a full sdn.xml load, 23 Sep 2026 publication):")
            expected = {"sdn_entries": 19391, "sdn_aliases": 24622,
                        "sdn_features": 78581, "sdn_addresses": 21951}
            for table, n in counts.items():
                flag = "OK" if n == expected[table] else f"expected ~{expected[table]}"
                print(f"  {table:<16} {n:>8}   {flag}")

            if counts["sdn_entries"] == 0:
                print("\nDATABASE IS EMPTY. The schema exists but nothing was loaded.")
                print("Run the loader first, then re-run this command:")
                print("    uv run load_sdn.py --xml sdn.xml")
                return
            if counts["sdn_features"] == 0:
                print("\nEntries loaded but NO FEATURES. The benchmark cannot sample without")
                print("year of birth / location / registration country, so re-run the loader.")
                return

            cur.execute("""
                SELECT e.entry_type, count(*) AS n
                FROM sdn_entries e GROUP BY 1 ORDER BY 2 DESC
            """)
            print("\nentry_type:")
            for r in cur.fetchall():
                print(f"  {r['entry_type']:<12} {r['n']}")

            cur.execute("""
                SELECT e.entry_type, f.feature_type, count(*) AS n
                FROM sdn_features f JOIN sdn_entries e USING (uid)
                GROUP BY 1, 2 ORDER BY 1, 3 DESC
            """)
            rows_out = cur.fetchall()
            print(f"\nfeature_type distribution ({len(rows_out)} entry_type/feature_type pairs):")
            for r in rows_out:
                print(f"  {r['entry_type']:<12} {r['feature_type']:<34} {r['n']}")
            return

        idf = build_idf(cur)
        inds = fetch_sample(cur, "individual", args.individuals, INDIVIDUAL_ATTRS)
        ents = fetch_sample(cur, "entity", args.entities, ("registration_number",))
        print(f"sampled {len(inds)} individuals, {len(ents)} entities")

        positives = make_positives(cur, inds + ents)
        negatives = make_negatives(cur, fake, rng, args.negatives, idf)
        queries = positives + negatives
        checked = check(cur, queries)
        problems, informational = checked["problems"], checked["informational"]

    # Assign dev/test before writing, so a regenerated benchmark is always split.
    assign_splits(queries)
    split_errors = verify_split(queries)
    if split_errors:
        raise SystemExit("split failed verification: " + "; ".join(split_errors))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    by_pert = Counter(q["perturbation_type"] for q in positives)
    by_neg = Counter(q["negative_kind"] for q in negatives)
    summary = {
        "positives": len(positives), "negatives": len(negatives),
        "positives_by_perturbation": dict(by_pert), "negatives_by_kind": dict(by_neg),
        "idf_tokens": len(idf), "conformance_problems": problems,
        "informational": informational, "seed": SEED,
        "splits": split_report(queries), "dev_fraction": 0.25,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if problems:
        print("\nCONFORMANCE PROBLEMS — investigate before trusting any metric.")


if __name__ == "__main__":
    main()

"""Load OFAC's SDN list (sdn.xml) into Postgres.

Schema: schema.sql (plan section 1.2 of Sanctions-Adjudicator-Plan.md).
Tables: sdn_entries, sdn_aliases, sdn_features, sdn_addresses.

Memory note (plan section 1, "Parsing on 8 GB"): sdn.xml is ~29 MB, but
etree.parse() would build a full DOM in memory, costing several hundred MB of
Python objects. Instead this uses lxml.etree.iterparse() and clears each
<sdnEntry> element (plus its now-processed preceding siblings) as soon as it
has been read, so memory stays flat regardless of file size — the same
loader would work unchanged against sdn_advanced.xml or cons_advanced.xml.

Row buffers are flushed to Postgres every --batch-size entries (default
1,000) rather than held for the whole file, and committed per batch so a
long run can be interrupted without losing prior progress.

Usage:
    uv run load_sdn.py --reset                 # fresh load, ~19,391 entries
    uv run load_sdn.py --limit 100 --reset      # smoke test first
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from lxml import etree

NS = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/XML"


def qn(tag: str) -> str:
    return f"{{{NS}}}{tag}"


ENTRY_TAG = qn("sdnEntry")

# OFAC formats a fully-specified DOB as "10 Dec 1948". Year-only ("1963"),
# "circa 1951" and range forms ("1946 to 31 Dec 1948") are common and are
# left unparsed (value_date NULL, raw text kept in value_text) — see the
# plan's "DOB degradation" perturbation, which treats this as normal data.
_DATE_FORMATS = ("%d %b %Y",)


def parse_ofac_date(raw: str | None) -> date | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def normalize_feature_type(id_type: str) -> str:
    """'SWIFT/BIC' -> 'swift_bic', 'Tax ID No.' -> 'tax_id_no'."""
    s = re.sub(r"[^a-z0-9]+", "_", id_type.strip().lower())
    return s.strip("_")


def text(elem: etree._Element, tag: str) -> str | None:
    val = elem.findtext(qn(tag))
    if val is None:
        return None
    val = val.strip()
    return val or None


def full_name(last: str | None, first: str | None) -> str | None:
    """OFAC's own display convention: 'LASTNAME, Firstname'. Entities/vessels/
    aircraft have no firstName and lastName already holds the full name."""
    if not last:
        return None
    return f"{last}, {first}" if first else last


def build_entry_row(entry: etree._Element) -> tuple:
    uid = int(text(entry, "uid"))
    entry_type = (text(entry, "sdnType") or "").lower()
    first_name = text(entry, "firstName")
    last_name = text(entry, "lastName")
    programs = [
        p.text.strip()
        for p in entry.findall(f"{qn('programList')}/{qn('program')}")
        if p.text and p.text.strip()
    ]
    return (
        uid,
        entry_type,
        full_name(last_name, first_name),
        first_name,
        last_name,
        text(entry, "title"),
        programs,
        text(entry, "remarks"),
    )


def build_alias_rows(uid: int, entry: etree._Element) -> list[tuple]:
    rows = []
    for aka in entry.findall(f"{qn('akaList')}/{qn('aka')}"):
        alias_name = full_name(text(aka, "lastName"), text(aka, "firstName"))
        if not alias_name:
            continue
        is_weak = (text(aka, "category") or "").lower() == "weak"
        rows.append((uid, alias_name, text(aka, "type"), is_weak))
    return rows


def build_feature_rows(uid: int, entry: etree._Element) -> list[tuple]:
    rows: list[tuple] = []

    for dob in entry.findall(f"{qn('dateOfBirthList')}/{qn('dateOfBirthItem')}"):
        raw = text(dob, "dateOfBirth")
        if raw:
            rows.append((uid, "dob", raw, parse_ofac_date(raw), None))

    for pob in entry.findall(f"{qn('placeOfBirthList')}/{qn('placeOfBirthItem')}"):
        raw = text(pob, "placeOfBirth")
        if raw:
            rows.append((uid, "place_of_birth", raw, None, None))

    for nat in entry.findall(f"{qn('nationalityList')}/{qn('nationality')}"):
        raw = text(nat, "country")
        if raw:
            rows.append((uid, "nationality", raw, None, None))

    for cit in entry.findall(f"{qn('citizenshipList')}/{qn('citizenship')}"):
        raw = text(cit, "country")
        if raw:
            rows.append((uid, "citizenship", raw, None, None))

    # idList mixes genuine identity documents (Passport, Registration Number,
    # SWIFT/BIC, Tax ID No., Website, Email Address, ...) with regulatory
    # citations (e.g. "Secondary sanctions risk:", EO directive information).
    # All are kept as raw features here; mapping the ones that matter for
    # adjudication (registration_number, jurisdiction, ...) into the coarse
    # attribute categories in plan section 4.3 is a Day-2 concern, not a
    # migration concern — this loader stays lossless.
    for id_elem in entry.findall(f"{qn('idList')}/{qn('id')}"):
        id_type = text(id_elem, "idType")
        if not id_type:
            continue
        rows.append(
            (
                uid,
                normalize_feature_type(id_type),
                text(id_elem, "idNumber"),
                parse_ofac_date(text(id_elem, "issueDate")),
                text(id_elem, "idCountry"),
            )
        )

    vessel_info = entry.find(qn("vesselInfo"))
    if vessel_info is not None:
        for child_tag, feature_type in (
            ("callSign", "vessel_call_sign"),
            ("vesselType", "vessel_type"),
            ("vesselFlag", "vessel_flag"),
            ("vesselOwner", "vessel_owner"),
            ("tonnage", "vessel_tonnage"),
            ("grossRegisteredTonnage", "vessel_gross_registered_tonnage"),
        ):
            val = text(vessel_info, child_tag)
            if val:
                rows.append((uid, feature_type, val, None, None))

    return rows


def build_address_rows(uid: int, entry: etree._Element) -> list[tuple]:
    rows = []
    for addr in entry.findall(f"{qn('addressList')}/{qn('address')}"):
        parts = [text(addr, t) for t in ("address1", "address2", "address3")]
        address = ", ".join(p for p in parts if p) or None
        city = text(addr, "city")
        state = text(addr, "stateOrProvince")
        postal_code = text(addr, "postalCode")
        country = text(addr, "country")
        if not any((address, city, state, postal_code, country)):
            continue
        rows.append((uid, address, city, state, postal_code, country))
    return rows


def iter_entries(xml_path: Path) -> Iterator[etree._Element]:
    """Stream <sdnEntry> elements with flat memory usage (plan section 1,
    'Parsing on 8 GB'): clear each element, and its now-dead preceding
    siblings, right after it is yielded and consumed."""
    context = etree.iterparse(str(xml_path), events=("end",), tag=ENTRY_TAG)
    for _, elem in context:
        yield elem
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


INSERT_ENTRIES = """
    INSERT INTO sdn_entries
        (uid, entry_type, primary_name, first_name, last_name, title, programs, remarks)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""
INSERT_ALIASES = """
    INSERT INTO sdn_aliases (uid, alias_name, alias_type, is_weak)
    VALUES (%s, %s, %s, %s)
"""
INSERT_FEATURES = """
    INSERT INTO sdn_features (uid, feature_type, value_text, value_date, value_country)
    VALUES (%s, %s, %s, %s, %s)
"""
INSERT_ADDRESSES = """
    INSERT INTO sdn_addresses (uid, address, city, state, postal_code, country)
    VALUES (%s, %s, %s, %s, %s, %s)
"""


def flush(cur: psycopg.Cursor, entries, aliases, features, addresses) -> None:
    if entries:
        cur.executemany(INSERT_ENTRIES, entries)
    if aliases:
        cur.executemany(INSERT_ALIASES, aliases)
    if features:
        cur.executemany(INSERT_FEATURES, features)
    if addresses:
        cur.executemany(INSERT_ADDRESSES, addresses)
    entries.clear()
    aliases.clear()
    features.clear()
    addresses.clear()


def load(
    conn: psycopg.Connection,
    xml_path: Path,
    batch_size: int,
    limit: int | None,
) -> dict[str, int]:
    counts = {"entries": 0, "aliases": 0, "features": 0, "addresses": 0}
    entry_buf: list[tuple] = []
    alias_buf: list[tuple] = []
    feature_buf: list[tuple] = []
    address_buf: list[tuple] = []

    with conn.cursor() as cur:
        for i, entry in enumerate(iter_entries(xml_path), start=1):
            if limit is not None and i > limit:
                break

            entry_row = build_entry_row(entry)
            uid = entry_row[0]
            new_aliases = build_alias_rows(uid, entry)
            new_features = build_feature_rows(uid, entry)
            new_addresses = build_address_rows(uid, entry)

            entry_buf.append(entry_row)
            alias_buf.extend(new_aliases)
            feature_buf.extend(new_features)
            address_buf.extend(new_addresses)

            counts["entries"] += 1
            counts["aliases"] += len(new_aliases)
            counts["features"] += len(new_features)
            counts["addresses"] += len(new_addresses)

            if i % batch_size == 0:
                flush(cur, entry_buf, alias_buf, feature_buf, address_buf)
                conn.commit()
                print(f"  ... {i:,} entries loaded", file=sys.stderr)

        flush(cur, entry_buf, alias_buf, feature_buf, address_buf)
        conn.commit()

    return counts


def reset_schema(conn: psycopg.Connection, schema_sql: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS sdn_addresses, sdn_features, sdn_aliases, sdn_entries CASCADE"
        )
        cur.execute(schema_sql.read_text())
    conn.commit()


def ensure_schema(conn: psycopg.Connection, schema_sql: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(schema_sql.read_text())
    conn.commit()


def main() -> None:
    load_dotenv()
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", type=Path, default=here / "sdn.xml")
    parser.add_argument("--schema-path", type=Path, default=here / "schema.sql")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="defaults to $DATABASE_URL (see .env)",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="load only the first N entries (smoke test)")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop and recreate sdn_* tables before loading (default: fail on duplicate uid if already loaded)",
    )
    args = parser.parse_args()

    if not args.database_url:
        parser.error("--database-url not given and $DATABASE_URL is not set")
    if not args.xml_path.exists():
        parser.error(f"XML file not found: {args.xml_path}")

    with psycopg.connect(args.database_url) as conn:
        if args.reset:
            print(f"Resetting schema from {args.schema_path.name} ...", file=sys.stderr)
            reset_schema(conn, args.schema_path)
        else:
            ensure_schema(conn, args.schema_path)

        print(f"Loading {args.xml_path.name} ...", file=sys.stderr)
        counts = load(conn, args.xml_path, args.batch_size, args.limit)

    print(
        "Done. "
        f"entries={counts['entries']:,} aliases={counts['aliases']:,} "
        f"features={counts['features']:,} addresses={counts['addresses']:,}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic dev/test split of the benchmark (adds a "split" field to every query).

    uv run split.py                      # rewrites benchmark/queries.jsonl in place
    uv run split.py --dev-fraction 0.25

Why two disjoint splits rather than "a subset plus the full set"
-----------------------------------------------------------------
You will tune prompts and thresholds while looking at results. Whatever you
look at while tuning is no longer an honest measurement — the numbers drift
towards flattering it. So:

    dev  (~25%)  iterate here, as often as you like      (~1,100 LLM calls/run)
    test (~75%)  run ONCE at the end; the README reports these numbers

If the headline came from the full set, a quarter of it would be data you had
tuned against.

How the split is drawn
----------------------
Positives are split by SOURCE ENTRY, not by query: all perturbations of one SDN
entry land in the same split. Otherwise the transliteration variant of entry X
could sit in dev while its typo variant sits in test, and lessons learned on X
during tuning would leak into the test score.

Within each stratum (positives: subject_type; negatives: negative_kind), keys
are ordered by a hash and the first round(fraction * n) go to dev. Ordering by
hash rather than thresholding it gives an exact fraction per stratum, and the
hash makes the split identical on every run and every machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_PATH = Path("benchmark/queries.jsonl")


def _h(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"split|{key}".encode(), digest_size=8).digest(), "big")


def _pick_dev(keys: list[str], fraction: float) -> set[str]:
    ordered = sorted(keys, key=_h)
    return set(ordered[: round(fraction * len(ordered))])


def assign_splits(queries: list[dict], dev_fraction: float = 0.25) -> list[dict]:
    """Return the same queries with "split" set to "dev" or "test"."""
    # Positives: group by source entry, stratify groups by subject_type.
    groups_by_type: dict[str, set[str]] = defaultdict(set)
    for q in queries:
        if q["is_true_match"]:
            groups_by_type[q["subject_type"]].add(str(q["source_uid"]))
    dev_sources: set[str] = set()
    for keys in groups_by_type.values():
        dev_sources |= _pick_dev(sorted(keys), dev_fraction)

    # Negatives: independent queries, stratify by negative_kind.
    neg_by_kind: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        if not q["is_true_match"]:
            neg_by_kind[q["negative_kind"]].append(q["query_id"])
    dev_negs: set[str] = set()
    for ids in neg_by_kind.values():
        dev_negs |= _pick_dev(ids, dev_fraction)

    for q in queries:
        in_dev = (str(q["source_uid"]) in dev_sources) if q["is_true_match"] else (q["query_id"] in dev_negs)
        q["split"] = "dev" if in_dev else "test"
    return queries


def load_queries(path: Path = DEFAULT_PATH, split: str | None = None) -> list[dict]:
    """What blocking (section 3) and the eval (section 5) should call."""
    with path.open(encoding="utf-8") as fh:
        qs = [json.loads(line) for line in fh if line.strip()]
    return [q for q in qs if split is None or q.get("split") == split]


def split_report(queries: list[dict]) -> dict:
    report: dict = {}
    for split in ("dev", "test"):
        qs = [q for q in queries if q["split"] == split]
        pos = [q for q in qs if q["is_true_match"]]
        neg = [q for q in qs if not q["is_true_match"]]
        report[split] = {
            "positives": len(pos),
            "negatives": len(neg),
            "source_entries": len({q["source_uid"] for q in pos}),
            "positives_by_subject": dict(Counter(q["subject_type"] for q in pos)),
            "positives_by_perturbation": dict(sorted(Counter(q["perturbation_type"] for q in pos).items())),
            "positives_by_expected_outcome": dict(sorted(Counter(
                f'{q["subject_type"]}:{q.get("expected_outcome")}' for q in pos).items())),
            "negatives_by_kind": dict(sorted(Counter(q["negative_kind"] for q in neg).items())),
            "est_llm_calls": len(pos) * 4 + round(len(neg) * 0.08 * 2),
        }
    return report


def verify(queries: list[dict]) -> list[str]:
    """Invariants. An empty list means the split is sound."""
    errs = []
    if any(q.get("split") not in ("dev", "test") for q in queries):
        errs.append("query without a valid split")
    src_splits: dict = defaultdict(set)
    for q in queries:
        if q["is_true_match"]:
            src_splits[q["source_uid"]].add(q["split"])
    leaked = [uid for uid, s in src_splits.items() if len(s) > 1]
    if leaked:
        errs.append(f"{len(leaked)} source entries appear in both splits")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--dev-fraction", type=float, default=0.25)
    args = ap.parse_args()

    queries = load_queries(args.path)
    assign_splits(queries, args.dev_fraction)
    errs = verify(queries)
    if errs:
        raise SystemExit("split failed verification: " + "; ".join(errs))

    with args.path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    report = split_report(queries)
    summary_path = args.path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["splits"] = report
        summary["dev_fraction"] = args.dev_fraction
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\nverified: every query has a split; no source entry appears in both.")


if __name__ == "__main__":
    main()

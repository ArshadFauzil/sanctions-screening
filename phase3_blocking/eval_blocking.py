#!/usr/bin/env python3
"""Measure blocking against the answer key. No LLM.

    uv run python -m phase3_blocking.eval_blocking --split dev

Reports, and writes to benchmark/blocking_report_<split>.json:

1. Recall@k — is a positive's source_uid in its top-k shortlist? Broken down by
   subject type and perturbation type. This is the CEILING on the whole system:
   an entry missing here can never be matched later.
2. Operating points — recall vs. how many (query, candidate) pairs reach the
   adjudicator, for each shortlist size k and score floor. This is the
   recall/LLM-budget trade-off you choose from.
3. Fuzzy-only baseline — "alert on every list entry scoring above a threshold",
   which is what keyword/fuzzy screening tools do. The "before" column of the
   project's headline result.
4. Misses — every positive whose source entry is not in its top-10, for reading.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from common.paths import BENCHMARK_DIR
from phase2_benchmark.split import load_queries

KS = (1, 3, 5, 10)
FLOORS = (0.0, 0.5, 0.6, 0.7, 0.8)
THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def pct(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--queries", type=Path, default=BENCHMARK_DIR / "queries.jsonl")
    ap.add_argument("--database-url", default=None, help="optional: names the source entry of each miss")
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in load_queries(args.queries, None if args.split == "all" else args.split)}
    cand_path = args.queries.parent / f"candidates_{args.split}.jsonl"
    cands = {}
    for line in cand_path.open(encoding="utf-8"):
        r = json.loads(line)
        cands[r["query_id"]] = r["candidates"]
    missing = set(queries) - set(cands)
    if missing:
        raise SystemExit(f"{len(missing)} queries have no candidates line — re-run run_blocking.py")

    pos = [q for q in queries.values() if q["is_true_match"]]
    rank = {}                       # query_id -> 1-based rank of source entry, or None
    for q in pos:
        uids = [c["uid"] for c in cands[q["query_id"]]]
        rank[q["query_id"]] = uids.index(q["source_uid"]) + 1 if q["source_uid"] in uids else None

    in_shortlist = {q["query_id"]: any(c["uid"] == q["source_uid"] for c in cands[q["query_id"]]) for q in pos}

    def recall_at(group: list[dict], k: int) -> float:
        return pct(sum(1 for q in group if rank[q["query_id"]] and rank[q["query_id"]] <= k), len(group))

    report: dict = {"split": args.split, "queries": len(queries), "positives": len(pos)}

    # 1. recall@k
    groups = {"all": pos}
    for st in ("individual", "entity"):
        groups[st] = [q for q in pos if q["subject_type"] == st]
    by_pert = defaultdict(list)
    for q in pos:
        by_pert[f'{q["subject_type"]}:{q["perturbation_type"]}'].append(q)
    # @k = ranked within the top k. "shortlist" = anywhere in what blocking
    # returned, which can exceed k when an ambiguous acronym tie group is
    # appended. "shortlist" is the true ceiling on the adjudicator.
    report["recall_at_k"] = {g: {f"@{k}": recall_at(qs, k) for k in KS}
                             | {"shortlist": pct(sum(in_shortlist[q["query_id"]] for q in qs), len(qs)),
                                "n": len(qs)}
                             for g, qs in groups.items()}
    report["recall_at_10_by_perturbation"] = {
        p: {"recall": pct(sum(in_shortlist[q["query_id"]] for q in qs), len(qs)), "n": len(qs)}
        for p, qs in sorted(by_pert.items())}

    # 2. operating points: recall vs pairs sent to the adjudicator
    ops = []
    for k in (3, 5, 10):
        for floor in FLOORS:
            kept = {qid: [c for i, c in enumerate(cs) if (i < k or c.get("ambiguous")) and c["score"] >= floor]
                    for qid, cs in cands.items()}
            hit = sum(1 for q in pos if any(c["uid"] == q["source_uid"] for c in kept[q["query_id"]]))
            pairs = sum(len(v) for v in kept.values())
            ops.append({"k": k, "floor": floor, "recall": pct(hit, len(pos)),
                        "pairs": pairs, "queries_with_candidates": sum(1 for v in kept.values() if v)})
    report["operating_points"] = ops

    # 3. fuzzy-only baseline, per subject type (score scales differ between them)
    base = {}
    for st in ("individual", "entity"):
        qs = [q for q in queries.values() if q["subject_type"] == st]
        n_pos = sum(q["is_true_match"] for q in qs)
        rows = []
        for t in THRESHOLDS:
            alerts = true_alerts = 0
            caught = set()
            for q in qs:
                for c in cands[q["query_id"]]:
                    if c["score"] >= t:
                        alerts += 1
                        if q["is_true_match"] and c["uid"] == q["source_uid"]:
                            true_alerts += 1
                            caught.add(q["query_id"])
            fp = alerts - true_alerts
            rows.append({"threshold": t, "alerts": alerts, "true_hits_alerted": true_alerts,
                         "recall": pct(len(caught), n_pos), "false_positives": fp,
                         "fp_per_true_hit": round(fp / true_alerts, 2) if true_alerts else None})
        base[st] = rows
    report["fuzzy_only_baseline"] = base

    # 4. misses
    misses = [q for q in pos if not in_shortlist[q["query_id"]]]
    names = {}
    url = args.database_url or (os.environ.get("DATABASE_URL") if _load_env() else None)
    if url and misses:
        import psycopg
        with psycopg.connect(url, connect_timeout=10) as conn, conn.cursor() as cur:
            cur.execute("SELECT uid, primary_name FROM sdn_entries WHERE uid = ANY(%s)",
                        ([q["source_uid"] for q in misses],))
            names = dict(cur.fetchall())
    report["misses"] = [{
        "query_id": q["query_id"], "perturbation": q["perturbation_type"], "query_name": q["query_name"],
        "source_name": names.get(q["source_uid"]),
        "top_candidate": (cands[q["query_id"]][0]["matched_name"] if cands[q["query_id"]] else None),
    } for q in misses]

    out = args.queries.parent / f"blocking_report_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(report)
    print(f"\nfull report: {out}")


def _load_env() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return True
    except ImportError:
        return False


def _print(r: dict) -> None:
    print(f"\n=== Blocking — {r['split']} split: {r['queries']:,} queries, {r['positives']:,} positives ===")
    print("\nRecall     " + "  ".join(f"{'@'+str(k):>6}" for k in KS) + "  shortlist      n")
    for g, v in r["recall_at_k"].items():
        print(f"  {g:<10}" + "  ".join(f"{v['@'+str(k)]:>6.3f}" for k in KS)
              + f"  {v['shortlist']:>9.3f}  {v['n']:>5}")
    print("\nShortlist recall by perturbation")
    for p, v in r["recall_at_10_by_perturbation"].items():
        flag = "   <-- below 0.98" if v["recall"] < 0.98 else ""
        print(f"  {p:<36} {v['recall']:.3f}  (n={v['n']}){flag}")
    print("\nOperating points (recall vs pairs sent to the adjudicator)")
    print("   k  floor  recall    pairs  queries_with_candidates")
    for o in r["operating_points"]:
        print(f"  {o['k']:>2}  {o['floor']:>5.2f}  {o['recall']:.3f}  {o['pairs']:>7,}  {o['queries_with_candidates']:>7,}")
    print("\nFuzzy-only baseline (alert on every candidate scoring >= threshold)")
    for st, rows in r["fuzzy_only_baseline"].items():
        print(f"  {st}")
        for x in rows:
            print(f"    >= {x['threshold']:.2f}: {x['alerts']:>6,} alerts, recall {x['recall']:.3f}, "
                  f"{x['false_positives']:>6,} false positives, {x['fp_per_true_hit']} per true hit")
    print(f"\nMisses (source entry not in the shortlist): {len(r['misses'])}")
    for m in r["misses"][:12]:
        print(f"  {m['perturbation']:<20} {m['query_name'][:34]!r:<36} source={str(m['source_name'])[:30]!r}")


if __name__ == "__main__":
    main()

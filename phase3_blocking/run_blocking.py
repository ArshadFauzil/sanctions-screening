#!/usr/bin/env python3
"""Run blocking for every query in a split -> benchmark/candidates_<split>.jsonl.

    uv run python -m phase3_blocking.run_blocking --split dev
    uv run python -m phase3_blocking.run_blocking --split test     # once, at the end — see split.py

No LLM anywhere: ~3 SQL statements per query against the sdn_names index.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from common.paths import BENCHMARK_DIR
from phase3_blocking.blocking import Blocker
from phase2_benchmark.split import load_queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=Path, default=BENCHMARK_DIR / "queries.jsonl")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()
    load_dotenv()
    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        ap.error("--database-url not given and $DATABASE_URL is not set")

    queries = load_queries(args.queries, None if args.split == "all" else args.split)
    out_path = args.queries.parent / f"candidates_{args.split}.jsonl"
    t0 = time.time()
    with psycopg.connect(url, connect_timeout=10) as conn:
        blocker = Blocker(conn, k=args.k)
        with out_path.open("w", encoding="utf-8") as fh:
            for i, q in enumerate(queries, 1):
                cands = blocker.block(q["query_name"], q["subject_type"])
                fh.write(json.dumps({"query_id": q["query_id"],
                                     "candidates": [c.to_dict() for c in cands]},
                                    ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    print(f"  {i:,}/{len(queries):,} queries blocked")
    dt = time.time() - t0
    print(f"{len(queries):,} queries -> {out_path} in {dt:.1f}s "
          f"({1000 * dt / max(len(queries), 1):.1f} ms/query)")


if __name__ == "__main__":
    main()

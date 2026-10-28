# Sanctions Screening Adjudicator

Screens customer records (individuals and businesses) against the OFAC SDN list and adjudicates
each potential hit as `true_match`, `possible_match` or `false_positive`.

## Layout

```
common/               shared by every phase
    names.py              name normalisation (one definition for benchmark and blocking)
    reference_tables.py   transliterations, legal forms, generic corporate tokens
    paths.py              project-root paths (data/, benchmark/)
phase1_data/          OFAC sdn.xml → Postgres
    schema.sql, load_sdn.py
phase2_benchmark/     labelled test queries with free ground truth
    perturbations.py, make_benchmark.py, split.py
phase3_blocking/      cheap candidate retrieval: 19k listed entries → a shortlist of ~5
    build_name_index.py, blocking.py, run_blocking.py, eval_blocking.py
    blocking.md           design, measurements, scaling notes
phase4_adjudication/  LangGraph adjudicator (planned)
    phase4-adjudication-plan.md
phase5_evaluation/    evaluation of the adjudicator (planned)
    phase5-evaluation-plan.md
data/                 sdn.xml, sdn_advanced.xml — not committed (download from OFAC)
benchmark/            generated: queries.jsonl, candidates_<split>.jsonl, reports — not committed
```

## Running

Every script runs **from the project root as a module** (`-m`), so imports such as
`from common.names import normalize` resolve against the project root:

```bash
uv run python -m phase1_data.load_sdn --reset               # data/sdn.xml → Postgres
uv run python -m phase2_benchmark.make_benchmark            # → benchmark/queries.jsonl
uv run python -m phase3_blocking.build_name_index           # → sdn_names table + indexes
uv run python -m phase3_blocking.run_blocking --split dev   # → benchmark/candidates_dev.jsonl
uv run python -m phase3_blocking.eval_blocking --split dev  # → benchmark/blocking_report_dev.json
```

Requires `DATABASE_URL` in `.env` (see `.env.example`).

"""Step 3 — run_blocking.py and eval_blocking.py.

Unit part: eval_blocking.main() on a hand-made answer key whose every metric was
worked out by hand (see the comments), so a bug in a recall or operating-point
formula cannot hide. run_blocking.main() with a fake database and blocker.

Slow part: the full dev split, end to end, against the numbers blocking.md
reports. It writes into a temporary folder, never over benchmark/.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from common.paths import BENCHMARK_DIR
from phase3_blocking import eval_blocking, run_blocking
from phase3_blocking.blocking import Candidate


def cand(uid, score, ambiguous=False, name=None):
    return {"uid": uid, "score": score, "ambiguous": ambiguous, "matched_name": name or f"N{uid}",
            "entry_type": "individual", "name_kind": "primary", "is_weak": False, "generators": ["trigram"]}


def q(qid, st, src=None, pert=None, split="dev"):
    return {"query_id": qid, "subject_type": st, "query_name": qid, "is_true_match": src is not None,
            "source_uid": src, "perturbation_type": pert, "split": split}


QUERIES = [
    q("Q1", "individual", 1, "typo"),         # true entry ranked 1
    q("Q2", "individual", 2, "typo"),         # true entry ranked 3
    q("Q3", "individual", 3, "token_drop"),   # true entry missing
    q("Q4", "entity", 4, "acronym"),          # true entry ranked 11: beyond @10, inside the tie group
    q("N1", "individual"),
    q("N2", "entity"),
    q("T1", "individual", 99, "typo", split="test"),   # other split: must be ignored
]
CANDS = {
    "Q1": [cand(1, 0.95)],
    "Q2": [cand(9, 0.9), cand(8, 0.8), cand(2, 0.7)],
    "Q3": [cand(7, 0.6, name="Someone Else")],
    "Q4": [cand(u, 0.9, ambiguous=True) for u in range(20, 30)] + [cand(4, 0.9, ambiguous=True)],
    "N1": [cand(5, 0.85)],
    "N2": [],
}


@pytest.fixture
def report(tmp_path, monkeypatch, capsys):
    (tmp_path / "queries.jsonl").write_text("".join(json.dumps(x) + "\n" for x in QUERIES))
    (tmp_path / "candidates_dev.jsonl").write_text(
        "".join(json.dumps({"query_id": k, "candidates": v}) + "\n" for k, v in CANDS.items()))
    monkeypatch.setattr(eval_blocking, "_load_env", lambda: False)    # never touch a real DB
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["eval_blocking", "--split", "dev", "--queries", str(tmp_path / "queries.jsonl")])
    eval_blocking.main()
    capsys.readouterr()
    return json.loads((tmp_path / "blocking_report_dev.json").read_text())


def test_counts(report):
    assert (report["queries"], report["positives"]) == (6, 4)


def test_recall_at_k(report):
    r = report["recall_at_k"]
    # all: Q1@1, Q2@3, Q4@11 (shortlist only), Q3 missing
    assert r["all"] == {"@1": 0.25, "@3": 0.5, "@5": 0.5, "@10": 0.5, "shortlist": 0.75, "n": 4}
    assert r["individual"] == {"@1": 0.3333, "@3": 0.6667, "@5": 0.6667, "@10": 0.6667, "shortlist": 0.6667, "n": 3}
    assert r["entity"] == {"@1": 0.0, "@3": 0.0, "@5": 0.0, "@10": 0.0, "shortlist": 1.0, "n": 1}


def test_recall_by_perturbation(report):
    assert report["recall_at_10_by_perturbation"] == {
        "entity:acronym": {"recall": 1.0, "n": 1},
        "individual:token_drop": {"recall": 0.0, "n": 1},
        "individual:typo": {"recall": 1.0, "n": 2},
    }


@pytest.mark.parametrize("k, floor, recall, pairs, qwc", [
    # k=3, floor 0: Q1 1 + Q2 3 + Q3 1 + Q4 11 (all ambiguous: kept beyond k) + N1 1 = 17 pairs
    (3, 0.0, 0.75, 17, 5),
    # floor 0.8 drops Q2's 0.7 (its true entry) and Q3's 0.6
    (3, 0.8, 0.5, 15, 4),
    # k=10, floor 0.5: Q3's 0.6 survives, Q2's true entry 0.7 survives
    (10, 0.5, 0.75, 17, 5),
    (5, 0.6, 0.75, 17, 5),
    (10, 0.7, 0.75, 16, 4),
])
def test_operating_points(report, k, floor, recall, pairs, qwc):
    [op] = [o for o in report["operating_points"] if o["k"] == k and o["floor"] == floor]
    assert (op["recall"], op["pairs"], op["queries_with_candidates"]) == (recall, pairs, qwc)


def test_fuzzy_only_baseline(report):
    ind = {r["threshold"]: r for r in report["fuzzy_only_baseline"]["individual"]}
    # >= 0.80: Q1 0.95 (true), Q2 0.9 + 0.8 (false), N1 0.85 (false)
    assert ind[0.80] == {"threshold": 0.80, "alerts": 4, "true_hits_alerted": 1, "recall": 0.3333,
                         "false_positives": 3, "fp_per_true_hit": 3.0}
    assert ind[0.95]["alerts"] == 1 and ind[0.95]["fp_per_true_hit"] == 0.0
    ent = {r["threshold"]: r for r in report["fuzzy_only_baseline"]["entity"]}
    assert ent[0.90]["alerts"] == 11 and ent[0.90]["recall"] == 1.0 and ent[0.90]["fp_per_true_hit"] == 10.0
    assert ent[0.95]["alerts"] == 0 and ent[0.95]["fp_per_true_hit"] is None


def test_misses(report):
    assert report["misses"] == [{"query_id": "Q3", "perturbation": "token_drop", "query_name": "Q3",
                                 "source_name": None, "top_candidate": "Someone Else"}]


def test_missing_candidates_line_is_an_error(tmp_path, monkeypatch):
    (tmp_path / "queries.jsonl").write_text(json.dumps(q("Q1", "individual", 1, "typo")) + "\n")
    (tmp_path / "candidates_dev.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--queries", str(tmp_path / "queries.jsonl")])
    with pytest.raises(SystemExit, match="no candidates line"):
        eval_blocking.main()


# ----------------------------------------------------------------------------
# run_blocking with a fake connection and blocker
# ----------------------------------------------------------------------------

class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeBlocker:
    calls: list = []

    def __init__(self, conn, k=10):
        self.k = k

    def block(self, name, subject_type):
        _FakeBlocker.calls.append((name, subject_type, self.k))
        return [Candidate(uid=1, score=0.123456, entry_type=subject_type, matched_name="X", name_kind="primary",
                          is_weak=False, generators=["trigram"])]


def test_run_blocking_writes_one_line_per_query(tmp_path, monkeypatch, capsys):
    (tmp_path / "queries.jsonl").write_text("".join(json.dumps(x) + "\n" for x in QUERIES))
    monkeypatch.setattr(run_blocking.psycopg, "connect", lambda *a, **k: _Conn())
    monkeypatch.setattr(run_blocking, "Blocker", _FakeBlocker)
    _FakeBlocker.calls = []
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--k", "7", "--database-url", "fake",
                                      "--queries", str(tmp_path / "queries.jsonl")])
    run_blocking.main()
    lines = [json.loads(x) for x in (tmp_path / "candidates_dev.jsonl").read_text().splitlines()]
    assert [x["query_id"] for x in lines] == ["Q1", "Q2", "Q3", "Q4", "N1", "N2"]      # dev only, in order
    assert lines[0]["candidates"][0]["score"] == 0.1235
    assert {k for _, _, k in _FakeBlocker.calls} == {7}
    # blocking sees the NAME and SUBJECT TYPE only — never the labels
    assert _FakeBlocker.calls[0][:2] == ("Q1", "individual")


def test_run_blocking_requires_a_database_url(tmp_path, monkeypatch):
    monkeypatch.setattr(run_blocking, "load_dotenv", lambda: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--queries", str(tmp_path / "q.jsonl")])
    with pytest.raises(SystemExit):
        run_blocking.main()


# ----------------------------------------------------------------------------
# the real dev split, end to end (slow)
# ----------------------------------------------------------------------------

@pytest.mark.slow
def test_dev_split_matches_blocking_md(conn, tmp_path, monkeypatch, capsys):
    """blocking.md 'Results on the dev split' and 'Operating points', re-measured."""
    src = BENCHMARK_DIR / "queries.jsonl"
    if not src.exists():
        pytest.skip("benchmark/queries.jsonl not generated")
    shutil.copy(src, tmp_path / "queries.jsonl")
    url = os.environ["DATABASE_URL"]       # conn.info.dsn omits the password
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--database-url", url,
                                      "--queries", str(tmp_path / "queries.jsonl")])
    run_blocking.main()
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--queries", str(tmp_path / "queries.jsonl")])
    eval_blocking.main()
    capsys.readouterr()
    r = json.loads((tmp_path / "blocking_report_dev.json").read_text())

    rk = r["recall_at_k"]
    assert rk["individual"]["n"] == 151 and rk["entity"]["n"] == 95
    assert rk["individual"]["shortlist"] == rk["entity"]["shortlist"] == 1.0
    assert rk["individual"]["@10"] == rk["entity"]["@10"] == 1.0    # Day-1 exit criterion: >= 0.98
    ops = {(o["k"], o["floor"]): o for o in r["operating_points"]}
    assert (ops[(5, 0.5)]["recall"], ops[(5, 0.5)]["pairs"], ops[(5, 0.5)]["queries_with_candidates"]) == (1.0, 3229, 835)
    assert ops[(5, 0.6)]["recall"] == 1.0 and ops[(5, 0.7)]["recall"] < 1.0      # the cliff
    base = {x["threshold"]: x for x in r["fuzzy_only_baseline"]["individual"]}
    assert base[0.85]["recall"] == 1.0 and base[0.80]["alerts"] == 996
    ent = {x["threshold"]: x for x in r["fuzzy_only_baseline"]["entity"]}
    assert max(x["recall"] for x in ent.values()) < 1.0    # name-only screening never catches every entity

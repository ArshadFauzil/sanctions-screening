# Phase 5 — Evaluating the Adjudicator: eval plan

**Goal.** Measure the adjudicator (Phase 4) well enough that you can trust its output, improve it
deliberately, and make claims about it that hold up with a sceptical technical interviewer.

**Why this phase is different.** Phases 1–3 are deterministic: run them twice, get the same bytes.
Phase 4 contains an LLM. Its output varies between runs, it can be confidently wrong, and it can
be right for the wrong reason. Evaluation is therefore not a final check — it is the instrument
you steer by. This plan is written before the Phase 4 code on purpose (Phase 4 §0).

▸ Concept boxes explain the evaluation ideas as they come up.

---

## 1. Errors are not equal — decide what matters before measuring

In sanctions screening the four kinds of error have very different costs:

| error | example | cost |
|---|---|---|
| **missed hit** | a listed person's record ends `false_positive` | regulatory failure — the worst outcome |
| **wrong-entity confirmation** | `true_match` against the *wrong* listed person, or against an innocent customer | an innocent customer's assets frozen |
| **policy violation** | auto-confirmed (`true_match`) a case the rule says is ambiguous | process failure; an examiner finding |
| **over-escalation** | `possible_match` where `true_match` or `false_positive` was correct | an analyst spends a few minutes |

So the metrics are tiered. Tier 1 are **gates**: if they fail, nothing else about the run matters.
A single accuracy number would hide exactly the errors that count — a system that is 99% accurate
by escalating everything ambiguous is excellent; one that is 99% accurate by occasionally
clearing a true hit is unusable.

---

## 2. What the ground truth says, unit by unit

The benchmark gives exact truth for more than it first appears:

| unit | ground truth | acceptable outcomes |
|---|---|---|
| (positive query, its source uid) | `expected_outcome` from the labeller | exactly that one |
| (positive query, any other uid) | a **different** listed entity | `false_positive` or `possible_match` — never `true_match` |
| (negative query, any uid) | a different entity | `false_positive`; the new name-only negatives (§3) expect `possible_match` |
| positive query whose source uid was not shortlisted | blocking miss | counts as a missed hit |

The second row matters: every shortlisted candidate that is *not* the true entity is a labelled
negative pair, so wrong-entity confirmations are measurable on positives as well as negatives.

---

## 3. The dataset — what to add before Phase 4 is built

The current benchmark has a blind spot measured in Phase 4 §1: the code precheck clears almost
every negative on location before the LLM sees it (only 21 of 750 dev negatives reach the LLM).
So the LLM's most important skill — telling a same-named stranger apart *when every other
attribute agrees* — is barely tested. Four additions fix that, including a safety test for the precheck:

| addition | how it is built | expected outcome | tests |
|---|---|---|---|
| **near-miss negatives** (~150) | copy a listed individual's year of birth, city and country exactly; change the given name to a *different* but similar name from a hand-curated near-miss table (Hassan→Hussein, Mohammed→Mahmoud, Omar→Amr, Yusuf→Yunus, Sergei→Semyon); entities: change one distinctive token (`PARS OIL`→`PARS GAS`) | `false_positive` | LLM precision on names — the core skill; these survive the precheck |
| **name-only negatives** (~50) | a listed name exactly, with no year of birth and no location | `possible_match` | escalation instead of clearing *or* confirming on a name alone |
| **nickname positives** (~40) | given name replaced by a nickname from `NICKNAME_GROUPS`; extra source entries sampled whose given name is in the table | as the base rule (NICKNAME = name match) | the Phase 3 carry-forward |
| **attribute-noise positives** (~80) | positives with country written differently (`Russian Federation`, `UAE`, `Iran, Islamic Republic of`, `Syrian Arab Republic`) and city transliterations (`Moskva`) | unchanged from the base query | **safety of the precheck**: a spelling difference must never clear a true hit |

The near-miss table must be **disjoint** from the transliteration and nickname tables — a unit test
asserts no pair appears in two. Hussein is not a transliteration of Hassan; if it were in both,
the label would contradict itself.

▸ **Concept — test the tails, not the average.** Random samples are dominated by easy cases. An
eval that only reports the average over easy cases tells you nothing about the cases that cause
incidents. Strata like these are built to hit the decisions you are actually worried about.

**Do not reshuffle the existing split.** `split.py` assigns dev by taking the first 25% of each
stratum in hash order, so adding new queries to a stratum would move some *existing* queries
between dev and test — leaking queries you have already looked at into test. Freeze the current
`split` field for existing queries and assign only new ones (by the same hash, with a fixed
threshold rather than a count). A test asserts no existing query changed split.

---

## 4. Metrics

### Tier 1 — gates (must hold, on every run)

**Missed-hit rate** — positives whose true entity ends `false_positive` or was never shortlisted.
Target: zero.

▸ **Concept — "zero" needs an upper bound.** Zero misses in a sample does not mean zero misses in
general. If 0 of n cases fail, the 95% upper bound on the true failure rate is about 3/n (the
*rule of three*). Computed exactly:

| positives observed with 0 misses | 95% upper bound on the miss rate |
|---|---|
| 246 (dev) | 1.21% |
| 739 (test) | 0.40% |
| 985 (dev + test) | 0.30% |

So the honest claim from a clean test run is "no missed hits in 739 cases; the miss rate is below
0.4% with 95% confidence" — not "it never misses". (Counts after the 3-character acronym minimum
removed 19 two-letter acronym queries, 28 Oct 2026; the §3 additions will raise them.)

**Wrong-entity confirmations** — any `true_match` on a different entity (§2 rows 2–3). Target:
zero; reported with the same upper bound over all different-entity pairs that reached `decide`.

### Tier 2 — policy correctness

A 3 × 3 confusion matrix on (positive query, source uid) pairs, expected vs actual:

|  | actual `true_match` | actual `possible_match` | actual `false_positive` |
|---|---|---|---|
| **expected `true_match`** | correct confirmation | over-escalation (workload) | **missed hit** (Tier 1) |
| **expected `possible_match`** | **policy violation** | correct escalation | **missed hit** (Tier 1) |

Plus, for negatives, the outcome distribution per negative kind. The near-miss row is the LLM's
precision: of near-miss negatives reaching the LLM, how many end `false_positive`.

Worked example (illustrative numbers, not results). Expected `true_match` = 124 cases: 110
confirmed, 14 escalated → confirmation rate 88.7%, 95% interval 81.9–93.2% (Wilson). Expected
`possible_match` = 126 cases: 2 auto-confirmed → policy-violation rate 1.6%, interval 0.4–5.6%.
Note how wide the intervals are at this size; this is why §6 matters.

### Tier 3 — what it is worth: workload and cost

- **The funnel** — the picture that explains the whole system at a glance: customers screened →
  shortlisted pairs → cleared by rules → sent to the LLM → `true_match` / `possible_match` /
  `false_positive`. Measured on dev on 14 Oct: 1,000 customers → 3,349 pairs → 2,807 cleared by
  rules (84%) → 542 pairs across 271 customers needing the LLM. Since the 3-character acronym
  minimum (28 Oct): 996 customers → 3,229 pairs; the rest is re-measured with `rules.py`.
- **Analyst workload vs name-only screening at the same recall.** Name-only screening has to
  lower its threshold until it stops missing hits, and every alert above the threshold goes to an
  analyst. Measured in Phase 3 on dev: to catch every true individual it must alert at ≥ 0.80 —
  **996 alerts**; for entities it **never reaches 100% recall** at any tested threshold (83.8% at
  0.80). The adjudicator's comparable number is the count of `possible_match` dispositions at zero
  misses. This comparison is the project's business case.
- **Cost** — LLM calls, input/output tokens, and latency (p50, p95) per 1,000 customers screened.

### Tier 4 — LLM behaviour (diagnostics)

- **Schema-valid on first attempt**, **grounding failures** (evidence not found in the input),
  **retry rate**, **fail-safe rate** (`llm_unreliable`). A rising grounding-failure rate after a
  prompt or model change is an early hallucination warning.
- **Right for the right reason** — the name reason code against the perturbation that produced the
  query: transliteration → `TRANSLITERATION`, initialize → `INITIAL`, acronym → `ACRONYM`, nickname
  → `NICKNAME`, name order → `WORD_ORDER`, dropped middle name → `TOKEN_MISSING`, near-miss →
  `DIFFERENT_GIVEN_NAME`. A correct verdict with the wrong reason is luck, and luck does not
  survive a model change.
- **Run-to-run consistency** — §5.

Every metric is reported overall, by subject type, by perturbation or negative kind, and by
expected outcome. Aggregates hide the stratum that is failing.

---

## 5. Measuring non-determinism

Temperature 0 does not make a hosted LLM deterministic (Phase 4 §6). So: run the full dev set
**three times with the response cache off**, then report

- **flip rate** — the share of candidate pairs whose outcome differs between any two runs;
- **every flip on a true entity**, listed individually: a flip between `true_match` and
  `possible_match` is workload noise; **a flip into `false_positive` is a Tier 1 finding** — the
  system missed a hit on one run in three;
- the spread of every headline metric across the three runs.

During day-to-day development the cache stays **on**: the same input returns the same stored
answer, so a change in results means *your* change caused it, not provider noise.

---

## 6. Statistics that keep you honest

**Clustered confidence intervals.** The 246 dev positives come from 50 source entries, roughly
five variants each. If the model mishandles one listed name, it tends to mishandle all of that
entry's variants together, so the 246 cases are not 246 independent observations. Compute intervals
with a **cluster bootstrap**: resample *source entries* (each negative is its own cluster) 1,000
times, recompute the metric each time, take the 2.5th and 97.5th percentiles. Intervals come out
wider — and true.

**Comparing two configurations: paired, not side by side.** Both configurations (prompt v1 vs v2,
model A vs model B) run on the *same* queries, so compare them query by query with **McNemar's
test**, which looks only at the queries where they disagree. Example: on 250 positives, v2 fixes
9 cases v1 got wrong and breaks 2 that v1 got right → p = 0.07: *not* a demonstrated improvement,
despite +7 net. Fixes 14, breaks 2 → p = 0.006: a real one. Two accuracy figures side by side
would have called both an improvement.

▸ **Concept — most "improvements" are noise.** At a few hundred cases, a 2–3 point change in a
metric is routinely within run-to-run variation. Accept a change only when the paired test says
so — or, for Tier 1, when it removes a specific failure you can point to.

---

## 7. Ablations — proving each design decision earned its place

Each row removes or swaps one component and reruns dev. Together they turn "I built a pipeline"
into "I can show what each part contributes".

| | configuration | LLM calls (dev) | question it answers |
|---|---|---|---|
| A0 | fuzzy threshold only (Phase 3 baseline) | 0 | what is the status quo? |
| A1 | precheck + **name verdict from string similarity**, no LLM | 0 | **does the LLM add anything at all?** |
| A2 | full system, prompt v1 | ~300–450 | the headline |
| A3 | LLM given the match rule and asked for the outcome directly; no `decide()` | ~300–450 | does "LLM judges facts, code applies policy" (Phase 4 Decision 2) matter? |
| A4 | no precheck: every shortlisted pair to the LLM | ~1,000 | what do the code rules save, in cost *and* in errors? (run on a 200-query sample if quota is short) |
| A5 | no grounding validation | 0 (from A2 logs) | how many invented evidence strings would have reached a decision? |
| A6 | model swap: Gemini Flash vs Groq `gpt-oss-120b` | ~300–450 | is the result a property of the design or of one model? |
| A7 | without nickname facts in the prompt | ~40 (nickname cases only) | does giving facts beat relying on model recall? |

A1 deserves special care: if string similarity plus your rules does almost as well as the LLM, that
is a finding to report, not hide — and the LLM's value is then concentrated in the near-miss and
nickname strata, which the table will show. A3 is the most interesting result for an interview:
it puts a number on the architectural principle.

Quota: with the response cache, A5 is free and A7 reuses most of A2. Schedule A2, A3, A6 on
separate days if the free tier's daily token limit is tight.

---

## 8. Day-to-day loop: golden set, error analysis, regression

**Golden set** — `phase5_evaluation/golden.jsonl`, ~40 cases you choose and label as the domain expert, at
least three per difficult category: transliteration, nickname, initial, acronym (unique and
ambiguous), multiple DOBs, country-only location, near-miss names, attribute noise, vessel-name
collisions. Run it after **every** prompt or code change (40 calls, a minute). Rule: every safety
case in the golden set must pass before a change is accepted; a change that fixes three dev errors
and breaks one golden safety case is rejected.

**Error analysis** — `eval_adjudication.py` writes `errors_<split>.csv`: query, candidate, the
model's verdicts and evidence, expected vs actual, and the checkpoint `thread_id` so the full state
can be replayed. Classify every error before fixing anything:

| class | fix where |
|---|---|
| blocking miss | Phase 3 |
| precheck cleared wrongly (country normalisation, missing value treated as conflict) | `rules.py` — a safety bug |
| LLM name judgement wrong | prompt definitions / examples |
| LLM location judgement wrong | prompt, or move the check into code |
| `decide()` disagrees with the labeller | the rule, in one place — definition error, not model error |
| label wrong | the benchmark generator |

**Fix the class, not the instance.** Do not paste a failing dev case into the prompt as an
example; that tunes the prompt to the dev set and the improvement will not transfer to test.
Write a general definition or a fresh example of the same pattern, then confirm on golden and dev.

**Unit tests that need no LLM** — `pytest`: `rules.py` on hand-written fixtures, `decide()` against
the benchmark labeller, validation rejecting invented evidence, and the whole graph with a fake LLM
that replays recorded responses. These run in seconds and catch most regressions before any quota
is spent.

---

## 9. The test run — once, pre-registered

▸ **Concept — pre-registration.** Writing down what you will measure and what would count as
success *before* seeing the results stops you, unconsciously, from choosing whichever metric
looks best afterwards.

1. Freeze: model and version, prompt file hash, git commit, operating point (k, floor), precheck
   rules. Record them in `phase5_evaluation/PREREGISTRATION.md` with the date.
2. In the same file, state the claims you intend to make and their thresholds — for example:
   *"missed hits: 0 of 739 (95% upper bound below 0.5%); wrong-entity confirmations: 0;
   policy-violation rate below 3%"*.
3. Run blocking and the adjudicator on test once; run the evaluator.
4. Report every pre-registered metric, including any that failed. A pre-registered claim that
   failed, reported honestly with its error analysis, is more credible than a clean table.

---

## 10. Reporting — and marketing the project

`eval_adjudication.py --split dev|test` writes `phase5_evaluation/reports/report_<split>.json` and a rendered
`report_<split>.md`. The README is built from the test report:

1. **One headline sentence**, with numbers filled from the report — template: *"On N held-out
   screenings, zero true sanctions hits were cleared (95% upper bound X%), and analyst alerts fell
   from A (name-only screening at full recall) to B."* Never fill it with numbers you did not measure.
2. **The funnel** (§4 Tier 3) — what happened to every customer.
3. **Workload vs name-only screening** at matched recall — the business case.
4. **The confusion matrix** on true entities, and outcomes on near-miss negatives.
5. **The ablation table** — what each design decision contributed, especially A1 and A3.
6. **Consistency** — the three-run flip rate.
7. **Analyst review** — you, as an AML analyst, read 50 random `possible_match` and 25
   `true_match` disposition notes and record whether you would accept each. "Disposition notes
   accepted by a practising AML analyst: x/75" is evidence few portfolio projects can offer.
8. **What this does not show**: synthetic customers, a single sanctions list, labels defined by
   one analyst's rule, a free-tier model. A reviewer trusts a project that states its limits.

---

## 11. Files

```
phase2_benchmark/make_benchmark.py, perturbations.py
                                         + near-miss, name-only, nickname, attribute-noise cases (§3)
common/reference_tables.py               + NEAR_MISS_GIVEN_NAMES, NEAR_MISS_ENTITY_TOKENS
phase2_benchmark/split.py                + freeze existing splits; threshold assignment for new queries
phase5_evaluation/
    __init__.py
    golden.jsonl                         hand-labelled hard cases (§8)
    PREREGISTRATION.md                   written before the test run (§9)
    eval_adjudication.py                 Tier 1–4 metrics, funnel, slices, bootstrap CIs, errors CSV, reports
    consistency_runs.py                  3 cache-off runs + flip-rate report (§5)
    ablate.py                            runs A1–A7 configurations; McNemar comparisons (§6, §7)
    run_golden.py                        golden-set gate (§8)
    reports/                             report_<split>.json / .md, errors_<split>.csv
tests/test_benchmark_v2.py               near-miss tables disjoint; no existing query changed split
```

## 12. Order of work

| when | work |
|---|---|
| before Phase 4 code | §3 dataset additions; freeze splits; golden set drafted (≈ ½ day) |
| during Phase 4 | unit tests and golden set on every change; dev reports with cache on |
| end of Phase 4 | full dev evaluation; A1, A5 (free); error analysis; one prompt revision if needed |
| following days (quota) | A2 / A3 / A6 / A7 ablations; consistency runs |
| last | pre-register; single test run; README |

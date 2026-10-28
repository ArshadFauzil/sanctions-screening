# Phase 4 — The Adjudicator: implementation plan

**Goal.** For every customer record, take blocking's shortlist and decide, for each candidate SDN
entry: `true_match`, `possible_match` or `false_positive` — with an auditable reason, using the
match rule you set, and without ever clearing a true sanctions hit.

**Who this is written for.** Someone who knows software engineering but not agentic development.
Every design choice states *what* was chosen, *why*, and *what the alternative would have cost*.
Concept boxes (▸) explain the agentic-development ideas as they come up.

**Read with:** `phase3_blocking/blocking.md` (Phase 3, the input to this phase) and
`phase5_evaluation/phase5-evaluation-plan.md` (how this phase is measured — written first, on purpose; see §0).

---

## 0. Build the evaluation first

The eval plan (Phase 5) is written before this one, and its new benchmark cases (§3 of that
document) are generated **before any adjudicator code exists**. This is not ceremony. If you
write the adjudicator first, every design decision is tuned against cases you happened to look
at, and the benchmark you build afterwards will — without you meaning it to — test what the
system already does well. Labels written first are the only honest labels.

▸ **Concept — eval-first development.** In ordinary software a function either returns the right
value or it does not, and a unit test settles it. An LLM component returns *plausible* output that
is right most of the time. The only way to know "most" is a measured number, and the only way to
make design decisions is to compare those numbers. So the measuring instrument comes first.

---

## 1. What the measurements already tell us

Measured on the dev split before writing this plan (no LLM involved):

| | |
|---|---|
| dev queries | 1,000 (250 true entities, 750 innocent customers) |
| shortlisted pairs at the chosen operating point (k = 5, score floor 0.50) | 3,349 |
| pairs settled by plain-code rules (§4) | **2,807 (84%)** |
| queries that still need the LLM | **271** (250 positives, 21 negatives), ~2 candidates each |
| true (customer, listed entity) pairs wrongly cleared by those rules | **0** |

*Measured 14 Oct 2026. On 28 Oct the 3-character acronym minimum (`phase3_blocking/blocking.md`,
Step 1c) removed 4 dev positives (two-letter acronyms): dev is now 996 queries (246 true
entities) and 3,229 shortlisted pairs. The precheck split is re-measured in build step 1 (§9),
once `rules.py` exists; expect figures within a few pairs of the ones above.*

Two consequences shape everything below. First, the LLM is needed for a minority of decisions,
so it can be used carefully rather than cheaply. Second, the benchmark's innocent customers are
almost all Canada-based, so code clears them on location before the LLM sees them — the LLM's
ability to *reject* a same-named stranger is barely tested by the current benchmark. Phase 5 §3
fixes that with new "near-miss" negatives; read it before trusting any precision number.

---

## 2. The architecture in one picture

```
candidates_<split>.jsonl  (Phase 3)
        │
  load_case        fetch each candidate's listed attributes (DOBs, addresses, place of
        │          birth, registration countries, entry type, aliases)
  precheck         CODE: entry type, year of birth ±1, location / registration country
        │          → decisive conflicts become false_positive here; no LLM
        ├─ nothing left to judge ──────────────────────┐
  adjudicate       LLM: ONE call per customer, all surviving candidates;
        │          answers narrow factual questions — is this the same name? same city?
  validate         CODE: schema valid? every candidate answered once? every quoted
        │          value present in the input?  → retry once, else fail safe
  decide   ◄───────┘ CODE: your match rule turns verdicts into the three outcomes
        │
        ├─ any possible_match ──► human_review   (pause; analyst decides; resume)
  finalize         disposition record + audit trail
        ▼
adjudications_<split>.jsonl  (Phase 5 reads this)
```

---

## 3. The six decisions that define the design

### Decision 1 — A workflow with fixed steps, not an autonomous agent

▸ **Concept — workflow vs agent.** An *agent* is an LLM in a loop that decides for itself which
tool to call next and when it is finished. A *workflow* is code that calls an LLM at fixed points
along a path you designed. Both are "agentic development"; most production systems are workflows,
with agent-style autonomy reserved for problems where the path genuinely cannot be known in
advance.

This problem has a known path: gather facts, check rules, judge names, apply policy. An
autonomous agent would add a failure mode (it might skip a check, loop, or call tools in a
different order on each run) and buy nothing. In a regulated process the order of checks is
itself something an examiner asks about. So: a **LangGraph state graph with fixed edges**. The
LLM is one node among seven.

*What it would have cost to choose an agent:* nondeterministic control flow, unbounded cost per
customer, and an audit trail that differs run to run.

### Decision 2 — The LLM judges facts; code applies policy

Your match rule is *policy*: individual = name + year of birth (±1) + city/country; entity =
full name + place of registration; anything ambiguous is a possible match. Policy belongs in code:

- **Consistency.** An LLM asked to "apply the rule" will apply it slightly differently across
  calls. A Python function applies it identically every time.
- **Change.** When you refine the rule (you already did, twice), you change one function, not a
  prompt whose behaviour you then have to re-measure from scratch.
- **Measurement.** If the LLM only answers "is `Hassen` the same name as `Hassan`?", you can
  measure *that* skill on its own. If it also applies the rule, a wrong outcome could be either.
- **Audit.** "Rule 3 fired because the year of birth differs by 23" is an explanation an examiner
  accepts. "The model said so" is not.

What the LLM *is* for: judgements code is bad at — whether two names in different scripts and
conventions refer to the same name (transliteration, word order, a dropped middle name, an
initial, a legal-form difference, an acronym), and whether two place descriptions refer to the
same place (`Qatif-Bab al Shamal, Saudi Arabia` vs a customer in `Qatif`).

### Decision 3 — Deterministic precheck before any LLM call

The cheapest, most reliable LLM call is the one you never make. The precheck applies only the
parts of your rule that need no judgement, and **may only clear, never confirm**:

| Rule | Example | Effect |
|---|---|---|
| entry-type mismatch | company vs a listed vessel | `false_positive` |
| year of birth | customer 1985; every listed DOB year outside 1984–1986 | `false_positive` |
| location (individuals) | customer country absent from every listed address and place of birth | `false_positive` |
| place of registration (entities) | customer registered in Canada; listed entity's registration and address countries do not include Canada | `false_positive` |

Two safety constraints, both enforced in code and tested:

1. **Country names are normalised before comparing** (`Russia` / `Russian Federation`, `UAE` /
   `United Arab Emirates`, `Iran` / `Iran, Islamic Republic of`) using `pycountry` plus a small
   alias table. A country that cannot be normalised is treated as *not decisive* and goes to the
   LLM. Failing to normalise must never produce a conflict.
2. **A missing value is never a conflict.** If either side lacks a year of birth, that criterion
   is *absent*, not *disagreeing* — absent criteria lead to `possible_match` (your rule), never to
   `false_positive`.

The 84% / 271-query / 0-wrongly-cleared figures in §1 come from a crude version of this precheck
(lower-cased exact country match). The real one is stricter about what it calls decisive, so
expect a few more pairs to reach the LLM, not fewer.

### Decision 4 — One LLM call per customer, not per candidate pair

All of a customer's surviving candidates go into one prompt (on dev, ~2 on average; ambiguous
acronym groups can be much larger — see §6). Fewer calls fits the free-tier request quota; the
model also sees the candidates side by side, which helps it notice that two listed people share a
surname but not a given name. The risk — the model mixing up candidates — is handled in §5:
each candidate has an ID, and validation rejects any response that does not answer every ID
exactly once.

### Decision 5 — Structured output, reason codes, and grounding

▸ **Concept — structured output.** Instead of asking for prose and parsing it, you give the model
a schema (here, a Pydantic class) and the API constrains its reply to valid JSON of that shape.
LangChain exposes this as `llm.with_structured_output(Schema)`. Free text is unmeasurable;
enumerated fields are.

```python
class NameVerdict(str, Enum):
    MATCH = "match"          # same name: transliteration, word order, dropped middle name,
                             # legal form / punctuation differences, a known nickname
    PARTIAL = "partial"      # consistent but incomplete: an initial, an acronym
    MISMATCH = "mismatch"    # a different name (Hassan vs Hussein, John vs Joan)

class NameReason(str, Enum):
    IDENTICAL = "identical"; TRANSLITERATION = "transliteration"; WORD_ORDER = "word_order"
    TOKEN_MISSING = "token_missing"; TYPO = "typo"; NICKNAME = "nickname"
    INITIAL = "initial"; ACRONYM = "acronym"; LEGAL_FORM = "legal_form"
    DIFFERENT_GIVEN_NAME = "different_given_name"; DIFFERENT_SURNAME = "different_surname"
    DIFFERENT_ENTITY_NAME = "different_entity_name"

class CandidateJudgement(BaseModel):
    candidate_id: str                       # must be one of the IDs in the prompt
    name_verdict: NameVerdict
    name_reason: NameReason
    customer_name_evidence: str             # the exact customer string compared
    listed_name_evidence: str               # the exact listed name/alias it matched
    city_verdict: Literal["agree", "disagree", "absent"]   # individuals only
    city_evidence: str | None               # exact listed location string used

class Adjudication(BaseModel):
    judgements: list[CandidateJudgement]
```

Three things make this anti-hallucination rather than just tidy:

- **Reason codes** turn "why" into data. Phase 5 checks them against the benchmark: a
  transliteration query should come back `TRANSLITERATION`, an acronym query `ACRONYM`.
- **Evidence fields must quote the input.** Validation checks that `listed_name_evidence` is
  literally one of that candidate's names or aliases, and `city_evidence` literally one of its
  locations. A model that invents an alias fails validation.
- **The model is not asked for a final outcome at all.** It cannot hallucinate a `true_match`
  because it is never asked for one.

### Decision 6 — Fail safe towards escalation, never towards clearing

When something goes wrong — invalid JSON, a missing candidate, a quote not found in the input,
a provider timeout — the system retries once with the error message attached. If it fails again,
every affected candidate becomes `possible_match` with flag `llm_unreliable`. Never
`false_positive`. In sanctions screening the two errors are not symmetric: an unnecessary
escalation costs an analyst a few minutes; a cleared true hit is a regulatory failure.

---

## 4. Your match rule, as code

`decide()` is a pure function — no I/O, no LLM — and the most heavily unit-tested code in the
project. Per candidate:

```
if any decisive conflict (precheck, or name_verdict == MISMATCH) ...... false_positive
elif individual and name MATCH and YOB within ±1 and exactly one distinct listed birth year
     and country agrees and city agrees ........................... true_match
elif entity and name MATCH and registration country agrees ........ true_match
else .............................................................. possible_match
```

Location is decisive only at **country** level. Same country but a different city is
`possible_match`, not `false_positive`: OFAC address data is incomplete, so a listed person can
live in a city the list never mentions. Treating a city difference as a conflict would clear true
hits on missing data — the error this whole design exists to avoid.

`possible_match` therefore covers exactly the cases you named: a criterion missing, a
country-only location, several listed DOBs, a partial name (initial, acronym), an ambiguous
acronym group, and any LLM failure.

**Query-level disposition** = the most serious outcome among its candidates
(`true_match` > `possible_match` > `false_positive`), plus the list of uids at that level.

The rule must match the one `make_benchmark.expected_outcome()` used to label the benchmark. They
were written from the same rule, but if `decide()` and the labeller ever disagree, Phase 5 will
report the disagreement as model error when it is a definition error. A unit test runs both on
the same fixtures and fails if they differ.

---

## 5. The graph, node by node

▸ **Concept — LangGraph.** A *state graph*: you define a typed **state** dict, **nodes** (Python
functions that read the state and return updates to it) and **edges** (which node runs next).
*Conditional edges* choose the next node from the state. A **checkpointer** saves the state after
every node, so a run can pause and resume — and so every intermediate value is recorded, which is
your audit trail.

```python
class CaseState(TypedDict):
    query: dict                        # customer record from queries.jsonl (name + attributes)
    candidates: list[dict]             # shortlist + each candidate's listed attributes
    precheck: dict[str, str]           # candidate_id -> "fp:<rule>" | "llm"
    adjudication: Adjudication | None  # validated LLM output
    errors: list[str]                  # validation messages fed back on retry
    attempts: int
    outcomes: dict[str, str]           # candidate_id -> outcome
    disposition: dict                  # query-level result, audit fields
```

| node | kind | does |
|---|---|---|
| `load_case` | code | reads the shortlist (k = 5, floor 0.50, plus ambiguous acronym ties) and fetches each candidate's listed attributes from Postgres |
| `precheck` | code | §3 Decision 3; marks each candidate `fp:<rule>` or `llm` |
| `adjudicate` | **LLM** | one call with every `llm` candidate; returns `Adjudication` |
| `validate` | code | schema, IDs answered exactly once, evidence quoted from input |
| `decide` | code | §4; pure function |
| `human_review` | pause | `interrupt()`; an analyst resolves `possible_match` cases |
| `finalize` | code | disposition record, rendered note, audit fields |

```python
g = StateGraph(CaseState)
for name, fn in [("load_case", load_case), ("precheck", precheck), ("adjudicate", adjudicate),
                 ("validate", validate), ("decide", decide_node),
                 ("human_review", human_review), ("finalize", finalize)]:
    g.add_node(name, fn)
g.add_edge(START, "load_case")
g.add_edge("load_case", "precheck")
g.add_conditional_edges("precheck", lambda s: "adjudicate" if "llm" in s["precheck"].values() else "decide")
g.add_edge("adjudicate", "validate")
g.add_conditional_edges("validate", route_after_validate)   # "decide" | "adjudicate" (retry) | "decide" (fail safe)
g.add_conditional_edges("decide", lambda s: "human_review" if needs_review(s) else "finalize")
g.add_edge("human_review", "finalize")
g.add_edge("finalize", END)
app = g.compile(checkpointer=checkpointer)
```

▸ **Concept — human-in-the-loop with `interrupt()`.** Inside `human_review`, calling
`interrupt(payload)` stops the graph and saves its state under the run's `thread_id` (use the
query ID). Days later an analyst's decision resumes it with `app.invoke(Command(resume=decision),
config)`, continuing exactly where it stopped. In **batch evaluation mode** the node does not
interrupt; it records the case as escalated, because Phase 5 measures the system, not a person.

▸ **Concept — checkpointer.** Use `PostgresSaver` (package `langgraph-checkpoint-postgres`) on the
database you already run. Every node's input and output for every customer is then queryable by
`thread_id` — "show me exactly what the model saw and said for P-26084" is one lookup.

LangGraph's API has moved between versions. Pin the versions in `pyproject.toml` when you
install, and check these names against the docs for that version before writing the graph.

---

## 6. The LLM layer

**Prompt as a versioned file.** `phase4_adjudication/prompts/adjudicate_v1.md`, never an f-string buried
in code. Its hash is written into every output record, so any result can be traced to the exact
prompt that produced it, and Phase 5 can compare `v1` against `v2`.

**What goes in the prompt.** Definitions of every `NameVerdict` and `NameReason` with one example
each; the customer record; per candidate an ID, entry type, primary name, up to 5 aliases (the
ones closest to the customer name, to bound length), listed locations and — for individuals —
whether the year of birth already agreed (so the model does not re-litigate settled facts).
Nothing else: no instructions about outcomes, no policy.

**Give the model facts instead of asking it to recall them.** The nickname table (§7) is applied
*before* the call: if `Pepe` and `José` are listed equivalents, the prompt says so
(`"nickname table: PEPE ≡ JOSE"`). The model does not need to know Spanish hypocorisms; it needs
to read. The same applies to transliteration groups already in `reference_tables.py`.

**Provider.** Free tiers, behind one function `get_llm(role)` so the provider is configuration:
Gemini Flash (free tier; per-project limits shown in Google AI Studio) as primary, Groq
`gpt-oss-120b` (~1,000 requests and ~200K tokens per day, Sept 2026) as secondary. The real limit
is **tokens per day, not requests**: at an estimated ~1,300 tokens per call, one dev run (≈271
calls, more after Phase 5's new cases) is ~350K–550K tokens — more than one Groq day. Check your
AI Studio quota before planning runs, and measure real token counts on the first 20 calls.

**Temperature 0, and still not deterministic.** Providers batch requests and run on
non-deterministic GPU kernels, so the same prompt can return different JSON on different days.
Phase 5 measures this directly (run-to-run flip rate) instead of assuming it away.

**Cache every response.** Key = hash(model, prompt version, rendered input). A re-run of an
unchanged case costs nothing and returns the same answer, which makes development reproducible and
saves quota. Phase 5 turns the cache **off** when it measures run-to-run consistency.

**Retries.** Rate-limit (429) and server errors: exponential backoff, at most 5 tries, then the
case is recorded as `llm_unreliable` (fail safe, §3 Decision 6) and the batch continues.

**Ambiguous acronym groups.** Blocking looks up acronyms of 3+ characters only (2-letter ones such
as `AS`, shared by 100 entities, are out of scope by design), so the largest tie group is `FYY7`,
62 candidates. The precheck usually clears most on place of registration; if more than 20 survive,
send them in chunks of 20 (several calls, one customer) rather than one enormous prompt.

---

## 7. Carried forward from Phase 3: nicknames

Deferred from Phase 3 and measured there (`phase3_blocking/blocking.md`, "Known limitation: nicknames"):
nickname + common surname records were missed by blocking 4 times in 20 (`Pepe Lopez`, `Paco
Gonzalez`). The adjudicator cannot judge a candidate blocking never shortlisted, so the fix is
in both places:

1. `reference_tables.py`: `NICKNAME_GROUPS`, hand-curated for this list's population — English
   (Bill/Will/William, Jack/John, Bob/Robert), Spanish (Pepe/José, Paco/Francisco, Chucho/Jesús,
   Lupe/Guadalupe), Russian (Sasha/Aleksandr, Dima/Dmitry, Volodya/Vladimir, Misha/Mikhail,
   Kolya/Nikolai, Zhenya/Yevgeniy).
2. `blocking.py`: query expansion in generation (`Pepe Lopez` also searches `Jose Lopez`), and
   `token_sim()` returns 0.95 for a nickname pair. Re-run `eval_blocking.py` and confirm dev
   recall stays 1.000 and the nickname cases are now shortlisted.
3. The adjudicator prompt receives matching nickname pairs as facts (§6).
4. The benchmark gets a `nickname` perturbation (Phase 5 §3).

---

## 8. Files

```
common/reference_tables.py      + NICKNAME_GROUPS, COUNTRY_ALIASES
common/names.py                 + normalize_country()
phase3_blocking/blocking.py     + nickname expansion and nickname-aware token_sim
phase4_adjudication/
    __init__.py
    schemas.py              Pydantic: NameVerdict, NameReason, CandidateJudgement, Adjudication
    profiles.py             fetch a candidate's listed attributes from Postgres
    rules.py                precheck() and decide() — pure functions, no I/O
    llm.py                  get_llm(role), response cache, retries, token accounting
    prompts/adjudicate_v1.md
    graph.py                CaseState, nodes, edges, compile(checkpointer)
    run_adjudication.py     every query in a split → benchmark/adjudications_<split>.jsonl
                            (run: uv run python -m phase4_adjudication.run_adjudication --split dev)
tests/                      (project root, so pytest finds every phase)
    test_rules.py           precheck/decide on hand-written fixtures; decide == benchmark labeller
    test_validate.py        grounding checks reject invented evidence
    test_graph.py           whole graph with a fake LLM returning recorded responses
```

New dependencies: `langgraph`, `langgraph-checkpoint-postgres`, `langchain-core`,
`langchain-google-genai`, `langchain-groq`, `pycountry`, `pytest`.

**Output record** (`adjudications_<split>.jsonl`, one line per query):

```json
{"query_id": "P-26084-transliteration",
 "disposition": "true_match", "matched_uids": [26084],
 "candidates": [
   {"uid": 26084, "outcome": "true_match", "decided_by": "llm+rules",
    "name_verdict": "match", "name_reason": "transliteration",
    "evidence": {"customer": "Waleed Abdalla M. Alsehri", "listed": "Waleed Abdullah M. ALSEHRI"},
    "yob": "agree", "location": "agree", "flags": []},
   {"uid": 12146, "outcome": "false_positive", "decided_by": "rules:year_of_birth", "flags": []}],
 "audit": {"model": "gemini-…", "prompt_sha": "…", "attempts": 1, "cache_hit": false,
           "input_tokens": 1180, "output_tokens": 240, "latency_ms": 2100}}
```

The analyst-facing **disposition note** is rendered from these fields by a template, not written
by the LLM — a regulatory record should contain no free prose the model invented.

---

## 9. Build order (~3–4 days)

| day | build | done when |
|---|---|---|
| 1 | Phase 5 §3 benchmark additions; nickname table + blocking changes; re-run blocking eval | dev shortlist recall still 1.000, nickname cases shortlisted |
| 1 | `profiles.py`, `rules.py` (precheck + decide), `test_rules.py` | precheck clears ≈84% of dev pairs, 0 true pairs cleared; decide agrees with the labeller on every fixture |
| 2 | `schemas.py`, prompt v1, `llm.py` (cache, retries, token log); 20 hand-picked cases by hand | 20/20 parse and validate; token cost per call measured |
| 3 | `graph.py`, `validate`, fail-safe path, `test_graph.py` with a fake LLM; `run_adjudication.py` | full dev run completes, every query has a disposition, failures land as `llm_unreliable` |
| 4 | Phase 5 evaluation on dev; read the errors; one round of prompt fixes (v2) | Phase 5 dev report produced; golden set passes |

Test split: untouched until Phase 5 §9.

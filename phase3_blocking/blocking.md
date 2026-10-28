# Blocking — how it is implemented (Phase 3)

**Job:** for each customer record, return the ~10 SDN entries worth comparing seriously, in
milliseconds, with no LLM. The adjudicator (Phase 4) only ever sees this shortlist.

**The one rule:** blocking is tuned for **recall**, not precision. If the true entry is not in
the shortlist, no downstream model can recover it. Wrong candidates are cheap — the adjudicator
clears them. Missing the right one is a missed sanctions hit.

Every number in this document was measured on this repo's real data (`sdn.xml`, 23 Sep 2026
publication, and `benchmark/queries.jsonl`), not made up.

---

## The pipeline

```
0. NAME INDEX (built once)                              build_name_index.py
   sdn_entries + sdn_aliases  →  sdn_names, 44,013 rows, 5 indexes

customer name
   │
   ├─ normalize()                                       names.py (same as the benchmark)
   │
   ├─ 1. CANDIDATE GENERATION   — cheap, generous      blocking.py  GENERATE_SQL
   │      a. trigram similarity        typos, transliteration, word order
   │      b. phonetic key              spelling variants that sound alike
   │      c. acronym lookup (entities) "SCD" — nothing else can find it (3+ characters)
   │      → union, typically 5–50 entries
   │
   ├─ 2. RESCORING              — precise, in Python   blocking.py  individual_score /
   │      individuals: per-token Jaro-Winkler (Monge-Elkan)          entity_score
   │      entities:    IDF-weighted token-set similarity
   │      → one score per SDN entry (best of its primary name and aliases)
   │
   └─ 3. top k = 10 (+ ambiguous acronym ties)         blocking.py  Blocker.block
          every query in a split  →  candidates_<split>.jsonl      run_blocking.py
          measured against the answer key                          eval_blocking.py
```

[Which file does what](#which-file-does-what) maps every step to its file and function.

Two stages because the jobs conflict: generation must find candidates among **all** names
cheaply (so it uses an index and a crude measure), rescoring must be **accurate** (so it uses an
expensive measure, but only on a few dozen rows). At today's 44,013 names a brute-force scan
would also work — see [How it scales](#how-it-scales) for when the index starts to matter.

---

## Step 0 — the name index

**Implemented in:** `build_name_index.py` — `build_rows()` reads `sdn_entries` and `sdn_aliases`; `natural_order()`, `phonetic_keys()` and `acronyms()` compute the columns; `DDL` and `INDEXES` create the table and its indexes. Run once after `load_sdn.py`, and again whenever the list is reloaded.

One table row per name — every primary name **and every alias** — because a customer record
often matches an a.k.a. rather than the primary name.

| | rows |
|---|---|
| primary names | 19,391 |
| aliases | 24,622 |
| **total** | **44,013** |

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE sdn_names (
    id            BIGSERIAL PRIMARY KEY,
    uid           INTEGER NOT NULL REFERENCES sdn_entries(uid),
    entry_type    TEXT NOT NULL,          -- individual / entity / vessel / aircraft
    name_kind     TEXT NOT NULL,          -- primary / alias
    is_weak       BOOLEAN NOT NULL,       -- OFAC's own "weak alias" flag
    display       TEXT NOT NULL,          -- natural order: "Vadim Yevgenyevich Dengin"
    name_norm     TEXT NOT NULL,          -- normalize(display)
    stem          TEXT,                   -- entities: legal form stripped
    phonetic_keys TEXT[] NOT NULL,        -- metaphone of each token, len >= 3
    acronyms      TEXT[]                  -- entities: initials under 3 stripping rules, 3+ chars
);
CREATE INDEX sdn_names_trgm  ON sdn_names USING gin (name_norm gin_trgm_ops);
CREATE INDEX sdn_names_phon  ON sdn_names USING gin (phonetic_keys);
CREATE INDEX sdn_names_acro  ON sdn_names USING gin (acronyms);
```

**Store individuals in natural order.** OFAC writes `DENGIN, Vadim Yevgenyevich`; customer
records say `Vadim Yevgenyevich Dengin`. Measured on a typo query `Vadim Yevgendevich Dengin`:

| compared against | Jaro-Winkler |
|---|---|
| OFAC form `DENGIN, Vadim Yevgenyevich` | 0.802 |
| natural order `Vadim Yevgenyevich Dengin` | **0.959** |

Same person, same typo, 0.16 apart purely because of word order.

**GIN index**, briefly: an inverted index — for every trigram (or phonetic key) it stores the
list of rows containing it, exactly like the index at the back of a book. "Which names contain
`has`?" becomes a lookup, not a scan of 44,013 rows.

---

## Step 1a — trigram similarity (`pg_trgm`)

**Implemented in:** `blocking.py`, first branch of `GENERATE_SQL`. The `%` threshold is set once per connection in `Blocker.__init__`.

A trigram is every 3-character window of a word, padded with two spaces in front and one behind.
Similarity is shared trigrams divided by all distinct trigrams (Jaccard).

Worked example, `Hassen` vs `Hassan`:

```
Hassen:  '  h'  ' ha'  'has'  'ass'  'sse'  'sen'  'en '
Hassan:  '  h'  ' ha'  'has'  'ass'  'ssa'  'san'  'an '
shared 4, union 10   →   similarity = 4/10 = 0.40
```

Two properties make it right for candidate generation:

- **Typo-tolerant.** One wrong letter only breaks the ~3 trigrams that contain it.
- **Order-insensitive.** Trigrams are a bag, so `Yazbeck Ahmad Mohamad` vs
  `Ahmad Mohamad Yazbeck` scores **1.000** — while whole-string Jaro-Winkler gives it **0.676**.

Query, with the threshold at 0.3:

```sql
SET pg_trgm.similarity_threshold = 0.3;
SELECT uid, display, similarity(name_norm, %(q)s) AS trgm
FROM sdn_names
WHERE name_norm % %(q)s                      -- "%" = similar above threshold; uses the GIN index
  AND entry_type = ANY(%(types)s)            -- individual → individual;
                                             -- entity → entity, vessel, aircraft
ORDER BY trgm DESC
LIMIT 50;
```

Measured: `Vadim Yevgendevich Dengin` → 5 name rows pass, true entry first (trgm 0.821).
`Hassen Mofakhami Shahrestani` → 3 rows pass, all belonging to the true entry.

## Step 1b — phonetic keys

**Implemented in:** stored by `build_name_index.phonetic_keys()`; at query time `Blocker.block()` computes the query's keys and the second branch of `GENERATE_SQL` looks them up.

Metaphone reduces a word to an approximation of how it sounds. Measured (jellyfish):

| spellings | key |
|---|---|
| MOHAMMED, MOHAMED, MUHAMMAD | `MHMT` |
| HASSAN, HASSEN, HASAN | `HSN` |
| SHAHRESTANI / SHAHRASHTANI | `XRSTN` / `XRXTN` — different |

So it catches romanisation variants that trigram similarity can score low, and misses others —
which is why it is a *second* generator, not a replacement. Lookup: `phonetic_keys && %(q_keys)s`
(array overlap, GIN-indexed). Only tokens of 3+ characters get keys, so `AL` and `EL` (which would
match thousands of rows) are skipped. Three-letter particles — `BIN`, `IBN`, `BEN`, `ABU` — *do* get
keys (`BN`, `IBN`, `AB`); harmless, because each generator returns only its 50 most similar rows.

## Step 1c — acronym lookup (entities only)

**Implemented in:** stored by `build_name_index.acronyms()`; at query time `Blocker._query_acronym_inputs()` decides whether the query might be an acronym (or expand to one), and the third branch of `GENERATE_SQL` looks it up. Tie groups are appended in `Blocker.block()`.

Measured on a real benchmark query:

| `SCD` vs `SINA COMPOSITE DELIJAN COMPANY` | score |
|---|---|
| Jaro-Winkler | 0.730 |
| IDF-weighted token set | **0.000** |
| trigram | below 0.3 |

Every similarity measure fails, because an acronym shares no tokens and almost no characters with
its expansion. The only fix is an exact lookup. Each entity stores its initials under three
stripping rules — `CHINA 49 GROUP CO., LIMITED` → `C4GCL`, `C4GC`, `C4G` — because a customer
might be onboarded under any of them, and the lookup is `acronyms && ARRAY[query]`.

**Minimum length: 3 characters (domain-owner decision, Oct 2026).** A 2-letter acronym is a long
shot for a true match: on this list `AS` is shared by 100 entities, `TS` by 41, `AR` by 35. The
name carries almost no evidence, so the shortlist would be a tie group the adjudicator could only
escalate. So acronyms shorter than `MIN_ACRONYM_LENGTH = 3` (`common/reference_tables.py`) are
not stored in the index, never looked up for a query, and not generated by the benchmark. A
2-letter query still goes through the trigram and phonetic generators like any short name.
Effect: 718 of 19,140 distinct acronyms dropped, and 19 two-letter acronym queries removed from
the benchmark (4 dev, 15 test).

**Ambiguous acronyms (domain-owner decision).** A 3-letter acronym can still be shared:
the largest group is `FYY7`, 62 entities. All tie on name, so a 10-candidate cutoff would pick the
survivors by tie order — luck. Instead, when more than one entity shares the query's acronym,
blocking returns the **whole tie group** (up to `ACRONYM_LIMIT` = 100), each flagged
`ambiguous: true`, beyond k if necessary. The adjudicator narrows it on place of registration; an
ambiguous name is a `possible_match` at best. This is the one generator that exists for a single
perturbation type — and the benchmark is what tells you it is needed.

Note that the SQL `LIMIT` counts index **rows**, not entities, and an entity can carry the same
acronym on several aliases. Before the 3-character minimum, `AS` had 124 rows over 100 entities,
so 21 entities were silently cut. The largest group now has 62 rows; a test
(`tests/phase3_blocking/test_index_integrity.py`) fails if a list reload ever pushes a group to
the limit.

---

## Step 2 — rescoring

**Implemented in:** `blocking.py`. `Blocker.block()` fetches every name of each generated entry (`NAMES_SQL`) and scores it with `Blocker._score()`, which dispatches to `individual_score()` or `entity_score()`. Both are built on `token_sim()` (rapidfuzz Jaro-Winkler, plus the initials rule). Legal forms are stripped with `names.strip_legal_forms()`.

### Individuals: score token by token, not the whole string

Whole-string Jaro-Winkler has a known flaw, visible in real data. Query
`Hassen Mofakhami Shahrestani` against two names on the same SDN entry:

| SDN name | differences | whole-string JW |
|---|---|---|
| `Hassan Mofakhami Shahrestani` (primary) | 1 letter | **0.912** |
| `Hassan Mofakhmi Shahrashtani` (alias) | 3 spread out | **0.925** ← ranked higher |

Confirmed identical in `jellyfish` and `rapidfuzz`. Why: Jaro matches characters greedily within
a window. One early substitution (`E`→`A`) leaves an unmatched `A` that later characters pair
with, shifting every downstream match and inflating the transposition count. One edit early in a
long string costs more than three edits spread out — the opposite of what an analyst means by
"similar".

Fix — **Monge-Elkan**: for each query token, take its best Jaro-Winkler against any target token,
then average:

```
primary:  HASSEN~HASSAN 0.933   MOFAKHAMI~MOFAKHAMI 1.000   SHAHRESTANI~SHAHRESTANI 1.000  → 0.978
alias:    HASSEN~HASSAN 0.933   MOFAKHAMI~MOFAKHMI  0.978   SHAHRESTANI~SHAHRASHTANI 0.928 → 0.947
```

Correct order restored, and the score is order-insensitive by construction. Compute it in both
directions and average, so a target with extra tokens (an additional given name) is penalised.

### Entities: weight tokens by how rare they are

IDF (inverse document frequency) measures how much a token identifies a company:
`idf = ln((N+1)/(df+1)) + 1`, where `df` = how many of the N = 9,975 entity names contain it.
Measured on the list:

| token | names containing it | idf |
|---|---|---|
| COMPANY | 140 | 5.26 |
| TRADING | 495 | 4.00 |
| INTERNATIONAL | 266 | 4.62 |
| GENERAL | 121 | 5.40 |
| SINA | 12 | 7.64 |
| PROPITIOUS | 1 | 9.51 |

IDF is computed over legal-form-free **stems**, so `COMPANY` counts only where it is not a trailing
legal form (it appears in 2,685 primary entity names). Note `GENERAL` (121) is far rarer than `TRADING`
(495) — the list decides what is generic, not intuition. Similarity = idf-weight of shared tokens ÷ idf-weight of all tokens:

| pair | JW | IDF token-set |
|---|---|---|
| `Bishop Trading Ltd` vs `PROPITIOUS FOREVER TRADING CO LTD` (benchmark negative) | 0.645 | **0.228** |
| `MAROC INTERNATIONAL PVT` vs `MAROC INTERNATIONAL PVT LTD` (true, suffix dropped) | 0.970 | 0.674 → **1.000** after the fix below |

**Bug this exposed, fix in Phase 3:** the true pair scores only 0.674 because `PVT LTD` is a
*two-token* legal form and `strip_legal_suffix()` removes only the last one, leaving `PVT` on one
side. Strip legal forms repeatedly until none remain.

### One score per entry

An entry has several names. Its score is the **maximum** over its primary name and aliases.
Weak aliases are kept (dropping them costs recall) but recorded, so the adjudicator weighs them
lower.

---

## Step 3 — output and measurement

**Implemented in:** `run_blocking.py` writes `benchmark/candidates_<split>.jsonl`; `eval_blocking.py` reads it alongside `queries.jsonl` and writes `benchmark/blocking_report_<split>.json`.

`candidates.jsonl`, one line per query:

```json
{"query_id": "P-36778-typo", "candidates": [
  {"uid": 36778, "score": 0.984, "matched_name": "Vadim Yevgenyevich Dengin",
   "name_kind": "primary", "generators": ["trigram", "phonetic"]},
  {"uid": 35570, "score": 0.713, "matched_name": "Vadim Yevgenyevich Bulavinov", ...}
]}
```

Then, without any LLM:

- **Blocking recall@k** — is `source_uid` in the top k? By subject type and perturbation type.
  Exit criterion: ≥ 0.98 at k = 10 for both individuals and entities, on dev.
- **Fuzzy-only baseline** — sweep a threshold on the top score; above it = alert. This is what
  keyword/fuzzy screening tools do, and it is the "before" column of the headline result.

A hard negative shows why blocking must be generous. `Andrew Al-Libi` surfaces five different
listed `AL-LIBI` individuals (scores 0.82–0.89), including the one the negative was built from.
That is correct behaviour: the shortlist contains the collisions, and clearing them on the given
name and year of birth is the adjudicator's job — exactly the alert an analyst would otherwise
work by hand.

---

## Libraries

Use `rapidfuzz` (C++, fast; Jaro-Winkler and token functions) and `jellyfish` (Metaphone) — not
hand-written implementations. A hand-rolled Jaro-Winkler used while writing this document
differed from both libraries by 0.004 on one pair; small, but a benchmark should not depend on
it.

## Results on the dev split (measured, Oct 2026)

| | recall@1 | @3 | @5 | @10 | shortlist | n |
|---|---|---|---|---|---|---|
| individuals | 0.993 | 1.000 | 1.000 | 1.000 | **1.000** | 151 |
| entities | 0.968 | 0.968 | 0.979 | 1.000 | **1.000** | 95 |
| all | 0.984 | 0.988 | 0.992 | 1.000 | **1.000** | 246 |

Re-measured on 28 Oct 2026 after the 3-character acronym minimum (19 two-letter acronym queries
left the benchmark, 4 of them in dev). Before it, entities scored 0.929 / 0.939 / 0.950 / 0.980
(n = 99): two `AR`/`TS` tie-group entries ranked 33rd and 39th, and 97/99 = 0.9798 was in fact
just under the 0.98 criterion.

"Shortlist" counts a true entry anywhere in what blocking returned (including appended acronym
tie groups) — the true ceiling on the adjudicator. Exit criterion (≥ 0.98 at k = 10, both subject
types): **met**, 1.000 for both. 13–19 ms per query, no LLM. Test split deliberately not
evaluated yet.

Three things the dev run found, all fixed before these numbers:
- acronym queries were capped at 8 characters, missing a unique 13-letter acronym
  (`GGRASPRDRLDCV`, one entity on the whole list);
- acronym candidates were cut at k by tie order — now the full tie group is returned;
- `PVT LTD`-style two-token legal forms (multi-form stripping, `strip_legal_forms`).

Found later by the test suite (`tests/`) and resolved by the 3-character minimum: the `AS` tie
group lost 21 of its 100 entities to the row-counting `LIMIT` (see Step 1c).

**Operating points** — recall vs pairs reaching the adjudicator (dev, 996 queries):

| k | score floor | recall | pairs | queries with ≥1 candidate |
|---|---|---|---|---|
| 5 | 0.00 | 1.000 | 4,746 | 986 |
| 5 | **0.50** | **1.000** | **3,229** | **835** |
| 5 | 0.60 | 1.000 | 2,880 | 797 |
| 5 | 0.70 | 0.976 | 2,314 | 734 |
| 10 | 0.50 | 1.000 | 5,445 | 835 |

Recall holds to a floor of 0.60 and drops at 0.70, so **k = 5, floor = 0.50** keeps a 0.1
safety margin below the cliff. If Phase 4 sends all of a query's candidates to the LLM in **one
call**, the cost is the last column — 835 calls for dev, inside one day of a free tier — not the
pair count.

**Fuzzy-only baseline** (alert on every candidate above a threshold — the "before" column):

| | threshold | recall | false positives per true hit |
|---|---|---|---|
| individuals | 0.90 | 0.993 | 0.34 |
| individuals | 0.85 | 1.000 | 2.01 |
| entities | 0.90 | 0.768 | 3.14 |
| entities | 0.80 | 0.832 | 3.73 |

Entities are where name-only screening fails: at 0.90 it misses 23% of true hits *and* raises
3.1 false alerts per true one. Caveat: this baseline thresholds this project's own improved
scorers (Monge-Elkan, IDF); a plain whole-string Jaro-Winkler tool would do worse, so the
comparison is conservative.

## How it scales

Everything above was measured at 44,013 names. Screening in production runs against more than
one OFAC file — OFAC plus the UN, EU, UK and Canadian lists, and commercial PEP databases with
millions of profiles — so the question is what happens as `sdn_names` grows.

**Method.** `sdn_names` was copied into 10× and 25× tables (440,130 and 1,100,325 names) with
the same five indexes, then the three generator queries were timed on 200 dev queries each.
Measured on Postgres 16, one connection, warm cache — indicative, not a hardware benchmark.

**Caveat on the method.** Growth was simulated by duplicating every name. That makes the number
of near-identical names grow as fast as possible — 25 copies of every `Mohammed` — which is close
to a worst case. Real list growth adds diverse names, so real slowdown should be smaller.

### Is the index needed at all at today's size?

Honestly, no. Scoring a query against all 44,013 names with rapidfuzz (no index) takes
**18.4 ms**; the trigram index lookup takes **8.4 ms**. Both are fine. The index is a scale
decision: a brute-force scan does work proportional to list size on every query, so at a million
names it would be ~0.4 s per customer — about 117 hours for a million-customer book. Kept
deliberately, at the cost of one Postgres extension.

### Do five indexes cause problems as the table grows? No.

Indexes cost on **writes** and **disk**. This table is written rarely (OFAC publishes small
updates a few times a week) and read constantly — the workload indexes are best at.

| | 44k (1×) | 440k (10×) | 1.1M (25×) |
|---|---|---|---|
| trigram GIN | 3.6 MB | 21.1 MB | 46.8 MB |
| phonetic GIN | 1.4 MB | 3.9 MB | 7.5 MB |
| acronym GIN | 1.3 MB | 2.3 MB | 3.6 MB |
| uid btree | 0.7 MB | 7.1 MB | 17.8 MB |
| entry_type btree | 0.3 MB | 3.1 MB | 7.7 MB |
| **all five** | **7.3 MB** | **37.5 MB** | **83.4 MB** |
| rebuild all five from scratch | — | 3.7 s | 9.1 s |

Inserting a 1,000-name delta into the 1.1M-name table with all five indexes maintained: **38 ms**.

Every index earns its place. Scans during 1,000 real blocking queries (1×):

| index | scans | used for |
|---|---|---|
| `uid` | 32,217 | fetching every name of every candidate entry for rescoring |
| `trgm` | 1,518 | trigram generator |
| `phon` | 1,000 | phonetic generator |
| `acro` | 1,000 | acronym generator |
| `entry_type` | 294 | type filter, when the planner judges it selective |
| primary key | 0 | created automatically; not used by blocking |

### What does degrade: lookups on common names

| generator | p50 at 1× → 10× → 25× | p95 at 1× → 10× → 25× |
|---|---|---|
| trigram | 4.1 → 19.0 → 49.5 ms | 25.8 → 164.9 → **391.7 ms** |
| phonetic | 1.5 → 5.5 → 14.6 ms | 30.3 → 164.5 → **255.7 ms** |
| acronym | 0.4 → 0.5 → 0.6 ms | 0.8 → 0.9 → 1.1 ms |

The cause is the query shape, not the number of indexes. Trigram and phonetic generators run
`... ORDER BY similarity(...) DESC LIMIT 50`. The index narrows the table to names that share
trigrams or a phonetic key with the query — but Postgres must then **score every one of those
rows** before it can return the best 50. A rare name matches a handful of rows; a common one
(`Mohammed`, any `AL-` surname) matches a large share of the list, and that share grows with the
list. Hence the median rises modestly and the p95 — the common names — rises steeply. Acronym
lookups stay flat because they are exact matches on small groups.

### The "standard" fix that did not work: GiST nearest-neighbour search

pg_trgm also supports a GiST index that can walk names nearest-first (`ORDER BY name_norm <-> q
LIMIT 50`) and stop after 50, in principle avoiding "score everything that matches". Measured at
1.1M names, GiST alone vs GIN alone:

| trigram lookup, 1.1M names | p50 | p95 | index size |
|---|---|---|---|
| GIN, filter then sort (current) | **42.7 ms** | 414.2 ms | **47 MB** |
| GiST, nearest-first | 361.0 ms | 480.7 ms | 157 MB |

GiST stores each name's trigrams as a lossy signature, so the nearest-first walk still rechecks a
large part of the table — 8× slower at the median and 3× larger. On short strings like names,
GIN wins. (A first version of this comparison put both indexes on one table and forced planner
settings; the planner then used the wrong index and the numbers were meaningless. The figures
above isolate one index at a time.)

### Levers if the list grows toward a million names

None is needed at 44k names. In order of effort — each must be re-checked with
`eval_blocking.py`, because every one of them trades against recall:

1. **Raise the trigram threshold** (0.3 → 0.4). Fewer rows pass the filter, so fewer get scored.
2. **Skip very common phonetic keys**, as IDF discounts common words. `MHMT` (every Mohammed)
   adds work and tells blocking almost nothing.
3. **Parallelise across customers.** Screening a customer book is independent per customer;
   N workers cut wall time ~N× without touching per-query latency.
4. **Split the table by subject type** (individuals vs entities/vessels/aircraft), halving the
   rows any one query's index must cover.

## Known limitation: nicknames (deferred to Phase 4)

Blocking has no notion that Bill = William, Jack = John, Pepe = José or Sasha = Aleksandr. The
only name-equivalence data is the transliteration table (spelling variants, not nicknames), and
the benchmark has no nickname perturbation — so the dev recall of 1.000 says nothing about them.
Measured against real listed individuals:

| customer record | shortlisted | ranked 1st |
|---|---|---|
| nickname + rest of name (`Jack Yvan Mejia Magnani`) | 24/24 | 23/24 |
| full given name + one surname (`Jose Lopez`) | 19/20 | 16/20 |
| **nickname + one surname** (`Pepe Lopez`) | **16/20** | **7/20** |

With middle names, patronymics or a second surname present, those tokens match exactly and carry
the entry. With only nickname + surname, the surname is the sole evidence, and when many listed
people share it the true entry is outranked: `Pepe Lopez` (40 listed Lopez names), `Pepe
Rodriguez` (39), `Paco Gonzalez` (29) and `Paco Flores` (22) were missed outright. Spanish
nicknames fail worst — Pepe/José and Paco/Francisco share almost no letters or sounds — and
this list is dominated by Spanish, Russian and Arabic names, so English nicknames are the least
important case.

**Planned for Phase 4:**

1. A hand-curated nickname table in `reference_tables.py` (English, Spanish, Russian
   diminutives), built like the transliteration groups.
2. Use it in **blocking as well as adjudication**. The adjudicator cannot recover an entry that
   blocking never shortlisted, so the four misses above need query expansion in generation
   (`Pepe Lopez` → also search `Jose Lopez`) and nickname-aware `token_sim()` in rescoring.
3. Pass the nickname match to the adjudicator as evidence — under the match rule an adjudicator
   that does not know Pepe is José would call it a `false_positive`, i.e. a missed hit.
4. A `nickname` perturbation in the benchmark so the effect is measured, not assumed.

## Which file does what

### Run order

```
# from the project root
uv run python -m phase1_data.load_sdn --reset              # once: data/sdn.xml → sdn_entries / sdn_aliases / sdn_features / sdn_addresses
uv run python -m phase2_benchmark.make_benchmark           # once: → benchmark/queries.jsonl (with dev/test split)
uv run python -m phase3_blocking.build_name_index          # Step 0: → sdn_names + 5 indexes   (~4 s)
uv run python -m phase3_blocking.run_blocking --split dev  # Steps 1–3 for every dev query → candidates_dev.jsonl  (~13 s)
uv run python -m phase3_blocking.eval_blocking --split dev # recall, operating points, baseline → blocking_report_dev.json
```

`--split test` is run once, at the very end of the project.

### Step → file → function

| step | file | function / object |
|---|---|---|
| 0. name index | `build_name_index.py` | `build_rows()`, `natural_order()`, `phonetic_keys()`, `acronyms()`, `DDL`, `INDEXES` |
| normalisation | `common/names.py` | `normalize()`, `strip_legal_forms()` (blocking), `strip_legal_suffix()` (benchmark) |
| 1a. trigram generator | `blocking.py` | `GENERATE_SQL` (branch 1); threshold set in `Blocker.__init__` |
| 1b. phonetic generator | `blocking.py` | `GENERATE_SQL` (branch 2); query keys in `Blocker.block()` |
| 1c. acronym generator | `blocking.py` | `GENERATE_SQL` (branch 3); `Blocker._query_acronym_inputs()` |
| 2. rescoring | `blocking.py` | `NAMES_SQL`, `Blocker._score()`, `individual_score()`, `entity_score()`, `IDF`, `token_sim()` |
| 3. shortlist | `blocking.py` | `Blocker.block()` → `list[Candidate]` (top k + ambiguous acronym ties) |
| 3. batch run | `run_blocking.py` | `main()` |
| 3. measurement | `eval_blocking.py` | `main()`: recall@k, operating points, fuzzy-only baseline, misses |

### Supporting files

| file | role in Phase 3 |
|---|---|
| `common/reference_tables.py` | legal-form suffixes and `LEADING_LEGAL_FORMS` used to build stems; generic corporate tokens; `MIN_ACRONYM_LENGTH` (3) |
| `phase2_benchmark/split.py` | `load_queries(split=...)` — what `run_blocking.py` and `eval_blocking.py` read |
| `phase1_data/schema.sql`, `phase1_data/load_sdn.py` | the source tables `build_name_index.py` reads from (Phase 1) |
| `phase2_benchmark/make_benchmark.py`, `phase2_benchmark/perturbations.py` | produce the answer key `eval_blocking.py` scores against (Phase 2) |

### Inputs and outputs

| file | written by | read by |
|---|---|---|
| table `sdn_names` | `build_name_index.py` | `blocking.py` |
| `benchmark/queries.jsonl` | `make_benchmark.py` | `run_blocking.py` (names only), `eval_blocking.py` (labels) |
| `benchmark/candidates_<split>.jsonl` | `run_blocking.py` | `eval_blocking.py`, Phase 4 |
| `benchmark/blocking_report_<split>.json` | `eval_blocking.py` | you; the README |

`run_blocking.py` passes only `query_name` and `subject_type` to the blocker. The labels in
`queries.jsonl` (`source_uid`, `is_true_match`, `expected_outcome`) are read only by
`eval_blocking.py`, so blocking can never see the answer key.

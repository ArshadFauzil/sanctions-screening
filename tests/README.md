# Tests

Phase 3 (blocking) has 1,086 test cases. Property tests add about 5,500 randomly generated inputs on
top of those. The slow sweeps run about 10,600 real blocking queries against the full OFAC list.

## Running

```bash
uv add --dev pytest hypothesis              # once

uv run pytest                               # fast suite: unit + database tests, ~20 s
uv run pytest -m slow                       # exhaustive sweeps over the real list, ~3 min
uv run pytest -m "slow or not slow"         # everything
uv run pytest tests/phase3_blocking/test_scoring.py -v      # one file, one line per test
SSA_EXHAUSTIVE=1 uv run pytest -m slow      # every name and every acronym, ~45 min
```

Database tests need Postgres with the Phase 1 tables and `sdn_names` (`DATABASE_URL` in `.env`).
Without them, those tests are **skipped**, not failed, so the unit tests still run anywhere.

A healthy run ends like this:

```
1068 passed, 14 deselected, 4 xfailed      (fast; the 14 deselected are the slow ones)
9 passed, 1072 deselected, 5 xfailed       (-m slow)
```

`xfailed` means "expected to fail". Each one is a known gap with its reason attached (see below).

## Which technique is tested where

| blocking step | unit tests (no DB) | against real Postgres | exhaustive sweep (`-m slow`) |
|---|---|---|---|
| normalisation and legal forms | `test_normalization.py`: every legal suffix × every leading form × 5 base names; random-input properties | — | — |
| step 0: name index | `test_index_columns.py`: natural order, phonetic keys, acronyms, `build_rows` | `test_index_integrity.py`: all 44,013 rows rebuilt and compared column by column | — |
| 1a. trigram | trigram model checked against the documented 0.40 | `test_generators_db.py`: threshold, ordering, limit, type filter, threshold subsets, variants found by trigram alone; Python model = pg_trgm on 200 random pairs | — |
| 1b. phonetic | all 168 transliteration pairs: shared key or not | rows share a key, no-key queries return nothing, MOHAMMED spellings connected | every transliteration sibling substituted into real names (839 queries) |
| 1c. acronym (3+ characters) | query acronym inputs, the 3-character minimum, the legal-form gap | each documented acronym, 2-letter queries never looked up, uid ordering, row limit vs largest group, full name → listed acronym | all 2,325 shared acronyms returned whole; 1,000 unique acronyms (all with `SSA_EXHAUSTIVE`) |
| 2. rescoring | `test_scoring.py`: token_sim, Monge-Elkan, IDF, entity_score; examples plus symmetry, bounds and order-invariance properties | IDF table on the real list | — |
| 3. merge, rank, k, tie groups | `test_blocker_logic.py` with a fake database, plus 400 random result sets | union = branches, determinism, normalisation invariance, k on real data | every name finds itself at 1.0 (2,000 names; all 44,013 with `SSA_EXHAUSTIVE`) |
| end to end | — | `test_documented_cases.py`: every worked example in `blocking.md` | every perturbation (1,040 queries) and every ordered pair of perturbations (50 pairs, 2,300 queries) on entries outside the benchmark; nicknames |
| run / eval scripts | `test_run_and_eval.py`: a hand-computed answer key | — | the full dev split compared with the `blocking.md` results |

## Concepts used (brief)

- **Fixture**: a function pytest calls for you. A test lists its name as a parameter (for example
  `blocker`) and pytest passes in the result. `scope="session"` creates it once per run, so all
  tests share one database connection. Fixtures live in `conftest.py`.
- **`@pytest.mark.parametrize`**: runs one test body once per row of inputs. Stacking two decorators
  gives the cross product. That is how 51 legal suffixes × 5 base names become 255 tests from one
  function.
- **Property-based testing (`hypothesis`)**: instead of picking inputs yourself, you state a rule that
  must hold for every input, for example "`individual_score(a, b) == individual_score(b, a)`".
  Hypothesis then generates hundreds of inputs that try to break it. When one does, it shrinks the
  input to the smallest one that still fails.
- **Test double (`helpers.FakeConnection`)**: a stand-in for the database that returns scripted
  rows. It lets the ranking and tie-group logic in `Blocker.block()` be tested exactly, in
  milliseconds, without Postgres.
- **`xfail(strict=True)`**: "this test is expected to fail because of a known gap". If the gap is
  ever fixed, the test passes, and strict mode turns that unexpected pass into a failure. That
  forces you to remove the marker, so a fixed gap can't silently come back.
- **Ratchet** (the slow sweeps): one test runs thousands of queries and compares its misses with a
  pinned list of known misses. A new miss fails the test as a regression. A known miss that is now
  found also fails, telling you to delete it from the list. The list can only shrink. Pair recall
  works the same way against `baselines/perturbation_pairs.json`; after a deliberate improvement,
  run `SSA_UPDATE_BASELINES=1 uv run pytest -m slow` to lock the new values in.

## Findings

The tests found these. Each one is pinned by a test, so a fix shows up immediately.

| # | finding | severity | pinned by |
|---|---|---|---|
| 1 | **The acronym tie group for `AS` was incomplete.** The acronym SQL uses `LIMIT 100`, which counts *rows*, not entities: `AS` had 124 rows across 100 entities, so 21 were silently dropped. **Resolved 2026-10-28** by the 3-character acronym minimum: 2-letter acronyms are no longer looked up, and the largest remaining group (FYY7) has 62 rows. The row-vs-entity limit is still latent, so a test fails if a list reload ever pushes a group to the limit. | resolved; guarded | `test_index_integrity::test_acronym_group_sizes`, `test_documented_cases::test_two_letter_acronyms_are_not_looked_up` |
| 2 | **An acronym with a legal form is not tried as an acronym.** Queries like `SCD LLC` or `OOO SCD` are two tokens, so the acronym generator skips them. The acronym + legal-suffix pair is found only 9/106 times. Fix: also treat `strip_legal_forms(query)` as an acronym when it is one token. | medium | `test_blocker_logic::test_acronym_with_legal_form_is_tried_as_acronym` |
| 3 | **One-word weak aliases outrank true matches.** 674 individual names are a single token, 670 of them weak aliases. The alias→query direction of symmetric Monge-Elkan scores 1.0, so `HASSAN` (uid 8187) scores 0.948 against *any* query containing Hassan. It outranks the true entry in `Hassan Shahrestani`. Recall held in every sweep, but these aliases use up top-k slots and pass the 0.5 floor, so they also become LLM pairs in Phase 4. | medium: precision and cost | `test_generators_db::...ranks_it_first[Hassan Shahrestani]`, `test_single_token_weak_alias_inflation` |
| 4 | **Ł, Ø and Đ become a space.** These letters have no NFKD decomposition, so `Søren` → `S REN`, which splits the token. Fix: a small fold table (Ł→L, Ø→O, Đ→D) in `normalize()`. | low | `test_normalization::test_letters_without_nfkd_decomposition` |
| 5 | **`&` adds an A to acronyms.** `M&S TRADING` → `MAST`, so a customer onboarded as `MST` is not found. | low | `test_index_columns::test_ampersand_contributes_an_A` |
| 6 | **BIN, IBN, BEN and ABU do get phonetic keys.** `blocking.md` said they were skipped (now corrected), but they have 3 letters. This is harmless (LIMIT 50, best similarity first), but every "bin" name hits a very common key. | info | `test_index_columns::test_phonetic_keys` |
| 7 | **Names with two commas are split only at the first comma.** `KNOWLES, JR., Samuel` → `JR., Samuel KNOWLES` (5 rows). Rescoring is order-free, so it costs nothing. | info | `test_index_columns::test_natural_order_multi_comma` |
| 8 | **Known misses from the sweeps (names only, so these are expected limits):** common two-token names (`Song Choe`, `MUHAMMAD OMAR`), MEHMET↔MOHAMMED (no shared key, low trigram), the two-token sibling `ABD AL`, and YOUSEF↔JOSEPH on a one-word name. 43 of the 168 transliteration pairs share neither a phonetic key nor a 0.3 trigram score as single tokens. | info | `KNOWN_*` lists in `test_exhaustive_db.py`, `KNOWN_UNREACHABLE_TOKEN_PAIRS` |

### Corrections made to `blocking.md`

These numbers in `blocking.md` predated later code changes. The tests exposed them, and the
document was corrected on 2026-10-28:

| was | now |
|---|---|
| IDF: COMPANY in 1,920 names, idf 2.65 | 140 / 5.26. IDF is built over *stems*, and COMPANY is a legal suffix |
| `Bishop Trading Ltd` vs `PROPITIOUS…` IDF score 0.108 | 0.228 (still well below the 0.5 floor) |
| `Andrew Al-Libi` scores 0.67–0.73 | 0.82–0.89 |
| natural-order JW 0.955; word-order JW 0.665; alias ME 0.946 | 0.959; 0.676; 0.947 |
| entities recall@10 = 0.980, "exit criterion met" | it was 97/99 = 0.9798, just under 0.98. With the 3-character acronym minimum it is now 95/95 = 1.000 |
| "AL and BIN" get no phonetic key | BIN does (finding 6) |

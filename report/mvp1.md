# MVP1 — environment, verifier, and a working baseline table

MVP1 asked one question: does the pipeline work end to end? It does. Every component of the
full system exists at reduced strength, one path runs from question to verified answer, and
the whole sweep replays offline for nothing.

**None of the numbers below are results.** They come from 20 questions about one company,
answered by the cheapest available model. They demonstrate that the measurement apparatus
works, not that anything is true about agents. The scale that would make them results is
MVP2.2's job.

## Baselines

20 questions, AAPL, ground truth from XBRL. Analyst and judge both Haiku 4.5.

| policy | D1 accuracy | D2 unsupported | D3 step pass | E1 $/correct | E2 p95 | mean steps | list cost |
|---|---|---|---|---|---|---|---|
| **B1** single-shot RAG | 35% | 16% | 20% | $0.0132 | 3,608 ms | 2.0 | $0.0926 |
| **B2** tools | **85%** | **0%** | **81%** | **$0.0128** | 12,290 ms | 2.9 | $0.2178 |

The comparison is deliberately fair to B1: same corpus, same as-of horizon, same model, same
verifier. The only thing it lacks is the ability to look again after seeing what came back.

Two things worth drawing out.

**B2 costs 2.4× more in total and is still cheaper per correct answer** — $0.0128 against
$0.0132 — because B1 is wrong about two-thirds of the time. A per-token comparison would have
picked the worse system. This is the entire argument for measuring E1 rather than spend. Note
the margin is thin: on this split the two policies cost almost the same per correct answer.

**B2's 15% failures are all the same failure.** Three of 20 runs ended in prose without ever
calling `finish`. Not wrong answers — no answer. Every one was on the P4 path, where the agent
holds all five tools, and the rollouts below show the same pattern far more starkly.

**B2 is slower, by a lot.** 12.3 s at p95 against 3.6 s. Accuracy was bought with latency, and
on this evidence the trade is worth it; on a latency-sensitive product it might not be.

**D3 is the interesting column.** Only 81% of B2's steps verify — roughly a fifth are wasted
or wrong, including inside trajectories that end well. Outcome filtering keeps all of them;
step filtering keeps 81%. That divergence is the mechanism H1 proposes to exploit, and the
rollout section below measures it directly.

## A1 — lookahead leak rate

24 horizons swept across the corpus. Leakage is judged against acceptance time regardless of
how each strategy filtered.

| filter | filings leaked | pooled | horizons hit | worst horizon | facts pooled |
|---|---|---|---|---|---|
| no filter | 48/96 | 50.00% | 21/24 | 100% | 2.18% |
| `period_end` (naive) | 10/58 | **17.24%** | **10/24** | 100% | 1.34% |
| `accepted_at` (FinanceVault) | 0/48 | **0.00%** | **0/24** | 0.00% | **0.00%** |

The `period_end` row is the comparison that counts. It is not a strawman — filtering on the
fiscal period a document *describes* is the obvious, careful-looking choice, and it is wrong.

Concretely: a 10-Q for the period ending 2025-12-27 was visible at an as-of of 2025-12-27,
**34 days before it was accepted** on 2026-01-30. A month of future knowledge, from a filter
that looks correct.

Three qualifications:

- **The filings corpus is 4 documents.** 96 = 4 filings × 24 horizons. The rate is real but
  thin. The 25,135-fact figures are better powered and lower — 1.34% — simply because most
  facts are old enough that no horizon reaches them.
- **"Worst horizon 100%" is less dramatic than it reads.** It means every filing returned at
  that horizon was unpublished, at a point early in the corpus where only one or two pass the
  filter at all.
- **The leak is concentrated, not uniform.** It occurs inside reporting lags — which is
  exactly when someone asks "what did they just report?". The concentration is the argument,
  more than the pooled rate.

## B2 — seeded-error detection

Trajectories the verifier already passed, with a known defect injected. Ground truth is
constructed, so there is no ambiguity about whether the seeded step is wrong.

| error class | signal | seeded | caught | recall |
|---|---|---|---|---|
| off_by_1000 | s3 | 20 | 20 | **100%** |
| wrong_scale_word | s3 | 19 | 19 | **100%** |
| wrong_period | s3 | 20 | 20 | **100%** |
| fabricated_value | s3 | 20 | 20 | **100%** |
| fabricated_citation | s2 | 20 | 19 | 95% |
| hallucinated_tool | s1 | 20 | 20 | **100%** |
| malformed_args | s1 | 15 | 15 | **100%** |

**Control false-positive rate: 0%** — none of the 20 clean trajectories was flagged. Recall
without this number would be meaningless; a verifier that rejects everything scores 100%.

Recall is reported per class and never pooled, so a weak class cannot hide behind strong ones.

### What this run changed

The first pass scored `wrong_scale_word` at 79%, and the cause was not a detection gap. `s3`
*detected* every corruption — it named them "off by 1000x" — but an answer restating a figure
both correctly and with the error scored 0.6, and 0.6 cleared the 0.5 pass threshold. A
detected fabrication was being averaged against correct claims and passing.

`s3` now carries a threshold of 1.0: "does every figure trace to evidence" is a conjunction,
not an average. The score stays a fraction for ranking and diagnosis; passing requires all of
them. The baseline table above is unaffected — in real trajectories `s3` was always 1.0 or
0.0, never partial — so the stricter rule bites only on the corruption it was built to catch.

## C1 — trajectory generation

50 trajectories: 10 questions under 5 variants. Variants pin the execution path, because with
sampling parameters removed and every call journalled, the same question asked twice returns
the identical cached trajectory — diversity has to come from the input.

| | rate |
|---|---|
| C1 accept rate (every step verifies) | **26%** |
| outcome accept rate (answer correct) | **60%** |
| **correct but flawed** | **34%** |
| step pass rate | 62% |

| variant | tools | accept | correct |
|---|---|---|---|
| facts_only | XBRL lookup | 60% | 90% |
| compute | SQL + Python | 40% | 70% |
| text_only | passage retrieval | 30% | 80% |
| full | all five | **0%** | **30%** |
| tight | all five, 3 steps | **0%** | **30%** |

Exported: 13 accepted, 10 repaired, 20 hard-negative, 23 step-filtered.

**The divergence is the point.** Outcome filtering keeps 60% of trajectories; step filtering
keeps 26%. The 34% in between — right answer, at least one bad step — is exactly the data the
two conditions disagree about, and it bounds how large an effect H1 could possibly detect. A
divergence near zero would have meant the experiment was unrunnable; 34% means there is
something to measure.

**The `full` and `tight` rows are a finding, not noise.** Giving the agent every tool makes it
*worse*: 0% of those trajectories pass step filtering and only 30% reach a correct answer,
against 90% when restricted to XBRL lookup. The failure mode is consistent — it ends in prose
without calling `finish`. Constraining the toolset is doing more work here than the agent's
own routing, which is an argument for the router that MVP2.2 should test properly rather than
assume.

Two caveats on these numbers:

- **Variant diversity is engineered, not sampled.** Trajectories differ in approach, not in
  the model's moment-to-moment choices, so they under-represent the near-misses a stochastic
  policy produces. Forcing `compute` on a simple lookup also manufactures awkwardness a
  free-running policy would avoid, which inflates the divergence somewhat.
- **Hard negatives are constructed, not collected.** Organic near-misses are too rare at this
  scale, so they are built by perturbing accepted answers in ways the verifier is known to
  catch. Synthetic and easy; a real model's mistakes are subtler.

## Reproducibility

| | live | replayed |
|---|---|---|
| Sweep cost | $0.3313 | **$0.0000** |
| B2 p95 latency | 12,290 ms | 77 ms |
| Metric fields identical | — | **14/14 (100%)** |

```
make eval        # live sweep
make replay      # same sweep from the journal
make determinism # compares the two
```

**What F1 does and does not claim.** It says the journal key is stable, so a replay resolves
to the same rows and reproduces the same table. It says nothing about model reproducibility:
sampling parameters were removed from current models, so two live calls with identical inputs
may legitimately differ. The determinism is the pipeline's, not the model's.

## Component status

| Component | State |
|---|---|
| EDGAR/XBRL ingest | 25,135 facts, **0 undatable**, idempotent |
| Point-in-time enforcement | Full strength — required at the store boundary, enforced in SQL views for agent-written queries |
| Journal + replay | Full strength — sweep replays at $0.00 |
| Budget ledger | Full strength — aborts on breach; tracks spent and list cost separately |
| Five tools | All present; `python` is process isolation, not a sandbox |
| Router | Fixed rubric over P0–P4, unlearned |
| Five-signal verifier | All five; `s2`/`s4` rubrics uncalibrated |
| Eval harness | 20 questions, B1 and B2 |

## Known weaknesses

Listing these is the point of the stage, not an apology for it.

0. **A correctness-accounting bug was found and fixed after the first table was published.**
   `s5` scores 1.0 when it does not apply, and a run that never calls `finish` has no terminal
   step — so reading the last verdict's score counted "no answer" as "right answer". B2's
   accuracy was reported as 100% and is actually 85%. Signals now carry an `applicable` flag
   and correctness goes through one function, with regression tests. The earlier figure is
   corrected above.
1. **Ground truth is circular.** Expected values come from the same XBRL facts the environment
   serves, so an answer can be right without anything being understood. The published
   benchmark needs values read off the filing document itself.
2. **`s3` is weaker on prose than on facts.** A figure quoted in filing text carries no unit —
   statements are tabulated "in millions" and that context lives in a column header the
   extractor never sees — so scale-equivalent matches are accepted for text evidence. The
   off-by-1000 detection that motivates the signal holds only for structured evidence.
3. **`s2` and `s4` are uncalibrated.** No κ against human labels yet, so the judged signals
   are unvalidated. This is MVP2.2's gate and nothing downstream should be trusted until it
   passes.
4. **One company, 20 questions, one model.** No confidence intervals, and B0/B3 do not exist.
5. **Chunking is naive.** Word windows, no awareness of 10-K item structure.

## What this stage bought

The apparatus is real: an environment that cannot leak, a verifier that catches silent numeric
errors, a benchmark that reproduces for free, and a baseline that the system beats on a metric
chosen before the result was known. Everything after this is scale and rigour, not
architecture.

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
| **B2** tools | **100%** | **0%** | **81%** | **$0.0109** | 12,290 ms | 2.9 | $0.2178 |

The comparison is deliberately fair to B1: same corpus, same as-of horizon, same model, same
verifier. The only thing it lacks is the ability to look again after seeing what came back.

Two things worth drawing out.

**B2 costs 2.4× more in total and is still cheaper per correct answer** — $0.0109 against
$0.0132 — because B1 is wrong about two-thirds of the time. A per-token comparison would have
picked the worse system. This is the entire argument for measuring E1 rather than spend.

**B2 is slower, by a lot.** 12.3 s at p95 against 3.6 s. Accuracy was bought with latency, and
on this evidence the trade is worth it; on a latency-sensitive product it might not be.

**D3 is the interesting column.** B2 answers every question correctly while only 81% of its
steps verify — so roughly a fifth of its steps are wasted or wrong inside trajectories that
end well. Outcome filtering keeps all of them; step filtering keeps 81%. That divergence is
the mechanism H1 proposes to exploit, and it is now measured rather than assumed.

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

# FinanceVault

A point-in-time-correct financial tool-use environment with step-level verification.

Answering a question about a company's financials is a multi-step problem: find the right
filing, pull the right tagged fact, check its unit and reporting period, compute a derived
figure, cite the source. Retrieval-augmented generation handles the first step and guesses at
the rest. Agents that chain tool calls do better, but they are judged only on whether the final
answer is right, which gives no signal about *where* a wrong answer went wrong.

Two things make finance harder than general tool use:

- **Numeric errors are silent.** An answer off by a factor of a thousand, or drawn from the
  wrong fiscal quarter, reads as fluent and correct.
- **Corpora have a time dimension.** A retriever that ignores filing acceptance dates will
  answer questions using documents that did not yet exist.

FinanceVault addresses both: `as_of` filtering is enforced at the store boundary so no call
site can omit it, and a five-signal verifier scores every step of a trajectory rather than
only its final answer. Those step scores then filter the data used for fine-tuning.

**Hypothesis (H1).** Filtering agent trajectories at the step level produces better
fine-tuning data than filtering on final-answer correctness, at equal trajectory budget.

## Status

MVP1, days 1–2 complete: repo skeleton, container stack, and the seven-table schema.

## Quick start

```bash
make venv
make up
make migrate
```

`make help` lists every target. Postgres listens on **55432** and Redis on **55433** —
deliberately unusual ports, because this host already runs both for other projects.

Copy `.env.example` to `.env` and set `FV_ANTHROPIC_API_KEY` before anything that calls a model.

## Layout

```
src/financevault/
  config.py    process-wide settings, read once from the environment
  env/         EDGAR and XBRL ingest, chunking, as-of filtering
  store/       schema.sql and Postgres access
  tools/       the five agent tools behind Pydantic contracts
  runtime/     router, step executor, budget ledger, journal, replay
  verify/      the five verifier signals
  data/        rollout, repair, hard-negative mining, export
  train/       LoRA SFT
  api/         FastAPI service
eval/          frozen splits, harness, baselines
bench/         calibration, leak-rate contrast, frontier figure
```

## Design notes

**`accepted_at`, not `period_end`.** The point-in-time key is the SEC acceptance timestamp —
the moment a document became knowable to the public. It is `NOT NULL` everywhere it appears so
a row can never silently escape an as-of filter.

**Unit and scale are stored separately from value.** An answer that is right in millions and
wrong in units is exactly the silent error this project is built to catch, so `xbrl_facts`
keeps them as distinct columns rather than folding them into a single number.

**Verification is a separate pass from generation.** Signals on `steps` are nullable and
written later, so either stage can be re-run without redoing the other.

**The journal is content-addressed.** Every model call is keyed on a hash of model, prompt,
tool specs and sampling params, so `FV_REPLAY=true` reproduces any past run offline at zero
token cost. Under replay a cache miss raises rather than falling back to a live call — a
replay can never silently spend money.

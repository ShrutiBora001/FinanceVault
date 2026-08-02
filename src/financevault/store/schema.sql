-- FinanceVault schema.
--
-- Point-in-time note: `accepted_at` is the SEC acceptance timestamp, i.e. the moment a
-- document became knowable to the public. It is NOT `period_end` (when the fiscal period
-- closed) nor `filed_at` (the filing date, which can differ from acceptance). Every
-- as-of filter in this system keys on `accepted_at`. It is NOT NULL everywhere it appears
-- so a row can never silently escape the filter.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------- environment

CREATE TABLE IF NOT EXISTS filings (
    id           BIGSERIAL PRIMARY KEY,
    cik          TEXT        NOT NULL,
    ticker       TEXT,
    accession    TEXT        NOT NULL UNIQUE,
    form         TEXT        NOT NULL,
    period_end   DATE,
    filed_at     TIMESTAMPTZ,
    accepted_at  TIMESTAMPTZ NOT NULL,
    url          TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS filings_cik_idx         ON filings (cik);
CREATE INDEX IF NOT EXISTS filings_accepted_at_idx ON filings (accepted_at);
CREATE INDEX IF NOT EXISTS filings_form_idx        ON filings (form);

CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    filing_id  BIGINT NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    section    TEXT,
    idx        INT    NOT NULL,
    text       TEXT   NOT NULL,
    n_tokens   INT,
    embedding  VECTOR(384),
    tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (filing_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_filing_id_idx ON chunks (filing_id);
-- HNSW over cosine distance. Built after the first bulk load; harmless when empty.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- XBRL numeric facts. These are the ground truth for the `s3` numeric-correctness signal,
-- so unit and scale are stored explicitly rather than folded into `value`: an answer that
-- is right in millions and wrong in units is exactly the silent error we are hunting.
CREATE TABLE IF NOT EXISTS xbrl_facts (
    id            BIGSERIAL PRIMARY KEY,
    cik           TEXT        NOT NULL,
    taxonomy      TEXT        NOT NULL,           -- us-gaap, dei, srt, ...
    tag           TEXT        NOT NULL,           -- e.g. RevenueFromContractWithCustomer...
    unit          TEXT        NOT NULL,           -- USD, shares, USD/shares, ...
    value         NUMERIC     NOT NULL,
    scale         INT,                            -- SEC `decimals`; -6 means reported in millions
    fy            INT,
    fp            TEXT,                           -- FY, Q1, Q2, Q3
    period_start  DATE,
    period_end    DATE,
    form          TEXT,
    accession     TEXT,
    accepted_at   TIMESTAMPTZ NOT NULL,
    frame         TEXT,
    -- NULLS NOT DISTINCT is load-bearing, not stylistic. Instant facts (balance-sheet
    -- items) have no period_start, and under the default NULLS DISTINCT every re-ingest
    -- would insert them again -- ON CONFLICT never fires because NULL <> NULL. Requires
    -- Postgres 15+.
    UNIQUE NULLS NOT DISTINCT (accession, taxonomy, tag, unit, period_start, period_end)
);
CREATE INDEX IF NOT EXISTS xbrl_cik_tag_idx     ON xbrl_facts (cik, tag);
CREATE INDEX IF NOT EXISTS xbrl_period_end_idx  ON xbrl_facts (period_end);
CREATE INDEX IF NOT EXISTS xbrl_accepted_at_idx ON xbrl_facts (accepted_at);

CREATE TABLE IF NOT EXISTS prices (
    ticker    TEXT   NOT NULL,
    date      DATE   NOT NULL,
    open      NUMERIC,
    high      NUMERIC,
    low       NUMERIC,
    close     NUMERIC,
    adj_close NUMERIC,
    volume    BIGINT,
    PRIMARY KEY (ticker, date)
);

-- ---------------------------------------------------------------- as-of views

-- The SQL tool executes agent-written queries, so it cannot be trusted to apply its own
-- point-in-time filter. These views apply it in the database, reading the horizon from a
-- session variable that the tool sets per call.
--
-- `current_setting('fv.as_of')` with no second argument RAISES when the variable is unset,
-- which is the point: an unfiltered query fails closed rather than silently returning the
-- whole corpus. The SQL tool exposes only these views, never the base tables.

CREATE OR REPLACE VIEW v_filings AS
    SELECT * FROM filings
    WHERE accepted_at <= current_setting('fv.as_of')::timestamptz;

CREATE OR REPLACE VIEW v_xbrl_facts AS
    SELECT * FROM xbrl_facts
    WHERE accepted_at <= current_setting('fv.as_of')::timestamptz;

CREATE OR REPLACE VIEW v_chunks AS
    SELECT c.*, f.cik, f.form, f.period_end, f.accepted_at
    FROM chunks c JOIN filings f ON f.id = c.filing_id
    WHERE f.accepted_at <= current_setting('fv.as_of')::timestamptz;

-- Prices have no acceptance instant: a daily bar for date D is knowable at D's close.
-- adj_close is excluded because later splits and dividends rewrite it retroactively, which
-- makes it unsafe to read at a historical horizon.
CREATE OR REPLACE VIEW v_prices AS
    SELECT ticker, date, open, high, low, close, volume FROM prices
    WHERE date <= (current_setting('fv.as_of')::timestamptz)::date;

-- ---------------------------------------------------------------- trajectories

-- One agent episode. `as_of` is recorded on the run so a trajectory can be replayed or
-- re-verified later under exactly the information horizon it was produced under.
CREATE TABLE IF NOT EXISTS runs (
    id           UUID PRIMARY KEY,
    question     TEXT        NOT NULL,
    as_of        TIMESTAMPTZ NOT NULL,
    policy       TEXT        NOT NULL,            -- b1-rag, b2-tools, mvp-adapter, ...
    path         TEXT,                            -- P0..P4, chosen by the router
    budget_usd   NUMERIC,
    budget_steps INT,
    outcome      TEXT,                            -- ok, budget_exceeded, tool_error, max_steps
    answer       TEXT,
    citations    JSONB,
    n_steps      INT,
    cost_usd     NUMERIC     NOT NULL DEFAULT 0,
    latency_ms   INT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS runs_policy_idx     ON runs (policy);
CREATE INDEX IF NOT EXISTS runs_created_at_idx ON runs (created_at);

-- The trajectory itself, one row per step, carrying the five verifier signals.
-- Signals are NULL until the verifier runs, which is deliberately a separate pass:
-- generation and verification are decoupled so either can be re-run alone.
CREATE TABLE IF NOT EXISTS steps (
    run_id      UUID NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    idx         INT  NOT NULL,
    thought     TEXT,
    tool        TEXT,
    args        JSONB,
    obs         JSONB,
    s1          REAL,                             -- tool validity        (programmatic)
    s2          REAL,                             -- citation support     (llm-judged)
    s3          REAL,                             -- numeric correctness  (programmatic)
    s4          REAL,                             -- retrieval relevance  (llm-judged)
    s5          REAL,                             -- answer correctness   (terminal only)
    s_reasons   JSONB,
    tokens_in   INT,
    tokens_out  INT,
    cost_usd    NUMERIC,
    latency_ms  INT,
    PRIMARY KEY (run_id, idx)
);
CREATE INDEX IF NOT EXISTS steps_tool_idx ON steps (tool);

-- ---------------------------------------------------------------- journal

-- Content-addressed record of every model call. `hash` covers model, prompt, tool specs
-- and sampling params, so an identical request always resolves to the same row. This is
-- what makes REPLAY=1 free and byte-deterministic.
CREATE TABLE IF NOT EXISTS journal (
    hash        TEXT PRIMARY KEY,
    model       TEXT        NOT NULL,
    request     JSONB       NOT NULL,
    response    JSONB       NOT NULL,
    tokens_in   INT,
    tokens_out  INT,
    cost_usd    NUMERIC,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS journal_model_idx ON journal (model);

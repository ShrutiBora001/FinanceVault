"""Process-wide configuration, read once from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FV_", extra="ignore")

    # storage
    pg_dsn: str = "postgresql://financevault:financevault@localhost:55432/financevault"
    redis_url: str = "redis://localhost:55433/0"

    # Models. Aliases, not dated snapshots -- the alias is the documented form.
    #
    # Defaults are the cheapest configuration that works: Haiku 4.5 at $1/$5 per MTok, on both
    # the routing/judging path and the analyst path. Sonnet 5 is 3x the price ($3/$15) and is
    # a one-line switch when it earns it:
    #
    #     FV_ANALYST_MODEL=claude-sonnet-5
    #
    # The switch is expected. MVP1 measures whether the pipeline works, so the cheap model is
    # the right default; MVP2.2 stands up B0-B3 as separate policies, at which point the
    # analyst model becomes an experimental variable rather than a setting. Note the efficiency
    # benchmark needs at least one strong-model run to have a frontier to plot against.
    anthropic_api_key: str = ""
    judge_model: str = "claude-haiku-4-5"
    analyst_model: str = "claude-haiku-4-5"
    # B3 only. The ceiling the efficiency frontier is plotted against; never the default,
    # because a benchmark whose baseline is the expensive model has no cost story to tell.
    frontier_model: str = "claude-sonnet-5"
    # Thinking depth and overall spend on the analyst path. Ignored on models that predate the
    # parameter -- Haiku 4.5 rejects it outright, so `llm.call` drops it rather than passing
    # it through. Kept configurable so the efficiency benchmark can sweep it.
    analyst_effort: str = "medium"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # execution budget; the ledger aborts a run that breaches either
    max_steps: int = 6
    max_usd: float = 0.05

    # When true, every model call must resolve from the journal. A miss is an error
    # rather than a live call, so a replay can never silently cost money.
    replay: bool = False

    # SEC requires a descriptive User-Agent with contact details on every request.
    sec_user_agent: str = "FinanceVault research (sb9880@nyu.edu)"

    @property
    def sec_headers(self) -> dict[str, str]:
        return {"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate"}


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()

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
    anthropic_api_key: str = ""
    judge_model: str = "claude-haiku-4-5"
    analyst_model: str = "claude-sonnet-5"
    # Effort controls thinking depth and overall spend; it defaults to `high` server-side,
    # which is more than a bounded tool loop needs. Kept configurable so the efficiency
    # benchmark can sweep it rather than guess.
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

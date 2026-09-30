"""Journal key stability and the replay contract.

If hashing is unstable, the F1 determinism metric measures serialization noise instead of
pipeline behaviour, and every "free replay" claim in the report is unfounded.
"""

from __future__ import annotations

import pytest

from financevault.config import settings
from financevault.runtime import journal
from financevault.runtime.budget import Ledger
from financevault.store import pg

MESSAGES = [{"role": "user", "content": "What was Apple's FY2025 revenue?"}]
TOOLS = [{"name": "lookup_fact", "input_schema": {"type": "object", "properties": {}}}]


@pytest.fixture
def db() -> bool:
    if pg.missing_tables():
        pytest.skip("schema not applied; run `make migrate`")
    return True


# ---------------------------------------------------------------- key stability


def test_key_is_deterministic() -> None:
    assert journal.key("m", MESSAGES) == journal.key("m", MESSAGES)


def test_key_is_insensitive_to_dict_ordering() -> None:
    """The same logical request must hash identically regardless of key insertion order."""
    a = journal.key("m", [{"role": "user", "content": "hi"}])
    b = journal.key("m", [{"content": "hi", "role": "user"}])
    assert a == b


def test_key_changes_with_every_parameter_that_steers_output() -> None:
    base = journal.key("m", MESSAGES, tools=TOOLS, system="s", max_tokens=100)
    variants = {
        "model": journal.key("m2", MESSAGES, tools=TOOLS, system="s", max_tokens=100),
        "messages": journal.key(
            "m", [{"role": "user", "content": "x"}], tools=TOOLS, system="s", max_tokens=100
        ),
        "tools": journal.key("m", MESSAGES, tools=[], system="s", max_tokens=100),
        "system": journal.key("m", MESSAGES, tools=TOOLS, system="other", max_tokens=100),
        "thinking": journal.key(
            "m", MESSAGES, tools=TOOLS, system="s", thinking={"type": "adaptive"}, max_tokens=100
        ),
        "effort": journal.key(
            "m", MESSAGES, tools=TOOLS, system="s", effort="high", max_tokens=100
        ),
        "max_tokens": journal.key("m", MESSAGES, tools=TOOLS, system="s", max_tokens=200),
        "extra": journal.key(
            "m", MESSAGES, tools=TOOLS, system="s", max_tokens=100, extra={"seed": 1}
        ),
    }
    for name, variant in variants.items():
        assert variant != base, f"changing {name} must change the key"


def test_keys_are_distinct_across_variants() -> None:
    """No two different requests may collide onto one address."""
    keys = {
        journal.key("m", MESSAGES),
        journal.key("m", MESSAGES, system="s"),
        journal.key("m", MESSAGES, tools=TOOLS),
        journal.key("m", MESSAGES, effort="low"),
    }
    assert len(keys) == 4


def test_canonical_json_escapes_unicode_consistently() -> None:
    """Non-ASCII must serialize identically everywhere, or hashes diverge across machines."""
    payload = {"text": "café — ünïcode"}
    assert journal.canonical(payload) == journal.canonical(payload)
    assert "\\u" in journal.canonical(payload)


def test_scheme_is_part_of_the_key() -> None:
    assert journal.SCHEME in journal.canonical({"scheme": journal.SCHEME, "model": "m"})


# ---------------------------------------------------------------- round trip


def test_put_then_get_round_trips(db: bool) -> None:
    hash_ = journal.key("test-model", [{"role": "user", "content": "round trip"}])
    response = {"content": [{"type": "text", "text": "42"}], "stop_reason": "end_turn"}
    journal.put(
        hash_,
        "test-model",
        {"model": "test-model"},
        response,
        tokens_in=10,
        tokens_out=2,
        cost_usd=0.001,
    )

    entry = journal.get(hash_)
    assert entry is not None
    assert entry.response == response
    assert entry.tokens_in == 10
    assert entry.cost_usd == pytest.approx(0.001)

    pg.execute("DELETE FROM journal WHERE hash = %s", (hash_,))


def test_put_is_first_write_wins(db: bool) -> None:
    """A differing response for the same key is a determinism failure worth detecting."""
    hash_ = journal.key("test-model", [{"role": "user", "content": "first write"}])
    first = {"content": [{"type": "text", "text": "original"}]}
    second = {"content": [{"type": "text", "text": "overwritten"}]}
    journal.put(hash_, "test-model", {}, first, tokens_in=1, tokens_out=1, cost_usd=0.0)
    journal.put(hash_, "test-model", {}, second, tokens_in=1, tokens_out=1, cost_usd=0.0)

    entry = journal.get(hash_)
    assert entry is not None and entry.response == first

    pg.execute("DELETE FROM journal WHERE hash = %s", (hash_,))


def test_get_returns_none_for_unknown_hash(db: bool) -> None:
    assert journal.get("0" * 64) is None


# ---------------------------------------------------------------- replay contract


def test_replay_miss_raises_instead_of_calling_live(
    db: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract: under replay, a miss is fatal. It must never cost money silently."""
    from financevault.runtime import llm

    settings.cache_clear()
    monkeypatch.setenv("FV_REPLAY", "true")
    monkeypatch.setenv("FV_ANTHROPIC_API_KEY", "")

    ledger = Ledger(max_usd=1.0, max_steps=10)
    with pytest.raises(journal.ReplayMiss, match="no journal entry"):
        llm.call(
            "claude-haiku-4-5-20251001", [{"role": "user", "content": "never seen"}], ledger=ledger
        )

    assert ledger.usd == 0.0
    settings.cache_clear()


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-5"])
def test_replay_hit_costs_nothing(db: bool, monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """Runs on both request surfaces, since they produce different keys for the same call.

    The expected key is derived through the same helper `call` uses rather than hardcoded --
    a literal here would silently drift from the code and turn a real key regression into a
    passing test.
    """
    from financevault.runtime import llm

    messages = [{"role": "user", "content": f"cached question ({model})"}]
    hash_ = journal.key(
        model,
        messages,
        thinking=llm._thinking_param(model, False, 4096),
        max_tokens=4096,
    )
    journal.put(
        hash_,
        model,
        {},
        {"content": [{"type": "text", "text": "cached answer"}]},
        tokens_in=500,
        tokens_out=50,
        cost_usd=0.00075,
    )

    settings.cache_clear()
    monkeypatch.setenv("FV_REPLAY", "true")
    # The ceiling must be able to afford the call. It used to be set below the call's price to
    # prove the response came from cache, but a cached call now consumes list budget -- the
    # ceiling is a property of the policy, so it cannot depend on whether the journal is warm.
    # REPLAY mode is the stronger proof anyway: a miss raises rather than quietly going live.
    ledger = Ledger(max_usd=1.0, max_steps=10)

    result = llm.call(model, messages, ledger=ledger)
    assert result.text == "cached answer"
    assert result.cached is True
    assert result.cost_usd == 0.0
    assert ledger.usd == 0.0, "a journal hit is free"
    assert ledger.list_usd > 0.0, "and still consumes the run's budget"

    pg.execute("DELETE FROM journal WHERE hash = %s", (hash_,))
    settings.cache_clear()

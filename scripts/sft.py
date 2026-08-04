"""Run the LoRA smoke fine-tune and load the adapter back.

Proves the last MVP1 seam: exported trajectories become a trained adapter that loads. It is
not an experiment and its loss numbers mean nothing about H1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from financevault.train.sft_lora import DEFAULT_MODEL, TrainConfig, train, verify_adapter

console = Console()
OUT = Path("eval/results/sft_smoke.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="step_filtered")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    config = TrainConfig(
        model=args.model,
        split=args.split,
        epochs=args.epochs,
        max_examples=args.max_examples,
    )

    console.print(
        f"[bold]LoRA smoke run[/bold] — {config.model} on [cyan]{config.split}[/cyan]\n"
        "[dim]a pipeline test, not an experiment: the adapter is worthless as a model[/dim]\n"
    )

    try:
        result = train(config)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    table = Table(header_style="bold")
    table.add_column("field")
    table.add_column("value", justify="right")
    for key in (
        "n_examples",
        "device",
        "steps",
        "trainable_params",
        "total_params",
        "trainable_fraction",
        "first_loss",
        "last_loss",
    ):
        value = result[key]
        table.add_row(key, f"{value:,}" if isinstance(value, int) else str(value))
    console.print(table)

    console.print("\nloading the adapter back onto the base model…")
    loaded = verify_adapter(Path(result["adapter_dir"]), config.model)
    result["verify"] = loaded
    console.print(
        f"[green]ok[/green] — adapter loaded, {loaded['adapter_params']:,} LoRA parameters"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    console.print(f"wrote {OUT} and {result['adapter_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Apply the schema and report what exists. Safe to re-run."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from financevault.store import pg

console = Console()


def main() -> int:
    try:
        pg.migrate()
    except Exception as exc:  # noqa: BLE001 - surface the cause plainly to the operator
        console.print(f"[red]migration failed:[/red] {exc}")
        console.print("[dim]is the stack up? try `make up`[/dim]")
        return 1

    missing = pg.missing_tables()
    if missing:
        console.print(f"[red]tables still missing after migrate:[/red] {', '.join(missing)}")
        return 1

    table = Table(title="financevault schema", header_style="bold")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, count in pg.table_counts().items():
        table.add_row(name, f"{count:,}")
    console.print(table)
    console.print(f"[green]ok[/green] — all {len(pg.TABLES)} tables present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

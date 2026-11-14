#!/usr/bin/env python
"""Entry point for the terminal agent.

    python run_cli.py --candidate-id c001
    python run_cli.py --list
    python run_cli.py --reset
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console

# Windows consoles still default to cp1252, which cannot render the box drawing
# rich uses. Force UTF-8 before anything is printed.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
from rich.table import Table

from agent.core import Agent
from agent.db import CandidateDB, CandidateDBError
from agent.llm import LLM, LLMError
from agent.validators import missing_fields, suspect_fields
from ui.cli import TerminalUI, run_chat


def show_workspace(db, console):
    table = Table(box=None, pad_edge=False)
    for column in ("id", "name", "status", "missing", "suspect"):
        table.add_column(column, style="dim" if column == "id" else None)
    for candidate in db.list_candidates():
        missing = missing_fields(candidate)
        suspect = [s["field"] for s in suspect_fields(candidate)]
        table.add_row(
            candidate["id"],
            candidate.get("full_name", ""),
            candidate.get("collection_status", ""),
            ", ".join(missing) or "-",
            ", ".join(suspect) or "-",
        )
    console.print()
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="ACME candidate information collection agent")
    parser.add_argument("--candidate-id", "-c", help="e.g. c001")
    parser.add_argument("--list", action="store_true", help="list your workspace and exit")
    parser.add_argument("--reset", action="store_true", help="restore the workspace and exit")
    parser.add_argument("--no-trace", action="store_true", help="hide tool and HTTP lines")
    parser.add_argument("--idle-timeout", type=int,
                        default=int(os.getenv("IDLE_TIMEOUT_SECONDS", "120")),
                        help="seconds to wait for a reply before nudging")
    args = parser.parse_args()

    load_dotenv()
    console = Console(legacy_windows=False)
    ui = TerminalUI(console=console, show_trace=not args.no_trace)

    try:
        db = CandidateDB(on_request=ui.on_http if not args.no_trace else None)
    except CandidateDBError as exc:
        console.print("[bold red]{}[/]".format(exc))
        return 2

    if args.reset:
        db.reset()
        console.print("[green]workspace reset to the original data[/]")
        return 0

    if args.list or not args.candidate_id:
        if not args.list:
            console.print("[yellow]no --candidate-id given; showing the workspace[/]")
        show_workspace(db, console)
        return 0 if args.list else 1

    try:
        llm = LLM()
    except LLMError as exc:
        console.print("[bold red]{}[/]".format(exc))
        return 2

    try:
        agent = Agent(args.candidate_id, db, llm, emit=ui.on_event)
        needs_chat, summary = agent.bootstrap()
    except CandidateDBError as exc:
        console.print("[bold red]{}[/]".format(exc))
        return 2

    if not needs_chat:
        ui.nothing_to_do(summary)
        return 0

    ui.banner(args.candidate_id, agent.state.record, summary)

    try:
        run_chat(agent, ui, idle_timeout=args.idle_timeout)
    except KeyboardInterrupt:
        console.print()
        console.print("[dim]interrupted - recording what was collected[/]")
        agent.close("operator interrupted the chat")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

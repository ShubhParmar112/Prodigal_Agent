"""Terminal chat.

The safe front end for a live demo: no browser, no port, and the evaluator can
take the keyboard straight away. Shows the tool calls and HTTP status codes
inline and dimmed, so what the agent is doing is visible without cluttering the
conversation.
"""
from __future__ import annotations

import queue
import sys
import threading

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agent.events import AGENT, CANDIDATE, ERROR, HTTP, OUTCOME, SYSTEM, TOOL

QUIT_WORDS = {"/quit", "/exit", "/q", "/bye"}
SILENCE = "/silence"

STATUS_STYLE = {
    "completed": "bold green",
    "in_progress": "bold yellow",
    "no_answer": "bold red",
    "refused": "bold magenta",
    "needs_human": "bold red",
}


class Reader:
    """Reads stdin on a background thread so the agent can time out waiting."""

    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        for line in sys.stdin:
            self._queue.put(line.rstrip("\n"))
        self._queue.put(None)  # EOF - the candidate closed the chat

    def read(self, timeout):
        try:
            return self._queue.get(timeout=timeout), False
        except queue.Empty:
            return None, True  # timed out


class TerminalUI:
    def __init__(self, console=None, show_trace=True):
        self.console = console or Console()
        self.show_trace = show_trace

    # ------------------------------------------------------------- render

    def on_event(self, event):
        if event.kind == AGENT:
            self.console.print(Panel(Text(event.text), title="Shubh (ACME)",
                                     title_align="left", border_style="cyan", padding=(0, 1)))
        elif event.kind == TOOL and self.show_trace:
            style = "dim green" if event.meta.get("result") else "dim cyan"
            self.console.print(Text("  " + event.text, style=style))
        elif event.kind == HTTP and self.show_trace:
            self.console.print(Text("  " + event.text, style="dim"))
        elif event.kind == SYSTEM:
            self.console.print(Text("  " + event.text, style="dim italic"))
        elif event.kind == ERROR:
            self.console.print(Text("  ! " + event.text, style="bold red"))
        elif event.kind == OUTCOME:
            self.console.print()
            self.console.print(Rule(style="dim"))
            self.console.print(Text("  outcome: ", style="dim")
                               + Text(event.text, style=STATUS_STYLE.get(event.text, "bold")))
            data = event.data or {}
            if data.get("written"):
                self.console.print(Text("  written: ", style="dim")
                                   + Text(", ".join(data["written"]), style="green"))
            if data.get("still_needed"):
                self.console.print(Text("  still needed: ", style="dim")
                                   + Text(", ".join(data["still_needed"]), style="yellow"))
            if data.get("notes"):
                self.console.print(Text("  notes: " + data["notes"], style="dim"))

    def on_http(self, entry):
        if not self.show_trace:
            return
        colour = "dim green" if entry["status"] < 300 else "dim red"
        self.console.print(Text("  {} {} -> {}".format(
            entry["method"], entry["path"], entry["status"]), style=colour))

    # -------------------------------------------------------------- chrome

    def banner(self, candidate_id, record, summary):
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column()
        table.add_row("candidate", "{}  ({})".format(record.get("full_name", "?"), candidate_id))
        table.add_row("version", str(record.get("version")))
        table.add_row("to collect", summary)
        self.console.print()
        self.console.print(Panel(table, title="ACME candidate collection agent",
                                 title_align="left", border_style="dim", padding=(0, 1)))
        self.console.print(Text(
            "  you are the candidate - type replies below.  "
            "{} to leave, {} to simulate going quiet".format("/quit", SILENCE),
            style="dim italic"))
        self.console.print()

    def nothing_to_do(self, reason):
        self.console.print()
        self.console.print(Panel(Text(reason), title="no conversation needed",
                                 title_align="left", border_style="green", padding=(0, 1)))
        self.console.print()

    def ask(self, timeout, reader):
        self.console.print(Text("you > ", style="bold magenta"), end="")
        text, timed_out = reader.read(timeout)
        if timed_out:
            self.console.print(Text("(no reply)", style="dim italic"))
        return text, timed_out


def run_chat(agent, ui, idle_timeout=120):
    """Drive the conversation to its end and return the outcome."""
    reader = Reader()

    ui.console.print()
    agent.open()

    idle_rounds = 0
    while not agent.state.ended:
        text, timed_out = ui.ask(idle_timeout, reader)

        if timed_out or text == SILENCE:
            idle_rounds += 1
            if idle_rounds == 1:
                ui.console.print(Text("  candidate went quiet - nudging once", style="dim italic"))
                agent.nudge()
                continue
            ui.console.print(Text("  still no reply - closing the chat", style="dim italic"))
            return agent.close("candidate stopped replying", gave_up=True)

        if text is None:  # EOF
            return agent.close("chat window closed")

        text = text.strip()
        if not text:
            continue
        if text.lower() in QUIT_WORDS:
            return agent.close("candidate left the chat")

        idle_rounds = 0
        agent.reply(text)

    return agent.finalise()

"""Streamlit chat.

Same agent, same tools, different skin. The sidebar is the point: the record
updates live as the conversation fills it, so you can watch validation and
writes happen rather than taking the chat's word for it.
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from agent.core import Agent
from agent.db import CandidateDB, CandidateDBError
from agent.events import AGENT, ERROR, OUTCOME, SYSTEM, TOOL
from agent.llm import LLM, LLMError
from agent.validators import COLLECTABLE, OPTIONAL, validate_field

STATUS_COLOUR = {
    "completed": "#1a7f37",
    "in_progress": "#9a6700",
    "no_answer": "#a40e26",
    "refused": "#8250df",
    "needs_human": "#a40e26",
    "pending": "#57606a",
}

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 980px;}
  .field-row {display:flex; justify-content:space-between; gap:.5rem;
              padding:.3rem .55rem; border-radius:7px; margin-bottom:.22rem;
              font-size:.82rem; line-height:1.3; border:1px solid transparent;}
  .field-row .k {opacity:.75;}
  .field-row .v {font-weight:600; text-align:right; word-break:break-word;}
  .ok      {background:rgba(26,127,55,.10);  border-color:rgba(26,127,55,.25);}
  .missing {background:rgba(140,140,140,.10);border-color:rgba(140,140,140,.22);}
  .bad     {background:rgba(191,54,12,.11);  border-color:rgba(191,54,12,.30);}
  .pill {display:inline-block; padding:.18rem .6rem; border-radius:999px;
         color:#fff; font-size:.74rem; font-weight:700; letter-spacing:.02em;}
  .trace {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.73rem;
          white-space:pre-wrap; opacity:.85; margin:.08rem 0;}
  .trace-res {color:#1a7f37;}
  .trace-err {color:#bf360c;}
</style>
"""


# ------------------------------------------------------------------ helpers

def _state():
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("trace", [])
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("outcome", None)
    st.session_state.setdefault("banner", None)


def _emit(event):
    """Route agent events into session state so a rerun can draw them."""
    if event.kind == AGENT:
        st.session_state.chat.append(("assistant", event.text))
    elif event.kind == TOOL:
        st.session_state.trace.append(("res" if event.meta.get("result") else "req", event.text))
    elif event.kind == SYSTEM:
        st.session_state.trace.append(("sys", event.text))
        if "switched to" in event.text:
            st.session_state.banner = ("info", event.text)
    elif event.kind == ERROR:
        st.session_state.trace.append(("err", event.text))
        st.session_state.banner = ("error", event.text)
    elif event.kind == OUTCOME:
        st.session_state.outcome = {"status": event.text, **(event.data or {})}


def _on_http(entry):
    st.session_state.trace.append((
        "res" if entry["status"] < 300 else "err",
        "{} {} -> {}".format(entry["method"], entry["path"], entry["status"]),
    ))


def _start(candidate_id):
    load_dotenv()
    st.session_state.chat = []
    st.session_state.trace = []
    st.session_state.outcome = None
    st.session_state.agent = None
    st.session_state.banner = None

    try:
        db = CandidateDB(on_request=_on_http)
        llm = LLM()
    except (CandidateDBError, LLMError) as exc:
        st.session_state.banner = ("error", str(exc))
        return

    try:
        agent = Agent(candidate_id.strip(), db, llm, emit=_emit)
        needs_chat, summary = agent.bootstrap()
    except CandidateDBError as exc:
        st.session_state.banner = ("error", str(exc))
        return

    if not needs_chat:
        st.session_state.agent = agent
        st.session_state.banner = ("done", summary)
        return

    st.session_state.agent = agent
    st.session_state.banner = ("info", "To collect - " + summary)
    agent.open()


# ------------------------------------------------------------------ sidebar

def _field_row(label, value, css, hint=""):
    shown = "not provided" if value in (None, "", []) else value
    if isinstance(shown, list):
        shown = ", ".join(str(s) for s in shown)
    st.markdown(
        '<div class="field-row {}" title="{}"><span class="k">{}</span>'
        '<span class="v">{}</span></div>'.format(css, hint, label, shown),
        unsafe_allow_html=True,
    )


def _record_panel(agent):
    record = agent.state.record
    suspect = {s["field"]: s["problem"] for s in agent.state.suspect()}

    st.markdown("**Mandatory**")
    for field in COLLECTABLE:
        value = record.get(field)
        if field in suspect:
            css, hint = "bad", suspect[field]
        elif value in (None, "", []):
            css, hint = "missing", "still to collect"
        else:
            css, hint = "ok", "valid"
        _field_row(field, value, css, hint)

    _field_row("phone", record.get("phone"), "ok", "read-only, OTP verified")

    with st.expander("Optional fields"):
        for field in OPTIONAL:
            value = record.get(field)
            css = "ok" if value not in (None, "", []) else "missing"
            _field_row(field, value, css)

    st.caption("record version {} · attempt {} · {}".format(
        record.get("version"), record.get("attempt_count"),
        record.get("collection_status")))


def _sidebar():
    with st.sidebar:
        st.markdown("### Candidate")
        candidate_id = st.text_input("Candidate ID", value="c001",
                                     label_visibility="collapsed",
                                     placeholder="c001").strip()
        left, right = st.columns(2)
        if left.button("Start chat", type="primary", use_container_width=True):
            _start(candidate_id)
            st.rerun()
        if right.button("Reset data", use_container_width=True,
                        help="Restore your workspace to the original data"):
            try:
                load_dotenv()
                CandidateDB().reset()
                st.session_state.banner = ("info", "Workspace reset to the original data.")
            except CandidateDBError as exc:
                st.session_state.banner = ("error", str(exc))
            st.rerun()

        agent = st.session_state.agent
        if agent:
            st.divider()
            _record_panel(agent)

            if not agent.state.ended and st.button("End chat", use_container_width=True):
                agent.close("the operator ended the chat")
                st.rerun()

            st.divider()
            with st.expander("Tool calls and API", expanded=False):
                if not st.session_state.trace:
                    st.caption("nothing yet")
                for kind, line in st.session_state.trace[-200:]:
                    css = {"res": "trace trace-res", "err": "trace trace-err"}.get(kind, "trace")
                    st.markdown('<div class="{}">{}</div>'.format(css, line),
                                unsafe_allow_html=True)


# --------------------------------------------------------------------- main

def render():
    st.set_page_config(page_title="ACME candidate agent", page_icon="*",
                       layout="centered", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    _state()
    _sidebar()

    st.markdown("### ACME internship — application follow-up")
    st.caption("An autonomous agent completes one candidate's application in a single chat. "
               "You are the candidate.")

    banner = st.session_state.banner
    if banner:
        kind, text = banner
        {"error": st.error, "done": st.success, "info": st.info}[kind](text)

    agent = st.session_state.agent
    if agent is None:
        st.info("Enter a candidate ID in the sidebar and press **Start chat**.")
        return

    for role, text in st.session_state.chat:
        with st.chat_message(role, avatar="💬" if role == "assistant" else "🧑"):
            st.write(text)

    outcome = st.session_state.outcome
    if outcome:
        colour = STATUS_COLOUR.get(outcome["status"], "#57606a")
        st.markdown(
            '<span class="pill" style="background:{}">{}</span>'.format(
                colour, outcome["status"].replace("_", " ")),
            unsafe_allow_html=True)
        if outcome.get("written"):
            st.caption("written: " + ", ".join(outcome["written"]))
        if outcome.get("still_needed"):
            st.caption("still needed: " + ", ".join(outcome["still_needed"]))
        if outcome.get("notes"):
            st.caption("notes: " + outcome["notes"])
        return

    if agent.state.ended:
        return

    reply = st.chat_input("Type as the candidate…")
    if reply:
        st.session_state.chat.append(("user", reply))
        agent.reply(reply)
        # The model can call end_conversation on its own turn. If it did, the
        # outcome still has to be written before the loop stops.
        if agent.state.ended and not st.session_state.outcome:
            agent.finalise()
        st.rerun()

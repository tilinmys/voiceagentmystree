"""Tests for the adaptive VAD state machine (agent.DynamicVADController).

The controller toggles the session's endpointing min_delay between fast /
slow / fragment-recovery based on what the agent just asked - via the PUBLIC
session.update_options(endpointing_opts=...) API, never by mutating the
shared Silero VAD instance (which is shared across concurrent calls via
proc.userdata and would contaminate other live calls).

Standalone runnable: python tests/test_adaptive_vad.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {label}")
    else:
        FAIL += 1
        print(f"[FAIL] {label}")


# --- state machine transitions ------------------------------------------------
session = MagicMock()
c = agent.DynamicVADController(session)
check("starts TRANSACTIONAL", c.state == "TRANSACTIONAL")

c.evaluate_agent_text("May I please have your name?")
check("name question -> HIGH_COGNITIVE_LOAD", c.state == "HIGH_COGNITIVE_LOAD")
check(
    "slow delay applied via public API",
    session.update_options.call_args.kwargs["endpointing_opts"]["min_delay"] == 1.0,
)

c.evaluate_agent_text("Is ten thirty tomorrow okay for you?")
check("closed question -> TRANSACTIONAL", c.state == "TRANSACTIONAL")
check(
    "fast delay restored",
    session.update_options.call_args.kwargs["endpointing_opts"]["min_delay"] == 0.35,
)

c.fragment_recovery()
check("fragment -> FRAGMENT_RECOVERY", c.state == "FRAGMENT_RECOVERY")
check(
    "fragment delay applied",
    session.update_options.call_args.kwargs["endpointing_opts"]["min_delay"] == 1.3,
)

c.on_user_turn()
check("fragment window is one-shot", c.state == "TRANSACTIONAL")

n = session.update_options.call_count
c.evaluate_agent_text("okay, booked for tomorrow")
check("idempotent: no redundant update on same state", session.update_options.call_count == n)

# --- failure safety -------------------------------------------------------------
bad = MagicMock()
bad.update_options.side_effect = RuntimeError("session closed")
c2 = agent.DynamicVADController(bad)
c2.evaluate_agent_text("could you spell your name?")
check("dead session fails safe, state unchanged", c2.state == "TRANSACTIONAL")

# --- slow trigger coverage -------------------------------------------------------
for t in [
    "Could you spell that for me?",
    "letter by letter please",
    "what brings you in today?",
    "do you have a specific doctor in mind?",
    "please describe your concern",
]:
    c3 = agent.DynamicVADController(MagicMock())
    c3.evaluate_agent_text(t)
    check(f"slow trigger: {t!r}", c3.state == "HIGH_COGNITIVE_LOAD")

for t in [
    "Is that right?",
    "Booked. Appointment ID 42.",
    "Thank you... anything else?",
]:
    c4 = agent.DynamicVADController(MagicMock())
    c4.evaluate_agent_text("your name please")  # push to slow first
    c4.evaluate_agent_text(t)
    check(f"fast trigger: {t!r}", c4.state == "TRANSACTIONAL")


# --- full hook integration --------------------------------------------------------
async def hook_tests() -> None:
    gracy = agent.GracyAgent(instructions="t", chat_ctx=agent.llm.ChatContext())
    mock_session = MagicMock()
    mock_session.userdata = agent.CallState()
    gracy._activity = MagicMock(session=mock_session)
    vc = agent.DynamicVADController(MagicMock())
    gracy._vad_controller = vc

    tc = MagicMock()
    tc.items = []
    msg = MagicMock()
    msg.text_content = "My name is."  # cut-off fragment
    try:
        await gracy.on_user_turn_completed(tc, msg)
    except agent.StopResponse:
        pass
    check("incomplete fragment engages FRAGMENT_RECOVERY", vc.state == "FRAGMENT_RECOVERY")

    # the continuation turn consumes the window; the fast-path name reply
    # ("May I have your phone number?") is transactional
    msg2 = MagicMock()
    msg2.text_content = "My name is Priya"
    tc2 = MagicMock()
    tc2.items = [MagicMock(role="assistant", text_content="May I please have your name?")]
    try:
        await gracy.on_user_turn_completed(tc2, msg2)
    except agent.StopResponse:
        pass
    check("continuation turn consumes fragment window", vc.state == "TRANSACTIONAL")


asyncio.run(hook_tests())

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

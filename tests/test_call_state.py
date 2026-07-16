"""Tests for the explicit CallState machine (agent.CallState) and its wiring
into GracyAgent.on_user_turn_completed / book_appointment.

Fixes a real bug observed live (2026-07-14): the agent re-asked for the
caller's name and phone number well after both had already been given,
because the only "memory" of what was collected was the LLM re-reading raw
transcript history. CallState is the explicit source of truth instead - this
file proves it updates correctly and that it resolves phone confirmation
deterministically (no LLM round-trip) without ever misfiring on an ordinary
bare yes/no elsewhere in the call.

Standalone runnable: python tests/test_call_state.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import db_helper  # noqa: E402

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


def make_gracy(state: "agent.CallState"):
    gracy = agent.GracyAgent(instructions="test", chat_ctx=agent.llm.ChatContext())
    mock_session = MagicMock()
    mock_session.userdata = state
    gracy._activity = MagicMock(session=mock_session)
    return gracy, mock_session


def make_turn_ctx(last_agent_text: str):
    tc = MagicMock()
    tc.items = [MagicMock(role="assistant", text_content=last_agent_text)]
    return tc


def make_msg(text: str):
    m = MagicMock()
    m.text_content = text
    return m


async def main() -> None:
    # --- summary() formatting ------------------------------------------------
    s = agent.CallState()
    check("empty state shows NOT YET COLLECTED for name", "NOT YET COLLECTED" in s.summary())
    s.name, s.name_confirmed = "Priya", True
    check("name reflected in summary", "Priya (confirmed)" in s.summary())
    s.phone_pending = "7-0-1-2-8-1-2-4-7-6"
    check("pending phone tells the LLM not to re-ask", "do NOT ask for the number again" in s.summary())
    s.phone, s.phone_confirmed, s.phone_pending = s.phone_pending, True, None
    check("confirmed phone reflected in summary", "7-0-1-2-8-1-2-4-7-6 (confirmed)" in s.summary())

    # --- full flow: name -> phone -> deterministic yes confirmation ----------
    state = agent.CallState()
    gracy, session = make_gracy(state)

    tc = make_turn_ctx("Thank you for calling MyStree Clinic... This is Gracy. May I please have your name?")
    try:
        await gracy.on_user_turn_completed(tc, make_msg("My name is Priya"))
    except agent.StopResponse:
        pass
    check("name captured into state", state.name == "Priya" and state.name_confirmed)

    tc = make_turn_ctx("Thanks, Priya. May I have your phone number?")
    try:
        await gracy.on_user_turn_completed(tc, make_msg("7012812476"))
    except agent.StopResponse:
        pass
    check("phone staged as pending confirmation", state.phone_pending is not None and not state.phone_confirmed)

    tc = make_turn_ctx("Thank you... just to confirm, that's 7-0-1-2-8-1-2-4-7-6... is that right?")
    fired = False
    try:
        await gracy.on_user_turn_completed(tc, make_msg("Yes"))
    except agent.StopResponse:
        fired = True
    check("bare yes on pending phone resolved without the LLM", fired)
    check("phone confirmed and stored", state.phone_confirmed and state.phone == "7-0-1-2-8-1-2-4-7-6")
    check("state never regresses: name still set after phone turn", state.name == "Priya")

    # --- negative confirmation -------------------------------------------------
    state2 = agent.CallState()
    state2.phone_pending = "9-9-9-9-9-9-9-9-9-9"
    gracy2, session2 = make_gracy(state2)
    tc = make_turn_ctx("...is that right?")
    fired2 = False
    try:
        await gracy2.on_user_turn_completed(tc, make_msg("No"))
    except agent.StopResponse:
        fired2 = True
    check("bare no on pending phone resolved without the LLM", fired2)
    check("phone_pending cleared, NOT confirmed", state2.phone_pending is None and not state2.phone_confirmed)

    # --- safety boundary: ordinary bare yes/no elsewhere must NOT be intercepted --
    state3 = agent.CallState()  # nothing pending
    gracy3, session3 = make_gracy(state3)
    tc = make_turn_ctx("Do you have a specific doctor in mind, or should I suggest one?")
    fired3 = False
    try:
        await gracy3.on_user_turn_completed(tc, make_msg("Yes"))
    except agent.StopResponse:
        fired3 = True
    check("ordinary yes with nothing pending still falls through to the LLM", not fired3)
    check("no deterministic reply spoken for the ordinary yes case", not session3.say.called)

    # --- LLM-path identity harvesting (the cancellation-flow re-ask bug) -------
    # Caller gives phone INSIDE a sentence -> fast path doesn't fire -> LLM
    # handles the turn and calls lookup_appointments(phone). That tool call
    # must update CallState, or the state summary tells the model to re-ask.
    state5 = agent.CallState()
    ctx5 = MagicMock()
    ctx5.userdata = state5
    agent._harvest_identity(ctx5, phone="7012812476")
    check("tool-arg phone harvested as confirmed", state5.phone_confirmed and state5.phone == "7-0-1-2-8-1-2-4-7-6")
    agent._harvest_identity(ctx5, name="Priya Sharma")
    check("tool-arg name harvested as confirmed", state5.name_confirmed and state5.name == "Priya Sharma")
    # garbage name must never be harvested
    state6 = agent.CallState()
    ctx6 = MagicMock()
    ctx6.userdata = state6
    agent._harvest_identity(ctx6, name="yes", phone="123")
    check("invalid tool args are not harvested", state6.name is None and not state6.phone_confirmed)

    # --- sentence-embedded phone staged as pending on LLM-path turns ----------
    state7 = agent.CallState()
    gracy7, session7 = make_gracy(state7)
    tc7 = make_turn_ctx("How can I help you today?")
    fired7 = False
    try:
        await gracy7.on_user_turn_completed(tc7, make_msg("I want to cancel my appointment, my number is 7012812476"))
    except agent.StopResponse:
        fired7 = True
    check("sentence phone turn still goes to the LLM", not fired7)
    check("sentence phone staged as pending", state7.phone_pending == "7-0-1-2-8-1-2-4-7-6")

    # --- spelled name captured through the full hook ---------------------------
    state8 = agent.CallState()
    gracy8, session8 = make_gracy(state8)
    tc8 = make_turn_ctx("Sorry, could you spell that for me, one letter at a time?")
    fired8 = False
    try:
        await gracy8.on_user_turn_completed(tc8, make_msg("p as in papa, r as in romeo, i as in india, y as in yankee, a as in alpha"))
    except agent.StopResponse:
        fired8 = True
    check("spelled name resolves deterministically", fired8 and state8.name == "Priya")

    # spelled name when phone is ALREADY confirmed must not re-ask for phone
    state9 = agent.CallState()
    state9.phone = "7-0-1-2-8-1-2-4-7-6"
    state9.phone_confirmed = True
    gracy9, session9 = make_gracy(state9)
    fired9 = False
    try:
        await gracy9.on_user_turn_completed(tc8, make_msg("P R I Y A"))
    except agent.StopResponse:
        fired9 = True
    spoken9 = session9.say.call_args[0][0] if session9.say.called else ""
    check("post-phone name capture does not re-ask for phone", fired9 and "phone" not in spoken9.lower())

    # --- book_appointment updates state on success ----------------------------
    state4 = agent.CallState()
    ctx = MagicMock()
    ctx.userdata = state4
    ctx.session = MagicMock()
    today = db_helper.date.today().isoformat()
    # Self-healing: a previously interrupted run can leave this exact slot
    # booked (its cleanup never ran), which fails THIS run's booking and
    # passes the rerun - a confusing flake. Force the slot open first.
    _conn = db_helper._connect()
    _c = _conn.cursor()
    _c.execute(
        "UPDATE slots SET status='available', booked_via=NULL "
        "WHERE doctor_name='Dr. Surbhi Sinha' AND slot_date=? AND slot_time='12:00'",
        (today,),
    )
    _c.execute(
        "DELETE FROM appointments WHERE doctor_name='Dr. Surbhi Sinha' "
        "AND appointment_date=? AND appointment_time='12:00' AND status='Scheduled'",
        (today,),
    )
    _conn.commit()
    _conn.close()
    result = await agent.book_appointment(ctx, "Test State Patient", "7099900001", "Dr. Surbhi Sinha", today, "12:00")
    check("book_appointment returned a booked confirmation", "Booked" in result)
    check("CallState.booking_confirmed set by the tool", state4.booking_confirmed)
    check("CallState.doctor set by the tool", state4.doctor == "Dr. Surbhi Sinha")
    check("CallState.appointment_id set by the tool", state4.appointment_id is not None)

    # cleanup - this is a real DB write, must not linger as fake dashboard data
    conn = db_helper._connect()
    c = conn.cursor()
    c.execute(
        "UPDATE slots SET status='available', booked_via=NULL WHERE doctor_name=? AND slot_date=? AND slot_time=?",
        ("Dr. Surbhi Sinha", today, "12:00"),
    )
    c.execute("DELETE FROM appointments WHERE appointment_id = ?", (state4.appointment_id,))
    c.execute("DELETE FROM patients WHERE phone = ?", ("+917099900001",))
    c.execute("DELETE FROM slot_events WHERE patient_name = ?", ("Test State Patient",))
    conn.commit()
    conn.close()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())

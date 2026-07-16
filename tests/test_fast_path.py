"""Unit tests for the deterministic fast-path router (agent.fast_path_reply).

Standalone runnable: python tests/test_fast_path.py
The router must ONLY fire on unambiguous, state-free turns - a false positive
here means a wrong canned reply on a live clinic call, so the negative cases
matter more than the positive ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import assemble_spelled_name, fast_path_reply, looks_incomplete  # noqa: E402

PHONE_PROMPT = "Thank you... and the best mobile number to reach you on?"
NAME_PROMPT = "Namaste... may I have your name, please?"

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


# --- positive: repeat requests -------------------------------------------
for text in [
    "Can you repeat that?",
    "repeat",
    "sorry, say that again",
    "Pardon?",
    "I didn't catch that",
    "could you please repeat",
]:
    result = fast_path_reply(text, "I have a slot tomorrow at eleven thirty.")
    check(f"repeat fires: {text!r}", result is not None and result[0] == "repeat")

# repeat must include the previous agent utterance
result = fast_path_reply("repeat that", "The slot is at four in the evening.")
check("repeat echoes last agent text", result is not None and "four in the evening" in result[1])

# --- positive: bare phone number after a phone prompt ----------------------
for text in [
    "7012812476",
    "70128 12476",
    "my number is 7012812476",
    "it's 7012-812-476",
    "+91 7012812476",
]:
    result = fast_path_reply(text, PHONE_PROMPT)
    check(f"phone fires: {text!r}", result is not None and result[0] == "phone_confirm")

result = fast_path_reply("7012812476", PHONE_PROMPT)
check("phone reply is digit-by-digit", result is not None and "7-0-1-2-8-1-2-4-7-6" in result[1])

# --- positive: deterministic name collection and incomplete-turn guard -----
for text in [
    "My name is Dylan",
    "My name it Tilin",
    "My names Tilin",
    "My name's Tilin",
    "This is Asha",
    "I am Tilin",
    "I'm Tilin",
    "Tilin",
]:
    result = fast_path_reply(text, NAME_PROMPT)
    check(f"name fires: {text!r}", result is not None and result[0] == "name_captured")

for text in ["My name is Tilin", "My name it Tilin", "My names Tilin"]:
    result = fast_path_reply(text, NAME_PROMPT)
    check(
        f"name introduction removed: {text!r}",
        result == ("name_captured", "Thanks, Tilin. May I have your phone number?"),
    )

for text in ["My name is.", "My number is", "I want to..."]:
    result = fast_path_reply(text, NAME_PROMPT)
    check(f"incomplete waits: {text!r}", result is not None and result[0] == "incomplete")
    check(f"incomplete detector: {text!r}", looks_incomplete(text))

# --- positive: spelled-out names reassembled deterministically --------------
SPELL_PROMPT = "Sorry, could you spell that for me, one letter at a time?"
for text, expected in [
    ("P R I Y A", "Priya"),
    ("P, R, I, Y, A", "Priya"),
    ("p as in papa, r as in romeo, i as in india, y as in yankee, a as in alpha", "Priya"),
    ("A Y E S H A", "Ayesha"),
    ("yeah it's spelled T I L I N", "Tilin"),
    ("N double E T A", "Neeta"),
]:
    check(f"assembler: {text!r} -> {expected}", assemble_spelled_name(text) == expected)
    result = fast_path_reply(text, SPELL_PROMPT)
    check(
        f"spelled name fires after spell prompt: {text!r}",
        result is not None and result[0] == "name_captured" and expected in result[1],
    )

# ordinary sentences must NOT be mistaken for spellings
for text in [
    "I want to book an appointment",
    "yes that works",
    "my name is Priya",
    "a b",  # too short to be a spelling
]:
    check(f"assembler rejects: {text!r}", assemble_spelled_name(text) is None)

# --- negative: things that must NEVER be captured as a patient name ---------
# "Hello? Are you there?" was captured as a name on a live call - every one of
# these must fall through to the LLM (result is None), never "name_captured".
for text in [
    "Hello?",
    "Hello? Are you there?",
    "Are you there?",
    "Are you there",
    "Can you hear me?",
    "Can you hear me",
    "Yeah",
    "Correct",
    "Tomorrow",
    "Appointment",
    "Doctor",
    "Thank you",
    "Nothing",
]:
    result = fast_path_reply(text, NAME_PROMPT)
    check(
        f"never a name: {text!r}",
        result is None or result[0] != "name_captured",
    )

# --- negative: must NOT fire ------------------------------------------------
negatives = [
    # bare yes/no is state-dependent - must go to the LLM
    ("yes", PHONE_PROMPT),
    ("no", PHONE_PROMPT),
    ("correct", PHONE_PROMPT),
    # phone number when the agent did NOT ask for one
    ("7012812476", NAME_PROMPT),
    ("7012812476", ""),
    # number embedded in a longer, meaningful sentence
    ("my number is 7012812476 but call only after 5pm", PHONE_PROMPT),
    ("change it to 7012812476 and cancel tomorrow's booking", PHONE_PROMPT),
    # partial / invalid numbers
    ("70128", PHONE_PROMPT),
    ("call me maybe", PHONE_PROMPT),
    # 'repeat' embedded in a real question
    ("do I need to repeat the scan next month?", PHONE_PROMPT),
    # empty
    ("", PHONE_PROMPT),
    ("   ", PHONE_PROMPT),
    # normal conversational turns
    ("I want to book an appointment", ""),
    ("tomorrow evening", "Which day works for you?"),
]
for text, prompt in negatives:
    result = fast_path_reply(text, prompt)
    check(f"does NOT fire: {text!r} (after {prompt[:24]!r})", result is None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

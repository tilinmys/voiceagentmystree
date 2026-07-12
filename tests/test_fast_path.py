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

from agent import fast_path_reply  # noqa: E402

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
    ("Tilin", NAME_PROMPT),
]
for text, prompt in negatives:
    result = fast_path_reply(text, prompt)
    check(f"does NOT fire: {text!r} (after {prompt[:24]!r})", result is None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

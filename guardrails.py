"""Deterministic safety guardrails for the MyStree voice agent.

Design principle: anything safety-critical must be deterministic, not
generated. A model that never receives a medical question cannot hallucinate
an answer to it (input gate), and a reply that is scanned before synthesis
cannot speak a drug name even if the model produced one (output gate). The
two layers fail independently - both would have to miss for medical content
to reach a caller's ears.

Deliberately dependency-free (regex only, no livekit imports) so it can be
unit-tested standalone and can never add latency beyond a few regex matches.

False-positive discipline: this clinic's LEGITIMATE flow includes symptom
words - the agent is supposed to ask "what brings you in today?" and route
"I have PCOS" to the right doctor. So the input gate triggers on
ADVICE-SEEKING patterns (what medicine / should I take / is it safe to
take / what dose), never on bare symptom mentions. "I have a fever, book me
with a doctor" must flow through untouched; "I have a fever, what medicine
should I take" must never reach the LLM.
"""

from __future__ import annotations

import re

# --- Approved scripts (spoken VERBATIM - never let the LLM improvise a ------
# --- refusal; improvised refusals leak advice while refusing) ---------------

MEDICAL_REFUSAL_SCRIPT = (
    "I'm really not able to advise on medicines or symptoms... only our "
    "doctors can do that safely. What I can do is book you in right away... "
    "shall I find you a slot?"
)

EMERGENCY_SCRIPT = (
    "This sounds urgent... please don't wait for an appointment. Go to the "
    "nearest hospital emergency room right away, or call one-zero-eight for "
    "an ambulance. Please stay safe."
)

SELF_HARM_SCRIPT = (
    "I hear you, and I'm glad you told me... please reach out right now to "
    "someone you trust, or call the Tele-MANAS helpline at "
    "one-four-four-one-six... it is free and always open. If you are in "
    "immediate danger, please call one-zero-eight."
)

JAILBREAK_SCRIPT = (
    "I'm just the receptionist here, so let's stay with your appointment... "
    "would you like to book, change, or cancel one?"
)

ABUSE_WARNING_SCRIPT = (
    "I understand you may be frustrated... I'm here to help with your "
    "appointment. Shall we continue?"
)

ABUSE_GOODBYE_SCRIPT = (
    "I'm sorry, I have to end this call now. You're welcome to call back "
    "anytime. Take care."
)

# --- Input gate patterns ------------------------------------------------------

# Emergencies outrank everything - flat refusal to a bleeding caller is a
# worse failure than the paracetamol case. Never triage severity; escalate.
_EMERGENCY_RE = re.compile(
    r"bleeding\s+(?:a\s+lot|heavily|badly|so\s+much)|heavy\s+bleeding"
    r"|severe\s+(?:pain|bleeding|cramp)|unbearable\s+pain|worst\s+pain"
    r"|collaps(?:ed|ing)|unconscious|fainted|fainting|passed\s+out"
    r"|chest\s+pain|can'?t\s+breathe|cannot\s+breathe|difficulty\s+breathing|not\s+breathing"
    r"|water\s+(?:has\s+|just\s+)?broke|labou?r\s+pain|having\s+(?:a\s+)?seizure",
    re.IGNORECASE,
)

_SELF_HARM_RE = re.compile(
    r"suicid|kill\s+myself|end\s+my\s+life|hurt\s+myself|harm\s+myself"
    r"|self[\s-]?harm|don'?t\s+want\s+to\s+live|no\s+reason\s+to\s+live"
    r"|want\s+to\s+die",
    re.IGNORECASE,
)

_JAILBREAK_RE = re.compile(
    r"ignore\s+(?:all\s+|your\s+|previous\s+|prior\s+|the\s+)*(?:instructions?|prompts?|rules?)"
    r"|system\s+prompt|your\s+instructions?\b|forget\s+(?:your|all|the)\s+(?:instructions?|rules?)"
    r"|pretend\s+(?:you'?re|you\s+are|to\s+be)|act\s+as\s+(?:a|an|the)\s+(?:doctor|nurse|physician|pharmacist)"
    r"|you\s+are\s+now\s+(?:a|an)\b|repeat\s+after\s+me|role[\s-]?play",
    re.IGNORECASE,
)

# Common Indian OTC/prescription drug names heard on clinic calls. A bare
# mention alone ("I'm on metformin") is information-sharing and does NOT
# trigger - it must co-occur with an advice-seeking verb.
_DRUG_WORDS_RE = re.compile(
    r"\b(?:paracetamol|dolo|crocin|calpol|ibuprofen|brufen|combiflam|meftal"
    r"|aspirin|disprin|azithromycin|azithral|amoxicillin|augmentin|antibiotics?"
    r"|metformin|insulin|cetirizine|allegra|pantoprazole|omeprazole|antacids?"
    r"|painkillers?|folic\s+acid|iron\s+tablets?)\b",
    re.IGNORECASE,
)
_ADVICE_VERBS_RE = re.compile(
    r"\b(?:should\s+i|can\s+i|shall\s+i|safe\s+to|how\s+many|how\s+much"
    r"|what\s+dose|dosage|suggest|recommend|prescribe|take|stop|start|skip)\b",
    re.IGNORECASE,
)

# "should/can I take/use/stop/skip..." is only medical when a medicine-shaped
# object exists somewhere in the turn - "can I take my husband along" and
# "can I take the morning slot" are ordinary booking talk and must pass.
_TAKE_ADVICE_RE = re.compile(
    r"\b(?:should|can|shall)\s+i\s+(?:take|use|stop|start|skip|continue)\b", re.IGNORECASE
)
_MEDICINE_NOUNS_RE = re.compile(
    r"\b(?:medicines?|medications?|tablets?|pills?|doses?|dosage|syrups?"
    r"|antibiotics?|painkillers?|injections?|capsules?)\b",
    re.IGNORECASE,
)

_MEDICAL_ADVICE_PATTERNS = [
    # "what/which medicine should I..." / "suggest me a tablet..."
    re.compile(
        r"\b(?:what|which|suggest|recommend|give\s+me|tell\s+me\s+about|any)"
        r"[\w\s,']{0,30}\b(?:medicines?|medications?|tablets?|drugs?|antibiotics?"
        r"|syrups?|dose|dosage|painkillers?|remedy|remedies|injections?)\b",
        re.IGNORECASE,
    ),
    # "is it safe/ok to take/use/eat/drink ..."
    re.compile(r"\bis\s+it\s+(?:safe|ok|okay|fine)\s+to\s+(?:take|use|eat|drink|have)\b", re.IGNORECASE),
    # "is this/that normal" - symptom triage disguised as a yes/no question
    re.compile(r"\bis\s+(?:this|that|it)\s+normal\b", re.IGNORECASE),
    # "home remedy for ..."
    re.compile(r"\bhome\s+remed(?:y|ies)\b", re.IGNORECASE),
    # "what should I do for the fever/pain..." (bounded to symptom objects)
    re.compile(
        r"\bwhat\s+(?:should|can|do)\s+i\s+do\s+(?:for|about)\s+(?:the\s+|my\s+|this\s+)?"
        r"(?:fever|pain|headache|cold|cough|vomit|nausea|acidity|cramps?|swelling|rash)",
        re.IGNORECASE,
    ),
    # test-result interpretation: "is my report normal/bad..."
    re.compile(r"\b(?:is|are)\s+(?:my|the|this|that)\s+(?:reports?|results?|scans?|readings?)\b", re.IGNORECASE),
]

_PROFANITY_RE = re.compile(
    r"\b(?:fuck(?:ing|er)?|bitch|bastard|asshole|motherfucker|dickhead"
    r"|chutiya|madarchod|bhenchod|behenchod|randi|kutti|kamini)\b",
    re.IGNORECASE,
)


def classify_turn(text: str) -> str | None:
    """Classify one caller turn. Returns one of:
    'emergency' | 'self_harm' | 'jailbreak' | 'medical_advice' | 'abuse' | None.

    Order matters: an emergency mentioned alongside anything else is still an
    emergency; self-harm outranks jailbreak/advice; abuse is checked last so
    a frustrated caller describing an emergency still gets the emergency
    path, not a warning.
    """
    t = (text or "").strip()
    if not t:
        return None
    if _EMERGENCY_RE.search(t):
        return "emergency"
    if _SELF_HARM_RE.search(t):
        return "self_harm"
    if _JAILBREAK_RE.search(t):
        return "jailbreak"
    if any(p.search(t) for p in _MEDICAL_ADVICE_PATTERNS):
        return "medical_advice"
    if _DRUG_WORDS_RE.search(t) and _ADVICE_VERBS_RE.search(t):
        return "medical_advice"
    if _TAKE_ADVICE_RE.search(t) and (_MEDICINE_NOUNS_RE.search(t) or _DRUG_WORDS_RE.search(t)):
        return "medical_advice"
    if _PROFANITY_RE.search(t):
        return "abuse"
    return None


SCRIPTS = {
    "emergency": EMERGENCY_SCRIPT,
    "self_harm": SELF_HARM_SCRIPT,
    "jailbreak": JAILBREAK_SCRIPT,
    "medical_advice": MEDICAL_REFUSAL_SCRIPT,
}


# --- Output gate (scan generated text BEFORE it reaches TTS) -----------------
# The agent's legitimate vocabulary contains no drug names and no dosage
# phrasing at all, so this can be aggressive with near-zero false-positive
# risk: if the model ever generates one (jailbreak that survived the input
# gate, or unprompted volunteering), the sentence is cut and replaced.

_OUTPUT_BLOCK_RE = re.compile(
    r"\b\d+\s*(?:mg|milligrams?|ml|millilitres?)\b"
    r"|\b(?:paracetamol|dolo|crocin|calpol|ibuprofen|brufen|combiflam|meftal"
    r"|aspirin|disprin|azithromycin|azithral|amoxicillin|augmentin"
    r"|metformin|cetirizine|allegra|pantoprazole|omeprazole"
    r"|take\s+(?:one|two|three|a|1|2|3)\s+(?:tablets?|pills?|capsules?|spoons?)"
    r"|twice\s+a\s+day\s+(?:before|after)\s+(?:food|meals?))\b",
    re.IGNORECASE,
)

OUTPUT_REPLACEMENT = (
    " ...actually, our doctors are the right people to guide you on that. "
    "Shall I book you an appointment?"
)


def output_flagged(text: str) -> bool:
    return bool(_OUTPUT_BLOCK_RE.search(text or ""))

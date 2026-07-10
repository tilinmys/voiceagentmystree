"""Race-condition and slot-logic tests for the MyStree booking layer.

Simulates the website and the voice agent booking the same slot at the same
moment, plus phone normalization, cancel-frees-slot, and nearest-slot ranking.

Run:  python scripts/test_double_booking.py
"""
import concurrent.futures
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# isolated throwaway database for the test run
os.environ["SQLITE_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_slots.sqlite3")

import db_helper  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {label} {extra}")


def main() -> None:
    db_helper.init_db(reset=True)

    open_slots = db_helper.get_open_slots()
    check("slots seeded", len(open_slots) > 50, f"(open={len(open_slots)})")

    target = open_slots[0]
    doctor, date, time_ = target["doctor_name"], target["slot_date"], target["slot_time"]
    print(f"    racing for: {doctor} {date} {time_}")

    # --- RACE: website vs voice agent booking the exact same slot concurrently ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_agent = pool.submit(db_helper.book_slot, 1, doctor, date, time_, "voice_agent")
        f_web = pool.submit(db_helper.book_slot, 2, doctor, date, time_, "website")
        agent_result = f_agent.result()
        web_result = f_web.result()

    winners = [r for r in (agent_result, web_result) if r[0] is not None]
    losers = [r for r in (agent_result, web_result) if r[0] is None]
    check("exactly one booking wins the race", len(winners) == 1 and len(losers) == 1,
          f"(agent={agent_result}, website={web_result})")
    check("loser is told the slot is taken", losers and losers[0][1] == "taken")

    # --- the slot is really gone from availability ---
    still_open = any(
        s["doctor_name"] == doctor and s["slot_date"] == date and s["slot_time"] == time_
        for s in db_helper.get_open_slots()
    )
    check("won slot no longer listed as open", not still_open)

    # --- booking a nonexistent slot is rejected cleanly ---
    bad_id, bad_reason = db_helper.book_slot(1, doctor, date, "03:15")
    check("nonexistent slot rejected", bad_id is None and bad_reason == "no_such_slot")

    # --- cancel frees the slot for rebooking ---
    appt_id = winners[0][0]
    check("cancel succeeds", db_helper.cancel_appointment(appt_id))
    rebook_id, _ = db_helper.book_slot(3, doctor, date, time_)
    check("freed slot can be rebooked", rebook_id is not None)

    # --- phone normalization ---
    cases = {
        "9876543210": "+919876543210",
        "98765 43210": "+919876543210",
        "+91 98765-43210": "+919876543210",
        "09876543210": "+919876543210",
        "919876543210": "+919876543210",
    }
    ok = all(db_helper.normalize_phone(raw) == want for raw, want in cases.items())
    check("phone normalization variants", ok)
    found = db_helper.get_patient_by_phone("98765 43210")
    check("lookup works with unformatted spoken number", bool(found and found["name"] == "Tilin Bijoy"))

    # --- same-call reschedule: atomic slot swap ---
    open_now = db_helper.get_open_slots()
    a = open_now[0]
    b = next(s for s in open_now[1:] if (s["slot_date"], s["slot_time"]) != (a["slot_date"], a["slot_time"]))
    appt, _ = db_helper.book_slot(1, a["doctor_name"], a["slot_date"], a["slot_time"])
    check("reschedule setup: booked", appt is not None)
    ok, why = db_helper.reschedule_appointment(appt, b["slot_date"], b["slot_time"], b["doctor_name"])
    check("reschedule succeeds", ok, f"(reason={why})")
    opens = {(s["doctor_name"], s["slot_date"], s["slot_time"]) for s in db_helper.get_open_slots()}
    check("old slot freed after reschedule", (a["doctor_name"], a["slot_date"], a["slot_time"]) in opens)
    check("new slot taken after reschedule", (b["doctor_name"], b["slot_date"], b["slot_time"]) not in opens)
    # reschedule into a TAKEN slot must fail and leave the booking unchanged
    c_slot = next(s for s in db_helper.get_open_slots() if (s["slot_date"], s["slot_time"]) != (b["slot_date"], b["slot_time"]))
    blocker, _ = db_helper.book_slot(2, c_slot["doctor_name"], c_slot["slot_date"], c_slot["slot_time"])
    ok2, why2 = db_helper.reschedule_appointment(appt, c_slot["slot_date"], c_slot["slot_time"], c_slot["doctor_name"])
    check("reschedule into taken slot rejected", not ok2 and why2 == "taken", f"(reason={why2})")
    opens2 = {(s["doctor_name"], s["slot_date"], s["slot_time"]) for s in db_helper.get_open_slots()}
    check("failed reschedule keeps current slot booked", (b["doctor_name"], b["slot_date"], b["slot_time"]) not in opens2)
    check("cleanup", db_helper.cancel_appointment(appt) and db_helper.cancel_appointment(blocker))

    # --- Sunday closure ---
    check("Sunday detected as closed", not db_helper.is_clinic_open("2026-07-12"))
    check("Monday detected as open", db_helper.is_clinic_open("2026-07-13"))
    sunday_slots = [s for s in db_helper.get_open_slots() if s["slot_date"] == "2026-07-12"]
    check("no Sunday slots ever seeded", len(sunday_slots) == 0)

    # --- doctor closes slots (leave / schedule change) ---
    open_before = [s for s in db_helper.get_open_slots() if "Surbhi" in s["doctor_name"]]
    day = open_before[0]["slot_date"]
    closed_n = db_helper.close_slots("Dr. Surbhi", day, reason="on leave")
    check("doctor day-off closes open slots", closed_n > 0, f"(closed={closed_n} on {day})")
    still = [s for s in db_helper.get_open_slots() if "Surbhi" in s["doctor_name"] and s["slot_date"] == day]
    check("closed slots vanish from availability", len(still) == 0)
    t_closed = next(s["slot_time"] for s in open_before if s["slot_date"] == day)
    blocked_id, blocked_reason = db_helper.book_slot(1, "Dr. Surbhi", day, t_closed)
    check("booking a closed slot says doctor unavailable",
          blocked_id is None and blocked_reason == "doctor_unavailable", f"(reason={blocked_reason})")
    reopened = db_helper.reopen_slots("Dr. Surbhi", day)
    check("reopen restores the day", reopened == closed_n)
    again_id, _ = db_helper.book_slot(1, "Dr. Surbhi", day, t_closed)
    check("reopened slot is bookable again", again_id is not None)

    # --- cancellation with reason recorded ---
    check("cancel with reason succeeds", db_helper.cancel_appointment(again_id, "travelling out of town"))
    import sqlite3 as _sq
    conn = _sq.connect(db_helper.DB_PATH)
    stored = conn.execute(
        "SELECT cancel_reason FROM appointments WHERE appointment_id = ?", (again_id,)
    ).fetchone()[0]
    conn.close()
    check("cancel reason stored", stored == "travelling out of town")

    # --- concern -> doctor routing across the real 11-member team ---
    cases = [
        ("I am pregnant", "Dr. Swathi S Pai"),
        ("high risk pregnancy with twins", "Dr. Smitha A.P."),
        ("skin rash and hair fall", "Dr Nivetha"),
        ("we are trying to conceive", "Dr. Chaitra Nayak"),
        ("need a diet plan for weight", "Ms. Priyanka Savina"),
        ("feeling a lot of stress and anxiety", "Ms. Jigyasa Thakur"),
        ("need an ultrasound scan", "Dr. Shreyashi Bhattacharyya"),
        ("prenatal yoga classes", "Ms. Nupur Karmarkar"),
        ("fever and cold", "Dr. Priyadarshini Sumanohar"),
        ("back pain after delivery", "Dr. Jasmine Flora"),
        ("something unclear", "Dr. Surbhi Sinha"),  # default: general gynecologist
    ]
    for concern, expected in cases:
        got = db_helper.suggest_doctor_for_concern(concern)["name"]
        check(f"route: {concern[:32]} -> {expected}", got == expected, f"(got {got})")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

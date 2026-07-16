"""Unit tests for the dashboard schedule layer in db_helper:
doctors CRUD, add_slot validation, double-booking/cancel edge cases,
slot_events feed, and the Mon-Fri week schedule query.

Standalone runnable: python tests/test_schedule_db.py
Uses a throwaway SQLite file - never touches data/mystree_demo.sqlite3.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from datetime import date, timedelta
from pathlib import Path

os.environ["SQLITE_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_schedule.sqlite3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db_helper  # noqa: E402

db_helper.init_db()

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


# --- doctors -----------------------------------------------------------------
docs = db_helper.get_doctors()
check("seeded doctors from table", len(docs) == 11)
d, err = db_helper.add_doctor("Dr. Test Kumar", "Cardiologist")
check("add doctor ok", d is not None and err is None)
d2, err2 = db_helper.add_doctor("dr. test kumar", "Dupe")
check("case-insensitive dedup", d2 is None and err2 == "already_exists")
d3, err3 = db_helper.add_doctor("X")
check("too-short name rejected", err3 == "name_too_short")
check("doctor visible in list", any(x["name"] == "Dr. Test Kumar" for x in db_helper.get_doctors()))

# --- add_slot validation ------------------------------------------------------
mon = date.today() + timedelta(days=(7 - date.today().weekday()))  # next Monday
mon_s = mon.isoformat()
ok, err = db_helper.add_slot("Dr. Test Kumar", mon_s, "08:00"); check("add 08:00 ok", ok)
ok, err = db_helper.add_slot("Dr. Test Kumar", mon_s, "19:30"); check("add 19:30 ok", ok)
ok, err = db_helper.add_slot("Dr. Test Kumar", mon_s, "20:00"); check("20:00 rejected (slot must END by 8pm)", not ok and err == "bad_time")
ok, err = db_helper.add_slot("Dr. Test Kumar", mon_s, "07:30"); check("07:30 rejected", err == "bad_time")
ok, err = db_helper.add_slot("Dr. Test Kumar", mon_s, "08:15"); check("non-30min rejected", err == "bad_time")
ok, err = db_helper.add_slot("Dr. Test Kumar", mon_s, "08:00"); check("duplicate rejected", err == "slot_exists")
ok, err = db_helper.add_slot("Dr. Test Kumar", "2020-01-06", "08:00"); check("past date rejected", err == "date_in_past")
sunday = (mon + timedelta(days=6)).isoformat()
ok, err = db_helper.add_slot("Dr. Test Kumar", sunday, "08:00"); check("sunday rejected", err == "sunday_closed")
ok, err = db_helper.add_slot("Dr. Nobody", mon_s, "09:00"); check("unknown doctor rejected", err == "no_such_doctor")
ok, err = db_helper.add_slot("dr. test kumar", mon_s, "09:00"); check("case-insensitive doctor match", ok)

# --- booking + slot events ----------------------------------------------------
pat = db_helper.get_or_create_patient("Asha Verma", "9000011111")
appt, reason = db_helper.book_slot(pat["patient_id"], "Dr. Test Kumar", mon_s, "08:00")
check("book ok", appt is not None)
appt2, reason2 = db_helper.book_slot(pat["patient_id"], "Dr. Test Kumar", mon_s, "08:00")
check("double booking blocked", appt2 is None and reason2 == "taken")

# concurrent race: two threads fight for the same slot; exactly one may win
results: list = []


def race() -> None:
    p = db_helper.get_or_create_patient("Race Tester", "9000022222")
    results.append(db_helper.book_slot(p["patient_id"], "Dr. Test Kumar", mon_s, "19:30"))


t1, t2 = threading.Thread(target=race), threading.Thread(target=race)
t1.start(); t2.start(); t1.join(); t2.join()
wins = [r for r in results if r[0] is not None]
check("concurrent race: exactly one winner", len(wins) == 1)

ev = db_helper.get_slot_events(0)
booked_evs = [e for e in ev["events"] if e["event_type"] == "booked"]
check("booked events recorded with names", len(booked_evs) == 2 and booked_evs[0]["patient_name"] == "Asha Verma")

# --- reschedule + cancel events -------------------------------------------------
okr, rr = db_helper.reschedule_appointment(appt, mon_s, "09:00")
check("reschedule ok", okr)
ev2 = db_helper.get_slot_events(ev["next"])
kinds = [e["event_type"] for e in ev2["events"]]
check("reschedule = cancelled+booked events", kinds == ["cancelled", "booked"])
okc = db_helper.cancel_appointment(appt, "test cancel")
check("cancel ok", okc)
ev3 = db_helper.get_slot_events(ev2["next"])
check("cancel event recorded", ev3["events"][0]["event_type"] == "cancelled" and ev3["events"][0]["patient_name"] == "Asha Verma")
okc2 = db_helper.cancel_appointment(appt, "again")
check("double cancel blocked", not okc2)

# --- week schedule --------------------------------------------------------------
sched = db_helper.get_week_schedule("Dr. Test Kumar", mon_s)
check("week has 5 days", len(sched["days"]) == 5)
check("grid exposes 24 half-hour rows (08:00-19:30)", len(sched["times"]) == 24)
check("09:00 slot shows available after cancel", sched["grid"][mon_s]["09:00"]["status"] == "available")
booked_slot = sched["grid"][mon_s].get("19:30")
check("19:30 booked with patient name", bool(booked_slot) and booked_slot["status"] == "booked" and booked_slot["patient_name"] == "Race Tester")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

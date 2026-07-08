"""Clinic-side slot management CLI — simulates the backend a doctor/admin would use.

The voice agent's slot cache refreshes every SLOT_CACHE_REFRESH_SECONDS (default 10s),
so any change made here reaches live calls within seconds.

Usage:
  python scripts/manage_slots.py list [DATE]
  python scripts/manage_slots.py close  "Dr. Anita" [DATE] [TIME] [--reason "on leave"]
  python scripts/manage_slots.py reopen "Dr. Anita" [DATE] [TIME]

Examples:
  python scripts/manage_slots.py close "Dr. Anita" 2026-07-09              # whole day off
  python scripts/manage_slots.py close "Dr. Rajesh" 2026-07-08 17:00       # one slot
  python scripts/manage_slots.py reopen "Dr. Anita" 2026-07-09
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db_helper


def cmd_list(args) -> None:
    conn = sqlite3.connect(db_helper.DB_PATH)
    cursor = conn.cursor()
    query = "SELECT doctor_name, slot_date, slot_time, status, booked_via FROM slots"
    params = []
    if args.date:
        query += " WHERE slot_date = ?"
        params.append(args.date)
    query += " ORDER BY slot_date, slot_time, doctor_name"
    counts: dict[str, int] = {}
    for doctor, date, time_, status, via in cursor.execute(query, params):
        counts[status] = counts.get(status, 0) + 1
        if args.date:
            print(f"{date} {time_}  {doctor:<12} {status:<10} {via or ''}")
    conn.close()
    print("totals:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def cmd_close(args) -> None:
    n = db_helper.close_slots(args.doctor, args.date, args.time, args.reason)
    print(f"closed {n} slot(s) for {args.doctor}"
          + (f" on {args.date}" if args.date else "")
          + (f" at {args.time}" if args.time else ""))


def cmd_reopen(args) -> None:
    n = db_helper.reopen_slots(args.doctor, args.date, args.time)
    print(f"reopened {n} slot(s) for {args.doctor}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show slot counts, or all slots for a date")
    p_list.add_argument("date", nargs="?", default=None)
    p_list.set_defaults(fn=cmd_list)

    p_close = sub.add_parser("close", help="close a doctor's open slots")
    p_close.add_argument("doctor")
    p_close.add_argument("date", nargs="?", default=None)
    p_close.add_argument("time", nargs="?", default=None)
    p_close.add_argument("--reason", default="doctor unavailable")
    p_close.set_defaults(fn=cmd_close)

    p_reopen = sub.add_parser("reopen", help="reopen a doctor's closed slots")
    p_reopen.add_argument("doctor")
    p_reopen.add_argument("date", nargs="?", default=None)
    p_reopen.add_argument("time", nargs="?", default=None)
    p_reopen.set_defaults(fn=cmd_reopen)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

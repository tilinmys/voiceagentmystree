import logging
import os
import random
import re
import sqlite3
from datetime import date, datetime, timedelta

logger = logging.getLogger("db_helper")

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/mystree_demo.sqlite3")

# The real MyStree Indiranagar clinical team (provided by the clinic).
# `concerns` keywords route a caller's stated concern to the right specialist.
DOCTORS = [
    {
        "name": "Dr. Smitha A.P.",
        "speciality": "High Risk Obstetrician and Fertility Expert",
        "concerns": ["high risk", "risky pregnancy", "complicated pregnancy", "twins", "miscarriage"],
    },
    {
        "name": "Dr. Surbhi Sinha",
        "speciality": "Gynecologist & Fertility Specialist, Obstetrician",
        "concerns": ["gynec", "pcos", "period", "periods", "menstrual", "menopause", "white discharge"],
    },
    {
        "name": "Ms. Priyanka Savina",
        "speciality": "Therapist, Dietitian, Nutritionist",
        "concerns": ["diet", "nutrition", "weight", "food", "eating", "thyroid diet", "obesity"],
    },
    {
        "name": "Dr. Chaitra Nayak",
        "speciality": "Infertility Specialist & Reproductive Endocrinologist",
        "concerns": ["infertility", "fertility", "ivf", "iui", "conceive", "conception", "hormone", "family planning"],
    },
    {
        "name": "Dr. Priyadarshini Sumanohar",
        "speciality": "General Physician",
        "concerns": ["fever", "cold", "cough", "general", "checkup", "bp", "blood pressure", "diabetes", "sugar"],
    },
    {
        "name": "Dr. Swathi S Pai",
        "speciality": "Obstetrics & Gynaecology",
        "concerns": ["pregnancy", "pregnant", "prenatal", "antenatal", "delivery", "obstetric", "trimester"],
    },
    {
        "name": "Dr. Jasmine Flora",
        "speciality": "Obstetrics and Gynaecology Physiotherapy",
        "concerns": ["physio", "physiotherapy", "back pain", "pelvic pain", "postnatal exercise", "posture"],
    },
    {
        "name": "Dr Nivetha",
        "speciality": "Dermatologist",
        "concerns": ["skin", "hair", "acne", "pimple", "dermat", "rash", "pigmentation", "hair fall"],
    },
    {
        "name": "Dr. Shreyashi Bhattacharyya",
        "speciality": "Radiologist",
        "concerns": ["scan", "ultrasound", "sonography", "x-ray", "xray", "anomaly scan", "imaging"],
    },
    {
        "name": "Ms. Nupur Karmarkar",
        "speciality": "Certified Yoga Therapist",
        "concerns": ["yoga", "prenatal yoga", "garbh sanskar", "breathing", "meditation"],
    },
    {
        "name": "Ms. Jigyasa Thakur",
        "speciality": "Consultant Psychologist & Women's Mental Health Specialist",
        "concerns": ["mental", "stress", "anxiety", "depression", "counselling", "counseling", "postpartum depression", "mood", "sleep"],
    },
]

# Clinic slot grid: Mon-Sat, morning and evening OPD, 30-minute slots.
SLOT_TIMES = [
    "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
    "17:00", "17:30", "18:00", "18:30", "19:00", "19:30",
]
SLOT_DAYS_AHEAD = 7
PREBOOKED_RATIO = 0.35  # simulate a busy clinic: ~35% of slots already taken via website


def normalize_phone(phone: str) -> str:
    """Normalize any spoken/typed Indian phone to +91XXXXXXXXXX; return input if not parseable."""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10:
        return "+91" + digits
    return (phone or "").strip()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(reset=False):
    dir_name = os.path.dirname(DB_PATH)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f"Created directory for database: {dir_name}")

    if reset and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            logger.info("Removed existing SQLite database file for reset.")
        except Exception as e:
            logger.error(f"Error resetting database file: {e}")

    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT UNIQUE NOT NULL,
        dob TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        doctor_name TEXT NOT NULL,
        appointment_date TEXT NOT NULL,
        appointment_time TEXT NOT NULL,
        status TEXT DEFAULT 'Scheduled',
        cancel_reason TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    """)
    # migration for databases created before cancel_reason existed
    try:
        cursor.execute("ALTER TABLE appointments ADD COLUMN cancel_reason TEXT")
    except sqlite3.OperationalError:
        pass

    # One row per bookable slot. The UNIQUE constraint plus the guarded UPDATE in
    # book_slot() is what makes double booking impossible, even when the website
    # and the voice agent race for the same slot.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS slots (
        slot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_name TEXT NOT NULL,
        slot_date TEXT NOT NULL,
        slot_time TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'available',
        booked_via TEXT,
        UNIQUE (doctor_name, slot_date, slot_time)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS call_reports (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_name TEXT,
        caller_phone TEXT,
        patient_id INTEGER,
        call_summary TEXT NOT NULL,
        user_sentiment TEXT NOT NULL,
        follow_up_required INTEGER NOT NULL DEFAULT 0,
        report_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    """)

    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        logger.info("Seeding mock patients...")
        mock_patients = [
            ("Tilin Bijoy", "+919876543210", "1995-05-15"),
            ("Vinayak Sharma", "+919999988888", "1988-12-01"),
            ("Priya Patel", "+918888877777", "1992-08-20"),
        ]
        cursor.executemany("INSERT INTO patients (name, phone, dob) VALUES (?, ?, ?)", mock_patients)

    _seed_slots(cursor)
    _ensure_demo_followup_patient(cursor)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


def _seed_slots(cursor) -> None:
    """Create the slot grid for the coming week; pre-book a share to simulate website traffic."""
    today = date.today()
    rng = random.Random(20260707)  # deterministic seeding so tests are reproducible
    created = 0
    for day_offset in range(SLOT_DAYS_AHEAD):
        slot_day = today + timedelta(days=day_offset)
        if slot_day.weekday() == 6:  # clinic closed on Sunday
            continue
        for doctor in DOCTORS:
            for slot_time in SLOT_TIMES:
                prebooked = rng.random() < PREBOOKED_RATIO
                status = "booked" if prebooked else "available"
                booked_via = "website" if prebooked else None
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO slots (doctor_name, slot_date, slot_time, status, booked_via)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doctor["name"], slot_day.isoformat(), slot_time, status, booked_via),
                )
                created += cursor.rowcount
    if created:
        logger.info(f"Seeded {created} clinic slots for the next {SLOT_DAYS_AHEAD} days.")


def _ensure_demo_followup_patient(cursor) -> None:
    """Keep a deterministic prior follow-up patient available in every DB."""
    phone = normalize_phone("7012812476")
    cursor.execute(
        "INSERT OR IGNORE INTO patients (name, phone, dob) VALUES (?, ?, ?)",
        ("Angel", phone, None),
    )
    cursor.execute("UPDATE patients SET name = ?, dob = NULL WHERE phone = ?", ("Angel", phone))
    cursor.execute("SELECT patient_id FROM patients WHERE phone = ?", (phone,))
    row = cursor.fetchone()
    if not row:
        return
    patient_id = row[0]
    prior_date = (date.today() - timedelta(days=21)).isoformat()
    cursor.execute(
        "SELECT appointment_id FROM appointments WHERE patient_id = ? AND status = 'Completed' LIMIT 1",
        (patient_id,),
    )
    if cursor.fetchone():
        return
    cursor.execute(
        "INSERT INTO appointments (patient_id, doctor_name, appointment_date, appointment_time, status) "
        "VALUES (?, ?, ?, ?, 'Completed')",
        (patient_id, "Dr. Surbhi Sinha", prior_date, "11:00"),
    )

def get_patient_by_phone(phone: str):
    normalized = normalize_phone(phone)
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE phone IN (?, ?)", (normalized, (phone or "").strip()))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_patients_by_name(name: str, phone: str | None = None, limit: int = 5):
    """Find patients by spoken name, optionally narrowing with phone."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    name_like = f"%{(name or '').strip()}%"
    normalized_phone = normalize_phone(phone or "") if phone else None
    if normalized_phone:
        cursor.execute(
            "SELECT * FROM patients WHERE lower(name) LIKE lower(?) AND phone IN (?, ?) ORDER BY patient_id DESC LIMIT ?",
            (name_like, normalized_phone, (phone or "").strip(), limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM patients WHERE lower(name) LIKE lower(?) ORDER BY patient_id DESC LIMIT ?",
            (name_like, limit),
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_or_create_patient(name: str, phone: str) -> dict:
    """Return an existing patient by phone or create a lightweight record without DOB."""
    normalized = normalize_phone(phone)
    conn = _connect()
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE phone IN (?, ?)", (normalized, (phone or "").strip()))
        row = cursor.fetchone()
        if row:
            conn.commit()
            return dict(row)
        cursor.execute(
            "INSERT INTO patients (name, phone, dob) VALUES (?, ?, NULL)",
            ((name or "Patient").strip(), normalized),
        )
        patient_id = cursor.lastrowid
        conn.commit()
        return {"patient_id": patient_id, "name": (name or "Patient").strip(), "phone": normalized, "dob": None}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_visit_history_by_patient_id(patient_id: int, limit: int = 3):
    """Return most recent completed or scheduled visits for follow-up routing."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    today = date.today().isoformat()
    cursor.execute(
        "SELECT * FROM appointments WHERE patient_id = ? "
        "AND status != 'Cancelled' AND (status = 'Completed' OR appointment_date <= ?) "
        "ORDER BY appointment_date DESC, appointment_time DESC LIMIT ?",
        (patient_id, today, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def register_patient(name: str, phone: str, dob: str) -> int:
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO patients (name, phone, dob) VALUES (?, ?, ?)",
            (name, normalize_phone(phone), dob),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_appointments_by_patient_id(patient_id: int):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM appointments WHERE patient_id = ? AND status = 'Scheduled' "
        "ORDER BY appointment_date, appointment_time",
        (patient_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient_context_by_phone(phone: str):
    """Fetch caller context for zero-latency greeting/prompt preload."""
    patient = get_patient_by_phone(phone)
    if not patient:
        return None
    return {
        "patient": patient,
        "appointments": get_appointments_by_patient_id(patient["patient_id"]),
        "history": get_visit_history_by_patient_id(patient["patient_id"], limit=3),
    }


def save_call_report(report: dict) -> int:
    """Persist post-call analysis without blocking the live conversation."""
    import json

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO call_reports (
                room_name, caller_phone, patient_id, call_summary,
                user_sentiment, follow_up_required, report_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.get("room_name"),
                normalize_phone(report.get("caller_phone", "")) if report.get("caller_phone") else None,
                report.get("patient_id"),
                report.get("call_summary") or "No summary available.",
                report.get("user_sentiment") or "unknown",
                1 if report.get("follow_up_required") else 0,
                json.dumps(report, ensure_ascii=True),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_open_slots():
    """All future available slots — feeds the in-memory cache the agent reads from."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    now = datetime.now()
    cursor.execute(
        "SELECT doctor_name, slot_date, slot_time FROM slots WHERE status = 'available' "
        "ORDER BY slot_date, slot_time"
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    # keep only slots in the future
    out = []
    for r in rows:
        try:
            slot_dt = datetime.strptime(f"{r['slot_date']} {r['slot_time']}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if slot_dt > now:
            out.append(r)
    return out


def book_slot(patient_id: int, doctor_name: str, slot_date: str, slot_time: str, via: str = "voice_agent"):
    """Atomically claim a slot and create the appointment.

    Returns (appointment_id, None) on success or (None, reason) on failure.
    The guarded UPDATE ... WHERE status='available' inside BEGIN IMMEDIATE is the
    double-booking protection: whichever caller commits first wins, the other
    sees rowcount 0.
    """
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE slots SET status = 'booked', booked_via = ? "
            "WHERE doctor_name LIKE ? AND slot_date = ? AND slot_time = ? AND status = 'available'",
            (via, f"%{doctor_name}%", slot_date, slot_time),
        )
        if cursor.rowcount == 0:
            conn.rollback()
            cursor.execute(
                "SELECT status FROM slots WHERE doctor_name LIKE ? AND slot_date = ? AND slot_time = ?",
                (f"%{doctor_name}%", slot_date, slot_time),
            )
            row = cursor.fetchone()
            if row is None:
                reason = "no_such_slot"
            elif row[0] == "closed":
                reason = "doctor_unavailable"
            else:
                reason = "taken"
            return None, reason
        cursor.execute(
            "INSERT INTO appointments (patient_id, doctor_name, appointment_date, appointment_time, status) "
            "VALUES (?, ?, ?, ?, 'Scheduled')",
            (patient_id, doctor_name, slot_date, slot_time),
        )
        appointment_id = cursor.lastrowid
        conn.commit()
        return appointment_id, None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reschedule_appointment(appointment_id: int, new_date: str, new_time: str, new_doctor: str | None = None):
    """Atomic reschedule: claim the new slot and free the old one in ONE transaction.

    Returns (True, None) on success or (False, reason) where reason is
    'not_found' | 'taken' | 'doctor_unavailable' | 'no_such_slot'.

    Both slot flips happen inside a single BEGIN IMMEDIATE, so there is no
    window where the caller holds zero slots or two slots, and a website
    booking racing for the new slot simply wins or loses the guarded UPDATE.
    """
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT doctor_name, appointment_date, appointment_time FROM appointments "
            "WHERE appointment_id = ? AND status = 'Scheduled'",
            (appointment_id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.rollback()
            return False, "not_found"
        old_doctor, old_date, old_time = row
        doctor = new_doctor or old_doctor

        # 1) claim the NEW slot (guarded — loses cleanly to concurrent bookings)
        cursor.execute(
            "UPDATE slots SET status = 'booked', booked_via = 'voice_agent' "
            "WHERE doctor_name LIKE ? AND slot_date = ? AND slot_time = ? AND status = 'available'",
            (f"%{doctor}%", new_date, new_time),
        )
        if cursor.rowcount == 0:
            cursor.execute(
                "SELECT status FROM slots WHERE doctor_name LIKE ? AND slot_date = ? AND slot_time = ?",
                (f"%{doctor}%", new_date, new_time),
            )
            status_row = cursor.fetchone()
            conn.rollback()
            if status_row is None:
                return False, "no_such_slot"
            return False, "doctor_unavailable" if status_row[0] == "closed" else "taken"

        # 2) free the OLD slot and move the appointment
        cursor.execute(
            "UPDATE slots SET status = 'available', booked_via = NULL "
            "WHERE doctor_name = ? AND slot_date = ? AND slot_time = ?",
            (old_doctor, old_date, old_time),
        )
        cursor.execute(
            "UPDATE appointments SET doctor_name = ?, appointment_date = ?, appointment_time = ? "
            "WHERE appointment_id = ?",
            (doctor, new_date, new_time, appointment_id),
        )
        conn.commit()
        return True, None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def cancel_appointment(appointment_id: int, reason: str | None = None):
    """Cancel (recording an optional caller-given reason) and release the slot."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT doctor_name, appointment_date, appointment_time FROM appointments "
            "WHERE appointment_id = ? AND status = 'Scheduled'",
            (appointment_id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.rollback()
            return False
        cursor.execute(
            "UPDATE appointments SET status = 'Cancelled', cancel_reason = ? WHERE appointment_id = ?",
            (reason, appointment_id),
        )
        cursor.execute(
            "UPDATE slots SET status = 'available', booked_via = NULL "
            "WHERE doctor_name = ? AND slot_date = ? AND slot_time = ?",
            (row[0], row[1], row[2]),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_doctors():
    return [{"name": d["name"], "speciality": d["speciality"]} for d in DOCTORS]


def suggest_doctor_for_concern(concern: str):
    """Keyword-match a health concern to the right specialist.

    Longest-matching keyword wins so 'high risk pregnancy' routes to the high-risk
    obstetrician instead of the general pregnancy match. Unmatched concerns default
    to the general gynecologist — the safest first door for a women's clinic.
    """
    text = (concern or "").lower()
    best, best_len = None, 0
    for doctor in DOCTORS:
        for k in doctor["concerns"]:
            if k in text and len(k) > best_len:
                best, best_len = doctor, len(k)
    chosen = best or next(d for d in DOCTORS if d["name"] == "Dr. Surbhi Sinha")
    return {"name": chosen["name"], "speciality": chosen["speciality"]}


def is_clinic_open(date_str: str) -> bool:
    """Clinic is closed on Sundays."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() != 6
    except ValueError:
        return True


def close_slots(doctor_name: str, slot_date: str | None = None, slot_time: str | None = None,
                reason: str = "doctor unavailable") -> int:
    """Doctor closes some/all of their open slots (leave, schedule change).

    Booked appointments are untouched — only 'available' slots become 'closed'.
    Returns how many slots were closed. The agent's cache picks this up on its
    next refresh, so callers are told the doctor is unavailable within seconds.
    """
    conn = _connect()
    try:
        query = "UPDATE slots SET status = 'closed', booked_via = ? WHERE doctor_name LIKE ? AND status = 'available'"
        params: list = [reason, f"%{doctor_name}%"]
        if slot_date:
            query += " AND slot_date = ?"
            params.append(slot_date)
        if slot_time:
            query += " AND slot_time = ?"
            params.append(slot_time)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def reopen_slots(doctor_name: str, slot_date: str | None = None, slot_time: str | None = None) -> int:
    """Reopen previously closed slots for a doctor. Returns how many reopened."""
    conn = _connect()
    try:
        query = "UPDATE slots SET status = 'available', booked_via = NULL WHERE doctor_name LIKE ? AND status = 'closed'"
        params: list = [f"%{doctor_name}%"]
        if slot_date:
            query += " AND slot_date = ?"
            params.append(slot_date)
        if slot_time:
            query += " AND slot_time = ?"
            params.append(slot_time)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_booking_timings(doctor_name: str, appointment_date: str):
    """Available times for a doctor on a date, from the slots table."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT slot_time FROM slots WHERE doctor_name LIKE ? AND slot_date = ? AND status = 'available' "
        "ORDER BY slot_time",
        (f"%{doctor_name}%", appointment_date),
    )
    times = [r[0] for r in cursor.fetchall()]
    conn.close()
    return times

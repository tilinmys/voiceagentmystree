import os
import sqlite3
import logging

logger = logging.getLogger("db_helper")

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/mystree_demo.sqlite3")

def init_db(reset=False):
    # Ensure parent directory exists
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

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
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
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    """)

    # Seed mock data if empty
    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        logger.info("Seeding mock database tables with patients and appointments...")
        mock_patients = [
            ("Tilin Bijoy", "+919876543210", "1995-05-15"),
            ("Vinayak Sharma", "+919999988888", "1988-12-01"),
            ("Priya Patel", "+918888877777", "1992-08-20")
        ]
        cursor.executemany("INSERT INTO patients (name, phone, dob) VALUES (?, ?, ?)", mock_patients)

        mock_appointments = [
            (1, "Dr. Anita (Gynecologist)", "2026-07-08", "10:00 AM", "Scheduled"),
            (2, "Dr. Rajesh (Cardiologist)", "2026-07-10", "02:30 PM", "Scheduled"),
            (3, "Dr. Sunita (Pediatrician)", "2026-07-09", "11:30 AM", "Scheduled")
        ]
        cursor.executemany("""
        INSERT INTO appointments (patient_id, doctor_name, appointment_date, appointment_time, status)
        VALUES (?, ?, ?, ?, ?)
        """, mock_appointments)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def get_patient_by_phone(phone: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE phone = ?", (phone,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_appointments_by_patient_id(patient_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM appointments WHERE patient_id = ? AND status = 'Scheduled'", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def book_appointment(patient_id: int, doctor_name: str, appointment_date: str, appointment_time: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO appointments (patient_id, doctor_name, appointment_date, appointment_time, status)
    VALUES (?, ?, ?, ?, 'Scheduled')
    """, (patient_id, doctor_name, appointment_date, appointment_time))
    appointment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return appointment_id

def cancel_appointment(appointment_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE appointments SET status = 'Cancelled' WHERE appointment_id = ?", (appointment_id,))
    changes = conn.total_changes
    conn.commit()
    conn.close()
    return changes > 0

def get_doctors():
    return [
        {"name": "Dr. Anita", "speciality": "Gynecologist"},
        {"name": "Dr. Rajesh", "speciality": "Cardiologist"},
        {"name": "Dr. Sunita", "speciality": "Pediatrician"}
    ]

def get_booking_timings(doctor_name: str, appointment_date: str):
    all_slots = ["10:00 AM", "11:30 AM", "02:30 PM", "04:00 PM"]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Support lookup with partial name match e.g. "Anita" matches "Dr. Anita (Gynecologist)"
    cursor.execute("""
        SELECT appointment_time FROM appointments 
        WHERE doctor_name LIKE ? AND appointment_date = ? AND status = 'Scheduled'
    """, (f"%{doctor_name}%", appointment_date))
    booked_slots = [r[0] for r in cursor.fetchall()]
    conn.close()
    
    available_slots = [s for s in all_slots if s not in booked_slots]
    return available_slots


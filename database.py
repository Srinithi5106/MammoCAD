"""
database.py — All database operations for MammoCAD
Tables: users, patients, analyses
"""
import sqlite3
import bcrypt
import os
import json
from datetime import datetime
from config import DB_PATH


# ══════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('doctor','lab_assistant')),
    full_name   TEXT,
    email       TEXT DEFAULT '',
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT UNIQUE NOT NULL,
    full_name       TEXT NOT NULL,
    age             INTEGER,
    gender          TEXT DEFAULT 'Female',
    contact         TEXT,
    history         TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    created_by      TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          TEXT NOT NULL,
    image_path          TEXT,
    prediction          TEXT,
    benign_prob         REAL,
    malignant_prob      REAL,
    birads_category     TEXT,
    birads_desc         TEXT,
    features_json       TEXT,
    notes               TEXT,
    analysed_by         TEXT,
    analysed_at         TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
"""

# Migration: add email column if it doesn't exist (for existing DBs)
MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''",
]


# ══════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables, run migrations, and seed default users."""
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()

    # Run migrations safely (ignore if column already exists)
    for migration in MIGRATIONS:
        try:
            conn.execute(migration)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    _seed_default_users(conn)
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


def _seed_default_users(conn):
    defaults = [
        ("doctor1",   "doc123",  "doctor",        "Dr. Aishwarya Rajan",  ""),
        ("labtech1",  "lab123",  "lab_assistant",  "Lab Tech Priya Nair",  ""),
    ]
    for username, pw, role, name, email in defaults:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=?", (username,)
        ).fetchone()
        if not existing:
            hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (username, password, role, full_name, email) VALUES (?,?,?,?,?)",
                (username, hashed, role, name, email)
            )
    conn.commit()


# ══════════════════════════════════════════════════════════════
# Auth
# ══════════════════════════════════════════════════════════════

def verify_user(username: str, password: str):
    """Returns user dict or None."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    conn.close()
    if row and bcrypt.checkpw(password.encode(), row["password"].encode()):
        return dict(row)
    return None


def register_user(username: str, password: str, role: str,
                  full_name: str, email: str = "") -> tuple[bool, str]:
    """
    Register a new user account.
    Returns (True, "success message") or (False, "error message").
    """
    # Check username isn't taken
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM users WHERE username=?", (username,)
    ).fetchone()

    if existing:
        conn.close()
        return False, f"Username '{username}' is already taken. Please choose another."

    # Hash password and insert
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password, role, full_name, email) VALUES (?,?,?,?,?)",
            (username, hashed, role, full_name, email)
        )
        conn.commit()
        conn.close()
        return True, f"Account created for {full_name}."
    except Exception as e:
        conn.close()
        return False, f"Registration failed: {str(e)}"


def get_all_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, role, full_name, email, created_at FROM users"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
# Patients
# ══════════════════════════════════════════════════════════════

def add_patient(patient_id, full_name, age, contact, history, created_by):
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO patients (patient_id, full_name, age, contact, history, created_by)
               VALUES (?,?,?,?,?,?)""",
            (patient_id, full_name, age, contact, history, created_by)
        )
        conn.commit()
        return True, "Patient added."
    except sqlite3.IntegrityError:
        return False, "Patient ID already exists."
    finally:
        conn.close()


def get_patient(patient_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_patients():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_patients(query):
    conn = get_conn()
    q = f"%{query}%"
    rows = conn.execute(
        "SELECT * FROM patients WHERE patient_id LIKE ? OR full_name LIKE ? ORDER BY created_at DESC",
        (q, q)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
# Analyses
# ══════════════════════════════════════════════════════════════

def save_analysis(patient_id, image_path, prediction, benign_prob, malignant_prob,
                  birads_category, birads_desc, features_dict, notes, analysed_by):
    conn = get_conn()
    conn.execute(
        """INSERT INTO analyses
           (patient_id, image_path, prediction, benign_prob, malignant_prob,
            birads_category, birads_desc, features_json, notes, analysed_by)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            patient_id, image_path, prediction, benign_prob, malignant_prob,
            birads_category, birads_desc,
            json.dumps(features_dict), notes, analysed_by
        )
    )
    conn.commit()
    conn.close()


def get_analyses_for_patient(patient_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM analyses WHERE patient_id=? ORDER BY analysed_at DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features_json"]) if d["features_json"] else {}
        result.append(d)
    return result


def get_all_analyses():
    conn = get_conn()
    rows = conn.execute("""
        SELECT a.*, p.full_name, p.age
        FROM analyses a
        JOIN patients p ON a.patient_id = p.patient_id
        ORDER BY a.analysed_at DESC
    """).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features_json"]) if d["features_json"] else {}
        result.append(d)
    return result


def get_latest_analysis(patient_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM analyses WHERE patient_id=? ORDER BY analysed_at DESC LIMIT 1",
        (patient_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["features"] = json.loads(d["features_json"]) if d["features_json"] else {}
        return d
    return None


def get_stats():
    """Dashboard statistics."""
    conn = get_conn()
    total_patients  = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_analyses  = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    malignant_count = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE prediction='Malignant'"
    ).fetchone()[0]
    benign_count    = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE prediction='Benign'"
    ).fetchone()[0]
    conn.close()
    return {
        "total_patients":  total_patients,
        "total_analyses":  total_analyses,
        "malignant_count": malignant_count,
        "benign_count":    benign_count,
    }


# ══════════════════════════════════════════════════════════════
# Run directly to initialize
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    print("[DB] Default users seeded.")
    print("  doctor1  / doc123  (doctor)")
    print("  labtech1 / lab123  (lab_assistant)")
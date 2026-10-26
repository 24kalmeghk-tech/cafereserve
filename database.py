"""SQLite storage for Cafe Insight demo reservations."""
from pathlib import Path
import sqlite3
from contextlib import contextmanager

DB_PATH = Path(__file__).with_name("cafe_insight.db")

@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                booking_date TEXT NOT NULL,
                booking_time TEXT NOT NULL,
                party_size INTEGER NOT NULL CHECK (party_size BETWEEN 1 AND 30),
                phone TEXT DEFAULT '',
                advance_days INTEGER NOT NULL DEFAULT 0,
                past_visits INTEGER NOT NULL DEFAULT 0,
                estimated_spend REAL NOT NULL DEFAULT 0,
                weather TEXT NOT NULL DEFAULT 'Sunny',
                status TEXT NOT NULL DEFAULT 'Booked',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

def add_reservation(customer_name, booking_date, booking_time, party_size, phone='', advance_days=0,
                    past_visits=0, estimated_spend=0, weather='Sunny'):
    with connection() as conn:
        cur = conn.execute("""
            INSERT INTO reservations
            (customer_name, booking_date, booking_time, party_size, phone, advance_days,
             past_visits, estimated_spend, weather, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Booked')
        """, (customer_name.strip(), str(booking_date), str(booking_time), int(party_size), phone.strip(),
              int(advance_days), int(past_visits), float(estimated_spend), weather))
        return cur.lastrowid

def get_reservations():
    with connection() as conn:
        rows = conn.execute("SELECT * FROM reservations ORDER BY booking_date DESC, booking_time DESC, id DESC").fetchall()
    import pandas as pd
    return pd.DataFrame([dict(r) for r in rows])

def update_status(reservation_id, status):
    allowed = {'Booked', 'Confirmed', 'Arrived', 'Completed', 'Cancelled', 'No-show'}
    if status not in allowed:
        raise ValueError('Invalid reservation status')
    with connection() as conn:
        conn.execute("UPDATE reservations SET status=? WHERE id=?", (status, int(reservation_id)))

def delete_reservation(reservation_id):
    with connection() as conn:
        conn.execute("DELETE FROM reservations WHERE id=?", (int(reservation_id),))

def seed_demo_data():
    """Insert sample bookings only when the database is empty."""
    import random
    from datetime import date, timedelta
    with connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM reservations").fetchone()[0]
    if count:
        return
    names = ['Aarav Shah', 'Diya Patil', 'Kabir Joshi', 'Anaya Rao', 'Ishaan Mehta', 'Sara Khan', 'Vivaan Deshmukh', 'Mira Kulkarni']
    times = ['09:00', '10:30', '12:00', '13:00', '16:30', '18:00', '19:00', '20:00']
    rng = random.Random(42)
    for i in range(16):
        day = date.today() + timedelta(days=i % 5)
        add_reservation(rng.choice(names), day.isoformat(), rng.choice(times), rng.randint(1, 6),
                        advance_days=rng.randint(0, 14), past_visits=rng.randint(0, 12),
                        estimated_spend=round(rng.uniform(250, 1800), 2),
                        weather=rng.choice(['Sunny', 'Rainy', 'Cloudy', 'Windy']))

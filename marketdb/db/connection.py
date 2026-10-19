"""
SQLite connection and table creation.
"""
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path):
    """Open the database (the file and its folder are created if missing)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # SQLite does not check REFERENCES unless this is turned on for each connection
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn):
    """Create the tables if they do not exist yet."""
    conn.executescript(SCHEMA_PATH.read_text(encoding='utf-8'))
    conn.commit()

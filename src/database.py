import sqlite3
import json
from datetime import datetime

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_db()

    def init_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT
                );
            """)

    def log_detection(self, event_type, details_dict=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        details_json = json.dumps(details_dict) if details_dict else None
        with self.conn:
            self.conn.execute(
                "INSERT INTO detections (timestamp, event_type, details) VALUES (?, ?, ?)",
                (timestamp, event_type, details_json)
            )

    def close(self):
        if self.conn:
            self.conn.close()

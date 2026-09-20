import sqlite3
import os
import datetime
import app.config as config


class DatabaseManager:
    def __init__(self):
        self.db_path = config.DB_PATH
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                timestamp DATETIME,
                evidence_path TEXT,
                status TEXT,
                reason TEXT,
                plate_number TEXT
            )
            """
        )
        # Migrate older DBs that lack reason / plate_number
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(incidents)")}
        if "reason" not in cols:
            cursor.execute("ALTER TABLE incidents ADD COLUMN reason TEXT")
        if "plate_number" not in cols:
            cursor.execute("ALTER TABLE incidents ADD COLUMN plate_number TEXT")
        conn.commit()
        conn.close()

    def log_incident(self, camera_id, evidence_path, reason=None, plate_number=None):
        conn = self._connect()
        cursor = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO incidents (camera_id, timestamp, evidence_path, status, reason, plate_number)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (camera_id, now, evidence_path, "New", reason or "unknown", plate_number),
        )
        conn.commit()
        incident_id = cursor.lastrowid
        conn.close()
        plate_str = f" | plate: {plate_number}" if plate_number else ""
        print(f"[DB] Incident #{incident_id} logged at {now} ({reason}){plate_str}")
        return incident_id

    def update_status(self, incident_id, status):
        conn = self._connect()
        conn.execute(
            "UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id)
        )
        conn.commit()
        conn.close()

    def get_all_incidents(self):
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()
        return rows

    def purge_old_evidence(self, retention_days=None):
        """Delete evidence files and DB rows older than retention policy."""
        days = retention_days if retention_days is not None else config.EVIDENCE_RETENTION_DAYS
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, evidence_path FROM incidents WHERE timestamp < ?",
            (cutoff_str,),
        ).fetchall()
        removed = 0
        for row in rows:
            path = row["evidence_path"]
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            conn.execute("DELETE FROM incidents WHERE id = ?", (row["id"],))
            removed += 1
        conn.commit()
        conn.close()
        if removed:
            print(f"[DB] Purged {removed} incident(s) older than {days} days")
        return removed

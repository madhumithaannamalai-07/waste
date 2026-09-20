"""
Daily Incident Report Sender.

Queries the SQLite database for all incidents logged today (or a custom date),
builds an HTML email report, and sends it via SMTP.

Usage
-----
    # Send today's report (reads REPORT_TO_EMAIL from env / config)
    python scripts/send_daily_report.py

    # Send a specific date
    python scripts/send_daily_report.py --date 2026-08-30 --to warden@example.com

    # Just print the report (no email)
    python scripts/send_daily_report.py --dry-run

Environment variables (or set in app/config.py)
-------------------------------------------------
    REPORT_TO_EMAIL       Recipient address(es), comma-separated
    REPORT_FROM_EMAIL     Sender address
    REPORT_SMTP_HOST      e.g. smtp.gmail.com
    REPORT_SMTP_PORT      587 (TLS) or 465 (SSL)
    REPORT_SMTP_PASSWORD  App password / SMTP password

For Gmail: enable "App Passwords" under your Google Account 2FA settings
and use the 16-character app password as REPORT_SMTP_PASSWORD.
"""

import argparse
import datetime
import os
import smtplib
import sqlite3
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import app.config as config


# ── HTML template ────────────────────────────────────────────────────── #

def _build_html(date_str: str, rows: list) -> str:
    total = len(rows)
    cameras = len({r["camera_id"] for r in rows})

    row_html = ""
    for r in rows:
        plate = r.get("plate_number") or "—"
        reason = r.get("reason") or "unknown"
        status = r.get("status") or "New"
        status_color = {
            "New": "#ef4444",
            "Under Review": "#f59e0b",
            "Confirmed": "#8b5cf6",
            "Resolved": "#10b981",
            "False Positive": "#64748b",
        }.get(status, "#94a3b8")
        row_html += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #334155;">{r['id']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #334155;">{r['camera_id']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #334155;">{r['timestamp']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #334155;">{reason}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #334155;">{plate}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #334155;">
            <span style="background:{status_color};color:#fff;padding:3px 8px;
                         border-radius:9999px;font-size:12px;">{status}</span>
          </td>
        </tr>"""

    if not rows:
        row_html = (
            '<tr><td colspan="6" style="padding:20px;text-align:center;'
            'color:#94a3b8;">No incidents recorded today.</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>EcoWatch Daily Report</title></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:Arial,sans-serif;color:#f8fafc;">
  <div style="max-width:780px;margin:32px auto;background:#1e293b;
              border-radius:12px;overflow:hidden;border:1px solid #334155;">
    <!-- Header -->
    <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:28px 32px;">
      <h1 style="margin:0;color:#38bdf8;font-size:22px;">🚯 EcoWatch — Daily Incident Report</h1>
      <p style="margin:6px 0 0;color:#94a3b8;font-size:14px;">Date: {date_str} &nbsp;|&nbsp; Generated at {datetime.datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    <!-- Summary cards -->
    <div style="display:flex;gap:16px;padding:20px 32px;background:#0f172a;">
      <div style="flex:1;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;">
        <div style="font-size:28px;font-weight:700;color:#ef4444;">{total}</div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">Total Incidents</div>
      </div>
      <div style="flex:1;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;">
        <div style="font-size:28px;font-weight:700;color:#38bdf8;">{cameras}</div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">Cameras Triggered</div>
      </div>
      <div style="flex:1;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px;">
        <div style="font-size:28px;font-weight:700;color:#f59e0b;">{sum(1 for r in rows if r.get('plate_number'))}</div>
        <div style="font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">Vehicles Detected</div>
      </div>
    </div>
    <!-- Table -->
    <div style="padding:0 32px 28px;">
      <h2 style="color:#f8fafc;font-size:16px;margin-bottom:12px;">Incident Log</h2>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#0f172a;">
            <th style="padding:10px 12px;text-align:left;color:#38bdf8;font-weight:600;">ID</th>
            <th style="padding:10px 12px;text-align:left;color:#38bdf8;font-weight:600;">Camera</th>
            <th style="padding:10px 12px;text-align:left;color:#38bdf8;font-weight:600;">Time</th>
            <th style="padding:10px 12px;text-align:left;color:#38bdf8;font-weight:600;">Reason</th>
            <th style="padding:10px 12px;text-align:left;color:#38bdf8;font-weight:600;">Plate</th>
            <th style="padding:10px 12px;text-align:left;color:#38bdf8;font-weight:600;">Status</th>
          </tr>
        </thead>
        <tbody>{row_html}</tbody>
      </table>
    </div>
    <!-- Footer -->
    <div style="padding:16px 32px;background:#0f172a;border-top:1px solid #334155;
                font-size:12px;color:#64748b;text-align:center;">
      EcoWatch AI Surveillance System &nbsp;|&nbsp; This is an automated report.
      Please do not reply to this email.
    </div>
  </div>
</body>
</html>"""


# ── Database query ───────────────────────────────────────────────────── #

def _fetch_incidents(date_str: str) -> list:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM incidents WHERE timestamp LIKE ? ORDER BY timestamp",
        (f"{date_str}%",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Email sender ─────────────────────────────────────────────────────── #

def _send_email(to_addrs: list, subject: str, html_body: str) -> bool:
    from_addr = config.REPORT_FROM_EMAIL
    host = config.REPORT_SMTP_HOST
    port = config.REPORT_SMTP_PORT
    password = config.REPORT_SMTP_PASSWORD

    if not from_addr or not password:
        print("[Report] REPORT_FROM_EMAIL / REPORT_SMTP_PASSWORD not set. Skipping send.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html"))

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port)
        else:
            server = smtplib.SMTP(host, port)
            server.starttls()
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addrs, msg.as_string())
        server.quit()
        print(f"[Report] Email sent to {', '.join(to_addrs)}")
        return True
    except Exception as e:
        print(f"[Report] Email failed: {e}")
        return False


# ── Save local HTML copy ─────────────────────────────────────────────── #

def _save_local(date_str: str, html_body: str) -> str:
    reports_dir = os.path.join(config.DATA_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"report_{date_str}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_body)
    return path


# ── Main ─────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Send EcoWatch daily incident report.")
    parser.add_argument("--date", default=datetime.date.today().isoformat(),
                        help="Date to report on (YYYY-MM-DD, default: today)")
    parser.add_argument("--to", default="",
                        help="Recipient email(s), comma-separated (overrides config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the report but do not send email")
    args = parser.parse_args()

    rows = _fetch_incidents(args.date)
    html = _build_html(args.date, rows)

    local_path = _save_local(args.date, html)
    print(f"[Report] {len(rows)} incident(s) on {args.date}.")
    print(f"[Report] HTML report saved: {local_path}")

    if args.dry_run:
        print("[Report] --dry-run — email not sent.")
        return

    recipients_str = args.to or config.REPORT_TO_EMAIL
    if not recipients_str:
        print("[Report] No recipient configured. Set REPORT_TO_EMAIL env var or --to flag.")
        return

    to_list = [a.strip() for a in recipients_str.split(",") if a.strip()]
    subject = f"EcoWatch | {len(rows)} Waste Dumping Incident(s) — {args.date}"
    _send_email(to_list, subject, html)


if __name__ == "__main__":
    main()

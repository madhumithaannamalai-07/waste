import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import datetime
import cv2
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app.config as config
from app.db import DatabaseManager

# Page Configuration
st.set_page_config(
    page_title="AI Waste Dumping Detection Dashboard",
    page_icon="🚯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark & Glassmorphism Theme
st.markdown("""
<style>
    /* Dark Theme Customization */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .dashboard-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-top: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 6px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-new { background-color: #ef4444; color: #ffffff; }
    .status-review { background-color: #f59e0b; color: #ffffff; }
    .status-confirmed { background-color: #8b5cf6; color: #ffffff; }
    .status-resolved { background-color: #10b981; color: #ffffff; }
    .status-false { background-color: #64748b; color: #ffffff; }

    /* Hide Streamlit Default Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
</style>
""", unsafe_allow_html=True)


def require_login():
    if st.session_state.get("authenticated"):
        return True
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="background: #1e293b; padding: 36px; border-radius: 16px; border: 1px solid #334155; text-align: center;">
            <h1 style="color: #38bdf8; margin-bottom: 8px;">🚯 Warden Control Portal</h1>
            <p style="color: #94a3b8; margin-bottom: 24px;">Authorized Personnel Authentication Required</p>
        </div>
        """, unsafe_allow_html=True)
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign In to Dashboard", use_container_width=True):
            if password == config.DASHBOARD_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please check DASHBOARD_PASSWORD.")
        st.caption("Default password: `warden123`")
    return False


if not require_login():
    st.stop()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/trash.png", width=64)
    st.title("Warden Portal")
    st.caption("AI Surveillance System v1.0")
    st.markdown("---")
    
    # Quick Actions & Filters
    st.subheader("⚙️ System Control")
    st.info(f"🟢 **Monitoring Active**\n\nCameras Configured: `{len(config.CAMERAS)}`")
    
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

db = DatabaseManager()
db.purge_old_evidence()

try:
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql_query("SELECT * FROM incidents ORDER BY timestamp DESC", conn)
    conn.close()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    df = pd.DataFrame()

# Main Header
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">🚯 AI Waste Dumping Incident Dashboard</div>
    <div class="dashboard-subtitle">Real-time surveillance monitoring, privacy-blurred evidence inspection & warden audit log</div>
</div>
""", unsafe_allow_html=True)

# Top Metrics Row
total_incidents = len(df)
today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
today_incidents = len(df[df["timestamp"].astype(str).str.startswith(today_str)]) if not df.empty else 0
unreviewed_count = int((df["status"] == "New").sum()) if not df.empty and "status" in df else 0
active_cams = len(config.CAMERAS)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_incidents}</div>
        <div class="metric-label">Total Incidents</div>
        <div class="metric-sub">Lifetime logged</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #ef4444;">{today_incidents}</div>
        <div class="metric-label">Incidents Today</div>
        <div class="metric-sub">Date: {today_str}</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #f59e0b;">{unreviewed_count}</div>
        <div class="metric-label">Needs Review</div>
        <div class="metric-sub">Status: New</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #10b981;">{active_cams}</div>
        <div class="metric-label">Active Cameras</div>
        <div class="metric-sub">Live feeds connected</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Show info if no incidents but don't stop — we still want to show live feed
no_incidents = df.empty

# Filter Controls
with st.expander("🔍 Filter & Export Incident Log", expanded=False):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cameras_list = ["All"] + list(df["camera_id"].unique()) if "camera_id" in df else ["All"]
        selected_cam = st.selectbox("Filter by Camera", cameras_list)
    with f_col2:
        status_list = ["All"] + list(df["status"].unique()) if "status" in df else ["All"]
        selected_status = st.selectbox("Filter by Status", status_list)
    with f_col3:
        search_query = st.text_input("Search (ID / Reason)", "")

filtered_df = df.copy()
if selected_cam != "All":
    filtered_df = filtered_df[filtered_df["camera_id"] == selected_cam]
if selected_status != "All":
    filtered_df = filtered_df[filtered_df["status"] == selected_status]
if search_query:
    filtered_df = filtered_df[
        filtered_df["reason"].astype(str).str.contains(search_query, case=False) |
        filtered_df["id"].astype(str).str.contains(search_query)
    ]

# Main Tabs Layout
tab_live, tab_incidents, tab_analytics, tab_gallery = st.tabs(["📹 Live Camera Feed", "📋 Incident Log & Evidence", "📊 Analytics & Insights", "🖼️ Evidence Gallery"])

with tab_live:
    st.subheader("📹 Live Camera Feed")
    st.caption("Real-time view from configured surveillance cameras")

    cam_source = config.CAMERAS[0]["source"] if config.CAMERAS else 0
    cam_name = config.CAMERAS[0]["id"] if config.CAMERAS else "Camera"

    col_feed, col_info = st.columns([3, 1])

    with col_info:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Camera</div>
            <div class="metric-value" style="font-size:1.2rem;">{cam_name}</div>
            <div class="metric-sub">Source: {cam_source}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value" style="font-size:1.2rem; color: #10b981;">● Live</div>
            <div class="metric-sub">Streaming active</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tip:** Run `python run.py` in a separate terminal for full AI detection with alerts.")

    with col_feed:
        start_feed = st.button("▶️ Start Live Feed", use_container_width=True, type="primary")
        frame_placeholder = st.empty()

        if start_feed:
            if not os.path.exists(config.LIVE_FRAME_PATH):
                st.error("❌ No live feed found. Make sure `python run.py` is running in another terminal first!")
            else:
                stale_count = 0
                last_mod_time = 0
                for _ in range(3000):  # max ~5 minutes then auto-stop
                    if not os.path.exists(config.LIVE_FRAME_PATH):
                        frame_placeholder.warning("⚠️ Detector stopped. Run `python run.py` in another terminal to restart.")
                        break
                    # Check if file is being updated
                    try:
                        mod_time = os.path.getmtime(config.LIVE_FRAME_PATH)
                        if mod_time == last_mod_time:
                            stale_count += 1
                        else:
                            stale_count = 0
                            last_mod_time = mod_time
                        if stale_count > 30:  # 3 seconds with no update
                            frame_placeholder.warning("⚠️ Feed stopped — detector is no longer running. Run `python run.py` to restart.")
                            break
                        frame = cv2.imread(config.LIVE_FRAME_PATH)
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    except Exception:
                        pass
                    time.sleep(0.1)

with tab_incidents:
    if no_incidents:
        st.info("ℹ️ No waste dumping incidents logged yet. Start the detector (`python run.py`) to trigger live surveillance events.")
        st.stop()
    col_table, col_evidence = st.columns([1.6, 1])

    with col_table:
        st.subheader(f"Incidents Log ({len(filtered_df)})")
        
        # Display dataframe neatly
        show_cols = [c for c in ["id", "camera_id", "timestamp", "status", "reason", "plate_number"] if c in filtered_df.columns]
        st.dataframe(
            filtered_df[show_cols],
            use_container_width=True,
            hide_index=True,
            height=360
        )

        st.markdown("### ✏️ Update Incident Status")
        up_c1, up_c2, up_c3 = st.columns([1, 1, 1])
        with up_c1:
            incident_ids = filtered_df["id"].tolist()
            pick_id = st.selectbox("Select Incident ID", incident_ids)
        with up_c2:
            new_status = st.selectbox(
                "New Status",
                ["New", "Under Review", "Confirmed", "False Positive", "Resolved"],
            )
        with up_c3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Save Status Update", use_container_width=True):
                db.update_status(int(pick_id), new_status)
                st.success(f"Incident #{pick_id} updated to '{new_status}'")
                st.rerun()

        # CSV Export
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Log to CSV",
            data=csv_data,
            file_name=f"waste_dumping_incidents_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with col_evidence:
        st.subheader("📷 Evidence Inspection (Privacy Blurred)")
        if not filtered_df.empty:
            options = {
                f"#{int(row.id)} | {row.timestamp} ({row.camera_id})": row
                for _, row in filtered_df.iterrows()
            }
            selected_label = st.selectbox("Choose Incident Record", list(options.keys()))
            selected = options[selected_label]
            img_path = selected["evidence_path"]

            if img_path and os.path.exists(img_path):
                st.image(
                    img_path,
                    caption=f"Incident #{selected['id']} | Camera: {selected['camera_id']}",
                    use_container_width=True,
                )
                reason_val = selected["reason"] if "reason" in selected and pd.notna(selected["reason"]) else "N/A"
                st.info(f"**Dumping Detection Reason:** `{reason_val}`\n\n**Status:** `{selected['status']}`\n\n🔒 *Faces automatically blurred for privacy protection.*")
            else:
                st.warning("⚠️ Evidence image snapshot not found on disk.")

with tab_analytics:
    st.subheader("📊 Surveillance Analytics & Pattern Insights")
    
    chart_c1, chart_c2 = st.columns(2)
    
    with chart_c1:
        st.markdown("#### Incidents by Camera Feed")
        if "camera_id" in df and not df.empty:
            cam_counts = df["camera_id"].value_counts().reset_index()
            cam_counts.columns = ["Camera ID", "Incidents"]
            st.bar_chart(cam_counts.set_index("Camera ID"))

    with chart_c2:
        st.markdown("#### Dumping Trigger Reason Breakdown")
        if "reason" in df and not df.empty:
            reason_counts = df["reason"].value_counts().reset_index()
            reason_counts.columns = ["Reason", "Count"]
            st.bar_chart(reason_counts.set_index("Reason"))

    st.markdown("#### Incident Status Distribution")
    if "status" in df and not df.empty:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.dataframe(status_counts, use_container_width=True, hide_index=True)

    # ── Daily Report download ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📄 Daily Incident Report")
    report_date = st.date_input("Report date", value=datetime.date.today())
    if st.button("Generate & Download HTML Report", use_container_width=True):
        date_str = report_date.isoformat()
        day_df = df[df["timestamp"].astype(str).str.startswith(date_str)] if not df.empty else pd.DataFrame()
        rows_for_report = day_df.to_dict("records") if not day_df.empty else []
        # Build inline HTML
        total_r = len(rows_for_report)
        row_html = ""
        for r in rows_for_report:
            plate = r.get("plate_number") or "—"
            row_html += f"<tr><td>{r['id']}</td><td>{r['camera_id']}</td><td>{r['timestamp']}</td><td>{r.get('reason','')}</td><td>{plate}</td><td>{r.get('status','')}</td></tr>"
        html_report = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
        <style>body{{font-family:Arial;background:#0f172a;color:#f8fafc;padding:20px}}
        table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 12px;border:1px solid #334155;text-align:left}}
        th{{background:#1e3a5f;color:#38bdf8}}h1{{color:#38bdf8}}</style></head>
        <body><h1>🚯 EcoWatch — Daily Report: {date_str}</h1>
        <p>Total incidents: <strong>{total_r}</strong></p>
        <table><tr><th>ID</th><th>Camera</th><th>Timestamp</th><th>Reason</th><th>Plate</th><th>Status</th></tr>
        {row_html}</table></body></html>"""
        st.download_button(
            label=f"📥 Download Report ({date_str})",
            data=html_report.encode("utf-8"),
            file_name=f"ecowatch_report_{date_str}.html",
            mime="text/html",
        )
        st.success(f"{total_r} incident(s) included in the report.")

with tab_gallery:
    st.subheader("🖼️ Evidence Snapshot Gallery")
    st.caption("Grid view of captured incident evidence with face-blurring.")
    
    gallery_cols = st.columns(3)
    for i, (_, row) in enumerate(filtered_df.head(12).iterrows()):
        with gallery_cols[i % 3]:
            path = row["evidence_path"]
            if path and os.path.exists(path):
                st.image(path, use_container_width=True)
                st.caption(f"**#{int(row.id)}** | `{row.camera_id}` | Status: `{row.status}`")
            else:
                st.warning(f"#{int(row.id)} — Snapshot missing")


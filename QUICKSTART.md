# EcoWatch — Quick Start Guide

## 1. Install (once)

```powershell
cd c:\Users\Madhumitha\OneDrive\Desktop\study\pro\waste_dumping_system
python -m pip install -r requirements.txt
```

> **Note:** `easyocr` downloads ~600 MB of model weights on first ANPR use.
> Disable ANPR with `$env:ANPR_ENABLED="false"` if you want a faster start.

---

## 2. Run live detection

```powershell
python run.py
```

### What you'll see on screen

| Overlay | Meaning |
|---|---|
| 🟦 Blue box | Person (tracked by ID) |
| 🟩 Green box | Waste object (bottle, cup, bag…) |
| 🟠 Orange "ABANDONED" | Object left after person walks away → alert |
| 🔴 Red box | Person confirmed dumping |
| 🔴 "DUMPING" banner | Active alert — evidence being saved |
| 🩵 "PLATE: …" text | ANPR plate read (if vehicle present) |
| Top-right bar | Video classifier confidence (0–1) |

Press **q** to quit.

---

## 3. Run the warden dashboard (second terminal)

```powershell
streamlit run app/dashboard.py
```

- **Password:** `warden123` (change with `$env:DASHBOARD_PASSWORD="yourpassword"`)
- Dashboard shows: Incident log with plate numbers · Analytics · Evidence gallery
- **Analytics tab** → "Generate & Download HTML Report" → pick any date → download

---

## 4. Send the daily report by email

```powershell
# Dry-run (saves HTML locally, no email sent)
python scripts/send_daily_report.py --dry-run

# Send to a specific address
python scripts/send_daily_report.py --to warden@college.edu

# Configure email via env vars (for scheduled daily sends)
$env:REPORT_FROM_EMAIL  = "yourecowatch@gmail.com"
$env:REPORT_SMTP_PASSWORD = "your-app-password"   # Gmail App Password
$env:REPORT_TO_EMAIL    = "warden@college.edu"
python scripts/send_daily_report.py
```

HTML reports are always saved locally to `data/reports/report_YYYY-MM-DD.html`.

### Schedule daily auto-send (Windows Task Scheduler)
Create a basic task that runs:
```
python C:\...\waste_dumping_system\scripts\send_daily_report.py
```
Trigger: Daily at 08:00.

---

## 5. Train the custom waste model (optional but recommended)

```powershell
# After preparing your labeled dataset as data.yaml:
python scripts/train_waste_model.py --data path\to\data.yaml --epochs 50
```
Best weights are automatically copied to `models/waste_yolo.pt`.
Restart the detector — it will auto-load the custom model.

---

## 6. Demo tips (works reliably without training a custom model)

1. Clear plastic bottle or cup in good light, full body visible.
2. Stand 1–2 m from webcam, hips and shoulders visible.
3. **Method A — abandonment:** Hold bottle → place on floor → step back 2–3 steps → wait → alert.
4. **Method B — pose:** Hold bottle → bend near it for ~1 s → alert.
5. Watch the top-right **CLF bar** grow as the video classifier picks up the sequence.

---

## 7. Stricter production mode

```powershell
$env:DEMO_MODE = "false"
python run.py
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| No webcam | Change `source` in `CAMERAS` in `app/config.py` to `1` or a video file path |
| Slow FPS | Set `DETECT_EVERY_N_FRAMES = 2` in config, or disable ANPR |
| No voice | Install pyttsx3; alert still prints in console |
| Models missing | Keep `yolo11n.pt` and `yolo11n-pose.pt` in project root |
| ANPR slow | First run downloads EasyOCR weights; subsequent runs are faster |
| Email not sending | Use a Gmail **App Password** (not your regular password) |

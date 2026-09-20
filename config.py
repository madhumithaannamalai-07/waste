import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVIDENCE_DIR = os.path.join(DATA_DIR, "evidence")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "incidents.db")
LIVE_FRAME_PATH = os.path.join(DATA_DIR, "live_frame.jpg")

# ---------------------------------------------------------------------------
# Cameras — USB index (0), video file path, or RTSP URL
# ---------------------------------------------------------------------------
CAMERAS = [
    {"id": "Cam_01_Hostel_Entrance", "source": 0},
]

# Set False for stricter production-style rules
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")

# Detection
PERSON_CONFIDENCE = 0.35 if DEMO_MODE else 0.45
WASTE_CONFIDENCE = 0.18 if DEMO_MODE else 0.35
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# 1 = every frame (best accuracy for demo laptop); 2 = faster on weak CPU
DETECT_EVERY_N_FRAMES = 1 if DEMO_MODE else 2
POSE_EVERY_N_DETECTS = 1 if DEMO_MODE else 2

# Waste proxies (COCO IDs: bottle, cup, bowl, bag, box, book, phone, food, containers, toys...)
WASTE_CLASSES = [
    24, 26, 28,             # backpack, handbag, suitcase
    39, 40, 41, 42, 43, 44, 45, # bottle, wine glass, cup, fork, knife, spoon, bowl
    46, 47, 48, 49, 50, 51, 52, 53, 54, # fruits/food items
    63, 64, 65, 66, 67,     # laptop, mouse, remote, keyboard, cell phone
    73, 74, 75, 76, 77, 79  # book, clock, vase, scissors, teddy bear, toothbrush
]

CUSTOM_WASTE_MODEL = os.getenv(
    "CUSTOM_WASTE_MODEL",
    os.path.join(MODELS_DIR, "waste_yolo.pt"),
)
OBJ_MODEL = os.path.join(BASE_DIR, os.getenv("OBJ_MODEL", "yolo11n.pt"))
POSE_MODEL = os.path.join(BASE_DIR, os.getenv("POSE_MODEL", "yolo11n-pose.pt"))

# Decision
PROXIMITY_THRESHOLD = 380 if DEMO_MODE else 280
BEND_ANGLE_THRESHOLD = 18.0 if DEMO_MODE else 28.0
CONFIRMATION_FRAMES = 3 if DEMO_MODE else 5

TRACK_MATCH_THRESHOLD = 100
STATIONARY_PIXEL_THRESHOLD = 30
ABANDONMENT_SECONDS = 1.5 if DEMO_MODE else 3.0
ABANDONMENT_PERSON_CLEARANCE = 220 if DEMO_MODE else 350
# Keep tracks alive through missed detections (was 2s — too short, broke abandonment)
TRACK_TTL_SECONDS = 8.0
REQUIRE_PERSON_AT_BIRTH = not DEMO_MODE  # demo: also track objects placed in view
PICKUP_CLEAR_STREAK = 4

BLUR_FACES_IN_EVIDENCE = True
EVIDENCE_RETENTION_DAYS = 30

COOLDOWN_SECONDS = 8
ALERT_MESSAGE = (
    "Warning! Waste dumping is restricted in this area. "
    "Please use the designated dustbin."
)

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "warden123")

# ---------------------------------------------------------------------------
# Video classifier fusion
# ---------------------------------------------------------------------------
# Confidence score above this threshold makes the classifier "vote" to confirm.
CLASSIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.55"))
# If classifier is very confident (>= this), it can confirm even without pose streak.
CLASSIFIER_HIGH_CONFIDENCE = float(os.getenv("CLASSIFIER_HIGH_CONFIDENCE", "0.72"))

# ---------------------------------------------------------------------------
# ANPR — set False to skip vehicle plate detection entirely
# ---------------------------------------------------------------------------
ANPR_ENABLED = os.getenv("ANPR_ENABLED", "true").lower() in ("1", "true", "yes")
# Run ANPR on vehicles within this many pixels of a detected dump
ANPR_SEARCH_RADIUS = 500

# ---------------------------------------------------------------------------
# Daily report (email via SMTP)
# ---------------------------------------------------------------------------
REPORT_TO_EMAIL = os.getenv("REPORT_TO_EMAIL", "")       # comma-separated
REPORT_FROM_EMAIL = os.getenv("REPORT_FROM_EMAIL", "")
REPORT_SMTP_HOST = os.getenv("REPORT_SMTP_HOST", "smtp.gmail.com")
REPORT_SMTP_PORT = int(os.getenv("REPORT_SMTP_PORT", "587"))
REPORT_SMTP_PASSWORD = os.getenv("REPORT_SMTP_PASSWORD", "")

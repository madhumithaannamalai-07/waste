"""
ANPR - Automatic Number Plate Recognition.

Pipeline: vehicle detection (YOLO classes 2,3,5,7) → lower-crop → EasyOCR.
Falls back gracefully if EasyOCR is not installed.
"""
import re
from typing import List, Optional, Tuple
import numpy as np

try:
    import easyocr as _easyocr
    _EASYOCR_OK = True
except ImportError:
    _easyocr = None
    _EASYOCR_OK = False

VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck

# Indian plate: MH12AB1234 / KA 01 AB 1234
_PLATE_RE = re.compile(
    r"[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4}", re.IGNORECASE
)

_reader = None


def _get_reader():
    global _reader
    if _reader is None and _EASYOCR_OK:
        print("[ANPR] Initialising EasyOCR (first run ~10 s)...")
        _reader = _easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[ANPR] EasyOCR ready.")
    return _reader


def extract_vehicles(yolo_results) -> List[dict]:
    """Return list of {box, cls, conf} dicts for all vehicle detections."""
    vehicles: List[dict] = []
    if yolo_results is None:
        return vehicles
    for result in yolo_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            vehicles.append({
                "box": box.xyxy[0].cpu().numpy(),
                "cls": cls_id,
                "conf": float(box.conf[0]),
            })
    return vehicles


def read_plate(frame: np.ndarray, vehicle_box: np.ndarray) -> Optional[str]:
    """Read license plate text from a vehicle bounding box. Returns str or None."""
    reader = _get_reader()
    if reader is None:
        return None

    h, w = frame.shape[:2]
    x1, y1 = max(0, int(vehicle_box[0])), max(0, int(vehicle_box[1]))
    x2, y2 = min(w, int(vehicle_box[2])), min(h, int(vehicle_box[3]))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Plates are usually in the lower 55% of a vehicle crop
    plate_region = crop[int(crop.shape[0] * 0.45):, :]

    try:
        results = reader.readtext(
            plate_region,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
            detail=1, paragraph=False,
        )
    except Exception:
        return None

    candidates: List[Tuple[float, str]] = []
    for (_, text, conf) in results:
        clean = text.strip().upper()
        alnum = re.sub(r"[^A-Z0-9]", "", clean)
        if _PLATE_RE.match(clean):
            candidates.append((conf + 1.0, clean))
        elif len(alnum) >= 5:
            candidates.append((conf, clean))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def detect_and_read(frame: np.ndarray, yolo_results) -> List[dict]:
    """Convenience: extract vehicles + read plates. Returns list of dicts."""
    vehicles = extract_vehicles(yolo_results)
    for v in vehicles:
        v["plate"] = read_plate(frame, v["box"])
    return vehicles

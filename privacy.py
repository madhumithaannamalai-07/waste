import cv2
import app.config as config


_cascade = None


def _face_cascade():
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


def blur_faces(frame):
    """Blur detected faces for privacy before storing evidence."""
    if not config.BLUR_FACES_IN_EVIDENCE:
        return frame
    out = frame.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi = out[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        out[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (51, 51), 0)
    return out


def prepare_evidence_frame(frame, person_box=None):
    """
    Privacy-safe evidence: blur faces; if no face found, blur upper body of person box.
    """
    out = blur_faces(frame)
    if person_box is not None and config.BLUR_FACES_IN_EVIDENCE:
        x1, y1, x2, y2 = map(int, person_box)
        # Upper 40% of person box as head region fallback
        hy = y1 + max(1, int((y2 - y1) * 0.4))
        x1, y1 = max(0, x1), max(0, y1)
        x2, hy = min(out.shape[1], x2), min(out.shape[0], hy)
        roi = out[y1:hy, x1:x2]
        if roi.size > 0:
            out[y1:hy, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
    return out

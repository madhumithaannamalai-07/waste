import cv2
import time
import os
import math
import datetime
import threading
from app.detector import Detector
from app.decision import DecisionEngine
from app.video_classifier import VideoClassifier
from app.alert import AlertSystem
from app.db import DatabaseManager
from app.privacy import prepare_evidence_frame
import app.config as config

# ANPR is optional — only imported when enabled
if config.ANPR_ENABLED:
    try:
        from app import anpr as _anpr
        _ANPR_AVAILABLE = True
    except Exception:
        _ANPR_AVAILABLE = False
else:
    _ANPR_AVAILABLE = False


def open_capture(source):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _try_read_plate(frame, detector):
    """Detect vehicles in a frame and return the first readable plate (or None)."""
    if not _ANPR_AVAILABLE:
        return None
    try:
        results = detector.detect_vehicles(frame)
        vehicles = _anpr.detect_and_read(frame, results)
        for v in vehicles:
            if v.get("plate"):
                return v["plate"]
    except Exception as e:
        print(f"[ANPR] Error: {e}")
    return None


def _box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist_centers(b1, b2):
    c1, c2 = _box_center(b1), _box_center(b2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def process_camera(camera, shared_alert_lock, last_alert_times, db_manager, alert_system):
    camera_id = camera["id"]
    source = camera["source"]

    print(f"[{camera_id}] Loading AI models (first run may take a minute)...")
    detector = Detector()
    decision_engine = DecisionEngine()
    video_classifier = VideoClassifier()

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"[{camera_id}] Error: could not open source {source}")
        return

    # Warm up camera
    for _ in range(5):
        cap.read()

    window = f"EcoWatch — {camera_id}"
    print(f"[{camera_id}] LIVE. Press 'q' to quit.")
    if config.DEMO_MODE:
        print(f"[{camera_id}] DEMO_MODE=on. Set DEMO_MODE=false for stricter rules.")

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Stream ended.")
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        people, waste_items, poses, people_by_id = detector.process_frame(frame)

        # Feed temporal classifier every frame
        video_classifier.push_frame(people, waste_items, poses)
        clf_conf, clf_reason = video_classifier.classify()

        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

        # ── Draw detections ────────────────────────────────────────────────
        if not people_by_id and people:
            for p_box in people:
                cv2.rectangle(frame, (int(p_box[0]), int(p_box[1])),
                              (int(p_box[2]), int(p_box[3])), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (int(p_box[0]), int(p_box[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        for tid, p_box in people_by_id.items():
            cv2.rectangle(frame, (int(p_box[0]), int(p_box[1])),
                          (int(p_box[2]), int(p_box[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"Person {tid}", (int(p_box[0]), int(p_box[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        for w_box in waste_items:
            cv2.rectangle(frame, (int(w_box[0]), int(w_box[1])),
                          (int(w_box[2]), int(w_box[3])), (0, 255, 0), 2)
            cv2.putText(frame, "Waste", (int(w_box[0]), int(w_box[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ── Run fused decision engine ──────────────────────────────────────
        is_dumping, person_box, waste_box, reason = decision_engine.is_dumping(
            people, waste_items, poses, people_by_id,
            classifier_confidence=clf_conf,
        )

        streak = decision_engine._dump_streak
        clf_bar = int(clf_conf * 10)
        status = "Monitoring"
        if streak > 0:
            status = f"Checking {streak}/{config.CONFIRMATION_FRAMES}"
        if clf_conf >= config.CLASSIFIER_CONFIDENCE_THRESHOLD:
            status += f" | CLF:{clf_conf:.2f}"

        for track in decision_engine.tracked_objects.values():
            tb = track["box"]
            if track["abandoned"]:
                color = (0, 140, 255)
                cv2.putText(frame, "ABANDONED", (int(tb[0]), int(tb[3]) + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = (120, 120, 120)
            cv2.rectangle(frame, (int(tb[0]), int(tb[1])),
                          (int(tb[2]), int(tb[3])), color, 1)

        # ── HUD ───────────────────────────────────────────────────────────
        hud = (f"FPS {fps_display:.1f} | {status} | "
               f"waste:{len(waste_items)} people:{len(people)}")
        cv2.putText(frame, hud, (10, config.FRAME_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # Classifier confidence bar (top-right corner)
        bar_x = config.FRAME_WIDTH - 120
        cv2.rectangle(frame, (bar_x, 8), (bar_x + 100, 22), (50, 50, 50), -1)
        bar_fill = int(clf_conf * 100)
        bar_color = (0, 200, 100) if clf_conf < 0.55 else (0, 100, 255)
        if bar_fill > 0:
            cv2.rectangle(frame, (bar_x, 8), (bar_x + bar_fill, 22), bar_color, -1)
        cv2.putText(frame, f"CLF {clf_conf:.2f}", (bar_x - 65, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.rectangle(frame, (bar_x, 8), (bar_x + 100, 22), (100, 100, 100), 1)

        # ── Alert & logging ───────────────────────────────────────────────
        if is_dumping:
            now = time.time()
            with shared_alert_lock:
                last = last_alert_times.get(camera_id, 0)
                cooled = now - last > config.COOLDOWN_SECONDS
                if cooled:
                    last_alert_times[camera_id] = now

            if cooled and person_box is not None and waste_box is not None:
                print(f">>> [{camera_id}] DUMPING ({reason}) <<<")

                # Draw alert overlays
                cv2.rectangle(frame,
                              (int(person_box[0]), int(person_box[1])),
                              (int(person_box[2]), int(person_box[3])),
                              (0, 0, 255), 3)
                cv2.rectangle(frame,
                              (int(waste_box[0]), int(waste_box[1])),
                              (int(waste_box[2]), int(waste_box[3])),
                              (0, 165, 255), 3)
                cv2.putText(frame, f"DUMPING ({reason})", (40, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # ANPR — try to read a plate if enabled
                plate_number = None
                if _ANPR_AVAILABLE:
                    plate_number = _try_read_plate(frame, detector)
                    if plate_number:
                        print(f"[{camera_id}] Plate detected: {plate_number}")
                        cv2.putText(frame, f"PLATE: {plate_number}",
                                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 200, 255), 2)

                # Save evidence + log
                evidence_frame = prepare_evidence_frame(frame, person_box)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                evidence_path = os.path.join(
                    config.EVIDENCE_DIR, f"incident_{camera_id}_{ts}.jpg"
                )
                cv2.imwrite(evidence_path, evidence_frame)
                db_manager.log_incident(camera_id, evidence_path,
                                        reason=reason, plate_number=plate_number)
                alert_system.trigger_alert(config.ALERT_MESSAGE)

                # Reset classifier window after a confirmed event
                video_classifier.reset()

                cv2.imshow(window, frame)
                cv2.waitKey(800)

        # Save frame for dashboard live feed
        try:
            cv2.imwrite(config.LIVE_FRAME_PATH, frame)
        except Exception:
            pass

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    # Clean up shared frame so dashboard knows feed stopped
    try:
        if os.path.exists(config.LIVE_FRAME_PATH):
            os.remove(config.LIVE_FRAME_PATH)
    except Exception:
        pass
    try:
        cv2.destroyWindow(window)
    except cv2.error:
        pass


def main():
    print("=" * 60)
    print("  EcoWatch — AI Waste Dumping Detection System")
    print("=" * 60)
    if _ANPR_AVAILABLE:
        print("  ANPR: enabled (EasyOCR)")
    else:
        print("  ANPR: disabled (install easyocr to enable)")
    print("=" * 60)

    db_manager = DatabaseManager()
    db_manager.purge_old_evidence()
    alert_system = AlertSystem()

    if not config.CAMERAS:
        print("No cameras in app/config.py CAMERAS list.")
        return

    lock = threading.Lock()
    last_alert_times = {}

    if len(config.CAMERAS) == 1:
        process_camera(config.CAMERAS[0], lock, last_alert_times, db_manager, alert_system)
        cv2.destroyAllWindows()
        print("System shut down.")
        return

    threads = []
    for cam in config.CAMERAS:
        t = threading.Thread(
            target=process_camera,
            args=(cam, lock, last_alert_times, db_manager, alert_system),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    print("System shut down.")


if __name__ == "__main__":
    main()

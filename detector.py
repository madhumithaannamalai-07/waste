import os
from ultralytics import YOLO
import app.config as config


def _resolve_model(path):
    if os.path.isfile(path):
        return path
    base = os.path.basename(path)
    alt = os.path.join(config.BASE_DIR, base)
    return alt if os.path.isfile(alt) else path


class Detector:
    """Single-pass person tracking + waste detection; pose on a configurable schedule."""

    def __init__(self):
        obj_path = _resolve_model(config.OBJ_MODEL)
        pose_path = _resolve_model(config.POSE_MODEL)
        self.obj_model = YOLO(obj_path)
        self.pose_model = YOLO(pose_path)

        self.waste_model = None
        if os.path.isfile(config.CUSTOM_WASTE_MODEL):
            self.waste_model = YOLO(config.CUSTOM_WASTE_MODEL)
            print(f"[Detector] Custom waste model: {config.CUSTOM_WASTE_MODEL}")
        else:
            print("[Detector] Using COCO waste proxies (bottle, cup, bag, etc.).")

        self._frame_i = 0
        self._detect_i = 0
        self._cache = ([], [], [], {})

    def process_frame(self, frame):
        self._frame_i += 1
        if self._frame_i % config.DETECT_EVERY_N_FRAMES != 0:
            return self._cache

        self._detect_i += 1
        people, waste_items, people_by_id = self._detect_people_and_waste(frame)

        poses = []
        if people and self._detect_i % config.POSE_EVERY_N_DETECTS == 0:
            poses = self._estimate_poses(frame)
        elif people:
            poses = self._cache[2]
        else:
            poses = []

        self._cache = (people, waste_items, poses, people_by_id)
        return self._cache

    def _detect_people_and_waste(self, frame):
        people = []
        waste_items = []
        people_by_id = {}

        if self.waste_model is not None:
            people, people_by_id = self._track_people(frame)
            waste_items = self._detect_waste_custom(frame)
            return people, waste_items, people_by_id

        target = [0] + list(config.WASTE_CLASSES)
        results = self.obj_model.track(
            frame,
            classes=target,
            conf=min(config.PERSON_CONFIDENCE, config.WASTE_CONFIDENCE),
            persist=True,
            verbose=False,
            iou=0.5,
        )
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                if cls_id == 0:
                    if conf < config.PERSON_CONFIDENCE:
                        continue
                    people.append(xyxy)
                    if box.id is not None:
                        people_by_id[int(box.id[0])] = xyxy
                elif cls_id in config.WASTE_CLASSES:
                    if conf < config.WASTE_CONFIDENCE:
                        continue
                    waste_items.append(xyxy)

        return people, waste_items, people_by_id

    def _track_people(self, frame):
        people = []
        people_by_id = {}
        results = self.obj_model.track(
            frame,
            classes=[0],
            conf=config.PERSON_CONFIDENCE,
            persist=True,
            verbose=False,
        )
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                people.append(xyxy)
                if box.id is not None:
                    people_by_id[int(box.id[0])] = xyxy
        return people, people_by_id

    def _detect_waste_custom(self, frame):
        waste_items = []
        results = self.waste_model(frame, conf=config.WASTE_CONFIDENCE, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                waste_items.append(box.xyxy[0].cpu().numpy())
        return waste_items

    def _estimate_poses(self, frame):
        poses = []
        results = self.pose_model(
            frame, classes=[0], conf=config.PERSON_CONFIDENCE, verbose=False
        )
        for result in results:
            if result.keypoints is None:
                continue
            for kp in result.keypoints.xy:
                poses.append(kp.cpu().numpy())
        return poses

    def detect_vehicles(self, frame):
        """One-shot vehicle detection (no tracking) — used during ANPR on dump events."""
        VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck
        results = self.obj_model(
            frame,
            classes=VEHICLE_CLASSES,
            conf=0.40,
            verbose=False,
        )
        return results

import math
import time
import app.config as config


class DecisionEngine:
    def __init__(self):
        self.tracked_objects = {}
        self._next_track_id = 1
        self._dump_streak = 0
        self._last_suspect = (None, None)
        self._missing_streak = {}
        self._clf_streak = 0   # consecutive frames where classifier is confident

    def calculate_angle(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if abs(dy) < 1e-3:
            return 90.0
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def box_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def distance(self, box1, box2):
        c1, c2 = self.box_center(box1), self.box_center(box2)
        return math.hypot(c2[0] - c1[0], c2[1] - c1[1])

    def point_to_box_distance(self, point, box):
        cx, cy = self.box_center(box)
        return math.hypot(point[0] - cx, point[1] - cy)

    def nearest_person(self, box, people, people_by_id=None):
        best_box, best_id, best_d = None, None, float("inf")
        sources = list(people_by_id.items()) if people_by_id else [(None, b) for b in people]
        for pid, pbox in sources:
            d = self.distance(box, pbox)
            if d < best_d:
                best_d, best_box, best_id = d, pbox, pid
        return best_box, best_id, best_d

    def match_pose_to_person(self, pose, people):
        if len(pose) < 17 or not people:
            return None
        hip_mid = (
            (pose[11][0] + pose[12][0]) / 2,
            (pose[11][1] + pose[12][1]) / 2,
        )
        if hip_mid[0] <= 0 and hip_mid[1] <= 0:
            return None
        best_idx, best_dist = None, float("inf")
        for i, box in enumerate(people):
            d = self.point_to_box_distance(hip_mid, box)
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx

    def _keypoint_ok(self, pt):
        return pt[0] > 0 and pt[1] > 0

    def _update_object_tracks(self, waste_items, people, people_by_id):
        now = time.time()
        used_detections = set()

        for track_id, track in list(self.tracked_objects.items()):
            best_j, best_dist = None, float("inf")
            for j, box in enumerate(waste_items):
                if j in used_detections:
                    continue
                d = self.distance(track["box"], box)
                if d < best_dist:
                    best_dist, best_j = d, j

            if best_j is not None and best_dist < config.TRACK_MATCH_THRESHOLD:
                box = waste_items[best_j]
                movement = self.distance(track["box"], box)
                track["box"] = box
                track["last_seen"] = now
                self._missing_streak[track_id] = 0

                pbox, pid, pdist = self.nearest_person(box, people, people_by_id)
                if pbox is not None and pdist < config.PROXIMITY_THRESHOLD:
                    track["last_person_box"] = pbox
                    track["person_id"] = pid

                if movement < config.STATIONARY_PIXEL_THRESHOLD:
                    if track["stationary_since"] is None:
                        track["stationary_since"] = now
                else:
                    track["stationary_since"] = None
                    track["abandoned"] = False
                used_detections.add(best_j)
            else:
                _, _, dist = self.nearest_person(track["box"], people, people_by_id)
                if dist < config.PROXIMITY_THRESHOLD:
                    self._missing_streak[track_id] = self._missing_streak.get(track_id, 0) + 1
                    if self._missing_streak[track_id] >= config.PICKUP_CLEAR_STREAK:
                        track["is_pickup"] = True
                else:
                    self._missing_streak[track_id] = 0

        for j, box in enumerate(waste_items):
            if j in used_detections:
                continue
            pbox, pid, dist = self.nearest_person(box, people, people_by_id)
            near_person = dist < config.PROXIMITY_THRESHOLD
            if config.REQUIRE_PERSON_AT_BIRTH and not near_person:
                continue

            tid = self._next_track_id
            self._next_track_id += 1
            self.tracked_objects[tid] = {
                "box": box,
                "first_seen": now,
                "last_seen": now,
                "stationary_since": now,
                "abandoned": False,
                "alerted": False,
                "is_pickup": False,
                "born_near_person": near_person,
                "person_id": pid,
                "last_person_box": pbox if near_person else None,
                "reason": "drop_candidate",
            }
            self._missing_streak[tid] = 0

        for tid in list(self.tracked_objects.keys()):
            track = self.tracked_objects[tid]
            if track.get("is_pickup"):
                del self.tracked_objects[tid]
                self._missing_streak.pop(tid, None)
                continue
            if now - track["last_seen"] > config.TRACK_TTL_SECONDS:
                del self.tracked_objects[tid]
                self._missing_streak.pop(tid, None)

        for track in self.tracked_objects.values():
            if track["stationary_since"] is None:
                continue
            if now - track["stationary_since"] < config.ABANDONMENT_SECONDS:
                continue
            _, _, nearest = self.nearest_person(track["box"], people, people_by_id)
            if nearest > config.ABANDONMENT_PERSON_CLEARANCE:
                track["abandoned"] = True
                track["reason"] = "abandoned_after_drop"

    def _pose_suggests_dump(self, pose):
        if len(pose) < 17:
            return False

        l_shoulder, r_shoulder = pose[5], pose[6]
        l_hip, r_hip = pose[11], pose[12]
        l_wrist, r_wrist = pose[9], pose[10]

        if not (self._keypoint_ok(l_hip) or self._keypoint_ok(r_hip)):
            return False
        if not (self._keypoint_ok(l_shoulder) or self._keypoint_ok(r_shoulder)):
            return False

        shoulder_mid = (
            (l_shoulder[0] + r_shoulder[0]) / 2,
            (l_shoulder[1] + r_shoulder[1]) / 2,
        )
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

        bend_angle = self.calculate_angle(shoulder_mid, hip_mid)

        wrist_dropped = False
        if self._keypoint_ok(l_wrist) and l_wrist[1] > hip_mid[1] - 15:
            wrist_dropped = True
        if self._keypoint_ok(r_wrist) and r_wrist[1] > hip_mid[1] - 15:
            wrist_dropped = True

        return bend_angle > config.BEND_ANGLE_THRESHOLD and wrist_dropped

    def is_dumping(self, people, waste_items, poses, people_by_id=None,
                   classifier_confidence: float = 0.0):
        people_by_id = people_by_id or {}
        self._update_object_tracks(waste_items, people, people_by_id)

        # ── Abandoned-object path (unchanged) ──────────────────────────
        for track in self.tracked_objects.values():
            if track["abandoned"] and not track["alerted"]:
                track["alerted"] = True
                person_box = track.get("last_person_box")
                if person_box is None and people:
                    person_box, _, _ = self.nearest_person(
                        track["box"], people, people_by_id
                    )
                if person_box is None:
                    person_box = track["box"]
                self._dump_streak = 0
                self._clf_streak = 0
                return True, person_box, track["box"], track.get("reason", "abandoned")

        # ── Pose-rule path ─────────────────────────────────────────────
        frame_hit = False
        suspect_person, suspect_waste = None, None
        for pose in poses:
            if not self._pose_suggests_dump(pose):
                continue
            person_idx = self.match_pose_to_person(pose, people)
            if person_idx is None:
                continue
            person_box = people[person_idx]
            for waste_box in waste_items:
                if self.distance(person_box, waste_box) < config.PROXIMITY_THRESHOLD:
                    frame_hit = True
                    suspect_person, suspect_waste = person_box, waste_box
                    break
            if frame_hit:
                break

        if frame_hit:
            self._dump_streak += 1
            self._last_suspect = (suspect_person, suspect_waste)
        else:
            self._dump_streak = 0
            self._last_suspect = (None, None)

        # ── Classifier path ────────────────────────────────────────────
        if classifier_confidence >= config.CLASSIFIER_CONFIDENCE_THRESHOLD:
            self._clf_streak += 1
        else:
            self._clf_streak = 0

        # Find nearest person+waste pair for classifier confirmation
        clf_person, clf_waste = None, None
        if people and waste_items:
            clf_person, _, _ = self.nearest_person(waste_items[0], people, people_by_id)
            clf_waste = waste_items[0]

        # Very high classifier confidence → confirm immediately if pair exists
        if (classifier_confidence >= config.CLASSIFIER_HIGH_CONFIDENCE
                and clf_person is not None and clf_waste is not None
                and self._clf_streak >= 3):
            self._dump_streak = 0
            self._clf_streak = 0
            reason = f"classifier_high({classifier_confidence:.2f})"
            return True, clf_person, clf_waste, reason

        # ── Determine effective confirmation threshold ─────────────────
        # Classifier voting lowers the pose streak needed
        clf_boost = 0
        if classifier_confidence >= config.CLASSIFIER_CONFIDENCE_THRESHOLD:
            clf_boost = 2
        elif classifier_confidence >= 0.35:
            clf_boost = 1
        effective_threshold = max(1, config.CONFIRMATION_FRAMES - clf_boost)

        if self._dump_streak >= effective_threshold:
            self._dump_streak = 0
            self._clf_streak = 0
            if clf_boost > 0:
                reason = f"pose_confirmed+clf({classifier_confidence:.2f})"
            else:
                reason = "pose_confirmed"
            return True, self._last_suspect[0], self._last_suspect[1], reason

        return False, None, None, None

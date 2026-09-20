"""
Temporal video classifier for waste-dumping detection.

Two modes
---------
1.  Heuristic (always available):
    Analyses a 30-frame rolling window of detections for the classic
    dumping signature — wrist-drop event, waste object appearing mid-sequence
    and becoming stationary, sustained person-waste proximity.

2.  Model (optional):
    If models/dump_classifier.pt exists it is loaded via torch.jit and
    used instead.  The file is a TorchScript module that accepts a
    (1, WINDOW, feature_dim) float tensor and returns a scalar confidence.

Either way, .classify() returns (confidence: float, reason: str).
The DecisionEngine fuses this with its rule-based result.
"""

import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np

import app.config as config

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class VideoClassifier:
    """Sliding-window temporal dump detector."""

    WINDOW: int = 30       # frames kept in the buffer
    MIN_FRAMES: int = 6    # minimum before classification attempt

    def __init__(self):
        self._buf: List[dict] = []
        self._model = None

        mp = os.path.join(config.MODELS_DIR, "dump_classifier.pt")
        if _TORCH_OK and os.path.isfile(mp):
            try:
                self._model = torch.jit.load(mp, map_location="cpu")
                self._model.eval()
                print(f"[VideoClassifier] Loaded trained model: {mp}")
            except Exception as e:
                print(f"[VideoClassifier] Model load failed ({e}); using heuristic mode.")
        else:
            print("[VideoClassifier] Heuristic temporal-analysis mode active.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push_frame(self, people, waste_items, poses) -> None:
        """Add the latest detection frame to the rolling buffer."""
        self._buf.append(
            {
                "people": list(people),
                "waste_items": list(waste_items),
                "poses": [np.asarray(p, dtype=float) for p in poses] if poses else [],
                "ts": time.time(),
            }
        )
        if len(self._buf) > self.WINDOW:
            self._buf.pop(0)

    def classify(self) -> Tuple[float, str]:
        """Return (confidence in [0,1], reason_string)."""
        if len(self._buf) < self.MIN_FRAMES:
            return 0.0, "buffering"
        if self._model is not None:
            return self._model_classify()
        return self._heuristic_classify()

    def reset(self) -> None:
        """Clear the rolling buffer (call after a confirmed alert)."""
        self._buf.clear()

    # ------------------------------------------------------------------ #
    #  Heuristic classifier                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _kp_ok(pt) -> bool:
        return float(pt[0]) > 0 or float(pt[1]) > 0

    def _wrist_drop_scores(self) -> List[Optional[float]]:
        """Per-frame: wrist_y - hip_y (positive = wrist below hip)."""
        out: List[Optional[float]] = []
        for f in self._buf:
            if not f["poses"]:
                out.append(None)
                continue
            pose = f["poses"][0]
            if len(pose) < 17:
                out.append(None)
                continue
            lw, rw = pose[9], pose[10]
            lh, rh = pose[11], pose[12]
            hip_y = (float(lh[1]) + float(rh[1])) / 2.0
            wrist_y: Optional[float] = None
            if self._kp_ok(lw):
                wrist_y = float(lw[1])
            elif self._kp_ok(rw):
                wrist_y = float(rw[1])
            out.append((wrist_y - hip_y) if wrist_y is not None else None)
        return out

    def _avg_waste_movement(self) -> float:
        """Average per-frame pixel movement of the principal waste centroid."""
        centroids = []
        for f in self._buf:
            if not f["waste_items"]:
                continue
            b = f["waste_items"][0]
            cx = (float(b[0]) + float(b[2])) / 2
            cy = (float(b[1]) + float(b[3])) / 2
            centroids.append((cx, cy))
        if len(centroids) < 2:
            return float("inf")
        total = sum(
            math.hypot(centroids[k][0] - centroids[k - 1][0],
                       centroids[k][1] - centroids[k - 1][1])
            for k in range(1, len(centroids))
        )
        return total / (len(centroids) - 1)

    def _heuristic_classify(self) -> Tuple[float, str]:
        score = 0.0
        reasons: List[str] = []
        n = len(self._buf)

        # ── Signal 1: wrist-drop event ──────────────────────────────────
        wrist_scores = [v for v in self._wrist_drop_scores() if v is not None]
        if wrist_scores:
            peak_drop = max(wrist_scores)
            if peak_drop > 30:
                score += 0.35
                reasons.append("wrist_below_hip")
            elif peak_drop > 10:
                score += 0.15
                reasons.append("wrist_at_hip")

        # ── Signal 2: waste appears mid-sequence and becomes stationary ──
        waste_frame_indices = [i for i, f in enumerate(self._buf) if f["waste_items"]]
        if waste_frame_indices:
            first_idx = waste_frame_indices[0]
            if first_idx > n * 0.15:         # appeared after initial frames
                score += 0.20
                reasons.append("waste_appeared")

            avg_move = self._avg_waste_movement()
            if avg_move < 12:
                score += 0.20
                reasons.append("waste_stationary")
            elif avg_move < 25:
                score += 0.08
                reasons.append("waste_slow")

        # ── Signal 3: sustained person–waste proximity ───────────────────
        prox_count = 0
        for f in self._buf[-10:]:
            if not f["people"] or not f["waste_items"]:
                continue
            pb = f["people"][0]
            wb = f["waste_items"][0]
            dist = math.hypot(
                (float(pb[0]) + float(pb[2])) / 2 - (float(wb[0]) + float(wb[2])) / 2,
                (float(pb[1]) + float(pb[3])) / 2 - (float(wb[1]) + float(wb[3])) / 2,
            )
            if dist < config.PROXIMITY_THRESHOLD:
                prox_count += 1
        if prox_count >= 5:
            score += 0.20
            reasons.append("sustained_proximity")
        elif prox_count >= 2:
            score += 0.08
            reasons.append("brief_proximity")

        # ── Dampen when evidence is thin ────────────────────────────────
        strong_signals = {"wrist_below_hip", "waste_stationary", "sustained_proximity"}
        has_strong = bool(strong_signals & set(reasons))
        if not has_strong or len(reasons) < 2:
            score *= 0.45

        score = round(min(score, 1.0), 3)
        return score, ("+".join(reasons) if reasons else "no_signal")

    # ------------------------------------------------------------------ #
    #  Model-based inference (activated when dump_classifier.pt is ready) #
    # ------------------------------------------------------------------ #

    def _model_classify(self) -> Tuple[float, str]:
        """
        Build a feature tensor from the current buffer and run the model.

        Feature vector per frame (9 dimensions):
          [has_person, has_waste, person_cx, person_cy,
           waste_cx, waste_cy, wrist_drop, proximity, waste_area_norm]
        """
        try:
            feats = []
            for f in self._buf:
                has_p = 1.0 if f["people"] else 0.0
                has_w = 1.0 if f["waste_items"] else 0.0
                pcx = pcy = wcx = wcy = 0.0
                if f["people"]:
                    b = f["people"][0]
                    pcx = (float(b[0]) + float(b[2])) / (2 * config.FRAME_WIDTH)
                    pcy = (float(b[1]) + float(b[3])) / (2 * config.FRAME_HEIGHT)
                if f["waste_items"]:
                    b = f["waste_items"][0]
                    wcx = (float(b[0]) + float(b[2])) / (2 * config.FRAME_WIDTH)
                    wcy = (float(b[1]) + float(b[3])) / (2 * config.FRAME_HEIGHT)
                    waste_area = ((float(b[2]) - float(b[0])) *
                                  (float(b[3]) - float(b[1]))) / (
                        config.FRAME_WIDTH * config.FRAME_HEIGHT)
                else:
                    waste_area = 0.0
                wd = 0.0
                if f["poses"]:
                    pose = f["poses"][0]
                    if len(pose) >= 17:
                        hip_y = (float(pose[11][1]) + float(pose[12][1])) / 2
                        for idx in [9, 10]:
                            if self._kp_ok(pose[idx]):
                                wd = max(wd, (float(pose[idx][1]) - hip_y) /
                                         config.FRAME_HEIGHT)
                prox = math.hypot(pcx - wcx, pcy - wcy) if has_p and has_w else 1.0
                feats.append([has_p, has_w, pcx, pcy, wcx, wcy, wd, prox, waste_area])

            # Pad / truncate to WINDOW length
            while len(feats) < self.WINDOW:
                feats.insert(0, [0.0] * 9)
            feats = feats[-self.WINDOW:]

            tensor = torch.tensor([feats], dtype=torch.float32)   # (1, W, 9)
            with torch.no_grad():
                conf = float(self._model(tensor).squeeze())
            conf = max(0.0, min(1.0, conf))
            return round(conf, 3), "model"
        except Exception as e:
            return 0.0, f"model_err:{e}"

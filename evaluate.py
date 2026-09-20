"""
Evaluate detection decisions on a folder of images or a video.

Usage:
  python scripts/evaluate.py --source data/eval_images
  python scripts/evaluate.py --source path/to/clip.mp4 --labels data/eval_labels.json

labels JSON (optional):
  {"frame_0001.jpg": true, "frame_0002.jpg": false, ...}
  or for video: {"42": true, "100": false}  # frame indices where dumping occurs
"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.detector import Detector
from app.decision import DecisionEngine
import app.config as config


def iter_frames(source):
    if os.path.isdir(source):
        files = sorted(
            f
            for f in os.listdir(source)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        for name in files:
            path = os.path.join(source, name)
            frame = cv2.imread(path)
            if frame is not None:
                yield name, frame
        return

    cap = cv2.VideoCapture(source)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield str(idx), frame
        idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Evaluate waste dumping detector")
    parser.add_argument("--source", required=True, help="Image folder or video path")
    parser.add_argument("--labels", default=None, help="Optional JSON ground truth")
    args = parser.parse_args()

    labels = {}
    if args.labels and os.path.isfile(args.labels):
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = json.load(f)

    detector = Detector()
    engine = DecisionEngine()

    tp = fp = tn = fn = 0
    positives_fired = 0
    total = 0

    for key, frame in iter_frames(args.source):
        total += 1
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        people, waste, poses, people_by_id = detector.process_frame(frame)
        is_dump, _, _, reason = engine.is_dumping(people, waste, poses, people_by_id)
        pred = bool(is_dump)
        if pred:
            positives_fired += 1
            print(f"[HIT] {key} reason={reason}")

        if key in labels or (isinstance(key, str) and key in labels):
            gt = bool(labels[key])
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1

    print("\n=== Evaluation Summary ===")
    print(f"Frames processed : {total}")
    print(f"Detections fired : {positives_fired}")
    if labels:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    else:
        print("No labels provided — reported detection count only.")
        print("Add --labels to compute precision / recall / F1.")


if __name__ == "__main__":
    main()

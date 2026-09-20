"""
Fine-tune a YOLO model on a litter/waste dataset (turns COCO proxies into real waste detection).

1. Prepare YOLO-format dataset with data.yaml, e.g. from TACO or a custom litter set.
2. Run:
     python scripts/train_waste_model.py --data path/to/data.yaml --epochs 50
3. Copy best weights to models/waste_yolo.pt (done automatically below).

Example data.yaml:
  path: ../datasets/litter
  train: images/train
  val: images/val
  names: {0: litter}
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ultralytics import YOLO
import app.config as config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", default="yolo11n.pt", help="Base weights")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"data.yaml not found: {args.data}")
        sys.exit(1)

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    dest = config.CUSTOM_WASTE_MODEL
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(best):
        shutil.copy2(best, dest)
        print(f"Custom waste model saved to {dest}")
        print("Restart the detector — it will auto-load this model.")
    else:
        print(f"Training finished but best.pt not found at {best}")


if __name__ == "__main__":
    main()

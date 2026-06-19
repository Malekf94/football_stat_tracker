"""
Fine-tune YOLO11 on the corrected dataset (run AFTER label_tool.py).

Starts from pretrained yolo11n (the size the pipeline uses), so the result is a
drop-in replacement that detects OUR players + ball better than the generic model.

Usage:
    .\\venv\\Scripts\\python train.py                  # 100 epochs, yolo11n
    .\\venv\\Scripts\\python train.py --epochs 60 --model yolo11m.pt

Best weights end up at: runs/football_finetune/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="base weights to fine-tune from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="match the pipeline's imgsz")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data="dataset/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=-1,              # auto-fit GPU memory
        patience=25,           # early-stop if no improvement
        project="runs",
        name="football_finetune",
        exist_ok=True,
    )
    print("\nDone. Fine-tuned weights: runs/football_finetune/weights/best.pt")


if __name__ == "__main__":
    main()

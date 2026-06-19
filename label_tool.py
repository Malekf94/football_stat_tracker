"""
Local labelling tool — correct the pre-labelled detector dataset.

Loads the frames + pre-labels made by make_dataset.py and lets you fix them:
delete wrong boxes, add missed players, fix the ball. No cloud, no limits.

Usage:
    .\\venv\\Scripts\\python label_tool.py            # uses ./dataset

Controls:
    Left-drag         draw a NEW box (in the current class)
    Right-click       delete the box under the cursor
    V                 set draw class = player (blue)
    B                 set draw class = ball   (yellow)
    D / Space         next image (auto-saves)
    A                 previous image (auto-saves)
    R                 reload this image's labels (discard edits on it)
    Q / Esc           save everything and quit

Tips: the boxes are already drawn by the model — most frames just need a quick
glance and a small fix. The ball is the important one to get right.
"""
import os
import glob
import cv2
import numpy as np

DATASET = "dataset"
CLASS_COLORS = {0: (255, 130, 0), 1: (0, 220, 255)}   # player=blue, ball=yellow
CLASS_NAMES = {0: "player", 1: "ball"}
MAX_W = 1600


class Labeller:
    def __init__(self, dataset=DATASET):
        self.img_dir = os.path.join(dataset, "images")
        self.lbl_dir = os.path.join(dataset, "labels")
        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if not self.images:
            raise SystemExit(f"No images in {self.img_dir}. Run make_dataset.py first.")
        os.makedirs(self.lbl_dir, exist_ok=True)
        self.idx = 0
        self.draw_class = 0
        self.boxes: list[list] = []        # [cls, x1, y1, x2, y2] in original px
        self.frame = None
        self.scale = 1.0
        self.drag_start = None
        self.cur_mouse = (0, 0)
        self.win = "Labeller — V player | B ball | drag=add | right-click=delete | D/A nav | Q save+quit"
        self._load()

    # ---- label IO (YOLO format) ----
    def _lbl_path(self, i):
        name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return os.path.join(self.lbl_dir, name + ".txt")

    def _load(self):
        self.frame = cv2.imread(self.images[self.idx])
        h, w = self.frame.shape[:2]
        self.scale = min(1.0, MAX_W / w)
        self.boxes = []
        p = self._lbl_path(self.idx)
        if os.path.exists(p):
            for ln in open(p):
                parts = ln.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = parts
                c = int(c)
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                self.boxes.append([c, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])

    def _save(self):
        h, w = self.frame.shape[:2]
        with open(self._lbl_path(self.idx), "w") as f:
            for c, x1, y1, x2, y2 in self.boxes:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param):
        ox, oy = x / self.scale, y / self.scale
        self.cur_mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x1, y1 = self.drag_start
            if abs(ox - x1) > 4 and abs(oy - y1) > 4:
                self.boxes.append([self.draw_class, min(x1, ox), min(y1, oy), max(x1, ox), max(y1, oy)])
            self.drag_start = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit, area = None, float("inf")
            for i, (c, a, b, cc, d) in enumerate(self.boxes):
                if a <= ox <= cc and b <= oy <= d and (cc - a) * (d - b) < area:
                    area = (cc - a) * (d - b)
                    hit = i
            if hit is not None:
                self.boxes.pop(hit)

    # ---- render ----
    def render(self):
        disp = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale)
        for c, x1, y1, x2, y2 in self.boxes:
            col = CLASS_COLORS.get(c, (200, 200, 200))
            cv2.rectangle(disp, (int(x1 * self.scale), int(y1 * self.scale)),
                          (int(x2 * self.scale), int(y2 * self.scale)), col, 2)
        if self.drag_start:
            x1, y1 = self.drag_start
            cv2.rectangle(disp, (int(x1 * self.scale), int(y1 * self.scale)),
                          self.cur_mouse, CLASS_COLORS[self.draw_class], 1)
        n_ball = sum(1 for b in self.boxes if b[0] == 1)
        bar = (f"{self.idx+1}/{len(self.images)} | draw:{CLASS_NAMES[self.draw_class]} | "
               f"players:{sum(1 for b in self.boxes if b[0]==0)} ball:{n_ball}")
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(disp, bar, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.win, disp)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)
        while True:
            self.render()
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('q'), 27):
                self._save()
                break
            elif k == ord('v'):
                self.draw_class = 0
            elif k == ord('b'):
                self.draw_class = 1
            elif k in (ord('d'), 32):
                self._save()
                self.idx = min(self.idx + 1, len(self.images) - 1)
                self._load()
            elif k == ord('a'):
                self._save()
                self.idx = max(self.idx - 1, 0)
                self._load()
            elif k == ord('r'):
                self._load()
        cv2.destroyAllWindows()
        print(f"Saved labels for {len(self.images)} images in {self.lbl_dir}")


if __name__ == "__main__":
    Labeller().run()

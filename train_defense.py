"""
train_defense.py — fine-tune a Kinetics-pretrained R3D-18 video model to classify
PLAYER-CENTRED clips as defensive-action vs not, from make_defense_dataset.py output.

This is a PROOF-OF-SIGNAL run: with only ~150 positives it tells us whether the
concept is learnable at all, not a finished model.

    python train_defense.py [--epochs 20] [--batch 8] [--val-match "testing match"]

--val-match: hold ALL of one match out as validation (honest cross-match test).
             Default: random stratified 80/20 split (in-distribution, optimistic).
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from torchvision.models.video import r3d_18, R3D_18_Weights

DATA = "dataset_defense"
OUT = "runs_defense"
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

random.seed(0); np.random.seed(0); torch.manual_seed(0)


class Clips(torch.utils.data.Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        self.train = train
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r["clip"] not in self.cache:
            self.cache[r["clip"]] = np.load(f"{DATA}/clips/{r['clip']}")  # [T,H,W,3] uint8
        clip = self.cache[r["clip"]]
        x = torch.from_numpy(clip).float() / 255.0          # [T,H,W,3]
        x = x.permute(3, 0, 1, 2)                            # [3,T,H,W]
        if self.train and random.random() < 0.5:
            x = torch.flip(x, dims=[3])                      # horizontal flip
        x = (x - MEAN) / STD
        return x, int(r["label"])


def load_rows():
    with open(f"{DATA}/manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split(rows, val_match):
    if val_match:
        tr = [r for r in rows if r["match"] != val_match]
        va = [r for r in rows if r["match"] == val_match]
        return tr, va
    # random stratified 80/20
    by = {"0": [], "1": []}
    for r in rows:
        by[r["label"]].append(r)
    tr, va = [], []
    for lab, rs in by.items():
        random.shuffle(rs)
        k = int(len(rs) * 0.8)
        tr += rs[:k]; va += rs[k:]
    random.shuffle(tr); random.shuffle(va)
    return tr, va


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        ps += out.cpu().tolist(); ys += y.tolist()
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    pred = [1 if p >= 0.5 else 0 for p in ps]
    cm = confusion_matrix(ys, pred, labels=[0, 1])
    acc = sum(int(a == b) for a, b in zip(pred, ys)) / len(ys)
    return auc, acc, cm, ys, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-match", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_rows()
    tr, va = split(rows, args.val_match)
    ntr_pos = sum(int(r["label"]) for r in tr)
    nva_pos = sum(int(r["label"]) for r in va)
    print(f"train {len(tr)} ({ntr_pos} pos) | val {len(va)} ({nva_pos} pos)"
          + (f" | val-match={args.val_match}" if args.val_match else " | random split"))

    tl = torch.utils.data.DataLoader(Clips(tr, train=True), batch_size=args.batch,
                                     shuffle=True, num_workers=0)
    vl = torch.utils.data.DataLoader(Clips(va), batch_size=args.batch, num_workers=0)

    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # class weights (negatives outnumber positives ~2:1)
    npos = max(1, ntr_pos); nneg = max(1, len(tr) - ntr_pos)
    w = torch.tensor([len(tr) / (2 * nneg), len(tr) / (2 * npos)], device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_auc, best = -1, None
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            tot += loss.item() * len(y)
        sched.step()
        auc, acc, cm, _, _ = evaluate(model, vl, device)
        flag = ""
        if auc > best_auc:
            best_auc, best = auc, {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best, f"{OUT}/best.pt"); flag = "  <- best"
        print(f"ep{ep+1:02d}  loss {tot/len(tr):.3f}  val_auc {auc:.3f}  val_acc {acc:.3f}  "
              f"cm[tn,fp;fn,tp]={cm.tolist()}{flag}")

    print(f"\n[Done] best val AUC = {best_auc:.3f}  (0.5 = chance). Weights -> {OUT}/best.pt")
    print("Reminder: random-split AUC is IN-DISTRIBUTION (optimistic). Use --val-match "
          "to test cross-match generalisation once more matches are labelled.")


if __name__ == "__main__":
    main()

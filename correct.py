"""
Manual correction tool — player gallery.

After analyze.py runs (it writes a tracking log), this shows ONE thumbnail per
player who appears in the stats, grouped by team. You fix the two things the
automatic pipeline can't on plain bibs, without scrubbing the video:

  - one real player split into several ids  -> click each fragment, press M to merge
  - a player in the wrong team colour        -> click them, press T to flip team

Press S to save and write *_player_stats_corrected.csv with the merged totals.
The thumbnail ids match the CSV rows so you can find any player's stats.

Usage:
    .\venv\Scripts\python correct.py "..\testing match.mp4"

Controls:
    Left-click   select / deselect a player
    A            select look-alikes of the selected player ON THIS PAGE
    M            merge all selected players into one
    T            flip team of just the selected players
    B            toggle sort: by involvement  <->  grouped BY TEAM (spot wrong colours)
    G            swap team colours globally (use this if ALL colours are inverted)
    Z            UNDO the last action (merge / flip / swap) — go back a step
    U            revert corrections on the currently-selected players
    C            clear selection
    N / P        next / previous page
    S            save + write corrected stats CSV
    Q / Esc      quit
"""
import os
import sys
import json
from collections import Counter, defaultdict

import cv2
import numpy as np

from tracker.replay import reaggregate, UnionFind

TEAM_COLORS_BGR = {0: (255, 100, 0), 1: (0, 80, 255), -1: (160, 160, 160)}
GALLERY_MIN_POSS_S = 1.5     # show players with at least this much possession (a
                             # bit below the 3s CSV cutoff so near-threshold
                             # fragments can still be merged up)
CELL_W, CELL_H = 120, 175
THUMB_W, THUMB_H = 96, 140
CANVAS_W = 1320
PAGE_ROWS = 6          # thumbnails per page = (CANVAS_W // CELL_W) * PAGE_ROWS
SIMILAR_THRESH = 1.2   # z-distance for "select look-alikes" (higher = more permissive,
                       # but risks grabbing similar-looking teammates; press A again to grow)


class Corrector:
    def __init__(self, arg: str):
        self.base = os.path.splitext(os.path.basename(arg))[0]
        meta_path = os.path.join("output", f"{self.base}_tracks_meta.json")
        self.tracks_path = os.path.join("output", f"{self.base}_tracks.jsonl")
        if not (os.path.exists(meta_path) and os.path.exists(self.tracks_path)):
            sys.exit(f"No tracking log for '{self.base}' in output/. Run analyze.py first.")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.final_teams = {int(k): int(v) for k, v in self.meta.get("final_teams", {}).items()}
        self.display_ids = {int(k): int(v) for k, v in self.meta.get("display_map", {}).items()}

        self.frames = []
        with open(self.tracks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.frames.append(json.loads(line))

        video_path = self.meta.get("video_path", "")
        if not os.path.exists(video_path):
            alt = os.path.join("..", f"{self.base}.mp4")
            video_path = alt if os.path.exists(alt) else video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open video: {video_path}")

        # corrections
        self.overrides: dict[int, int] = {}
        self.merges: list[list[int]] = []
        self.swap_teams: bool = False
        self.corr_path = os.path.join("output", f"{self.base}_corrections.json")
        if os.path.exists(self.corr_path):
            # resume a previous session (its merges already include the auto-stitch)
            c = json.load(open(self.corr_path, encoding="utf-8"))
            self.overrides = {int(k): int(v) for k, v in c.get("team_overrides", {}).items()}
            self.merges = [[int(a), int(b)] for a, b in c.get("merges", [])]
            self.swap_teams = bool(c.get("swap_teams", False))
        else:
            # first open: auto-stitch fragmented tracks as a starting point
            from tracker.reid import auto_merge
            self.merges = auto_merge(self.meta, self.tracks_path)
            print(f"[Auto-ReID] stitched fragments into {len(self.merges)} links.")

        self.selected: set[int] = set()
        self.page = 0
        self.sort_mode = "poss"                  # "poss" (most-involved first) or "team"
        self.history: list[tuple] = []          # snapshots for undo (Z)
        self._page_canons: set[int] = set()      # canons shown on the current page
        self.status = "Click players that are the same person, then press M to merge."
        self.win = "Gallery — A look-alikes | M merge | T flip | B by-team | G swap | Z undo | N/P page | S save | Q quit"
        self.cell_rects: list[tuple[int, int, int, int, int]] = []
        self._crop_cache: dict[int, np.ndarray] = {}

        self._scan_candidates()
        self._refresh_gallery()

    # ---- per-track clearest-isolated-frame candidate (computed once) ----
    def _scan_candidates(self):
        self.framecount: Counter = Counter()
        self.best: dict[int, tuple] = {}     # raw_id -> (pollution, -area, frame, bbox)
        for rec in self.frames:
            boxes = rec["players"]
            for i, (rid, x1, y1, x2, y2, team) in enumerate(boxes):
                self.framecount[rid] += 1
                area = (x2 - x1) * (y2 - y1)
                if area <= 0:
                    continue
                pollution = 0.0
                for j, (_, a1, b1, a2, b2, _) in enumerate(boxes):
                    if j == i:
                        continue
                    iw = max(0.0, min(x2, a2) - max(x1, a1))
                    ih = max(0.0, min(y2, b2) - max(y1, b1))
                    if iw > 0 and ih > 0:
                        pollution = max(pollution, (iw * ih) / area)
                key = (round(pollution, 3), -area)
                if rid not in self.best or key < self.best[rid][:2]:
                    self.best[rid] = (round(pollution, 3), -area, rec["f"], (x1, y1, x2, y2))
        self.all_raw = list(self.framecount.keys())

    def _crop_for(self, rid: int) -> np.ndarray | None:
        if rid in self._crop_cache:
            return self._crop_cache[rid]
        if rid not in self.best:
            return None
        _, _, f, (x1, y1, x2, y2) = self.best[rid]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = self.cap.read()
        if not ok:
            return None
        crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if crop.size == 0:
            return None
        self._crop_cache[rid] = crop
        return crop

    # ---- correction helpers ----
    def _uf(self) -> UnionFind:
        uf = UnionFind()
        for a, b in self.merges:
            uf.union(a, b)
        return uf

    def effective_team(self, canon: int, members: list[int]) -> int:
        if canon in self.overrides:
            return self.overrides[canon]
        votes = Counter(self.final_teams.get(m, -1) for m in members)
        return votes.most_common(1)[0][0]

    def _rep_member(self, members: list[int]) -> int:
        """The member with the cleanest/largest crop (for the thumbnail + id)."""
        return min(members, key=lambda m: self.best.get(m, (9, 0))[:2])

    @staticmethod
    def _appearance(crop) -> np.ndarray:
        """Appearance signature emphasising what distinguishes TEAMMATES.

        The torso (bib) is identical within a team, so it carries no info for
        telling teammates apart — we use the head band (skin tone + hair) and the
        leg band (shorts/socks/shoes), which is what actually differs per player.
        """
        c = cv2.resize(crop, (24, 48))
        head, legs = c[0:16], c[32:48]
        feat = []
        for b in (head, legs):
            feat += [float(b[:, :, 0].mean()), float(b[:, :, 1].mean()), float(b[:, :, 2].mean())]
        return np.array(feat, dtype=float)

    # ---- build the gallery from the players who actually appear in the stats ----
    def _refresh_gallery(self):
        stats, _, _ = reaggregate(self.meta, self.tracks_path, self.merges, self.overrides, self.swap_teams)
        self.stats = stats
        skip, fps = self.meta["skip_frames"], self.meta["fps"]

        uf = self._uf()
        members = defaultdict(list)
        for rid in self.all_raw:
            members[uf.find(rid)].append(rid)

        gallery = []
        for canon, s in stats.players.items():
            poss = s["possession_frames"] * skip / fps
            if poss < GALLERY_MIN_POSS_S and s["goals"] == 0 and s["assists"] == 0:
                continue
            mem = members.get(canon, [canon])
            rep = self._rep_member(mem)
            crop = self._crop_for(rep)
            if crop is None:
                continue
            gallery.append({
                "canon": canon, "members": mem, "crop": crop, "poss": poss,
                "team": s["team"] if s["team"] in (0, 1) else self.effective_team(canon, mem),
                "disp": self.display_ids.get(rep, rep),
                "feat": self._appearance(crop),
            })

        # Normalise appearance features (z-score per dimension) for "select similar".
        if gallery:
            F = np.array([g["feat"] for g in gallery])
            mu, sd = F.mean(0), F.std(0) + 1e-6
            for g in gallery:
                g["zfeat"] = (g["feat"] - mu) / sd

        self.gallery = gallery
        self._sort_gallery()

    def _sort_gallery(self):
        if self.sort_mode == "team":
            # group by assigned team (0, then 1, then unknown) so a wrong-coloured
            # player sticks out among their team's thumbnails
            self.gallery.sort(key=lambda g: (g["team"] if g["team"] in (0, 1) else 2, -g["poss"]))
        else:
            # most-involved players first; the long tail of fragments goes to later pages
            self.gallery.sort(key=lambda g: -g["poss"])

    def id_map(self) -> dict[int, int]:
        return {g["canon"]: g["disp"] for g in self.gallery}

    # ---- rendering ----
    def render(self):
        cv2.imshow(self.win, self._build_canvas())

    def _build_canvas(self):
        cols = max(1, CANVAS_W // CELL_W)
        per_page = cols * PAGE_ROWS
        n_pages = max(1, (len(self.gallery) + per_page - 1) // per_page)
        self.page = max(0, min(self.page, n_pages - 1))
        page_items = self.gallery[self.page * per_page:(self.page + 1) * per_page]

        canvas = np.full((PAGE_ROWS * CELL_H + 60, CANVAS_W, 3), 30, np.uint8)
        self.cell_rects = []
        self._page_canons = {g["canon"] for g in page_items}

        for i, g in enumerate(page_items):
            r, c = divmod(i, cols)
            cx, cy = c * CELL_W + 8, r * CELL_H + 50
            color = TEAM_COLORS_BGR.get(g["team"], TEAM_COLORS_BGR[-1])
            canvas[cy:cy + THUMB_H, cx:cx + THUMB_W] = cv2.resize(g["crop"], (THUMB_W, THUMB_H))
            sel = g["canon"] in self.selected
            cv2.rectangle(canvas, (cx - 2, cy - 2), (cx + THUMB_W + 2, cy + THUMB_H + 2),
                          (0, 255, 255) if sel else color, 3 if sel else 2)
            tag = f"#{g['disp']}"
            if g["canon"] in self.overrides:
                tag += "*"
            if len(g["members"]) > 1:
                tag += f" x{len(g['members'])}"
            cv2.putText(canvas, tag, (cx, cy + THUMB_H + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            self.cell_rects.append((cx, cy, cx + THUMB_W, cy + THUMB_H, g["canon"]))

        cv2.rectangle(canvas, (0, 0), (CANVAS_W, 40), (0, 0, 0), -1)
        head = (f"{len(self.gallery)} players (page {self.page+1}/{n_pages}, N/P) | sort:{self.sort_mode} (B) | "
                f"selected:{len(self.selected)} | merges:{len(self.merges)} | overrides:{len(self.overrides)}")
        cv2.putText(canvas, head, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, self.status, (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return canvas

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, canon in self.cell_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected.discard(canon) if canon in self.selected else self.selected.add(canon)
                return

    # ---- actions ----
    def _push(self):
        """Snapshot current state so Z can undo the next action."""
        self.history.append(([list(m) for m in self.merges], dict(self.overrides), self.swap_teams))
        if len(self.history) > 100:
            self.history.pop(0)

    def undo_last(self):
        if not self.history:
            self.status = "Nothing to undo."
            return
        self.merges, self.overrides, self.swap_teams = self.history.pop()
        self.selected.clear()
        self._refresh_gallery()
        self.status = f"Undid last action. ({len(self.history)} more undos available)"

    def merge_selected(self):
        sel = list(self.selected)
        if len(sel) < 2:
            self.status = "Select 2+ players to merge."
            return
        self._push()
        for other in sel[1:]:
            self.merges.append([sel[0], other])
        self.selected.clear()
        self._refresh_gallery()
        self.status = f"Merged {len(sel)} players into one.  (Z to undo)"

    def flip_selected(self):
        if not self.selected:
            self.status = "Select players first."
            return
        self._push()
        n = len(self.selected)
        uf = self._uf()
        cur_team = {g["canon"]: g["team"] for g in self.gallery}
        for canon in self.selected:
            # clear any stale overrides on other members of this merged group
            for m in [r for r in self.all_raw if uf.find(r) == canon]:
                self.overrides.pop(m, None)
            cur = cur_team.get(canon, -1)
            self.overrides[canon] = 1 - cur if cur in (0, 1) else 0
        self._refresh_gallery()
        self.status = f"Flipped team on {n} player(s)."

    def toggle_sort(self):
        self.sort_mode = "team" if self.sort_mode == "poss" else "poss"
        self._sort_gallery()
        self.page = 0
        self.status = ("Sorted BY TEAM — wrong-coloured players stick out; click + T to fix."
                       if self.sort_mode == "team" else "Sorted by involvement (most touches first).")

    def swap_all_teams(self):
        self._push()
        self.swap_teams = not self.swap_teams
        self._refresh_gallery()
        self.status = f"Swapped team colours globally (now {'ON' if self.swap_teams else 'OFF'}). Manual fixes kept."

    def select_similar(self):
        """Grow the selection to all same-team players that look like the selected one."""
        if len(self.selected) != 1:
            self.status = "Select exactly ONE player, then A to grab all who look like them."
            return
        canon = next(iter(self.selected))
        ref = next((g for g in self.gallery if g["canon"] == canon), None)
        if ref is None or "zfeat" not in ref:
            return
        n0 = len(self.selected)
        # Only consider players on the CURRENT page, so you can see everything
        # that gets selected before merging.
        for g in self.gallery:
            if (g["canon"] in self._page_canons and g["team"] == ref["team"] and "zfeat" in g):
                if float(np.linalg.norm(g["zfeat"] - ref["zfeat"])) <= SIMILAR_THRESH:
                    self.selected.add(g["canon"])
        self.status = (f"Selected {len(self.selected)} look-alikes on this page (was {n0}). "
                       "Deselect any wrong ones, then M to merge.")

    def undo_selected(self):
        self._push()
        for canon in self.selected:
            self.overrides.pop(canon, None)
            self.merges = [m for m in self.merges if canon not in m]
        n = len(self.selected)
        self.selected.clear()
        self._refresh_gallery()
        self.status = f"Reverted {n} player(s)."

    def save(self):
        with open(self.corr_path, "w", encoding="utf-8") as f:
            json.dump({"team_overrides": {str(k): v for k, v in self.overrides.items()},
                       "merges": self.merges, "swap_teams": self.swap_teams}, f, indent=2)
        df = self.stats.get_dataframe(self.meta["fps"], self.meta["skip_frames"], id_map=self.id_map())
        out_csv = os.path.join("output", f"{self.base}_player_stats_corrected.csv")
        if not df.empty:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        self.status = f"Saved -> {out_csv}  ({len(df)} players)"
        print(self.status)

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)
        while True:
            self.render()
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('m'):
                self.merge_selected()
            elif k == ord('t'):
                self.flip_selected()
            elif k == ord('g'):
                self.swap_all_teams()
            elif k == ord('a'):
                self.select_similar()
            elif k == ord('b'):
                self.toggle_sort()
            elif k == ord('z'):
                self.undo_last()
            elif k == ord('u'):
                self.undo_selected()
            elif k == ord('c'):
                self.selected.clear()
            elif k == ord('n'):
                self.page += 1
            elif k == ord('p'):
                self.page = max(0, self.page - 1)
            elif k == ord('s'):
                self.save()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python correct.py <video_or_base_name>")
    Corrector(sys.argv[1]).run()

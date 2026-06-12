# Football Stat Tracker

Automatically extracts stats from Veo 3 match recordings using GPU-accelerated object detection and player tracking.

## What it detects

| Stat | Notes |
|------|-------|
| Passes (completed / attempted / %) | Per player |
| Defensive contributions | Ball won from opposing team (tackles, blocks, interceptions) |
| Possession time | Per player |
| Goals + Assists | Requires one-time goal zone setup |
| Annotated video | Bounding boxes, team colours, score HUD, event notifications |

---

## How to open a terminal in the project folder

**Option A — VS Code (recommended):** open the `football_stat_tracker` folder in VS Code, then open the integrated terminal with `` Ctrl+` ``.

**Option B — Windows Explorer:** navigate to `e:\footballVids\football_stat_tracker`, then right-click an empty area and choose **Open in Terminal**.

All commands below are typed into that terminal and run with Enter. The `.\venv\Scripts\python` prefix tells Windows to use the project's own Python installation rather than any other Python on your machine.

---

## Setup (first time only)

**Set up goal zones for your pitch**

Run this once per pitch/camera angle. A window opens — click 4 corners around each goal mouth, press Enter.
```
.\venv\Scripts\python setup_goals.py "..\match-farhat-fc-2026-06-08.mp4"
```
The zones save as `match-farhat-fc-2026-06-08_goals.json` next to the video and are picked up automatically on future runs.

---

## Running an analysis

**Stats only (fast — good for full matches):**
```
.\venv\Scripts\python analyze.py "..\match-farhat-fc-2026-06-08.mp4" --no-video --skip 3
```

**Stats + annotated video:**
```
.\venv\Scripts\python analyze.py "..\match-farhat-fc-2026-06-08.mp4" --skip 3
```

**Quick test on a short goal clip:**
```
.\venv\Scripts\python analyze.py "..\001228-goal.mp4" --no-video --skip 2
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--skip N` | 3 | Process every Nth frame. Higher = faster, slightly less accurate. |
| `--model` | `yolov8n.pt` | Detection model. `yolov8m.pt` is more accurate, ~2× slower. |
| `--no-video` | off | Skip annotated video output — much faster if you only need stats. |
| `--conf` | 0.3 | Detection confidence threshold. |
| `--goals` | auto | Path to `*_goals.json`. Auto-detected if it sits next to the video. |

---

## Outputs

All files go to the `output/` folder.

| File | What's in it |
|------|-------------|
| `*_player_stats.csv` | One row per tracked player: goals, assists, passes, defensive contributions, possession time |
| `*_events.csv` | Every detected event with timestamp: passes, goals, defensive actions |
| `*_annotated.mp4` | Video with bounding boxes, team colours, score HUD, event pop-ups |

**Open CSVs directly in Excel** — they are saved as UTF-8 with BOM so Excel reads them correctly.

---

## Finding yourself in the stats

1. Watch the annotated video — each player has a `#ID` label on their bounding box.
2. Note your ID (e.g. `#32`).
3. Look up `#32` in `*_player_stats.csv`.

Your ID may change if you leave the frame for an extended period and re-enter — this is a known limitation of the tracker.

---

## Performance

Tested on RTX 4070 SUPER with `yolov8n.pt`:

| Video | Skip | Speed | Time for 90 min match |
|-------|------|-------|----------------------|
| 1080p | 3 | ~90 fps | ~8 min |
| 1080p | 5 | ~120 fps | ~5 min |

---

## Known limitations

- **Team classification** uses jersey colour. If both teams wear similar colours the split may be wrong — check the annotated video to verify.
- **Ball detection** can miss when the ball is small or occluded. Goals won't register if the ball isn't detected near the goal zone.
- **Goalkeeper** will likely be classified into the wrong team (different jersey colour) — their defensive contributions still count.
- **Stats only start after calibration** — the first ~3–5 seconds of each video are used to learn team colours, so very early events may be missed.

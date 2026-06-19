# How to run the tracker on a full game

A practical checklist for analysing a full match. No Python knowledge needed —
everything is done through the app window.

---

## 1. Before you start

- **Stop the PC sleeping.** A full game takes a while; if the machine sleeps or
  the screensaver kicks in, the run is lost.
  Windows Settings → System → Power → set **Sleep = Never** while it runs.
- **Close the output CSVs** if you have them open in Excel (a locked file blocks
  the save at the end).
- Make sure there are a few GB free on the drive.

---

## 2. Run the analysis

1. Double-click **`Open Football Tracker.bat`**.
2. Drag the game video into the box (or click **Browse**).
3. Settings:
   - **Speed/accuracy:** *Balanced* (skip 3) is the right default for a long game.
     *Faster* (skip 5) is ~40% quicker but less precise.
   - **Leave "Also generate annotated video" OFF.** Writing the video is the
     slowest part and you don't need it — the correction gallery reads the
     original video directly. Stats-only is far faster.
   - **Leave "Enable goal detection" OFF** — goals aren't reliable yet (the
     panning camera breaks the fixed goal zones).
4. Click **"2. Run Analysis"** and leave it running.
   **Do not close the window** — that stops the job.

**Roughly how long:** the 4-minute test clip is ~3-5 min stats-only, so a
90-minute match is roughly **1–2 hours**. Your sessions may be shorter.

---

## 3. When it finishes

Click **"Open Output Folder"**. You get:

| File | What it is |
|------|-----------|
| `<game>_events.csv` | Every detected pass/turnover with a timestamp |
| `<game>_player_stats.csv` | Per-player totals (well-tracked players only) |
| `<game>_tracks.jsonl` + `_tracks_meta.json` | Tracking log used by the correction tool |

**Reading the numbers:**
- **Total passes** = count the `PASS` rows in `_events.csv` — *not* by summing the
  player-stats file (that one is filtered to players with enough possession).
- **Passes and possession are the trustworthy outputs.**
- **Ignore the "Defensive contributions" column for now** — it under-counts and
  isn't reliable until the action-recognition model is built.
- Goals are not produced (detection is off).

---

## 4. Optional: clean up per-player stats

Click **"3. Correct Results"** to open the player gallery:

- If the team colours are inverted (red players in blue boxes etc.), press **G**
  to swap them all at once.
- Click the thumbnails that are the **same person** (one player often splits into
  several tracks over a long game) and press **M** to merge them.
- Click a player + **T** to fix a wrong team colour.
- Press **S** to save → writes `<game>_player_stats_corrected.csv` with the
  merged totals.

Over a full 90 minutes there will be more split tracks than in a short clip, so
expect more merging here if you want clean per-player numbers.

---

## What's reliable vs not (current state)

| Stat | Status |
|------|--------|
| Passes (count, completion %) | ✅ Reliable (validated against manual count) |
| Possession time | ✅ Reliable |
| Defensive contributions | ⚠️ Under-counts — not trustworthy yet (needs action recognition) |
| Goals / assists | ❌ Off — needs pitch-mapping work |

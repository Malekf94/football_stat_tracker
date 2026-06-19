# Football Stat Tracker — Project Status & Roadmap

A living document for the project. Records what it does, what's reliable, the key
decisions/lessons so far, and where we're heading. Update as we build.

_Last updated: 2026-06-16_

---

## 1. Goal

Automatically extract per-player and per-team stats from Veo-camera match
recordings — replacing the manual "watch it back and note goals/assists/defensive
contributions" process. Target stats: **goals, assists, passes, defensive
contributions**, later dribbles/distance/heatmaps.

Footage characteristics that drive every design decision:
- **Veo camera auto-pans/zooms to follow the ball** (no fixed view).
- **Plain coloured bibs, no numbers** (e.g. red vs light-blue); bib colours
  change between sessions.
- Amateur pitch, variable lighting, players occlude each other constantly.
- Full matches ~80-90 min (huge files, e.g. one is 17 GB).

---

## 2. Pipeline (how it works now)

```
video ─► YOLO11 detection ─► BoT-SORT tracking (ReID + camera-motion comp)
                              │
                              ├─ ball: separate low-confidence detection pass
                              ├─ teams: SigLIP torso-embedding clustering (per match)
                              ├─ camera motion: logged per frame (optical flow)
                              └─ events: possession → PASS / DEFENSIVE_ACTION (+ GOAL if enabled)
                                          │
        tracking log (jsonl) ◄───────────┘
                              │
        auto re-ID (stitch fragments, camera-stabilised) ─► stats CSV + events CSV
                              │
        correct.py (manual gallery cleanup) ─► corrected stats CSV
```

**Run it:** see `HOW_TO_RUN.md`. In short: GUI (`Open Football Tracker.bat`) →
select video → *Run Analysis* (stats-only, no goals) → *Correct Results*.

---

## 3. What's reliable vs not (current honest state)

| Stat / capability | Status |
|---|---|
| **Passes** (count, completion %) | ✅ Reliable on clips; validated 12/13 in a manual check |
| **Possession time** | ✅ Reliable |
| **Player/ball detection** | ✅ Good (ball fixed via low-conf pass: ~10% → ~90%) |
| **Team split** | 🟡 Mostly right, a few players mis-coloured — fix in gallery (A+T) |
| **Per-player stats on a full match** | 🟡 Needs manual cleanup; fragments heavily (see §5) |
| **Defensive contributions** | ❌ Proxy only; under-counts. Needs action-recognition |
| **Goals / assists** | ❌ Off — fixed goal zones break under the panning camera |

---

## 4. Key decisions & lessons (why things are the way they are)

- **Tracking: ByteTrack → BoT-SORT (ReID + GMC).** ByteTrack is motion-only and
  re-numbered players constantly under the pan. BoT-SORT adds appearance + camera
  motion compensation.
- **Ball detection was the big bug.** The small, blurry ball scores low
  confidence; the tracker's threshold discarded it (~10% kept). Fix: a **separate
  low-confidence detection pass** for the ball (~90%). This is what made passes
  work at all (19 → ~50 → sensible counts).
- **Possession is measured at the feet**, not the torso centre (the ball is at
  the feet).
- **Event logic is heavily guarded** to avoid phantom events: debounce, a
  "contested scrum" rule (don't assign possession when two players are equally
  close), a "sustained turnover" rule (an opponent must hold the ball ~1s — a
  brief touch during your passing stays a pass), and skipping events when a
  team label is unknown. These turned wildly-inflated turnover counts into sane
  numbers.
- **Team classification: tried HSV → SigLIP → colour (BGR/hue) → back to SigLIP.**
  Lesson: you can't *train* a fixed team classifier because the two teams are
  different colours every match — it's inherently per-match **clustering**.
  SigLIP torso-embedding clustering splits the squad most evenly; colour methods
  misread blue bibs in shadow as red. Residual errors are fixed in the gallery.
- **Goals disabled.** Fixed pixel goal-zones don't work because the camera pans
  the goal out of the zone. Proper fix needs per-frame goal tracking / pitch
  mapping.
- **Auto re-ID by spatio-temporal stitching, camera-stabilised.** Can't use
  appearance to tell same-bib teammates apart, so we stitch a player's broken
  trail by *where they were* — corrected for camera motion so a player lost
  during a pan stays linkable. Roughly halves fragments. (Also fixed an O(n²)
  performance bug → full-game stitch now <1s.)
- **Manual correction tool (`correct.py`)** — a paginated player gallery:
  click + **A** to select look-alikes, **M** merge, **T** flip team, **G** swap
  all team colours, **S** save. The safety net for whatever automation misses.

---

## 5. Known limitations / the hard problems

- **Full-match fragmentation.** ~6,000 tracks for ~16 players over 82 min (a
  track breaks every ~14s) because of the panning camera + occlusions + identical
  bibs. Camera-comp stitching halves it but doesn't get to a clean 16-22. Per-
  player full-match stats therefore still need gallery cleanup; **team-level
  totals (from `events.csv`) are the reliable full-match output.**
- **Telling teammates apart** (same bib, build) is at/beyond what appearance can
  do — the "select look-alikes" helper works for *distinctive* players (e.g. a
  dark-skinned player) but not lookalikes.
- **Defensive contributions** as defined (tackles/blocks/clearances/saves) are
  *actions*, not positions — the possession proxy can't capture them.
- **Off-pitch players** on an adjacent pitch are filtered by a size threshold
  (works because they're far/small), but it's a heuristic.

---

## 6. Roadmap / future steps

Ordered roughly by value-for-effort:

1. **Fine-tune the detector (players + ball) on our own matches.** Highest-
   leverage training: better ball + fewer missed players lifts passes and
   reduces fragmentation. Bootstrap labelling by correcting the current model's
   pre-labels (Roboflow). Robust across the different bib colours we have.
2. **Action-recognition for defensive contributions (and goals).** The genuine
   "train a model" task — tag tackle/block/clearance/save (and goal) examples
   across our many matches, train a classifier that fires on the short windows
   around ball events. Build a labelling tool first.
3. **Better full-match per-player identity.** Improve stitching (tune camera-comp
   gaps), consider a football-tuned re-ID. Accept gallery cleanup until then.
4. **Pitch mapping (homography) — exploratory.** Would unlock panning-robust
   goals + heatmaps + distances, but the moving camera makes it risky on an
   amateur pitch (same reason fixed goal-zones failed). Needs a feasibility test.
5. **Match report output** — PDF/HTML with tables + heatmaps once the above land.

**What training will NOT fix** (so we don't over-invest): team classification
(per-match clustering, varying bibs) and the core tracking fragmentation (a
camera/occlusion problem, not a detection one).

---

## 7. Key files

| File | Purpose |
|---|---|
| `analyze.py` | Main pipeline: detect → track → teams → events → stats + logs |
| `tracker/detection.py` | YOLO11 + BoT-SORT; separate low-conf ball pass; size filter |
| `tracker/teams.py` | SigLIP torso-embedding team clustering (per-track voting) |
| `tracker/events.py` | Possession → PASS/DEFENSIVE/GOAL with the guards in §4 |
| `tracker/camera.py` | Per-frame camera-motion estimation (for stitching) |
| `tracker/reid.py` | Auto re-ID: camera-stabilised tracklet stitching |
| `tracker/replay.py` | Re-aggregate stats/events from the log + corrections |
| `tracker/stats.py` | Stat tallying + CSV output |
| `correct.py` | Manual correction gallery (merge / flip team / look-alikes) |
| `setup_goals.py` | One-off goal-zone drawing (goals currently disabled) |
| `gui.py` + `Open Football Tracker.bat` | Desktop GUI launcher |
| `HOW_TO_RUN.md` | Step-by-step run guide |

Outputs (in `output/`): `*_player_stats.csv`, `*_events.csv`,
`*_tracks.jsonl` + `*_tracks_meta.json` (log for correction),
`*_player_stats_corrected.csv` (after gallery cleanup).

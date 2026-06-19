"""
Football Video Stat Analyzer
============================
Usage:
    python analyze.py <video_path> [options]

Options:
    --goals    Path to <video>_goals.json (created by setup_goals.py).
               Auto-detected if the file sits next to the video.
    --model    YOLO weights: yolo11n.pt (fast) | yolo11m.pt | yolo11l.pt (accurate)
               Default: yolo11n.pt
    --skip     Process every Nth frame.  3 = ~3× faster, minimal quality loss. Default: 3
    --output   Output directory.  Default: output/
    --no-video Skip writing annotated video (much faster if you only want stats).
    --conf     Detection confidence threshold.  Default: 0.3

Examples:
    # Quick test on a goal clip
    python analyze.py "../001228-goal.mp4" --no-video

    # Full match with GPU, better model, annotated video
    python analyze.py "../match-farhat-fc-2026-06-08.mp4" --model yolov8m.pt --skip 3

    # Stats only, no video output
    python analyze.py "../match-farhat-fc-2026-06-08.mp4" --no-video --skip 2
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm

from tracker.camera import CameraMotion
from tracker.detection import FootballDetector, DEFAULT_MODEL
from tracker.teams import TeamClassifier
from tracker.events import EventDetector
from tracker.stats import StatsAggregator


# ------------------------------------------------------------------
# Annotation helpers
# ------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _draw_players(frame: np.ndarray, players, team_clf: TeamClassifier,
                  current_possessor: int | None, display_map: dict | None = None) -> np.ndarray:
    if players is None or len(players) == 0:
        return frame

    tracker_ids = players.tracker_id
    if tracker_ids is None:
        return frame

    for i, tid in enumerate(tracker_ids):
        if tid is None:
            continue

        x1, y1, x2, y2 = map(int, players.xyxy[i])
        team = team_clf.track_team_map.get(int(tid), -1)
        color = team_clf.get_color(team)
        thickness = 3 if tid == current_possessor else 2

        # Glow effect for possessor
        if tid == current_possessor:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)

        label_id = display_map.get(int(tid), int(tid)) if display_map else int(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"#{label_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame


def _draw_ball(frame: np.ndarray, ball_pos: np.ndarray | None) -> np.ndarray:
    if ball_pos is None:
        return frame
    bx, by = int(ball_pos[0]), int(ball_pos[1])
    cv2.circle(frame, (bx, by), 14, (0, 230, 230), 2)
    cv2.circle(frame, (bx, by), 4, (0, 230, 230), -1)
    return frame


def _draw_goal_zones(frame: np.ndarray, goal_zones: list, alpha: float = 0.25) -> np.ndarray:
    overlay = frame.copy()
    colors = [(0, 200, 0), (0, 140, 255)]
    for i, zone in enumerate(goal_zones):
        if len(zone) >= 3:
            pts = np.array(zone, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], colors[i % 2])
            cv2.polylines(frame, [pts], True, colors[i % 2], 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def _draw_notifications(frame: np.ndarray,
                         notifications: list[tuple[dict, float]],
                         display_map: dict | None = None) -> np.ndarray:
    """Fade-out event banners."""
    h = frame.shape[0]
    y_base = 80

    def d(pid):
        if display_map is None or pid is None:
            return pid if pid is not None else "?"
        return display_map.get(pid, pid)

    for event, alpha in notifications:
        t = event["type"]
        if t == "GOAL":
            assister = event.get("assister")
            text = f"  GOAL!  Scorer: #{d(event.get('scorer'))}  Assist: #{d(assister) if assister is not None else '-'}  "
            bg_color, txt_color = (0, 0, 180), (255, 255, 255)
        elif t == "DEFENSIVE_ACTION":
            text = f"  Defensive contribution - #{d(event.get('player_id'))}  "
            bg_color, txt_color = (0, 130, 0), (255, 255, 255)
        else:
            continue

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, y_base - th - 6), (tw + 14, y_base + 6), bg_color, -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        cv2.putText(frame, text, (10, y_base),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, txt_color, 2, cv2.LINE_AA)
        y_base += th + 16
    return frame


def _draw_hud(frame: np.ndarray, frame_num: int, fps: float,
              stats: StatsAggregator, calibrated: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    time_str = _fmt_time(frame_num / fps)
    cv2.putText(frame, time_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1, cv2.LINE_AA)

    if not calibrated:
        status = "Calibrating teams..."
        cv2.putText(frame, status, (w // 2 - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 220, 255), 1, cv2.LINE_AA)
    else:
        g0, g1 = stats.team_goals[0], stats.team_goals[1]
        score = f"Team 0 (blue)  {g0} : {g1}  Team 1 (red)"
        (sw, _), _ = cv2.getTextSize(score, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.putText(frame, score, (w // 2 - sw // 2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 230), 1, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------
# Main analysis pipeline
# ------------------------------------------------------------------

def _calibrate_teams(video_path, detector, team_clf, conf,
                     target_crops=300, max_frames_scanned=80):
    """Pre-scan the video at a stride, collect player crops, fit the team classifier.

    Sampling across the whole match (rather than only the first seconds) gives
    SigLIP a diverse set of both teams to cluster, and means no early stats are
    lost waiting for calibration.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    stride = max(1, total // max_frames_scanned)
    crops, fidx = [], 0
    while len(crops) < target_crops and fidx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            break
        min_h = int(team_clf.CLEAR_MIN_FRAC * frame.shape[0])  # calibrate on CLEAR (close) players only
        res = detector.model.predict(
            frame, conf=conf, imgsz=detector.imgsz,
            classes=[detector.person_class], verbose=False, device=detector.device,
        )[0]
        det = sv.Detections.from_ultralytics(res)
        for xyxy in det.xyxy:
            c = team_clf.crop_player(frame, xyxy, min_h=min_h)
            if c is not None:
                crops.append(c)
        fidx += stride
    cap.release()
    team_clf.calibrate(crops)


def run_analysis(
    video_path: str,
    goals_path: str | None = None,
    model_name: str = DEFAULT_MODEL,
    skip_frames: int = 3,
    output_dir: str = "output",
    no_video: bool = False,
    conf: float = 0.25,
) -> tuple[StatsAggregator, list[dict]]:

    os.makedirs(output_dir, exist_ok=True)

    # ---- Goal zones ----
    goal_zones: list = []
    if goals_path and os.path.exists(goals_path):
        with open(goals_path) as f:
            goal_zones = json.load(f).get("goal_zones", [])
        print(f"[Goals] {len(goal_zones)} zone(s) loaded from {goals_path}")
    else:
        print("[Goals] None configured — goal/assist detection disabled.")
        print("        Run:  python setup_goals.py <video_path>  to set up goals.\n")

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_m  = total_f / fps / 60

    print(f"[Video] {os.path.basename(video_path)}")
    print(f"        {W}×{H} @ {fps:.1f} fps | {duration_m:.1f} min ({total_f} frames)")
    print(f"        Sampling every {skip_frames} frame(s) → ≈{total_f // skip_frames:,} inference passes")

    # ---- Components ----
    print(f"\n[Model] Loading {model_name} …")
    detector  = FootballDetector(model_name=model_name, conf=conf)
    team_clf  = TeamClassifier()
    _poss_px = max(60, int(min(W, H) * 0.065))
    event_det = EventDetector(
        goal_zones=goal_zones,
        possession_px=_poss_px,
        # ~0.3s of held possession before a turnover/pass counts — expressed in
        # processed frames so it means the same time regardless of --skip.
        possession_confirm_frames=max(1, round(0.3 * fps / skip_frames)),
        # the ball must move at least this far for a pass/turnover to count
        # (filters phantom passes from a player's id switching in place).
        possession_min_separation=_poss_px * 0.7,
        # a 2nd player within this margin = contested scrum → possession unchanged
        contest_margin=_poss_px * 0.6,
        # the opposing team must hold the ball ~1s before it counts as a turnover
        turnover_confirm_frames=max(2, round(1.0 * fps / skip_frames)),
    )
    stats = StatsAggregator()

    # ---- Team calibration (pre-scan the whole match for both teams) ----
    print("\n[Teams] Pre-scanning video to learn the two teams …")
    _calibrate_teams(video_path, detector, team_clf, conf)
    calibrated = team_clf.calibrated

    # ---- Video writer ----
    writer = None
    out_video_path = ""
    if not no_video:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_video_path = os.path.join(output_dir, f"{base}_annotated.mp4")
        writer = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,          # keep original fps — we'll write skip_frames copies per detection
            (W, H),
        )

    # ---- Notification state ----
    NOTIF_FADE_FRAMES = 70    # processed frames to show each notification
    notifications: list[list] = []   # [[event, alpha], ...]

    frame_num = 0
    processed = 0

    # Compact display IDs: raw BoT-SORT ids climb into the thousands (the counter
    # ticks for every track ever created). Remap them to #1, #2, … in order of
    # first appearance for the video labels and the CSV — internal ids stay raw.
    display_map: dict[int, int] = {}

    # Tracking log — one JSON line per processed frame (player boxes + team + ball).
    # Lets correct.py replay the match and re-aggregate stats after manual fixes
    # without re-running detection. Written incrementally to keep memory low.
    base = os.path.splitext(os.path.basename(video_path))[0]
    tracks_path = os.path.join(output_dir, f"{base}_tracks.jsonl")
    tracks_file = open(tracks_path, "w", encoding="utf-8")

    # Camera-motion tracker: logs the cumulative pan/zoom each frame so fragments
    # can later be stitched in a stabilised frame (see tracker/reid.py).
    cammotion = CameraMotion()

    print("\n[Running] Press Ctrl+C to stop early and save partial results.\n")

    try:
        with tqdm(total=total_f, unit="fr", desc="Analyzing", dynamic_ncols=True) as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % skip_frames == 0:
                    # ---- Detection + tracking ----
                    players, ball_pos, ball_xyxy = detector.process_frame(frame)

                    # ---- Camera motion (mask players so only the background counts) ----
                    _boxes = players.xyxy if (players is not None and len(players) > 0) else []
                    _M = cammotion.update(frame, _boxes)
                    cam6 = [round(float(_M[0, 0]), 5), round(float(_M[0, 1]), 5), round(float(_M[0, 2]), 2),
                            round(float(_M[1, 0]), 5), round(float(_M[1, 1]), 5), round(float(_M[1, 2]), 2)]

                    # ---- Team classification (teams pre-calibrated before the loop) ----
                    if calibrated and players is not None and len(players) > 0:
                        tids = players.tracker_id
                        if tids is not None:
                            for i, tid in enumerate(tids):
                                if tid is None:
                                    continue
                                team_clf.classify_player(int(tid), frame, players.xyxy[i])
                                # Assign a compact display id on first sighting
                                if int(tid) not in display_map:
                                    display_map[int(tid)] = len(display_map) + 1

                    # ---- Build player_info list (+ tracking-log records) ----
                    player_info: list[tuple[int, np.ndarray, int]] = []
                    frame_players: list[list] = []
                    if calibrated and players is not None and len(players) > 0:
                        tids = players.tracker_id
                        if tids is not None:
                            for i, tid in enumerate(tids):
                                if tid is None:
                                    continue
                                x1, y1, x2, y2 = players.xyxy[i]
                                center = np.array([(x1 + x2) / 2, y2])  # feet — ball is at the feet
                                team = team_clf.track_team_map.get(int(tid), -1)
                                player_info.append((int(tid), center, team))
                                frame_players.append([int(tid), round(float(x1), 1), round(float(y1), 1),
                                                      round(float(x2), 1), round(float(y2), 1), int(team)])

                    # ---- Events ----
                    new_events = event_det.update(frame_num, fps, player_info, ball_pos)
                    for ev in new_events:
                        stats.process_event(ev)
                        if ev["type"] in ("GOAL", "DEFENSIVE_ACTION"):
                            notifications.append([ev, 1.0])

                    stats.tick_possession(event_det.current_possessor, event_det.current_possessor_team)

                    # ---- Tracking-log line for this processed frame ----
                    _ball = [round(float(ball_pos[0]), 1), round(float(ball_pos[1]), 1)] if ball_pos is not None else None
                    tracks_file.write(json.dumps({"f": frame_num, "ball": _ball, "players": frame_players, "cam": cam6}) + "\n")

                    # ---- Fade notifications ----
                    notifications = [[e, a - 1.0 / NOTIF_FADE_FRAMES]
                                     for e, a in notifications if a > 0]

                    # ---- Annotate + write ----
                    if writer:
                        annotated = frame.copy()
                        annotated = _draw_goal_zones(annotated, goal_zones)
                        annotated = _draw_players(annotated, players, team_clf,
                                                  event_det.current_possessor, display_map)
                        annotated = _draw_ball(annotated, ball_pos)
                        annotated = _draw_notifications(annotated, notifications, display_map)
                        annotated = _draw_hud(annotated, frame_num, fps, stats, calibrated)
                        # Write skip_frames copies so output runs at original speed
                        for _ in range(skip_frames):
                            writer.write(annotated)

                    processed += 1

                frame_num += 1
                bar.update(1)
                bar.set_postfix(goals=f"{stats.team_goals[0]}-{stats.team_goals[1]}")

    except KeyboardInterrupt:
        print("\n[!] Stopped early — saving results.")
    finally:
        cap.release()
        if writer:
            writer.release()
        tracks_file.close()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"\n[Done] Processed {processed:,} frames out of {frame_num:,}")

    # Tracking-log metadata (config + final teams + display ids) for correct.py
    meta = {
        "fps": fps, "skip_frames": skip_frames, "frame_w": W, "frame_h": H,
        "possession_px": event_det.possession_px,
        "possession_confirm_frames": event_det.possession_confirm_frames,
        "possession_min_separation": event_det.possession_min_separation,
        "contest_margin": event_det.contest_margin,
        "turnover_confirm_frames": event_det.turnover_confirm_frames,
        "goal_cooldown_frames": event_det.goal_cooldown_frames,
        "goal_zones": goal_zones,
        "video_path": os.path.abspath(video_path),
        "display_map": {str(k): v for k, v in display_map.items()},
        "final_teams": {str(k): int(v) for k, v in team_clf.track_team_map.items()},
    }
    with open(os.path.join(output_dir, f"{base}_tracks_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf)
    print(f"[Output] Tracking log   → {tracks_path}")

    # ---- Auto re-identification: stitch a player's fragments, then recompute ----
    from tracker.reid import auto_merge
    from tracker.replay import reaggregate
    auto_merges = auto_merge(meta, tracks_path)
    stats, display_map, events = reaggregate(meta, tracks_path, auto_merges)
    print(f"[Auto-ReID] stitched {len(auto_merges)} fragment links; recomputed stats.")

    stats.print_summary(fps, skip_frames, id_map=display_map)

    def _disp(pid):
        """Canonical id -> compact display id (falls back to raw)."""
        try:
            return display_map.get(int(pid), pid)
        except (TypeError, ValueError):
            return pid

    # Player stats CSV (written first — more important)
    df = stats.get_dataframe(fps, skip_frames, id_map=display_map)
    if not df.empty:
        stats_path = os.path.join(output_dir, f"{base}_player_stats.csv")
        try:
            df.to_csv(stats_path, index=False, encoding="utf-8-sig")
            print(f"[Output] Player stats   → {stats_path}")
        except PermissionError:
            print(f"[!] Could not write {stats_path} — close it in Excel first, then re-run.")

    # Events CSV
    if events:
        rows = []
        for ev in events:
            row = {
                "Time":  _fmt_time(ev.get("time_sec", 0)),
                "Frame": ev.get("frame", 0),
                "Type":  ev["type"],
            }
            if ev["type"] == "GOAL":
                assister = ev.get("assister")
                row["Player"] = _disp(ev.get("scorer", "?"))
                row["Detail"] = f"Assist: #{_disp(assister) if assister is not None else '—'}  Team {ev.get('scoring_team','?')} scores"
            elif ev["type"] == "PASS":
                row["Player"] = _disp(ev.get("from_player", "?"))
                row["Detail"] = f"-> #{_disp(ev.get('to_player','?'))} (Team {ev.get('team','?')})"
            elif ev["type"] == "DEFENSIVE_ACTION":
                row["Player"] = _disp(ev.get("player_id", "?"))
                row["Detail"] = f"Team {ev.get('team','?')} wins ball from Team {ev.get('from_team','?')}"
            rows.append(row)

        events_path = os.path.join(output_dir, f"{base}_events.csv")
        try:
            pd.DataFrame(rows).to_csv(events_path, index=False, encoding="utf-8-sig")
            print(f"[Output] Events log     → {events_path}")
        except PermissionError:
            print(f"[!] Could not write {events_path} — close it in Excel first, then re-run.")

    if writer:
        print(f"[Output] Annotated video → {out_video_path}")

    return stats, events


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Football video stat analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video",       help="Path to football video file")
    parser.add_argument("--goals",     help="Path to <video>_goals.json")
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help="detection weights [default: fine-tuned finetuned.pt if present, else yolo11n.pt]")
    parser.add_argument("--skip",      type=int, default=3,
                        help="Process every Nth frame [default: 3]")
    parser.add_argument("--output",    default="output",
                        help="Output directory [default: output/]")
    parser.add_argument("--no-video",  action="store_true",
                        help="Skip annotated video (much faster)")
    parser.add_argument("--no-goals",  action="store_true",
                        help="Disable goal zone detection even if a _goals.json exists")
    parser.add_argument("--conf",      type=float, default=0.25,
                        help="Detection confidence [default: 0.25 — low so the small ball still detects]")

    args = parser.parse_args()

    # Auto-detect goals file next to the video
    goals_path = None if args.no_goals else args.goals
    if not args.no_goals and goals_path is None:
        auto = os.path.splitext(args.video)[0] + "_goals.json"
        if os.path.exists(auto):
            goals_path = auto
            print(f"[Auto] Goal zones found: {auto}")

    run_analysis(
        video_path=args.video,
        goals_path=goals_path,
        model_name=args.model,
        skip_frames=args.skip,
        output_dir=args.output,
        no_video=args.no_video,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()

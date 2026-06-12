"""
Football Video Stat Analyzer
============================
Usage:
    python analyze.py <video_path> [options]

Options:
    --goals    Path to <video>_goals.json (created by setup_goals.py).
               Auto-detected if the file sits next to the video.
    --model    YOLOv8 weights: yolov8n.pt (fast) | yolov8m.pt | yolov8l.pt (accurate)
               Default: yolov8n.pt
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
from tqdm import tqdm

from tracker.detection import FootballDetector
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
                  current_possessor: int | None) -> np.ndarray:
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"#{int(tid)}", (x1, y1 - 5),
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
                         notifications: list[tuple[dict, float]]) -> np.ndarray:
    """Fade-out event banners."""
    h = frame.shape[0]
    y_base = 80
    for event, alpha in notifications:
        t = event["type"]
        if t == "GOAL":
            text = f"  GOAL!  Scorer: #{event.get('scorer','?')}  Assist: #{event.get('assister','—')}  "
            bg_color, txt_color = (0, 0, 180), (255, 255, 255)
        elif t == "DEFENSIVE_ACTION":
            text = f"  Defensive contribution — #{event.get('player_id','?')}  "
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

def run_analysis(
    video_path: str,
    goals_path: str | None = None,
    model_name: str = "yolov8n.pt",
    skip_frames: int = 3,
    output_dir: str = "output",
    no_video: bool = False,
    conf: float = 0.3,
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
    event_det = EventDetector(
        goal_zones=goal_zones,
        possession_px=max(60, int(min(W, H) * 0.065)),
    )
    stats = StatsAggregator()

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

    # ---- Calibration state ----
    CAL_SAMPLES_NEEDED = 80   # jersey colour samples before K-Means runs
    cal_features: list[np.ndarray] = []
    calibrated = False

    # ---- Notification state ----
    NOTIF_FADE_FRAMES = 70    # processed frames to show each notification
    notifications: list[list] = []   # [[event, alpha], ...]

    frame_num = 0
    processed = 0

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

                    # ---- Team calibration ----
                    if not calibrated:
                        if players is not None and len(players) > 0:
                            tids = players.tracker_id
                            if tids is not None:
                                for i, tid in enumerate(tids):
                                    if tid is None:
                                        continue
                                    feat = team_clf.extract_jersey_feature(frame, players.xyxy[i])
                                    if feat is not None:
                                        cal_features.append(feat)

                        if len(cal_features) >= CAL_SAMPLES_NEEDED:
                            team_clf.calibrate(cal_features)
                            calibrated = True
                    else:
                        # Classify any newly tracked player
                        if players is not None and len(players) > 0:
                            tids = players.tracker_id
                            if tids is not None:
                                for i, tid in enumerate(tids):
                                    if tid is None:
                                        continue
                                    team_clf.classify_player(int(tid), frame, players.xyxy[i])

                    # ---- Build player_info list ----
                    player_info: list[tuple[int, np.ndarray, int]] = []
                    if calibrated and players is not None and len(players) > 0:
                        tids = players.tracker_id
                        if tids is not None:
                            for i, tid in enumerate(tids):
                                if tid is None:
                                    continue
                                x1, y1, x2, y2 = players.xyxy[i]
                                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                                team = team_clf.track_team_map.get(int(tid), -1)
                                player_info.append((int(tid), center, team))

                    # ---- Events ----
                    new_events = event_det.update(frame_num, fps, player_info, ball_pos)
                    for ev in new_events:
                        stats.process_event(ev)
                        if ev["type"] in ("GOAL", "DEFENSIVE_ACTION"):
                            notifications.append([ev, 1.0])

                    stats.tick_possession(event_det.current_possessor, event_det.current_possessor_team)

                    # ---- Fade notifications ----
                    notifications = [[e, a - 1.0 / NOTIF_FADE_FRAMES]
                                     for e, a in notifications if a > 0]

                    # ---- Annotate + write ----
                    if writer:
                        annotated = frame.copy()
                        annotated = _draw_goal_zones(annotated, goal_zones)
                        annotated = _draw_players(annotated, players, team_clf,
                                                  event_det.current_possessor)
                        annotated = _draw_ball(annotated, ball_pos)
                        annotated = _draw_notifications(annotated, notifications)
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

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"\n[Done] Processed {processed:,} frames out of {frame_num:,}")
    stats.print_summary(fps, skip_frames)

    base = os.path.splitext(os.path.basename(video_path))[0]

    # Events CSV
    if event_det.events:
        rows = []
        for ev in event_det.events:
            row = {
                "Time":  _fmt_time(ev.get("time_sec", 0)),
                "Frame": ev.get("frame", 0),
                "Type":  ev["type"],
            }
            if ev["type"] == "GOAL":
                row["Player"] = ev.get("scorer", "?")
                row["Detail"] = f"Assist: #{ev.get('assister', '—')}  Team {ev.get('scoring_team','?')} scores"
            elif ev["type"] == "PASS":
                row["Player"] = ev.get("from_player", "?")
                row["Detail"] = f"-> #{ev.get('to_player','?')} (Team {ev.get('team','?')})"
            elif ev["type"] == "DEFENSIVE_ACTION":
                row["Player"] = ev.get("player_id", "?")
                row["Detail"] = f"Team {ev.get('team','?')} wins ball from Team {ev.get('from_team','?')}"
            rows.append(row)

        events_path = os.path.join(output_dir, f"{base}_events.csv")
        pd.DataFrame(rows).to_csv(events_path, index=False, encoding="utf-8-sig")
        print(f"[Output] Events log     → {events_path}")

    # Player stats CSV
    df = stats.get_dataframe(fps, skip_frames)
    if not df.empty:
        stats_path = os.path.join(output_dir, f"{base}_player_stats.csv")
        df.to_csv(stats_path, index=False, encoding="utf-8-sig")
        print(f"[Output] Player stats   → {stats_path}")

    if writer:
        print(f"[Output] Annotated video → {out_video_path}")

    return stats, event_det.events


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
    parser.add_argument("--model",     default="yolov8n.pt",
                        help="YOLOv8 model (yolov8n/m/l.pt) [default: yolov8n.pt]")
    parser.add_argument("--skip",      type=int, default=3,
                        help="Process every Nth frame [default: 3]")
    parser.add_argument("--output",    default="output",
                        help="Output directory [default: output/]")
    parser.add_argument("--no-video",  action="store_true",
                        help="Skip annotated video (much faster)")
    parser.add_argument("--conf",      type=float, default=0.3,
                        help="Detection confidence [default: 0.3]")

    args = parser.parse_args()

    # Auto-detect goals file next to the video
    goals_path = args.goals
    if goals_path is None:
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

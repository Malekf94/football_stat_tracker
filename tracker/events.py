from __future__ import annotations
from collections import deque
import numpy as np


class EventDetector:
    """
    Heuristic event detection from frame-by-frame tracking data.

    Detects:
      GOAL              – ball enters a defined goal zone polygon
      PASS              – ball transferred between two same-team players
      DEFENSIVE_ACTION  – ball gained by the opposing team (tackle/block/clearance/save)

    Assist logic: the player who made the last PASS in the 10 seconds before a GOAL
    is credited as the assister.

    goal_zones: list of two polygons, each a list of (x, y) tuples.
                  zones[0] = defending team 1's goal  → team 0 scores
                  zones[1] = defending team 0's goal  → team 1 scores
    possession_px: max pixel distance from ball centre to count as possessing the ball
    goal_cooldown_frames: minimum processed frames between registered goals (debounce)
    """

    def __init__(
        self,
        goal_zones: list[list[tuple[int, int]]] | None = None,
        possession_px: int = 100,
        goal_cooldown_frames: int = 90,
    ):
        self.goal_zones = goal_zones or []
        self.possession_px = possession_px
        self.goal_cooldown_frames = goal_cooldown_frames

        # Possession state
        self.current_possessor: int | None = None
        self.current_possessor_team: int = -1

        # Assist tracking
        self.last_pass_from: int | None = None
        self.last_pass_frame: int = 0

        # Goal debounce
        self._frames_since_goal = goal_cooldown_frames  # start ready

        self.events: list[dict] = []

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        frame_num: int,
        fps: float,
        player_info: list[tuple[int, np.ndarray, int]],
        ball_pos: np.ndarray | None,
    ) -> list[dict]:
        """
        player_info: [(tracker_id, center_xy, team), ...]
        Returns list of new events detected this frame.
        """
        self._frames_since_goal += 1
        new_events: list[dict] = []

        if ball_pos is None or len(player_info) == 0:
            return new_events

        # ---- Possession: nearest player within threshold ----
        min_dist = float("inf")
        nearest: tuple[int, int] | None = None  # (tracker_id, team)

        for pid, pos, team in player_info:
            d = float(np.linalg.norm(pos - ball_pos))
            if d < min_dist:
                min_dist = d
                nearest = (pid, team)

        if nearest and min_dist < self.possession_px:
            new_pid, new_team = nearest

            if new_pid != self.current_possessor:
                prev_pid = self.current_possessor
                prev_team = self.current_possessor_team

                if prev_pid is not None:
                    if prev_team == new_team:
                        # Same team → PASS
                        event = self._make_event("PASS", frame_num, fps,
                                                 from_player=prev_pid,
                                                 to_player=new_pid,
                                                 team=new_team)
                        new_events.append(event)
                        self.last_pass_from = prev_pid
                        self.last_pass_frame = frame_num

                    else:
                        # Opposing team gained ball → DEFENSIVE_ACTION
                        event = self._make_event("DEFENSIVE_ACTION", frame_num, fps,
                                                 player_id=new_pid,
                                                 team=new_team,
                                                 from_team=prev_team)
                        new_events.append(event)
                        # Interception breaks the assist chain
                        self.last_pass_from = None

                self.current_possessor = new_pid
                self.current_possessor_team = new_team

        # ---- Goal detection ----
        if self._frames_since_goal >= self.goal_cooldown_frames:
            for zone_idx, zone in enumerate(self.goal_zones):
                if _point_in_polygon(ball_pos, zone):
                    scoring_team = 1 - zone_idx  # zone 0 = team 0 scores, zone 1 = team 1 scores

                    # Assist: last same-team pass within the last 10 seconds
                    assister = None
                    time_since_pass = (frame_num - self.last_pass_frame) / fps
                    if (self.last_pass_from is not None
                            and self.current_possessor_team == scoring_team
                            and time_since_pass < 10.0):
                        assister = self.last_pass_from

                    event = self._make_event("GOAL", frame_num, fps,
                                             scorer=self.current_possessor,
                                             assister=assister,
                                             scoring_team=scoring_team,
                                             goal_zone=zone_idx)
                    new_events.append(event)
                    self._frames_since_goal = 0
                    self.last_pass_from = None
                    break

        self.events.extend(new_events)
        return new_events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_event(event_type: str, frame: int, fps: float, **kwargs) -> dict:
        return {"type": event_type, "frame": frame, "time_sec": frame / fps, **kwargs}


def _point_in_polygon(point: np.ndarray, polygon: list[tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test (no external dependencies)."""
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside

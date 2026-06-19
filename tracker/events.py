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
        goal_cooldown_frames: int = 300,
        possession_confirm_frames: int = 3,
        possession_min_separation: float = 50.0,
        contest_margin: float = 45.0,
        turnover_confirm_frames: int = 10,
    ):
        self.goal_zones = goal_zones or []
        self.possession_px = possession_px
        self.goal_cooldown_frames = goal_cooldown_frames
        # A new player must be nearest the ball for this many processed frames
        # before we count the possession change. Stops one-frame ball jitter from
        # firing phantom passes / defensive contributions.
        self.possession_confirm_frames = max(1, possession_confirm_frames)
        # A turnover (ball won by the OTHER team) must be sustained, not a brief
        # touch as the ball passes an opponent — so it needs a longer hold than a
        # same-team pass. This keeps quick passing from registering as turnovers.
        self.turnover_confirm_frames = max(1, turnover_confirm_frames)
        # A real pass/turnover moves the ball to a player in a DIFFERENT spot. If
        # the new possessor is essentially where the old one was, it's an id
        # switch of the same physical player, not a pass — so don't count it.
        self.possession_min_separation = possession_min_separation
        # The ball is only "possessed" if ONE player is clearly closest. If a
        # second player is within this margin too, it's a contested scrum — we
        # don't reassign possession (stops the possessor flickering between teams
        # in a cluster and inventing turnovers).
        self.contest_margin = contest_margin

        # Possession state
        self.current_possessor: int | None = None
        self.current_possessor_team: int = -1
        self.current_possessor_pos: np.ndarray | None = None

        # Debounce: a candidate possessor must hold for N frames to be confirmed
        self._candidate: int | None = None
        self._candidate_team: int = -1
        self._candidate_count: int = 0

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

        # ---- Possession: the player CLEARLY closest to the ball ----
        d1 = d2 = float("inf")
        nearest: tuple[int, int] | None = None  # (tracker_id, team)
        nearest_pos: np.ndarray | None = None

        for pid, pos, team in player_info:
            d = float(np.linalg.norm(pos - ball_pos))
            if d < d1:
                d2 = d1
                d1 = d
                nearest = (pid, team)
                nearest_pos = pos
            elif d < d2:
                d2 = d

        # Possession only counts if one player is clearly on the ball; if a
        # second player is nearly as close, it's contested — leave possession
        # unchanged so a scrum doesn't generate phantom passes/turnovers.
        clearly_possessed = (
            nearest is not None
            and d1 < self.possession_px
            and (d2 - d1) > self.contest_margin
        )

        if clearly_possessed:
            new_pid, new_team = nearest

            if new_pid == self.current_possessor:
                # Still the confirmed holder — no pending change.
                self._candidate = None
                self._candidate_count = 0
            else:
                # A different player is nearest: build confidence before acting.
                if new_pid == self._candidate:
                    self._candidate_count += 1
                else:
                    self._candidate = new_pid
                    self._candidate_team = new_team
                    self._candidate_count = 1

                # Same-team change confirms quickly (a pass); a change to the
                # opposing team must be held longer (a real turnover, not the ball
                # momentarily passing an opponent).
                is_turnover = (
                    self.current_possessor_team in (0, 1)
                    and self._candidate_team in (0, 1)
                    and self.current_possessor_team != self._candidate_team
                )
                needed = self.turnover_confirm_frames if is_turnover else self.possession_confirm_frames

                if self._candidate_count >= needed:
                    prev_pid = self.current_possessor
                    prev_team = self.current_possessor_team

                    # Did the ball actually change location? If the new possessor
                    # is basically where the old one stood, it's an id switch of
                    # the same player, not a pass/turnover — update silently.
                    moved = (
                        self.current_possessor_pos is None or nearest_pos is None or
                        float(np.linalg.norm(nearest_pos - self.current_possessor_pos))
                        >= self.possession_min_separation
                    )

                    # Only judge pass-vs-turnover when BOTH teams are known —
                    # acting on an uncertain label is what invents phantom events.
                    teams_known = prev_team in (0, 1) and new_team in (0, 1)

                    if prev_pid is not None and moved and teams_known:
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
                                                     from_team=prev_team,
                                                     lost_by=prev_pid)
                            new_events.append(event)
                            # Interception breaks the assist chain
                            self.last_pass_from = None

                    self.current_possessor = new_pid
                    self.current_possessor_team = new_team
                    self.current_possessor_pos = nearest_pos
                    self._candidate = None
                    self._candidate_count = 0
        else:
            # Ball loose (no one within range): drop any pending candidate but
            # keep the last confirmed possessor.
            self._candidate = None
            self._candidate_count = 0

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

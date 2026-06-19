"""
Replay a saved tracking log and re-aggregate stats, applying manual corrections
(track merges + team reassignments) made in correct.py.

This reuses the live EventDetector + StatsAggregator so that, with no corrections,
the re-aggregated stats match the original run.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict

import numpy as np

from .events import EventDetector
from .stats import StatsAggregator


class UnionFind:
    """Group raw track ids that the user merged into one player."""
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:        # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_meta(meta_path: str) -> dict:
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def iter_frames(tracks_path: str):
    with open(tracks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def reaggregate(meta: dict, tracks_path: str,
                merges: list[list[int]] | None = None,
                team_overrides: dict[int, int] | None = None,
                swap_teams: bool = False):
    """Replay the log applying merges + team overrides.

    swap_teams globally flips team 0<->1 for every player that has NOT been
    manually overridden (fixes the case where the automatic classifier assigned
    the two clusters to the wrong colours). Manual overrides are left as set.

    Returns (stats, display_map) where display_map maps canonical id -> #N.
    """
    merges = merges or []
    raw_overrides = {int(k): int(v) for k, v in (team_overrides or {}).items()}

    uf = UnionFind()
    for a, b in merges:
        uf.union(int(a), int(b))

    # Map overrides to the canonical id so clicking any merged member works.
    team_overrides = {uf.find(k): v for k, v in raw_overrides.items()}

    final_teams = {int(k): int(v) for k, v in meta.get("final_teams", {}).items()}

    # Group size + a stable team per merged group (majority of members' final teams)
    group_size: Counter = Counter(uf.find(rid) for rid in final_teams)
    group_votes: dict[int, Counter] = defaultdict(Counter)
    for rid, t in final_teams.items():
        group_votes[uf.find(rid)][t] += 1
    # Prefer a KNOWN team for the group: if any fragment was confidently classified
    # (0/1), the whole merged player inherits the majority of those — an unknown
    # fragment shouldn't drag a player to "unknown" when a sibling fragment knows it.
    def _group_team(v: Counter) -> int:
        known = {t: n for t, n in v.items() if t in (0, 1)}
        return max(known, key=known.get) if known else v.most_common(1)[0][0]
    canonical_team = {c: _group_team(v) for c, v in group_votes.items()}

    def team_for(canon: int, frame_team: int) -> int:
        if canon in team_overrides:
            return team_overrides[canon]        # manual fix — never swapped
        base = canonical_team.get(canon, frame_team) if group_size.get(canon, 1) > 1 else frame_team
        if swap_teams and base in (0, 1):
            base = 1 - base
        return base

    event_det = EventDetector(
        goal_zones=[[tuple(p) for p in zone] for zone in meta.get("goal_zones", [])],
        possession_px=meta["possession_px"],
        goal_cooldown_frames=meta["goal_cooldown_frames"],
        possession_confirm_frames=meta["possession_confirm_frames"],
        possession_min_separation=meta.get("possession_min_separation", 50.0),
        contest_margin=meta.get("contest_margin", 45.0),
        turnover_confirm_frames=meta.get("turnover_confirm_frames", 10),
    )
    stats = StatsAggregator()
    fps = meta["fps"]

    display_map: dict[int, int] = {}
    for rec in iter_frames(tracks_path):
        ball = rec["ball"]
        ball_pos = np.array(ball, dtype=float) if ball is not None else None
        player_info = []
        for raw_id, x1, y1, x2, y2, team in rec["players"]:
            canon = uf.find(int(raw_id))
            if canon not in display_map:
                display_map[canon] = len(display_map) + 1
            cx, cy = (x1 + x2) / 2.0, float(y2)   # feet — ball is at the feet
            player_info.append((canon, np.array([cx, cy]), team_for(canon, int(team))))
        for ev in event_det.update(rec["f"], fps, player_info, ball_pos):
            stats.process_event(ev)
        stats.tick_possession(event_det.current_possessor, event_det.current_possessor_team)

    return stats, display_map, event_det.events

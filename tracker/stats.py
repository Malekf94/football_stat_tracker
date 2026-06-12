from __future__ import annotations
from collections import defaultdict
import pandas as pd


def _blank_player():
    return {
        "team": -1,
        "goals": 0,
        "assists": 0,
        "passes_completed": 0,
        "passes_attempted": 0,
        "defensive_contributions": 0,
        "possession_frames": 0,
    }


class StatsAggregator:
    """Tallies per-player and per-team stats from detected events."""

    def __init__(self):
        self.players: dict = defaultdict(_blank_player)
        self.team_goals = {0: 0, 1: 0}

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def process_event(self, event: dict) -> None:
        t = event["type"]

        if t == "GOAL":
            scorer = event.get("scorer")
            assister = event.get("assister")
            team = event.get("scoring_team", -1)
            if scorer is not None:
                self.players[scorer]["goals"] += 1
                self.players[scorer]["team"] = team
            if assister is not None:
                self.players[assister]["assists"] += 1
            if team in self.team_goals:
                self.team_goals[team] += 1

        elif t == "PASS":
            from_p = event.get("from_player")
            to_p = event.get("to_player")
            team = event.get("team", -1)
            if from_p is not None:
                self.players[from_p]["passes_completed"] += 1
                self.players[from_p]["passes_attempted"] += 1
                self.players[from_p]["team"] = team
            if to_p is not None:
                self.players[to_p]["team"] = team

        elif t == "DEFENSIVE_ACTION":
            pid = event.get("player_id")
            team = event.get("team", -1)
            if pid is not None:
                self.players[pid]["defensive_contributions"] += 1
                self.players[pid]["team"] = team

    def tick_possession(self, tracker_id: int | None, team: int) -> None:
        """Call once per processed frame for the current ball-holder."""
        if tracker_id is not None:
            self.players[tracker_id]["possession_frames"] += 1
            self.players[tracker_id]["team"] = team

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_dataframe(self, fps: float, skip_frames: int) -> pd.DataFrame:
        rows = []
        for pid, s in self.players.items():
            pass_pct = (
                f"{100 * s['passes_completed'] / s['passes_attempted']:.0f}%"
                if s["passes_attempted"] > 0 else "—"
            )
            poss_sec = s["possession_frames"] * skip_frames / fps
            rows.append({
                "Player ID":               pid,
                "Team":                    s["team"] if s["team"] >= 0 else "?",
                "Goals":                   s["goals"],
                "Assists":                 s["assists"],
                "Passes (completed)":      s["passes_completed"],
                "Passes (attempted)":      s["passes_attempted"],
                "Pass %":                  pass_pct,
                "Defensive contributions": s["defensive_contributions"],
                "Possession (s)":          f"{poss_sec:.0f}",
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(["Team", "Goals"], ascending=[True, False])
        return df.reset_index(drop=True)

    def print_summary(self, fps: float, skip_frames: int) -> None:
        line = "=" * 70
        print(f"\n{line}")
        print("  MATCH STATS")
        print(line)
        print(f"  Team 0 goals: {self.team_goals[0]}   |   Team 1 goals: {self.team_goals[1]}")
        print(line)
        df = self.get_dataframe(fps, skip_frames)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  No stats collected (calibration may not have completed).")
        print(f"{line}\n")

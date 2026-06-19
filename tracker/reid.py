"""
Automatic re-identification by tracklet stitching.

A player whose tracking id keeps changing (lost behind others, leaves frame,
crosses an opponent) shows up as many short tracks. We can't tell same-team
players apart by appearance (identical bibs), so instead we stitch tracks by
SPATIO-TEMPORAL continuity: if track B starts shortly after track A ends, near
where A ended, it's almost certainly the same player resuming.

Returns a `merges` list (pairs of raw track ids) compatible with replay.reaggregate
and correct.py — so stitched tracks are treated as one player, and that player's
team becomes the majority vote over all their fragments.
"""
from __future__ import annotations
import bisect
from collections import Counter, defaultdict

from .replay import UnionFind, iter_frames


def _stab(cam_by_frame, frame, pos):
    """Map a feet position into the stabilised frame (or return it unchanged)."""
    if not cam_by_frame:
        return pos
    M = cam_by_frame.get(frame) or cam_by_frame.get(str(frame))
    if M is None:
        return pos
    a, b, tx, c, d, ty = M
    return (a * pos[0] + b * pos[1] + tx, c * pos[0] + d * pos[1] + ty)


def auto_merge(meta: dict, tracks_path: str,
               max_gap_s: float = 2.5,
               base_px: float = 70.0,
               speed_px_s: float = 350.0,
               same_spot_gap_s: float = 0.25,
               same_spot_px: float = 55.0,
               cam_by_frame: dict | None = None) -> list[list[int]]:
    """Stitch fragmented tracks of the same player. Returns merges [[a,b],...].

    If cam_by_frame (frame -> 2x3 affine) is given, positions are compared in the
    camera-stabilised frame so a player lost during a pan can still be linked.
    """
    fps = float(meta.get("fps", 30.0))

    # ---- gather per-track span, end/start position, team votes ----
    tr: dict[int, dict] = {}
    log_cam: dict = {}
    for rec in iter_frames(tracks_path):
        f = rec["f"]
        if "cam" in rec:
            log_cam[f] = rec["cam"]
        for rid, x1, y1, x2, y2, team in rec["players"]:
            feet = ((x1 + x2) / 2.0, float(y2))
            t = tr.get(rid)
            if t is None:
                tr[rid] = {"start": f, "end": f, "start_pos": feet,
                           "end_pos": feet, "teams": Counter()}
                t = tr[rid]
            t["end"] = f
            t["end_pos"] = feet
            if team in (0, 1):
                t["teams"][team] += 1

    # camera transforms: explicit arg wins, else use what's logged
    cam = cam_by_frame if cam_by_frame is not None else (log_cam or None)
    if cam is not None:
        max_gap_s = max(max_gap_s, 6.0)   # camera-stable positions allow wider gaps

    for t in tr.values():
        t["team"] = t["teams"].most_common(1)[0][0] if t["teams"] else -1
        # positions used for linking — stabilised for camera motion if available
        t["ep"] = _stab(cam, t["end"], t["end_pos"])
        t["sp"] = _stab(cam, t["start"], t["start_pos"])

    ids = list(tr)

    # Index tracks by start frame so each track only checks the few candidates
    # whose start falls in its short "continue" window (O(n·k), not O(n²)).
    by_start = sorted(ids, key=lambda i: tr[i]["start"])
    starts = [tr[i]["start"] for i in by_start]
    max_gap_frames = max_gap_s * fps

    # ---- candidate links: B can continue A ----
    cands = []
    for a in ids:
        ta = tr[a]
        fa_end = ta["end"]
        j = bisect.bisect_right(starts, fa_end)        # first B starting after A ends
        while j < len(by_start) and starts[j] <= fa_end + max_gap_frames:
            b = by_start[j]
            j += 1
            if b == a:
                continue
            tb = tr[b]
            gap_s = (tb["start"] - fa_end) / fps
            ax, ay = ta["ep"]
            bx, by = tb["sp"]
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            # how far a player could plausibly move during the gap
            if dist > base_px + speed_px_s * gap_s:
                continue
            # Teams must agree when both are known — UNLESS it's an obvious
            # in-place id switch (tiny gap + tiny move), which is the same player
            # even if one fragment was mislabelled.
            in_place = gap_s <= same_spot_gap_s and dist <= same_spot_px
            if (not in_place and ta["team"] in (0, 1) and tb["team"] in (0, 1)
                    and ta["team"] != tb["team"]):
                continue
            score = dist + 250.0 * gap_s
            cands.append((score, a, b))

    cands.sort(key=lambda c: c[0])

    # ---- greedy chaining: each track gets at most one successor/predecessor ----
    has_succ: set[int] = set()
    has_pred: set[int] = set()
    uf = UnionFind()
    for _, a, b in cands:
        if a in has_succ or b in has_pred:
            continue
        if uf.find(a) == uf.find(b):     # already in same chain
            continue
        uf.union(a, b)
        has_succ.add(a)
        has_pred.add(b)

    # ---- emit merges (link every non-root member to its chain root) ----
    groups: dict[int, list[int]] = defaultdict(list)
    for rid in ids:
        groups[uf.find(rid)].append(rid)
    merges = [[root, m] for root, members in groups.items()
              for m in members if m != root]
    return merges

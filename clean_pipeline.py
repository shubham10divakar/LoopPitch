"""
Football chapter — Day 1–2 data pipeline.

Builds the pass-level dataset from StatsBomb Open Data:
1. ingest:   download events + 360 freeze-frames (cached as raw JSON)
2. clean:    keep open-play passes with a usable freeze-frame; log every drop
3. label:    y_success, y_shot10, y_xg10
4. features: state + action features (Section 5 of the design doc)
5. write:    passes.parquet, frames_long.parquet, drop_log.csv, qa_report.json

Usage:
    python clean_pipeline.py --out data            # all three tournaments
    python clean_pipeline.py --out data --limit 2  # quick smoke test

Data: StatsBomb Open Data (https://github.com/statsbomb/open-data).
Credit StatsBomb as the data source in any publication.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# competition_id, season_id — verify with sb.competitions() if anything changes
COMPETITIONS = {
    "euro2020": (55, 43),
    "wc2022": (43, 106),
    "euro2024": (55, 282),
}
SPLIT = {"euro2020": "train", "euro2024": "train", "wc2022": "test"}

SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
# Open-play pass types ("Recovery", "Interception") are KEPT on purpose.
DROP_OUTCOMES = {"Injury Clearance", "Unknown"}  # not genuine pass attempts / unlabeled
SHOT_WINDOW_S = 10.0
MIN_VISIBLE_PLAYERS = 6
GOAL = np.array([120.0, 40.0])
POST_L, POST_R = np.array([120.0, 36.0]), np.array([120.0, 44.0])


# ----------------------------------------------------------------------------- ingest
def fetch_json(url: str, path: Path, retries: int = 3):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            path.write_bytes(data)
            return json.loads(data)
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def load_match(raw: Path, comp: str, match_id: int):
    ev = fetch_json(f"{RAW_BASE}/events/{match_id}.json", raw / comp / "events" / f"{match_id}.json")
    ff = fetch_json(f"{RAW_BASE}/three-sixty/{match_id}.json", raw / comp / "three-sixty" / f"{match_id}.json")
    return ev, ff


# ----------------------------------------------------------------------------- helpers
def ts_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def in_bounds(p) -> bool:
    return p is not None and 0 <= p[0] <= 120 and 0 <= p[1] <= 80


def angle_to_goal(p: np.ndarray) -> float:
    """Opening angle (radians) between the two posts as seen from p."""
    a, b = POST_L - p, POST_R - p
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.arccos(np.clip(cos, -1, 1)))


def in_triangle(pts: np.ndarray, a, b, c) -> np.ndarray:
    def sign(p, q, r):
        return (p[:, 0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[:, 1] - r[1])
    d1, d2, d3 = sign(pts, a, b), sign(pts, b, c), sign(pts, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    t = np.clip(((pts - a) @ v) / (v @ v + 1e-9), 0, 1)
    proj = a + t[:, None] * v
    return np.linalg.norm(pts - proj, axis=1)


def score_states(events):
    """Return {event_id: (goals_for_team_A, goals_for_team_B)} BEFORE each event."""
    tally, out = Counter(), {}
    for e in events:  # events are ordered by 'index'
        out[e["id"]] = dict(tally)
        t = e["type"]["name"]
        if t == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            tally[e["team"]["id"]] += 1
        elif t == "Own Goal For":
            tally[e["team"]["id"]] += 1
    return out


# ----------------------------------------------------------------------------- per-match build
def build_match(events, frames, comp: str, match_id: int, drop: Counter, qa: Counter):
    frame_by_id = {f["event_uuid"]: f for f in frames}
    scores = score_states(events)
    team_ids = {e["team"]["id"] for e in events if "team" in e}

    shots = [e for e in events if e["type"]["name"] == "Shot"]
    shots_by_poss = {}
    for s in shots:
        shots_by_poss.setdefault((s["period"], s["possession"]), []).append(s)

    seen, rows, frame_rows = set(), [], []
    passes = [e for e in events if e["type"]["name"] == "Pass"]
    drop["0_all_passes"] += len(passes)

    for p in passes:
        pz = p["pass"]
        ptype = pz.get("type", {}).get("name")
        outcome = pz.get("outcome", {}).get("name")

        if ptype in SET_PIECES:
            drop["1_set_piece"] += 1; continue
        if p["period"] == 5:
            drop["2_shootout"] += 1; continue
        if outcome in DROP_OUTCOMES:
            drop["3_injury_or_unknown_outcome"] += 1; continue
        if p["id"] not in frame_by_id:
            drop["4_no_360_frame"] += 1; continue
        if not (in_bounds(p.get("location")) and in_bounds(pz.get("end_location"))):
            drop["5_out_of_bounds_xy"] += 1; continue
        if p["id"] in seen:
            drop["6_duplicate_id"] += 1; continue

        fr = frame_by_id[p["id"]]
        ff = [q for q in fr.get("freeze_frame", []) if in_bounds(q.get("location"))]
        if len(ff) < MIN_VISIBLE_PLAYERS:
            drop["7_too_few_visible"] += 1; continue
        seen.add(p["id"])

        ball = np.array(p["location"][:2], float)
        end = np.array(pz["end_location"][:2], float)

        # orientation QA: actor in frame should sit on the event location
        actor = [q for q in ff if q.get("actor")]
        if actor:
            d = float(np.linalg.norm(np.array(actor[0]["location"]) - ball))
            qa["actor_present"] += 1
            qa["actor_mismatch_gt2"] += int(d > 2.0)
        else:
            qa["actor_missing"] += 1

        locs = np.array([q["location"] for q in ff], float)
        is_tm = np.array([bool(q["teammate"]) for q in ff])
        is_actor = np.array([bool(q.get("actor")) for q in ff])
        is_gk = np.array([bool(q.get("keeper")) for q in ff])
        opp = locs[~is_tm]
        tm = locs[is_tm & ~is_actor]

        d_ball_opp = np.linalg.norm(opp - ball, axis=1) if len(opp) else np.array([])
        d_end_opp = np.linalg.norm(opp - end, axis=1) if len(opp) else np.array([])
        outfield_opp_x = np.sort(locs[~is_tm & ~is_gk][:, 0]) if (~is_tm & ~is_gk).any() else np.array([])

        va = fr.get("visible_area") or []
        try:
            vis_area = Polygon(np.array(va, float).reshape(-1, 2)).area if len(va) >= 6 else np.nan
        except Exception:  # noqa: BLE001
            vis_area = np.nan

        # labels
        y_success = int(outcome is None)
        t0 = ts_seconds(p["timestamp"])
        window = [
            s for s in shots_by_poss.get((p["period"], p["possession"]), [])
            if s["team"]["id"] == p["team"]["id"] and 0 <= ts_seconds(s["timestamp"]) - t0 <= SHOT_WINDOW_S
            and s["index"] > p["index"]
        ]
        y_shot10 = int(len(window) > 0)
        y_xg10 = float(sum(s["shot"].get("statsbomb_xg", 0.0) for s in window))

        sc = scores[p["id"]]
        own = sc.get(p["team"]["id"], 0)
        other = sum(v for k, v in sc.items() if k != p["team"]["id"])

        rows.append({
            "pass_id": p["id"], "match_id": match_id, "competition": comp, "split": SPLIT[comp],
            "period": p["period"], "minute": p["minute"], "second": p["second"],
            "team": p["team"]["name"], "player": p["player"]["name"], "possession": p["possession"],
            "play_pattern": p["play_pattern"]["name"], "pass_type": ptype or "Open Play",
            "body_part": pz.get("body_part", {}).get("name", "Unknown"),
            "pass_height": pz.get("height", {}).get("name", "Unknown"),
            "under_pressure": int(bool(p.get("under_pressure", False))),
            "score_diff": own - other,
            # state features
            "ball_x": ball[0], "ball_y": ball[1],
            "ball_dist_goal": float(np.linalg.norm(GOAL - ball)), "ball_angle_goal": angle_to_goal(ball),
            "opp_within_5": int((d_ball_opp < 5).sum()),
            "nearest_opp_dist": float(d_ball_opp.min()) if len(d_ball_opp) else np.nan,
            "opp_in_cone": int(in_triangle(opp, ball, POST_L, POST_R).sum()) if len(opp) else 0,
            "def_line_x": float(outfield_opp_x[-1]) if len(outfield_opp_x) else np.nan,
            "tm_ahead": int((tm[:, 0] > ball[0]).sum()) if len(tm) else 0,
            "n_visible_teammates": int(is_tm.sum()), "n_visible_opponents": int((~is_tm).sum()),
            "visible_area": vis_area,
            # action features
            "end_x": end[0], "end_y": end[1],
            "pass_len": float(np.linalg.norm(end - ball)),
            "pass_angle": float(math.atan2(end[1] - ball[1], end[0] - ball[0])),
            "progress_x": float(end[0] - ball[0]),
            "end_dist_goal": float(np.linalg.norm(GOAL - end)), "end_angle_goal": angle_to_goal(end),
            "opp_near_target_3": int((d_end_opp < 3).sum()),
            "lane_opp_2": int((dist_to_segment(opp, ball, end) < 2).sum()) if len(opp) else 0,
            "target_in_box": int(end[0] >= 102 and 18 <= end[1] <= 62),
            # labels
            "y_success": y_success, "y_shot10": y_shot10, "y_xg10": y_xg10,
        })
        for i, q in enumerate(ff):
            frame_rows.append({
                "pass_id": p["id"], "slot": i, "x": q["location"][0], "y": q["location"][1],
                "teammate": bool(q["teammate"]), "actor": bool(q.get("actor")), "keeper": bool(q.get("keeper")),
            })

    drop["8_kept"] += len(rows)
    qa["teams_seen"] = max(qa["teams_seen"], len(team_ids))
    return rows, frame_rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None, help="matches per competition (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    raw, proc = out / "raw", out / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    all_rows, all_frames, drop_log, qa = [], [], [], Counter()
    for comp, (cid, sid) in COMPETITIONS.items():
        matches = fetch_json(f"{RAW_BASE}/matches/{cid}/{sid}.json", raw / comp / "matches.json")
        matches = [m for m in matches if m.get("match_status_360") == "available"][: args.limit]
        drop = Counter()
        for i, m in enumerate(matches, 1):
            ev, ff = load_match(raw, comp, m["match_id"])
            rows, frows = build_match(ev, ff, comp, m["match_id"], drop, qa)
            all_rows += rows; all_frames += frows
            print(f"[{comp}] {i}/{len(matches)} match {m['match_id']}: {len(rows)} passes", flush=True)
        drop_log.append({"competition": comp, "matches": len(matches), **dict(sorted(drop.items()))})

    passes = pd.DataFrame(all_rows)
    frames = pd.DataFrame(all_frames)
    passes.to_parquet(proc / "passes.parquet", index=False)
    frames.to_parquet(proc / "frames_long.parquet", index=False)
    pd.DataFrame(drop_log).to_csv(proc / "drop_log.csv", index=False)

    comp_rows = passes[passes.y_success == 1]
    report = {
        "n_passes": len(passes),
        "n_frame_rows": len(frames),
        "pass_success_rate": round(passes.y_success.mean(), 4),
        "shot10_rate_completed": round(comp_rows.y_shot10.mean(), 4),
        "per_split": passes.groupby("split").agg(n=("pass_id", "size"),
                                                  success=("y_success", "mean")).round(4).to_dict(),
        "orientation_qa": dict(qa),
        "missing_values": {k: int(v) for k, v in passes.isna().sum().items() if v},
    }
    (proc / "qa_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

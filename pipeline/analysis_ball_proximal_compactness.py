"""
Idea #3 (deferred phase-detection-definition candidate): "ball-proximal
compactness replacing whole-team inter-line."

Literature basis (literature-and-data-foundations.md): Forcher et al. found
whole-team compactness barely predicts defensive success, but compactness of
the 5 defenders NEAREST THE BALL does. The current `inter_line` metric
(tracking_features.py) is whole-team span (deepest defender to most advanced
attacker) -- exactly the kind of measure that finding says is the weaker one.

This script computes a genuinely new per-frame metric (not already sitting in
match_features_with_pressure.pkl -- that only has the whole-team version) for
the FULL match, using the already-exported, already-smoothed full-match
player positions in chapters/anim_01.json..anim_26.json (built in round 17;
same positions already plotted on the pitch canvas), then compares it against
the existing inter_line metric.

Definition: ball-proximal compactness (BPC) for a team at a frame = the
maximum pairwise distance ("diameter") among that team's nearest
min(5, n_available) outfield (non-GK) players to the ball's current
position. Same unit (meters) and same "smaller = more compact" direction as
inter_line, so it's a like-for-like candidate replacement, not a rescaled
quantity that needs new thresholds invented from nothing.
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
from config import load_roster, HOME_TEAM_ID, AWAY_TEAM_ID, CHAPTERS_DIR, pkl_path

N_PROXIMAL = 5

roster, jersey_lookup, _ = load_roster()
home_pos = {j: pg for (t, j), (nick, pg) in jersey_lookup.items() if t == str(HOME_TEAM_ID)}
away_pos = {j: pg for (t, j), (nick, pg) in jersey_lookup.items() if t == str(AWAY_TEAM_ID)}


def outfield_positions(players, team_prefix, pos_lookup):
    """players: frame['players'] list of {'id': 'H4', 'x':.., 'y':..}. Returns
    list of (x,y) for outfield (non-GK) players of the given team, skipping
    anyone not in the roster lookup (shouldn't happen) or the keeper."""
    out = []
    for p in players:
        pid = p['id']
        if not pid.startswith(team_prefix):
            continue
        jersey = pid[1:]
        pg = pos_lookup.get(jersey)
        if pg is None or pg == 'GK':
            continue
        out.append((p['x'], p['y']))
    return out


def bpc(positions, bx, by):
    if bx is None or by is None or len(positions) < 2:
        return np.nan
    arr = np.array(positions)
    d_to_ball = np.hypot(arr[:, 0] - bx, arr[:, 1] - by)
    nearest = arr[np.argsort(d_to_ball)[:N_PROXIMAL]]
    if len(nearest) < 2:
        return np.nan
    # max pairwise distance ("diameter") among the ball-proximal group
    diffs = nearest[:, None, :] - nearest[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return float(dists.max())


# chapter count varies by match (match 1: 26 chapters over ~130.6 min; match 2
# will differ) -- discover it from whatever anim_NN.json files actually exist
# in this match's chapters dir, rather than assuming 26.
chapter_paths = sorted(glob.glob(os.path.join(CHAPTERS_DIR, 'anim_*.json')))
chapter_indices = sorted(
    int(re.search(r'anim_(\d+)\.json$', p).group(1)) for p in chapter_paths
)
if not chapter_indices:
    raise SystemExit(f"No anim_NN.json chapter files found in {CHAPTERS_DIR} -- "
                      f"run full_match_export.py for this match first.")

rows = []
for idx in chapter_indices:
    with open(f'{CHAPTERS_DIR}/anim_{idx:02d}.json') as f:
        d = json.load(f)
    fps = d['meta']['fps']
    for fr in d['frames']:
        frame_num = round(fr['t'] * fps)
        ball = fr.get('ball')
        bx, by = (ball['x'], ball['y']) if ball else (None, None)
        home_xy = outfield_positions(fr['players'], 'H', home_pos)
        away_xy = outfield_positions(fr['players'], 'A', away_pos)
        rows.append((frame_num, bpc(home_xy, bx, by), bpc(away_xy, bx, by)))
    print(f"chapter {idx:02d}: {len(d['frames'])} frames processed")

bpc_df = pd.DataFrame(rows, columns=['frameNum', 'home_bpc', 'away_bpc'])
bpc_df = bpc_df.drop_duplicates(subset='frameNum', keep='first')
out_path = pkl_path('ball_proximal_compactness')
bpc_df.to_pickle(out_path)
print(f"\nSaved {out_path} ({len(bpc_df)} frames)")
print(bpc_df[['home_bpc', 'away_bpc']].describe().round(2))

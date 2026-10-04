"""
Space-in-behind model, built on the DAS (Dangerous Accessible Space) framework
from Bischofberger & Baca (2026), via the official `accessible-space` PyPI
package -- which ships the same physics-based pass-completion simulation and
the same fitted distance+angle "danger" (value) model described in the paper,
already trained on public Metrica Sports tracking data. No re-fitting needed.

For every sampled frame in the same 5-minute tracking window used by the
Pressure Read animation:
  1. Build a tracking snapshot in accessible_space's schema (players + ball,
     velocities via finite difference at native frame rate, attacking
     direction from the match's known orientation -- not inferred, since we
     already have it).
  2. Run get_dangerous_accessible_space() for the whole pitch (context).
  3. Re-integrate that same simulation result restricted to the region behind
     the DEFENDING team's back line -- from their own goal line out to their
     single deepest outfield defender's actual x position (the same
     live, exact-extreme convention already used for the inter-line band in
     the animation, not a smoothed/mean value) -- giving DAS/AS specifically
     for "space in behind" at that instant.
  4. Tag the frame with whichever team is defending and classification.py's
     phase label for that team (High Press / Mid Block / Low Block /
     Defensive Transition), already present in the exported animation data.

Output: space_behind_line.json, consumed by the companion chart.
"""
from config import json_path
import json
import time
import pandas as pd
pd.set_option('future.infer_string', False)  # accessible-space's internals predate pandas 3's default string dtype
import numpy as np
import accessible_space as accsp

ANIM_PATH = json_path('animation_data')
SAMPLE_EVERY_S = 0.5  # ~600 samples over the 5-minute window

PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0


def team_of(pid):
    return 'home' if pid[0] == 'H' else 'away'


def build_velocity_lookup(frames):
    """vx, vy per (frame_index, player_id) via central difference on the
    already Kalman-smoothed positions, at the animation's native frame rate --
    same approach pressure.py already uses for the pressure metric."""
    n = len(frames)
    pos = [{p['id']: (p['x'], p['y']) for p in fr['players']} for fr in frames]
    vel = [dict() for _ in range(n)]
    for i in range(n):
        t0 = frames[max(i - 1, 0)]['clockS']
        t1 = frames[min(i + 1, n - 1)]['clockS']
        dt = t1 - t0
        if dt <= 0:
            continue
        prev_pos = pos[max(i - 1, 0)]
        next_pos = pos[min(i + 1, n - 1)]
        for pid, (x, y) in pos[i].items():
            if pid in prev_pos and pid in next_pos:
                px, py = prev_pos[pid]
                nx, ny = next_pos[pid]
                vel[i][pid] = ((nx - px) / dt, (ny - py) / dt)
            else:
                vel[i][pid] = (0.0, 0.0)
    return vel


def nearest_teammate_to_ball(frame, team):
    bx, by = frame['ball']['x'], frame['ball']['y']
    best, best_d = None, 1e9
    for p in frame['players']:
        if team_of(p['id']) != team:
            continue
        d = (p['x'] - bx) ** 2 + (p['y'] - by) ** 2
        if d < best_d:
            best_d, best = d, p['id']
    return best


def deepest_defender_x(frame, team, gk_ids, own_goal_x):
    cands = [p for p in frame['players'] if team_of(p['id']) == team and p['id'] not in gk_ids]
    if not cands:
        return None
    return min(cands, key=lambda p: abs(own_goal_x - p['x']))['x']


def frame_to_tracking_df(frame, vel_lookup, home_positive, frame_id):
    poss = frame['possessionTeam']
    if poss not in ('home', 'away') or frame.get('ball') is None or frame['ball'].get('x') is None:
        return None
    attack_dir = -1.0 if (poss == 'home') == home_positive else 1.0
    carrier = nearest_teammate_to_ball(frame, poss)
    rows = []
    for p in frame['players']:
        team = team_of(p['id'])
        vx, vy = vel_lookup.get(p['id'], (0.0, 0.0))
        rows.append({'frame_id': frame_id, 'player_id': p['id'], 'team_id': team,
                     'x': p['x'], 'y': p['y'], 'vx': vx, 'vy': vy,
                     'team_in_possession': poss, 'period_id': 1,
                     'attacking_direction': attack_dir, 'player_in_possession': carrier})
    rows.append({'frame_id': frame_id, 'player_id': 'ball', 'team_id': None,
                 'x': frame['ball']['x'], 'y': frame['ball']['y'], 'vx': 0.0, 'vy': 0.0,
                 'team_in_possession': poss, 'period_id': 1,
                 'attacking_direction': attack_dir, 'player_in_possession': carrier})
    return pd.DataFrame(rows)


def process_frame(frame, vel_at_frame, home_positive, gk_ids, frame_id):
    df = frame_to_tracking_df(frame, vel_at_frame, home_positive, frame_id)
    if df is None:
        return None

    defending = 'away' if frame['homeStructure'] == 'In Possession' else 'home'
    phase = frame['awayStructure'] if defending == 'away' else frame['homeStructure']
    if phase not in ('High Press', 'Mid Block', 'Low Block', 'Defensive Transition'):
        return None

    own_goal_x = 52.5 if (defending == 'home') == home_positive else -52.5
    dmin_x = deepest_defender_x(frame, defending, gk_ids, own_goal_x)
    if dmin_x is None:
        return None
    x_lo, x_hi = sorted([own_goal_x, dmin_x])
    region_width_m = x_hi - x_lo

    ret = accsp.get_dangerous_accessible_space(
        df, frame_col='frame_id', player_col='player_id', team_col='team_id', ball_player_id='ball',
        x_col='x', y_col='y', vx_col='vx', vy_col='vy', team_in_possession_col='team_in_possession',
        period_col='period_id', attacking_direction_col='attacking_direction', infer_attacking_direction=False,
        player_in_possession_col='player_in_possession', use_progress_bar=False,
        x_pitch_min=PITCH_X_MIN, x_pitch_max=PITCH_X_MAX, y_pitch_min=PITCH_Y_MIN, y_pitch_max=PITCH_Y_MAX,
    )
    as_whole = float(ret.acc_space.iloc[0])
    das_whole = float(ret.das.iloc[0])

    areas_as_behind = accsp.integrate_surfaces(ret.simulation_result, x_lo, x_hi, PITCH_Y_MIN, PITCH_Y_MAX)
    areas_das_behind = accsp.integrate_surfaces(ret.dangerous_result, x_lo, x_hi, PITCH_Y_MIN, PITCH_Y_MAX)
    as_behind = float(areas_as_behind.attack_poss[0])
    das_behind = float(areas_das_behind.attack_poss[0])

    return {
        'clockS': frame['clockS'],
        'defending_team': defending,
        'phase': phase,
        'possession_team': frame['possessionTeam'],
        'backline_x': dmin_x,
        'region_width_m': region_width_m,
        'region_area_m2': region_width_m * (PITCH_Y_MAX - PITCH_Y_MIN),
        'AS_whole': as_whole,
        'DAS_whole': das_whole,
        'AS_behind': as_behind,
        'DAS_behind': das_behind,
    }


def main():
    with open(ANIM_PATH) as f:
        data = json.load(f)
    meta = data['meta']
    frames = data['frames']
    home_positive = meta['homeDefendsPositiveX']
    gk_ids = {pid for pid, info in meta['players'].items() if info['pos'] == 'GK'}
    fps = meta['fps']

    print(f"Building velocity lookup over {len(frames)} frames...")
    vel_full = build_velocity_lookup(frames)

    step = max(1, round(SAMPLE_EVERY_S * fps))
    sample_idxs = list(range(0, len(frames), step))
    print(f"Sampling every {step} frames (~{SAMPLE_EVERY_S}s) -> {len(sample_idxs)} candidate samples")

    results = []
    t0 = time.time()
    for n, i in enumerate(sample_idxs):
        r = process_frame(frames[i], vel_full[i], home_positive, gk_ids, frame_id=i)
        if r is not None:
            results.append(r)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(sample_idxs)} samples, {time.time()-t0:.1f}s elapsed")
    print(f"Done: {len(results)} usable samples in {time.time()-t0:.1f}s "
          f"(skipped {len(sample_idxs)-len(results)} loose-ball/edge frames)")

    df = pd.DataFrame(results)
    print(df.groupby(['defending_team', 'phase'])[['DAS_behind', 'AS_behind', 'region_width_m']].mean().round(4))

    out = {
        'window': {'startMin': meta['windowStartMin'], 'endMin': meta['windowEndMin'], 'sampleEveryS': SAMPLE_EVERY_S},
        'teams': {'home': meta['homeTeam'], 'away': meta['awayTeam']},
        'samples': results,
    }
    with open(json_path('space_behind_line'), 'w') as f:
        json.dump(out, f)
    print("Wrote space_behind_line.json")


if __name__ == '__main__':
    main()

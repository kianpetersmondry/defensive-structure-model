"""
Export a per-frame JSON payload for one clean, phase-diverse match segment,
for the interactive animation artifact.

WINDOW SELECTION (v2): the original window (period 2, video-relative
53.5-58.5 min) was picked mainly for phase diversity and looked reasonable
on paper (68% ball coverage), but the user correctly reported the ball still
"completely disappears" too often after the Kalman/RTS smoothing fix. That
smoother can only clean noise -- it can't recover ball positions the raw
tracking never captured, and that window turned out to have 6 real gaps of
9-23s (about 32% of its 300s span with literally no ball position in the
raw data).

WINDOW SELECTION (v3): the first replacement window (period 1, frames
64579-73569) was picked from a full-match scan that measured the worst gap
*between* consecutive raw ball samples inside each candidate window -- but
that scan had a blind spot: it couldn't see a gap that runs off either edge
of the window itself, because there's no earlier/later sample inside the
window to diff against. That window happened to start mid-dropout: the raw
ball data has no position at all for the first 19.1s after WINDOW_LO_FRAME
(a real gap that straddled the window boundary), so the rendered animation
opened on a 19.1s ball disappearance even though the internal-gap scan
reported a clean 7.4s max. Re-running the scan with edge-inclusive gap
measurement (treating "no sample before the window's own start/end" as part
of the gap, exactly like a mid-window dropout) surfaced a real improvement:
period 1, frame range 21783-30758 (periodElapsedTime ~11:27-16:26 of the
first half) -- 86.9% ball coverage, longest single gap 8.5s *including both
edges*, and still gets all five defensive-structure labels for the home
team (In Possession, Defensive Transition, High Press, Mid Block, Low
Block) and four of five for the away team (all but Low Block). No 5-minute
window in the whole match had zero gaps over ~8s -- broadcast ball tracking
in this match just has a baseline rate of multi-second dropouts -- so this
is close to the best available, not a perfect window; the Kalman/RTS
smoothing still does real work on top of it for the noise that remains.

The window is specified as an explicit frame range (WINDOW_LO_FRAME /
WINDOW_HI_FRAME) rather than start/end minutes multiplied by fps, precisely
because frameNum (video-relative) and periodElapsedTime (true broadcast
match clock) are NOT the same axis in this dataset -- an earlier version of
this pipeline had a real bug from conflating them (see
literature-and-data-foundations project notes). Using frame bounds found
directly from the raw file sidesteps that whole class of bug; windowStartMin
/ windowEndMin in the exported metadata are derived from the actual
periodElapsedTime of the boundary frames, purely for display.

RAW-TRACKING QUALITY HANDLING (added after the first published cut showed the
ball "teleporting"/gliding oddly): the raw PFF broadcast tracking has two
distinct real defects, confirmed by direct measurement against this match's
tracking.jsonl.bz2, not assumed:

  1. Single-frame ball detection spikes -- ~176 occurrences match-wide (~1 per
     33s of ball tracking), where one frame reports the ball tens of meters
     from where it was a frame earlier (and a frame later), i.e. an implied
     speed of hundreds of m/s. This is sensor/detector noise (a misdetection
     against an advertising board, a player's boot, etc.), not a real ball
     position. `reject_speed_spikes` drops any sample whose neighbor-in and
     neighbor-out speeds both exceed a plausible max AND which is not itself
     part of a real fast passage (checked by seeing that skipping the sample
     entirely gives a plausible speed between its two neighbors).
  2. Genuine multi-second ball tracking gaps -- broadcast cameras lose the
     ball off-screen regularly; in just this one 5-minute window there are 6
     gaps up to 23.2s long (out of ~176 total in the raw ball samples for the
     window, most much shorter). The OLD version of this script bridged every
     gap with a straight-line linear interpolation regardless of length, so a
     23-second real gap rendered as the ball smoothly, unrealistically
     gliding across a huge span of pitch in one dead-straight line -- the
     other half of what looked like "teleporting". `interpolate_series` now
     takes a `max_gap_frames` cutoff: gaps at or under it are still linearly
     interpolated (normal brief occlusion/detector miss), gaps beyond it are
     left un-filled, so the ball/player legitimately disappears from the
     animation and the JS's existing "ball position unavailable" fallback
     (already written for the null-possession case) handles it correctly
     with no further JS changes needed.

Players did not show this problem in this match (checked match-wide: zero
tracking gaps over 2s, and only one single-frame speed spike in all 235,670
frames), but the same two-stage cleaning is applied to player series too --
broadcast cameras can and do lose off-ball players for real stretches, and
this makes the pipeline safe to point at other matches/windows where that
does happen, rather than something that happens to work here by luck.

SMOOTHING (v2): the two fixes above removed the worst offenders (gross
single-frame spikes, and long fabricated glides across real gaps), but
measuring the CLEANED ball trajectory's frame-to-frame acceleration still
showed pervasive small-scale noise throughout -- 41% of frames implied
>50 m/s^2, 23% >100 m/s^2, far beyond anything a real ball does -- i.e.
broadcast single-camera ball detection carries continuous positional noise
(tens of cm per frame), not just occasional gross misses. A first attempt
at fixing this with an ad hoc stack (Savitzky-Golay smoothing + two
different spike-masking heuristics layered on top) helped but was still
heuristic and didn't fully resolve short (~0.3-0.4s) sustained wobbles.
Replaced with `kalman_rts_smooth`: a constant-velocity Kalman filter with
per-step Mahalanobis-distance gating (an implausible observation is skipped
in favor of the filter's own prediction, handling both single bad frames
and short bad runs the same way) followed by an RTS backward pass, which is
the standard technique for exactly this problem -- see that function's
docstring for the full reasoning.
"""
from config import json_path
import bz2
import json
import math
import pandas as pd
from config import (TRACKING_PATH, HOME_TEAM_ID, AWAY_TEAM_ID, load_roster,
                     load_metadata, orientation_lookup)

WINDOW_LO_FRAME = 21783
WINDOW_HI_FRAME = 30758
WINDOW_PERIOD = 1
DOWNSAMPLE = 1  # 1 = native ~30fps, 2 = ~15fps, etc.

BALL_MAX_SPEED_MS = 35.0     # ~126 km/h -- generous ceiling for a struck ball
PLAYER_MAX_SPEED_MS = 10.5   # above elite-sprint territory
BALL_MAX_GAP_S = 1.5         # longer real gaps are left untracked, not glided
PLAYER_MAX_GAP_S = 2.0
FPS_NOMINAL = 29.97

# Kalman/RTS tuning -- process_noise_std is "how much real acceleration (m/s^2)
# to allow between samples" (higher = tracks sharp real direction changes
# better but smooths less); measurement_noise_std is "assumed per-frame
# detector noise (m)". Ball values are looser on both since its raw signal is
# both noisier AND capable of much sharper real accelerations (a struck ball)
# than a player ever produces.
BALL_PROCESS_NOISE_STD = 12.0
BALL_MEASUREMENT_NOISE_STD = 1.0
PLAYER_PROCESS_NOISE_STD = 4.0
PLAYER_MEASUREMENT_NOISE_STD = 0.3


def reject_speed_spikes(samples, max_speed_ms, fps=FPS_NOMINAL):
    """samples: sorted list of (frame, val_tuple) where val_tuple[0:2] is (x, y)
    (a ball sample's val_tuple may also carry z -- untouched either way). Drops
    single-sample detector spikes: a point whose speed in AND out both exceed
    max_speed_ms, but whose two neighbors are themselves a plausible speed
    apart (i.e. removing just this one point resolves the anomaly, rather
    than it being real fast play). Isolated spikes only -- confirmed
    empirically to be how these occur in this dataset (a single bad frame,
    not a run of them)."""
    if len(samples) < 3:
        return samples
    keep = [True] * len(samples)
    for i in range(1, len(samples) - 1):
        f0, v0 = samples[i - 1]
        f1, v1 = samples[i]
        f2, v2 = samples[i + 1]
        x0, y0 = v0[0], v0[1]
        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]
        dt_in, dt_out, dt_skip = f1 - f0, f2 - f1, f2 - f0
        if dt_in <= 0 or dt_out <= 0 or dt_skip <= 0:
            continue
        speed_in = math.hypot(x1 - x0, y1 - y0) / (dt_in / fps)
        speed_out = math.hypot(x2 - x1, y2 - y1) / (dt_out / fps)
        speed_skip = math.hypot(x2 - x0, y2 - y0) / (dt_skip / fps)
        if speed_in > max_speed_ms and speed_out > max_speed_ms and speed_skip <= max_speed_ms:
            keep[i] = False
    return [s for s, k in zip(samples, keep) if k]


def reject_speed_spikes_iterative(samples, max_speed_ms, fps=FPS_NOMINAL, max_passes=5):
    """Runs reject_speed_spikes repeatedly: a short run of >1 consecutive bad
    samples isn't resolved by a single pass (removing one point still leaves
    its neighbor looking anomalous against the next real point), but each
    pass peels off at least the clearest single-point spikes in the run, so a
    handful of passes converges on a clean series. Stops as soon as a pass
    removes nothing."""
    for _ in range(max_passes):
        cleaned = reject_speed_spikes(samples, max_speed_ms, fps)
        if len(cleaned) == len(samples):
            return cleaned
        samples = cleaned
    return samples


def interpolate_series(samples, max_gap_frames=None):
    """samples: sorted list of (frame, val_tuple). Returns frame->val_tuple via
    linear interpolation between the two nearest known samples. Gaps longer
    than max_gap_frames are left un-filled (no entry for those frames) rather
    than bridged -- a real, extended tracking dropout should render as
    "not tracked", not as a fabricated straight-line glide. max_gap_frames=None
    keeps the old hold/bridge-everything behavior."""
    if not samples:
        return {}
    frames = [s[0] for s in samples]
    out = {}
    lo_f, hi_f = frames[0], frames[-1]
    import bisect as _bisect
    for f in range(lo_f, hi_f + 1):
        idx = _bisect.bisect_right(frames, f) - 1
        idx = max(0, min(idx, len(samples) - 2)) if len(samples) > 1 else 0
        f0, v0 = samples[idx]
        f1, v1 = samples[min(idx + 1, len(samples) - 1)]
        if max_gap_frames is not None and (f1 - f0) > max_gap_frames:
            continue
        frac = 0.0 if f1 == f0 else max(0.0, min(1.0, (f - f0) / (f1 - f0)))
        out[f] = tuple(v0[i] + (v1[i] - v0[i]) * frac for i in range(len(v0)))
    return out


def kalman_rts_smooth(samples, fps=FPS_NOMINAL, process_noise_std=8.0,
                       measurement_noise_std=1.0, gate_sigma=4.5):
    """samples: sorted list of (frame, val_tuple), val_tuple[0:2] = (x, y);
    any extra dims (ball's z) pass through unchanged. Returns a new list of
    (frame, val_tuple) at the SAME frames, with x,y replaced by the RTS-
    smoothed estimate.

    Replaces the earlier ad hoc stack (Savitzky-Golay smoothing + two
    separate spike-masking passes) with the standard technique for this
    exact problem: noisy, irregularly-timed position observations of
    something that moves by physics (roughly constant velocity between
    real accelerations), where you want both a smooth position AND a
    principled way to down-weight an implausible observation instead of
    hand-writing "isolated point" / "sustained wobble" special cases.

    Model: constant-velocity motion, state [x, y, vx, vy], propagated with
    the true elapsed time between successive raw samples (so it's correct
    across the small natural gaps that already exist between raw samples,
    not just a fixed 1/fps step). Forward pass is a standard Kalman filter;
    each step gates its own measurement by Mahalanobis distance against the
    filter's *current* belief -- a step whose observation is implausible
    given recent real motion has its measurement skipped for that step
    (the filter coasts on its own prediction instead), which handles both
    a single bad frame AND a short run of bad frames (e.g. the ~2-2.5m
    per-frame back-and-forth wobble found in this match around t=3496s)
    without needing separate isolated-point vs. sustained-run logic --
    each frame is judged against the filter's live estimate, not a fixed
    neighbor pattern. The backward RTS pass then uses the *whole* trajectory
    (not just past frames) to produce the final smoothed estimate, so it
    doesn't lag behind real direction changes the way a causal-only filter
    or a moving average would.

    process_noise_std: how much real acceleration (m/s^2, roughly) the
    ball/player is allowed between samples -- higher lets the filter track
    sharper real direction changes (a kick) but smooths less; lower smooths
    harder but can lag a genuine fast change. measurement_noise_std: assumed
    per-frame detector noise (m) -- tuned per key (ball vs. player) since
    they measured very differently in this match's raw data."""
    import numpy as np

    n = len(samples)
    if n < 2:
        return samples
    frames = [s[0] for s in samples]
    z = np.array([[s[1][0], s[1][1]] for s in samples])  # (n, 2) observations

    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R = (measurement_noise_std ** 2) * np.eye(2)
    qc = process_noise_std ** 2
    gate_d2 = gate_sigma ** 2  # squared-Mahalanobis gate (2 dof)

    def F_of(dt):
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    def Q_of(dt):
        return qc * np.array([
            [dt**4/4, 0,        dt**3/2, 0],
            [0,        dt**4/4, 0,        dt**3/2],
            [dt**3/2,  0,        dt**2,    0],
            [0,        dt**3/2,  0,        dt**2],
        ])

    x_filt = [None] * n
    P_filt = [None] * n
    x_pred = [None] * n
    P_pred = [None] * n
    F_used = [None] * n

    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.diag([measurement_noise_std**2, measurement_noise_std**2, 25.0, 25.0])
    x_filt[0], P_filt[0], x_pred[0], P_pred[0] = x, P, x, P

    for k in range(1, n):
        dt = max((frames[k] - frames[k - 1]) / fps, 1e-3)
        F = F_of(dt)
        Q = Q_of(dt)
        xp = F @ x
        Pp = F @ P @ F.T + Q
        x_pred[k], P_pred[k], F_used[k] = xp, Pp, F

        innovation = z[k] - H @ xp
        S = H @ Pp @ H.T + R
        try:
            d2 = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            d2 = 0.0
        if d2 > gate_d2:
            x, P = xp, Pp  # implausible observation -- coast on the prediction
        else:
            K = Pp @ H.T @ np.linalg.inv(S)
            x = xp + K @ innovation
            P = (np.eye(4) - K @ H) @ Pp
        x_filt[k], P_filt[k] = x, P

    x_smooth = [None] * n
    P_smooth = [None] * n
    x_smooth[-1], P_smooth[-1] = x_filt[-1], P_filt[-1]
    for k in range(n - 2, -1, -1):
        Pp_next = P_pred[k + 1]
        try:
            C = P_filt[k] @ F_used[k + 1].T @ np.linalg.inv(Pp_next)
        except np.linalg.LinAlgError:
            x_smooth[k], P_smooth[k] = x_filt[k], P_filt[k]
            continue
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - Pp_next) @ C.T

    out = []
    for k in range(n):
        rest = samples[k][1][2:]
        out.append((frames[k], (float(x_smooth[k][0]), float(x_smooth[k][1])) + rest))
    return out


def export_window():
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    fps = meta['fps']

    lo_frame = WINDOW_LO_FRAME
    hi_frame = WINDOW_HI_FRAME

    id_to_jersey = {str(r['player']['id']): r['shirtNumber'] for r in roster}
    id_to_name = {str(r['player']['id']): r['player']['nickname'] for r in roster}
    jersey_name = {}  # (team_id, jersey) -> nickname
    jersey_pos = {}   # (team_id, jersey) -> positionGroupType
    for r in roster:
        key = (int(r['team']['id']), r['shirtNumber'])
        jersey_name[key] = r['player']['nickname']
        jersey_pos[key] = r['positionGroupType']

    raw_samples = {'ball': []}  # key -> [(frame, (x,y[,z])), ...]
    period_elapsed_by_frame = {}

    with bz2.open(TRACKING_PATH, 'rt') as f:
        for line in f:
            # cheap pre-filter on frameNum without full json parse would need a
            # regex; full-file decompression dominates cost regardless, so just parse
            d = json.loads(line)
            frame = d['frameNum']
            if frame < lo_frame or frame > hi_frame:
                continue
            if d['period'] != WINDOW_PERIOD:
                continue

            period_elapsed_by_frame[frame] = d.get('periodElapsedTime')

            ball = d.get('ballsSmoothed')
            if isinstance(ball, list):
                ball = ball[0] if ball else None
            if ball is None or ball.get('x') is None:
                raw_ball = d.get('balls') or []
                b0 = raw_ball[0] if raw_ball else None
                if b0 is not None and b0.get('x') is not None:
                    ball = b0
                else:
                    ball = None
            if ball is not None:
                raw_samples['ball'].append((frame, (ball['x'], ball['y'], max(ball.get('z', 0) or 0, 0))))

            for side, team_id in [('home', HOME_TEAM_ID), ('away', AWAY_TEAM_ID)]:
                players = d.get(f'{side}PlayersSmoothed') or d.get(f'{side}Players') or []
                seen = set()
                for pl in players:
                    jn = pl['jerseyNum']
                    if jn in seen or pl.get('x') is None:
                        continue
                    seen.add(jn)
                    key = (team_id, jn)
                    raw_samples.setdefault(key, []).append((frame, (pl['x'], pl['y'])))

    fps_val = fps
    interpolated = {}
    n_ball_dropped = n_player_dropped = 0
    for key, samples in raw_samples.items():
        samples = sorted(samples, key=lambda s: s[0])
        if key == 'ball':
            cleaned = reject_speed_spikes_iterative(samples, BALL_MAX_SPEED_MS, fps_val)
            n_ball_dropped += len(samples) - len(cleaned)
            smoothed = kalman_rts_smooth(cleaned, fps_val, BALL_PROCESS_NOISE_STD, BALL_MEASUREMENT_NOISE_STD)
            interpolated[key] = interpolate_series(smoothed, max_gap_frames=int(BALL_MAX_GAP_S * fps_val))
        else:
            cleaned = reject_speed_spikes_iterative(samples, PLAYER_MAX_SPEED_MS, fps_val)
            n_player_dropped += len(samples) - len(cleaned)
            smoothed = kalman_rts_smooth(cleaned, fps_val, PLAYER_PROCESS_NOISE_STD, PLAYER_MEASUREMENT_NOISE_STD)
            interpolated[key] = interpolate_series(smoothed, max_gap_frames=int(PLAYER_MAX_GAP_S * fps_val))
    print(f"Speed-spike samples dropped: ball={n_ball_dropped}, players={n_player_dropped}")

    # merge in the pipeline outputs (phase/pressure/depth/etc) computed earlier
    from config import pkl_path
    df = pd.read_pickle(pkl_path('match_features_with_pressure'))
    df = df.drop_duplicates(subset='frameNum').set_index('frameNum')

    frames_out = []
    player_meta = {}  # "T-jersey" -> {name, team, pos}

    for frame in range(lo_frame, hi_frame + 1, DOWNSAMPLE):
        row = df.loc[frame] if frame in df.index else None

        players_out = []
        for side, team_id in [('home', HOME_TEAM_ID), ('away', AWAY_TEAM_ID)]:
            for key, series in interpolated.items():
                if key == 'ball' or not isinstance(key, tuple):
                    continue
                t, jn = key
                if t != team_id:
                    continue
                if frame not in series:
                    continue
                x, y = series[frame]
                pid = f"{'H' if side == 'home' else 'A'}{jn}"
                if pid not in player_meta:
                    nm = jersey_name.get((team_id, jn), f"#{jn}")
                    pg = jersey_pos.get((team_id, jn), '')
                    player_meta[pid] = {'name': nm, 'team': side, 'jersey': jn, 'pos': pg}
                players_out.append({'id': pid, 'x': round(x, 2), 'y': round(y, 2)})

        ball_xy = interpolated.get('ball', {}).get(frame)
        ball_out = None
        if ball_xy:
            ball_out = {'x': round(ball_xy[0], 2), 'y': round(ball_xy[1], 2), 'z': round(ball_xy[2], 2)}

        if row is not None:
            def _r(v, nd=2):
                return None if pd.isna(v) else round(float(v), nd)

            presser_id = None
            if pd.notna(row.get('pressing_team_id')) and pd.notna(row.get('primary_presser_jersey')):
                side_code = 'H' if row['pressing_team_id'] == HOME_TEAM_ID else 'A'
                presser_id = f"{side_code}{row['primary_presser_jersey']}"

            def _s(v):
                # pandas' string dtype (this environment's pandas infers it for
                # object columns built via df.apply() that mix None with str --
                # see engagement.py) uses NaN as ITS missing-value sentinel
                # instead of preserving Python None, unlike legacy object
                # dtype. json.dump serializes a bare float NaN as the literal
                # token `NaN`, which is not valid JSON and makes
                # JSON.parse() throw in the browser -- caught by checking the
                # freshly-exported file for literal NaN tokens before this
                # fix. Route every column that can carry a null label
                # (home/away engaged; structure defensively, in case a future
                # column hits the same dtype) through this before it lands
                # in frame_rec.
                return None if pd.isna(v) else v

            frame_rec = {
                't': round(frame / fps, 2),
                'clockS': period_elapsed_by_frame.get(frame),
                'players': players_out,
                'ball': ball_out,
                'possessionTeam': ('home' if row.get('possession_team_id') == HOME_TEAM_ID
                                    else ('away' if row.get('possession_team_id') == AWAY_TEAM_ID else None)),
                'homeStructure': _s(row.get('home_structure')),
                'awayStructure': _s(row.get('away_structure')),
                'homeDepth': _r(row.get('home_depth_smooth')),
                'awayDepth': _r(row.get('away_depth_smooth')),
                'homeInterLine': _r(row.get('home_inter_line_smooth')),
                'awayInterLine': _r(row.get('away_inter_line_smooth')),
                'homeBacklineGap': _r(row.get('home_backline_gap')),
                'awayBacklineGap': _r(row.get('away_backline_gap')),
                'pressureScore': _r(row.get('pressure_score'), 3),
                'pressingTeam': ('home' if row.get('pressing_team_id') == HOME_TEAM_ID
                                  else ('away' if row.get('pressing_team_id') == AWAY_TEAM_ID else None)),
                'primaryPresser': presser_id,
                'homeEngaged': _s(row.get('home_engaged')),
                'awayEngaged': _s(row.get('away_engaged')),
                'homeMarking': _s(row.get('home_marking')),
                'awayMarking': _s(row.get('away_marking')),
            }
        else:
            frame_rec = {'t': round(frame / fps, 2), 'clockS': period_elapsed_by_frame.get(frame),
                         'players': players_out, 'ball': ball_out,
                         'possessionTeam': None, 'homeStructure': None, 'awayStructure': None,
                         'homeDepth': None, 'awayDepth': None, 'homeInterLine': None, 'awayInterLine': None,
                         'homeBacklineGap': None, 'awayBacklineGap': None, 'pressureScore': None,
                         'pressingTeam': None, 'primaryPresser': None,
                         'homeEngaged': None, 'awayEngaged': None,
                         'homeMarking': None, 'awayMarking': None}

        frames_out.append(frame_rec)

    # display minutes come from the boundary frames' real periodElapsedTime,
    # not from the frame-bound constants -- see module docstring on why
    # frameNum and periodElapsedTime are not the same axis in this dataset
    pet_lo = period_elapsed_by_frame.get(lo_frame)
    pet_hi = period_elapsed_by_frame.get(hi_frame)
    window_start_min = pet_lo / 60 if pet_lo is not None else None
    window_end_min = pet_hi / 60 if pet_hi is not None else None

    meta_out = {
        'gameId': '10508',
        'homeTeam': 'Morocco',
        'awayTeam': 'Spain',
        'homeTeamId': HOME_TEAM_ID,
        'awayTeamId': AWAY_TEAM_ID,
        'pitchLength': 105.0,
        'pitchWidth': 68.0,
        'fps': fps / DOWNSAMPLE,
        'windowStartMin': window_start_min,
        'windowEndMin': window_end_min,
        'period': WINDOW_PERIOD,
        'homeDefendsPositiveX': orientation.get(WINDOW_PERIOD),
        'players': player_meta,
    }

    return {'meta': meta_out, 'frames': frames_out}


if __name__ == '__main__':
    import time
    t0 = time.time()
    payload = export_window()
    print(f"Export took {time.time()-t0:.1f}s, {len(payload['frames'])} frames, "
          f"{len(payload['meta']['players'])} players")

    out_path = json_path('animation_data')
    with open(out_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    import os
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")

    # quick sanity: frames missing ball / missing all players
    n_no_ball = sum(1 for fr in payload['frames'] if fr['ball'] is None)
    n_no_players = sum(1 for fr in payload['frames'] if len(fr['players']) == 0)
    print(f"Frames with no ball: {n_no_ball}/{len(payload['frames'])}")
    print(f"Frames with no players: {n_no_players}/{len(payload['frames'])}")
    print(f"Structure label sample (frame 100): {payload['frames'][100]}")

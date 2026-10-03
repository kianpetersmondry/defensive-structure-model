"""
Bekkers-inspired pressure-on-ball-carrier metric, computed directly from
tracking data (geometric, not PFF's binary event-tagged pressure).

Reference: Bekkers, J. (2025) "Pressing Intensity: An Intuitive Measure for
Pressing in Soccer" (arXiv:2501.04712) -- defines pressing via TIME-TO-INTERCEPT
derived from pitch-control components (player velocity, reaction time, max
speed), converted to a probability via a logistic function.

SIMPLIFICATIONS vs. the published method / the `unravelsports` package (being
explicit about this, not passing it off as a literal reimplementation):
  - We do not build the full 2D pitch-control surface. We only evaluate
    time-to-intercept at ONE target point per frame: the ball's location
    (a stand-in for "where the carrier is"), not a grid over the whole pitch.
  - Max speed and reaction time are constants (data-driven per-player max
    speed where we have enough samples, else a default), not fit per-player
    from a larger multi-match sample.
  - Combining multiple defenders' contributions uses a simple probabilistic-OR
    (1 - product of "misses"), not a full team pitch-control integral.

Time-to-intercept model per defender:
  1. During the reaction time tau, the player continues at their current
     velocity: p_react = p + v * tau
  2. They then close the remaining distance to the target at their max speed:
     time_to_intercept = tau + |target - p_react| / max_speed

Pressure contribution of one defender, via logistic decay:
  contribution = 1 / (1 + exp((time_to_intercept - T0) / SCALE))
  (T0 = 1.5s, SCALE = 0.5s -- a defender arriving in <1s contributes ~1,
  one arriving in >3s contributes ~0)

Frame-level pressure_score = 1 - product(1 - contribution_i) over defenders
(diminishing returns for multiple pressers, saturates at 1). The defender
with the single highest contribution is recorded as the primary presser.
"""
import bz2
import json
import math
import numpy as np
import pandas as pd
from config import TRACKING_PATH, HOME_TEAM_ID, AWAY_TEAM_ID

REACTION_TIME_S = 0.7
DEFAULT_MAX_SPEED_MS = 8.0
T0_S = 1.5
SCALE_S = 0.5
VELOCITY_LOOKBACK_MAX_S = 0.5   # if the previous sample for this player is older
                                 # than this, treat velocity as unknown (0,0)


def logistic_contribution(ttc):
    return 1.0 / (1.0 + math.exp((ttc - T0_S) / SCALE_S))


def compute_pressure_for_match(df_meta, orientation, jersey_lookup, fps=29.97):
    """df_meta: the match_features_classified dataframe (indexed for lookup by
    frameNum), providing possession_team_id and ball_x/ball_y per frame -- kept
    consistent with the earlier pipeline stage rather than recomputed here."""
    home_lookup = {j: v for (t, j), v in jersey_lookup.items() if t == str(HOME_TEAM_ID)}
    away_lookup = {j: v for (t, j), v in jersey_lookup.items() if t == str(AWAY_TEAM_ID)}

    # a small number of frameNums appear multiple times in the source file
    # (PFF notes frames tied to multiple possession events get duplicated) --
    # keep the first occurrence for a stable 1:1 frame->info lookup
    dedup = df_meta.drop_duplicates(subset='frameNum', keep='first')
    frame_info = dedup.set_index('frameNum')[['possession_team_id', 'ball_x', 'ball_y']].to_dict('index')

    # Pass 1: per-player empirical max speed.
    # Preferred source: the raw tracking file's own per-frame 'speed' field
    # (present on game 10508 / Morocco vs Spain). Some matches' exports omit
    # this field entirely -- confirmed: game 10515 / Morocco vs France has NO
    # 'speed' key on any player record, which previously meant every single
    # player fell back to one shared DEFAULT_MAX_SPEED_MS=8.0 constant, a much
    # cruder assumption than match 1 got. Fix: for any player with zero raw
    # speed samples, derive their own empirical max speed instead, from
    # frame-to-frame displacement of their SMOOTHED position (the exact same
    # positions the live pressure loop below already differentiates for
    # reaction-time projection) divided by dt. A player with real raw-speed
    # data is completely unaffected -- this only changes players who have no
    # raw speed samples at all, which today is every player in game 10515.
    raw_speed_samples = {}       # (team_id, jersey) -> [speed, ...] from raw 'speed' field
    derived_speed_samples = {}   # (team_id, jersey) -> [speed, ...] from smoothed-position deltas
    _prev_pos_speed_pass = {}

    def _collect_raw_speeds(players, team_id):
        seen = set()
        for pl in players:
            jn = pl['jerseyNum']
            if jn in seen or pl.get('speed') is None:
                continue
            seen.add(jn)
            raw_speed_samples.setdefault((team_id, jn), []).append(pl['speed'])

    def _collect_derived_speeds(players, team_id, video_t):
        seen = set()
        for pl in players:
            jn = pl['jerseyNum']
            if jn in seen or pl.get('x') is None:
                continue
            seen.add(jn)
            key = (team_id, jn)
            x, y = pl['x'], pl['y']
            prev = _prev_pos_speed_pass.get(key)
            if prev is not None:
                pt, px, py = prev
                dt = video_t - pt
                if 0 < dt <= VELOCITY_LOOKBACK_MAX_S:
                    derived_speed_samples.setdefault(key, []).append(math.hypot(x - px, y - py) / dt)
            _prev_pos_speed_pass[key] = (video_t, x, y)

    with bz2.open(TRACKING_PATH, 'rt') as f:
        for line in f:
            d = json.loads(line)
            video_t = d['frameNum'] / fps
            _collect_raw_speeds(d.get('homePlayers') or [], HOME_TEAM_ID)
            _collect_raw_speeds(d.get('awayPlayers') or [], AWAY_TEAM_ID)
            _collect_derived_speeds(d.get('homePlayersSmoothed') or [], HOME_TEAM_ID, video_t)
            _collect_derived_speeds(d.get('awayPlayersSmoothed') or [], AWAY_TEAM_ID, video_t)

    max_speed = {}
    n_from_raw, n_from_derived = 0, 0
    for key in set(raw_speed_samples) | set(derived_speed_samples):
        if raw_speed_samples.get(key):
            arr = np.array(raw_speed_samples[key])
            n_from_raw += 1
        else:
            arr = np.array(derived_speed_samples.get(key) or [])
            n_from_derived += 1
        if len(arr) == 0:
            continue
        p99 = np.percentile(arr, 99)
        max_speed[key] = max(p99, 4.0)  # floor to avoid degenerate tiny max-speeds
    print(f"Derived empirical max-speed for {len(max_speed)} player-slots "
          f"({n_from_raw} from raw tracking speed, {n_from_derived} from smoothed-position "
          f"deltas used as fallback where raw speed is unavailable) "
          f"(mean {np.mean(list(max_speed.values())):.2f} m/s)")

    prev_pos = {}  # (team_id, jersey) -> (video_t, x, y)  [smoothed position]
    rows = []

    with bz2.open(TRACKING_PATH, 'rt') as f:
        for line in f:
            d = json.loads(line)
            frame = d['frameNum']
            info = frame_info.get(frame)
            if info is None:
                continue
            possession_team = info['possession_team_id']
            bx, by = info['ball_x'], info['ball_y']
            if pd.isna(possession_team) or bx is None or by is None or pd.isna(bx) or pd.isna(by):
                continue
            defending_team = AWAY_TEAM_ID if possession_team == HOME_TEAM_ID else HOME_TEAM_ID

            video_t = frame / fps
            players = (d.get('homePlayersSmoothed') if defending_team == HOME_TEAM_ID
                       else d.get('awayPlayersSmoothed')) or []
            lookup = home_lookup if defending_team == HOME_TEAM_ID else away_lookup

            best_contribution = 0.0
            best_jersey = None
            miss_product = 1.0
            n_considered = 0
            seen = set()

            for pl in players:
                jn = pl['jerseyNum']
                if jn in seen or pl.get('x') is None:
                    continue
                seen.add(jn)
                info_p = lookup.get(jn)
                if info_p is None or info_p[1] == 'GK':
                    continue

                key = (defending_team, jn)
                x, y = pl['x'], pl['y']
                vx, vy = 0.0, 0.0
                prev = prev_pos.get(key)
                if prev is not None:
                    pt, px, py = prev
                    dt = video_t - pt
                    if 0 < dt <= VELOCITY_LOOKBACK_MAX_S:
                        vx, vy = (x - px) / dt, (y - py) / dt
                prev_pos[key] = (video_t, x, y)

                react_x, react_y = x + vx * REACTION_TIME_S, y + vy * REACTION_TIME_S
                dist_remaining = math.hypot(bx - react_x, by - react_y)
                vmax = max_speed.get(key, DEFAULT_MAX_SPEED_MS)
                ttc = REACTION_TIME_S + dist_remaining / vmax

                contribution = logistic_contribution(ttc)
                miss_product *= (1.0 - contribution)
                n_considered += 1
                if contribution > best_contribution:
                    best_contribution = contribution
                    best_jersey = jn

            if n_considered == 0:
                continue

            pressure_score = 1.0 - miss_product
            rows.append((frame, pressure_score, defending_team, best_jersey, best_contribution))

    pdf = pd.DataFrame(rows, columns=['frameNum', 'pressure_score', 'pressing_team_id',
                                       'primary_presser_jersey', 'primary_presser_contribution'])
    return pdf


if __name__ == '__main__':
    import time
    from config import load_roster, load_metadata, orientation_lookup, pkl_path

    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    df_meta = pd.read_pickle(pkl_path('match_features_classified'))

    t0 = time.time()
    pdf = compute_pressure_for_match(df_meta, orientation, jersey_lookup, fps=meta['fps'])
    print(f"Pressure computation took {time.time()-t0:.1f}s over {len(pdf)} frames")
    print(pdf['pressure_score'].describe().round(3))
    print()
    print("Pressing team distribution:", pdf['pressing_team_id'].value_counts().to_dict())
    print()
    print("Sample high-pressure frames:")
    print(pdf.sort_values('pressure_score', ascending=False).head(5).to_string())

    out = pkl_path('match_pressure')
    pdf.to_pickle(out)
    print(f"Saved {out}")

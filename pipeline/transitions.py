"""
Phase classification: In Possession / Defensive Transition / Organized Defense,
per team, per frame.

Transition length is detected per loss-of-possession event via a joint
depth + inter-line "settling" detector: a team is considered to have
re-organized once BOTH its smoothed depth and smoothed inter-line compactness
hold near a stable reference value for most of the remaining time before the
next event. This follows Bauer & Anzer (2021)'s finding that there's no single
clean duration threshold for transition -- it's genuinely event-dependent, so
a settling detector is more defensible than a fixed cutoff.

Rewritten cleanly (self-contained, no cross-notebook state) from the
reference notebook's Cell 9 concept.
"""
import bisect
import pandas as pd
from config import HOME_TEAM_ID, AWAY_TEAM_ID

DEPTH_SMOOTHING_S = 3.0
MAX_TRANSITION_S = 20.0
SAMPLE_STEP_S = 0.5
REFERENCE_WINDOW_S = 3.0
SETTLE_TOLERANCE_M = 3.0       # depth tolerance band
IL_TOLERANCE_M = 8.0           # inter-line tolerance band (wider natural range)
SETTLE_SUCCESS_FRACTION = 0.85


def add_smoothed_columns(df):
    idx = pd.to_timedelta(df['video_t'], unit='s')
    for col in ['home_depth', 'away_depth', 'home_inter_line', 'away_inter_line',
                'home_backline_gap', 'away_backline_gap']:
        s = df.set_index(idx)[col].astype(float)
        df[col + '_smooth'] = s.rolling(f'{DEPTH_SMOOTHING_S}s').mean().values
    return df


def add_possession_column(df, otb_times, otb_teams):
    otb_df = pd.DataFrame({'time_s': otb_times, 'possession_team_id': otb_teams})
    df = pd.merge_asof(df.sort_values('video_t'), otb_df, left_on='video_t', right_on='time_s',
                        direction='backward')
    df.drop(columns=['time_s'], inplace=True)
    return df


def _get_smoothed_series(df, col):
    out = df[['video_t']].copy()
    out['val'] = df[col].values
    return out


def _detect_settle_duration(depth_s, il_s, loss_time, horizon):
    """Returns (duration_s_after_loss, settled_bool)."""
    depth_samples, il_samples = [], []
    t = loss_time
    # both series share the same video_t index/spacing, so a single nearest-index
    # lookup per timestep is enough
    while t <= horizon:
        idx = (depth_s['video_t'] - t).abs().idxmin()
        if abs(depth_s.loc[idx, 'video_t'] - t) <= SAMPLE_STEP_S:
            vd, vil = depth_s.loc[idx, 'val'], il_s.loc[idx, 'val']
            if pd.notna(vd) and pd.notna(vil):
                depth_samples.append((t - loss_time, vd))
                il_samples.append((t - loss_time, vil))
        t += SAMPLE_STEP_S

    if len(depth_samples) < 2:
        return 0.0, False

    ref_d = [v for (tr, v) in depth_samples if tr >= depth_samples[-1][0] - REFERENCE_WINDOW_S]
    ref_il = [v for (tr, v) in il_samples if tr >= il_samples[-1][0] - REFERENCE_WINDOW_S]
    if not ref_d or not ref_il:
        return depth_samples[-1][0], False
    reference_d = sum(ref_d) / len(ref_d)
    reference_il = sum(ref_il) / len(ref_il)

    for i in range(len(depth_samples)):
        t_rel = depth_samples[i][0]
        remaining_d = [v for (_, v) in depth_samples[i:]]
        remaining_il = [v for (_, v) in il_samples[i:]]
        n_ok_d = sum(1 for v in remaining_d if abs(v - reference_d) <= SETTLE_TOLERANCE_M)
        n_ok_il = sum(1 for v in remaining_il if abs(v - reference_il) <= IL_TOLERANCE_M)
        if (n_ok_d / len(remaining_d) >= SETTLE_SUCCESS_FRACTION and
                n_ok_il / len(remaining_il) >= SETTLE_SUCCESS_FRACTION):
            return t_rel, True

    return depth_samples[-1][0], False


def compute_transition_durations(df, lost_ball_times_for_team, depth_col, il_col):
    depth_s = _get_smoothed_series(df, depth_col)
    il_s = _get_smoothed_series(df, il_col)
    results = []
    times = lost_ball_times_for_team
    for i, loss_t in enumerate(times):
        next_t = times[i + 1] if i + 1 < len(times) else None
        horizon = loss_t + MAX_TRANSITION_S
        if next_t is not None:
            horizon = min(horizon, next_t)
        duration, settled = _detect_settle_duration(depth_s, il_s, loss_t, horizon)
        results.append({'loss_time': loss_t, 'transition_duration_s': duration, 'settled': settled})
    return pd.DataFrame(results)


def classify_possession_phase(df, team_id, lost_ball_times_for_team, transition_lookup):
    times_arr = lost_ball_times_for_team
    out = []
    for _, row in df.iterrows():
        if row['possession_team_id'] == team_id:
            out.append('In Possession')
            continue
        idx = bisect.bisect_right(times_arr, row['video_t']) - 1
        if idx < 0:
            out.append('Organized Defense')
            continue
        loss_t = times_arr[idx]
        duration = transition_lookup.get(loss_t, 5.0)
        if (row['video_t'] - loss_t) <= duration:
            out.append('Defensive Transition')
        else:
            out.append('Organized Defense')
    return out


if __name__ == '__main__':
    import time
    from config import load_roster, load_metadata, orientation_lookup, pkl_path
    from events_processing import load_events, build_event_signals

    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    events = load_events()
    sig = build_event_signals(events, player_id_lookup, orientation)

    df = pd.read_pickle(pkl_path('match_features'))
    df = add_smoothed_columns(df)
    df = add_possession_column(df, sig['otb_times'], sig['otb_teams'])
    print("Possession distribution:", df['possession_team_id'].value_counts(dropna=False).to_dict())

    t0 = time.time()
    home_trans = compute_transition_durations(df, sig['lost_ball_times'][HOME_TEAM_ID], 'home_depth_smooth', 'home_inter_line_smooth')
    away_trans = compute_transition_durations(df, sig['lost_ball_times'][AWAY_TEAM_ID], 'away_depth_smooth', 'away_inter_line_smooth')
    print(f"Transition duration detection took {time.time()-t0:.1f}s")
    from config import HOME_TEAM_NAME, AWAY_TEAM_NAME
    print(f"{HOME_TEAM_NAME} transition durations:", home_trans['transition_duration_s'].describe().round(2).to_dict())
    print(f"{AWAY_TEAM_NAME} transition durations:", away_trans['transition_duration_s'].describe().round(2).to_dict())
    print(f"{HOME_TEAM_NAME} settled fraction:", home_trans['settled'].mean())
    print(f"{AWAY_TEAM_NAME} settled fraction:", away_trans['settled'].mean())

    home_lookup_ = dict(zip(home_trans['loss_time'], home_trans['transition_duration_s']))
    away_lookup_ = dict(zip(away_trans['loss_time'], away_trans['transition_duration_s']))

    t0 = time.time()
    df['home_phase'] = classify_possession_phase(df, HOME_TEAM_ID, sig['lost_ball_times'][HOME_TEAM_ID], home_lookup_)
    df['away_phase'] = classify_possession_phase(df, AWAY_TEAM_ID, sig['lost_ball_times'][AWAY_TEAM_ID], away_lookup_)
    print(f"Classification took {time.time()-t0:.1f}s")
    print(f"{HOME_TEAM_NAME} phase dist:", df['home_phase'].value_counts().to_dict())
    print(f"{AWAY_TEAM_NAME} phase dist:", df['away_phase'].value_counts().to_dict())

    out = pkl_path('match_features_phased')
    df.to_pickle(out)
    print(f"Saved {out}")

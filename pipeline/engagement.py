"""
Idea #1 (engaged vs. passive defense), final scoped-down version per explicit
user direction: this is NOT a new phase-detection definition (no new
High Press/Mid/Low Block variants, no new hub version) -- it's a live add-on
readout next to the existing phase label, describing whether a team's
current organized-defense phase involves genuine 1v1 engagement or is just
holding its shape/depth passively. User's own words: "I think its an
interesting find but dosent really work as its own parameter, more of an
add on to help describe the press."

Reuses pressure.py's already-computed, already-validated pressure_score
(time-to-intercept metric) rather than a new signal. Two differences from
the companion-artifact version (analysis_engaged_passive.py /
build_engaged_passive.py):
  1. Smoothed like depth/inter-line already are (transitions.py's
     DEPTH_SMOOTHING_S=3.0 trailing rolling mean), not the companion chart's
     whole-phase-instance majority vote -- this is a LIVE per-frame readout
     next to a label that itself updates live, so it needs frame-to-frame
     stability from smoothing, not segment-level aggregation.
  2. The engaged threshold is re-derived against the SMOOTHED metric (not
     reused from the raw-score threshold in analysis_engaged_passive.py),
     since smoothing shifts the score distribution -- re-validating keeps
     this honest rather than assuming the raw threshold still applies.
"""
import json
import numpy as np
import pandas as pd
from config import HOME_TEAM_ID, AWAY_TEAM_ID

PRESSURE_SMOOTHING_S = 3.0  # matches transitions.py's DEPTH_SMOOTHING_S convention
ORGANIZED_STATES = ('High Press', 'Mid Block', 'Low Block')


def add_smoothed_pressure(df):
    """Adds home_pressure_smooth / away_pressure_smooth: a trailing 3s rolling
    mean of pressure_score, computed only over each team's own pressing
    frames (pressure_score is null whenever that team has the ball, exactly
    like depth/inter-line are null nowhere but pressure only applies to the
    defending team) -- then reindexed back onto the full frame grid."""
    df = df.sort_values('video_t').reset_index(drop=True)
    for team_id, prefix in [(HOME_TEAM_ID, 'home'), (AWAY_TEAM_ID, 'away')]:
        mask = df['pressing_team_id'] == team_id
        sub = df.loc[mask, ['video_t', 'pressure_score']].copy()
        sub = sub.set_index(pd.to_timedelta(sub['video_t'], unit='s'))
        smooth = sub['pressure_score'].astype(float).rolling(f'{PRESSURE_SMOOTHING_S}s').mean()
        col = f'{prefix}_pressure_smooth'
        df[col] = np.nan
        df.loc[mask, col] = smooth.values
    return df


def derive_engaged_threshold(df, events_path=None):
    """Youden's-J threshold against PFF's own analyst-tagged pressure field
    (initialPressureType P/N), matched to the SMOOTHED metric this time.

    events_path defaults to THIS match's own events file (config.EVENTS_PATH),
    not a hardcoded one. Bug found round 33: the old hardcoded default always
    pointed at game 10508's (Morocco vs Spain) events.json regardless of which
    match MATCH_ID was actually pointed at, so match 2's (Morocco vs France)
    threshold was silently derived by matching ITS pressure scores against
    MATCH 1's tagged-pressure timestamps -- comparing two different games'
    events to each other. That alone fully explains match 2's previously
    "notably weaker" AUC (0.541): re-derived against its own match's events,
    match 2's AUC is 0.778, actually slightly stronger than match 1's 0.766."""
    if events_path is None:
        from config import EVENTS_PATH
        events_path = EVENTS_PATH
    with open(events_path) as f:
        events = json.load(f)
    tagged = []
    for e in events:
        it = e.get('initialTouch') or {}
        ptype = it.get('initialPressureType')
        if ptype not in ('P', 'N'):
            continue
        ge = e.get('gameEvents') or {}
        team_id, t = ge.get('teamId'), e.get('startTime')
        if team_id is None or t is None:
            continue
        defending_team = AWAY_TEAM_ID if team_id == HOME_TEAM_ID else HOME_TEAM_ID
        tagged.append((t, defending_team, ptype))
    tagged_df = pd.DataFrame(tagged, columns=['t', 'defending_team', 'ptype'])

    home_p = df[['video_t', 'home_pressure_smooth']].dropna().rename(columns={'home_pressure_smooth': 'score'})
    away_p = df[['video_t', 'away_pressure_smooth']].dropna().rename(columns={'away_pressure_smooth': 'score'})
    streams = {HOME_TEAM_ID: home_p.sort_values('video_t'), AWAY_TEAM_ID: away_p.sort_values('video_t')}

    def nearest(row):
        sub = streams[row['defending_team']]
        if sub.empty:
            return np.nan
        idx = (sub['video_t'] - row['t']).abs().idxmin()
        return sub.loc[idx, 'score'] if abs(sub.loc[idx, 'video_t'] - row['t']) <= 0.5 else np.nan

    tagged_df['score'] = tagged_df.apply(nearest, axis=1)
    matched = tagged_df.dropna(subset=['score'])
    p = matched.loc[matched.ptype == 'P', 'score']
    n = matched.loc[matched.ptype == 'N', 'score']

    best_j, best_t = -1, 0.5
    for thresh in np.arange(0.05, 0.96, 0.01):
        j = (p >= thresh).mean() - (n >= thresh).mean()
        if j > best_j:
            best_j, best_t = j, thresh
    from scipy import stats
    auc = stats.mannwhitneyu(p, n, alternative='greater').statistic / (len(p) * len(n))
    return {
        'threshold': round(float(best_t), 2), 'auc': round(float(auc), 3), 'j': round(float(best_j), 3),
        'tpr': round(float((p >= best_t).mean()), 3), 'fpr': round(float((n >= best_t).mean()), 3),
        'p_n': int(len(p)), 'n_n': int(len(n)),
        'p_mean': round(float(p.mean()), 3), 'n_mean': round(float(n.mean()), 3),
    }


def add_engagement_labels(df, threshold):
    for team_id, prefix in [(HOME_TEAM_ID, 'home'), (AWAY_TEAM_ID, 'away')]:
        struct_col = f'{prefix}_structure'
        smooth_col = f'{prefix}_pressure_smooth'
        out_col = f'{prefix}_engaged'

        def _row(r):
            if r[struct_col] not in ORGANIZED_STATES or pd.isna(r[smooth_col]):
                return None
            return 'Engaged' if r[smooth_col] >= threshold else 'Passive'

        df[out_col] = df.apply(_row, axis=1)
    return df


if __name__ == '__main__':
    from config import pkl_path, json_path
    PATH = pkl_path('match_features_with_pressure')
    df = pd.read_pickle(PATH)
    df = add_smoothed_pressure(df)
    thresh_info = derive_engaged_threshold(df)
    print('Threshold info:', thresh_info)
    df = add_engagement_labels(df, thresh_info['threshold'])

    print('\nhome_engaged counts:')
    print(df['home_engaged'].value_counts(dropna=False))
    print('\naway_engaged counts:')
    print(df['away_engaged'].value_counts(dropna=False))

    df.to_pickle(PATH)
    with open(json_path('engagement_threshold'), 'w') as f:
        json.dump(thresh_info, f)
    print(f'\nSaved {PATH} with home_engaged/away_engaged columns, and engagement_threshold.json')

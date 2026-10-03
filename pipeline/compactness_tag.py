"""
Idea #3 (ball-proximal compactness), final scoped-down version per explicit
user direction: NOT a third defining variable in the phase classification
(analysis_bpc_turnover_validation.py found no clean threshold and no
independent predictive value over inter_line for predicting an imminent
turnover) -- instead a descriptive add-on tag next to the phase pill, the
same treatment as Idea #1's Engaged/Passive tag. User's own words: "...have
it just listed on the key like the Engaged/Passive pattern."

Reuses ball_proximal_compactness.py's already-computed BPC metric (the
"diameter" of the 5 outfield players nearest the ball, per Forcher et al.'s
finding that ball-proximal compactness -- not whole-team compactness -- is
the one that predicts defensive success), smoothed with the SAME trailing
3s rolling-window convention transitions.py already uses for depth/
inter-line (video_t-indexed, whole-match trailing mean) -- this is a LIVE
per-frame readout next to a label that itself updates live, so it needs the
same convention the rest of the live shape metrics already use, not the
true-clock/period-grouped centered smoothing the one-off analysis scripts
used.

Threshold: a balanced median split (23m) over all organized-defense frames
(High Press/Mid/Low Block). This is descriptive, not predictive, so unlike
Engaged/Passive's threshold it doesn't need Youden's-J validation against an
outcome -- it just needs to split the population meaningfully, which 23m
does (51%/49% Tight/Loose over organized defense).
"""
import pandas as pd
from config import HOME_TEAM_ID, AWAY_TEAM_ID

BPC_SMOOTHING_S = 3.0
TIGHT_LOOSE_THRESHOLD_M = 23.0
ORGANIZED_STATES = ('High Press', 'Mid Block', 'Low Block')


def add_smoothed_bpc(df, bpc_df):
    df = df.merge(bpc_df, on='frameNum', how='left')
    idx = pd.to_timedelta(df['video_t'], unit='s')
    for col in ['home_bpc', 'away_bpc']:
        s = df.set_index(idx)[col].astype(float)
        df[col + '_smooth'] = s.rolling(f'{BPC_SMOOTHING_S}s').mean().values
    return df


def add_marking_labels(df, threshold=TIGHT_LOOSE_THRESHOLD_M):
    for team_id, prefix in [(HOME_TEAM_ID, 'home'), (AWAY_TEAM_ID, 'away')]:
        struct_col = f'{prefix}_structure'
        smooth_col = f'{prefix}_bpc_smooth'
        out_col = f'{prefix}_marking'

        def _row(r):
            if r[struct_col] not in ORGANIZED_STATES or pd.isna(r[smooth_col]):
                return None
            return 'Tight' if r[smooth_col] <= threshold else 'Loose'

        df[out_col] = df.apply(_row, axis=1)
    return df


if __name__ == '__main__':
    from config import pkl_path
    PATH = pkl_path('match_features_with_pressure')
    df = pd.read_pickle(PATH)
    bpc_df = pd.read_pickle(pkl_path('ball_proximal_compactness'))
    df = add_smoothed_bpc(df, bpc_df)
    df = add_marking_labels(df)

    print('home_marking counts:')
    print(df['home_marking'].value_counts(dropna=False))
    print('\naway_marking counts:')
    print(df['away_marking'].value_counts(dropna=False))

    df.to_pickle(PATH)
    print(f'\nSaved {PATH} with home_marking/away_marking columns '
          f'(Tight/Loose, threshold={TIGHT_LOOSE_THRESHOLD_M}m)')

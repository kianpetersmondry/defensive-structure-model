"""
Idea #2 (deferred phase-detection-definition candidate): "does wide ball-side
pressure intensity predict back-line height in the following seconds" -- the
professor's coaching-note hypothesis (see literature-and-data-foundations.md,
"External input worth logging"). Pure analysis, no new modeling: both
ingredients (pressure_score, depth_smooth) already exist in
match_features_with_pressure{suffix}.pkl.

Made match-aware round 33 (previously hardcoded to game 10508 only, with a
hardcoded HOME_TEAM_ID/AWAY_TEAM_ID and pet_by_frame.json path) so this idea
could finally be checked against BOTH matches before writing a conclusion,
instead of generalizing from Morocco-Spain alone. Run as:
    python3 analysis_wide_pressure_leadlag.py          # game 10508
    MATCH_ID=10515 python3 analysis_wide_pressure_leadlag.py   # game 10515

Definitions:
  - "Wide" frame: |ball_y| > WIDE_Y (five-channel model on this pitch's real
    68m width -- channels are 68/5 = 13.6m each; "wide" = outer channel,
    boundary at 1.5 channel-widths from center = 20.4m). Nothing in the
    existing codebase defines a wide-channel cutoff, so this is a fresh,
    literature-grounded (five-channel/half-space tactical convention)
    choice, not a re-derivation of an existing constant.
  - "Ball-side wide pressure" at a wide frame = pressure_score as already
    computed (pressure.py evaluates time-to-intercept AT THE BALL'S actual
    location every frame) -- restricting to wide frames is exactly
    "pressure on the ball carrier while the ball is out wide", no
    recomputation of pressure.py needed.
  - "Defending team's depth" = home_depth_smooth when the away team has the
    ball, away_depth_smooth when the home team has the ball -- stitched
    into one continuous "whoever is currently defending" series. Flagged
    as a simplification: this series jumps discontinuously at a turnover
    (from one team's shape to the other's), since it's not a single
    physical quantity -- see the printed caveat below.

Method: resample both series onto a uniform 1s grid (period-local true
clock, from pet_by_frame{suffix}.json, never crossing a period boundary), then
compute Pearson correlation between wide-pressure at time t and defending
depth at time t+lag for lag in [-20, +20]s, using only 1s bins that have at
least one wide-pressure sample. Positive lag = does wide pressure precede a
depth change; negative lag = does depth level precede pressure (reverse
check, since a team already pushed up may find fewer wide 1v1s available).
"""
import json
import numpy as np
import pandas as pd
from config import HOME_TEAM_ID, AWAY_TEAM_ID, HOME_TEAM_NAME, AWAY_TEAM_NAME, MATCH_LABEL, pkl_path, json_path

print(f"=== Wide ball-side pressure vs. back-line depth: {MATCH_LABEL} ===\n")

df = pd.read_pickle(pkl_path('match_features_with_pressure'))
df = df.drop_duplicates(subset='frameNum', keep='first')

pet = json.load(open(json_path('pet_by_frame')))
pet_by_frame = {int(k): v for k, v in pet['pet_by_frame'].items()}
period_by_frame = {int(k): v for k, v in pet['period_by_frame'].items()}

df['clockS'] = df['frameNum'].map(pet_by_frame)
df['true_period'] = df['frameNum'].map(period_by_frame)
df = df.dropna(subset=['clockS', 'true_period'])

WIDE_Y = 68.0 / 5 * 1.5  # = 20.4m, outer of 5 equal-width tactical channels

df['defending_depth'] = np.where(
    df['possession_team_id'] == HOME_TEAM_ID, df['away_depth_smooth'],
    np.where(df['possession_team_id'] == AWAY_TEAM_ID, df['home_depth_smooth'], np.nan)
)
df['is_wide'] = df['ball_y'].abs() > WIDE_Y
df['wide_pressure'] = np.where(df['is_wide'], df['pressure_score'], np.nan)

print(f"Total frames: {len(df)}")
print(f"Frames with resolved possession + ball position: {df['defending_depth'].notna().sum()}")
print(f"Frames with ball in wide channel (|y|>{WIDE_Y:.1f}m): {df['is_wide'].sum()} "
      f"({100*df['is_wide'].mean():.1f}% of all frames)")
print(f"Frames with wide + resolved pressure_score: {df['wide_pressure'].notna().sum()}")

# --- resample to a uniform 1s grid, per true period, never crossing a break ---
rows = []
for period, g in df.groupby('true_period'):
    g = g.sort_values('clockS')
    t_max = g['clockS'].max()
    g = g.copy()
    g['bin'] = np.floor(g['clockS']).astype(int)
    agg = g.groupby('bin').agg(
        wide_pressure=('wide_pressure', 'mean'),
        defending_depth=('defending_depth', 'mean'),
        n_wide=('is_wide', 'sum'),
    ).reindex(range(int(t_max) + 1))
    agg['period'] = period
    rows.append(agg)

grid = pd.concat(rows).reset_index().rename(columns={'index': 'bin'})
print(f"\n1s-bin grid: {len(grid)} bins across {grid['period'].nunique()} periods, "
      f"{grid['wide_pressure'].notna().sum()} bins with a wide-pressure reading")

# --- lagged correlation, computed per period then combined (never shifts
# across a period boundary since each period's series is handled separately
# and re-concatenated with NaN padding at the join) ---
LAGS = range(-20, 21)
results = []
for lag in LAGS:
    xs, ys = [], []
    for period, g in grid.groupby('period'):
        g = g.sort_values('bin').reset_index(drop=True)
        x = g['wide_pressure'].values
        y = g['defending_depth'].values
        if lag >= 0:
            xa, yb = x[:len(x) - lag] if lag > 0 else x, y[lag:] if lag > 0 else y
        else:
            xa, yb = x[-lag:], y[:len(y) + lag]
        xs.append(xa)
        ys.append(yb)
    xall = np.concatenate(xs)
    yall = np.concatenate(ys)
    mask = ~np.isnan(xall) & ~np.isnan(yall)
    n = mask.sum()
    if n < 20:
        results.append((lag, np.nan, n))
        continue
    r = np.corrcoef(xall[mask], yall[mask])[0, 1]
    results.append((lag, r, n))

res_df = pd.DataFrame(results, columns=['lag_s', 'pearson_r', 'n'])
print("\nLagged correlation (wide ball-side pressure at t  vs.  defending depth at t+lag):")
print(res_df.to_string(index=False))

best = res_df.loc[res_df['pearson_r'].abs().idxmax()]
print(f"\nStrongest |r| at lag={best['lag_s']:.0f}s: r={best['pearson_r']:.3f} (n={best['n']:.0f})")

res_df.to_csv(json_path('wide_pressure_leadlag').replace('.json', '.csv'), index=False)
grid.to_csv(json_path('wide_pressure_grid').replace('.json', '.csv'), index=False)
print(f"\nSaved {json_path('wide_pressure_leadlag').replace('.json', '.csv')} and "
      f"{json_path('wide_pressure_grid').replace('.json', '.csv')}")

# ---------------------------------------------------------------------------
# Follow-up refinements, since the raw whole-match/all-lags correlation above
# is often weak -- checking whether it's masked by (a) the two teams'
# responses canceling out, (b) using depth LEVEL instead of the
# following-seconds CHANGE in depth, or (c) transition-phase noise
# contaminating "defending depth" the same way it contaminated the phase-vote
# work in rounds 22-23.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("REFINEMENT A: split by which team is defending")
print("=" * 70)
df['defending_team'] = np.select(
    [df['possession_team_id'] == HOME_TEAM_ID, df['possession_team_id'] == AWAY_TEAM_ID],
    ['away_defending', 'home_defending'],
    default=None,
)

refinement_a = {}
for team_flag, label in [('home_defending', HOME_TEAM_NAME), ('away_defending', AWAY_TEAM_NAME)]:
    sub = df[df['defending_team'] == team_flag]
    xs = []
    for period, g in sub.groupby('true_period'):
        g = g.sort_values('clockS').copy()
        g['bin'] = np.floor(g['clockS']).astype(int)
        agg = g.groupby('bin').agg(wp=('wide_pressure', 'mean'), dd=('defending_depth', 'mean'))
        xs.append(agg)
    grid_t = pd.concat(xs).reset_index()
    best_r, best_lag, best_n = 0, 0, 0
    for lag in range(-20, 21):
        x = grid_t['wp'].values
        y = grid_t['dd'].values
        if lag >= 0:
            xa, yb = (x[:len(x) - lag] if lag > 0 else x), (y[lag:] if lag > 0 else y)
        else:
            xa, yb = x[-lag:], y[:len(y) + lag]
        mask = ~np.isnan(xa) & ~np.isnan(yb)
        if mask.sum() < 20:
            continue
        r = np.corrcoef(xa[mask], yb[mask])[0, 1]
        if abs(r) > abs(best_r):
            best_r, best_lag, best_n = r, lag, int(mask.sum())
    refinement_a[label] = (best_r, best_lag, best_n)
    print(f"  {label} defending: strongest |r|={best_r:.3f} at lag={best_lag:+d}s (n bins={grid_t['wp'].notna().sum()})")

print("\n" + "=" * 70)
print("REFINEMENT B: target = CHANGE in defending depth over the next N seconds")
print("(instead of the depth level itself)")
print("=" * 70)
for horizon in [3, 5, 10, 15]:
    xs = []
    for period, g in df.groupby('true_period'):
        g = g.sort_values('clockS').copy()
        g['bin'] = np.floor(g['clockS']).astype(int)
        agg = g.groupby('bin').agg(wp=('wide_pressure', 'mean'), dd=('defending_depth', 'mean')).reindex(
            range(int(g['bin'].max()) + 1))
        agg['delta'] = agg['dd'].shift(-horizon) - agg['dd']
        xs.append(agg[['wp', 'delta']])
    comb = pd.concat(xs)
    mask = comb['wp'].notna() & comb['delta'].notna()
    r = np.corrcoef(comb.loc[mask, 'wp'], comb.loc[mask, 'delta'])[0, 1]
    print(f"  horizon={horizon:2d}s: r(wide_pressure_t, depth_change_[t,t+{horizon}])={r:+.3f}  (n={mask.sum()})")

print("\n" + "=" * 70)
print("REFINEMENT C: restrict to defending team in ORGANIZED defense only")
print("(exclude Defensive Transition -- matches round 23's phase-vote fix)")
print("=" * 70)
df['defending_structure'] = np.select(
    [df['defending_team'] == 'home_defending', df['defending_team'] == 'away_defending'],
    [df['home_structure'], df['away_structure']],
    default=None,
)
organized = df[df['defending_structure'].isin(['High Press', 'Mid Block', 'Low Block'])]
print(f"  organized-defense frames: {len(organized)} of {df['defending_structure'].notna().sum()} "
      f"defending-resolved frames ({100*len(organized)/df['defending_structure'].notna().sum():.1f}%)")
xs = []
for period, g in organized.groupby('true_period'):
    g = g.sort_values('clockS').copy()
    g['bin'] = np.floor(g['clockS']).astype(int)
    agg = g.groupby('bin').agg(wp=('wide_pressure', 'mean'), dd=('defending_depth', 'mean'))
    xs.append(agg)
grid_o = pd.concat(xs).reset_index()
best_r, best_lag = 0, 0
for lag in range(-20, 21):
    x = grid_o['wp'].values
    y = grid_o['dd'].values
    if lag >= 0:
        xa, yb = (x[:len(x) - lag] if lag > 0 else x), (y[lag:] if lag > 0 else y)
    else:
        xa, yb = x[-lag:], y[:len(y) + lag]
    mask = ~np.isnan(xa) & ~np.isnan(yb)
    if mask.sum() < 20:
        continue
    r = np.corrcoef(xa[mask], yb[mask])[0, 1]
    if abs(r) > abs(best_r):
        best_r, best_lag = r, lag
print(f"  organized-defense-only: strongest |r|={best_r:.3f} at lag={best_lag:+d}s (n bins={grid_o['wp'].notna().sum()})")

"""
Compares the new ball-proximal compactness (BPC) metric against the existing
whole-team inter_line metric, both restricted to the currently-DEFENDING
team, to test whether BPC would be a better Low Block compactness gate.

Current rule (classification.py): Low Block requires depth < 32m AND
inter_line <= 20m, else downgraded to Mid Block.
"""
import json
import numpy as np
import pandas as pd

HOME_TEAM_ID = 374
AWAY_TEAM_ID = 52

mfp = pd.read_pickle('/home/claude/project_work/match_features_with_pressure.pkl')
mfp = mfp.drop_duplicates(subset='frameNum', keep='first')
bpc = pd.read_pickle('/home/claude/project_work/ball_proximal_compactness.pkl')

df = mfp.merge(bpc, on='frameNum', how='inner')

pet = json.load(open('/home/claude/project_work/pet_by_frame.json'))
pet_by_frame = {int(k): v for k, v in pet['pet_by_frame'].items()}
period_by_frame = {int(k): v for k, v in pet['period_by_frame'].items()}
df['clockS'] = df['frameNum'].map(pet_by_frame)
df['true_period'] = df['frameNum'].map(period_by_frame)

df['defending_team'] = np.select(
    [df['possession_team_id'] == HOME_TEAM_ID, df['possession_team_id'] == AWAY_TEAM_ID],
    ['away_defending', 'home_defending'], default=None)
df['defending_depth'] = np.select(
    [df['defending_team'] == 'home_defending', df['defending_team'] == 'away_defending'],
    [df['home_depth_smooth'], df['away_depth_smooth']], default=np.nan)
df['defending_inter_line'] = np.select(
    [df['defending_team'] == 'home_defending', df['defending_team'] == 'away_defending'],
    [df['home_inter_line_smooth'], df['away_inter_line_smooth']], default=np.nan)
df['defending_bpc_raw'] = np.select(
    [df['defending_team'] == 'home_defending', df['defending_team'] == 'away_defending'],
    [df['home_bpc'], df['away_bpc']], default=np.nan)
df['defending_structure'] = np.select(
    [df['defending_team'] == 'home_defending', df['defending_team'] == 'away_defending'],
    [df['home_structure'], df['away_structure']], default=None)

# 3-second rolling smooth of BPC, per team per period, same convention as
# tracking_features.py/transitions.py use for depth/inter_line (DEPTH_SMOOTHING_S=3.0)
df = df.sort_values(['true_period', 'clockS'])
def smooth_group(g, col, window_s=3.0):
    g = g.set_index(pd.to_timedelta(g['clockS'], unit='s'))
    return g[col].rolling(f'{window_s}s', min_periods=1).mean().values

out = []
for (period, team), g in df.groupby(['true_period', 'defending_team']):
    g = g.sort_values('clockS').copy()
    g['defending_bpc_smooth'] = smooth_group(g, 'defending_bpc_raw')
    out.append(g)
df = pd.concat(out).sort_index()

valid = df.dropna(subset=['defending_depth', 'defending_inter_line', 'defending_bpc_smooth', 'defending_structure'])
print(f"Valid rows for comparison: {len(valid)} of {len(df)}")

print("\n--- Correlation: whole-team inter_line vs. ball-proximal compactness (defending team) ---")
r = np.corrcoef(valid['defending_inter_line'], valid['defending_bpc_smooth'])[0, 1]
print(f"Pearson r = {r:.3f}  (1.0 would mean BPC is just measuring the same thing)")

print("\n--- Distribution of each metric by defending structure (smoothed classification) ---")
summary = valid.groupby('defending_structure')[['defending_inter_line', 'defending_bpc_smooth']].agg(
    ['mean', 'std', 'count']).round(2)
print(summary)

# Separation quality: ratio of between-group variance to within-group variance
# (one-way ANOVA F-statistic, informal use -- higher = the metric separates
# High Press/Mid/Low Block/Transition more cleanly)
def anova_f(series, groups):
    overall_mean = series.mean()
    ss_between, ss_within = 0.0, 0.0
    for gname, gvals in series.groupby(groups):
        ss_between += len(gvals) * (gvals.mean() - overall_mean) ** 2
        ss_within += ((gvals - gvals.mean()) ** 2).sum()
    k = groups.nunique()
    n = len(series)
    if ss_within == 0 or n - k == 0:
        return np.nan
    return (ss_between / (k - 1)) / (ss_within / (n - k))

f_il = anova_f(valid['defending_inter_line'], valid['defending_structure'])
f_bpc = anova_f(valid['defending_bpc_smooth'], valid['defending_structure'])
print(f"\nANOVA F-stat by defending_structure:  inter_line={f_il:.1f}   ball-proximal compactness={f_bpc:.1f}")
print("(higher F = the metric's mean differs more sharply across High Press/Mid/Low Block/Transition)")

# --- Re-derive Low Block using BPC instead of inter_line ---
print("\n--- Low Block gate comparison (candidates = depth<32m frames only) ---")
candidates = valid[valid['defending_depth'] < 32]
print(f"Depth<32m candidate frames: {len(candidates)}")
current_low = candidates['defending_inter_line'] <= 20
print(f"Current rule (inter_line<=20m): {current_low.sum()} Low Block ({100*current_low.mean():.1f}% of candidates)")

for thresh in [12, 14, 16, 18, 20]:
    new_low = candidates['defending_bpc_smooth'] <= thresh
    agree = (new_low == current_low).mean()
    print(f"BPC<={thresh}m: {new_low.sum()} Low Block ({100*new_low.mean():.1f}% of candidates), "
          f"agrees with current rule on {100*agree:.1f}% of candidate frames")

valid.to_pickle('/home/claude/project_work/compactness_compare.pkl')
print("\nSaved compactness_compare.pkl")

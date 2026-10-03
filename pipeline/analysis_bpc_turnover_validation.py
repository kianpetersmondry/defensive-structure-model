"""
Idea #3 follow-up validation (user: "do more anylisis before we make a
decision... if we could find good thresholds and it gave us good and
accurate results").

The bucket-level PPDA-by-phase validation (round 23's method) is
underpowered for this question: Morocco has only 7 Low Block 300s-buckets
in the whole match, Spain effectively 1. This script instead uses a much
higher-power, genuinely independent, event-derived outcome: for every
1-second slice where a team is defending, does that team WIN THE BALL BACK
(an OTB event where they become the new possessing team -- a "turnover" in
their favor) within the next K seconds? That is tens of thousands of
observations instead of a couple dozen buckets, and "did a turnover happen"
comes straight from the raw event feed, not from tracking positions -- so
it cannot be circular with either ball-proximal compactness (BPC) or the
existing whole-team inter_line metric the way the ANOVA-by-structure check
in analysis_compactness_compare.py partly was.

Turnover event times are corrected onto the TRUE match clock exactly the
way build_ppda_vs_phase.py does it (its video_clock_offsets /
rebuild_events_on_true_clock functions) -- round 21 established the raw
event feed's startTime tracks the uncorrected frame/fps "video clock", not
periodElapsedTime -- reused here rather than re-derived, to stay consistent
with the already-published PPDA-vs-Phase companion's methodology. Turnover
lookups are done in PERIOD-LOCAL seconds (matching each frame's own
periodElapsedTime / clockS) and never cross a period boundary, avoiding the
match_t discontinuity that can occur right at a period boundary when a
period runs long on injury time.
"""
import json
import bisect
import numpy as np
import pandas as pd

from config import HOME_TEAM_ID, AWAY_TEAM_ID
from events_processing import load_events

PERIOD_START_S = {1: 0, 2: 2700, 3: 5400, 4: 6300}
TEAM_NAME = {HOME_TEAM_ID: 'Morocco', AWAY_TEAM_ID: 'Spain'}
ORGANIZED_STATES = ['High Press', 'Mid Block', 'Low Block']

# ---------------------------------------------------------------------------
# 1. True-clock offsets (reusing build_ppda_vs_phase.py's validated method)
# ---------------------------------------------------------------------------
mfp = pd.read_pickle('/home/claude/project_work/match_features_with_pressure.pkl')
mfp = mfp.drop_duplicates(subset='frameNum', keep='first')

pet = json.load(open('/home/claude/project_work/pet_by_frame.json'))
pet_by_frame = {int(k): v for k, v in pet['pet_by_frame'].items()}
period_by_frame = {int(k): v for k, v in pet['period_by_frame'].items()}
mfp['pet'] = mfp['frameNum'].map(pet_by_frame)
mfp['true_period'] = mfp['frameNum'].map(period_by_frame)
mfp = mfp.dropna(subset=['pet', 'true_period'])
mfp['true_period'] = mfp['true_period'].astype(int)
mfp['match_t'] = mfp['true_period'].map(PERIOD_START_S) + mfp['pet']

offsets = {}
for p, g in mfp.groupby('true_period'):
    diff = g['match_t'] - g['video_t']
    lo, hi = diff.min(), diff.max()
    assert abs(hi - lo) < 0.05, f'period {p} offset not constant: {lo} vs {hi}'
    offsets[int(p)] = float(diff.mean())
print('video-clock offsets (s), match_t - video_t:', offsets)

# ---------------------------------------------------------------------------
# 2. Rebuild OTB events on the TRUE clock, in PERIOD-LOCAL seconds
# ---------------------------------------------------------------------------
events = load_events()
otb_by_period = {}
for ev in events:
    ge = ev.get('gameEvents') or {}
    st = ev.get('startTime')
    if st is None or ge.get('gameEventType') != 'OTB':
        continue
    period = ge.get('period')
    if period not in offsets:
        continue
    true_t = st + offsets[period]
    local_t = true_t - PERIOD_START_S[period]
    otb_by_period.setdefault(period, []).append((local_t, ge.get('teamId')))

for p in otb_by_period:
    otb_by_period[p].sort(key=lambda r: r[0])

# Regain ("turnover in favor of") instants per (team, period): the local_t
# at the start of each possession spell for that team -- every point the
# OTB timeline's possessing team changes TO that team.
regain_times = {}
for period, rows in otb_by_period.items():
    for i in range(1, len(rows)):
        prev_team, cur_team = rows[i - 1][1], rows[i][1]
        if cur_team != prev_team and cur_team in (HOME_TEAM_ID, AWAY_TEAM_ID):
            regain_times.setdefault((cur_team, period), []).append(rows[i][0])

print("\nRegain (turnover-in-favor-of) counts by team/period:")
for (team, period), times in sorted(regain_times.items()):
    print(f"  {TEAM_NAME[team]}, period {period}: {len(times)} regains")

# ---------------------------------------------------------------------------
# 3. Merge BPC + inter_line + depth, build the defending-team series (same
#    stitching pattern as analysis_compactness_compare.py)
# ---------------------------------------------------------------------------
bpc = pd.read_pickle('/home/claude/project_work/ball_proximal_compactness.pkl')
df = mfp.merge(bpc, on='frameNum', how='inner')
df['clockS'] = df['pet']

df['defending_team_id'] = np.select(
    [df['possession_team_id'] == HOME_TEAM_ID, df['possession_team_id'] == AWAY_TEAM_ID],
    [AWAY_TEAM_ID, HOME_TEAM_ID], default=-1).astype(int)
df['defending_depth'] = np.select(
    [df['defending_team_id'] == HOME_TEAM_ID, df['defending_team_id'] == AWAY_TEAM_ID],
    [df['home_depth_smooth'], df['away_depth_smooth']], default=np.nan)
df['defending_inter_line'] = np.select(
    [df['defending_team_id'] == HOME_TEAM_ID, df['defending_team_id'] == AWAY_TEAM_ID],
    [df['home_inter_line_smooth'], df['away_inter_line_smooth']], default=np.nan)
df['defending_bpc_raw'] = np.select(
    [df['defending_team_id'] == HOME_TEAM_ID, df['defending_team_id'] == AWAY_TEAM_ID],
    [df['home_bpc'], df['away_bpc']], default=np.nan)
df['defending_structure'] = np.select(
    [df['defending_team_id'] == HOME_TEAM_ID, df['defending_team_id'] == AWAY_TEAM_ID],
    [df['home_structure'], df['away_structure']], default=None)

df = df[df['defending_team_id'].isin([HOME_TEAM_ID, AWAY_TEAM_ID])].sort_values(['true_period', 'clockS'])


def smooth_group(g, col, window_s=3.0):
    g = g.set_index(pd.to_timedelta(g['clockS'], unit='s'))
    return g[col].rolling(f'{window_s}s', min_periods=1).mean().values


out = []
for (period, team), g in df.groupby(['true_period', 'defending_team_id']):
    g = g.sort_values('clockS').copy()
    g['defending_bpc_smooth'] = smooth_group(g, 'defending_bpc_raw')
    out.append(g)
df = pd.concat(out).sort_index()

# ---------------------------------------------------------------------------
# 4. Resample to 1s bins per (period, defending team) -- de-autocorrelates
#    frame-level noise, same convention the lead-lag/PPDA-vs-phase scripts
#    already use for continuous-signal-vs-event comparisons.
# ---------------------------------------------------------------------------
rows = []
for (period, team), g in df.groupby(['true_period', 'defending_team_id']):
    g = g.sort_values('clockS').copy()
    g['bin'] = np.floor(g['clockS']).astype(int)
    agg = g.groupby('bin').agg(
        depth=('defending_depth', 'mean'),
        inter_line=('defending_inter_line', 'mean'),
        bpc=('defending_bpc_smooth', 'mean'),
        structure=('defending_structure', lambda s: s.value_counts().idxmax() if len(s) else None),
    ).reset_index()
    agg['period'] = int(period)
    agg['team_id'] = int(team)
    agg['t0'] = agg['bin'].astype(float)
    rows.append(agg)
grid = pd.concat(rows, ignore_index=True)
print(f"\n1s-bin grid: {len(grid)} defending-team-seconds")

# ---------------------------------------------------------------------------
# 5. Outcome: does a regain for that team occur within (t0, t0+K] the same
#    period? Computed for a few K horizons.
# ---------------------------------------------------------------------------
KS = (5, 8, 10)
for K in KS:
    def turnover_within(row, K=K):
        times = regain_times.get((row['team_id'], row['period']))
        if not times:
            return 0
        lo, hi = row['t0'], row['t0'] + K
        i = bisect.bisect_right(times, lo)
        return int(i < len(times) and times[i] <= hi)
    grid[f'turnover_{K}s'] = grid.apply(turnover_within, axis=1)

print("\nBase turnover-within-K rates (all defending-seconds):")
print(grid[[f'turnover_{K}s' for K in KS]].mean().round(4))

grid.to_pickle('/home/claude/project_work/bpc_turnover_grid.pkl')

# ---------------------------------------------------------------------------
# 6. Quantile-bin analysis: does tighter BPC predict a sooner turnover?
#    Run on (a) all organized-defense seconds, (b) Low-Block-depth
#    candidates only (depth<32m) -- the exact frame population any new
#    threshold would actually gate.
# ---------------------------------------------------------------------------
def quantile_report(sub, col, K, n_bins=5, label=''):
    sub = sub.dropna(subset=[col, f'turnover_{K}s'])
    if len(sub) < n_bins * 20:
        print(f"    [{label}] too few rows ({len(sub)}) for {n_bins}-bin quantile report")
        return None
    try:
        sub = sub.copy()
        sub['qbin'] = pd.qcut(sub[col], n_bins, duplicates='drop')
    except ValueError as e:
        print(f"    [{label}] qcut failed: {e}")
        return None
    rep = sub.groupby('qbin', observed=True).agg(
        n=(f'turnover_{K}s', 'size'),
        rate=(f'turnover_{K}s', 'mean'),
        mean_val=(col, 'mean'),
    )
    print(f"    [{label}] {col}, K={K}s:")
    print(rep.round(4).to_string())
    return rep


print("\n" + "=" * 78)
print("A. ORGANIZED DEFENSE (High Press / Mid Block / Low Block), all depths")
print("=" * 78)
organized = grid[grid['structure'].isin(ORGANIZED_STATES)]
print(f"n = {len(organized)} defending-seconds")
for K in KS:
    quantile_report(organized, 'bpc', K, label='BPC')
for K in KS:
    quantile_report(organized, 'inter_line', K, label='inter_line')

print("\n" + "=" * 78)
print("B. LOW-BLOCK-DEPTH CANDIDATES ONLY (depth < 32m) -- the exact frame")
print("   population the current Low Block gate operates on")
print("=" * 78)
candidates = grid[grid['depth'] < 32]
print(f"n = {len(candidates)} defending-seconds with depth<32m")
for K in KS:
    quantile_report(candidates, 'bpc', K, n_bins=4, label='BPC')
for K in KS:
    quantile_report(candidates, 'inter_line', K, n_bins=4, label='inter_line')

# ---------------------------------------------------------------------------
# 7. Does BPC add independent predictive signal beyond inter_line (and vice
#    versa)? Partial correlation within the depth<32m candidate population:
#    pcorr(turnover, X | Y) = corr(resid(turnover~Y), resid(X~Y))
# ---------------------------------------------------------------------------
def partial_corr(y, x, z):
    """corr(y,x) controlling for z, via linear-residual method."""
    mask = ~(y.isna() | x.isna() | z.isna())
    y, x, z = y[mask].values, x[mask].values, z[mask].values
    if mask.sum() < 30:
        return np.nan, int(mask.sum())
    bz_y = np.polyfit(z, y, 1)
    bz_x = np.polyfit(z, x, 1)
    resid_y = y - np.polyval(bz_y, z)
    resid_x = x - np.polyval(bz_x, z)
    r = np.corrcoef(resid_y, resid_x)[0, 1]
    return r, int(mask.sum())


print("\n" + "=" * 78)
print("C. PARTIAL CORRELATION -- does each metric predict turnover beyond the")
print("   other, within Low-Block-depth candidates (depth<32m)?")
print("=" * 78)
for K in KS:
    y = candidates[f'turnover_{K}s'].astype(float)
    r_bpc_raw = candidates[['bpc', f'turnover_{K}s']].dropna()
    r_bpc_raw_val = np.corrcoef(r_bpc_raw['bpc'], r_bpc_raw[f'turnover_{K}s'])[0, 1] if len(r_bpc_raw) > 30 else np.nan
    r_il_raw = candidates[['inter_line', f'turnover_{K}s']].dropna()
    r_il_raw_val = np.corrcoef(r_il_raw['inter_line'], r_il_raw[f'turnover_{K}s'])[0, 1] if len(r_il_raw) > 30 else np.nan

    r_bpc_given_il, n1 = partial_corr(y, candidates['bpc'], candidates['inter_line'])
    r_il_given_bpc, n2 = partial_corr(y, candidates['inter_line'], candidates['bpc'])
    print(f"  K={K:2d}s:  raw corr(turnover,BPC)={r_bpc_raw_val:+.3f}   "
          f"raw corr(turnover,inter_line)={r_il_raw_val:+.3f}")
    print(f"          corr(turnover,BPC | inter_line)={r_bpc_given_il:+.3f} (n={n1})   "
          f"corr(turnover,inter_line | BPC)={r_il_given_bpc:+.3f} (n={n2})")

candidates.to_pickle('/home/claude/project_work/bpc_turnover_candidates.pkl')
print("\nSaved bpc_turnover_grid.pkl and bpc_turnover_candidates.pkl")

"""
Idea #1 analysis: does splitting each organized-defense phase (High Press /
Mid Block / Low Block) into "engaged" (real 1v1 pressure) vs "passive"
(holding the shape/depth without actually closing anyone down) reveal
something the geometric phase alone doesn't -- per the professor's note
that a low back line isn't the same thing as a genuine low block unless
there's real engagement on the ball-side.

Step 1: validate/derive an "engaged" pressure_score threshold against PFF's
own analyst-tagged initialPressureType field (P = a defender was tagged as
applying pressure at that touch, N = not), the same ground truth already
cited in pressure.py's docstring (0.739 tagged vs 0.538 untagged) -- redoing
it here to get an actual threshold, not just a validation mean.

Step 2: using that threshold, and classification.py's own segment_phases()
(the same >=5s/gap-merged segmentation classification.py already uses to
validate PPDA-by-structure), label each organized-defense PHASE INSTANCE
(not raw frame -- that would just reproduce the flicker problem rounds
21-23 already dealt with) "engaged" or "passive" by whether a majority of
its (resolved) frames are above threshold.

Step 3: report the split's distribution and whether it's tactically
informative (PPDA by phase x engagement, not just by phase).
"""
import json
import numpy as np
import pandas as pd
from config import HOME_TEAM_ID, AWAY_TEAM_ID
from classification import segment_phases

df = pd.read_pickle('match_features_with_pressure.pkl')

# ---------- Step 1: derive threshold from PFF's tagged pressure events ----------
with open('/home/claude/wc2022/events.json') as f:
    events = json.load(f)

tagged = []
for e in events:
    it = e.get('initialTouch') or {}
    ptype = it.get('initialPressureType')
    if ptype not in ('P', 'N'):
        continue
    ge = e.get('gameEvents') or {}
    team_id = ge.get('teamId')
    t = e.get('startTime')
    if team_id is None or t is None:
        continue
    defending_team = AWAY_TEAM_ID if team_id == HOME_TEAM_ID else HOME_TEAM_ID
    tagged.append((t, defending_team, ptype))

tagged_df = pd.DataFrame(tagged, columns=['t', 'defending_team', 'ptype'])
print(f"Tagged touches: {len(tagged_df)} ({(tagged_df.ptype=='P').sum()} P, {(tagged_df.ptype=='N').sum()} N)")

# nearest-frame match within 0.5s, per defending team's own pressure_score stream
pdf = df[['video_t', 'pressing_team_id', 'pressure_score']].dropna(subset=['pressure_score']).sort_values('video_t')

def nearest_score(row):
    sub = pdf[pdf['pressing_team_id'] == row['defending_team']]
    if sub.empty:
        return np.nan
    idx = (sub['video_t'] - row['t']).abs().idxmin()
    if abs(sub.loc[idx, 'video_t'] - row['t']) > 0.5:
        return np.nan
    return sub.loc[idx, 'pressure_score']

tagged_df['score'] = tagged_df.apply(nearest_score, axis=1)
matched = tagged_df.dropna(subset=['score'])
print(f"Matched to a frame within 0.5s: {len(matched)}/{len(tagged_df)}")

p_scores = matched.loc[matched.ptype == 'P', 'score']
n_scores = matched.loc[matched.ptype == 'N', 'score']
print(f"P (tagged pressure): n={len(p_scores)}, mean={p_scores.mean():.3f}, median={p_scores.median():.3f}")
print(f"N (no pressure):      n={len(n_scores)}, mean={n_scores.mean():.3f}, median={n_scores.median():.3f}")

# Youden's J across candidate thresholds
best_j, best_t = -1, None
for thresh in np.arange(0.05, 0.96, 0.01):
    tpr = (p_scores >= thresh).mean()
    fpr = (n_scores >= thresh).mean()
    j = tpr - fpr
    if j > best_j:
        best_j, best_t = j, thresh
print(f"Best Youden's-J threshold: {best_t:.2f} (J={best_j:.3f}, TPR={((p_scores>=best_t).mean()):.3f}, FPR={((n_scores>=best_t).mean()):.3f})")


ENGAGED_THRESHOLD = 0.66  # Youden's-J-optimal split of pressure_score vs PFF's tagged pressure events (above)

# ---------- Step 2: segment-level engaged/passive labeling ----------
def label_segments(df, team_prefix, defending_team_id):
    struct_col = f'{team_prefix}_structure'
    segs = segment_phases(df, struct_col, min_duration_s=5.0, merge_gap_s=3.0)
    segs = [s for s in segs if s['state'] in ('High Press', 'Mid Block', 'Low Block')]
    press = df[['video_t', 'pressure_score', 'pressing_team_id']]
    press = press[press['pressing_team_id'] == defending_team_id].dropna(subset=['pressure_score'])
    press = press.sort_values('video_t')
    pv = press['video_t'].values
    ps = press['pressure_score'].values
    for s in segs:
        lo = np.searchsorted(pv, s['start_t'], side='left')
        hi = np.searchsorted(pv, s['end_t'], side='right')
        window = ps[lo:hi]
        s['n_resolved'] = len(window)
        s['frac_engaged'] = float((window >= ENGAGED_THRESHOLD).mean()) if len(window) else np.nan
        s['label'] = ('Engaged' if s['frac_engaged'] >= 0.5 else 'Passive/Screening') if len(window) >= 3 else 'Unresolved'
    return segs

home_segs = label_segments(df, 'home', HOME_TEAM_ID)
away_segs = label_segments(df, 'away', AWAY_TEAM_ID)

for name, segs in [('Morocco', home_segs), ('Spain', away_segs)]:
    print(f"\n=== {name}: {len(segs)} organized-defense phase instances ===")
    tab = pd.DataFrame(segs)
    print(tab.groupby(['state', 'label'])['duration_s'].agg(['count', 'sum']).round(1))
    total_by_state = tab.groupby('state')['duration_s'].sum()
    engaged_by_state = tab[tab.label == 'Engaged'].groupby('state')['duration_s'].sum()
    print("\nShare of each phase's TIME that is 'Engaged':")
    for state in ['High Press', 'Mid Block', 'Low Block']:
        tot = total_by_state.get(state, 0)
        eng = engaged_by_state.get(state, 0)
        print(f"  {state}: {eng/tot*100:.1f}% engaged ({eng:.0f}s / {tot:.0f}s)" if tot else f"  {state}: n/a")

import pickle
with open('engaged_passive_segments.pkl', 'wb') as f:
    pickle.dump({'home': home_segs, 'away': away_segs, 'threshold': ENGAGED_THRESHOLD}, f)
print("\nSaved engaged_passive_segments.pkl")

# ---------- Step 3: does the engaged/passive split matter for PPDA? ----------
from config import load_roster, load_metadata, orientation_lookup
from events_processing import load_events, build_event_signals, compute_ppda

roster, jersey_lookup, player_id_lookup = load_roster()
meta = load_metadata()
orientation = orientation_lookup(meta)
raw_events = load_events()
sig = build_event_signals(raw_events, player_id_lookup, orientation)

def ppda_by_label(segs, attacking_team_id, defending_team_id, name):
    print(f"\n{name}: PPDA by phase x engagement (pooled passes/actions, not averaged per-segment)")
    rows = []
    for state in ['High Press', 'Mid Block', 'Low Block']:
        for label in ['Engaged', 'Passive/Screening']:
            cell = [s for s in segs if s['state'] == state and s['label'] == label]
            tp, ta = 0, 0
            for s in cell:
                _, npass, nact = compute_ppda(sig['pass_events_for_ppda'], sig['action_events_for_ppda'],
                                                attacking_team_id, defending_team_id, s['start_t'], s['end_t'], use_zone=True)
                tp += npass; ta += nact
            ppda = tp / ta if ta else None
            rows.append((state, label, len(cell), tp, ta, ppda))
            ppda_str = f"{ppda:.2f}" if ppda else "n/a"
            print(f"  {state:12s} {label:20s} n_segs={len(cell):3d}  passes={tp:4d}  actions={ta:3d}  PPDA={ppda_str}")
    return rows

home_rows = ppda_by_label(home_segs, AWAY_TEAM_ID, HOME_TEAM_ID, "Morocco (defending)")
away_rows = ppda_by_label(away_segs, HOME_TEAM_ID, AWAY_TEAM_ID, "Spain (defending)")

# ---------- Step 3b: PPDA cells were too sparse (single-digit actions per cell) --
# fall back to a turnover-proximity check instead: does the ATTACKING team lose
# the ball during/soon after an Engaged defensive segment more often than a
# Passive one? Grace window matches MAX_TRANSITION_S (20s) used elsewhere in
# this project for "how long a turnover's effect should count".
GRACE_S = 20.0

def turnover_rate_by_label(segs, attacking_team_id, name):
    lost = np.array(sorted(sig['lost_ball_times'][attacking_team_id]))
    print(f"\n{name}: turnover-by-attacking-team within segment or {GRACE_S:.0f}s after, by phase x engagement")
    for state in ['High Press', 'Mid Block', 'Low Block']:
        for label in ['Engaged', 'Passive/Screening']:
            cell = [s for s in segs if s['state'] == state and s['label'] == label]
            if not cell:
                continue
            hits = 0
            for s in cell:
                lo, hi = s['start_t'], s['end_t'] + GRACE_S
                idx_lo = np.searchsorted(lost, lo, side='left')
                idx_hi = np.searchsorted(lost, hi, side='right')
                if idx_hi > idx_lo:
                    hits += 1
            rate = hits / len(cell)
            print(f"  {state:12s} {label:20s} n={len(cell):3d}  turnover-within-window rate={rate*100:.0f}%")

turnover_rate_by_label(home_segs, AWAY_TEAM_ID, "Morocco (defending; Spain is attacking team)")
turnover_rate_by_label(away_segs, HOME_TEAM_ID, "Spain (defending; Morocco is attacking team)")

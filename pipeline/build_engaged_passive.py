"""
Idea #1 from the phase-detection-definition discussion (professor's note):
a low back line isn't the same thing as a genuine low block unless there's
real 1v1 engagement on the ball-side wide player -- "real" pressure vs.
passive screening. pressure.py's time-to-intercept score already measures
engagement (not just proximity), so this doesn't need a new metric, just a
threshold and a way to attach it to the existing geometric phases without
re-introducing the frame-flicker problem rounds 21-23 already solved once.

Pipeline:
1. Reuse build_ppda_vs_phase.load_merged() for the true-match-clock dataframe
   (avoids re-introducing the video_t-vs-true-clock bug already fixed there).
2. Derive an "engaged" pressure_score threshold from PFF's own analyst-tagged
   initialPressureType field (P/N) via Youden's J -- not eyeballed. See
   analysis_engaged_passive.py for the exploratory version of this; this
   script is the clean, re-runnable build.
3. Segment each team's organized-defense structure into phase instances
   (>=5s, gap-merged <=3s -- the same convention classification.py already
   uses for its own PPDA-by-structure validation), then label each instance
   Engaged / Passive by majority vote of its own (resolved) frames.
4. Write engaged_passive_data.json for the companion artifact.
"""
from config import EVENTS_PATH
import json
import numpy as np
import pandas as pd
from config import HOME_TEAM_ID, AWAY_TEAM_ID
from build_ppda_vs_phase import load_merged, period_bounds, PERIOD_START_S, PERIOD_LABEL, TEAM_NAME

MIN_DURATION_S = 5.0
MERGE_GAP_S = 3.0
MIN_RESOLVED_FRAMES = 3


def derive_threshold():
    with open(EVENTS_PATH) as f:
        events = json.load(f)
    df = pd.read_pickle('match_features_with_pressure.pkl')
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

    pdf = df[['video_t', 'pressing_team_id', 'pressure_score']].dropna(subset=['pressure_score']).sort_values('video_t')

    def nearest_score(row):
        sub = pdf[pdf['pressing_team_id'] == row['defending_team']]
        if sub.empty:
            return np.nan
        idx = (sub['video_t'] - row['t']).abs().idxmin()
        return sub.loc[idx, 'pressure_score'] if abs(sub.loc[idx, 'video_t'] - row['t']) <= 0.5 else np.nan

    tagged_df['score'] = tagged_df.apply(nearest_score, axis=1)
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
        'threshold': round(float(best_t), 2),
        'auc': round(float(auc), 3),
        'j': round(float(best_j), 3),
        'tpr': round(float((p >= best_t).mean()), 3),
        'fpr': round(float((n >= best_t).mean()), 3),
        'p_n': int(len(p)), 'n_n': int(len(n)),
        'p_mean': round(float(p.mean()), 3), 'n_mean': round(float(n.mean()), 3),
    }


def segment_phases_true_clock(df, state_col, min_duration_s=MIN_DURATION_S, merge_gap_s=MERGE_GAP_S):
    sub = df[['match_t', state_col]].dropna(subset=[state_col]).reset_index(drop=True)
    if sub.empty:
        return []
    raw_runs = []
    cur_state, cur_start, prev_t = sub.loc[0, state_col], sub.loc[0, 'match_t'], sub.loc[0, 'match_t']
    for i in range(1, len(sub)):
        t, state = sub.loc[i, 'match_t'], sub.loc[i, state_col]
        if state != cur_state:
            raw_runs.append({'state': cur_state, 'start_t': cur_start, 'end_t': prev_t})
            cur_state, cur_start = state, t
        prev_t = t
    raw_runs.append({'state': cur_state, 'start_t': cur_start, 'end_t': prev_t})

    merged = [raw_runs[0]]
    for run in raw_runs[1:]:
        last = merged[-1]
        if run['state'] == last['state'] and (run['start_t'] - last['end_t']) <= merge_gap_s:
            last['end_t'] = run['end_t']
        else:
            merged.append(run)
    for r in merged:
        r['duration_s'] = r['end_t'] - r['start_t']
    return [r for r in merged if r['duration_s'] >= min_duration_s]


def label_segments(df, struct_col, defending_team_id, threshold):
    segs = segment_phases_true_clock(df, struct_col)
    segs = [s for s in segs if s['state'] in ('High Press', 'Mid Block', 'Low Block')]
    press = df[df['pressing_team_id'] == defending_team_id][['match_t', 'pressure_score']].dropna()
    press = press.sort_values('match_t')
    mv, ps = press['match_t'].values, press['pressure_score'].values
    for s in segs:
        lo = np.searchsorted(mv, s['start_t'], side='left')
        hi = np.searchsorted(mv, s['end_t'], side='right')
        window = ps[lo:hi]
        s['n_resolved'] = int(len(window))
        s['frac_engaged'] = round(float((window >= threshold).mean()), 3) if len(window) else None
        if len(window) >= MIN_RESOLVED_FRAMES:
            s['label'] = 'Engaged' if s['frac_engaged'] >= 0.5 else 'Passive'
        else:
            s['label'] = 'Unresolved'
    return segs


def summarize(segs):
    out = {}
    for state in ['High Press', 'Mid Block', 'Low Block']:
        cell = [s for s in segs if s['state'] == state]
        total = sum(s['duration_s'] for s in cell)
        engaged = sum(s['duration_s'] for s in cell if s['label'] == 'Engaged')
        passive = sum(s['duration_s'] for s in cell if s['label'] == 'Passive')
        unresolved = sum(s['duration_s'] for s in cell if s['label'] == 'Unresolved')
        n_engaged = sum(1 for s in cell if s['label'] == 'Engaged')
        n_passive = sum(1 for s in cell if s['label'] == 'Passive')
        out[state] = {
            'total_s': round(total, 1), 'engaged_s': round(engaged, 1), 'passive_s': round(passive, 1),
            'unresolved_s': round(unresolved, 1),
            'engaged_pct': round(engaged / total * 100, 1) if total else None,
            'passive_pct': round(passive / total * 100, 1) if total else None,
            'n_engaged': n_engaged, 'n_passive': n_passive, 'n_segments': len(cell),
        }
    return out


def main():
    df = load_merged()
    bounds = period_bounds(df)
    thresh_info = derive_threshold()
    threshold = thresh_info['threshold']
    print('Threshold info:', thresh_info)

    home_segs = label_segments(df, 'home_structure', HOME_TEAM_ID, threshold)
    away_segs = label_segments(df, 'away_structure', AWAY_TEAM_ID, threshold)

    home_summary = summarize(home_segs)
    away_summary = summarize(away_segs)

    def seg_out(segs):
        return [{'state': s['state'], 't0': round(s['start_t'], 1), 't1': round(s['end_t'], 1),
                  'dur': round(s['duration_s'], 1), 'label': s['label'], 'frac_engaged': s['frac_engaged'],
                  'n_resolved': s['n_resolved']} for s in segs]

    periods_out = [{'period': p, 't0': round(b[0], 1), 't1': round(b[1], 1), 'label': PERIOD_LABEL[p]}
                   for p, b in sorted(bounds.items())]
    t_max = max(b[1] for b in bounds.values())

    out = {
        'match': {'home': TEAM_NAME[HOME_TEAM_ID], 'away': TEAM_NAME[AWAY_TEAM_ID],
                  'home_id': HOME_TEAM_ID, 'away_id': AWAY_TEAM_ID},
        't_max_s': round(t_max, 1),
        'periods': periods_out,
        'threshold_info': thresh_info,
        'segments': {'home': seg_out(home_segs), 'away': seg_out(away_segs)},
        'summary': {TEAM_NAME[HOME_TEAM_ID]: home_summary, TEAM_NAME[AWAY_TEAM_ID]: away_summary},
    }
    with open('engaged_passive_data.json', 'w') as f:
        json.dump(out, f)
    print('Wrote engaged_passive_data.json')
    print(json.dumps(home_summary, indent=2))
    print(json.dumps(away_summary, indent=2))


if __name__ == '__main__':
    main()

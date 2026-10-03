"""
Builds the data payload for the new "PPDA vs. Phase Detector" full-match
comparison chart:

  Panel A - full-match defensive-phase timeline (Morocco / Spain), aggregated
            to the SAME 300s buckets as PPDA (dominant/highest-time-share
            structure per bucket from match_features_with_pressure.pkl's
            home_structure/away_structure columns), on the TRUE match clock
            (periodElapsedTime + PERIOD_START_S), not the frame/fps-derived
            `video_t` column. Matching PPDA's own bucket grid -- rather than a
            finer continuous strip -- means every phase block sits directly
            under the PPDA point it's paired with, at the same resolution
            PPDA itself is legible at (see note in bucket_dominant_phase()).

  Panel B - full-match PPDA (5-min buckets), reusing ppda_timeline.json's
            already-computed values but re-expressed on that SAME true match
            clock. ppda_timeline.json's own time axis turns out to be on the
            frame/fps-style "video clock" too (it matches video_t almost
            exactly, and drifts up to ~12.5 minutes behind the true match
            clock by extra time) -- corrected here with a per-period additive
            offset derived empirically from the tracking file.

  Panel C - PPDA-by-dominant-defensive-phase breakdown per team (5-min
            buckets), directly re-checking the methodology doc's open finding
            about this cross-tab looking inverted for Morocco under the old
            whole-team-depth metric. Uses the exact same per-bucket dominant
            state as Panel A, so the chart and the table are never showing
            two different aggregations of the same thing.

Writes ppda_vs_phase_data.json for the HTML chart to consume.
"""
import json
import pandas as pd

from config import load_roster, load_metadata, orientation_lookup, HOME_TEAM_ID, AWAY_TEAM_ID
from events_processing import load_events, _in_press_zone

PERIOD_START_S = {1: 0, 2: 2700, 3: 5400, 4: 6300}
PERIOD_LABEL = {1: '1st Half', 2: '2nd Half', 3: 'Extra Time 1', 4: 'Extra Time 2'}
BUCKET_S = 300.0

STATES = ['In Possession', 'Defensive Transition', 'High Press', 'Mid Block', 'Low Block']

TEAM_NAME = {374: 'Morocco', 52: 'Spain'}


def fmt_clock(t_s):
    m, s = divmod(int(round(t_s)), 60)
    return f'{m}:{s:02d}'


def load_merged():
    with open('pet_by_frame.json') as f:
        d = json.load(f)
    pet_by_frame = {int(k): v for k, v in d['pet_by_frame'].items()}

    df = pd.read_pickle('match_features_with_pressure.pkl')
    df['pet'] = df['frameNum'].map(pet_by_frame)
    df['match_t'] = df['period'].map(PERIOD_START_S) + df['pet']
    df = df.sort_values('match_t').reset_index(drop=True)
    return df


def period_bounds(df):
    bounds = {}
    for p, g in df.groupby('period'):
        bounds[p] = (float(g['match_t'].min()), float(g['match_t'].max()))
    return bounds


ORGANIZED_STATES = ['High Press', 'Mid Block', 'Low Block']


def bucket_dominant_phase(df, col, n_buckets, exclude=None):
    """Aggregate the (very high-frequency, frame-level) structure column to
    the dominant (highest time-share) state per BUCKET_S=300s bucket -- the
    exact same 0-anchored grid PPDA is computed on -- rather than rendering
    frame-level classification directly or smoothing it to some other,
    PPDA-unrelated resolution.

    Raw per-frame home_structure/away_structure flickers constantly -- over
    97% of raw same-state runs last under 2 seconds (median run length is a
    single frame). That's expected: classify_structure() thresholds smoothed
    depth/inter-line at fixed meter cutoffs (40m / 32m / 20m), and a team's
    line often hovers within noise distance of a cutoff for long stretches,
    so the *raw* label can toggle every frame without the underlying shape
    actually changing -- and 'In Possession' vs. everything else follows raw
    per-frame possession_team_id, which itself flips briefly on contested or
    loose balls. None of that is a bug in the frame-level classifier (the
    per-chapter animations show it moment-to-moment, where a viewer reads the
    live shape rather than the label), but rendering it directly as a
    130-minute strip would just be visual noise, and a display-only smoothing
    window picked independently of PPDA (an earlier version of this script
    used a 5-second vote) draws a phase chart and a PPDA chart at two
    unrelated resolutions glued together, not one apples-to-apples
    comparison. Matching PPDA's own 300s bucket grid instead means every
    phase block is literally the same time window as the PPDA point above
    it -- the two panels are two views of the identical bucketing, not a fine
    continuous trace next to a coarse scatter.

    `exclude`, if given, drops those states from the vote entirely before
    computing the per-bucket mode -- for this chart, 'In Possession' and
    'Defensive Transition' are excluded (see ORGANIZED_STATES): a team is on
    the ball most of any given 5 minutes, so leaving it in the vote lets
    'In Possession' win the bucket's label almost by default and bury the
    organized-defense phase (High Press/Mid/Low Block) that's actually the
    point of this comparison, even in buckets where a real, sustained press
    or low block happened for a meaningful share of the window. A bucket
    with zero non-excluded frames (the team was on the ball or transitioning
    for the whole window) is left out of the result entirely -- there is no
    defensive phase to report for it, not a zero-th one -- which shows up as
    a genuine gap in the rendered strip rather than a misleading label.

    Returns {bucket_idx: state} for every bucket that has qualifying frame
    data."""
    if exclude:
        df = df[~df[col].isin(exclude)]
    bucket_idx = (df['match_t'] // BUCKET_S).astype(int).clip(upper=n_buckets - 1)
    tmp = pd.DataFrame({'bucket': bucket_idx, 'state': df[col]})
    dominant = {}
    for b, g in tmp.groupby('bucket'):
        counts = g['state'].value_counts()
        dominant[int(b)] = counts.idxmax()
    return dominant


def video_clock_offsets(df):
    """Empirical per-period constant: match_t - video_t (both continuous,
    whole-match clocks; the difference is exactly constant within a period)."""
    offsets = {}
    for p, g in df.groupby('period'):
        diff = (g['match_t'] - g['video_t'])
        lo, hi = diff.min(), diff.max()
        assert abs(hi - lo) < 0.05, f'period {p} offset not constant: {lo} vs {hi}'
        offsets[p] = float(diff.mean())
    return offsets


def rebuild_events_on_true_clock(offsets):
    """Re-derive pass/action/OTB events straight from the raw event feed, but
    with each event's time corrected onto the TRUE match clock using its own
    `period` tag (events carry period directly, unlike the pre-built
    pass_events_for_ppda/action_events_for_ppda lists in events_processing.py,
    which drop it). This avoids re-bucketing an already-bucketed PPDA series
    onto a different grid -- PPDA is recomputed from scratch on true-clock,
    0-anchored 300s buckets, so it lines up exactly with the phase-detector
    buckets below."""
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    events = load_events()

    pass_events, action_events, otb_rows = [], [], []
    for ev in events:
        ge = ev.get('gameEvents') or {}
        pe = ev.get('possessionEvents') or {}
        st = ev.get('startTime')
        if st is None:
            continue
        period = ge.get('period')
        if period not in offsets:
            continue
        true_t = st + offsets[period]
        home_positive = orientation.get(period)

        if ge.get('gameEventType') == 'OTB':
            otb_rows.append((true_t, ge.get('teamId')))

        et = pe.get('possessionEventType')
        ball = ev.get('ball') or []
        if ball and ball[0].get('x') is not None and home_positive is not None:
            bx = ball[0]['x']
            if et == 'PA' and pe.get('passOutcomeType') == 'C':
                pass_events.append({'time_s': true_t, 'team_id': ge.get('teamId'),
                                     'x': bx, 'home_positive': home_positive})
            if et == 'CH':
                challenger_id = pe.get('challengerPlayerId')
                info = player_id_lookup.get(str(challenger_id)) if challenger_id is not None else None
                if info is not None:
                    action_events.append({'time_s': true_t, 'team_id': info[2], 'x': bx,
                                           'home_positive': home_positive})
            if et == 'CL':
                clearer_id = pe.get('clearerPlayerId')
                info = player_id_lookup.get(str(clearer_id)) if clearer_id is not None else None
                if info is not None:
                    action_events.append({'time_s': true_t, 'team_id': info[2], 'x': bx,
                                           'home_positive': home_positive})

    otb_rows.sort(key=lambda r: r[0])
    return pass_events, action_events, otb_rows


def compute_ppda_true_clock(pass_events, action_events, t_max):
    n_buckets = int(t_max // BUCKET_S) + 1
    points = []
    for team_id, opp_id in [(HOME_TEAM_ID, AWAY_TEAM_ID), (AWAY_TEAM_ID, HOME_TEAM_ID)]:
        for b in range(n_buckets):
            lo, hi = b * BUCKET_S, (b + 1) * BUCKET_S
            passes = [e for e in pass_events if e['team_id'] == opp_id
                      and lo <= e['time_s'] < hi and _in_press_zone(e, team_id)]
            actions = [e for e in action_events if e['team_id'] == team_id
                       and lo <= e['time_s'] < hi and _in_press_zone(e, team_id)]
            if not actions:
                continue
            ppda = len(passes) / len(actions)
            points.append({
                'team_id': team_id, 'team': TEAM_NAME[team_id],
                't_lo': round(lo, 2), 't_hi': round(hi, 2),
                't_mid_min': round((lo + hi) / 2 / 60.0, 2),
                'ppda': round(ppda, 2), 'opp_passes': len(passes), 'def_actions': len(actions),
            })
    return points


def compute_turnovers_true_clock(otb_rows, pass_events):
    spells = []
    i, n = 0, len(otb_rows)
    while i < n:
        team = otb_rows[i][1]
        start = otb_rows[i][0]
        j = i + 1
        while j < n and otb_rows[j][1] == team:
            j += 1
        end = otb_rows[j][0] if j < n else None
        spells.append((team, start, end))
        i = j

    turnovers = []
    for team, start, end in spells:
        if end is None or team not in (HOME_TEAM_ID, AWAY_TEAM_ID):
            continue
        n_passes = sum(1 for p in pass_events if p['team_id'] == team and start <= p['time_s'] < end)
        turnovers.append({'team_id': team, 'team': TEAM_NAME[team], 'time_s': round(end, 2),
                           't_min': round(end / 60.0, 2), 'passes_before': n_passes})
    return turnovers


def build_crosstab(dominant, ppda_points):
    """For each team, for each 5-min true-match-clock bucket, pair that
    bucket's dominant (highest time-share) defensive structure -- the exact
    same values plotted in Panel A -- with that bucket's PPDA value (as the
    DEFENDING team) if one exists."""
    # index ppda points by (team_id, bucket_idx)
    ppda_by_bucket = {}
    for pt in ppda_points:
        b = int(round(pt['t_lo'] / BUCKET_S))
        ppda_by_bucket[(pt['team_id'], b)] = pt['ppda']

    rows = []
    crosstab = {374: {s: [] for s in ORGANIZED_STATES}, 52: {s: [] for s in ORGANIZED_STATES}}
    for team_id in (374, 52):
        for b, state in dominant[team_id].items():
            ppda = ppda_by_bucket.get((team_id, b))
            if ppda is None:
                continue
            crosstab[team_id][state].append(ppda)
            rows.append({'team_id': team_id, 'bucket': b, 't_mid_min': round((b * BUCKET_S + BUCKET_S / 2) / 60, 2),
                         'state': state, 'ppda': ppda})

    summary = {}
    for team_id in (374, 52):
        summary[TEAM_NAME[team_id]] = {}
        for s in ORGANIZED_STATES:
            vals = crosstab[team_id][s]
            if vals:
                summary[TEAM_NAME[team_id]][s] = {
                    'n': len(vals),
                    'mean_ppda': round(sum(vals) / len(vals), 2),
                    'min_ppda': round(min(vals), 2),
                    'max_ppda': round(max(vals), 2),
                }
            else:
                summary[TEAM_NAME[team_id]][s] = {'n': 0, 'mean_ppda': None, 'min_ppda': None, 'max_ppda': None}

    return rows, summary


def main():
    df = load_merged()
    bounds = period_bounds(df)
    offsets = video_clock_offsets(df)
    print('period true-clock bounds (s):', bounds)
    print('video-clock offsets (s), match_t - video_t:', offsets)

    pass_events, action_events, otb_rows = rebuild_events_on_true_clock(offsets)
    t_max_events = max([e['time_s'] for e in pass_events + action_events] + [r[0] for r in otb_rows])
    t_max_true = max(df['match_t'].max(), t_max_events)
    n_buckets = int(t_max_true // BUCKET_S) + 1

    EXCLUDE = {'In Possession', 'Defensive Transition'}
    home_dominant = bucket_dominant_phase(df, 'home_structure', n_buckets, exclude=EXCLUDE)
    away_dominant = bucket_dominant_phase(df, 'away_structure', n_buckets, exclude=EXCLUDE)
    print(f'home organized-defense buckets: {len(home_dominant)}, away: {len(away_dominant)} (of {n_buckets} total)')

    def to_bucket_list(dominant):
        return [{'bucket': b, 't0': round(b * BUCKET_S, 2), 't1': round((b + 1) * BUCKET_S, 2), 'state': s}
                for b, s in sorted(dominant.items())]

    phase_buckets = {'home': to_bucket_list(home_dominant), 'away': to_bucket_list(away_dominant)}

    ppda_points = compute_ppda_true_clock(pass_events, action_events, t_max_true)
    ppda_turnovers = compute_turnovers_true_clock(otb_rows, pass_events)
    print(f'ppda points: {len(ppda_points)}, turnovers: {len(ppda_turnovers)}')

    dominant = {374: home_dominant, 52: away_dominant}
    crosstab_rows, crosstab_summary = build_crosstab(dominant, ppda_points)
    print(json.dumps(crosstab_summary, indent=2))

    periods_out = []
    for p in (1, 2, 3, 4):
        lo, hi = bounds[p]
        periods_out.append({'period': p, 'label': PERIOD_LABEL[p],
                             't_lo': round(lo, 2), 't_hi': round(hi, 2),
                             'clock_lo': fmt_clock(lo), 'clock_hi': fmt_clock(hi)})

    t_max_overall = max(df['match_t'].max(), max(p['t_hi'] for p in ppda_points))

    out = {
        'match': {'home': 'Morocco', 'away': 'Spain', 'home_id': 374, 'away_id': 52,
                   'competition': 'FIFA World Cup 2022 · Round of 16 · Education City Stadium'},
        't_max_s': round(float(t_max_overall), 2),
        'periods': periods_out,
        'bucket_s': BUCKET_S,
        'phase_buckets': phase_buckets,
        'ppda_points': ppda_points,
        'ppda_turnovers': ppda_turnovers,
        'crosstab_rows': crosstab_rows,
        'crosstab_summary': crosstab_summary,
        'states': ORGANIZED_STATES,
        'video_clock_offsets_s': offsets,
    }
    with open('ppda_vs_phase_data.json', 'w') as f:
        json.dump(out, f)
    import os
    print('wrote ppda_vs_phase_data.json', os.path.getsize('ppda_vs_phase_data.json'), 'bytes')


if __name__ == '__main__':
    main()

"""
Within Organized Defense, classify High Press / Mid Block / Low Block using
fixed, literature-anchored thresholds on smoothed team depth (primary axis),
with inter-line compactness as a secondary gate specific to Low Block.

Why fixed thresholds rather than per-match k-means (as the reference notebook
did): the goal is a system that generalizes across matches/teams, not a
description of one match's own relative distribution. Per-match clustering
also proved unstable in the reference notebook (cluster index identity isn't
stable across re-fits, requiring ad hoc relabeling every run). Fixed
thresholds are simpler, reproducible, and directly interpretable.

Thresholds were originally anchored to practitioner benchmark ranges
compiled during this project's literature review (defensive line height, in
meters from own goal): High Press ~35-45m, Mid Block ~25-35m, Low Block
~18-28m, taking the non-overlapping midpoints (35m / 27m). The FIFA
"Enhanced Football Intelligence" documentation is the one source that
explicitly ties Low Block to LOW PRESSING/ENGAGEMENT in addition to depth,
not depth alone -- so a team that is deep on average but not actually
compact (stretched, disorganized) is graded as Mid Block rather than Low
Block. The inter-line compactness gate below (20m) is set near the
empirical median inter-line spread for this match's sub-Mid-Block-depth
frames, i.e. it splits genuinely compact deep defending from
deep-but-stretched defending roughly in half rather than being an arbitrary
round number.

RECALIBRATION (v2): the 35m/27m cutoffs badly over-called High Press on this
match -- checked directly against full-match organized-defense time, Morocco
(the team known for sitting off: PPDA 18.44, the passive end of this
match's PPDA split) spent 44.3% of its organized-defense time in "High
Press", more than Mid Block (37.9%) or Low Block (17.8%). The 35m line sat
almost exactly at Morocco's own median defensive-line depth (33.8m), so
roughly half of their ordinary, unremarkable defending -- not just genuine
committed pressing -- was tipping over the line. Per explicit user
direction, both depth cutoffs were raised by 5m (40m / 32m); the
compactness gate was left as-is. This keeps the fixed, literature-anchored,
cross-match-comparable approach rather than switching to per-team-relative
thresholds -- it's a recalibration of where the same absolute-meter bands
sit, not a change in kind.
"""
import pandas as pd

HIGH_PRESS_MIN_DEPTH_M = 40.0
MID_BLOCK_MIN_DEPTH_M = 32.0
LOW_BLOCK_COMPACTNESS_MAX_M = 20.0  # inter-line gate for genuine Low Block (unchanged)


def classify_structure(depth, inter_line):
    if pd.isna(depth):
        return None
    if depth >= HIGH_PRESS_MIN_DEPTH_M:
        return 'High Press'
    if depth >= MID_BLOCK_MIN_DEPTH_M:
        return 'Mid Block'
    # depth < 27m: genuine Low Block requires tight compactness too
    if pd.notna(inter_line) and inter_line <= LOW_BLOCK_COMPACTNESS_MAX_M:
        return 'Low Block'
    return 'Mid Block'  # deep but stretched -- not a genuine low block siege


def add_defensive_structure(df, team_prefix):
    """team_prefix: 'home' or 'away'. Adds `{prefix}_structure` column, defined
    only where `{prefix}_phase == 'Organized Defense'`; elsewhere copies the phase
    label through (In Possession / Defensive Transition) so a single column
    gives the full category set the user asked for: Transition / High Press /
    Mid Block / Low Block / In Possession."""
    phase_col = f'{team_prefix}_phase'
    depth_col = f'{team_prefix}_depth_smooth'
    il_col = f'{team_prefix}_inter_line_smooth'
    out_col = f'{team_prefix}_structure'

    def _row(r):
        if r[phase_col] != 'Organized Defense':
            return r[phase_col]
        return classify_structure(r[depth_col], r[il_col])

    df[out_col] = df.apply(_row, axis=1)
    return df


def segment_phases(df, state_col, min_duration_s=5.0, merge_gap_s=3.0):
    """Contiguous-run segmentation with short-gap merging, rewritten cleanly.
    Returns a list of {state, start_t, end_t, duration_s} for runs of the same
    label, after merging gaps of <= merge_gap_s between same-label runs (a
    single brief interruption shouldn't fragment one sustained phase), then
    dropping runs shorter than min_duration_s."""
    sub = df[['video_t', state_col]].dropna(subset=[state_col]).reset_index(drop=True)
    if sub.empty:
        return []

    raw_runs = []
    cur_state = sub.loc[0, state_col]
    cur_start = sub.loc[0, 'video_t']
    prev_t = cur_start
    for i in range(1, len(sub)):
        t, state = sub.loc[i, 'video_t'], sub.loc[i, state_col]
        if state != cur_state:
            raw_runs.append({'state': cur_state, 'start_t': cur_start, 'end_t': prev_t})
            cur_state, cur_start = state, t
        prev_t = t
    raw_runs.append({'state': cur_state, 'start_t': cur_start, 'end_t': prev_t})

    # merge short gaps between same-label runs
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


if __name__ == '__main__':
    from config import pkl_path
    df = pd.read_pickle(pkl_path('match_features_phased'))
    df = add_defensive_structure(df, 'home')
    df = add_defensive_structure(df, 'away')

    from config import HOME_TEAM_NAME, AWAY_TEAM_NAME
    print(f"{HOME_TEAM_NAME} defensive structure distribution:")
    print(df['home_structure'].value_counts(dropna=False))
    print()
    print(f"{AWAY_TEAM_NAME} defensive structure distribution:")
    print(df['away_structure'].value_counts(dropna=False))

    home_phases = segment_phases(df, 'home_structure')
    away_phases = segment_phases(df, 'away_structure')
    print(f"\n{HOME_TEAM_NAME}: {len(home_phases)} phases (>=5s, gap-merged)")
    print(f"{AWAY_TEAM_NAME}: {len(away_phases)} phases (>=5s, gap-merged)")

    # validate against PPDA per structural state, like the reference notebook did
    from config import load_roster, load_metadata, orientation_lookup, HOME_TEAM_ID, AWAY_TEAM_ID
    from events_processing import load_events, build_event_signals, compute_ppda

    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    events = load_events()
    sig = build_event_signals(events, player_id_lookup, orientation)

    print(f"\n{HOME_TEAM_NAME} pooled PPDA by structural state (should be: High Press < Mid < Low, i.e. lower PPDA = more aggressive):")
    for name in ['High Press', 'Mid Block', 'Low Block']:
        phases = [p for p in home_phases if p['state'] == name]
        tp, ta = 0, 0
        for p in phases:
            _, np_, na_ = compute_ppda(sig['pass_events_for_ppda'], sig['action_events_for_ppda'],
                                        AWAY_TEAM_ID, HOME_TEAM_ID, p['start_t'], p['end_t'], use_zone=False)
            tp += np_; ta += na_
        ppda = tp / ta if ta else None
        print(f"  {name}: PPDA={ppda:.2f} ({tp} passes / {ta} actions, {len(phases)} phases, "
              f"{sum(p['duration_s'] for p in phases)/60:.1f} min)" if ppda else f"  {name}: N/A ({len(phases)} phases)")

    print(f"\n{AWAY_TEAM_NAME} pooled PPDA by structural state:")
    for name in ['High Press', 'Mid Block', 'Low Block']:
        phases = [p for p in away_phases if p['state'] == name]
        tp, ta = 0, 0
        for p in phases:
            _, np_, na_ = compute_ppda(sig['pass_events_for_ppda'], sig['action_events_for_ppda'],
                                        HOME_TEAM_ID, AWAY_TEAM_ID, p['start_t'], p['end_t'], use_zone=False)
            tp += np_; ta += na_
        ppda = tp / ta if ta else None
        print(f"  {name}: PPDA={ppda:.2f} ({tp} passes / {ta} actions, {len(phases)} phases, "
              f"{sum(p['duration_s'] for p in phases)/60:.1f} min)" if ppda else f"  {name}: N/A ({len(phases)} phases)")

    out = pkl_path('match_features_classified')
    df.to_pickle(out)
    print(f"\nSaved {out}")

"""
PPDA-over-time + passes-before-turnover data export for the PPDA scatterplot.

Computes, for fixed time buckets across the FULL match (all periods incl.
extra time), each team's PPDA (opponent passes allowed per defensive action
in the press zone, x >= 35m from the defending team's own goal) as the
DEFENDING side. Also derives, for every individual turnover (loss of
possession), how many of that team's completed passes were strung together
in the possession spell immediately before it.

Output: ppda_timeline.json, consumed by the HTML chart.
"""
import json
from config import load_roster, load_metadata, orientation_lookup, HOME_TEAM_ID, AWAY_TEAM_ID
from events_processing import load_events, build_event_signals, compute_ppda

BUCKET_S = 300.0  # 5-minute buckets

TEAM_NAME = {HOME_TEAM_ID: 'Morocco', AWAY_TEAM_ID: 'Spain'}


def build_buckets(sig):
    all_times = [e['time_s'] for e in sig['pass_events_for_ppda']] + \
                [e['time_s'] for e in sig['action_events_for_ppda']]
    t_min, t_max = min(all_times), max(all_times)
    n_buckets = int(t_max // BUCKET_S) + 1

    points = []
    for team_id, opp_id in [(HOME_TEAM_ID, AWAY_TEAM_ID), (AWAY_TEAM_ID, HOME_TEAM_ID)]:
        for b in range(n_buckets):
            lo = b * BUCKET_S
            hi = lo + BUCKET_S
            ppda, n_passes, n_actions = compute_ppda(
                sig['pass_events_for_ppda'], sig['action_events_for_ppda'],
                attacking_team_id=opp_id, defending_team_id=team_id,
                time_lo=lo, time_hi=hi, use_zone=True)
            if ppda is None:
                continue
            points.append({
                'team_id': team_id,
                'team': TEAM_NAME[team_id],
                't_lo': lo, 't_hi': hi,
                't_mid_min': round((lo + hi) / 2 / 60.0, 2),
                'ppda': round(ppda, 2),
                'opp_passes': n_passes,
                'def_actions': n_actions,
            })
    return points, t_min, t_max


def build_turnovers(sig):
    """Every possession spell that ends in a loss: passes completed during
    that spell, keyed to the turnover moment (when the opponent's first OTB
    event follows)."""
    otb_times = sig['otb_times']
    otb_teams = sig['otb_teams']
    passes = sig['pass_events_for_ppda']

    # group consecutive OTB rows into spells
    spells = []  # (team_id, start_time, end_time_exclusive)
    i = 0
    n = len(otb_times)
    while i < n:
        team = otb_teams[i]
        start = otb_times[i]
        j = i + 1
        while j < n and otb_teams[j] == team:
            j += 1
        end = otb_times[j] if j < n else None  # None = final spell, no turnover
        spells.append((team, start, end))
        i = j

    turnovers = []
    for team, start, end in spells:
        if end is None or team not in (HOME_TEAM_ID, AWAY_TEAM_ID):
            continue
        n_passes = sum(1 for p in passes if p['team_id'] == team and start <= p['time_s'] < end)
        turnovers.append({
            'team_id': team,
            'team': TEAM_NAME[team],
            'time_s': end,
            't_min': round(end / 60.0, 2),
            'passes_before': n_passes,
        })
    return turnovers


def main():
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    events = load_events()
    sig = build_event_signals(events, player_id_lookup, orientation)

    points, t_min, t_max = build_buckets(sig)
    turnovers = build_turnovers(sig)

    print(f"Bucket points: {len(points)}  (bucket={BUCKET_S}s, t_min={t_min:.1f}s, t_max={t_max:.1f}s)")
    print(f"Turnovers: {len(turnovers)}  "
          f"(Morocco={sum(1 for t in turnovers if t['team_id']==HOME_TEAM_ID)}, "
          f"Spain={sum(1 for t in turnovers if t['team_id']==AWAY_TEAM_ID)})")
    ppda_vals = [p['ppda'] for p in points]
    print(f"PPDA range: {min(ppda_vals):.2f} - {max(ppda_vals):.2f}")
    passes_vals = [t['passes_before'] for t in turnovers]
    print(f"Passes-before-turnover range: {min(passes_vals)} - {max(passes_vals)}, "
          f"mean={sum(passes_vals)/len(passes_vals):.2f}")

    out = {
        'bucket_s': BUCKET_S,
        't_min_s': t_min,
        't_max_s': t_max,
        'teams': {str(HOME_TEAM_ID): 'Morocco', str(AWAY_TEAM_ID): 'Spain'},
        'points': points,
        'turnovers': turnovers,
        # period boundaries (approx, for axis annotation), from prior diagnostics
        'periods': [
            {'period': 1, 'label': '1st Half', 't_lo_min': 0, 't_hi_min': round(2800.6/60, 1)},
            {'period': 2, 'label': '2nd Half', 't_lo_min': round(2854.8/60, 1), 't_hi_min': round(5866.6/60, 1)},
            {'period': 3, 'label': 'ET 1', 't_lo_min': round(5924/60, 1), 't_hi_min': round(6903.7/60, 1)},
            {'period': 4, 'label': 'ET 2', 't_lo_min': round(7053.3/60, 1), 't_hi_min': round(8581.5/60, 1)},
        ],
    }
    with open('ppda_timeline.json', 'w') as f:
        json.dump(out, f)
    print("Wrote ppda_timeline.json")


if __name__ == '__main__':
    main()

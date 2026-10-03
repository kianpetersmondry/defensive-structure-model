"""
Clean rebuild of event-derived signals: dead-ball masks, possession timeline,
lost-ball times per team, PPDA raw events, shot events, ball-carrier timeline.

Rewritten from scratch (not ported) from the reference Colab notebook, to avoid
carrying over its bugs / hidden cross-notebook state dependencies.
"""
import json
import bisect
from config import (EVENTS_PATH, HOME_TEAM_ID, AWAY_TEAM_ID,
                     DEADBALL_SETPIECE_TYPES, MASK_BEFORE_S, MASK_AFTER_S)


def load_events():
    with open(EVENTS_PATH) as f:
        return json.load(f)


def build_event_signals(events, player_id_lookup, orientation):
    """
    Single pass over the event list. Returns a dict of derived signals:
      - mask_windows: list of (lo, hi) dead-ball time windows
      - otb_times / otb_teams: sorted possession timeline (OTB = on-the-ball event)
      - carrier_times / carrier_events: sorted ball-carrier timeline (player-level)
      - pass_events_for_ppda: completed passes, attributed to attacking team
      - action_events_for_ppda: challenges + clearances, attributed to the ACTING
        player's team (not gameEvents.teamId, which is the team ON the ball)
      - shot_events
      - lost_ball_times: {team_id: sorted list of times team_id lost the ball}
    """
    mask_windows = []
    otb_rows = []          # (time_s, team_id)
    carrier_events = []    # (time_s, team_id, player_id, player_name)
    pass_events_for_ppda = []
    action_events_for_ppda = []
    shot_events = []

    for ev in events:
        ge = ev.get('gameEvents') or {}
        pe = ev.get('possessionEvents') or {}
        st = ev.get('startTime')
        if st is None:
            continue

        period = ge.get('period')
        home_positive = orientation.get(period)

        if ge.get('gameEventType') == 'OTB':
            team_id = ge.get('teamId')
            otb_rows.append((st, team_id))
            carrier_events.append({'time_s': st, 'team_id': team_id,
                                    'player_id': str(ge.get('playerId')),
                                    'player_name': ge.get('playerName')})

        if ge.get('setpieceType') in DEADBALL_SETPIECE_TYPES:
            mask_windows.append((st - MASK_BEFORE_S, st + MASK_AFTER_S))

        et = pe.get('possessionEventType')
        ball = ev.get('ball') or []
        if ball and ball[0].get('x') is not None and home_positive is not None:
            bx = ball[0]['x']
            if et == 'PA' and pe.get('passOutcomeType') == 'C':
                pass_events_for_ppda.append({'time_s': st, 'team_id': ge.get('teamId'),
                                              'x': bx, 'home_positive': home_positive})
            if et == 'CH':
                challenger_id = pe.get('challengerPlayerId')
                info = player_id_lookup.get(str(challenger_id)) if challenger_id is not None else None
                if info is not None:
                    action_events_for_ppda.append({'time_s': st, 'team_id': info[2], 'x': bx,
                                                    'home_positive': home_positive})
            if et == 'CL':
                clearer_id = pe.get('clearerPlayerId')
                info = player_id_lookup.get(str(clearer_id)) if clearer_id is not None else None
                if info is not None:
                    action_events_for_ppda.append({'time_s': st, 'team_id': info[2], 'x': bx,
                                                    'home_positive': home_positive})

        if et == 'SH':
            shot_events.append({'time_s': st, 'team_id': ge.get('teamId'),
                                 'shooter_name': pe.get('shooterPlayerName'),
                                 'outcome': pe.get('shotOutcomeType')})

    otb_rows.sort(key=lambda r: r[0])
    otb_times = [r[0] for r in otb_rows]
    otb_teams = [r[1] for r in otb_rows]

    carrier_events.sort(key=lambda e: e['time_s'])
    carrier_times = [e['time_s'] for e in carrier_events]

    # lost-ball times per team: an OTB event where the team on the ball changes
    lost_ball_times = {HOME_TEAM_ID: [], AWAY_TEAM_ID: []}
    for i in range(1, len(otb_rows)):
        prev_team, cur_team = otb_rows[i - 1][1], otb_rows[i][1]
        if prev_team != cur_team and prev_team in lost_ball_times:
            lost_ball_times[prev_team].append(otb_rows[i][0])

    return {
        'mask_windows': mask_windows,
        'otb_times': otb_times,
        'otb_teams': otb_teams,
        'carrier_times': carrier_times,
        'carrier_events': carrier_events,
        'pass_events_for_ppda': pass_events_for_ppda,
        'action_events_for_ppda': action_events_for_ppda,
        'shot_events': shot_events,
        'lost_ball_times': lost_ball_times,
    }


def in_dead_ball_mask(t, mask_windows):
    # mask_windows is short enough that a linear scan is fine; called rarely in bulk paths.
    return any(lo <= t <= hi for lo, hi in mask_windows)


def possession_team_at(t, otb_times, otb_teams):
    idx = bisect.bisect_right(otb_times, t) - 1
    if idx < 0:
        return None
    return otb_teams[idx]


def current_carrier_at(t, carrier_times, carrier_events):
    idx = bisect.bisect_right(carrier_times, t) - 1
    return carrier_events[idx] if idx >= 0 else None


PPDA_ZONE_MIN_DEPTH_M = 35.0


def _in_press_zone(ev, defending_team_id):
    own_goal_x = 52.5 if (ev['home_positive'] and defending_team_id == HOME_TEAM_ID) or \
                          (not ev['home_positive'] and defending_team_id == AWAY_TEAM_ID) else -52.5
    depth = abs(own_goal_x - ev['x'])
    return depth >= PPDA_ZONE_MIN_DEPTH_M


def compute_ppda(pass_events_for_ppda, action_events_for_ppda, attacking_team_id, defending_team_id,
                  time_lo=None, time_hi=None, use_zone=True):
    passes = [e for e in pass_events_for_ppda if e['team_id'] == attacking_team_id
              and (time_lo is None or time_lo <= e['time_s'] < time_hi)
              and (not use_zone or _in_press_zone(e, defending_team_id))]
    actions = [e for e in action_events_for_ppda if e['team_id'] == defending_team_id
               and (time_lo is None or time_lo <= e['time_s'] < time_hi)
               and (not use_zone or _in_press_zone(e, defending_team_id))]
    if not actions:
        return None, len(passes), len(actions)
    return len(passes) / len(actions), len(passes), len(actions)


if __name__ == '__main__':
    from config import load_roster, load_metadata, orientation_lookup
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    events = load_events()
    print(f"Total event records: {len(events)}")

    game_event_types = set()
    poss_event_types = set()
    for ev in events:
        ge = ev.get('gameEvents') or {}
        pe = ev.get('possessionEvents') or {}
        if ge.get('gameEventType'):
            game_event_types.add(ge['gameEventType'])
        if pe.get('possessionEventType'):
            poss_event_types.add(pe['possessionEventType'])
    print("game event types:", sorted(game_event_types))
    print("possession event types:", sorted(poss_event_types))

    sig = build_event_signals(events, player_id_lookup, orientation)
    print(f"Dead-ball mask windows: {len(sig['mask_windows'])}")
    print(f"OTB events: {len(sig['otb_times'])}")
    print(f"Morocco lost ball: {len(sig['lost_ball_times'][HOME_TEAM_ID])} times")
    print(f"Spain lost ball: {len(sig['lost_ball_times'][AWAY_TEAM_ID])} times")
    print(f"PPDA pass events: {len(sig['pass_events_for_ppda'])}, action events: {len(sig['action_events_for_ppda'])}")
    print(f"Shots: {len(sig['shot_events'])}")

    morocco_ppda, mp, ma = compute_ppda(sig['pass_events_for_ppda'], sig['action_events_for_ppda'], AWAY_TEAM_ID, HOME_TEAM_ID)
    spain_ppda, sp, sa = compute_ppda(sig['pass_events_for_ppda'], sig['action_events_for_ppda'], HOME_TEAM_ID, AWAY_TEAM_ID)
    print(f"Morocco PPDA: {morocco_ppda:.2f} ({mp}/{ma})")
    print(f"Spain PPDA: {spain_ppda:.2f} ({sp}/{sa})")

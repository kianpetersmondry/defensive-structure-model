"""Convert a DFL open-data match (Bundesliga, Sportec/TRACAB positions + DFL event feed) into the four
PFF-shaped files the pipeline reads: metadata.json, roster.json, events.json, tracking.jsonl.bz2.

    python3 providers/dfl_to_pff.py <dfl_dir> <DFL-MAT-id> <pipeline match id> <out_dir> [--home-name X --away-name Y]

What is converted, and how (see providers/README.md):
- Positions: DFL gives every player and the ball at 25 Hz in metres, centred on the spot, x along the
  length; the same axes as PFF (checked: a left-back attacking +x sits at +y in both). The pipeline has
  PFF's 29.97 fps baked into many frame-count thresholds, so positions are linearly interpolated onto a
  29.97 fps grid rather than changing those thresholds. Speed (DFL `S`, km/h) becomes PFF `speed` in m/s.
- Clock: video time = seconds since 10 s before the first-half kickoff frame; frameNum = video time x 29.97.
- Events: DFL events are single timestamps typed by a human operator. Each on-ball event (pass, cross,
  shot, other ball action, ball claim) is snapped to tracking: the moment within +/-0.6 s (after a global
  clock offset) when the ball was closest to the acting player. The PFF "on the ball" (OTB) spell for that
  player then starts where the ball first stayed within 2 m of him, walking back from that moment.
  Consecutive actions by the same player merge into one spell, as PFF records them.
- Ball out of play: DFL tracking carries BallStatus per frame; every live->dead switch becomes an OUT event.
- Not available in DFL and left empty: PFF's analyst "under pressure" tag (initialPressureType), which
  engagement.py uses to calibrate the engaged/passive threshold (it falls back to the other matches'
  pooled threshold), and clearances (CL), which only feed the printed PPDA.
"""
import bz2, json, math, os, re, sys, collections
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfl_parse

FPS = 29.97
DT = 0.04                                     # DFL frame spacing (25 Hz)
SECTIONS = {'firstHalf': 1, 'secondHalf': 2, 'firstHalfExtra': 3, 'secondHalfExtra': 4}
CLOCK0 = {1: 0, 2: 2700, 3: 5400, 4: 6300}
SETPIECE = {'KickOff': 'K', 'ThrowIn': 'T', 'FreeKick': 'F', 'CornerKick': 'C', 'GoalKick': 'G', 'Penalty': 'P'}
SHOT_OUT = {'SuccessfulShot': 'G', 'SavedShot': 'S', 'BlockedShot': 'B', 'ShotWide': 'W', 'ShotWoodWork': 'P'}
POS = {'TW': 'GK', 'IVL': 'LCB', 'IVR': 'RCB', 'IVZ': 'MCB', 'LV': 'LB', 'RV': 'RB', 'DLM': 'LWB', 'DRM': 'RWB',
       'DML': 'DM', 'DMR': 'DM', 'DMZ': 'DM', 'ZD': 'DM', 'ZM': 'CM', 'LM': 'LW', 'RM': 'RW', 'OLM': 'LW', 'ORM': 'RW',
       'ZO': 'AM', 'HL': 'CF', 'HR': 'CF', 'LA': 'LW', 'RA': 'RW', 'STZ': 'CF', 'STL': 'CF', 'STR': 'CF', 'ST': 'CF'}
NEAR = 2.0                                    # m: ball "with" a player
SYNC_WIN = 0.6                                # s: snap window around the (offset-corrected) event time


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def num_id(dfl_id):
    """DFL ids like DFL-CLU-00000G / DFL-OBJ-0002DR -> stable integers (base 36 of the suffix)."""
    return int(dfl_id.rsplit('-', 1)[1], 36)


def team_num(dfl_id):
    return 9_000_000 + num_id(dfl_id)          # well clear of PFF team ids (hundreds)


def fmt_clock(s):
    s = max(0, int(s)); return f'{s // 60:02d}:{s % 60:02d}'


def main():
    a = sys.argv[1:]
    dfl_dir, mat, mid, out = a[:4]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    f = lambda kind: [os.path.join(dfl_dir, x) for x in os.listdir(dfl_dir) if kind in x and mat in x][0]
    os.makedirs(out, exist_ok=True)
    notes = []

    # ---------------- match information: teams, roster
    mi = ET.parse(f('matchinformation')).getroot()
    gen = mi.find('.//General').attrib; env = mi.find('.//Environment').attrib
    teams = {t.get('Role'): t for t in mi.iter('Team')}
    home, away = teams['home'], teams['guest']
    hid, aid = home.get('TeamId'), away.get('TeamId')
    names = {hid: opt('--home-name', home.get('TeamName')), aid: opt('--away-name', away.get('TeamName'))}
    players = {}                                  # DFL person id -> dict
    roster = []
    for t in (home, away):
        for p in t.iter('Player'):
            pos = POS.get(p.get('PlayingPosition'), 'CM')
            players[p.get('PersonId')] = dict(team=t.get('TeamId'), shirt=p.get('ShirtNumber'), name=p.get('Shortname'),
                                              pid=num_id(p.get('PersonId')), pos=pos, started=p.get('Starting') == 'true')
            roster.append({'player': {'id': str(num_id(p.get('PersonId'))), 'nickname': p.get('Shortname')},
                           'positionGroupType': pos, 'shirtNumber': p.get('ShirtNumber'),
                           'started': p.get('Starting') == 'true',
                           'team': {'id': str(team_num(t.get('TeamId'))), 'name': names[t.get('TeamId')]}})
    unknown = [p.get('PlayingPosition') for p in mi.iter('Player') if p.get('PlayingPosition') and p.get('PlayingPosition') not in POS]
    if unknown: notes.append(f'unmapped DFL positions (set to CM): {sorted(set(unknown))}')
    px, py = float(env.get('PitchX', 105)), float(env.get('PitchY', 68))
    if abs(px - 105) > 0.5 or abs(py - 68) > 0.5:
        notes.append(f'pitch is {px}x{py} m; the pipeline assumes 105x68 (goal lines and thirds will be slightly off)')

    # ---------------- positions
    npz = os.path.join(out, '_positions.npz')
    if not os.path.exists(npz):
        print('parsing positions XML ...', flush=True)
        np.savez_compressed(npz, **dfl_parse.parse(f('positions')))
    P = np.load(npz)
    sets = collections.defaultdict(dict)
    for k in P.files:
        sec, team, pid, field = k.split('|'); sets[(sec, team, pid)][field] = P[k]
    periods = {}
    for (sec, team, pid), d in sets.items():
        if team != 'BALL': continue
        per = SECTIONS[sec]; n = d['n']
        periods[per] = dict(sec=sec, n0=int(n[0]), N=len(n), t0=float(d['t0'][0]), bx=d['x'], by=d['y'], bz=d['z'],
                            status=d['status'], poss=d['poss'], people={})
    for (sec, team, pid), d in sets.items():
        if team in ('BALL', 'referee') or pid not in players: continue
        pr = periods[SECTIONS[sec]]; idx = d['n'] - pr['n0']
        full = {c: np.full(pr['N'], np.nan, np.float32) for c in ('x', 'y', 's')}
        ok = (idx >= 0) & (idx < pr['N'])
        for c in full: full[c][idx[ok]] = d[c][ok]
        pr['people'][pid] = full
    T_REF = periods[1]['t0'] - 10.0
    vid = lambda t: t - T_REF

    def frame_i(per, t):
        pr = periods[per]; i = int(round((t - pr['t0']) / DT)); return min(max(i, 0), pr['N'] - 1)

    def period_of(t, slack=3.0):
        for per, pr in periods.items():
            if pr['t0'] - slack <= t <= pr['t0'] + (pr['N'] - 1) * DT + slack: return per
        return None

    def dist(per, pid, i0, i1):
        pr = periods[per]; pp = pr['people'].get(pid)
        if pp is None: return np.full(i1 - i0, np.nan)
        return np.hypot(pp['x'][i0:i1] - pr['bx'][i0:i1], pp['y'][i0:i1] - pr['by'][i0:i1])

    # home direction in period 1 from the keepers' positions (tracked, not labelled by hand)
    gk = {t: [pid for pid, p in players.items() if p['team'] == t and p['pos'] == 'GK' and pid in periods[1]['people']] for t in (hid, aid)}
    hx = np.nanmean(periods[1]['people'][gk[hid][0]]['x'])
    home_start_left = bool(hx < 0)                # home keeper in the -x goal = home defends -x

    # ---------------- events
    ev = ET.parse(f('events_raw')).getroot()
    touches, tackles, raw_goals = [], [], []
    for e in ev.iter('Event'):
        if len(e) == 0 or e[0].tag == 'Delete': continue
        t = ts(e.get('EventTime')); top = e[0]
        sp = SETPIECE.get(top.tag, 'O')
        act = None
        for tag in ('ShotAtGoal', 'Play', 'OtherBallAction', 'BallClaiming', 'TacklingGame'):
            act = top if top.tag == tag else top.find(f'.//{tag}')
            if act is not None: break
        if act is None: continue
        if act.tag == 'TacklingGame':
            ch = act.get('Winner') if act.get('WinnerRole') == 'withoutBallControl' else act.get('Loser') if act.get('LoserRole') == 'withoutBallControl' else None
            carrier = act.get('Loser') if act.get('LoserRole') == 'withBallControl' else act.get('Winner')
            if ch in players: tackles.append(dict(t=t, challenger=ch, carrier=carrier))
            continue
        pid = act.get('Player')
        if pid not in players: continue
        d = dict(t_raw=t, pid=pid, team=players[pid]['team'], sp=sp, eid=e.get('EventId'))
        if act.tag == 'ShotAtGoal':
            res = next((c.tag for c in act if c.tag in SHOT_OUT), None)
            d.update(kind='SH', outcome=SHOT_OUT.get(res), xg=act.get('xG'))
            if res == 'SuccessfulShot': raw_goals.append(d)
        elif act.tag == 'Play':
            d.update(kind='CR' if act.find('Cross') is not None else 'PA',
                     outcome='C' if act.get('Evaluation') in ('successfullyCompleted', 'successful') else 'D',
                     recipient=act.get('Recipient'))
        elif act.tag == 'OtherBallAction':
            d.update(kind='BC')
        else:
            d.update(kind='CLAIM')
        touches.append(d)

    ko = next((k for k in ev.iter('KickOff') if k.get('GameSection') == 'firstHalf'), None)
    if ko is not None and ko.get('TeamLeft') and (ko.get('TeamLeft') == hid) != home_start_left:
        notes.append('WARNING: DFL says the home team started on the left, but the keepers say otherwise')

    # global event->tracking clock offset: median of best-match lags over well-matched events
    lags = []
    for d in touches:
        per = period_of(d['t_raw'])
        if per is None: continue
        i0 = frame_i(per, d['t_raw'] - 1.5); i1 = frame_i(per, d['t_raw'] + 1.5) + 1
        dd = dist(per, d['pid'], i0, i1)
        if np.all(np.isnan(dd)): continue
        j = int(np.nanargmin(dd))
        if dd[j] < 1.0: lags.append(periods[per]['t0'] + (i0 + j) * DT - d['t_raw'])
    OFFSET = float(np.median(lags)) if lags else 0.0
    notes.append(f'event clock offset vs tracking: {OFFSET:+.2f} s (median over {len(lags)} events matched within 1 m)')

    snapped = 0
    for d in touches:
        t = d['t_raw'] + OFFSET; per = period_of(t); d['per'] = per
        if per is None: d['t'] = t; continue
        i0 = frame_i(per, t - SYNC_WIN); i1 = frame_i(per, t + SYNC_WIN) + 1
        dd = dist(per, d['pid'], i0, i1)
        if not np.all(np.isnan(dd)) and np.nanmin(dd) < 2.5:
            j = int(np.nanargmin(dd)); d['t'] = periods[per]['t0'] + (i0 + j) * DT; snapped += 1; d['snap'] = 1
        else:
            d['t'] = t; d['snap'] = 0
    touches = [d for d in touches if d['per'] is not None]
    touches.sort(key=lambda d: d['t'])
    by = collections.defaultdict(lambda: [0, 0])
    for d in touches: by[d['kind']][0] += d['snap']; by[d['kind']][1] += 1
    notes.append('snapped by type: ' + ', '.join(f'{k} {a}/{b}' for k, (a, b) in sorted(by.items())))
    notes.append(f'{snapped}/{len(touches)} on-ball events snapped to the tracked ball (rest kept at the corrected clock time)')

    # spell start: walk back while the ball stays within NEAR of the player (gaps up to 5 frames), not before the previous touch
    prev_t = {}
    for k, d in enumerate(touches):
        per = d['per']; pr = periods[per]; ie = frame_i(per, d['t'])
        lo_t = touches[k - 1]['t'] + DT if k and touches[k - 1]['per'] == per else pr['t0']
        lo = max(frame_i(per, lo_t), ie - int(15 / DT))
        dd = dist(per, d['pid'], lo, ie + 1)
        start = ie; gap = 0
        for j in range(len(dd) - 1, -1, -1):
            if not np.isnan(dd[j]) and dd[j] < NEAR: start = lo + j; gap = 0
            else:
                gap += 1
                if gap > 5: break
        d['t_start'] = pr['t0'] + start * DT

    # merge consecutive touches by the same player into one on-the-ball spell
    spells = []
    for d in touches:
        s = spells[-1] if spells else None
        if s and s['pid'] == d['pid'] and s['per'] == d['per'] and d['t'] - s['end'] < 4.0 and s['acts'][-1]['kind'] in ('BC', 'CLAIM'):
            s['acts'].append(d); s['end'] = d['t']
        else:
            spells.append(dict(pid=d['pid'], team=d['team'], per=d['per'], start=d['t_start'], end=d['t'], acts=[d], sp=d['sp']))

    # ball out of play
    outs = []
    for per, pr in periods.items():
        st = pr['status']; sw = np.where((st[:-1] == 1) & (st[1:] == 0))[0] + 1
        outs += [(per, pr['t0'] + i * DT) for i in sw]

    # ---------------- ids, frames, records
    fr = lambda t: int(round(vid(t) * FPS))
    def clock(per, t): return CLOCK0[per] + (t - periods[per]['t0'])
    def ball_at(per, t):
        pr = periods[per]; i = frame_i(per, t); return dict(x=round(float(pr['bx'][i]), 3), y=round(float(pr['by'][i]), 3), z=round(float(pr['bz'][i]), 3))

    ge_id, pe_id = 1_000_000, 2_000_000
    records, line_ge, line_pe, pe_ge = [], {}, {}, {}
    HT, AT = team_num(hid), team_num(aid)
    for s in spells:
        ge_id += 1; p = players[s['pid']]; per = s['per']; tid = team_num(s['team'])
        sf, ef = fr(s['start']), fr(s['end'])
        ge = {'game_id': int(mid), 'game_event_type': 'OTB', 'formatted_game_clock': fmt_clock(clock(per, s['start'])),
              'player_id': str(p['pid']), 'player_name': p['name'], 'shirt_number': p['shirt'], 'position_group_type': p['pos'],
              'team_id': str(tid), 'team_name': names[s['team']], 'start_time': round(vid(s['start']), 3), 'end_time': round(vid(s['end']), 3),
              'duration': round(s['end'] - s['start'], 3), 'home_team': int(tid == HT), 'sequence': None, 'home_ball': tid == HT,
              'start_frame': sf, 'end_frame': ef}
        s['ge'] = ge; s['ge_id'] = ge_id
        for k in range(sf, ef + 1): line_ge[k] = (ge_id, ge)
        base = lambda: {'gameId': int(mid), 'gameEventId': ge_id, 'startTime': round(vid(s['start']), 3), 'endTime': round(vid(s['end']), 3),
                        'duration': round(s['end'] - s['start'], 3),
                        'gameEvents': {'gameEventType': 'OTB', 'period': per, 'teamId': tid, 'teamName': names[s['team']], 'homeTeam': tid == HT,
                                       'playerId': p['pid'], 'playerName': p['name'], 'setpieceType': s['sp'],
                                       'startGameClock': int(clock(per, s['start'])), 'startFormattedGameClock': fmt_clock(clock(per, s['start']))},
                        'initialTouch': {'initialPressureType': None}}
        r = base(); r.update(possessionEventId=None, eventTime=r['startTime'], possessionEvents={'possessionEventType': 'IT'}, ball=[ball_at(per, s['start'])])
        records.append(r)
        for d in s['acts']:
            if d['kind'] == 'CLAIM': continue
            pe_id += 1; r = base(); t = d['t']
            pev = {'possessionEventType': d['kind']}
            if d['kind'] in ('PA', 'CR'):
                rc = players.get(d.get('recipient'))
                pev.update(passOutcomeType=d['outcome'], passerPlayerId=p['pid'], passerPlayerName=p['name'],
                           receiverPlayerId=rc['pid'] if rc else None, receiverPlayerName=rc['name'] if rc else None)
            if d['kind'] == 'SH':
                pev.update(shooterPlayerId=p['pid'], shooterPlayerName=p['name'], shotOutcomeType=d['outcome'], xg=d.get('xg'))
            r.update(possessionEventId=pe_id, eventTime=round(vid(t), 3), possessionEvents=pev, ball=[ball_at(per, t)], dflEventId=d['eid'])
            records.append(r)
            k = fr(t)
            while k in line_pe: k += 1
            pe_ge[k] = (ge_id, ge)              # the line carrying an action always carries that action's spell
            line_pe[k] = (pe_id, {'game_id': int(mid), 'game_event_id': ge_id, 'possession_event_type': d['kind'],
                                  'formatted_game_clock': fmt_clock(clock(per, t)), 'start_time': round(vid(t), 3), 'start_frame': k})
    # challenges: attached to the ball carrier's spell (PFF files them under the carrier's on-the-ball event)
    ch_kept = 0
    for c in tackles:
        t = c['t'] + OFFSET; per = period_of(t)
        if per is None: continue
        cand = [s for s in spells if s['per'] == per and s['start'] - 1.0 <= t <= s['end'] + 1.0]
        cand = [s for s in cand if s['pid'] == c['carrier']] or cand
        if not cand: continue
        s = min(cand, key=lambda s: abs(t - s['end'])); pe_id += 1; ch_kept += 1
        chp = players[c['challenger']]
        records.append({'gameId': int(mid), 'gameEventId': s['ge_id'], 'possessionEventId': pe_id, 'startTime': round(vid(s['start']), 3),
                        'endTime': round(vid(s['end']), 3), 'duration': round(s['end'] - s['start'], 3), 'eventTime': round(vid(t), 3),
                        'gameEvents': {'gameEventType': 'OTB', 'period': per, 'teamId': team_num(s['team']), 'teamName': names[s['team']],
                                       'homeTeam': team_num(s['team']) == HT, 'playerId': players[s['pid']]['pid'],
                                       'playerName': players[s['pid']]['name'], 'setpieceType': s['sp']},
                        'initialTouch': {'initialPressureType': None},
                        'possessionEvents': {'possessionEventType': 'CH', 'challengerPlayerId': chp['pid'], 'challengerPlayerName': chp['name']},
                        'ball': [ball_at(per, t)]})
    for per, t in outs:
        ge_id += 1; k = fr(t)
        ge = {'game_id': int(mid), 'game_event_type': 'OUT', 'formatted_game_clock': fmt_clock(clock(per, t)), 'player_id': None, 'player_name': None,
              'shirt_number': None, 'position_group_type': None, 'team_id': None, 'team_name': None, 'start_time': round(vid(t), 3), 'end_time': None,
              'duration': 0.0, 'home_team': None, 'sequence': None, 'home_ball': None, 'start_frame': k, 'end_frame': k}
        line_ge[k] = (ge_id, ge)
        records.append({'gameId': int(mid), 'gameEventId': ge_id, 'possessionEventId': None, 'startTime': round(vid(t), 3), 'endTime': round(vid(t), 3),
                        'duration': 0.0, 'eventTime': round(vid(t), 3), 'gameEvents': {'gameEventType': 'OUT', 'period': per, 'teamId': None},
                        'initialTouch': {}, 'possessionEvents': None, 'ball': [ball_at(per, t)]})
    line_ge.update(pe_ge)
    records.sort(key=lambda r: (r['startTime'], r['eventTime']))
    json.dump(records, open(os.path.join(out, 'events.json'), 'w'))

    # ---------------- metadata, roster
    meta = [{'id': mid, 'source': f'DFL open data {mat} (converted by providers/dfl_to_pff.py)',
             'homeTeam': {'id': str(HT), 'name': names[hid], 'shortName': opt('--home-code', '')},
             'awayTeam': {'id': str(AT), 'name': names[aid], 'shortName': opt('--away-code', '')},
             'competition': {'id': gen.get('CompetitionId'), 'name': gen.get('CompetitionName')},
             'date': gen.get('KickoffTime'), 'season': gen.get('Season'), 'week': int(gen.get('MatchDay', 0)), 'result': gen.get('Result'),
             'fps': FPS, 'homeTeamStartLeft': home_start_left,
             'stadium': {'id': env.get('StadiumId'), 'name': env.get('StadiumName'), 'pitches': [{'length': px, 'width': py}]}}]
    if 3 in periods: meta[0]['homeTeamStartLeftExtraTime'] = bool(np.nanmean(periods[3]['people'][gk[hid][0]]['x']) < 0)
    for per, pr in periods.items():
        meta[0][f'startPeriod{per}'] = round(vid(pr['t0']), 3); meta[0][f'endPeriod{per}'] = round(vid(pr['t0'] + (pr['N'] - 1) * DT), 3)
    json.dump(meta, open(os.path.join(out, 'metadata.json'), 'w'), indent=1, ensure_ascii=False)
    json.dump(roster, open(os.path.join(out, 'roster.json'), 'w'), indent=1, ensure_ascii=False)

    # ---------------- tracking lines at 29.97 fps
    print('writing tracking ...', flush=True)
    side = {pid: ('home' if p['team'] == hid else 'away') for pid, p in players.items()}
    nlines = 0
    with bz2.open(os.path.join(out, 'tracking.jsonl.bz2'), 'wt', compresslevel=6) as fo:
        for per in sorted(periods):
            pr = periods[per]; v0 = vid(pr['t0']); v1 = vid(pr['t0'] + (pr['N'] - 1) * DT)
            ks = np.arange(math.ceil(v0 * FPS), math.floor(v1 * FPS) + 1)
            fi = (ks / FPS - v0) / DT; i0 = np.clip(np.floor(fi).astype(int), 0, pr['N'] - 2); w = (fi - i0).astype(np.float32)
            lerp = lambda arr: arr[i0] * (1 - w) + arr[i0 + 1] * w
            bx, by, bz = lerp(pr['bx']), lerp(pr['by']), np.maximum(lerp(pr['bz']), 0)
            pp = {pid: {c: lerp(v) for c, v in d.items()} for pid, d in pr['people'].items()}
            for j, k in enumerate(ks):
                line = {'version': None, 'gameRefId': None, 'generatedTime': None, 'smoothedTime': None,
                        'videoTimeMs': round(k / FPS * 1000, 3), 'frameNum': int(k), 'period': per,
                        'periodElapsedTime': round(k / FPS - v0, 6), 'periodGameClockTime': round(CLOCK0[per] + k / FPS - v0, 6)}
                for sd in ('home', 'away'):
                    raw, sm = [], []
                    for pid, d in pp.items():
                        if side[pid] != sd or np.isnan(d['x'][j]): continue
                        x, y = round(float(d['x'][j]), 3), round(float(d['y'][j]), 3)
                        raw.append({'jerseyNum': players[pid]['shirt'], 'confidence': 'HIGH', 'visibility': 'VISIBLE', 'x': x, 'y': y,
                                    'speed': round(float(d['s'][j]) / 3.6, 3)})
                        sm.append({'jerseyNum': players[pid]['shirt'], 'confidence': 'HIGH', 'visibility': 'VISIBLE', 'x': x, 'y': y})
                    line[f'{sd}Players'] = raw; line[f'{sd}PlayersSmoothed'] = sm
                b = {'visibility': 'VISIBLE', 'x': round(float(bx[j]), 3), 'y': round(float(by[j]), 3), 'z': round(float(bz[j]), 3)}
                line['balls'] = [b]; line['ballsSmoothed'] = dict(b)
                g = line_ge.get(int(k)); p = line_pe.get(int(k))
                line['game_event_id'] = float(g[0]) if g else None; line['possession_event_id'] = float(p[0]) if p else None
                line['game_event'] = g[1] if g else None; line['possession_event'] = p[1] if p else None
                fo.write(json.dumps(line, ensure_ascii=False) + '\n'); nlines += 1
    os.remove(npz)

    # ---------------- report
    shots = collections.Counter(names[d['team']] for d in touches if d['kind'] == 'SH')
    goals = [(names[d['team']], int(clock(d['per'], d['t'] if 't' in d else d['t_raw']) // 60) + 1, players[d['pid']]['name'], d['sp']) for d in raw_goals]
    rep = dict(match=mid, dfl=mat, lines=nlines, spells=len(spells), touches=len(touches), challenges=ch_kept, outs=len(outs),
               shots=dict(shots), goals=goals, home_start_left=home_start_left,
               dfl_kickoff_team_left=ko.get('TeamLeft') if ko is not None else None, notes=notes)
    json.dump(rep, open(os.path.join(out, 'conversion_report.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

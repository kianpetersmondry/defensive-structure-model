"""Check a match's raw files and work out the facts the rest of the pipeline relies on.

    python3 pipeline/validate.py <match id>

Writes output/derived/<id>.json with:
- dirs: for each period, +1 if the home team defends the +x goal, -1 if it defends -x. Three independent
  sources are compared: the metadata's start sides, where each team's shots went (nobody shoots at their own
  goal), and where each team's labelled keeper stands. Shots decide; metadata, then keepers, are fallbacks.
- keeper_swap: periods where the labelled keepers stand at the goal the shots say the *other* team defends
  (the 2022 final's extra time has this), with the two shirt numbers to swap.
- fps, pitch size, periods, whether a per-player speed field exists, shot counts, warnings and problems.
A problem (missing file, team ids that don't match the registry, no keeper) stops with exit code 1.
"""
import bz2
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import R

mid = sys.argv[1]
m = R.get(mid)
problems, warnings, lines_out = [], [], []
say = lambda s: (lines_out.append(s), print(s))

say(f"{m['title']} ({mid}) · raw data in {R.raw_dir(mid)}")
need = ['metadata.json', 'roster.json', 'tracking.jsonl.bz2', 'events.json']
missing = [f for f in need if not os.path.exists(R.raw(mid, f))]
if missing:
    print('BLOCKING: missing files:', ', '.join(missing)); sys.exit(1)

# ---------------- metadata
meta = json.load(open(R.raw(mid, 'metadata.json'))); meta = meta[0] if isinstance(meta, list) else meta
fps = float(meta.get('fps') or 29.97)
try:
    pitch = (float(meta['stadium']['pitches'][0]['length']), float(meta['stadium']['pitches'][0]['width']))
except (KeyError, IndexError, TypeError):
    pitch = (105.0, 68.0); warnings.append('pitch size missing from metadata; assuming 105 x 68')
hsl, hsl_et = meta.get('homeTeamStartLeft'), meta.get('homeTeamStartLeftExtraTime')
hid, aid = str(m['home']['id']), str(m['away']['id'])
if str(meta.get('homeTeam', {}).get('id')) != hid:
    problems.append(f"metadata home team id {meta.get('homeTeam', {}).get('id')} != registry {hid}")
say(f"metadata: fps {fps} · pitch {pitch[0]} x {pitch[1]} m · homeTeamStartLeft {hsl} · ET {hsl_et}")
if abs(pitch[0] - 105) > 0.5 or abs(pitch[1] - 68) > 0.5:
    warnings.append(f'pitch is {pitch[0]} x {pitch[1]}; the pipeline assumes 105 x 68')

# ---------------- roster
roster = json.load(open(R.raw(mid, 'roster.json')))
gk = {hid: set(), aid: set()}
for side, tid in (('home', hid), ('away', aid)):
    rows = [r for r in roster if str(r['team']['id']) == tid]
    starters = sum(bool(r.get('started')) for r in rows)
    keepers = sorted({r['shirtNumber'] for r in rows if r.get('positionGroupType') == 'GK'}, key=lambda x: int(x))
    gk[tid] = set(keepers)
    say(f"roster {m[side]['name']}: {len(rows)} players, {starters} starters, keepers {keepers}")
    if not rows: problems.append(f'no roster rows for {m[side]["name"]} (team id {tid})')
    if not keepers: problems.append(f'no goalkeeper in the roster for {m[side]["name"]}')
    if starters != 11: warnings.append(f'{m[side]["name"]} has {starters} starters in the roster')

# ---------------- tracking: every 30th line, plus every line carrying a shot
periods = collections.Counter(); ball_ok = collections.Counter(); nplayers = []; speed_field = False
shots = {}                                    # game_event_id -> (team id, period, ball x)
gkx = collections.defaultdict(list)           # (period, team id, shirt) -> keeper x samples
with bz2.open(R.raw(mid, 'tracking.jsonl.bz2'), 'rt') as f:
    for k, line in enumerate(f):
        is_shot = '"possession_event_type":"SH"' in line or '"possession_event_type": "SH"' in line
        if k % 30 and not is_shot:
            continue
        d = json.loads(line); per = d.get('period')
        if per is None:
            continue
        b = d.get('ballsSmoothed') or {}
        if isinstance(b, list): b = b[0] if b else {}
        if is_shot:
            ge = d.get('game_event') or {}
            gid = d.get('game_event_id')
            if gid not in shots and ge.get('team_id') is not None and b.get('x') is not None:
                shots[gid] = (str(ge['team_id']), per, float(b['x']))
        if k % 30:
            continue
        periods[per] += 1; ball_ok[per] += b.get('x') is not None
        n = 0
        for side, tid in (('home', hid), ('away', aid)):
            pl = d.get(f'{side}PlayersSmoothed') or d.get(f'{side}Players') or []
            for p in pl:
                if p.get('x') is None: continue
                n += 1
                if str(p.get('jerseyNum')) in gk[tid]: gkx[(per, tid, str(p['jerseyNum']))].append(float(p['x']))
            if any(p.get('speed') is not None for p in d.get(f'{side}Players') or []): speed_field = True
        nplayers.append(n)
say(f"tracking: periods {sorted(periods)} · ~{sum(periods.values()) * 30:,} frames · players per frame median {int(np.median(nplayers))} · raw speed field {'yes' if speed_field else 'no'}")
for p in sorted(periods):
    say(f"  period {p}: ~{periods[p] * 30 / fps / 60:.1f} min of frames, ball tracked {100 * ball_ok[p] / periods[p]:.0f}%")

# ---------------- direction per period
def md_dir(p):
    if hsl is None: return None
    left = hsl if p in (1, 2) else (hsl_et if hsl_et is not None else hsl)
    home_neg = left if p in (1, 3) else not left     # start left = defend -x in the first half of that pair
    return -1 if home_neg else 1

def keeper_dir(p, tid):
    c = [(len(v), np.mean(v), s) for (pp, t, s), v in gkx.items() if pp == p and t == tid]
    if not c: return None, None
    n, x, shirt = max(c)
    return (1 if x > 0 else -1), shirt

dirs, table, swap_periods, swap_shirts = {}, [], [], {}
for p in sorted(periods):
    s = [x if t == hid else -x for t, per, x in shots.values() if per == p]   # + => home shooting at +x
    shot_dir = None
    if len(s) >= 2 and abs(np.sign(s).sum()) >= 0.5 * len(s):
        shot_dir = -1 if np.sign(s).sum() > 0 else 1                          # home attacks +x => defends -x
    md = md_dir(p)
    kh, shirt_h = keeper_dir(p, hid); ka, shirt_a = keeper_dir(p, aid)
    kp = kh if kh is not None else (-ka if ka is not None else None)
    final = shot_dir if shot_dir is not None else (md if md is not None else kp)
    if final is None:
        problems.append(f'period {p}: no way to tell attack direction'); continue
    dirs[p] = final
    table.append((p, md, shot_dir, kp, len(s), final))
    if shot_dir is not None and md is not None and shot_dir != md:
        warnings.append(f'period {p}: metadata start sides disagree with where the shots went; using the shots')
    if shot_dir is not None and kp is not None and kp != shot_dir:
        swap_periods.append(p); swap_shirts = {'home': shirt_h, 'away': shirt_a}
        warnings.append(f'period {p}: labelled keepers stand at the goal the shots say the other team defends (keeper labels swapped)')

fmt = lambda v: '?' if v is None else ('home defends +x' if v == 1 else 'home defends -x')
say('attack direction per period (metadata · shots · keepers → used):')
for p, md, sd, kd, n, fin in table:
    say(f"  period {p}: {fmt(md)} · {fmt(sd)} ({n} shots) · {fmt(kd)} → {fmt(fin)}")

keeper_swap = ({'periods': swap_periods, **swap_shirts,
                'why': 'labelled keepers stand at the goal the shots say the other team defends'} if swap_periods else None)
ev = json.load(open(R.raw(mid, 'events.json')))
by_team = collections.Counter(t for t, _, _ in shots.values())
say(f"events.json: {len(ev):,} events · shots in the tracking feed: {len(shots)} "
    f"({m['home']['name']} {by_team.get(hid, 0)}, {m['away']['name']} {by_team.get(aid, 0)})")
if m.get('goals'):
    say('goals from the registry: ' + ', '.join(f'{t} {mi}' for t, mi, _ in m['goals']))
for w in warnings: say('warning: ' + w)
for pr in problems: say('PROBLEM: ' + pr)

json.dump({'dirs': {str(k): v for k, v in dirs.items()}, 'periods': sorted(periods), 'fps': fps, 'pitch': pitch,
           'speed_field': speed_field, 'keeper_swap': keeper_swap,
           'shots': {m['home']['name']: by_team.get(hid, 0), m['away']['name']: by_team.get(aid, 0)},
           'warnings': warnings, 'problems': problems},
          open(R.derived_path(mid), 'w'), indent=1)
if problems:
    print('BLOCKING problems; see above'); sys.exit(1)
print('OK · written', R.derived_path(mid))

"""Chances conceded, traced back: every shot a team faced, rewound to the start of the attack.

    python3 analysis/chances.py <match id> [...]

Definitions (possession and live play as in turnovers.py):
- Shots are SH possession events; the shooting team is the team of the shot's on-the-ball (OTB) event.
- Possession before the shot: the match-features `possession_team_id` up to the frame before the shot (the
  features register a touch one frame late), with runs shorter than 1 s absorbed into the spell before them,
  except a final run by the shooting team: it won the ball just before shooting.
- Attack start: the latest of
    (a) the shooting team's possession start (the shot itself if the defending team still had the ball),
    (b) the last restart, i.e. the ball coming back into play (inplay.live_mask),
    (c) 20 s before the shot.
- Kind: after a restart if (b) sets the attack start (it wins ties); after a turnover if (a) does and the switch
  was in open play (ball live for the 2 s before it, as for turnovers), with the third the shooting team won the
  ball in, in its own frame (no third when the defending team still had the ball at the shot); otherwise long
  possession (an attack capped at 20 s, or a possession that began less than 2 s after a restart).
- Within 10 s: attack start to shot at most 10 s. Passes: the shooting team's PA/CR events from attack start to shot.
- Goals: each registry goal [team, minute, kind] is the scoring team's last shot in that minute.
- Minute labels: running clock minute + 1, stoppage time as 45+x', 90+x', 105+x', 120+x'.
- turnover_thirds counts where the shooting team won the ball, in its own frame (as in the panel captions:
  "final third" = high up the pitch, near the defending team's goal); turnover_thirds_lost gives the same
  turnovers from the defending team's side (own third = near its own goal).

Each panel shows the defending half in the defending team's frame (own goal on the left): outfield defenders at
the shot (filled) with a line back to where each stood at the attack start, the defending keeper and the
attackers at the shot (hollow), and the ball from the attack start to the shot (diamond). Goals are circled.
Players and the diamond come from the raw tracking sampled every 3rd line (~10 Hz): the first sample at or after the
shot and the attack start (the 5 Hz position cache is too coarse for this). The ball line uses the full-rate ball
track from the attack start to the shot.

Writes output/analysis/chances/<slug>.png/.jpg/.json, keyed by the defending team's name.
"""
import bz2
import json
import re
import sys

import numpy as np

from common import FPS, L, PEND, R, out_path, roster, side_flip
from style import BG, DIM, FAINT, INK, MONO, MONO_MED, Fig, pitch
from turnovers import Feed, absorb_flickers, possession_runs, third

CAP_S = 20.0                        # attack start at most this long before the shot
QUICK_S = 10.0                      # "within 10 s of the attack starting"
PASSES = ('PA', 'CR')
KIND_TEXT = {'restart': 'after a restart', 'long': 'long possession', 'turnover': 'after a turnover'}
THIRDS = ('own', 'middle', 'final')
LOST_AS = {'own': 'final', 'middle': 'middle', 'final': 'own'}     # shooter's third -> defending team's third
SAMPLE = 3                          # positions: every 3rd line of the raw tracking file (~10 Hz)
FRAME_RX = re.compile(r'"frameNum":\s*(\d+)')


def minute_label(clock_s, period):
    m = int(clock_s // 60) + 1
    end = PEND[int(period)]
    return f"{end}+{m - end}'" if m > end else f"{m}'"


def restarts(feed):
    """Frames where the ball comes back into play."""
    lv = feed.live
    return feed.fnum[1:][lv[1:] & ~lv[:-1]]


def possession_before(runs, frame, team):
    """[team, start] of the possession spell just before `frame`."""
    pre = [[t, a, min(b, frame - 1)] for t, a, b in runs if a < frame]
    return absorb_flickers(pre, keep_last=pre[-1][0] == team)[-1][:2]


def trace_shot(feed, runs, starts, passes, frame, ev):
    """One shot faced: attack start, kind, third, duration, passes, ball path (raw coordinates)."""
    team = ev['team']
    holder, since = possession_before(runs, frame, team)
    own = since if holder == team else frame
    back = starts[starts <= frame]
    restart = back[-1] if len(back) else -1
    start = max(own, restart, frame - CAP_S * FPS)
    i0, i1 = feed.index(start), feed.index(frame)
    won = None
    if restart == start:
        kind = 'restart'
    elif own == start and (holder != team or feed.open_play(own)):
        kind = 'turnover'
        if holder == team:
            won = third(feed.ball(own, feed.side_of[team])[0])
    else:
        kind = 'long'
    return dict(frame=int(frame), start=int(round(start)), period=int(feed.fr[i1, 1]), team=team,
                label=minute_label(feed.fr[i1, 2], feed.fr[i1, 1]), clock_s=float(feed.fr[i1, 2]),
                minute=round(float(feed.fr[i1, 2]) / 60, 2),
                player=ev['player'], shooter=ev['player'].split()[-1], kind=kind, third=won,
                seconds=float(frame - start) / FPS, passes=int(sum(team == t and start <= f < frame for f, t in passes)),
                path=feed.fr[i0:i1 + 1, 3:5])


def mark_goals(mid, shots, side_of):
    """Flag each registry goal's shot: the scoring team's last shot in the goal's minute."""
    names = {R.get(mid)[s]['name']: s for s in ('home', 'away')}
    for team, minute, _ in R.goals(mid):
        hits = [s for s in shots if side_of[s['team']] == names[team] and int(s['clock_s'] // 60) + 1 == minute]
        if hits:
            hits[-1]['goal'] = True
        else:
            print(f'{mid}: no shot found for {team} goal {minute}\'')


def read_line(d, swap):
    """Period, ball (x, y) and {side: {jersey: (x, y)}} from one raw tracking line (keeper labels swapped back where
    the registry says so, as in collect.py)."""
    per = d.get('period')
    players = {'home': {}, 'away': {}}
    for side in players:
        for p in d.get(f'{side}PlayersSmoothed') or d.get(f'{side}Players') or []:
            if p.get('x') is None or p.get('jerseyNum') is None:
                continue
            s, j = side, str(p['jerseyNum'])
            if swap and per in swap['periods'] and j == str(swap[side]):
                s, j = ('away', str(swap['away'])) if side == 'home' else ('home', str(swap['home']))
            players[s].setdefault(j, (p['x'], p['y']))
    b = d.get('ballsSmoothed') or {}
    if isinstance(b, list):
        b = b[0] if b else {}
    if b.get('x') is None:
        b = next(iter(d.get('balls') or []), {})
    ball = (np.nan, np.nan) if b.get('x') is None else (b['x'], b['y'])
    return dict(period=per, ball=np.array(ball, float), players=players)


class Snapshots:
    """Players and ball at the first sampled tracking line at or after each wanted frame (one pass over the file)."""

    def __init__(self, mid, frames):
        self.mid = mid
        self.gk = {s: {j for j, r in roster(mid)[s].items() if r['gk']} for s in ('home', 'away')}
        want, self.snap, k = sorted(set(frames)), {}, 0
        swap = R.keeper_swap(mid)
        with bz2.open(R.raw(mid, 'tracking.jsonl.bz2'), 'rt') as fh:
            for n, line in enumerate(fh):
                if n % SAMPLE:
                    continue
                m = FRAME_RX.search(line)
                if not m or int(m.group(1)) < want[k]:
                    continue
                snap = read_line(json.loads(line), swap)
                while k < len(want) and int(m.group(1)) >= want[k]:
                    self.snap[want[k]] = snap
                    k += 1
                if k == len(want):
                    break

    def players(self, frame, side, defending):
        """{jersey: (x, y)} of `side` in the defending team's frame."""
        sn = self.snap[frame]
        flip = side_flip(self.mid, sn['period'], defending)
        return {j: np.array(xy) * flip for j, xy in sn['players'][side].items()}

    def ball(self, frame, defending):
        sn = self.snap[frame]
        return sn['ball'] * side_flip(self.mid, sn['period'], defending)


def caption(s):
    n = s['passes']
    what = f"won in {s['third']} third" if s['third'] else KIND_TEXT[s['kind']]
    return f"{what} · {round(s['seconds'])} s · {n} pass{'es' if n != 1 else ''}"


def finite(v):
    return round(float(v), 2) if np.isfinite(v) else None


def summarise(shots, shots_for):
    turn = [s for s in shots if s['kind'] == 'turnover']
    won = {t: sum(s['third'] == t for s in turn) for t in THIRDS}
    return dict(
        shots_faced=len(shots),
        after_turnover=len(turn),
        after_restart=sum(s['kind'] == 'restart' for s in shots),
        long_possession=sum(s['kind'] == 'long' for s in shots),
        within_10s=sum(s['seconds'] <= QUICK_S for s in shots),
        turnover_thirds={f'{t} third': won[t] for t in THIRDS},
        turnover_thirds_lost={f'{LOST_AS[t]} third': won[t] for t in reversed(THIRDS)},
        shots_for=shots_for, shots_against=len(shots),
        goals_conceded=[f"{s['label']} {s['shooter']}" for s in shots if s.get('goal')],
        shots=[dict(minute=s['minute'], label=s['label'], shooter=s['shooter'], player=s['player'], kind=s['kind'],
                    third=s['third'], goal=bool(s.get('goal')), seconds=round(s['seconds'], 2), passes=s['passes'],
                    caption=caption(s), frame=s['frame'], attack_start_frame=s['start'],
                    x=finite(s['x']), y=finite(s['y'])) for s in shots])


# ---------------------------------------------------------------- drawing

COLS = 5
VIEW_X = (-L / 2 - 1.1, 9.1)        # the defending half, a metre behind the goal line and 9 m past halfway
VIEW_Y = (-34.15, 34.15)
SCALE = 0.0478                      # inches per metre
PANEL_W, PANEL_H = (VIEW_X[1] - VIEW_X[0]) * SCALE, (VIEW_Y[1] - VIEW_Y[0]) * SCALE
X0, COL_STEP, ROW_STEP = 1.078, 3.21, 3.975
TEXT_X = 1.08


def scatter(ax, pts, **kw):
    if pts:
        ax.scatter(*np.array(pts).T, **kw)


def draw_panel(fig, x, top, s, snaps, defending, colors):
    ax = fig.axes(x, top, PANEL_W, PANEL_H)
    pitch(ax, lw=0.8, arcs=False, half=VIEW_X)
    ax.set_ylim(*VIEW_Y)
    ax.set_aspect('auto')
    attacking = 'away' if defending == 'home' else 'home'
    now = snaps.players(s['frame'], defending, defending)
    then = snaps.players(s['start'], defending, defending)
    keepers = [p for j, p in now.items() if j in snaps.gk[defending]]
    outfield = {j: p for j, p in now.items() if j not in snaps.gk[defending]}
    for j, p in outfield.items():
        if j in then:
            ax.plot([then[j][0], p[0]], [then[j][1], p[1]], color=colors[defending], lw=0.8, alpha=0.55, zorder=4)
    path = s['path'] * side_flip(snaps.mid, s['period'], defending)
    path = path[np.isfinite(path[:, 0])]
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], color=INK, lw=1.2, alpha=0.9, zorder=6, solid_joinstyle='round')
        ax.scatter(*path[0], s=16, color=INK, lw=0, zorder=7)
    hollow = dict(s=23, facecolor='none', lw=0.85, zorder=5)
    scatter(ax, list(snaps.players(s['frame'], attacking, defending).values()),
            edgecolor=colors[attacking], **hollow)
    scatter(ax, keepers, edgecolor=colors[defending], **hollow)
    scatter(ax, list(outfield.values()), s=40, color=colors[defending], lw=0, zorder=5)
    scatter(ax, [(s['x'], s['y'])], s=32, marker='D', color='white', edgecolor=BG, lw=0.9, zorder=9)
    if s.get('goal'):
        scatter(ax, [(s['x'], s['y'])], s=230, facecolor='none', edgecolor='white', lw=1.3, zorder=8)
    fig.text(x, top + PANEL_H + 0.235, f"{s['label']} {s['shooter']}" + ('  GOAL' if s.get('goal') else ''),
             fontproperties=MONO_MED, fontsize=10.5, color=INK)
    fig.text(x, top + PANEL_H + 0.465, caption(s), fontproperties=MONO, fontsize=9.6, color=DIM)


def draw_block(fig, top, mid, defending, shots, st, snaps, colors):
    m = R.get(mid)
    opp = m['away' if defending == 'home' else 'home']['name']
    fig.team_chip(1.18, top, f"{m[defending]['name']} defending", colors[defending], size=28, chip=0.19, dy=0.05,
                  corner=0.5)
    goals = f" · goals conceded: {', '.join(st['goals_conceded'])}" if st['goals_conceded'] else ''
    lines = [f"{st['shots_faced']} shots faced: {st['after_turnover']} after losing the ball in open play · "
             f"{st['after_restart']} after a restart (set piece, throw-in, goal kick) · "
             f"{st['long_possession']} from sustained {opp} possession",
             f"{st['within_10s']} of the {st['shots_faced']} came within 10 s of the attack starting{goals}"]
    for k, s in enumerate(lines):
        fig.text(TEXT_X, top + 0.68 + 0.3 * k, s, fontproperties=MONO, fontsize=12, color=DIM)
    first = top + 1.435
    for k, s in enumerate(shots):
        draw_panel(fig, X0 + COL_STEP * (k % COLS), first + ROW_STEP * (k // COLS), s, snaps, defending, colors)
    return first + ROW_STEP * (max(len(shots) - 1, 0) // COLS)


def rows(n):
    return max(1, -(-n // COLS))


def draw(mid, faced, stats, snaps, png, jpg):
    m = R.get(mid)
    colors = {s: m[s]['color'] for s in ('home', 'away')}
    fig = Fig(7.445 + ROW_STEP * (rows(len(faced['home'])) + rows(len(faced['away']))))
    fig.header('Chances conceded, traced back', [
        f"{m['title']} · {m['sub']}",
        "Every shot each team faced, rewound to the start of the attack (up to 20 s) · defending team's own goal on the left"],
        x=TEXT_X, top=0.535, size=12, step=0.25, title_size=46, gap=0.58)
    top = 2.255
    for side in ('home', 'away'):
        last = draw_block(fig, top, mid, side, faced[side], stats[side], snaps, colors)
        top = last + 4.585
    fig.footnotes([
        'Filled dots: defenders at the moment of the shot, with a line back to where each stood when the attack began. '
        'Hollow dots: attackers at the shot.',
        'White line: the ball from the start of the attack to the shot (diamond). Attack start = when the shooting team '
        'got the ball, or play restarted, capped at 20 s before the shot.',
        R.credit(mid, 'Only the defending half is shown. Shots include blocked and off-target efforts. '
                      'Source: PFF FC tracking + event feed behind the 2-D animations.')],
        last + 4.715, color=FAINT, x=TEXT_X, step=0.27)
    fig.save(png, jpg)


def run(mid):
    m = R.get(mid)
    feed = Feed(mid)
    runs = possession_runs(mid)
    starts = restarts(feed)
    passes = [(p['frame'], feed.ev['gev'][p['gid']]['team']) for p in feed.ev['pev'].values()
              if p['type'] in PASSES and p['gid'] in feed.ev['gev']]
    shots = [trace_shot(feed, runs, starts, passes, f, ev) for f, ev in feed.shots]
    mark_goals(mid, shots, feed.side_of)
    faced = {'home': [], 'away': []}
    snaps = Snapshots(mid, [s['frame'] for s in shots] + [s['start'] for s in shots])
    for s in shots:
        defending = 'away' if feed.side_of[s['team']] == 'home' else 'home'
        s['x'], s['y'] = snaps.ball(s['frame'], defending)
        faced[defending].append(s)
    stats = {side: summarise(faced[side], len(faced['away' if side == 'home' else 'home'])) for side in faced}
    base = out_path('analysis', 'chances', m['slug'])
    draw(mid, faced, stats, snaps, base + '.png', base + '.jpg')
    json.dump({m[s]['name']: stats[s] for s in stats}, open(base + '.json', 'w'), ensure_ascii=False, indent=1)
    for side, st in stats.items():
        print(f"{mid} {m[side]['name']} faced {st['shots_faced']}: turnover {st['after_turnover']} "
              f"{st['turnover_thirds']} · restart {st['after_restart']} · long {st['long_possession']} · "
              f"<=10 s {st['within_10s']} · goals {st['goals_conceded']}")


if __name__ == '__main__':
    for a in sys.argv[1:]:
        run(a)

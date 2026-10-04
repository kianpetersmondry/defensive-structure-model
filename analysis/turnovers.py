"""Turnovers: where each team won and lost the ball in open play, and what came next.

    python3 analysis/turnovers.py <match id> [...]

Definitions:
- Possession is the match-features `possession_team_id` (~30 fps). Frames with no team are skipped, and a spell
  shorter than 1 s is absorbed into the spell before it (flickers), so a switch must hold for 1 s.
- A turnover is a switch of possession with the ball live (inplay.live_mask) for the whole 2 s before it, so
  throw-ins, goal kicks and other restarts are left out, as are switches with no ball position or with the ball
  over a goal line.
- The dot is the ball at the switch, drawn in the team's own frame (attacking left to right).
- Fast: the team that won it had the ball in the final third (x > L/6 in its frame) within 10 s, before it
  lost the ball again. Shot: it had a shot (SH possession event) within 15 s.
- Opponents' half: x >= 0. Thirds split at x = -L/6 and +L/6.
A team's losses are its opponent's wins, mirrored into the losing team's own frame.

Writes output/analysis/turn/<slug>.png/.jpg/.json, keyed by team name.
"""
import json
import sys

import numpy as np
import pandas as pd

from common import FPS, L, R, load_ev, out_path, side_flip, team_ids
from inplay import live_mask
from style import BG, DIM, FAINT, INK, MONO, MONO_MED, Fig, pitch

FLICKER = round(1.0 * FPS)          # frames: shorter possession spells are absorbed
LIVE_BEFORE = round(2.0 * FPS)      # frames of live ball needed before the switch
FAST_S, SHOT_S = 10.0, 15.0
THIRD = L / 6
THIRDS = ('own', 'middle', 'final')


class Feed:
    """One match's event feed, ball track (every tracking frame) and live-ball mask."""

    def __init__(self, mid):
        self.mid = mid
        self.ev = load_ev(mid)
        self.fr = self.ev['frames']
        self.fnum = self.fr[:, 0].astype(int)
        self.live = live_mask(self.ev, self.fnum)
        self.side_of = dict(zip(team_ids(mid), ('home', 'away')))
        self.shots = sorted((p['frame'], self.ev['gev'][p['gid']]) for p in self.ev['pev'].values()
                            if p['type'] == 'SH' and p['gid'] in self.ev['gev'])

    def index(self, frame):
        return min(np.searchsorted(self.fnum, frame), len(self.fnum) - 1)

    def ball(self, frame, side):
        """Ball (x, y) at `frame` in `side`'s own frame."""
        i = self.index(frame)
        flip = side_flip(self.mid, self.fr[i, 1], side)
        return self.fr[i, 3] * flip, self.fr[i, 4] * flip

    def open_play(self, frame):
        """A switch at `frame` is open play: ball live for the 2 s before, known and not over a goal line."""
        i = self.index(frame)
        x = self.fr[i, 3]
        return i >= LIVE_BEFORE and bool(self.live[i - LIVE_BEFORE:i].all()) and not np.isnan(x) and abs(x) <= L / 2


def possession_runs(mid):
    """Raw possession: [[team id, first frame, last frame]], frames with no team skipped."""
    f = pd.read_pickle(R.feat(mid))[['frameNum', 'possession_team_id']].drop_duplicates('frameNum').sort_values('frameNum')
    f = f[f.possession_team_id.notna()]
    runs = []
    for fn, t in zip(f.frameNum.to_numpy(), f.possession_team_id.to_numpy()):
        t = str(int(t))
        if runs and runs[-1][0] == t:
            runs[-1][2] = fn
        else:
            runs.append([t, fn, fn])
    return runs


def absorb_flickers(runs, keep_last=False):
    """Spells: runs shorter than 1 s are absorbed into the spell before them. With keep_last the final run is
    kept whatever its length (it is cut short by the moment of interest, not finished)."""
    spells = []
    for k, (t, a, b) in enumerate(runs):
        short = b - a + 1 < FLICKER and not (keep_last and k == len(runs) - 1)
        if spells and (short or spells[-1][0] == t):
            spells[-1][2] = b
        else:
            spells.append([t, a, b])
    return spells


def third(x):
    return 'own' if x < -THIRD else 'final' if x > THIRD else 'middle'


def find_turnovers(feed, spells):
    """Every open-play turnover: dict(side, frame, period, minute, x, y, fast, shot), ball in the winner's frame."""
    out = []
    for k in range(1, len(spells)):
        team, start = spells[k][0], spells[k][1]
        if not feed.open_play(start):
            continue
        side = feed.side_of[team]
        i = feed.index(start)
        flip = side_flip(feed.mid, feed.fr[i, 1], side)
        until = min(start + FAST_S * FPS, spells[k + 1][1] if k + 1 < len(spells) else np.inf)
        ahead = feed.fr[i:np.searchsorted(feed.fnum, until), 3] * flip
        fast = bool(np.isfinite(ahead).any() and np.nanmax(ahead) > THIRD)
        shot = any(g['team'] == team and start <= f <= start + SHOT_S * FPS for f, g in feed.shots)
        out.append(dict(side=side, frame=int(start), period=int(feed.fr[i, 1]), minute=round(float(feed.fr[i, 2]) / 60, 2),
                        x=round(float(feed.fr[i, 3] * flip), 2), y=round(float(feed.fr[i, 4] * flip), 2), fast=fast, shot=shot))
    return out


def summarise(events, side):
    """Counts for `side` from the list of turnovers."""
    won = [e for e in events if e['side'] == side]
    lost = [dict(e, x=-e['x'], y=-e['y']) for e in events if e['side'] != side]     # into the loser's own frame
    return dict(
        won=len(won),
        won_opp_half=sum(e['x'] >= 0 for e in won),
        won_opp_half_pct=round(100 * sum(e['x'] >= 0 for e in won) / max(len(won), 1)),
        won_to_final_third_10s=sum(e['fast'] for e in won),
        won_to_shot_15s=sum(e['shot'] for e in won),
        lost=len(lost),
        lost_own_half=sum(e['x'] <= 0 for e in lost),
        lost_to_opp_shot_15s=sum(e['shot'] for e in lost),
        won_by_third={t: sum(third(e['x']) == t for e in won) for t in THIRDS},
        lost_by_third={t: sum(third(e['x']) == t for e in lost) for t in THIRDS},
        won_events=won, lost_events=lost)


# ---------------------------------------------------------------- drawing

H = 18.27
BLOCK = 7.51                        # inches between the two teams' blocks
PITCH_W = 7.60                      # 105 m
PITCH_X = (1.15, 9.25)


def draw_pitch(fig, x0, top, events, color):
    ax = fig.axes(x0, top, PITCH_W, PITCH_W * 68 / 105)
    pitch(ax, thirds=True, lw=0.8, arcs=False)
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-34, 34)
    plain = [e for e in events if not e['fast'] and not e['shot']]
    fast = [e for e in events if e['fast'] and not e['shot']]
    shot = [e for e in events if e['shot']]
    kw = dict(clip_on=False, zorder=5)
    ax.scatter([e['x'] for e in plain], [e['y'] for e in plain], s=34, color=color, alpha=0.42, lw=0, **kw)
    ax.scatter([e['x'] for e in fast], [e['y'] for e in fast], s=62, color=color, edgecolor='white', lw=1.1, **kw)
    ax.scatter([e['x'] for e in shot], [e['y'] for e in shot], s=130, marker='D', color='white', edgecolor=BG, lw=1.1, **kw)
    return ax


def draw_block(fig, top, mid, side, s):
    m = R.get(mid)
    name, color = m[side]['name'], m[side]['color']
    fig.team_chip(1.175, top - 0.045, name, color, size=28, chip=0.2, dy=0.065)
    fig.text(16.92, top, f"{s['won']} ball wins, {s['won_opp_half_pct']}% in the opponents' half · "
                         f"{s['won_to_final_third_10s']} in or into the final third inside 10 s · "
                         f"{s['won_to_shot_15s']} led to a shot inside 15 s",
             fontproperties=MONO, fontsize=12, color=DIM, ha='right')
    panels = [('WON THE BALL', f"{s['won']} times · {name}: fast into final third {s['won_to_final_third_10s']} · shot {s['won_to_shot_15s']}",
               s['won_events'], s['won_by_third']),
              ('LOST THE BALL', f"{s['lost']} times · opponent: fast into final third {sum(e['fast'] for e in s['lost_events'])} · "
                                f"shot {s['lost_to_opp_shot_15s']}", s['lost_events'], s['lost_by_third'])]
    for (title, desc, events, thirds), x0 in zip(panels, PITCH_X):
        fig.text(x0 - 0.07, top + 0.75, title, fontproperties=MONO_MED, fontsize=12.5, color=INK)
        fig.text(x0 + PITCH_W + 0.07, top + 0.75, desc, fontproperties=MONO, fontsize=11, color=DIM, ha='right')
        draw_pitch(fig, x0, top + 1.08, events, color)
        for j, t in enumerate(THIRDS):
            fig.text(x0 + PITCH_W * (2 * j + 1) / 6, top + 6.24, f'{t} third  {thirds[t]}',
                     fontproperties=MONO, fontsize=11, color=DIM, ha='center')


def draw(mid, stats, png, jpg):
    m = R.get(mid)
    fig = Fig(H)
    fig.header('Turnovers', [f"{m['title']} · {m['sub']}",
                             'Where each team won and lost the ball in open play, and what came next · each team attacking left to right'],
               x=1.08, top=0.55, size=12, step=0.25, title_size=46, gap=0.57)
    for i, side in enumerate(('home', 'away')):
        draw_block(fig, 2.29 + i * BLOCK, mid, side, stats[side])
    y = 17.31
    leg = fig.axes(0, y - 0.1, 18, 0.2)
    leg.set_xlim(0, 18)
    leg.set_ylim(-0.1, 0.1)
    leg.axis('off')
    leg.scatter([1.18], [0], s=34, color=FAINT, lw=0)
    leg.scatter([2.81], [0], s=62, color=INK, lw=0)
    leg.scatter([9.83], [0], s=62, marker='D', color='white', edgecolor=BG, lw=1.1)
    for x, s in ((1.37, 'turnover'), (2.99, 'fast: the team that won it was in or into the final third inside 10 s'),
                 (10.01, 'led to a shot inside 15 s')):
        fig.text(x, y, s, fontproperties=MONO, fontsize=10.5, color=FAINT)
    fig.footnotes(['Open play only: a change of possession while the ball was live for the 2 s before it (throw-ins, goal kicks '
                   'and other restarts left out). Dot = ball position at the switch.',
                   R.credit(mid, 'Dotted lines split the pitch into thirds. Source: PFF FC tracking + event feed behind the 2-D animations.')],
                  17.775, color=FAINT, x=1.08, step=0.26)
    fig.save(png, jpg)


def run(mid):
    m = R.get(mid)
    events = find_turnovers(Feed(mid), absorb_flickers(possession_runs(mid)))
    stats = {side: summarise(events, side) for side in ('home', 'away')}
    base = out_path('analysis', 'turn', m['slug'])
    draw(mid, stats, base + '.png', base + '.jpg')
    json.dump({m[s]['name']: stats[s] for s in stats}, open(base + '.json', 'w'), ensure_ascii=False, indent=1)
    for side, s in stats.items():
        print(f"{mid} {m[side]['name']}: won {s['won']} (opp half {s['won_opp_half']}, fast {s['won_to_final_third_10s']}, "
              f"shot {s['won_to_shot_15s']}, thirds {s['won_by_third']}) · lost {s['lost']} (own half {s['lost_own_half']}, "
              f"opp shot {s['lost_to_opp_shot_15s']}, thirds {s['lost_by_third']})")


if __name__ == '__main__':
    for a in sys.argv[1:]:
        run(a)

"""Line-breaking passes: completed passes through the defending team's midfield or back line.

    python3 analysis/linebreak.py <match id> [...]

Definitions
- A completed pass is a pass (PA) or cross (CR) possession event whose on-the-ball (OTB) spell is followed,
  within 4 s of the pass, by a team-mate's next OTB spell (the reception). The pass must be played in live play.
- Ball positions come from the tracked ball at the pass and reception frames; when the ball is not tracked at
  that frame, the nearest tracked position within 1.5 s is used, and the pass is skipped if there is none.
- All positions are taken in the defending team's own frame (own goal at -x, its left at +y), with its
  outfielders at the 5 Hz sample of the pass.
- A pass is line-breaking when it moves the ball at least 5 m towards the defending goal and bypasses at least
  three of the defending team's deepest seven outfielders (its back and midfield lines): they are between the
  ball and goal when the pass is played and behind the ball when it is received.
- Received behind the back line: the reception is deeper than the mean x of the deepest four outfielders.
- Reception lanes by y in the defending team's frame: their right wing < -20.16 < their right half-space
  < -9.16 < centre < 9.16 < their left half-space < 20.16 < their left wing.
- Live defending time is the number of live 5 Hz samples with the team out of possession.

Writes output/analysis/lb/<slug>.png/.jpg/.json; the JSON is keyed by the defending team's name.
"""
import json
import sys
from collections import Counter

import numpy as np
from matplotlib.patches import FancyArrowPatch

from common import FPS, R, defending, flips, load_ev, load_pos, out_path, outfield_mask, team_ids
from style import DIM, FAINT, INK, MONO, MONO_MED, Fig, pitch

MAX_DT = 4.0                 # s: reception must start within this long of the pass
MIN_GAIN = 5.0               # m towards the defending goal
DEEPEST, MIN_BEATEN = 7, 3   # bypass at least MIN_BEATEN of the deepest DEEPEST outfielders
BALL_WINDOW = 45             # frames (1.5 s): reach for the nearest tracked ball
LANE_EDGES = (-20.16, -9.16, 9.16, 20.16)
LANES = ('right wing', 'right half-space', 'centre', 'left half-space', 'left wing')    # from -y to +y
TOP_N = 5
SAMPLE_S = 6 / FPS           # seconds per 5 Hz sample (every 6th tracking frame)


# ---------------------------------------------------------------- finding the passes

class Ball:
    """Tracked ball position at a frame, reaching for the nearest tracked frame within BALL_WINDOW."""

    def __init__(self, frames):
        self.fn = frames[:, 0].astype(int)
        self.period = frames[:, 1].astype(int)
        self.xy = frames[:, 3:5]
        self.ok = np.flatnonzero(~np.isnan(frames[:, 3]))

    def at(self, f):
        """(x, y, period) in raw pitch coordinates, x and y NaN when no ball is tracked close enough."""
        k = min(np.searchsorted(self.fn, f), len(self.fn) - 1)
        j = np.searchsorted(self.ok, k)
        near = [self.ok[c] for c in (j - 1, j) if 0 <= c < len(self.ok)]
        best = min(near, key=lambda c: abs(self.fn[c] - f), default=None)
        if best is None or abs(self.fn[best] - f) > BALL_WINDOW:
            return np.nan, np.nan, self.period[k]
        return self.xy[best, 0], self.xy[best, 1], self.period[k]


def outfield_positions(mid, P):
    """{side: (n, J, 2) outfield positions in that side's own frame}, NaN when off the pitch."""
    res = {}
    for side in ('home', 'away'):
        om = outfield_mask(mid, side, P[side]['jerseys'])
        res[side] = P[side]['xy'][:, om] * flips(mid, P['period'], side)[:, None, None]
    return res


def lane_of(y):
    return LANES[int(np.searchsorted(LANE_EDGES, y))]


def find_passes(mid, ev, P):
    """{defending team id: [pass dict, ...]} for every line-breaking pass."""
    hid, aid = team_ids(mid)
    gev = ev['gev']
    otb = sorted((v['start'], v['end'] if v['end'] is not None else v['start'], v['team'])
                 for v in gev.values() if v['type'] == 'OTB' and v['start'] is not None)
    ostart = np.array([o[0] for o in otb])
    sfr, live = P['frames'], P['live']
    pos = outfield_positions(mid, P)
    ball = Ball(ev['frames'])
    dirs = R.dirs(mid)
    out = {hid: [], aid: []}
    for p in sorted(ev['pev'].values(), key=lambda p: p['frame'] or 0):
        g = gev.get(p['gid'])
        if p['type'] not in ('PA', 'CR') or g is None:
            continue
        att, f0 = g['team'], p['frame']
        if att not in (hid, aid) or f0 is None:
            continue
        k = np.searchsorted(ostart, (g['end'] or f0) + 1)            # next on-ball spell after the passer's
        if k >= len(otb) or otb[k][2] != att or otb[k][0] - f0 > MAX_DT * FPS:
            continue                                                  # not completed
        f1 = otb[k][0]
        i = min(np.searchsorted(sfr, f0), len(sfr) - 1)
        if not live[i]:
            continue
        dfd = aid if att == hid else hid
        dside = 'home' if dfd == hid else 'away'
        x0, y0, per = ball.at(f0)
        x1, y1, _ = ball.at(f1)
        s = -dirs[per] if dside == 'home' else dirs[per]              # raw -> defending team's frame
        x0, y0, x1, y1 = x0 * s, y0 * s, x1 * s, y1 * s
        d = pos[dside][i]
        d = d[~np.isnan(d[:, 0])]
        if len(d) < 9 or not x0 - x1 >= MIN_GAIN:                   # NaN ball positions fail here too
            continue
        depth = np.sort(d[:, 0])
        beaten = int(((depth[:DEEPEST] < x0) & (depth[:DEEPEST] > x1)).sum())
        if beaten < MIN_BEATEN:
            continue
        out[dfd].append(dict(x0=round(float(x0), 2), y0=round(float(y0), 2), x1=round(float(x1), 2), y1=round(float(y1), 2),
                             beaten=beaten, behind=bool(x1 < depth[:4].mean()), passer=g['player'], shirt=g['shirt'],
                             lane=lane_of(y1), minute=round(float(P['clock_min'][i]), 1)))
    return out


def summarise(passes, live_s):
    who = Counter(p['passer'] for p in passes).most_common(TOP_N)
    lanes = Counter(p['lane'] for p in passes)
    return dict(conceded=len(passes), behind=sum(p['behind'] for p in passes),
                lanes={k: lanes.get(k, 0) for k in LANES}, top_passers=[f'{n} ({c})' for n, c in who],
                per_minute=round(len(passes) / (live_s / 60), 2), live_defending_s=round(live_s),
                seconds_per_pass=round(live_s / len(passes)) if passes else None, passes=passes)


# ---------------------------------------------------------------- drawing

HEIGHT = 21.13
BLOCK_TOP, BLOCK_STEP = 2.31, 9.02          # team label of the first block, distance between blocks
PITCH_X, PITCH_DY = 1.17, 1.19              # pitch left edge; pitch top below the team label
PITCH_W = 9.9                               # in, for 105 m
SIDE_X, BAR_X, BAR_W, BAR_H = 11.80, 14.46, 1.85, 0.13
ROW = 0.424
LEGEND_Y = 20.20
WHITE = INK


def arrow(ax, x0, y0, x1, y1, color, lw=1.2, zorder=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=0.5,head_width=0.2', mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=zorder, clip_on=False))


def pass_map(fig, top, passes, color):
    h = PITCH_W * 68 / 105
    ax = fig.axes(PITCH_X, top, PITCH_W, h)
    pitch(ax, thirds=True, arcs=False, lw=0.8)
    ax.set_xlim(-52.5, 52.5)
    ax.set_ylim(-34, 34)
    fig.text(PITCH_X + 0.11, top - 0.235, 'own goal', fontproperties=MONO, fontsize=9.5, color=FAINT)
    for p in sorted(passes, key=lambda p: p['behind']):                 # white arrows on top
        c = WHITE if p['behind'] else color
        ax.plot(p['x0'], p['y0'], 'o', ms=3.2, color=c, mec='none', zorder=5, clip_on=False)
        arrow(ax, p['x0'], p['y0'], p['x1'], p['y1'], c, zorder=5 if p['behind'] else 4)


def bar_list(fig, top, title, rows, color):
    """Title plus labelled horizontal bars scaled to the largest row."""
    fig.text(SIDE_X, top, title, fontproperties=MONO_MED, fontsize=12.5)
    vmax = max([v for _, v in rows] + [1])
    for k, (label, v) in enumerate(rows):
        y = top + 0.455 + k * ROW
        fig.text(SIDE_X, y, label, fontproperties=MONO, fontsize=11.5, color=DIM)
        w = BAR_W * v / vmax
        ax = fig.axes(BAR_X, y - BAR_H / 2, max(w, 0.001), BAR_H)
        ax.set_facecolor(color)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(BAR_X + w + 0.11, y, str(v), fontproperties=MONO, fontsize=11.5, color=INK)


def team_block(fig, mid, side, s, top):
    m = R.get(mid)
    team, opp = m[side], m['away' if side == 'home' else 'home']
    fig.team_label(1.17, top - 0.06, f"{team['name']} defending", team['color'], size=28, dot=0.095, gap=0.23, dot_dy=0.065)
    fig.text(1.08, top + 0.64, f"{opp['name']} played through {team['name']}'s lines {s['conceded']} times ({s['behind']} received "
                               f"behind the back line) · one every {s['seconds_per_pass']} seconds of live defending",
             fontproperties=MONO, fontsize=12, color=DIM)
    ptop = top + PITCH_DY
    pass_map(fig, ptop, s['passes'], opp['color'])
    who = [(t.rsplit(' (', 1)[0], int(t.rsplit('(', 1)[1][:-1])) for t in s['top_passers']]
    bar_list(fig, ptop - 0.01, 'WHO PLAYED THEM', who, opp['color'])
    lanes = [(('' if k == 'centre' else 'their ') + k, s['lanes'][k]) for k in LANES[::-1]]
    bar_list(fig, ptop + 2.90, 'WHERE THEY WERE RECEIVED', lanes, DIM)


def legend(fig, mid):
    ax = fig.axes(0, 0, 18, HEIGHT)
    ax.set_xlim(0, 18); ax.set_ylim(HEIGHT, 0); ax.axis('off')
    y = LEGEND_Y - 0.02
    arrow(ax, 1.10, y, 1.54, y, DIM, lw=1.2)
    t = fig.text(1.72, LEGEND_Y, 'through the lines, received in front of the back line (attacking team’s colour)',
                 fontproperties=MONO, fontsize=10.5, color=FAINT)
    x = fig.fig.transFigure.inverted().transform(t.get_window_extent(fig.fig.canvas.get_renderer()))[1, 0] * 18 + 0.35
    arrow(ax, x, y, x + 0.44, y, WHITE, lw=1.2)
    fig.text(x + 0.62, LEGEND_Y, 'received behind the back line (white)', fontproperties=MONO, fontsize=10.5, color=FAINT)
    notes = ['Line-breaking = bypasses 3+ of the defending team’s deepest seven outfielders (its back and midfield lines): '
             'between the ball and goal when played, behind it when received.',
             'Completed passes and crosses in live play moving the ball 5 m+ towards goal. Back line = the deepest four '
             'outfielders. ' + R.credit(mid, 'Source: PFF FC tracking + event feed.')]
    for i, s in enumerate(notes):
        fig.text(1.08, LEGEND_Y + 0.45 + 0.265 * i, s, fontproperties=MONO, fontsize=10.5, color=FAINT)


def draw(mid, summ):
    m = R.get(mid)
    fig = Fig(HEIGHT)
    fig.header('Line-breaking passes', [f"{m['title']} · {m['sub']}",
                                        "Completed passes through the defending team's midfield or back line · "
                                        "each defending team's own goal on the left"],
               x=1.08, size=12, step=0.25, title_size=46, gap=0.57)
    for k, side in enumerate(('home', 'away')):
        team_block(fig, mid, side, summ[side], BLOCK_TOP + k * BLOCK_STEP)
    legend(fig, mid)
    fig.save(out_path('analysis', 'lb', f"{m['slug']}.png"), out_path('analysis', 'lb', f"{m['slug']}.jpg"))


def run(mid):
    m = R.get(mid)
    P = load_pos(mid)
    passes = find_passes(mid, load_ev(mid), P)
    dfd = defending(P, mid)
    summ = {}
    for side in ('home', 'away'):
        live_s = sum(1 for lv, d in zip(P['live'], dfd) if lv and d == side) * SAMPLE_S
        summ[side] = summarise(passes[str(m[side]['id'])], live_s)
    draw(mid, summ)
    json.dump({m[side]['name']: summ[side] for side in ('home', 'away')},
              open(out_path('analysis', 'lb', f"{m['slug']}.json"), 'w'), ensure_ascii=False, indent=1)
    for side in ('home', 'away'):
        s = summ[side]
        print(mid, f"{m[side]['name']} conceded {s['conceded']} ({s['behind']} behind), one every {s['seconds_per_pass']} s · "
                   f"{s['lanes']} · {', '.join(s['top_passers'])}")


if __name__ == '__main__':
    for mid in sys.argv[1:]:
        run(mid)

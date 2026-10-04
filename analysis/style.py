"""Shared look for every analysis image: site palette, fonts, pitch drawing, figure helpers.

Images are 1800 px wide (18 in at 100 dpi), dark background, Barlow Condensed for headings and
IBM Plex Mono for text, matching the Match Library site.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Arc, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, 'fonts')
for f in os.listdir(FONT_DIR):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(FONT_DIR, f))

# palette (site tokens)
BG = '#0c1512'          # page background
PANEL = '#131f1a'
PITCH = '#1e3a2b'       # grass
INK = '#eef3ee'
DIM = '#93a89c'
FAINT = '#5e7268'
LINE = (0.86, 0.92, 0.88, 0.55)   # pitch markings
AMBER = '#ffb347'

L, W = 105.0, 68.0
WIDTH_IN, DPI = 18.0, 100


def _fp(path, size=None):
    return fm.FontProperties(fname=os.path.join(FONT_DIR, path), size=size)


# font properties (latin subset; matplotlib falls back to DejaVu for any glyph missing)
TITLE = _fp('barlow-condensed-latin-700.ttf')
SEMI = _fp('barlow-condensed-latin-600.ttf')
MED = _fp('barlow-condensed-latin-500.ttf')
MONO = _fp('ibm-plex-mono-latin-400.ttf')
MONO_MED = _fp('ibm-plex-mono-latin-500.ttf')
plt.rcParams['font.family'] = ['IBM Plex Mono', 'DejaVu Sans']


class Fig:
    """A figure laid out in inches from the top: fig.text(x, Y(inches), ...)."""

    def __init__(self, height_in):
        self.H = height_in
        self.fig = plt.figure(figsize=(WIDTH_IN, height_in), dpi=DPI, facecolor=BG)

    def Y(self, inches_from_top):
        return 1 - inches_from_top / self.H

    def axes(self, left_in, top_in, w_in, h_in):
        ax = self.fig.add_axes([left_in / WIDTH_IN, 1 - (top_in + h_in) / self.H, w_in / WIDTH_IN, h_in / self.H])
        ax.set_facecolor(BG)
        return ax

    def text(self, x_in, y_in, s, **kw):
        kw.setdefault('color', INK)
        return self.fig.text(x_in / WIDTH_IN, self.Y(y_in), s, va=kw.pop('va', 'center'), **kw)

    def header(self, title, lines, x=0.6, top=0.55, size=13, step=0.24, title_size=44, gap=0.62):
        """Big condensed title plus grey subtitle lines; returns the y (inches) below them."""
        self.text(x, top, title.upper(), fontproperties=TITLE, fontsize=title_size, va='center')
        y = top + gap
        for s in lines:
            self.text(x, y, s, fontproperties=MONO, fontsize=size, color=DIM)
            y += step
        return y

    def team_label(self, x, y, name, color, right=None, size=24, dot=0.07, gap=0.18, dot_dy=0.0):
        self.fig.patches.append(Circle((x / WIDTH_IN, self.Y(y + dot_dy)), dot / WIDTH_IN, transform=self.fig.transFigure,
                                       facecolor=color, edgecolor='none'))
        self.text(x + gap, y, name.upper(), fontproperties=SEMI, fontsize=size)
        if right:
            self.text(17.4, y, right, fontproperties=MONO, fontsize=13, color=DIM, ha='right')

    def team_chip(self, x, y, name, color, size=22, chip=0.17, dy=0.035, corner=0.3):
        """Team name in caps (centred on y) after a colour chip centred on (x, y + dy): a rounded square, or a
        circle with corner=0.5."""
        from matplotlib.patches import FancyBboxPatch
        h = chip / self.H
        w = chip / WIDTH_IN
        self.fig.patches.append(FancyBboxPatch((x / WIDTH_IN - w / 2, self.Y(y + dy) - h / 2), w, h,
                                               boxstyle=f'round,pad=0,rounding_size={corner * w}', mutation_aspect=h / w,
                                               transform=self.fig.transFigure, facecolor=color, edgecolor='none'))
        self.text(x + chip / 2 + 0.15, y, name.upper(), fontproperties=SEMI, fontsize=size)

    def footnotes(self, lines, y, color=None, x=0.6, step=0.27):
        for i, s in enumerate(lines):
            self.text(x, y + step * i, s, fontproperties=MONO, fontsize=10.5, color=color or (DIM if i < len(lines) - 1 else FAINT))

    def save(self, path_png, path_jpg=None, quality=88):
        self.fig.savefig(path_png, facecolor=BG, dpi=DPI)
        plt.close(self.fig)
        if path_jpg:
            from PIL import Image
            Image.open(path_png).convert('RGB').save(path_jpg, quality=quality, optimize=True)


def pitch(ax, L=L, W=W, grass=True, thirds=False, lw=1.0, color=LINE, half=None, arcs=True):
    """Draw a pitch centred on the spot (x in [-L/2, L/2], y in [-W/2, W/2]); arcs=False leaves out the D."""
    if grass:
        ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, facecolor=PITCH, edgecolor='none', zorder=0))
    kw = dict(fill=False, edgecolor=color, lw=lw, zorder=3)
    ax.add_patch(Rectangle((-L / 2, -W / 2), L, W, **kw))
    ax.plot([0, 0], [-W / 2, W / 2], color=color, lw=lw, zorder=3)
    ax.add_patch(Circle((0, 0), 9.15, **kw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * L / 2 - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((s * L / 2 - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, **kw))
        if arcs:
            ax.add_patch(Arc((s * (L / 2 - 11), 0), 18.3, 18.3, theta1=(127 if s > 0 else -53), theta2=(233 if s > 0 else 53), edgecolor=color, lw=lw, zorder=3))
    if thirds:
        for x in (-L / 6, L / 6):
            ax.plot([x, x], [-W / 2, W / 2], color=color, lw=lw * 0.7, ls=(0, (3, 4)), alpha=0.6, zorder=3)
    ax.set_xlim(-L / 2 - 0.5, L / 2 + 0.5) if half is None else ax.set_xlim(*half)
    ax.set_ylim(-W / 2 - 0.5, W / 2 + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

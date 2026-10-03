"""Parse a DFL (Sportec / TRACAB via DFL) positions XML into compact numpy arrays, one pass, streaming.

    python3 providers/dfl_parse.py <positions.xml> <out.npz>
"""
import re, sys, numpy as np
from datetime import datetime

HDR = re.compile(rb'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
FR = re.compile(rb'<Frame N="(\d+)" T="([^"]+)" X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")? D="[-\d.]+" S="([-\d.]+)"'
                rb'(?: A="[-\d.]+")?(?: M="-?\d+")?(?: BallPossession="(\d)")?(?: BallStatus="(\d)")?')


def ts(b):
    return datetime.fromisoformat(b.decode()).timestamp()


def parse(path):
    sets = {}; cur = None; rows = []
    def flush():
        if cur is not None:
            sets[cur] = rows
    with open(path, 'rb') as f:
        for line in f:
            if line.startswith(b'<Frame '):
                m = FR.match(line)
                if m: rows.append(m.groups())
            elif line.startswith(b'<FrameSet'):
                flush(); m = HDR.match(line); cur = tuple(x.decode() for x in m.groups()); rows = []
        flush()
    out = {}
    for (sec, team, pid), r in sets.items():
        n = np.array([int(x[0]) for x in r], np.int64)
        t0 = ts(r[0][1]); t = t0 + (n - n[0]) * 0.04          # frames are exactly 25 Hz; T checked below
        tl = ts(r[-1][1])
        if abs((t[-1] - tl)) > 0.05: raise SystemExit(f'{sec} {pid}: frame times are not a regular 25 Hz grid ({t[-1]-tl:.3f}s drift)')
        key = f'{sec}|{team}|{pid}'
        out[key + '|n'] = n; out[key + '|t0'] = np.array([t0])
        out[key + '|x'] = np.array([float(x[2]) for x in r], np.float32)
        out[key + '|y'] = np.array([float(x[3]) for x in r], np.float32)
        out[key + '|s'] = np.array([float(x[5]) for x in r], np.float32)
        if team == 'BALL':
            out[key + '|z'] = np.array([float(x[4] or 0) for x in r], np.float32)
            out[key + '|poss'] = np.array([int(x[6] or 0) for x in r], np.int8)
            out[key + '|status'] = np.array([int(x[7] or 0) for x in r], np.int8)
    return out


if __name__ == '__main__':
    out = parse(sys.argv[1]); np.savez_compressed(sys.argv[2], **out)
    print('framesets', len({k.rsplit('|', 1)[0] for k in out}), 'saved', sys.argv[2])

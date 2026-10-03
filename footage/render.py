"""Pass 5: draw the defensive read onto the original broadcast frames.

Everything tied to the pitch (heatmap, inter-line band, marking zone,
pressure ring, player markers) is drawn in pitch coordinates and projected
through that frame's camera homography, then painted only onto grass pixels,
so players and pitch lines stay on top -- the same way broadcast graphics sit
"under" the players. The camera parameters are interpolated between the 10 Hz
calibration samples so the graphics stay locked to the pitch at 30 fps.

Status handling (agreed with the user):
  live + camera locked  -> full overlay + read-out panel
  live, no lock         -> video untouched, badge "Broadcast view - no tactical read"
  replay / pre-kickoff  -> video untouched, badge "Replay - not live" / "Pre-kickoff"
Identities and the transition state are re-initialised after every cut.
"""
import cv2, numpy as np, pickle, subprocess, sys, math
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter
from calib import homography, project, HL, HW
from model_footage import team_family

FONT_DIR = '/home/claude/project_work/footage/models/fonts'
F_DISPLAY = f'{FONT_DIR}/fontsource-barlow-condensed-5.3.0/files/barlow-condensed-latin-700-normal.woff'
F_DISPLAY_SEMI = f'{FONT_DIR}/fontsource-barlow-condensed-5.3.0/files/barlow-condensed-latin-600-normal.woff'
F_MONO = f'{FONT_DIR}/fontsource-ibm-plex-mono-5.3.0/files/ibm-plex-mono-latin-500-normal.woff'
C = np.load('/home/claude/project_work/footage/camera_C.npy')
LIVE_START, LIVE_END = 8.4, 307.7
CLOCK_OFFSET = 8.4

# site tokens (artifact_head_10515.html)
INK, INK_DIM, INK_FAINT = (238, 243, 238), (147, 168, 156), (94, 114, 104)
PANEL, PANEL_RAISED, AMBER = (19, 31, 26), (23, 37, 31), (255, 176, 32)
TEAM_RGB = {'FRA': (47, 111, 224), 'MAR': (232, 50, 58)}
TEAM_NAME = {'FRA': 'FRANCE', 'MAR': 'MOROCCO'}
PHASE_RGB = {'High Press': (232, 50, 58), 'Mid Block': (217, 154, 43), 'Low Block': (63, 143, 92),
             'Transition': (181, 86, 58)}
TIGHT_RGB, LOOSE_RGB = (91, 141, 238), (147, 168, 156)


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


# ----------------------------------------------------------------- read-out
_font_cache = {}


def font(path, size):
    k = (path, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]


PANEL_W, PANEL_H = 318, 168
_hud_cache = {}


def hud(state):
    """Read-out panel as an RGBA array. state is a hashable tuple."""
    if state in _hud_cache:
        return _hud_cache[state]
    kind = state[0]
    im = Image.new('RGBA', (PANEL_W, PANEL_H if kind == 'read' else 62), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, im.width - 1, im.height - 1], 6, fill=PANEL + (228,), outline=(42, 58, 51, 255))
    mono10, mono12 = font(F_MONO, 11), font(F_MONO, 13)
    d.text((12, 9), 'DEFENSIVE STRUCTURE MONITOR', font=mono10, fill=INK_FAINT)
    if kind != 'read':
        _, clock, msg, sub = state
        if clock:
            d.text((im.width - 12, 9), clock, font=mono10, fill=INK_DIM, anchor='ra')
        d.text((12, 27), msg, font=font(F_DISPLAY, 21), fill=AMBER)
        _hud_cache[state] = np.array(im)
        return _hud_cache[state]
    (_, clock, dteam, phase, engaged, marking, press, depth, length, nvis, note) = state
    d.text((im.width - 12, 9), clock, font=mono10, fill=INK_DIM, anchor='ra')
    # defending team
    d.ellipse([12, 33, 22, 43], fill=TEAM_RGB[dteam])
    d.text((30, 25), f'{TEAM_NAME[dteam]} DEFENDING', font=font(F_DISPLAY_SEMI, 21), fill=INK)
    # phase pill + chips
    x = 12
    y = 56
    if phase:
        w = d.textlength(phase.upper(), font=font(F_DISPLAY, 19)) + 18
        d.rounded_rectangle([x, y, x + w, y + 26], 4, fill=PHASE_RGB[phase])
        d.text((x + 9, y + 2), phase.upper(), font=font(F_DISPLAY, 19), fill=(255, 255, 255))
        x += w + 6
    else:
        lbl = 'NOT ALL 10 IN SHOT'
        w = d.textlength(lbl, font=font(F_DISPLAY_SEMI, 16)) + 16
        d.rounded_rectangle([x, y, x + w, y + 26], 4, outline=INK_FAINT)
        d.text((x + 8, y + 4), lbl, font=font(F_DISPLAY_SEMI, 16), fill=INK_DIM)
        x += w + 6
    for label, on_rgb in ((engaged, None), (marking, None)):
        if not label:
            continue
        fnt = font(F_DISPLAY_SEMI, 16)
        w = d.textlength(label, font=fnt) + 14
        if label == 'ENGAGED':
            d.rounded_rectangle([x, y + 2, x + w, y + 24], 4, fill=AMBER)
            d.text((x + 7, y + 4), label, font=fnt, fill=(28, 22, 8))
        elif label == 'PASSIVE':
            d.rounded_rectangle([x, y + 2, x + w, y + 24], 4, outline=AMBER, width=1)
            d.text((x + 7, y + 4), label, font=fnt, fill=AMBER)
        elif label == 'TIGHT':
            d.rounded_rectangle([x, y + 2, x + w, y + 24], 4, fill=TIGHT_RGB)
            d.text((x + 7, y + 4), label, font=fnt, fill=(255, 255, 255))
        else:
            d.rounded_rectangle([x, y + 2, x + w, y + 24], 4, outline=LOOSE_RGB, width=1)
            for k in range(-24, int(w), 6):
                d.line([(x + k, y + 24), (x + k + 22, y + 2)], fill=LOOSE_RGB + (70,), width=1)
            d.text((x + 7, y + 4), label, font=fnt, fill=INK)
        x += w + 6
    # pressure bar
    y = 94
    d.text((12, y), 'PRESSURE ON BALL', font=mono10, fill=INK_DIM)
    bx0, bx1 = 132, im.width - 52
    d.rounded_rectangle([bx0, y + 3, bx1, y + 11], 3, fill=(38, 52, 46))
    if press is not None:
        d.rounded_rectangle([bx0, y + 3, bx0 + (bx1 - bx0) * press, y + 11], 3, fill=AMBER)
        d.text((im.width - 12, y), f'{press:.2f}', font=mono12, fill=INK, anchor='ra')
    else:
        d.text((im.width - 12, y), '—', font=mono12, fill=INK_FAINT, anchor='ra')
    # depth / length
    y = 116
    dep = f'{depth:.1f} m' if depth is not None else 'not read'
    ln = f'{length:.1f} m' if length is not None else '—'
    d.text((12, y), 'BACK-LINE DEPTH', font=mono10, fill=INK_DIM)
    d.text((im.width - 12, y), dep, font=mono12, fill=INK if depth is not None else INK_FAINT, anchor='ra')
    y = 134
    d.text((12, y), 'TEAM LENGTH', font=mono10, fill=INK_DIM)
    d.text((im.width - 12, y), ln, font=mono12, fill=INK if length is not None else INK_FAINT, anchor='ra')
    y = 151
    d.text((12, y), note, font=font(F_MONO, 10), fill=INK_FAINT)
    _hud_cache[state] = np.array(im)
    return _hud_cache[state]


def paste_rgba(frame, rgba, x, y):
    h, w = rgba.shape[:2]
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    src = rgba[:, :, 2::-1].astype(np.float32)
    frame[y:y + h, x:x + w] = (roi * (1 - a) + src * a).astype(np.uint8)


# --------------------------------------------------------- pitch graphics
def proj2(pts, params):
    P = np.c_[np.asarray(pts, float), np.zeros(len(pts))]
    uv, z = project(P, C, *params)
    return uv, z


def ground_circle(cx, cy, r, n=40):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def dashed_poly(img, pts, color, thick, dash=10, gap=7):
    """Closed dashed polyline drawn by walking cumulative arc length."""
    pts = np.asarray(pts, float)
    seg = np.r_[pts, pts[:1]]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
    total = d[-1]
    if total <= 0:
        return
    period = dash + gap
    starts = np.arange(0, total, period)
    for s0 in starts:
        s1 = min(s0 + dash, total)
        ss = np.linspace(s0, s1, max(int((s1 - s0) / 3) + 2, 2))
        xs = np.interp(ss, d, seg[:, 0]); ys = np.interp(ss, d, seg[:, 1])
        cv2.polylines(img, [np.round(np.c_[xs, ys]).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def hatch_fill(layer, alpha, poly, color, a_val, spacing=9):
    mask = np.zeros(layer.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    x, y, w, h = cv2.boundingRect(mask)
    hatch = np.zeros_like(mask)
    for k in range(-h, w + h, spacing):
        cv2.line(hatch, (x + k, y + h), (x + k + h, y), 255, 1, cv2.LINE_AA)
    m = (hatch > 0) & (mask > 0)
    layer[m] = color
    alpha[m] = np.maximum(alpha[m], a_val)


def grass_alpha(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (30, 50, 35), (88, 255, 255))
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g.astype(np.float32) / 255.0


# ------------------------------------------------------------------ prep
def prep(model, positions, calib):
    samples = {r['i']: r for r in model}
    pos = {r['i']: r for r in positions}
    live = {r['i']: r['live'] for r in calib}
    # smooth band edges within each run of readable samples
    runs = {}
    for r in positions:
        if r['params'] is not None:
            runs.setdefault(r['run'], []).append(r['i'])
    for run, idx in runs.items():
        for key in ('dmin_x', 'dmax_x'):
            known = [(k, samples[k][key]) for k in idx if key in samples[k]]
            if len(known) < 3:
                continue
            ks = np.array([k for k, _ in known]); vs = np.array([v for _, v in known])
            if len(vs) >= 7:
                vs = savgol_filter(vs, 7, 2)
            for k, v in zip(ks, vs):
                samples[k][key + '_sm'] = float(v)
    return samples, pos, live


def draw_frame(frame, i, samples, pos, sorted_idx):
    import bisect
    j = bisect.bisect_right(sorted_idx, i) - 1
    if j < 0:
        return frame, ('status', '', 'PRE-KICKOFF', '')
    s0 = sorted_idx[j]
    s1 = sorted_idx[j + 1] if j + 1 < len(sorted_idx) else s0
    r0 = samples[s0]
    t = r0['t'] + (i - s0) / 29.814
    clock_s = t - CLOCK_OFFSET
    clock = f'{int(clock_s // 60):02d}:{int(clock_s % 60):02d}' if clock_s >= 0 else ''
    if t < LIVE_START:
        return frame, ('status', clock, 'PRE-KICKOFF', '')
    if t > LIVE_END:
        return frame, ('status', '', 'REPLAY — NOT LIVE', '')
    p0 = pos[s0]['params']
    if p0 is None:
        return frame, ('status', clock, 'BROADCAST VIEW — NO TACTICAL READ', '')
    p1 = pos[s1]['params']
    a = (i - s0) / max(s1 - s0, 1)
    if p1 is not None and pos[s1]['run'] == pos[s0]['run'] and s1 != s0:
        params = (1 - a) * p0 + a * p1
    else:
        params = p0
    r = samples[s1] if (a > 0.5 and pos[s1]['params'] is not None and pos[s1]['run'] == pos[s0]['run']) else r0
    fr_pos = pos[r['i']]
    H = homography(C, *params)

    layer = np.zeros_like(frame)
    alpha = np.zeros(frame.shape[:2], np.float32)
    dteam = r.get('def')
    # 1. heatmap (nearest 5 Hz raster in the same run, within 0.35 s)
    heat_r = None
    for k in (r['i'], r['i'] - 3, r['i'] + 3, r['i'] - 6, r['i'] + 6):
        if k in samples and 'heat' in samples[k] and pos[k]['run'] == fr_pos['run']:
            heat_r = samples[k]; break
    if heat_r is not None and heat_r.get('poss'):
        raster = heat_r['heat'].T.astype(np.float32) / 255.0      # rows = Y, cols = X
        Hr, Wr = 170, 260
        big = cv2.resize(raster, (Wr, Hr), interpolation=cv2.INTER_CUBIC)
        S = np.array([[105 / Wr, 0, -HL + 0.5 * 105 / Wr], [0, 68 / Hr, -HW + 0.5 * 68 / Hr], [0, 0, 1]])
        M = H @ S
        warped = cv2.warpPerspective(big, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
        col = TEAM_RGB[heat_r['poss']]
        a_h = np.clip(warped, 0, 1) * 0.5
        m = a_h > alpha
        layer[m] = bgr(col)
        alpha = np.maximum(alpha, a_h)
    # 2. inter-line band (organised defence only, both edges readable)
    phase = r.get('phase')
    if dteam and phase in ('High Press', 'Mid Block', 'Low Block') and 'dmin_x_sm' in r and 'dmax_x_sm' in r and 'inter_line' in r:
        x0, x1 = r['dmin_x_sm'], r['dmax_x_sm']
        quad = np.array([[x0, -HW], [x1, -HW], [x1, HW], [x0, HW]])
        uv, z = proj2(quad, params)
        if (z > 1).all():
            poly = np.round(uv).astype(np.int32)
            band = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(band, [poly], 255)
            bm = band > 0
            layer[bm] = (0.55 * layer[bm] + 0.45 * np.array(bgr(TEAM_RGB[dteam]))).astype(np.uint8)
            alpha[bm] = np.maximum(alpha[bm], 0.22)
            for xe in (x0, x1):
                e_uv, _ = proj2([[xe, -HW], [xe, HW]], params)
                p1, p2 = tuple(np.round(e_uv[0]).astype(int)), tuple(np.round(e_uv[1]).astype(int))
                cv2.line(layer, p1, p2, bgr(TEAM_RGB[dteam]), 3, cv2.LINE_AA)
                cv2.line(alpha, p1, p2, 0.95, 3, cv2.LINE_AA)
    # 3. marking zone: hull of the 5 defenders nearest the ball
    if dteam and 'hull' in r and r.get('marking'):
        pts = np.array(r['hull'])
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        # 1.2 m outward buffer so a tight group still reads as an area
        cen = hull.mean(0)
        dirs = hull - cen
        nrm = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
        hull = hull + dirs / nrm * 1.2
        uv, z = proj2(hull, params)
        if (z > 1).all():
            if r['marking'] == 'TIGHT':
                zm = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(zm, [np.round(uv).astype(np.int32)], 255)
                m = zm > 0
                layer[m] = bgr(TIGHT_RGB)
                alpha[m] = np.maximum(alpha[m], 0.30)
            else:
                hatch_fill(layer, alpha, uv, bgr(LOOSE_RGB), 0.32, spacing=13)
            col = TIGHT_RGB if r['marking'] == 'TIGHT' else LOOSE_RGB
            poly = [np.round(uv).astype(np.int32)]
            cv2.polylines(layer, poly, True, bgr(col), 2, cv2.LINE_AA)
            cv2.polylines(alpha, poly, True, 0.9, 2, cv2.LINE_AA)
    # 4. player markers (interpolated between samples by track id)
    nxt = {pp['track']: pp for pp in pos[s1]['people']} if (p1 is not None and pos[s1]['run'] == pos[s0]['run']) else {}
    carrier_xy = None
    for pp in pos[s0]['people']:
        if pp.get('n_obs', 0) < 3:
            continue
        x, y = pp['x'], pp['y']
        if pp['track'] in nxt:
            q = nxt[pp['track']]
            x, y = (1 - a) * x + a * q['x'], (1 - a) * y + a * q['y']
        fam = team_family(pp['team'])
        if pp['track'] == r.get('carrier'):
            carrier_xy = (x, y)
        if fam is None:
            continue
        uv, z = proj2(ground_circle(x, y, 0.85, 24), params)
        if (z > 1).all():
            poly = [np.round(uv).astype(np.int32)]
            cv2.polylines(layer, poly, True, bgr(TEAM_RGB[fam]), 2, cv2.LINE_AA)
            cv2.polylines(alpha, poly, True, 0.95, 2, cv2.LINE_AA)
    # 5. pressure ring round the ball carrier
    press = r.get('pressure_s')
    if carrier_xy is not None and press is not None and dteam:
        uv, z = proj2(ground_circle(carrier_xy[0], carrier_xy[1], 2.3, 48), params)
        if (z > 1).all():
            col = tuple(int(INK_DIM[c] + (AMBER[c] - INK_DIM[c]) * min(press / 0.67, 1)) for c in range(3))
            if press >= 0.85:
                col = PHASE_RGB['High Press']
            if r.get('engaged'):
                poly = [np.round(uv).astype(np.int32)]
                cv2.polylines(layer, poly, True, bgr(col), 3, cv2.LINE_AA)
                cv2.polylines(alpha, poly, True, 0.95, 3, cv2.LINE_AA)
            else:
                dashed_poly(layer, uv, bgr(col), 2)
                dashed_poly(alpha, uv, 0.95, 2)

    # composite on grass only, so players and lines stay on top
    ga = grass_alpha(frame)
    a3 = (alpha * ga)[:, :, None]
    out = (frame.astype(np.float32) * (1 - a3) + layer.astype(np.float32) * a3).astype(np.uint8)

    if not dteam:
        return out, ('status', clock, 'READING POSSESSION…', '')
    n_def = r.get('n_def', 0)
    if r.get('bridged'):
        note = 'all 10 in shot (one hidden for a moment)'
    elif n_def == 10 and 'depth' in r:
        note = f'all 10 {TEAM_NAME[dteam].title()} outfield players in shot'
    elif n_def == 10:
        note = 'all 10 in shot, but pitch behind them is cut off'
    else:
        note = f'{min(n_def, 10)}/10 in shot — need all 10 to read the shape'
    state = ('read', clock, dteam, phase if phase else None,
             ('ENGAGED' if r.get('engaged') else 'PASSIVE') if 'engaged' in r else None,
             r.get('marking'),
             None if press is None else round(press, 2),
             None if 'depth_s' not in r else round(r['depth_s'], 1),
             None if 'inter_line_s' not in r else round(r['inter_line_s'], 1),
             n_def, note)
    return out, state


def main(t_from=0.0, t_to=1e9, out_path='overlay.mp4'):
    model = pickle.load(open('model_out.pkl', 'rb'))
    positions = pickle.load(open('positions.pkl', 'rb'))
    calib = pickle.load(open('calib.pkl', 'rb'))
    samples, pos, live = prep(model, positions, calib)
    sorted_idx = sorted(samples)
    cap = cv2.VideoCapture('clip.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720',
                           '-r', f'{fps:.5f}', '-i', '-', '-ss', f'{t_from:.3f}', '-t', f'{t_to - t_from:.3f}',
                           '-i', 'clip.mp4', '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'faster',
                           '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-shortest', out_path],
                          stdin=subprocess.PIPE)
    i = 0
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        if t >= t_to:
            break
        if t >= t_from:
            out, state = draw_frame(f, i, samples, pos, sorted_idx)
            panel = hud(state)
            paste_rgba(out, panel, 1280 - panel.shape[1] - 14, 70)
            ff.stdin.write(out.tobytes())
            n += 1
            if n % 600 == 0:
                print(f'rendered {n} frames (t={t:.1f}s)', flush=True)
        i += 1
    ff.stdin.close()
    ff.wait()
    print('DONE', n, 'frames ->', out_path, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.0, float(a[1]) if len(a) > 1 else 1e9, a[2] if len(a) > 2 else 'overlay.mp4')

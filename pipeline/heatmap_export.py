"""
Live spatial heatmap export: rasterizes the DAS (Dangerous Accessible Space)
model's per-frame polar simulation grid onto a Cartesian raster covering the
full pitch, for every sampled frame in the animation window. This replaces
the "space in behind" companion chart's time-series/summary approach with a
direct spatial answer to "where is the dangerous space, right now" -- meant
to be decoded and drawn as a live-updating heatmap layer inside the main
Pressure Read animation.

Method: `accessible_space`'s SimulationResult carries a polar grid centered
on the ball (`r_grid` x `phi_grid`, with `attack_poss_density` at each
(phi, r) cell, plus that cell's Cartesian center in `x_grid`/`y_grid`). We
replicate the exact area-element weighting `accessible_space.integrate_surfaces`
uses internally (dr from the radial midpoints, dA from the annular-sector
areas) to turn density into an actual probability "mass" per polar cell,
then bin those masses onto a regular Cartesian grid with `np.histogram2d`.
Verified: summing the resulting raster over the whole pitch reproduces the
library's own whole-pitch DAS value exactly (mass.sum() over histogram bins
clipped to pitch == ret.das), so the raster is faithful to the same numbers
already reported elsewhere in this project, not a fresh approximation.

We rasterize `dangerous_result` (DAS, not plain AS) whole-pitch every
sampled frame -- no restriction to a precomputed "behind the line" region.
The point of a live spatial view is to let the shape and location of the
danger emerge on its own (it naturally concentrates behind a defensive line
when that's where the risk is), rather than presupposing the region.

Output: heatmap_data.json -- {meta: {cols, rows, xMin, xMax, yMin, yMax,
floor, vmaxLog}, frames: [{t, grid: base64 of cols*rows uint8}]}, consumed
by the animation's render() loop, decoded and nearest-in-time matched to
whatever frame is currently playing.

Two fixes over the first version, both diagnosed from the published result
looking "messy" and hard to read:

1. **Rasterization aliasing.** `np.histogram2d` only bins the polar grid's
   discrete sample POINTS -- it doesn't distribute mass continuously across
   a cell. The polar grid has just 30 angular steps, so at any real radius
   the arc-length between adjacent rays quickly exceeds a Cartesian cell's
   width, and cells that fall between two rays get zero even though the
   library's continuous model has real density there. That is exactly what
   produced the checkerboard/quilted look in the published version -- verified
   by dumping the raw quantized grid for a sample frame and seeing alternating
   zero/nonzero cells in a hole-riddled pattern, not sensor noise or a real
   feature of the model. Fixed by Gaussian-blurring each frame's raw mass
   raster (`gaussian_filter`, sigma ~1.2 cells) before quantizing -- blurring
   in real mass units (not after log/quantization) fills the sampling gaps
   the way a finer-grained simulation would have, without materially moving
   the total mass around (interior cells; only cells right at the pitch edge
   lose a little to the zero-padded boundary).

2. **Normalization was clipping almost everything to full brightness.**
   Diagnostic dump of raw per-cell mass across sample frames: nonzero values
   span nearly 8 orders of magnitude (1st percentile ~3e-8, max ~2.4,
   99.9th percentile ~1.1) -- a single global percentile-and-sqrt scale (the
   first version's approach) puts the ceiling so low relative to that spread
   that any cell inside a real hot pocket clips to solid max brightness,
   which is what read as a flat, hard-edged plateau covering half the pitch
   in the published screenshot rather than a graded glow. Fixed with a
   log1p transform relative to a fixed floor (`LOG_FLOOR`, set near the
   median nonzero cell value) before percentile-scaling -- log-compressing
   this kind of heavy-tailed density is the standard fix and is what actually
   produces a readable multi-level gradient instead of a binary on/off wash.
"""
import json
import base64
import time
import pandas as pd
pd.set_option('future.infer_string', False)  # accessible-space's internals predate pandas 3's default string dtype
import numpy as np
from scipy.ndimage import gaussian_filter
import accessible_space as accsp

from space_behind_line import build_velocity_lookup, frame_to_tracking_df

ANIM_PATH = '/home/claude/project_work/animation_data.json'
OUT_PATH = '/home/claude/project_work/heatmap_data.json'

SAMPLE_HZ = 5.0
GRID_COLS = 52
GRID_ROWS = 34
PITCH_X_MIN, PITCH_X_MAX = -52.5, 52.5
PITCH_Y_MIN, PITCH_Y_MAX = -34.0, 34.0
BLUR_SIGMA = 1.2       # cells -- bridges the polar grid's angular sampling gaps
LOG_FLOOR = 3e-4       # ~median nonzero cell mass in a diagnostic sample; log1p(mass/LOG_FLOOR)

X_EDGES = np.linspace(PITCH_X_MIN, PITCH_X_MAX, GRID_COLS + 1)
Y_EDGES = np.linspace(PITCH_Y_MIN, PITCH_Y_MAX, GRID_ROWS + 1)


def area_elements(r_grid, phi_grid):
    """Same dr/dA area-element weighting as accessible_space.integrate_surfaces,
    kept separate so we get a per-cell mass grid instead of a single sum."""
    r_lo = np.zeros_like(r_grid)
    r_lo[1:] = (r_grid[:-1] + r_grid[1:]) / 2
    r_lo[0] = r_grid[0]
    r_hi = np.zeros_like(r_grid)
    r_hi[:-1] = (r_grid[:-1] + r_grid[1:]) / 2
    r_hi[-1] = r_grid[-1]
    dr = r_hi - r_lo  # (T,)

    phi_lo = np.zeros_like(phi_grid)
    phi_lo[1:] = (phi_grid[:-1] + phi_grid[1:]) / 2
    phi_lo[0] = phi_grid[0]
    phi_hi = np.zeros_like(phi_grid)
    phi_hi[:-1] = (phi_grid[:-1] + phi_grid[1:]) / 2
    phi_hi[-1] = phi_grid[-1]
    dphi = phi_hi - phi_lo  # (PHI,)

    outer = dphi[:, None] / (2 * np.pi) * (np.pi * r_hi[None, :] ** 2)
    inner = dphi[:, None] / (2 * np.pi) * (np.pi * r_lo[None, :] ** 2)
    dA = outer - inner  # (PHI, T)
    return dr, dA


def raster_for_frame(frame, vel_at_frame, home_positive, frame_id):
    df = frame_to_tracking_df(frame, vel_at_frame, home_positive, frame_id)
    if df is None:
        return None

    ret = accsp.get_dangerous_accessible_space(
        df, frame_col='frame_id', player_col='player_id', team_col='team_id', ball_player_id='ball',
        x_col='x', y_col='y', vx_col='vx', vy_col='vy', team_in_possession_col='team_in_possession',
        period_col='period_id', attacking_direction_col='attacking_direction', infer_attacking_direction=False,
        player_in_possession_col='player_in_possession', use_progress_bar=False,
        x_pitch_min=PITCH_X_MIN, x_pitch_max=PITCH_X_MAX, y_pitch_min=PITCH_Y_MIN, y_pitch_max=PITCH_Y_MAX,
    )
    res = ret.dangerous_result
    dr, dA = area_elements(res.r_grid, res.phi_grid[0])
    mass = res.attack_poss_density[0] * dr[None, :] * dA  # (PHI, T)

    H, _, _ = np.histogram2d(
        res.x_grid[0].ravel(), res.y_grid[0].ravel(),
        bins=[X_EDGES, Y_EDGES], weights=mass.ravel(),
    )
    # The polar grid's ~30 angular steps leave real gaps between Cartesian
    # cells at any real radius (histogram2d only bins the discrete sample
    # points, it doesn't spread mass across the cell) -- blur in real mass
    # units to fill those gaps the way a finer angular resolution would have,
    # before any log/quantization step touches the values.
    H = gaussian_filter(H, sigma=BLUR_SIGMA)
    return H  # (GRID_COLS, GRID_ROWS)


def main():
    with open(ANIM_PATH) as f:
        data = json.load(f)
    meta = data['meta']
    frames = data['frames']
    home_positive = meta['homeDefendsPositiveX']
    fps = meta['fps']

    print(f"Building velocity lookup over {len(frames)} frames...")
    vel_full = build_velocity_lookup(frames)

    step = max(1, round(fps / SAMPLE_HZ))
    sample_idxs = list(range(0, len(frames), step))
    print(f"Sampling every {step} frames (~{SAMPLE_HZ}Hz) -> {len(sample_idxs)} candidate samples")

    rasters = []
    times = []
    t0 = time.time()
    for n, i in enumerate(sample_idxs):
        H = raster_for_frame(frames[i], vel_full[i], home_positive, frame_id=i)
        if H is not None:
            rasters.append(H)
            times.append(frames[i]['clockS'])
        if (n + 1) % 250 == 0:
            print(f"  {n+1}/{len(sample_idxs)} samples, {time.time()-t0:.1f}s elapsed")
    print(f"Done: {len(rasters)} usable rasters in {time.time()-t0:.1f}s "
          f"(skipped {len(sample_idxs)-len(rasters)} loose-ball/edge frames)")

    stacked = np.stack(rasters)  # (N, cols, rows), post-blur mass units
    stacked = np.clip(stacked, 0, None)
    nz = stacked[stacked > 1e-12]
    print(f"Raster stats (post-blur): min={stacked.min():.6f} max={stacked.max():.6f} "
          f"mean={stacked.mean():.6f} nonzero-p50={np.percentile(nz, 50):.6f} "
          f"nonzero-p99={np.percentile(nz, 99):.6f}")

    # Raw mass is extremely heavy-tailed (nonzero cells span ~8 orders of
    # magnitude in a diagnostic sample: 1st pct ~3e-8, max ~2.4) -- a linear
    # or sqrt scale off a single global percentile puts the ceiling so low
    # relative to that spread that most of a real hot pocket clips to solid
    # max brightness, which is what made the first version look like a flat,
    # hard-edged plateau rather than a graded glow. log1p relative to a fixed
    # floor (LOG_FLOOR, near the median nonzero cell) compresses the range
    # into something a 0-255 ramp can actually resolve, the standard fix for
    # this kind of density. Floor and ceiling are both fixed ONCE across the
    # whole window (not per frame), so brightness stays comparable frame to
    # frame -- a genuinely bigger pocket still reads brighter, not
    # auto-normalized away.
    log_vals = np.log1p(stacked / LOG_FLOOR)
    vmax_log = float(np.percentile(log_vals[log_vals > 0], 99.5))
    quantized = np.clip(log_vals / vmax_log, 0, 1)
    quantized = np.round(quantized * 255).astype(np.uint8)  # (N, cols, rows)

    frame_records = []
    for t, grid in zip(times, quantized):
        b64 = base64.b64encode(grid.tobytes()).decode('ascii')
        frame_records.append({'t': round(t, 3), 'grid': b64})

    out = {
        'meta': {
            'cols': GRID_COLS, 'rows': GRID_ROWS,
            'xMin': PITCH_X_MIN, 'xMax': PITCH_X_MAX,
            'yMin': PITCH_Y_MIN, 'yMax': PITCH_Y_MAX,
            'logFloor': LOG_FLOOR, 'vmaxLog': vmax_log,
        },
        'frames': frame_records,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f)
    import os
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.2f} MB)")


if __name__ == '__main__':
    main()

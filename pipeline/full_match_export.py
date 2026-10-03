"""
Extends the animation export from one hand-picked 5-minute window to the
WHOLE match, split into ~5-minute chapters aligned to period boundaries.

Why chunked rather than one continuous page: a single published page has a
16MB hard cap. At the original window's fidelity (native ~30fps player
positions, 5Hz/52x34 heatmap), the whole match's animation data alone would
run roughly 276MB and the heatmap another ~89MB -- both wildly over budget,
and true regardless of how many periods are included. The user's explicit
call (given that math): keep today's exact fidelity/quality, and split the
match into a sequence of chapter pages instead of degrading quality to fit
one page. This script produces the per-chapter data; chapter HTML assembly
and publishing happen separately (see full_match_build.py).

Key difference from the original single-window `animation_export.py`: that
script did a full pass over the tracking file FILTERED to one frame range.
Doing that once per chapter (26 chapters) would mean 26 full decompressions
of the ~62MB tracking file (~1 minute each) for no benefit, and would also
Kalman-smooth each chapter's ball/player trajectories in isolation --
introducing a fresh discontinuity right at every chapter boundary since the
filter has no context before/after its own window. Instead this script:

  1. Reads the tracking file ONCE, collecting raw samples for the WHOLE
     match, kept separate per period (periods don't share a frame axis
     smoothing should cross, since each period is its own continuous shot).
  2. Runs the exact same two-stage cleaning (`reject_speed_spikes_iterative`
     + `kalman_rts_smooth` + `interpolate_series`, imported unchanged from
     `animation_export.py`) ONCE per period, over that period's FULL raw
     trajectory -- so the filter has proper context throughout and a chapter
     boundary falls at a normal point mid-trajectory, not at a smoothing
     seam.
  3. Slices the resulting smoothed/interpolated series into chapters only
     at the very end, purely by frame range -- smoothing quality is
     identical to what a from-scratch single-window run would have produced
     for that same span.

Chapter boundaries are chosen by TARGET TIME (periodElapsedTime), then
snapped to the nearest actual frame -- never by naively multiplying a
target minute by fps, for the same reason `animation_export.py`'s docstring
flags: frameNum (video-relative) and periodElapsedTime (true broadcast match
clock) are not the same axis in this dataset. Each period is split into N
chapters of roughly equal length (N chosen to land closest to a 5-minute
target -- see CHAPTER_TARGET_S), rather than fixed 5-minute chunks with a
short leftover chapter at the end of each period.
"""
import bz2
import json
import os
import time
import pandas as pd
from config import (TRACKING_PATH, HOME_TEAM_ID, AWAY_TEAM_ID, HOME_TEAM_NAME,
                     AWAY_TEAM_NAME, HOME_COLOR, HOME_COLOR_DIM, AWAY_COLOR,
                     AWAY_COLOR_DIM, GAME_ID, CHAPTERS_DIR, pkl_path,
                     load_roster, load_metadata, orientation_lookup)
from animation_export import (reject_speed_spikes_iterative, kalman_rts_smooth,
                               interpolate_series, BALL_MAX_SPEED_MS, PLAYER_MAX_SPEED_MS,
                               BALL_MAX_GAP_S, PLAYER_MAX_GAP_S, FPS_NOMINAL,
                               BALL_PROCESS_NOISE_STD, BALL_MEASUREMENT_NOISE_STD,
                               PLAYER_PROCESS_NOISE_STD, PLAYER_MEASUREMENT_NOISE_STD)

CHAPTER_TARGET_S = 300.0  # ~5 minutes, matching the original window's length
PERIOD_LABEL = {1: '1st half', 2: '2nd half', 3: 'Extra time 1', 4: 'Extra time 2'}
PKL_PATH = pkl_path('match_features_with_pressure')


def fmt_clock(pet_s):
    m, s = divmod(int(round(pet_s)), 60)
    return f"{m}:{s:02d}"


def scan_full_tracking():
    """One pass over the whole tracking file. Returns:
      raw_samples[period]['ball' | (team_id, jersey)] -> [(frame, (x,y[,z])), ...]
      pet_by_frame: frame -> periodElapsedTime (global dict, frame ids don't
                    repeat across periods in this file)
      period_frame_bounds: period -> (fr_min, fr_max)
    """
    raw_samples = {1: {'ball': []}, 2: {'ball': []}, 3: {'ball': []}, 4: {'ball': []}}
    pet_by_frame = {}
    period_frame_bounds = {}

    t0 = time.time()
    n = 0
    with bz2.open(TRACKING_PATH, 'rt') as f:
        for line in f:
            d = json.loads(line)
            period = d['period']
            frame = d['frameNum']
            pet = d.get('periodElapsedTime')
            pet_by_frame[frame] = pet
            bounds = period_frame_bounds.setdefault(period, [frame, frame])
            bounds[0] = min(bounds[0], frame)
            bounds[1] = max(bounds[1], frame)

            bucket = raw_samples[period]
            ball = d.get('ballsSmoothed')
            if isinstance(ball, list):
                ball = ball[0] if ball else None
            if ball is None or ball.get('x') is None:
                raw_ball = d.get('balls') or []
                b0 = raw_ball[0] if raw_ball else None
                ball = b0 if (b0 is not None and b0.get('x') is not None) else None
            if ball is not None:
                bucket['ball'].append((frame, (ball['x'], ball['y'], max(ball.get('z', 0) or 0, 0))))

            for side, team_id in [('home', HOME_TEAM_ID), ('away', AWAY_TEAM_ID)]:
                players = d.get(f'{side}PlayersSmoothed') or d.get(f'{side}Players') or []
                seen = set()
                for pl in players:
                    jn = pl['jerseyNum']
                    if jn in seen or pl.get('x') is None:
                        continue
                    seen.add(jn)
                    key = (team_id, jn)
                    bucket.setdefault(key, []).append((frame, (pl['x'], pl['y'])))
            n += 1
            if n % 50000 == 0:
                print(f"  scanned {n} lines, {time.time()-t0:.1f}s")

    print(f"Full tracking scan: {n} lines in {time.time()-t0:.1f}s")
    return raw_samples, pet_by_frame, period_frame_bounds


def clean_smooth_period(raw_samples_for_period, fps):
    """Same two-stage cleaning as animation_export.export_window, run once
    over a period's FULL raw trajectory (see module docstring for why)."""
    interpolated = {}
    for key, samples in raw_samples_for_period.items():
        samples = sorted(samples, key=lambda s: s[0])
        if not samples:
            interpolated[key] = {}
            continue
        if key == 'ball':
            cleaned = reject_speed_spikes_iterative(samples, BALL_MAX_SPEED_MS, fps)
            smoothed = kalman_rts_smooth(cleaned, fps, BALL_PROCESS_NOISE_STD, BALL_MEASUREMENT_NOISE_STD)
            interpolated[key] = interpolate_series(smoothed, max_gap_frames=int(BALL_MAX_GAP_S * fps))
        else:
            cleaned = reject_speed_spikes_iterative(samples, PLAYER_MAX_SPEED_MS, fps)
            smoothed = kalman_rts_smooth(cleaned, fps, PLAYER_PROCESS_NOISE_STD, PLAYER_MEASUREMENT_NOISE_STD)
            interpolated[key] = interpolate_series(smoothed, max_gap_frames=int(PLAYER_MAX_GAP_S * fps))
    return interpolated


def plan_chapters(period, frames_sorted, pet_by_frame):
    """frames_sorted: sorted list of every frame number seen in this period.
    Returns a list of (chapter_lo_frame, chapter_hi_frame, pet_lo, pet_hi)
    splitting the period into N chapters of roughly CHAPTER_TARGET_S each,
    boundaries snapped to real frames via periodElapsedTime (never a raw
    frame-count division -- see module docstring)."""
    import bisect
    pets = [pet_by_frame[fr] for fr in frames_sorted]
    pet_lo, pet_hi = pets[0], pets[-1]
    duration = pet_hi - pet_lo
    n_chapters = max(1, round(duration / CHAPTER_TARGET_S))

    bounds_frames = [frames_sorted[0]]
    for i in range(1, n_chapters):
        target_pet = pet_lo + duration * i / n_chapters
        idx = bisect.bisect_left(pets, target_pet)
        idx = max(0, min(idx, len(frames_sorted) - 1))
        bounds_frames.append(frames_sorted[idx])
    bounds_frames.append(frames_sorted[-1])

    chapters = []
    for i in range(n_chapters):
        lo = bounds_frames[i] if i == 0 else bounds_frames[i] + 1
        hi = bounds_frames[i + 1]
        # guard against a degenerate empty slice from two targets snapping
        # to the same frame (shouldn't happen at 5-min granularity, but cheap to guard)
        if lo > hi:
            lo = hi
        chapters.append((lo, hi, pet_by_frame[lo], pet_by_frame[hi]))
    return chapters


def build_frame_records(period, lo_frame, hi_frame, interpolated, pet_by_frame, df, fps):
    frames_out = []
    for frame in range(lo_frame, hi_frame + 1):
        players_out = []
        for side, team_id in [('home', HOME_TEAM_ID), ('away', AWAY_TEAM_ID)]:
            for key, series in interpolated.items():
                if key == 'ball' or not isinstance(key, tuple):
                    continue
                t, jn = key
                if t != team_id or frame not in series:
                    continue
                x, y = series[frame]
                pid = f"{'H' if side == 'home' else 'A'}{jn}"
                players_out.append({'id': pid, 'x': round(x, 2), 'y': round(y, 2), '_jn': jn, '_team_id': team_id})

        ball_xy = interpolated.get('ball', {}).get(frame)
        ball_out = None
        if ball_xy:
            ball_out = {'x': round(ball_xy[0], 2), 'y': round(ball_xy[1], 2), 'z': round(ball_xy[2], 2)}

        row = df.loc[frame] if frame in df.index else None
        if row is not None:
            def _r(v, nd=2):
                return None if pd.isna(v) else round(float(v), nd)

            def _s(v):
                # pandas' string dtype (inferred here for object columns built
                # via df.apply() mixing None with str -- see engagement.py)
                # uses NaN as its own missing-value sentinel instead of
                # preserving Python None, unlike legacy object dtype.
                # json.dump serializes a bare float NaN as the literal token
                # `NaN`, which is not valid JSON and makes the browser's
                # JSON.parse() throw -- caught by grepping a freshly-exported
                # file for literal NaN tokens before this fix. Route every
                # column that can carry a null label through this.
                return None if pd.isna(v) else v

            presser_id = None
            if pd.notna(row.get('pressing_team_id')) and pd.notna(row.get('primary_presser_jersey')):
                side_code = 'H' if row['pressing_team_id'] == HOME_TEAM_ID else 'A'
                presser_id = f"{side_code}{int(row['primary_presser_jersey'])}"
            frame_rec = {
                't': round(frame / fps, 2),
                'clockS': pet_by_frame.get(frame),
                'players': [{'id': p['id'], 'x': p['x'], 'y': p['y']} for p in players_out],
                'ball': ball_out,
                'possessionTeam': ('home' if row.get('possession_team_id') == HOME_TEAM_ID
                                    else ('away' if row.get('possession_team_id') == AWAY_TEAM_ID else None)),
                'homeStructure': _s(row.get('home_structure')),
                'awayStructure': _s(row.get('away_structure')),
                'homeDepth': _r(row.get('home_depth_smooth')),
                'awayDepth': _r(row.get('away_depth_smooth')),
                'homeInterLine': _r(row.get('home_inter_line_smooth')),
                'awayInterLine': _r(row.get('away_inter_line_smooth')),
                'homeBacklineGap': _r(row.get('home_backline_gap')),
                'awayBacklineGap': _r(row.get('away_backline_gap')),
                'pressureScore': _r(row.get('pressure_score'), 3),
                'pressingTeam': ('home' if row.get('pressing_team_id') == HOME_TEAM_ID
                                  else ('away' if row.get('pressing_team_id') == AWAY_TEAM_ID else None)),
                'primaryPresser': presser_id,
                'homeEngaged': _s(row.get('home_engaged')),
                'awayEngaged': _s(row.get('away_engaged')),
                'homeMarking': _s(row.get('home_marking')),
                'awayMarking': _s(row.get('away_marking')),
            }
        else:
            frame_rec = {'t': round(frame / fps, 2), 'clockS': pet_by_frame.get(frame),
                         'players': [{'id': p['id'], 'x': p['x'], 'y': p['y']} for p in players_out],
                         'ball': ball_out,
                         'possessionTeam': None, 'homeStructure': None, 'awayStructure': None,
                         'homeDepth': None, 'awayDepth': None, 'homeInterLine': None, 'awayInterLine': None,
                         'homeBacklineGap': None, 'awayBacklineGap': None, 'pressureScore': None,
                         'pressingTeam': None, 'primaryPresser': None,
                         'homeEngaged': None, 'awayEngaged': None,
                         'homeMarking': None, 'awayMarking': None}
        frames_out.append(frame_rec)
    return frames_out


def main():
    roster, jersey_lookup, player_id_lookup = load_roster()
    meta = load_metadata()
    orientation = orientation_lookup(meta)
    fps = meta['fps']

    jersey_name = {}
    jersey_pos = {}
    for r in roster:
        key = (int(r['team']['id']), r['shirtNumber'])
        jersey_name[key] = r['player']['nickname']
        jersey_pos[key] = r['positionGroupType']

    print("Scanning full tracking file (one pass, ~1 min)...")
    raw_samples, pet_by_frame, period_frame_bounds = scan_full_tracking()

    print("Loading full-match feature/pressure pipeline output...")
    df = pd.read_pickle(PKL_PATH)
    df = df.drop_duplicates(subset='frameNum').set_index('frameNum')

    chapters_manifest = []
    chapter_global_idx = 0
    total_chapters_by_period = {}

    for period in [1, 2, 3, 4]:
        if period not in period_frame_bounds:
            # this match has no extra time (e.g. a regulation-only result like
            # 10515) -- periods 3/4 simply never appear in the tracking file.
            continue
        lo, hi = period_frame_bounds[period]
        frames_sorted = [fr for fr in range(lo, hi + 1) if fr in pet_by_frame and pet_by_frame.get(fr) is not None]
        # frames_sorted should just be range(lo, hi+1) since every raw line has a frameNum;
        # filter defensively in case of any missing periodElapsedTime.
        chapters = plan_chapters(period, frames_sorted, pet_by_frame)
        total_chapters_by_period[period] = len(chapters)

        print(f"Period {period} ({PERIOD_LABEL[period]}): {len(chapters)} chapters")
        print(f"  Cleaning/smoothing full period trajectory ({len(frames_sorted)} frames)...")
        t0 = time.time()
        interpolated = clean_smooth_period(raw_samples[period], fps)
        print(f"  done in {time.time()-t0:.1f}s")

        for ci, (lo_f, hi_f, pet_lo, pet_hi) in enumerate(chapters):
            chapter_global_idx += 1
            player_meta = {}
            frames_out = build_frame_records(period, lo_f, hi_f, interpolated, pet_by_frame, df, fps)
            # backfill player_meta from whichever ids appeared
            seen_ids = set()
            for fr in frames_out:
                for p in fr['players']:
                    seen_ids.add(p['id'])
            for pid in seen_ids:
                side = 'home' if pid[0] == 'H' else 'away'
                team_id = HOME_TEAM_ID if side == 'home' else AWAY_TEAM_ID
                jn_str = pid[1:]
                # jersey_name/jersey_pos are keyed by roster.json's shirtNumber,
                # which is a STRING in this dataset (confirmed for both 10508
                # and 10515) -- looking this up with an int-converted jersey
                # silently missed every player and fell back to "#N" for the
                # name in every chapter ever exported (including all of match
                # 1's already-published chapters). Fix: key the lookup with
                # the string form throughout, matching how jersey_name/
                # jersey_pos were actually built above.
                nm = jersey_name.get((team_id, jn_str), f"#{jn_str}")
                pg = jersey_pos.get((team_id, jn_str), '')
                player_meta[pid] = {'name': nm, 'team': side, 'jersey': jn_str, 'pos': pg}

            meta_out = {
                'gameId': GAME_ID, 'homeTeam': HOME_TEAM_NAME, 'awayTeam': AWAY_TEAM_NAME,
                'homeColor': HOME_COLOR, 'homeColorDim': HOME_COLOR_DIM,
                'awayColor': AWAY_COLOR, 'awayColorDim': AWAY_COLOR_DIM,
                'homeTeamId': HOME_TEAM_ID, 'awayTeamId': AWAY_TEAM_ID,
                'pitchLength': 105.0, 'pitchWidth': 68.0,
                'fps': fps,
                'windowStartMin': pet_lo / 60, 'windowEndMin': pet_hi / 60,
                'period': period, 'periodLabel': PERIOD_LABEL[period],
                'homeDefendsPositiveX': orientation.get(period),
                'players': player_meta,
                'chapterIndex': chapter_global_idx,  # 1-based, across the whole match
                'chapterInPeriod': ci + 1,
                'chaptersInPeriod': len(chapters),
                'clockLabel': f"{fmt_clock(pet_lo)}–{fmt_clock(pet_hi)}",
            }
            payload = {'meta': meta_out, 'frames': frames_out}
            os.makedirs(CHAPTERS_DIR, exist_ok=True)
            out_path = f'{CHAPTERS_DIR}/anim_{chapter_global_idx:02d}.json'
            with open(out_path, 'w') as f:
                json.dump(payload, f, separators=(',', ':'))
            size_mb = os.path.getsize(out_path) / 1e6
            n_no_ball = sum(1 for fr in frames_out if fr['ball'] is None)
            print(f"    chapter {chapter_global_idx:02d} ({PERIOD_LABEL[period]} "
                  f"{meta_out['clockLabel']}): {len(frames_out)} frames, {size_mb:.2f}MB, "
                  f"no-ball {n_no_ball}/{len(frames_out)} ({100*n_no_ball/len(frames_out):.0f}%)")
            chapters_manifest.append({
                'chapterIndex': chapter_global_idx, 'period': period, 'periodLabel': PERIOD_LABEL[period],
                'clockLabel': meta_out['clockLabel'], 'animPath': out_path, 'sizeMb': round(size_mb, 2),
                'noBallPct': round(100 * n_no_ball / len(frames_out), 1),
            })

    # This script always rebuilds chapters_manifest from scratch and has no
    # notion of heatPath -- that key is added afterward, per chapter, by the
    # separate full_match_heatmap.py rasterization stage. Naively overwriting
    # manifest.json here silently drops any heatPath a prior heatmap run had
    # already added, which is exactly what happened three times (rounds 25,
    # 27, 29): every full re-run of this script for an unrelated reason blew
    # away the heatmap link and required a manual manifest patch afterward.
    # Fixed at the source: re-merge any existing heatPath from the manifest
    # already on disk (keyed by chapterIndex, which is stable across runs)
    # before writing the new one, so a full_match_heatmap.py run never has
    # to be repeated just because this script ran again.
    existing_manifest_path = f'{CHAPTERS_DIR}/manifest.json'
    existing_heat_paths = {}
    if os.path.exists(existing_manifest_path):
        try:
            with open(existing_manifest_path) as f:
                prev_manifest = json.load(f)
            for prev_ch in prev_manifest.get('chapters', []):
                if 'heatPath' in prev_ch:
                    existing_heat_paths[prev_ch['chapterIndex']] = prev_ch['heatPath']
        except (json.JSONDecodeError, OSError):
            pass  # no usable prior manifest -- nothing to carry forward
    carried_forward = 0
    for ch in chapters_manifest:
        hp = existing_heat_paths.get(ch['chapterIndex'])
        if hp is not None:
            ch['heatPath'] = hp
            carried_forward += 1

    with open(existing_manifest_path, 'w') as f:
        json.dump({'totalChapters': chapter_global_idx, 'chapters': chapters_manifest}, f, indent=2)
    print(f"\nDone: {chapter_global_idx} chapters total. Manifest written "
          f"({carried_forward} heatPath entries carried forward from the prior manifest).")


if __name__ == '__main__':
    main()

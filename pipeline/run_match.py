"""Run every step for one match, from the four raw files to chapter pages and the Analysis views.

    python3 pipeline/run_match.py <match id>              # run what hasn't run yet
    python3 pipeline/run_match.py <match id> --from pressure
    python3 pipeline/run_match.py <match id> --only gaps chances
    python3 pipeline/run_match.py <match id> --redo        # ignore earlier runs
    python3 pipeline/run_match.py hub                      # comparison board, reports and the hub page
    python3 pipeline/run_match.py --list                   # the steps, in order

The match must be in pipeline/matches.json and its raw files in its raw_dir (see docs/ADDING-A-MATCH.md).
Each step runs as its own process with MATCH_ID set and PYTHONHASHSEED=0 (some steps iterate over sets),
logs to output/logs/<id>/<step>.log and leaves a marker in output/runs/<id>/ when it succeeds, so a re-run
picks up where the last one stopped. A failing step stops the run and prints the end of its log.

Publishing the chapters is not automated: publish output/chapters_<id>/pressure_read_NN.html, save the links as
the registry's chapter_urls file, run full_match_patch.py on it, then
`python3 pipeline/build_match_hub.py --chapters <id>`.
"""
import argparse
import os
import subprocess
import sys
import time

from paths import ROOT, out
from registry import R

PIPE, ANALYSIS = 'pipeline', 'analysis'

# (name, folder, script, extra args, what it does)
STEPS = [
    ('validate', PIPE, 'validate.py', ['{mid}'], 'check raw files; attack direction per period; data quirks'),
    ('tracking_features', PIPE, 'tracking_features.py', [], 'clean tracking; back-line depth, team length, possession'),
    ('transitions', PIPE, 'transitions.py', [], 'defensive-transition windows after possession changes'),
    ('classification', PIPE, 'classification.py', [], 'High Press / Mid Block / Low Block / Transition per frame'),
    ('pressure', PIPE, 'pressure.py', [], 'pressure on the ball carrier (time to intercept)'),
    ('merge_pressure', PIPE, 'merge_pressure.py', [], 'join pressure onto the frame table'),
    ('engagement', PIPE, 'engagement.py', [], 'engaged/passive threshold and labels'),
    ('compactness', PIPE, 'analysis_ball_proximal_compactness.py', [], 'spread of the five players nearest the ball'),
    ('marking', PIPE, 'compactness_tag.py', [], 'TIGHT/LOOSE tag'),
    ('export', PIPE, 'full_match_export.py', [], 'smoothed 2-D frames, split into ~5-minute chapters'),
    ('heatmap', PIPE, 'full_match_heatmap.py', [], 'danger heatmap (DAS) per chapter at 5 Hz'),
    ('head', PIPE, 'make_head.py', ['{mid}'], 'chapter-page head template for this match'),
    ('build', PIPE, 'full_match_build.py', [], 'assemble the chapter pages'),
    ('events', ANALYSIS, 'collect_events.py', ['{mid}'], 'event feed, match clock and ball track'),
    ('positions', ANALYSIS, 'collect.py', ['{mid}'], '5 Hz player positions, possession and live play'),
    ('attack', ANALYSIS, 'lanes.py', ['{mid}'], 'view: where they attacked'),
    ('teams', ANALYSIS, 'team_heat.py', ['{mid}'], 'view: team heatmaps'),
    ('players', ANALYSIS, 'player_heat.py', ['{mid}'], 'view: player heatmaps'),
    ('shape', ANALYSIS, 'shape.py', ['{mid}'], 'view: in and out of possession'),
    ('time', ANALYSIS, 'shape_time.py', ['{mid}'], 'view: shape over time'),
    ('gaps', ANALYSIS, 'gaps.py', ['{mid}'], 'view: where the gaps open'),
    ('lb', ANALYSIS, 'linebreak.py', ['{mid}'], 'view: line-breaking passes'),
    ('turn', ANALYSIS, 'turnovers.py', ['{mid}'], 'view: turnovers'),
    ('chances', ANALYSIS, 'chances.py', ['{mid}'], 'view: chances conceded'),
    ('press', ANALYSIS, 'pressing.py', ['{mid}'], 'view: pressing'),
]
HUB_STEPS = [
    ('digest', ANALYSIS, 'digest.py', [], 'index of every Analysis image with alt text'),
    ('compare', ANALYSIS, 'compare.py', [], 'comparison board'),
    ('reports', ANALYSIS, 'reports.py', [], 'scouting reports with ranked chips'),
    ('hub', PIPE, 'build_match_hub.py', [], 'the Match Library page'),
]


def run_step(mid, step, redo):
    name, folder, script, args, _ = step
    marker = out('runs', mid, f'{name}.done')
    if os.path.exists(marker) and not redo:
        print(f'  {name:<18} done earlier')
        return
    log = out('logs', mid, f'{name}.log')
    env = dict(os.environ, PYTHONHASHSEED='0')
    if mid != 'hub':
        env['MATCH_ID'] = mid
    cmd = [sys.executable, script] + [a.format(mid=mid) for a in args]
    t0 = time.time()
    print(f'  {name:<18} ...', end='', flush=True)
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, cwd=os.path.join(ROOT, folder), env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc:
        print(f' FAILED (exit {rc}) after {time.time() - t0:.0f} s; last lines of {log}:')
        print(''.join(open(log).readlines()[-15:]))
        sys.exit(rc)
    open(marker, 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(f' {time.time() - t0:.0f} s')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mid', nargs='?', help="match id from pipeline/matches.json, or 'hub'")
    ap.add_argument('--from', dest='start', metavar='STEP', help='start at this step (it and later steps re-run)')
    ap.add_argument('--only', nargs='+', metavar='STEP', help='run just these steps (always re-run)')
    ap.add_argument('--redo', action='store_true', help='re-run steps that already succeeded')
    ap.add_argument('--list', action='store_true', help='list the steps and exit')
    a = ap.parse_args()
    if a.list or not a.mid:
        for name, folder, script, _, what in STEPS + HUB_STEPS:
            print(f'{name:<18} {folder}/{script:<40} {what}')
        return
    steps = HUB_STEPS if a.mid == 'hub' else STEPS
    if a.mid != 'hub':
        print(f"{R.get(a.mid)['title']} ({a.mid})")
    names = [s[0] for s in steps]
    for n in (a.only or []) + ([a.start] if a.start else []):
        if n not in names:
            sys.exit(f"unknown step {n!r}; steps: {', '.join(names)}")
    for i, step in enumerate(steps):
        if a.only and step[0] not in a.only:
            continue
        if a.start and i < names.index(a.start):
            continue
        run_step(a.mid, step, redo=a.redo or bool(a.only) or bool(a.start))
    if a.mid != 'hub':
        print(f'done. Next: publish the chapters in output/chapters{R.suffix(a.mid)}/, then run_match.py hub')


if __name__ == '__main__':
    main()

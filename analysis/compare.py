"""The comparison board: one row per team per match, pulled from the analysis views.

    python3 analysis/compare.py [<match id> ...]      # default: every match whose views exist

Reads output/analysis/<kind>/<slug>.json for the views below and writes output/analysis/compare.json, a list
of rows in hub order (registry order, home team first):

    possession     share of live, possessed play the team had the ball (%)          position cache
    back_line      back-line depth from own goal, match median (m)                  time   (match.back_line)
    length, width  block length and width out of possession, match medians (m)      time   (match.*)
    holes          open space inside the block (m², median)                         gaps   (holes)
    free           opponents free inside the block (players, mean)                  gaps   (free)
    wins           ball wins in open play                                           turn   (won)
    wins_high      share of those wins in the opponents' half (%)                   turn   (won_opp_half_pct)
    fast           wins that reached the final third inside 10 s                    turn   (won_to_final_third_10s)
    shots_for, shots_against                                                        chances
    live_min       minutes of live play in the match                                position cache
    press_won      share of presses that won the ball within 5 s (%)                press  (win_rate; None until built)

A match missing any of these views is skipped with a note, so the board only shows complete rows.
"""
import json
import os
import sys

import numpy as np

from common import FPS, R, load_pos, out_path

VIEWS = ('time', 'gaps', 'turn', 'chances')
STEP = 6                                     # the position cache keeps every 6th tracking frame


def live_play(mid):
    """Minutes of live play, and each team's share (%) of the live samples in which a team had the ball."""
    P = load_pos(mid)
    live = np.asarray(P['live'], bool)
    poss = np.array(['' if p is None else str(p) for p in P['poss']])[live]
    m = R.get(mid)
    n = {side: int(np.sum(poss == str(m[side]['id']))) for side in ('home', 'away')}
    share = {side: 100 * n[side] / max(n['home'] + n['away'], 1) for side in n}
    return round(float(live.sum()) * STEP / FPS / 60, 1), share


def rows_for(mid):
    m = R.get(mid)
    paths = {k: out_path('analysis', k, f"{m['slug']}.json") for k in VIEWS}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"{mid}: skipped, missing views: {', '.join(missing)}")
        return []
    v = {k: json.load(open(p, encoding='utf-8')) for k, p in paths.items()}
    press_path = out_path('analysis', 'press', f"{m['slug']}.json")
    press = json.load(open(press_path, encoding='utf-8')) if os.path.exists(press_path) else {}
    live, share = live_play(mid)
    rows = []
    for side, other in (('home', 'away'), ('away', 'home')):
        t = m[side]['name']
        rows.append(dict(
            mid=mid, slug=m['slug'], match=m['title'], team=t, opp=m[other]['name'], color=m[side]['color'],
            possession=round(share[side]),
            back_line=v['time'][t]['match']['back_line'],
            length=v['time'][t]['match']['length'],
            width=v['time'][t]['match']['width'],
            holes=v['gaps'][t]['holes'],
            free=v['gaps'][t]['free'],
            wins=v['turn'][t]['won'],
            wins_high=v['turn'][t]['won_opp_half_pct'],
            fast=v['turn'][t]['won_to_final_third_10s'],
            shots_for=v['chances'][t]['shots_for'],
            shots_against=v['chances'][t]['shots_against'],
            live_min=live,
            press_won=press.get(t, {}).get('win_rate'),
        ))
    return rows


def main(mids):
    rows = [r for mid in mids for r in rows_for(mid)]
    path = out_path('analysis', 'compare.json')
    json.dump(rows, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'{len(rows)} rows -> {path}')


if __name__ == '__main__':
    main(sys.argv[1:] or R.ids())

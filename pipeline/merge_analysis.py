"""Assemble the analysis set the website shows: the published set for the matches already live, plus the
freshly built views for every match it does not have yet.

    python3 pipeline/merge_analysis.py --published <published set> --out <merged set>

Why: the first six matches went live before the analysis code was rebuilt, and the rebuilt numbers sit a
point or two away from the published ones (docs/VERIFICATION.md). The site keeps the published images, board
rows and alt text for those matches, so nothing already live changes, and adds new matches from
output/analysis/. The scouting-report chips are then re-ranked against the whole merged board.

Inputs: <published>/an_kinds.json, compare.json, reports.json and the images under <kind>/<slug>.jpg;
output/analysis/ (run_match.py <id> for each new match, then run_match.py hub); content/reports.json.
Writes the same layout to --out.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))
from common import R, out_path  # noqa: E402
from paths import root  # noqa: E402
import reports as REP  # noqa: E402


def load(p):
    return json.load(open(p, encoding='utf-8'))


def save(obj, p):
    json.dump(obj, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--published', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pub, dst = a.published, a.out
    os.makedirs(dst, exist_ok=True)
    order = [R.get(mid)['slug'] for mid in R.ids()]

    # views: published entries win; new matches come from the fresh build
    pk, nk = load(os.path.join(pub, 'an_kinds.json')), load(out_path('analysis', 'an_kinds.json'))
    fresh = {k['name']: {e['slug']: e for e in k['data']} for k in nk}
    kinds, added = [], set()
    for k in pk:
        have = {e['slug']: e for e in k['data']}
        for slug, e in fresh.get(k['name'], {}).items():
            if slug not in have:
                have[slug] = e
                added.add(slug)
        data = sorted((have[s] for s in have if s in order), key=lambda e: order.index(e['slug']))
        kinds.append(dict(k, data=data))
        for e in data:
            src = os.path.join(pub if e['slug'] not in added else out_path('analysis'), e['src'])
            os.makedirs(os.path.dirname(os.path.join(dst, e['src'])), exist_ok=True)
            shutil.copy2(src, os.path.join(dst, e['src']))
    save(kinds, os.path.join(dst, 'an_kinds.json'))

    # board: published rows (with the press rate the published report chips carry), then new rows
    pb = load(os.path.join(pub, 'compare.json'))
    press = {}
    for r in load(os.path.join(pub, 'reports.json')):
        for c in r['chips']:
            if c['label'] == 'presses won in 5 s':
                press[(r['slug'], r['team'])] = int(c['value'].rstrip('%'))
    for r in pb:
        r.setdefault('press_won', press.get((r['slug'], r['team'])))
    seen = {r['slug'] for r in pb}
    nb = [r for r in load(out_path('analysis', 'compare.json')) if r['slug'] not in seen]
    board = sorted(pb + nb, key=lambda r: (order.index(r['slug']), r['team'] != R.by_slug(r['slug'])['home']['name']))
    save(board, os.path.join(dst, 'compare.json'))

    # reports: the hand-written text with chips ranked against the merged board
    rows = {(r['slug'], r['team']): r for r in board}
    out, missing = [], []
    for rep in load(root('content', 'reports.json')):
        row = rows.get((rep['slug'], rep['team']))
        if row is None:
            missing.append(f"{rep['team']} ({rep['slug']})")
            continue
        m = R.by_slug(rep['slug'])
        out.append(dict(slug=rep['slug'], team=rep['team'], opp=row['opp'], color=row['color'], match=m['title'],
                        sub=m['sub'], headline=rep['headline'], good=rep['good'], exploit=rep['exploit'],
                        views=rep['views'], chips=REP.chips(row, board)))
    save(out, os.path.join(dst, 'reports.json'))

    print(f"{len(kinds)} views x {len(kinds[0]['data'])} matches ({len(added)} new: {', '.join(sorted(added)) or '-'}); "
          f"{len(board)} board rows; {len(out)} reports -> {dst}")
    if missing:
        print('reports without a board row:', ', '.join(missing))
    no_report = [f"{r['team']} ({r['slug']})" for r in board if (r['slug'], r['team']) not in {(x['slug'], x['team']) for x in out}]
    if no_report:
        print('teams without a written report:', ', '.join(no_report))


if __name__ == '__main__':
    main()

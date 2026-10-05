"""Scouting reports for the hub: the hand-written text plus stat chips ranked against every team on the board.

    python3 analysis/reports.py

Inputs:
- content/reports.json: one report per team per match, written by hand from the analysis views:
  {slug, team, headline, good[3], exploit[3], views[]} (views name the Analysis tabs the report points to).
- output/analysis/compare.json (analysis/compare.py).

Each report gets the match facts from the registry and five chips. A chip ranks the team among all rows of the
comparison board that have the value: ties share the better rank ("5th" three times, then "8th"), and rank 1
carries a word saying which end of the scale is first.

Writes output/analysis/reports.json in the order of content/reports.json. A report whose team is not on the
board yet (its match's views are not built) is skipped with a note.
"""
import json
import os

from common import R, out_path
from paths import root

# (compare.json field, label, value format, True if higher ranks first, word for 1st place)
CHIPS = [('holes', 'holes inside its block', '{} m²', False, 'tightest'),
         ('back_line', 'back line from own goal', '{:.0f} m', True, 'highest'),
         ('wins_high', 'ball wins in opp. half', '{}%', True, 'highest'),
         ('press_won', 'presses won in 5 s', '{}%', True, 'best'),
         ('shots_against', 'shots faced', '{}', False, 'fewest')]


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def chips(row, board):
    out = []
    for field, label, fmt, high_first, best in CHIPS:
        v = row.get(field)
        vals = [r[field] for r in board if r.get(field) is not None]
        if v is None:
            continue
        rank = 1 + sum((x > v) if high_first else (x < v) for x in vals)
        text = f'{ordinal(rank)} of {len(vals)}' + (f' ({best})' if rank == 1 else '')
        out.append(dict(label=label, value=fmt.format(v), rank=rank, of=len(vals), rank_text=text))
    return out


def main():
    board = json.load(open(out_path('analysis', 'compare.json'), encoding='utf-8'))
    rows = {(r['slug'], r['team']): r for r in board}
    written = json.load(open(root('content', 'reports.json'), encoding='utf-8'))
    reports, skipped = [], []
    for rep in written:
        row = rows.get((rep['slug'], rep['team']))
        if row is None:
            skipped.append(f"{rep['team']} ({rep['slug']})")
            continue
        m = R.by_slug(rep['slug'])
        reports.append(dict(slug=rep['slug'], team=rep['team'], opp=row['opp'], color=row['color'], match=m['title'],
                            sub=m['sub'], headline=rep['headline'], good=rep['good'], exploit=rep['exploit'],
                            views=rep['views'], chips=chips(row, board)))
    path = out_path('analysis', 'reports.json')
    json.dump(reports, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'{len(reports)} reports -> {path}')
    if skipped:
        print('not on the board yet:', ', '.join(skipped))


if __name__ == '__main__':
    main()

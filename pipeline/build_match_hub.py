"""Build the Match Library hub: one page with the chapter picker, the Analysis views, the scouting reports and the
team comparison.

    python3 pipeline/build_match_hub.py                    # build output/hub/
    python3 pipeline/build_match_hub.py --chapters 3823    # first record 3823's published chapter links

Inputs:
- templates/hub.html: the page, with {{...}} slots for everything that depends on the matches.
- content/hub_matches.json: the published chapters of every match (period, clock range, ball-tracked share and
  link per chapter). `--chapters <id>` refreshes one match's entry from output/chapters_<id>/manifest.json and
  the match's chapter-links file (the registry's chapter_urls), after its chapters are published.
- output/analysis/an_kinds.json, compare.json and reports.json (analysis/digest.py, compare.py, reports.py).

A match appears in the Analysis tabs only when all ten views are built, because the page lines the views up by
position. The hub's images are copied to output/hub/<view>/<slug>.jpg next to index.html.
"""
import argparse
import html
import json
import os
import shutil

from paths import OUT, root, out
from registry import R

ANALYSIS = os.path.join(OUT, 'analysis')
MATCHES_FILE = root('content', 'hub_matches.json')
VERSION = {'id': 'v1-depth-interline', 'label': 'Baseline — depth + inter-line',
           'note': 'Classifies each team\'s out-of-possession phase (High Press / Mid Block / Low Block / Transition) '
                   'from back-line depth and inter-line compactness only.'}
NUMBER_WORDS = {2: 'two', 4: 'four', 6: 'six', 8: 'eight', 10: 'ten', 12: 'twelve', 14: 'fourteen', 16: 'sixteen'}
PERIODS = {1: '1st half', 2: '2nd half', 3: 'Extra time 1', 4: 'Extra time 2'}


def load(path):
    return json.load(open(path, encoding='utf-8'))


def chapters_from_build(mid):
    """The chapter list for one match from its build manifest and its published links."""
    manifest = load(os.path.join(R.chapters_dir(mid), 'manifest.json'))
    urls = load(R.chapter_urls(mid))
    urls = urls.get('chapters', urls)
    return [{'idx': ch['chapterIndex'], 'period': ch.get('periodLabel') or PERIODS[ch['period']],
             'clock': ch['clockLabel'], 'coverage': round(100 - ch['noBallPct']), 'url': urls[str(ch['chapterIndex'])]}
            for ch in manifest['chapters']]


def refresh_chapters(mid):
    m = R.get(mid)
    entries = load(MATCHES_FILE) if os.path.exists(MATCHES_FILE) else []
    version = dict(VERSION, note=m.get('hub_note') or VERSION['note'], chapters=chapters_from_build(mid))
    entry = {'id': m['hub_id'], 'home': m['home']['name'], 'away': m['away']['name'], 'label': m['title'],
             'competition': m['competition'], 'versions': [version]}
    entries = [e for e in entries if e['id'] != m['hub_id']] + [entry]
    order = [R.get(i)['hub_id'] for i in R.ids()]
    entries.sort(key=lambda e: order.index(e['id']) if e['id'] in order else len(order))
    json.dump(entries, open(MATCHES_FILE, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f"{m['title']}: {len(version['chapters'])} chapters recorded in {MATCHES_FILE}")


def js(value):
    """JSON for an inline <script>: keeps a '</' inside a string from closing the tag."""
    return json.dumps(value, ensure_ascii=False).replace('</', '<\\/')


def tab(label, color=None, small=None):
    dot = f'<i style="background:{color}"></i>' if color else ''
    sub = f' <small>{html.escape(small)}</small>' if small else ''
    return (f'<button type="button" class="lane-tab" role="tab" aria-selected="false">{dot}'
            f'{html.escape(label)}{sub}</button>')


def build():
    kinds = load(os.path.join(ANALYSIS, 'an_kinds.json'))
    board = load(os.path.join(ANALYSIS, 'compare.json'))
    reports = load(os.path.join(ANALYSIS, 'reports.json'))
    published = {e['id']: e for e in load(MATCHES_FILE)}

    # matches with every view built, in registry order
    have = [{e['slug'] for e in k['data']} for k in kinds]
    complete = [mid for mid in R.ids() if all(R.get(mid)['slug'] in s for s in have)]
    for mid in R.ids():
        if mid not in complete:
            print(f'{R.get(mid)["title"]}: left out of the Analysis tabs (not every view is built)')
    slugs = [R.get(mid)['slug'] for mid in complete]
    an_kinds = [dict(k, data=sorted((e for e in k['data'] if e['slug'] in slugs), key=lambda e: slugs.index(e['slug'])))
                for k in kinds]
    board = [r for r in board if r['slug'] in slugs]
    reports = [r for r in reports if r['slug'] in slugs]

    matches = []
    for mid in R.ids():
        hub_id = R.get(mid)['hub_id']
        if hub_id in published:
            matches.append(published[hub_id])
        else:
            print(f'{R.get(mid)["title"]}: no published chapters recorded (run --chapters {mid}); left out of the picker')

    n_teams = len(board)
    fill = {
        'MATCHUP': ' &nbsp;|&nbsp; '.join(f"<b>{html.escape(e['home'])}</b> vs <b>{html.escape(e['away'])}</b> "
                                          f"&middot; {html.escape(e['competition'])}" for e in matches),
        'MATCH_TABS': ''.join(tab(R.get(mid)['title']) for mid in complete),
        'TEAM_TABS': ''.join(tab(r['team'], r['color'], f"v {r['opp']}") for r in reports),
        'N_TEAMS': NUMBER_WORDS.get(n_teams, str(n_teams)),
        'N_MATCHES': NUMBER_WORDS.get(len(complete), str(len(complete))),
        'AN_KINDS': js(an_kinds), 'CMP': js(board), 'REPORTS': js(reports), 'MATCHES': js(matches),
    }
    page = open(root('templates', 'hub.html'), encoding='utf-8').read()
    for key, value in fill.items():
        page = page.replace('{{' + key + '}}', value)
    hub_dir = out('hub')
    with open(os.path.join(hub_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(page)
    n = 0
    for k in an_kinds:
        for e in k['data']:
            dst = out('hub', e['src'])
            shutil.copyfile(os.path.join(ANALYSIS, e['src']), dst)
            n += 1
    print(f'hub: {len(matches)} matches in the picker, {len(complete)} in Analysis, {len(reports)} reports, '
          f'{n_teams} board rows, {n} images -> {hub_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--chapters', metavar='MATCH_ID', help="record a match's published chapter links first")
    a = ap.parse_args()
    if a.chapters:
        refresh_chapters(a.chapters)
    build()

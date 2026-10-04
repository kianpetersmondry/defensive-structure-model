"""Write the chapter-page head template for a match, from the registry.

    python3 pipeline/make_head.py <match id> [--force]

The chapter pages are head + match data + heatmap data + tail (see full_match_build.py). The head carries the
per-match parts of the page: team colours, the competition line, team names and three-letter codes, the data
credit in the footer, and one sentence about how the engaged/passive threshold was set. This script takes
templates/artifact_head_3821.html as the base and swaps only those parts, each at a fixed anchor, failing loudly
if an anchor is missing (a silent blind replace once put the wrong match name into a footer).

Writes the registry's head_template path (templates/artifact_head_<id>.html). An existing file is left alone
unless --force is given.
"""
import argparse
import os
import re

from paths import root
from registry import R

BASE = root('templates', 'artifact_head_3821.html')

PROVIDERS = {
    'pff': {'data': 'PFF FC tracking data',
            'footer': 'PFF FC 2022 World Cup tracking data &middot; game {id} &middot; ~30Hz broadcast tracking, ball ~15Hz effective',
            'threshold': None},
    'dfl': {'data': 'DFL Bundesliga open data, converted to the PFF layout',
            'footer': 'DFL Bundesliga open data (&copy; DFL, CC-BY 4.0) &middot; game {id} &middot; 25Hz optical tracking, resampled to 30Hz',
            'threshold': ("Its threshold was validated against PFF's analyst-tagged pressure events in the World Cup matches "
                          "and is carried over here, since this Bundesliga data has no such tags; it isn't eyeballed")},
}
PFF_THRESHOLD = "It's validated against PFF's own analyst-tagged pressure events, not eyeballed"


def swap(text, pattern, repl, count=1, flags=0):
    new, n = re.subn(pattern, repl, text, flags=flags)
    if n != count:
        raise SystemExit(f'make_head: expected {count} match(es) for {pattern!r}, found {n}')
    return new


def make_head(mid):
    m = R.get(mid)
    prov = PROVIDERS[R.provider(mid)]
    h, a = m['home'], m['away']
    text = open(BASE, encoding='utf-8').read()

    text = swap(text, r'(Pressure Read -- defensive-structure monitor\n   ).*\n',
                lambda _: f"Pressure Read -- defensive-structure monitor\n   {m['title']} · {m['competition']} ({prov['data']})\n")
    for side, team in (('home', h), ('away', a)):
        text = swap(text, rf'(--{side}: )#[0-9a-fA-F]{{6}};', rf"\g<1>{team['color']};")
        text = swap(text, rf'(--{side}-dim: )#[0-9a-fA-F]{{6}};', rf"\g<1>{team['color_dim']};")
        text = swap(text, rf'(style="color:var\(--{side}\)">)[^<]*(</span>)', rf"\g<1>{team['code']}\2")
        text = swap(text, rf'(<span class="dot {side}"></span>)[A-Z]{{3}}(</span>|</td>)', rf"\g<1>{team['code']}\2", count=2)
    text = swap(text, r'(<div class="matchup">\n\s*).*?<br>\n(\s*)<b class="home">[^<]*</b> vs <b class="away">[^<]*</b>',
                lambda mm: (f"{mm.group(1)}{m['competition'].replace(' · ', ' &middot; ')}<br>\n{mm.group(2)}"
                            f"<b class=\"home\">{h['name']}</b> vs <b class=\"away\">{a['name']}</b>"), flags=re.S)
    text = swap(text, r'(<footer class="credit">).*?(</footer>)',
                lambda mm: mm.group(1) + prov['footer'].format(id=mid) + mm.group(2))
    if prov['threshold']:
        if PFF_THRESHOLD not in text:
            raise SystemExit('make_head: engaged/passive sentence not found in the base template')
        text = text.replace(PFF_THRESHOLD, prov['threshold'])
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mid')
    ap.add_argument('--force', action='store_true', help='overwrite an existing head template')
    args = ap.parse_args()
    path = R.head_template(args.mid)
    if os.path.exists(path) and not args.force:
        print(f'{path} exists; leaving it (use --force to regenerate)')
    else:
        open(path, 'w', encoding='utf-8').write(make_head(args.mid))
        print('written', path)

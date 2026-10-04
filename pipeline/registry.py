"""Read access to pipeline/matches.json, the one place every per-match fact lives.

    from registry import R
    R.ids()                      # match ids in hub order
    R.get('3823')['home']        # team id, name, code, colours
    R.raw('3823', 'roster.json') # path to a raw file
    R.feat('3823')               # path to the match-features table
    R.dirs('3823')               # {period: +1 if the home team defends +x}
"""
import json
import os

from paths import ROOT, OUT, data_dir, out

PW = ROOT          # kept for scripts written against the old name


class _Registry:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matches.json')

    # data credit on images and pages: PFF text is returned untouched; other providers get their
    # own source line (DFL open data is CC-BY 4.0 and must credit the DFL)
    CREDITS = {'dfl': [('PFF FC broadcast tracking behind the 2-D animations', 'DFL optical tracking (Bundesliga open data, CC-BY 4.0)'),
                       ('PFF FC broadcast tracking behind the 2-D animation', 'DFL optical tracking (Bundesliga open data, CC-BY 4.0)'),
                       ('PFF FC tracking behind the 2-D animations', 'DFL optical tracking (Bundesliga open data, CC-BY 4.0)'),
                       ('PFF FC tracking', 'DFL tracking (CC-BY)'),
                       ('engagement threshold calibrated for this match', 'engagement threshold pooled from the World Cup matches'),
                       ('PFF event feed', 'DFL event feed'), ('PFF FC', 'DFL')]}

    def __init__(self):
        self._data = None
        self._derived = {}

    def _load(self):
        if self._data is None:
            self._data = json.load(open(self.path, encoding='utf-8'))
        return self._data

    def ids(self):
        return [m['id'] for m in self._load()['matches']]

    def get(self, mid):
        for m in self._load()['matches']:
            if m['id'] == str(mid):
                return m
        raise KeyError(f'match {mid} is not in pipeline/matches.json')

    def by_slug(self, slug):
        return next(m for m in self._load()['matches'] if m['slug'] == slug)

    def suffix(self, mid):
        """Output-file suffix: '' for the original test match (10508), '_<id>' otherwise."""
        return '' if str(mid) == '10508' else f'_{mid}'

    def provider(self, mid):
        return self.get(mid).get('provider', 'pff')

    def credit(self, mid, text):
        for old, new in self.CREDITS.get(self.provider(mid), []):
            text = text.replace(old, new)
        return text

    # ---- paths
    def raw_dir(self, mid):
        return data_dir(self.get(mid)['raw_dir'])

    def raw(self, mid, fname):
        return os.path.join(self.raw_dir(mid), fname)

    def feat(self, mid, name='match_features_with_pressure'):
        return out(f'{name}{self.suffix(mid)}.pkl')

    def chapters_dir(self, mid):
        return os.path.join(OUT, f'chapters{self.suffix(mid)}')

    def head_template(self, mid):
        return os.path.join(ROOT, self.get(mid)['head_template'])

    def chapter_urls(self, mid):
        return os.path.join(OUT, self.get(mid)['chapter_urls'])

    def derived_path(self, mid):
        return out('derived', f'{mid}.json')

    # ---- teams
    def team_names(self, mid):
        m = self.get(mid)
        return m['home']['name'], m['away']['name']

    def team(self, mid, name_or_side):
        m = self.get(mid)
        if name_or_side in ('home', 'away'):
            return m[name_or_side]
        for side in ('home', 'away'):
            if m[side]['name'] == name_or_side or str(m[side]['id']) == str(name_or_side):
                return m[side]
        raise KeyError(name_or_side)

    def code(self, mid, name):
        return self.team(mid, name)['code']

    def goals(self, mid):
        return [tuple(g) for g in self.get(mid).get('goals', [])]

    # ---- facts worked out by pipeline/validate.py
    def derived(self, mid):
        if mid not in self._derived:
            p = self.derived_path(mid)
            self._derived[mid] = json.load(open(p)) if os.path.exists(p) else {}
        return self._derived[mid]

    def dirs(self, mid):
        """{period: +1 if the home team defends +x, -1 otherwise}, from validate.py."""
        d = self.derived(mid).get('dirs')
        if not d:
            raise RuntimeError(f'no attack directions for {mid}: run pipeline/validate.py {mid} first')
        return {int(k): int(v) for k, v in d.items()}

    def keeper_swap(self, mid):
        q = self.get(mid).get('quirks', {}).get('keeper_swap') or self.derived(mid).get('keeper_swap')
        return q or None

    def press_threshold(self, mid):
        m = self.get(mid)
        if m.get('press_threshold_override') is not None:
            return float(m['press_threshold_override'])
        return float(json.load(open(out(f'engagement_threshold{self.suffix(mid)}.json')))['threshold'])


R = _Registry()

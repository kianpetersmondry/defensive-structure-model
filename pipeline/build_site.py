"""Build the self-hosted static site: the hub, every chapter and the companion pages, with every claude.ai
link rewritten to a relative path, so the folder runs on any static host (or straight from disk).

    python3 pipeline/build_site.py [--extra DIR] [--out DIR]

Layout of the result (default output/site/):

    index.html                         the hub (output/hub/, from build_match_hub.py)
    <view>/<slug>.jpg                  the Analysis images
    matches/<site_folder>/chapter-NN.html
    behind-the-model/, broadcast/      companion pages, copied from --extra

Chapters come from output/chapters_<id>/pressure_read_NN.html when they have been built here, otherwise from
--extra/matches/<site_folder>/ (a previous export). The companion pages (the Behind the Model charts and the
broadcast-footage pages, whose video is not in this repository) are copied from --extra as they are.

Links are mapped from content/hub_matches.json (chapters), HUB_URL (the hub) and content/site_links.json
(companion pages); both spellings of an artifact link (claude.ai/artifact/<id> and claude.ai/code/artifact/<id>)
are matched. The build ends by listing any claude.ai link it could not map, and any page it could not find.
"""
import argparse
import json
import os
import re
import shutil

from config import HUB_URL
from paths import OUT, root
from registry import R

LINK = re.compile(r'https://claude\.ai/(?:code/)?artifact/([A-Za-z0-9-]+)')
COMPANION_DIRS = ('behind-the-model', 'broadcast')


def artifact_id(url):
    m = LINK.fullmatch(url)
    return m.group(1) if m else None


def link_map():
    """artifact id -> site path."""
    paths = {artifact_id(HUB_URL): 'index.html'}
    for url, path in json.load(open(root('content', 'site_links.json'), encoding='utf-8')).items():
        if not url.startswith('_'):
            paths[artifact_id(url)] = path
    by_hub_id = {R.get(mid)['hub_id']: mid for mid in R.ids()}
    for entry in json.load(open(root('content', 'hub_matches.json'), encoding='utf-8')):
        folder = R.get(by_hub_id[entry['id']])['site_folder']
        for ch in entry['versions'][0]['chapters']:
            paths[artifact_id(ch['url'])] = f"matches/{folder}/chapter-{ch['idx']:02d}.html"
    return paths


def chapter_sources(extra):
    """site path -> source file for every chapter; missing ones are reported."""
    by_hub_id = {R.get(mid)['hub_id']: mid for mid in R.ids()}
    found, missing = {}, []
    for entry in json.load(open(root('content', 'hub_matches.json'), encoding='utf-8')):
        mid = by_hub_id[entry['id']]
        folder = R.get(mid)['site_folder']
        for ch in entry['versions'][0]['chapters']:
            site_path = f"matches/{folder}/chapter-{ch['idx']:02d}.html"
            built = os.path.join(R.chapters_dir(mid), f"pressure_read_{ch['idx']:02d}.html")
            previous = os.path.join(extra, site_path) if extra else None
            src = built if os.path.exists(built) else previous if previous and os.path.exists(previous) else None
            if src:
                found[site_path] = src
            else:
                missing.append(site_path)
    return found, missing


def rewrite(text, page, paths, unresolved):
    here = os.path.dirname(page) or '.'

    def sub(m):
        target = paths.get(m.group(1))
        if target is None:
            unresolved.add(m.group(0))
            return m.group(0)
        return os.path.relpath(target, here).replace(os.sep, '/')
    return LINK.sub(sub, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--extra', help='a previous site export holding behind-the-model/, broadcast/ and chapters')
    ap.add_argument('--out', default=os.path.join(OUT, 'site'))
    a = ap.parse_args()
    hub = os.path.join(OUT, 'hub')
    if not os.path.exists(os.path.join(hub, 'index.html')):
        raise SystemExit('no hub yet: run pipeline/build_match_hub.py first')
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(hub, a.out)                      # index.html + the Analysis images
    if a.extra:
        for d in COMPANION_DIRS:
            if os.path.isdir(os.path.join(a.extra, d)):
                shutil.copytree(os.path.join(a.extra, d), os.path.join(a.out, d))

    paths, unresolved = link_map(), set()
    chapters, missing = chapter_sources(a.extra)
    for site_path, src in chapters.items():
        dst = os.path.join(a.out, site_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, encoding='utf-8') as f:
            text = f.read()
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(rewrite(text, site_path, paths, unresolved))

    pages = 0
    for dirpath, _, files in os.walk(a.out):
        for name in files:
            if not name.endswith('.html'):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, a.out)
            if rel in chapters:
                pages += 1
                continue
            text = open(path, encoding='utf-8').read()
            new = rewrite(text, rel, paths, unresolved)
            if new != text:
                open(path, 'w', encoding='utf-8').write(new)
            pages += 1

    absent = [p for p in set(paths.values()) if not os.path.exists(os.path.join(a.out, p))]
    n_files = sum(len(f) for _, _, f in os.walk(a.out))
    print(f'site: {n_files} files, {pages} pages, {len(chapters)} chapters -> {a.out}')
    if missing:
        print(f'{len(missing)} chapters not found (build them, or pass --extra):', ', '.join(missing[:5]),
              '...' if len(missing) > 5 else '')
    if absent:
        print(f'{len(absent)} linked pages are not in the site:', ', '.join(sorted(absent)[:8]))
    if unresolved:
        print(f'{len(unresolved)} claude.ai links left as they are:', ', '.join(sorted(unresolved)[:8]))


if __name__ == '__main__':
    main()

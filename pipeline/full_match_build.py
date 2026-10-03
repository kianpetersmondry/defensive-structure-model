"""
Assembles the 26 chapter HTML pages from the shared head/tail templates
plus each chapter's own anim_XX.json / heat_XX.json, the same head+data+tail
concatenation `build_pressure_read.py` already uses for the single curated
window -- just looped, with a per-chapter chapter-bar (prev/next/index nav
+ chapter label + ball-coverage note) spliced into the <!--CHAPTER_BAR-->
placeholder `artifact_head.html` now carries.

Two-pass nav, because artifact URLs don't exist until publish time:
  PASS 1 (this script, mode='placeholder'): every chapter's prev/next/index
    links are the literal tokens __PREV_URL__ / __NEXT_URL__ / __INDEX_URL__.
    First chapter has no prev, last has no next -- rendered as disabled
    spans, not placeholder links, since we already know those at build time.
  PASS 2 (mode='patch', after all 26 + the index have been published once
    and we know every real URL): re-reads each chapter's OWN already-built
    HTML file and does a plain string replace of the three tokens with real
    URLs -- cheap, no need to re-run the head/anim/heat/tail concatenation.
"""
import json
import sys

from config import CHAPTERS_DIR, HEAD_TEMPLATE, HOME_TEAM_NAME, AWAY_TEAM_NAME, HUB_URL

HEAD = HEAD_TEMPLATE
TAIL = '/home/claude/project_work/artifact_tail.html'


def chapter_bar_html(ch, total):
    idx = ch['chapterIndex']
    coverage = round(100 - ch['noBallPct'], 1)
    cov_class = ' low' if coverage < 70 else ''
    label = f"{ch['periodLabel']} &middot; {ch['clockLabel']}"

    if idx > 1:
        prev = '<a class="chapter-nav-btn" href="__PREV_URL__">&larr; Prev</a>'
    else:
        prev = '<span class="chapter-nav-btn disabled">&larr; Prev</span>'
    if idx < total:
        nxt = '<a class="chapter-nav-btn" href="__NEXT_URL__">Next &rarr;</a>'
    else:
        nxt = '<span class="chapter-nav-btn disabled">Next &rarr;</span>'

    info = (
        f'<a class="chapter-index-link" href="__INDEX_URL__">Chapter {idx} of {total}</a>'
        f'<span class="chapter-sep">&middot;</span>'
        f'<span class="chapter-label">{label}</span>'
        f'<span class="chapter-sep">&middot;</span>'
        f'<span class="chapter-coverage{cov_class}" title="Share of this chapter\'s frames with a tracked ball position">'
        f'Ball tracked {coverage:.0f}%</span>'
    )
    return f'<div class="chapter-bar">{prev}<div class="chapter-info">{info}</div>{nxt}</div>'


def build_chapter(ch, total, head_text, tail_text):
    idx = ch['chapterIndex']
    bar = chapter_bar_html(ch, total)
    head = head_text.replace('<!--CHAPTER_BAR-->', bar)
    # include the matchup in the title (not just "Defensive Structure Monitor")
    # so chapters from two different matches are distinguishable in a flat
    # artifact list -- added round 30 alongside France vs Morocco; applies to
    # a match-1 rebuild too since this is the shared build script, not a
    # per-match one. Renamed from "Pressure Read" (round 33) -- the tool no
    # longer uses that name anywhere in the published chapters.
    head = head.replace(
        '<title>Defensive Structure Monitor</title>',
        f"<title>Defensive Structure Monitor: {HOME_TEAM_NAME} v {AWAY_TEAM_NAME} — "
        f"Ch. {idx}: {ch['periodLabel']}, {ch['clockLabel']}</title>",
    )
    # Masthead title now links back to the Match Library hub (round 35 --
    # requested so a viewer can get "home" without hunting for the browser
    # back button). Fixed here, in the one shared generator every match's
    # chapters pass through, rather than in each per-match head template --
    # the exact mistake that let match 3's title regress to "Pressure Read"
    # (a head template built by copying an older one) is what this avoids
    # happening again for this feature.
    head = head.replace(
        '<h1>Defensive Structure Monitor</h1>',
        f'<h1><a class="home-link" href="{HUB_URL}" '
        f'title="Back to the Match Library">Defensive Structure Monitor</a></h1>',
    )
    head = head.replace(
        '</style>',
        '.masthead h1 a.home-link { color: inherit; text-decoration: none; }\n'
        '.masthead h1 a.home-link:hover, .masthead h1 a.home-link:focus-visible '
        '{ color: var(--amber); text-decoration: underline; }\n'
        '.masthead h1 a.home-link:focus-visible { outline: 2px solid var(--amber); outline-offset: 2px; }\n'
        '</style>',
    )

    out_path = f"{CHAPTERS_DIR}/pressure_read_{idx:02d}.html"
    with open(out_path, 'wb') as out:
        out.write(head.encode('utf-8'))
        with open(ch['animPath'], 'rb') as f:
            out.write(f.read())
        out.write(b'\n</script>\n<script id="heatmap-data" type="application/json">\n')
        with open(ch['heatPath'], 'rb') as f:
            out.write(f.read())
        out.write(tail_text.encode('utf-8'))

    import os
    return out_path, os.path.getsize(out_path)


def main():
    with open(f'{CHAPTERS_DIR}/manifest.json') as f:
        manifest = json.load(f)
    chapters = manifest['chapters']
    total = len(chapters)

    with open(HEAD, encoding='utf-8') as f:
        head_text = f.read()
    with open(TAIL, encoding='utf-8') as f:
        tail_text = f.read()

    for ch in chapters:
        out_path, size_b = build_chapter(ch, total, head_text, tail_text)
        ch['htmlPath'] = out_path
        ch['htmlSizeMb'] = round(size_b / 1e6, 2)
        print(f"chapter {ch['chapterIndex']:02d}: {out_path} ({size_b/1e6:.2f} MB)")

    with open(f'{CHAPTERS_DIR}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print("Manifest updated with htmlPath.")


if __name__ == '__main__':
    main()

"""
Builds the match-index page: the entry point for the full-match extension,
listing all 26 chapters grouped by period with their time range and ball-
tracking coverage, linking out to each chapter's own artifact. Chapter pages
also link back here (the "Chapter N of 26" link in each chapter-bar).

Run AFTER all 26 chapters have been published once (needs their real URLs).
Usage: fill CHAPTER_URLS below (chapterIndex -> url) and run; prints the
assembled HTML path. The chapter pages' own __INDEX_URL__ token gets patched
separately once this page is itself published (see full_match_patch.py).
"""
import json

CHAPTERS_DIR = '/home/claude/project_work/chapters'
OUT_PATH = '/home/claude/project_work/chapters/match_index.html'

PERIOD_ORDER = [1, 2, 3, 4]


def build(chapter_urls):
    with open(f'{CHAPTERS_DIR}/manifest.json') as f:
        manifest = json.load(f)
    chapters = manifest['chapters']
    total = len(chapters)
    by_period = {}
    for ch in chapters:
        by_period.setdefault(ch['period'], []).append(ch)

    period_label = {1: '1st half', 2: '2nd half', 3: 'Extra time 1', 4: 'Extra time 2'}

    sections = []
    for p in PERIOD_ORDER:
        if p not in by_period:
            continue
        rows = []
        for ch in by_period[p]:
            idx = ch['chapterIndex']
            url = chapter_urls.get(str(idx)) or chapter_urls.get(idx) or '#'
            coverage = round(100 - ch['noBallPct'], 1)
            cov_class = ' low' if coverage < 70 else ''
            rows.append(f'''
            <a class="chapter-row" href="{url}">
              <span class="chapter-row-num">{idx:02d}</span>
              <span class="chapter-row-clock">{ch['clockLabel']}</span>
              <span class="chapter-row-coverage{cov_class}">Ball tracked {coverage:.0f}%</span>
              <span class="chapter-row-go">Open &rarr;</span>
            </a>''')
        sections.append(f'''
        <section class="period-section">
          <h2>{period_label[p]}</h2>
          <div class="chapter-rows">{''.join(rows)}</div>
        </section>''')

    body = f'''<title>Pressure Read — Match Index</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg: #0c1512; --panel: #131f1a; --panel-raised: #17251f; --border: #253830;
  --ink: #eef3ee; --ink-dim: #93a89c; --ink-faint: #5e7268;
  --morocco: #e8323a; --spain: #4fd8d0; --amber: #ffb020; --state-mid: #d99a2b;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; background:var(--bg); color:var(--ink);
  font-family:'IBM Plex Sans', system-ui, sans-serif; -webkit-font-smoothing:antialiased; }}
body {{ min-height:100vh; padding:28px 20px 40px; }}
.wrap {{ max-width:820px; margin:0 auto; display:flex; flex-direction:column; gap:22px; }}
.masthead {{ border-bottom:1px solid var(--border); padding-bottom:14px; }}
.masthead h1 {{ font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:32px;
  letter-spacing:0.02em; text-transform:uppercase; margin:0 0 4px; }}
.masthead .eyebrow {{ font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.09em;
  text-transform:uppercase; color:var(--amber); }}
.masthead .matchup {{ font-family:'IBM Plex Mono', monospace; font-size:13px; color:var(--ink-dim); margin-top:8px; }}
.masthead .matchup b.mar {{ color:var(--morocco); }}
.masthead .matchup b.esp {{ color:var(--spain); }}
.about {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px 18px;
  font-size:13px; line-height:1.6; color:var(--ink-dim); }}
.about b {{ color:var(--ink); }}
.period-section h2 {{ font-family:'Barlow Condensed', sans-serif; font-size:16px; font-weight:600;
  letter-spacing:0.06em; text-transform:uppercase; color:var(--ink-faint); margin:0 0 10px;
  border-bottom:1px solid var(--border); padding-bottom:6px; }}
.chapter-rows {{ display:flex; flex-direction:column; gap:6px; }}
.chapter-row {{ display:flex; align-items:center; gap:14px; background:var(--panel);
  border:1px solid var(--border); border-radius:8px; padding:10px 14px; text-decoration:none;
  color:var(--ink); font-family:'IBM Plex Mono', monospace; font-size:13px; }}
.chapter-row:hover {{ border-color:var(--amber); }}
.chapter-row:hover .chapter-row-go {{ color:var(--amber); }}
.chapter-row-num {{ flex:none; width:26px; color:var(--ink-faint); font-weight:500; }}
.chapter-row-clock {{ flex:1; color:var(--ink); }}
.chapter-row-coverage {{ flex:none; color:var(--ink-faint); font-size:12px; }}
.chapter-row-coverage.low {{ color:var(--state-mid); }}
.chapter-row-go {{ flex:none; color:var(--ink-dim); }}
footer.credit {{ font-family:'IBM Plex Mono', monospace; font-size:10.5px; color:var(--ink-faint);
  text-align:center; padding-top:6px; }}
</style>
<div class="wrap">
  <div class="masthead">
    <h1>Pressure Read</h1>
    <span class="eyebrow">Full-match index</span>
    <div class="matchup">FIFA World Cup 2022 &middot; Round of 16 &middot; Education City Stadium<br>
      <b class="mar">Morocco</b> vs <b class="esp">Spain</b> &middot; full match, {total} chapters</div>
  </div>
  <div class="about">
    Same defensive-structure monitor as the original highlight window &mdash; live phase classification,
    pressure-on-ball-carrier, inter-line band and danger heatmap &mdash; now covering the <b>whole match</b>,
    split into ~5-minute chapters aligned to period breaks. A single continuous page isn't possible here:
    at this fidelity the full match's data would run several hundred megabytes, far past what one page can
    hold, so it's chaptered instead, like pages of a book &mdash; each chapter is the same quality as the
    original window, just a slice of it. Every chapter links to the next/previous and back here. <b>Ball
    tracked</b> is the share of that chapter's frames with a real tracked ball position &mdash; broadcast
    tracking has real gaps throughout this match (see the methodology doc), so it varies chapter to chapter;
    a lower number means more loose-ball/no-carrier stretches in that chapter, not a rendering problem.
  </div>
  {''.join(sections)}
  <footer class="credit">PFF FC 2022 World Cup tracking data &middot; game 10508 &middot; ~30Hz broadcast tracking, ball ~15Hz effective</footer>
</div>
'''
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(body)
    print(f"Wrote {OUT_PATH}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            urls = json.load(f)
    else:
        urls = {}
    build(urls)

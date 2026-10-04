"""Build the self-hosted website: home page, section pages and every chapter page, in the site design.

    python3 pipeline/build_web.py --analysis DIR --chapters-from DIR [--extra DIR] [--out DIR] [--only-match ID]

Pages (all relative links, so the folder works on any static host or straight from disk):

    index.html          hero with a short clip of real tracking, match cards, analysis cards, a report, the method
    matches.html        every match with its chapter list
    analysis.html       the ten Analysis views for any match (?m=<slug>&v=<view>)
    reports.html        the twelve scouting reports (?t=<slug>-<team>)
    compare.html        the comparison board, sortable
    broadcast.html      the broadcast-footage pages
    method.html         how the model works, and the Behind the Model pages
    matches/<site_folder>/chapter-NN.html

Inputs:
- the registry (pipeline/matches.json) and content/hub_matches.json (chapter labels and ball coverage)
- --analysis DIR: an_kinds.json, compare.json and reports.json (analysis/digest.py, compare.py, reports.py) and
  the images under <view>/<slug>.jpg. Point it at output/analysis, or at a folder holding the published set.
- chapter data, per chapter: output/chapters_<id>/anim_NN.json + heat_NN.json when the pipeline has exported the
  match here, otherwise the two data blocks are lifted out of an existing chapter page in
  --chapters-from DIR/matches/<site_folder>/chapter-NN.html (an earlier site build).
- --extra DIR: a previous site export; its behind-the-model/ and broadcast/ folders are copied as they are.
- templates/chapter_head.html + chapter_tail.html (the chapter player), templates/site/assets/ (CSS, JS), and
  assets/hero-clip.json (a 14-second tracking clip for the home page).
"""
import argparse
import html
import json
import os
import re
import shutil

from paths import OUT, root
from registry import R

NAV = [('index.html', 'Home'), ('matches.html', 'Matches'), ('analysis.html', 'Analysis'), ('reports.html', 'Reports'),
       ('compare.html', 'Compare'), ('broadcast.html', 'Broadcast'), ('method.html', 'Method')]
GITHUB = 'https://github.com/kianpetersmondry/defensive-structure-model'
CREDIT = {'pff': 'PFF FC 2022 World Cup tracking · game {id} · ~30 Hz broadcast tracking',
          'dfl': 'DFL Bundesliga open data (© DFL, CC-BY 4.0) · 25 Hz optical tracking, resampled to 30 Hz'}
SITE_CREDIT = 'World Cup: PFF FC tracking · Bundesliga: DFL open data (© DFL, CC-BY 4.0)'
ANIM_START = '<script id="match-data" type="application/json">'
HEAT_START = '<script id="heatmap-data" type="application/json">'
e = html.escape


# ---------------------------------------------------------------- match facts
def matches():
    published = {x['id']: x['versions'][0]['chapters'] for x in json.load(open(root('content', 'hub_matches.json'), encoding='utf-8'))}
    out = []
    for mid in R.ids():
        m = R.get(mid)
        h, a = m['home'], m['away']
        goals = [g[0] for g in m.get('goals', [])]
        sub = m['sub']
        note = sub.split(', ', 1)[1].replace('-', '–') if 'penalties' in sub else ''
        parts = m['competition'].split(' · ')
        comp = ' · '.join([parts[0].replace('FIFA ', '')] + parts[1:2])
        out.append(dict(mid=mid, slug=m['slug'], folder=m['site_folder'], provider=R.provider(mid),
                        home=h['name'], away=a['name'], hcode=h['code'], acode=a['code'], hc=h['color'], ac=a['color'],
                        hs=str(goals.count(h['name'])), as_=str(goals.count(a['name'])), note=note, comp=comp,
                        competition=m['competition'], chapters=published[m['hub_id']]))
    return out


def score(m):
    return f"{m['hs']}–{m['as_']}"


# ---------------------------------------------------------------- shared page parts
def nav(rootp, active):
    on = ' class="on"'
    links = ''.join(f'<a href="{rootp}{href}"{on if label == active else ""}>{label}</a>' for href, label in NAV)
    return (f'<nav class="nav"><div class="wrap"><a class="brand" href="{rootp}index.html"><span class="brand-mark" aria-hidden="true"></span>'
            f'Defensive Structure Model</a><div class="nav-links">{links}</div><div class="nav-cta">'
            f'<a class="btn small" href="{GITHUB}">GitHub</a><button type="button" class="menu-btn" aria-expanded="false">Menu</button></div></div></nav>')


def footer(rootp, credit=SITE_CREDIT):
    return (f'<footer class="site"><div class="wrap"><span>Built by Kian Peters Mondry · <a href="https://github.com/kianpetersmondry">GitHub</a>'
            f' · <a href="{rootp}method.html">Method</a></span><span>{e(credit)}</span></div></footer>'
            f'<script src="{rootp}assets/site.js"></script>')


def page(title, active, body, data=None, description=''):
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<' + chr(92) + '/') if data else ''
    data_js = f'<script>window.SITE = {payload};</script>' if data else ''
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(description)}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
{nav('', active)}
{body}
{data_js}
{footer('')}
</body>
</html>
'''


def mini_pitch(m):
    stripes = ''.join(f'<rect x="{i * 15}" width="15" height="50" fill="#183f2c"/>' for i in range(1, 7, 2))
    gid = f"g{m['slug']}"
    return (f'<svg class="mini" viewBox="0 0 105 50" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><defs><linearGradient id="{gid}" x1="0" x2="1">'
            f'<stop offset="0" stop-color="{m["hc"]}" stop-opacity=".32"/><stop offset=".5" stop-color="{m["hc"]}" stop-opacity="0"/>'
            f'<stop offset=".5" stop-color="{m["ac"]}" stop-opacity="0"/><stop offset="1" stop-color="{m["ac"]}" stop-opacity=".32"/></linearGradient></defs>'
            f'<rect width="105" height="50" fill="#163a28"/>{stripes}<rect width="105" height="50" fill="url(#{gid})"/>'
            '<g fill="none" stroke="rgba(235,245,238,.45)" stroke-width=".4"><rect x="2" y="-9" width="101" height="68"/><line x1="52.5" y1="-9" x2="52.5" y2="59"/>'
            '<circle cx="52.5" cy="25" r="9.15"/><rect x="2" y="4.8" width="16.5" height="40.3"/><rect x="86.5" y="4.8" width="16.5" height="40.3"/></g></svg>')


def match_card(m):
    note = f'<div class="note">{e(m["note"])}</div>' if m['note'] else ''
    return (f'<a class="match" href="matches/{m["folder"]}/chapter-01.html">{mini_pitch(m)}<div class="body"><div class="comp">{e(m["comp"])}</div>'
            f'<div class="teams"><span class="t"><i class="bar" style="background:{m["hc"]}"></i>{e(m["home"])}</span><span class="s">{m["hs"]}</span>'
            f'<span class="t"><i class="bar" style="background:{m["ac"]}"></i>{e(m["away"])}</span><span class="s">{m["as_"]}</span></div>{note}'
            f'<div class="foot"><span>{len(m["chapters"])} chapters</span><span class="go">Watch →</span></div></div></a>')


def view_card(k, thumb_src):
    return (f'<a class="view" href="analysis.html?v={k["key"]}"><img src="{thumb_src}" alt="" loading="lazy" width="560" height="315">'
            f'<div class="vt"><h3>{e(k["name"])}</h3><p>{e(k["blurb"])}</p></div></a>')


BLURBS = {'attack': 'Which flank each team attacked down, and how often it reached the final third.',
          'teams': 'Where each side spent the match, with every starter’s average spot.',
          'shape': 'How each team’s shape changes the moment it loses the ball.',
          'time': 'Back-line height, length and width as the match went on.',
          'gaps': 'The holes that keep opening inside the block, seen from behind it.',
          'lb': 'Every completed pass through the midfield or back line.',
          'chances': 'Every shot faced, rewound to the start of the attack.',
          'turn': 'Where the ball was won and lost, and what followed.',
          'press': 'What triggered each press, and whether it won the ball back.',
          'players': 'One small pitch per player, starters first.'}

COMPANIONS = [
    ('ppda-timeline.html', 'PPDA timeline', 'Pressing intensity (passes allowed per defensive action) in five-minute windows across the whole match, with every turnover.'),
    ('space-in-behind.html', 'Space in behind', 'How much dangerous, reachable space each team left behind its back line, phase by phase, and the moments worth pulling up on tape.'),
    ('ppda-vs-phase-detector.html', 'PPDA vs the phase detector', 'The tracking-based phases checked against an event-only pressing measure on one corrected match clock.'),
    ('engaged-vs-passive.html', 'Engaged vs passive defence', 'Each organised phase split into real pressure on the ball and passive shape-holding, against analysts’ pressure tags.'),
    ('ball-proximal-compactness.html', 'Ball-proximal compactness', 'Whether how tightly the five players nearest the ball are grouped predicts winning it back sooner.'),
]
BROADCAST = [
    ('broadcast/morocco-france/index.html', 'Broadcast vs 2-D animation', 'France vs Morocco · first six minutes',
     'The model drawn straight onto the World Cup semifinal broadcast, side by side with the 2-D animation of the same moments.'),
    ('broadcast/beach-lafc/part-1/index.html', 'Beach vs LAFC', 'Club match · 16 minutes of Veo footage',
     'The same read built from a club game’s own footage: players tracked from video, a 2-D map rebuilt from it, and both teams analysed.'),
]


# ---------------------------------------------------------------- pages
def home(ms, kinds, reports, clip, thumbs):
    n_ch = sum(len(m['chapters']) for m in ms)
    rep = next(r for r in reports if r['slug'] == 'mar-esp' and r['team'] == 'Morocco')
    chips = ''.join(chip_html(c) for c in rep['chips'][:3])
    goods = ''.join(f'<li>{e(x)}</li>' for x in (rep['good'][0], rep['exploit'][0]))
    cm = clip['meta']
    body = f'''<header class="hero"><div class="wrap">
<div><div class="eyebrow">Football tracking analysis</div><h1>How teams defend, frame by frame</h1>
<p class="lede">A model that reads every frame of tracking data to show each team’s shape without the ball, the pressure on the ball carrier, and the space it leaves to be exploited.</p>
<div class="hero-actions"><a class="btn primary" href="matches.html">Watch a match</a><a class="btn" href="analysis.html">Explore the analysis</a></div></div>
<div class="pitch-card"><canvas id="heroPitch" width="1050" height="680" aria-label="Animated 2-D tracking: {e(cm['homeTeam'])} defending against {e(cm['awayTeam'])}"></canvas>
<div class="pitch-overlay"><div class="readout"><span class="team"><span class="dot" style="background:{cm['homeColor']}"></span>{e(cm['homeTeam'])} defending</span><span class="pill phase" id="heroPhase">—</span><span class="pill tag" id="heroTag">—</span></div><div class="clock" id="heroClock" style="position:static">0:00</div></div>
<div class="pitch-caption"><span><b>{e(cm['homeTeam'])} vs {e(cm['awayTeam'])}</b> · {e(clip.get('caption', ''))}</span><a href="matches/{clip['folder']}/chapter-01.html" class="link">Watch this chapter →</a></div></div>
</div></header>
<div class="strip"><div class="wrap">
<div class="stat"><div class="n">{len(ms)}</div><div class="l">full matches, World Cup and Bundesliga</div></div>
<div class="stat"><div class="n">{n_ch}</div><div class="l">five-minute chapters of 2-D animation</div></div>
<div class="stat"><div class="n">{len(kinds)}</div><div class="l">match-level analysis views</div></div>
<div class="stat"><div class="n">{len(reports)}</div><div class="l">team scouting reports</div></div>
</div></div>
<section class="block" id="matches"><div class="wrap"><div class="sec-head"><div><div class="eyebrow">Watch the match</div><h2>Matches</h2>
<p>Every match is rendered in full, split into five-minute chapters you can play straight through.</p></div><a class="link" href="matches.html">All chapters →</a></div>
<div class="matches">{''.join(match_card(m) for m in ms)}</div></div></section>
<section class="block"><div class="wrap"><div class="sec-head"><div><div class="eyebrow">Analysis</div><h2>Ten ways to read a defence</h2>
<p>The same views for every match, each team drawn attacking left to right.</p></div><a class="link" href="analysis.html">Open the analysis →</a></div>
<div class="views">{''.join(view_card(k, thumbs[k['key']]) for k in kinds)}</div></div></section>
<section class="block"><div class="wrap"><div class="sec-head"><div><div class="eyebrow">Scouting reports</div><h2>What they did well, and how to get at them</h2></div>
<a class="link" href="reports.html">All {len(reports)} reports →</a></div>
<article class="report"><div class="left"><div class="who"><span class="dot lg" style="background:{rep['color']}"></span>{e(rep['team'])} · vs {e(rep['opp'])} · {e(rep['sub'].split(' · ')[0])}</div>
<blockquote>{e(rep['headline'])}</blockquote><ul>{goods}</ul>
<div class="views-links"><a class="btn small" href="reports.html?t=mar-esp-morocco">Read the full report →</a></div></div>
<div class="right"><div class="eyebrow" style="margin-bottom:6px">Ranked against all {len(reports)} teams</div>{chips}</div></article></div></section>
<section class="block"><div class="wrap"><div class="sec-head"><div><div class="eyebrow">Behind the model</div><h2>How it works</h2></div><a class="link" href="method.html">Read the methodology →</a></div>
<div class="method">
<a class="mcard" href="method.html#phase"><div class="num">01 · Phase</div><h3>Shape without the ball</h3><p>Back-line depth and team compactness classify every frame as high press, mid block, low block or transition.</p></a>
<a class="mcard" href="method.html#pressure"><div class="num">02 · Pressure</div><h3>Pressure on the ball</h3><p>Each defender’s time to reach the carrier, from position, speed and reaction time, checked against analysts’ pressure tags.</p></a>
<a class="mcard" href="method.html#space"><div class="num">03 · Space</div><h3>Space to exploit</h3><p>Where the attacking team could reach dangerous space with one realistic pass, and where the gaps open inside the block.</p></a>
</div></div></section>'''
    return page('Defensive Structure Model', 'Home', body, {'clip': clip},
                'How football teams defend, frame by frame: a tracking-data model of defensive shape, pressure and space.')


def chip_html(c):
    pos = (c['rank'] - 1) / 11 * 100
    label = c['label'][0].upper() + c['label'][1:]
    return (f'<div class="chip"><span class="k">{e(label)}</span><span class="v">{e(c["value"])}</span>'
            f'<span class="r"><span class="rankbar"><i style="left:{pos:.0f}%"></i></span>{e(c["rank_text"])}</span></div>')


def matches_page(ms):
    blocks = []
    for m in ms:
        items = ''.join(f'<a class="chitem" href="matches/{m["folder"]}/chapter-{c["idx"]:02d}.html"><b>Chapter {c["idx"]}</b>'
                        f'<span>{e(c["period"])} · {e(c["clock"])} · ball {c["coverage"]}%</span></a>' for c in m['chapters'])
        note = f' · {e(m["note"])}' if m['note'] else ''
        blocks.append(f'<div class="match-block" id="{m["slug"]}"><div class="comp">{e(m["competition"])}</div>'
                      f'<h2><span class="dot lg" style="background:{m["hc"]}"></span>{e(m["home"])} <span class="vs">{score(m)}</span> {e(m["away"])}'
                      f'<span class="dot lg" style="background:{m["ac"]}"></span></h2><p>{len(m["chapters"])} chapters{note}</p><div class="chlist">{items}</div></div>')
    body = (f'<header class="page-head"><div class="wrap"><div class="eyebrow">Watch the match</div><h1>Matches</h1>'
            f'<p>Each match is split into chapters of about five minutes, aligned to the period breaks. Every chapter links to the next, so you can watch straight through. '
            f'The ball percentage is the share of a chapter’s frames with a tracked ball.</p></div></header>'
            f'<section class="block" style="padding-top:28px"><div class="wrap">{"".join(blocks)}</div></section>')
    return page('Matches · Defensive Structure Model', 'Matches', body)


def analysis_page(ms, kinds):
    body = ('<header class="page-head"><div class="wrap"><div class="eyebrow">Analysis</div><h1>Ten ways to read a defence</h1>'
            '<p>Pick a match, then a view. Every team is drawn attacking left to right, and only live play counts.</p></div></header>'
            '<section class="block" style="padding-top:28px" id="viewer"><div class="wrap">'
            '<div class="tabs" id="viewerMatches" role="tablist" aria-label="Match"></div>'
            '<div class="subtabs" id="viewerKinds" role="tablist" aria-label="View"></div>'
            '<p class="viewer-desc" id="viewerDesc"></p>'
            '<figure class="viewer-fig"><img id="viewerImg" alt=""></figure>'
            '<div class="viewer-foot"><span id="viewerCap"></span><a id="viewerFull" href="#" target="_blank" rel="noopener">Open full size →</a></div>'
            '</div></section>')
    data = {'kinds': kinds, 'matches': [dict(m, **{'as': m['as_']}) for m in ms]}
    return page('Analysis · Defensive Structure Model', 'Analysis', body, data)


def reports_page(reports, kinds):
    body = ('<header class="page-head"><div class="wrap"><div class="eyebrow">Scouting reports</div><h1>What they did well, and how to get at them</h1>'
            '<p>One report per team per match, drawn from the Analysis views. The five stats are ranked against every team on the site.</p></div></header>'
            '<section class="block" style="padding-top:28px"><div class="wrap"><div class="tabs" id="reportTabs" role="tablist" aria-label="Team"></div>'
            '<article class="report" id="reportView" style="margin-top:18px"></article></div></section>')
    return page('Scouting reports · Defensive Structure Model', 'Reports', body,
                {'reports': reports, 'kinds': [{'name': k['name'], 'key': k['key']} for k in kinds]})


def compare_page(board):
    cols = [('possession', 'Possession', 'Share of live play with the ball'),
            ('back_line', 'Back line', 'Median height of the deepest four outfielders without the ball, metres from own goal'),
            ('length', 'Length', 'Median block length without the ball'), ('width', 'Width', 'Median block width without the ball'),
            ('holes', 'Holes', 'Open space inside the block more than 8 m from every defender'),
            ('free', 'Free opp.', 'Attackers inside the block with no defender within 5 m'),
            ('wins', 'Ball wins', 'Open-play ball wins'), ('wins_high', 'In opp. half', 'Share of ball wins in the opponents’ half'),
            ('fast', 'Fast attacks', 'Ball wins in or into the final third within 10 s'),
            ('shots_for', 'Shots', 'Shots taken'), ('shots_against', 'Faced', 'Shots faced')]
    heads = ''.join(f'<th scope="col"><button type="button" data-col="{c}" title="{e(t)}">{e(n)}</button></th>' for c, n, t in cols)
    body = (f'<header class="page-head"><div class="wrap"><div class="eyebrow">Team comparison</div><h1>All {len(board)} teams side by side</h1>'
            '<p>Live play only. Click a column to rank the teams by it.</p></div></header>'
            '<section class="block" style="padding-top:28px"><div class="wrap"><div class="cmp-scroll"><table class="cmp"><thead>'
            '<tr class="grp"><th></th><th>With the ball</th><th colspan="5">Without the ball</th><th colspan="3">Winning it back</th><th colspan="2">Shots</th></tr>'
            f'<tr><th scope="col">Team</th>{heads}</tr></thead><tbody id="cmpBody"></tbody></table></div>'
            '<p class="notes">Back line: deepest four outfielders, metres from own goal. Length and width: outfield spread. Holes: open space inside the block more than 8 m from every defender. '
            'Free opponents: attackers inside the block with no defender within 5 m. Fast attacks: ball wins where the team was in or into the final third within 10 s. '
            'Medians or totals over live play; shots from the event feed, extra time included for the two 120-minute matches.</p></div></section>')
    return page('Compare · Defensive Structure Model', 'Compare', body, {'board': board})


def broadcast_page():
    cards = ''.join(f'<a class="mcard" href="{href}"><div class="num">{e(sub)}</div><h3>{e(title)}</h3><p>{e(text)}</p></a>' for href, title, sub, text in BROADCAST)
    body = ('<header class="page-head"><div class="wrap"><div class="eyebrow">Live broadcast animations</div><h1>The model on real footage</h1>'
            '<p>The same defensive read, run on match video instead of tracking data: players detected and tracked from the broadcast or a club camera, '
            'the camera solved from the pitch lines, and the shape, pressure and danger drawn back onto the picture.</p></div></header>'
            f'<section class="block" style="padding-top:28px"><div class="wrap"><div class="method" style="grid-template-columns:repeat(2,1fr)">{cards}</div></div></section>')
    return page('Broadcast · Defensive Structure Model', 'Broadcast', body)


def method_page():
    cards = ''.join(f'<a class="mcard" href="behind-the-model/{href}"><div class="num">Morocco vs Spain</div><h3>{e(t)}</h3><p>{e(p)}</p></a>' for href, t, p in COMPANIONS)
    body = f'''<header class="page-head"><div class="wrap"><div class="eyebrow">Behind the model</div><h1>How it works</h1>
<p>Everything runs on tracking data: the position of every player and the ball, about 30 times a second, for the whole match.</p></div></header>
<section class="block" style="padding-top:20px"><div class="wrap"><div class="prose">
<h2 id="phase">Shape without the ball</h2>
<p>For every frame, the team without the ball is classed as <b>high press, mid block, low block</b> or <b>transition</b>. The call uses two measures from the literature: how far the back line (the deepest four outfielders) sits from its own goal, and how compact the team is from back to front. Transition covers the seconds after the ball changes hands, before the defence has reorganised.</p>
<h2 id="pressure">Pressure on the ball</h2>
<p>Each defender near the ball gets a time to intercept the carrier, from his position, current velocity, reaction time and his own top speed. The times combine into one 0–100% pressure score. Against PFF’s analyst-tagged pressure events, the score averages 0.74 on tagged moments and 0.54 elsewhere, and a per-match threshold splits each phase into <b>engaged</b> (someone is really closing the ball down) and <b>passive</b> shape-holding. A second tag, <b>tight</b> or <b>loose</b>, reads how closely the five players nearest the ball are grouped.</p>
<h2 id="space">Space to exploit</h2>
<p>The danger heatmap is Dangerous Accessible Space: a physics-based pass simulation combined with a danger model, showing where the team on the ball could reach dangerous space with one realistic pass, right now. The match-level views add where the gaps open inside the block, which passes break its lines, and how each chance conceded began.</p>
<h2>Checks</h2>
<p>The phases were checked against PPDA, a pressing statistic that uses only events: in organised defence it rises from high press to mid block to low block for both teams, as pressing theory predicts. The broadcast-footage prototype agrees with PFF’s tracking to a median 0.83 m per player. The full methodology, the literature behind it and every check are in the <a href="{GITHUB}">GitHub repository</a>.</p>
</div></div></section>
<section class="block"><div class="wrap"><div class="sec-head"><div><div class="eyebrow">Companion analyses</div><h2>Behind the model</h2>
<p>Five deeper studies from the first match, Morocco vs Spain, that shaped the definitions above.</p></div></div>
<div class="method">{cards}</div></div></section>'''
    return page('Method · Defensive Structure Model', 'Method', body)


# ---------------------------------------------------------------- chapters
def chapter_data(m, idx, src_dir):
    """(match-data json text, heatmap json text) for one chapter."""
    d = R.chapters_dir(m['mid'])
    anim, heat = os.path.join(d, f'anim_{idx:02d}.json'), os.path.join(d, f'heat_{idx:02d}.json')
    if os.path.exists(anim) and os.path.exists(heat):
        return open(anim, encoding='utf-8').read(), open(heat, encoding='utf-8').read()
    if not src_dir:
        return None
    path = os.path.join(src_dir, 'matches', m['folder'], f'chapter-{idx:02d}.html')
    if not os.path.exists(path):
        return None
    s = open(path, encoding='utf-8').read()
    a0 = s.index(ANIM_START) + len(ANIM_START)
    a1 = s.index('</script>', a0)
    h0 = s.index(HEAT_START, a1) + len(HEAT_START)
    h1 = s.index('</script>', h0)
    return s[a0:a1].strip(), s[h0:h1].strip()


def build_chapters(ms, src_dir, out_dir, only=None):
    head_t = open(root('templates', 'chapter_head.html'), encoding='utf-8').read()
    tail_t = open(root('templates', 'chapter_tail.html'), encoding='utf-8').read()
    rootp = '../../'
    done, missing = 0, []
    for m in ms:
        if only and m['mid'] != only:
            continue
        chs, n = m['chapters'], len(m['chapters'])
        cur = ' aria-current="page"'
        tabs = ''.join(f'<a class="mtab" href="{rootp}matches/{x["folder"]}/chapter-01.html"{cur if x is m else ""}>'
                       f'<span class="dot" style="background:{x["hc"]}"></span>{x["hcode"]} <span class="sc">{score(x)}</span> {x["acode"]}'
                       f'<span class="dot" style="background:{x["ac"]}"></span></a>' for x in ms)
        os.makedirs(os.path.join(out_dir, 'matches', m['folder']), exist_ok=True)
        credit = CREDIT[m['provider']].format(id=m['mid'])
        for i, c in enumerate(chs):
            idx = c['idx']
            data = chapter_data(m, idx, src_dir)
            if data is None:
                missing.append(f"{m['folder']}/chapter-{idx:02d}")
                continue
            label = lambda cc: f"{cc['period']} · {cc['clock']}"
            opts = ''.join(f'<option value="chapter-{cc["idx"]:02d}.html"{" selected" if cc is c else ""}>Ch {cc["idx"]} of {n} · {e(label(cc))}</option>' for cc in chs)
            prev_c = chs[i - 1] if i else None
            next_c = chs[i + 1] if i + 1 < n else None
            prev = (f'<a class="btn small" href="chapter-{prev_c["idx"]:02d}.html">← Prev</a>' if prev_c else '<span class="btn small disabled">← Prev</span>')
            nxt = (f'<a class="btn small" data-next href="chapter-{next_c["idx"]:02d}.html">Next →</a>' if next_c else '<span class="btn small disabled">Next →</span>')
            foot_prev = (f'<a class="chlink" href="chapter-{prev_c["idx"]:02d}.html">Previous chapter<b>← Ch {prev_c["idx"]} · {e(prev_c["clock"])}</b></a>'
                         if prev_c else '<span class="chlink off">First chapter<b>Kick-off</b></span>')
            foot_next = (f'<a class="chlink next" href="chapter-{next_c["idx"]:02d}.html">Next chapter<b>Ch {next_c["idx"]} · {e(next_c["clock"])} →</b></a>'
                         if next_c else f'<a class="chlink next" href="{rootp}matches.html">End of the match<b>All matches →</b></a>')
            comp = m['comp'] + (f" · {m['note']}" if m['note'] else '')
            fill = {'TITLE': f"{m['home']} {score(m)} {m['away']} · Chapter {idx} of {n} · Defensive Structure Model",
                    'ROOT': rootp, 'HOME_COLOR': m['hc'], 'AWAY_COLOR': m['ac'], 'NAV': nav(rootp, 'Matches'), 'MATCH_TABS': tabs,
                    'HOME': e(m['home']), 'AWAY': e(m['away']), 'SCORE': score(m), 'COMP': e(comp), 'CHAPTER_OPTIONS': opts,
                    'PREV': prev, 'NEXT': nxt, 'FOOT_PREV': foot_prev, 'FOOT_NEXT': foot_next,
                    'HOME_CODE': m['hcode'], 'AWAY_CODE': m['acode'], 'FOOTER': footer(rootp, credit)}
            head = head_t
            for k, v in fill.items():
                head = head.replace('{{' + k + '}}', v)
            with open(os.path.join(out_dir, 'matches', m['folder'], f'chapter-{idx:02d}.html'), 'w', encoding='utf-8') as f:
                f.write(head + data[0] + '\n</script>\n' + HEAT_START + '\n' + data[1] + tail_t + '</body>\n</html>\n')
            done += 1
    return done, missing


# ---------------------------------------------------------------- main
def load_kinds(adir):
    kinds = json.load(open(os.path.join(adir, 'an_kinds.json'), encoding='utf-8'))
    for k in kinds:
        k['key'] = k['data'][0]['src'].split('/')[0]
        k['blurb'] = BLURBS.get(k['key'], '')
    return kinds


def make_thumbs(adir, kinds, out_dir):
    """Card images for the home page: a crop of each view's image, below its title."""
    from PIL import Image
    picks = {'attack': 'arg-fra', 'teams': 'mar-esp', 'shape': 'fra-mar', 'time': 'ger-jpn', 'gaps': 'mar-esp',
             'lb': 'arg-fra', 'chances': 'bel-can', 'turn': 'ger-jpn', 'press': 'ger-jpn', 'players': 'arg-fra'}
    os.makedirs(os.path.join(out_dir, 'assets', 'thumbs'), exist_ok=True)
    out = {}
    for k in kinds:
        slug = picks.get(k['key'], k['data'][0]['slug'])
        im = Image.open(os.path.join(adir, k['key'], f'{slug}.jpg')).convert('RGB')
        w = im.size[0]
        top = int(w * 0.12)
        crop = im.crop((0, top, w, top + int(w * 0.5625))).resize((560, 315))
        rel = f"assets/thumbs/{k['key']}.jpg"
        crop.save(os.path.join(out_dir, rel), 'JPEG', quality=78, optimize=True)
        out[k['key']] = rel
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--analysis', default=os.path.join(OUT, 'analysis'), help='folder with an_kinds.json, compare.json, reports.json and the images')
    ap.add_argument('--chapters-from', help='an earlier site build to lift chapter data from')
    ap.add_argument('--extra', help='a previous site export holding behind-the-model/ and broadcast/')
    ap.add_argument('--out', default=os.path.join(OUT, 'web'))
    ap.add_argument('--only-match', help='build just this match’s chapter pages (into --out)')
    a = ap.parse_args()
    ms = matches()
    if a.only_match:
        done, missing = build_chapters(ms, a.chapters_from, a.out, a.only_match)
        print(f'{done} chapter pages -> {a.out}' + (f'; missing: {missing}' if missing else ''))
        return

    kinds = load_kinds(a.analysis)
    board = json.load(open(os.path.join(a.analysis, 'compare.json'), encoding='utf-8'))
    reports = json.load(open(os.path.join(a.analysis, 'reports.json'), encoding='utf-8'))
    order = [m['slug'] for m in ms]
    for k in kinds:
        k['data'].sort(key=lambda d: order.index(d['slug']))
        assert [d['slug'] for d in k['data']] == order, f"{k['name']}: needs an image for every match"
    os.makedirs(a.out, exist_ok=True)
    shutil.copytree(root('templates', 'site', 'assets'), os.path.join(a.out, 'assets'), dirs_exist_ok=True)
    for k in kinds:
        for d in k['data']:
            dst = os.path.join(a.out, d['src'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(a.analysis, d['src']), dst)
    thumbs = make_thumbs(a.analysis, kinds, a.out)
    clip = json.load(open(root('templates', 'site', 'hero-clip.json'), encoding='utf-8'))
    pages = {'index.html': home(ms, kinds, reports, clip, thumbs), 'matches.html': matches_page(ms),
             'analysis.html': analysis_page(ms, kinds), 'reports.html': reports_page(reports, kinds),
             'compare.html': compare_page(board), 'broadcast.html': broadcast_page(), 'method.html': method_page()}
    for name, text in pages.items():
        open(os.path.join(a.out, name), 'w', encoding='utf-8').write(text)
    if a.extra:
        for d in ('behind-the-model', 'broadcast'):
            if os.path.isdir(os.path.join(a.extra, d)):
                shutil.copytree(os.path.join(a.extra, d), os.path.join(a.out, d), dirs_exist_ok=True)
    done, missing = build_chapters(ms, a.chapters_from, a.out)
    print(f'site: {len(pages)} pages, {done} chapter pages, {sum(len(k["data"]) for k in kinds)} analysis images -> {a.out}')
    if missing:
        print(f'{len(missing)} chapters had no data:', ', '.join(missing[:6]), '...' if len(missing) > 6 else '')


if __name__ == '__main__':
    main()

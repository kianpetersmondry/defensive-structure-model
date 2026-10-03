"""
Builds the public-facing Defensive Structure Monitor site: a one-page "home" for the whole
project (hero + match/version/chapter picker + companion-analysis links),
meant to be shared as a single URL. Replaces the old flat chapter-index and
then the two-dropdown-only hub with a fuller page, same URL each time.

Data-driven by design: MATCHES is a list of match objects, each carrying a
list of phase-detection VERSIONS, and each version carrying its own
chapters[]. Originally exactly one match and exactly one version
("Baseline -- depth + inter-line", the classification logic currently
live). Adding a second version later (e.g. one that also weighs pressure
and/or PPDA) is just appending another entry to that match's versions list,
pointing at that version's own separately-exported/published chapter set,
and re-running this script -- nothing in the page logic assumes a single
version.

Round 30 added the second match this was designed for (France vs Morocco,
game 10515) -- confirms the MATCHES list needed no structural change, just
a second entry built the same way as the first, from that match's own
manifest.json + chapter_urls file. COMPANIONS lists the other published
analyses (PPDA Timeline, Space in Behind, etc. -- currently all scoped to
match 1 only, see the methodology doc) so they're discoverable from the
same public link instead of living only in project notes the site's
visitors never see.
"""
import json


def load_match_chapters(manifest_path, urls_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(urls_path) as f:
        urls = json.load(f)
    chapters = []
    for ch in manifest["chapters"]:
        idx = ch["chapterIndex"]
        chapters.append({
            "idx": idx,
            "period": ch["periodLabel"],
            "clock": ch["clockLabel"],
            "coverage": round(100 - ch["noBallPct"]),
            "url": urls["chapters"][str(idx)],
        })
    return chapters


match1_chapters = load_match_chapters("chapters/manifest.json", "final_urls.json")
match2_chapters = load_match_chapters("chapters_10515/manifest.json", "chapter_urls_10515.json")
match3_chapters = load_match_chapters("chapters_3821/manifest.json", "chapter_urls_3821.json")
match4_chapters = load_match_chapters("chapters_3823/manifest.json", "chapter_urls_3823.json")
match5_chapters = load_match_chapters("chapters_10517/manifest.json", "chapter_urls_10517.json")

MATCHES = [
    {
        "id": "mar-esp-2022-r16",
        "home": "Morocco",
        "away": "Spain",
        "label": "Morocco vs Spain",
        "competition": "FIFA World Cup 2022 · Round of 16 · Education City Stadium",
        "versions": [
            {
                "id": "v1-depth-interline",
                "label": "Baseline — depth + inter-line",
                "note": "Classifies each team's out-of-possession phase (High Press / Mid Block / "
                        "Low Block / Transition) from back-line depth and inter-line compactness only.",
                "chapters": match1_chapters,
            },
        ],
    },
    {
        "id": "fra-mar-2022-sf",
        "home": "France",
        "away": "Morocco",
        "label": "France vs Morocco",
        "competition": "FIFA World Cup 2022 · Semifinal · Al Bayt Stadium",
        "versions": [
            {
                "id": "v1-depth-interline",
                "label": "Baseline — depth + inter-line",
                "note": "Same classification logic as the Morocco vs Spain baseline: back-line depth + "
                        "inter-line compactness only. This match's raw tracking data has no per-player "
                        "speed field (unlike 10508), so the pressure-on-ball-carrier score falls back to "
                        "a fixed max-speed assumption per player instead of an empirically-derived one "
                        "-- see the methodology doc.",
                "chapters": match2_chapters,
            },
        ],
    },
    {
        "id": "ger-jpn-2022-grp",
        "home": "Germany",
        "away": "Japan",
        "label": "Germany vs Japan",
        "competition": "FIFA World Cup 2022 · Group Stage · Khalifa International Stadium",
        "versions": [
            {
                "id": "v1-depth-interline",
                "label": "Baseline — depth + inter-line",
                "note": "Same classification logic as the other two matches: back-line depth + inter-line "
                        "compactness only. Like 10515, this match's raw tracking has no per-player speed "
                        "field, so the pressure-on-ball-carrier score uses each player's own empirically "
                        "derived max speed (from smoothed-position deltas) rather than a flat assumption "
                        "-- see the methodology doc. Japan came back from 0-1 down to win 2-1.",
                "chapters": match3_chapters,
            },
        ],
    },
    {
        "id": "bel-can-2022-grp",
        "home": "Belgium",
        "away": "Canada",
        "label": "Belgium vs Canada",
        "competition": "FIFA World Cup 2022 · Group Stage · Ahmad Bin Ali Stadium",
        "versions": [
            {
                "id": "v1-depth-interline",
                "label": "Baseline — depth + inter-line",
                "note": "Same classification logic as the other matches: back-line depth + inter-line "
                        "compactness only. Like Germany vs Japan, this match's raw tracking has no "
                        "per-player speed field, so the pressure-on-ball-carrier score uses each player's "
                        "own empirically derived max speed rather than a flat assumption -- see the "
                        "methodology doc. Belgium sat deep to protect an early goal; Canada dominated "
                        "territorially and missed a penalty, and lost 1-0.",
                "chapters": match4_chapters,
            },
        ],
    },
    {
        "id": "arg-fra-2022-final",
        "home": "Argentina",
        "away": "France",
        "label": "Argentina vs France",
        "competition": "FIFA World Cup 2022 · Final · Lusail Stadium",
        "versions": [
            {
                "id": "v1-depth-interline",
                "label": "Baseline — depth + inter-line",
                "note": "Same classification logic as the other matches: back-line depth + inter-line "
                        "compactness only. Unlike matches 2-4, this match's raw tracking DOES carry a "
                        "per-player speed field (like the Morocco vs Spain baseline), so the "
                        "pressure-on-ball-carrier score uses each player's own real tracked max speed -- "
                        "see the methodology doc. One of the wildest finals in World Cup history: Argentina "
                        "led 2-0, France drew level 2-2 in a two-minute Mbappe burst late in regulation, "
                        "Argentina led again in extra time, Mbappe completed his hat-trick to make it 3-3, "
                        "and Argentina won 4-2 on penalties.",
                "chapters": match5_chapters,
            },
        ],
    },
]

COMPANIONS = [
    {
        "title": "PPDA Timeline",
        "icon": "\U0001F4C8",
        "url": "https://claude.ai/code/artifact/56e3112f-e014-4ea8-8de0-d64539289eb4",
        "blurb": "Passes allowed per defensive action for both teams across the full match, in 5-minute "
                 "windows, plus every turnover sized by how many passes came before it.",
    },
    {
        "title": "Space in Behind",
        "icon": "\U0001F573️",
        "url": "https://claude.ai/code/artifact/75d6a1ea-6984-4486-a2f7-13afb028bd96",
        "blurb": "How much dangerous, pass-accessible space each team leaves behind its own back line, "
                 "broken down by defensive phase, with the six riskiest moments flagged for review.",
    },
    {
        "title": "PPDA vs. Phase Detector",
        "icon": "\U0001F500",
        "url": "https://claude.ai/code/artifact/e03e03e1-f3a2-4a3b-9237-46308b26cb09",
        "blurb": "Overlays the geometric phase detector against PPDA on one shared, corrected time axis "
                 "with a synchronized hover crosshair — resolving PPDA cleanly by dominant defensive phase "
                 "for both teams across the full match.",
    },
    {
        "title": "Engaged vs. Passive Defense",
        "icon": "\U0001F3AF",
        "url": "https://claude.ai/code/artifact/dce35fa2-8885-46f9-8bd1-d128bc454c21",
        "blurb": "Splits each geometric phase into real 1v1 engagement vs. passive shape-holding, using a "
                 "pressure threshold validated against PFF's own analyst-tagged pressure events — every "
                 "phase turns out to be a genuine mix for both teams.",
    },
    {
        "title": "Ball-Proximal Compactness",
        "icon": "\U0001F9F2",
        "url": "https://claude.ai/code/artifact/905de7ea-08a1-46fe-8ee2-b9fb6d4e1562",
        "blurb": "Tests whether how tightly the 5 players nearest the ball are grouped predicts winning it "
                 "back sooner, using turnovers straight from the event feed as an independent outcome — it "
                 "doesn't clear the bar for a third classification variable, so it ships as a descriptive tag.",
    },
]

MATCHES_JSON = json.dumps(MATCHES, separators=(",", ":"), ensure_ascii=False)

HTML = """<title>Defensive Structure Monitor</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #0c1512; --panel: #131f1a; --panel-raised: #17251f; --border: #253830;
  --ink: #eef3ee; --ink-dim: #93a89c; --ink-faint: #5e7268;
  --amber: #ffb020; --state-mid: #d99a2b;
}
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:var(--bg); color:var(--ink);
  font-family:'IBM Plex Sans', system-ui, sans-serif; -webkit-font-smoothing:antialiased; }
body { min-height:100vh; }
.wrap { max-width:820px; margin:0 auto; padding:0 20px 56px; display:flex; flex-direction:column; gap:28px; }

/* ---- hero ---- */
.hero { padding:44px 0 22px; border-bottom:1px solid var(--border); }
.hero .eyebrow { font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:0.12em;
  text-transform:uppercase; color:var(--amber); }
.hero h1 { font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:52px; line-height:1.02;
  letter-spacing:0.01em; text-transform:uppercase; margin:8px 0 14px; text-wrap:balance; }
.hero p { font-size:15px; line-height:1.65; color:var(--ink-dim); max-width:58ch; margin:0 0 16px; }
.hero .matchup { display:inline-flex; align-items:center; gap:8px; font-family:'IBM Plex Mono', monospace;
  font-size:12.5px; color:var(--ink-faint); }
.hero .matchup b { color:var(--ink-dim); font-weight:600; }

/* ---- section framing ---- */
.section-label { font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.09em;
  text-transform:uppercase; color:var(--ink-faint); margin:0 0 12px; display:flex; align-items:center; gap:10px; }
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }

/* ---- picker ---- */
.picker { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:22px 22px 24px;
  display:flex; flex-direction:column; gap:18px; }
.field { display:flex; flex-direction:column; gap:7px; }
.field label { font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:0.07em;
  text-transform:uppercase; color:var(--ink-faint); }
.field label .step { color:var(--amber); }
select { appearance:none; -webkit-appearance:none; width:100%; background:var(--panel-raised);
  border:1px solid var(--border); border-radius:8px; color:var(--ink); font-family:'IBM Plex Sans', sans-serif;
  font-size:15px; font-weight:500; padding:12px 40px 12px 14px;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='9' viewBox='0 0 14 9'><path d='M1 1l6 6 6-6' stroke='%2393a89c' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>");
  background-repeat:no-repeat; background-position:right 14px center; cursor:pointer; }
select:focus-visible { outline:2px solid var(--amber); outline-offset:1px; }
select:disabled { color:var(--ink-faint); cursor:not-allowed; opacity:0.6; }

.version-note { font-size:12px; line-height:1.5; color:var(--ink-faint); margin-top:-1px; }
.version-note[hidden] { display:none; }

.preview { display:flex; align-items:center; gap:14px; background:var(--panel-raised);
  border:1px solid var(--border); border-radius:8px; padding:12px 14px;
  font-family:'IBM Plex Mono', monospace; font-size:13px; min-height:20px; }
.preview[hidden] { display:none; }
.preview-period { color:var(--ink-dim); }
.preview-clock { color:var(--ink); font-weight:500; }
.preview-coverage { margin-left:auto; color:var(--ink-faint); font-size:12px; }
.preview-coverage.low { color:var(--state-mid); }

.open-btn { display:flex; align-items:center; justify-content:center; gap:8px; text-decoration:none;
  background:var(--amber); color:#1a1400; font-family:'Barlow Condensed', sans-serif; font-weight:700;
  font-size:17px; letter-spacing:0.03em; text-transform:uppercase; border-radius:8px; padding:13px 18px;
  transition:filter 0.12s ease; }
.open-btn:hover { filter:brightness(1.08); }
.open-btn:focus-visible { outline:2px solid var(--ink); outline-offset:2px; }
.open-btn[aria-disabled="true"] { pointer-events:none; background:var(--panel-raised); color:var(--ink-faint); }

.picker-note { font-size:12.5px; line-height:1.6; color:var(--ink-faint); }
.picker-note b { color:var(--ink-dim); }

/* ---- companion cards ---- */
.companions { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }
@media (max-width:720px) { .companions { grid-template-columns:1fr 1fr; } }
@media (max-width:460px) { .companions { grid-template-columns:1fr; } }
.companion-card { display:flex; flex-direction:column; gap:10px; background:var(--panel);
  border:1px solid var(--border); border-radius:12px; padding:18px 20px; text-decoration:none; color:var(--ink); }
.companion-card:hover { border-color:var(--amber); }
.companion-card:hover .companion-go { color:var(--amber); }
.companion-icon { font-size:22px; line-height:1; }
.companion-title { font-family:'Barlow Condensed', sans-serif; font-weight:700; font-size:20px;
  text-transform:uppercase; letter-spacing:0.01em; }
.companion-blurb { font-size:13px; line-height:1.55; color:var(--ink-dim); flex:1; }
.companion-go { font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:0.04em;
  text-transform:uppercase; color:var(--ink-faint); }

footer.credit { font-family:'IBM Plex Mono', monospace; font-size:10.5px; color:var(--ink-faint);
  text-align:center; padding-top:6px; }
</style>
<div class="wrap">
  <div class="hero">
    <span class="eyebrow">Defensive structure analysis</span>
    <h1>Defensive Structure Monitor</h1>
    <p>Renders a full match and reads each team's defensive structure whenever they're out of possession —
      where the space and gaps are, what's working, and how it could be exploited. Live phase classification,
      a pressure-on-ball-carrier score, an inter-line band, and a danger heatmap, all updating frame by frame.</p>
    <div class="matchup">__MATCHUP_LINE__</div>
  </div>

  <div>
    <p class="section-label">Watch the match</p>
    <div class="picker">
      <div class="field">
        <label for="matchSelect"><span class="step">Step 1</span> · Match</label>
        <select id="matchSelect">
          <option value="" selected disabled>Select a match…</option>
        </select>
      </div>
      <div class="field">
        <label for="versionSelect"><span class="step">Step 2</span> · Phase-detection definition</label>
        <select id="versionSelect" disabled>
          <option value="" selected>Select a match first…</option>
        </select>
        <p class="version-note" id="versionNote" hidden></p>
      </div>
      <div class="field">
        <label for="chapterSelect"><span class="step">Step 3</span> · Chapter</label>
        <select id="chapterSelect" disabled>
          <option value="" selected>Select a definition first…</option>
        </select>
      </div>
      <div class="preview" id="preview" hidden>
        <span class="preview-period" id="previewPeriod"></span>
        <span class="preview-clock" id="previewClock"></span>
        <span class="preview-coverage" id="previewCoverage"></span>
      </div>
      <a class="open-btn" id="openBtn" href="#" aria-disabled="true">Open chapter &rarr;</a>
      <p class="picker-note">Each match is split into ~5-minute chapters aligned to period breaks — a full
        match at full quality runs past what a single page can hold. Every chapter links to the next and
        previous one so you can keep watching straight through. <b>Ball tracked</b> is the share of that
        chapter's frames with a real tracked ball position; broadcast tracking has real gaps, so it varies
        chapter to chapter. <b>Phase-detection definition</b> lets you pick which version of the classification
        logic is driving the High Press / Mid Block / Low Block / Transition calls — useful for comparing an
        earlier definition against a later one once more than one exists.</p>
    </div>
  </div>

  <div>
    <p class="section-label">Companion analyses</p>
    <p class="picker-note" style="margin-top:-8px">Scoped to <b>Morocco vs Spain</b> only — not yet rebuilt for the other matches.</p>
    <div class="companions">
      __COMPANION_CARDS__
    </div>
  </div>

  <footer class="credit">PFF FC tracking data &middot; ~30Hz broadcast tracking, ball ~15Hz effective</footer>
</div>
<script>
var MATCHES = __MATCHES_JSON__;

var matchSelect = document.getElementById('matchSelect');
var versionSelect = document.getElementById('versionSelect');
var versionNote = document.getElementById('versionNote');
var chapterSelect = document.getElementById('chapterSelect');
var preview = document.getElementById('preview');
var previewPeriod = document.getElementById('previewPeriod');
var previewClock = document.getElementById('previewClock');
var previewCoverage = document.getElementById('previewCoverage');
var openBtn = document.getElementById('openBtn');

MATCHES.forEach(function (m) {
  var opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = m.label + ' — ' + m.competition;
  matchSelect.appendChild(opt);
});

function findMatch(id) {
  for (var i = 0; i < MATCHES.length; i++) if (MATCHES[i].id === id) return MATCHES[i];
  return null;
}

function findVersion(match, id) {
  if (!match) return null;
  for (var i = 0; i < match.versions.length; i++) if (match.versions[i].id === id) return match.versions[i];
  return null;
}

function resetChapterPicker(message) {
  chapterSelect.innerHTML = '<option value="" selected>' + message + '</option>';
  chapterSelect.disabled = true;
  preview.hidden = true;
  openBtn.setAttribute('aria-disabled', 'true');
  openBtn.href = '#';
}

function resetVersionPicker() {
  versionSelect.innerHTML = '<option value="" selected>Select a match first…</option>';
  versionSelect.disabled = true;
  versionNote.hidden = true;
  resetChapterPicker('Select a definition first…');
}

function populateVersions(match) {
  versionSelect.innerHTML = '';
  var placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.selected = true;
  placeholder.disabled = true;
  placeholder.textContent = 'Select a definition…';
  versionSelect.appendChild(placeholder);

  match.versions.forEach(function (v) {
    var opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.label;
    versionSelect.appendChild(opt);
  });
  versionSelect.disabled = false;
}

function populateChapters(version) {
  chapterSelect.innerHTML = '';
  var placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.selected = true;
  placeholder.disabled = true;
  placeholder.textContent = 'Select a chapter…';
  chapterSelect.appendChild(placeholder);

  var currentGroup = null, groupEl = null;
  version.chapters.forEach(function (ch) {
    if (ch.period !== currentGroup) {
      currentGroup = ch.period;
      groupEl = document.createElement('optgroup');
      groupEl.label = currentGroup;
      chapterSelect.appendChild(groupEl);
    }
    var opt = document.createElement('option');
    opt.value = String(ch.idx);
    var num = ch.idx < 10 ? '0' + ch.idx : String(ch.idx);
    opt.textContent = num + '  ·  ' + ch.clock + '  ·  ball tracked ' + ch.coverage + '%';
    groupEl.appendChild(opt);
  });
  chapterSelect.disabled = false;
}

function showPreview(version, ch) {
  previewPeriod.textContent = ch.period;
  previewClock.textContent = ch.clock;
  previewCoverage.textContent = 'Ball tracked ' + ch.coverage + '%';
  previewCoverage.className = 'preview-coverage' + (ch.coverage < 70 ? ' low' : '');
  preview.hidden = false;
  openBtn.href = ch.url;
  openBtn.removeAttribute('aria-disabled');
}

function selectFirstChapter(version) {
  if (!version.chapters.length) return;
  chapterSelect.value = String(version.chapters[0].idx);
  showPreview(version, version.chapters[0]);
}

matchSelect.addEventListener('change', function () {
  var match = findMatch(matchSelect.value);
  if (!match) { resetVersionPicker(); return; }
  populateVersions(match);
  // Default to the first (baseline) definition so the chapter dropdown is
  // usable immediately, while still leaving the version picker in place for
  // whenever a second definition is added to compare against.
  if (match.versions.length) {
    versionSelect.value = match.versions[0].id;
    versionSelect.dispatchEvent(new Event('change'));
  }
});

versionSelect.addEventListener('change', function () {
  var match = findMatch(matchSelect.value);
  var version = findVersion(match, versionSelect.value);
  if (!version) { resetChapterPicker('Select a definition first…'); versionNote.hidden = true; return; }
  if (version.note) {
    versionNote.textContent = version.note;
    versionNote.hidden = false;
  } else {
    versionNote.hidden = true;
  }
  populateChapters(version);
  preview.hidden = true;
  openBtn.setAttribute('aria-disabled', 'true');
  openBtn.href = '#';
  selectFirstChapter(version);
});

chapterSelect.addEventListener('change', function () {
  var match = findMatch(matchSelect.value);
  var version = findVersion(match, versionSelect.value);
  if (!version || !chapterSelect.value) return;
  var idx = parseInt(chapterSelect.value, 10);
  var ch = version.chapters.filter(function (c) { return c.idx === idx; })[0];
  if (ch) showPreview(version, ch);
});

// Single match today: pick it automatically so the rest of the picker is
// usable immediately, while still leaving the UI in place for when a
// second match is added.
if (MATCHES.length === 1) {
  matchSelect.value = MATCHES[0].id;
  matchSelect.dispatchEvent(new Event('change'));
}
</script>
"""

card_html = []
for c in COMPANIONS:
    card_html.append(
        '<a class="companion-card" href="{url}">'
        '<span class="companion-icon">{icon}</span>'
        '<span class="companion-title">{title}</span>'
        '<span class="companion-blurb">{blurb}</span>'
        '<span class="companion-go">Open &rarr;</span>'
        '</a>'.format(url=c["url"], icon=c["icon"], title=c["title"], blurb=c["blurb"])
    )

matchup_parts = [
    f'<b>{m["home"]}</b> vs <b>{m["away"]}</b> &middot; {m["competition"]}'
    for m in MATCHES
]
matchup_line = ' &nbsp;|&nbsp; '.join(matchup_parts)

HTML = HTML.replace("__MATCHES_JSON__", MATCHES_JSON)
HTML = HTML.replace("__COMPANION_CARDS__", "\n      ".join(card_html))
HTML = HTML.replace("__MATCHUP_LINE__", matchup_line)

with open("hub.html", "w") as f:
    f.write(HTML)
print("wrote hub.html,", len(HTML), "bytes")

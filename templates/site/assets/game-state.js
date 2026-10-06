/* Game-state study page (game-state.html): charts drawn from window.SITE.gs (analysis/game_state.py site). */
(function () {
  'use strict';
  var D = (window.SITE || {}).gs;
  if (!D) return;
  var NS = 'http://www.w3.org/2000/svg';
  var COL = { leading: '#3d8ae6', trailing: '#d9733a', level: '#7d8f86' };
  var INK = '#eef3ee', DIM = '#9bb0a4', FAINT = '#6a8075', GRID = '#1f2f28', LINE2 = '#2c4038', BG = '#0a110e';
  var $ = function (id) { return document.getElementById(id); };
  var tip = document.createElement('div'); tip.className = 'gs-tip'; document.body.appendChild(tip);
  var esc = function (s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
  var el = function (tag, attrs, parent) { var n = document.createElementNS(NS, tag); for (var k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; };
  var txt = function (parent, x, y, s, o) { o = o || {}; var t = el('text', { x: x, y: y, fill: o.fill || FAINT, 'font-size': o.size || 11, 'font-family': 'IBM Plex Mono, monospace', 'text-anchor': o.anchor || 'middle' }, parent); t.textContent = s; return t; };
  var showTip = function (ev, html) { tip.innerHTML = html; tip.style.display = 'block'; var x = ev.clientX + 14; if (x + 270 > innerWidth) x = ev.clientX - 270; tip.style.left = x + 'px'; tip.style.top = (ev.clientY + 14) + 'px'; };
  var hideTip = function () { tip.style.display = 'none'; };
  var hover = function (node, html) { node.addEventListener('mousemove', function (ev) { showTip(ev, typeof html === 'function' ? html(ev) : html); }); node.addEventListener('mouseleave', hideTip); };
  var E = function (c, k) { return D.effects.filter(function (e) { return e.compare === c && e.key === k; })[0]; };
  var fmt = function (v, d) { return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d == null ? 1 : d); };
  var UNIT = { line: ' m', length: ' m', width: ' m', holes: ' m²', free: '', high: ' pts', low: ' pts', press_rate: '/min', possession: ' pts' };
  var DEC = { line: 1, length: 1, width: 1, holes: 0, free: 2, high: 1, low: 1, press_rate: 1, possession: 1 };
  var NICE = { line: 'Back-line height', length: 'Block length', width: 'Block width', holes: 'Holes inside the block', free: 'Opponents free inside', high: 'Time in a high press', low: 'Time in a low block', press_rate: 'Presses', possession: 'Possession' };
  var SUB = { line: 'm from own goal', length: 'm, back to front', width: 'm, side to side', holes: 'm² of open space', free: 'players, on average', high: '% of organised defending', low: '% of organised defending', press_rate: 'per minute defending', possession: '% of live play' };

  /* numbers in the text */
  var c = D.counts, dm = c.def_minutes;
  var ll = E('leading', 'line'), lp = E('leading', 'possession'), tp = E('trailing', 'press_rate'), tl = E('trailing', 'length'), tw = E('trailing', 'width'), tf = E('trailing', 'free');
  var set = function (id, v) { var n = $(id); if (n) n.textContent = v; };
  set('gsSample', c.matches + ' matches · ' + c.teams + ' teams · ' + c.goals + ' goals · ' + (dm.leading + dm.level + dm.trailing) + ' minutes of live defending (' + dm.leading + ' leading, ' + dm.level + ' level, ' + dm.trailing + ' trailing)');
  set('gsLeadLine', Math.abs(ll.est).toFixed(1) + ' m'); set('gsLeadSame', ll.same + ' of ' + ll.n); set('gsLeadPoss', Math.abs(lp.est).toFixed(0));
  set('gsTrPress', fmt(tp.est)); set('gsTrLen', Math.abs(tl.est).toFixed(1) + ' m'); set('gsTrWid', Math.abs(tw.est).toFixed(1) + ' m');
  set('gsTrFree', Math.abs(tf.est).toFixed(1)); set('gsTrSame', tl.same + ' of ' + tl.n);
  set('gsRaw', Math.abs(D.robust['leading|line'].raw).toFixed(1) + ' m'); set('gsAdj', Math.abs(ll.est).toFixed(1) + ' m');
  set('gsLateLead', D.late_share.leading + '%'); set('gsLateLevel', D.late_share.level + '%');

  /* effects: one row per measure, each on its own scale centred on zero */
  var ORDER = ['line', 'length', 'width', 'holes', 'free', 'press_rate', 'high', 'low', 'possession'];
  ['leading', 'trailing'].forEach(function (cmp) {
    var box = $(cmp === 'leading' ? 'gsLead' : 'gsTrail'); if (!box) return;
    ORDER.forEach(function (k) {
      var e = E(cmp, k); if (!e) return;
      var clear = e.lo > 0 || e.hi < 0;
      var row = document.createElement('div'); row.className = 'gs-row' + (clear ? '' : ' gs-muted');
      row.innerHTML = '<div class="gs-lab">' + esc(NICE[k]) + '<small>' + esc(SUB[k]) + '</small></div><div class="gs-bar"></div>' +
        '<div class="gs-val">' + fmt(e.est, DEC[k]) + esc(UNIT[k]) + '<small>' + e.same + ' of ' + e.n + ' teams</small></div>';
      box.appendChild(row);
      var bar = row.querySelector('.gs-bar');
      var lim = Math.max(Math.abs(e.lo), Math.abs(e.hi), Math.abs(e.est)) * 1.15 || 1;
      var P = function (v) { return 50 + v / lim * 46; };
      bar.innerHTML = '<i class="gs-zero"></i><i class="gs-ci" style="left:' + P(e.lo) + '%;width:' + (P(e.hi) - P(e.lo)) + '%;background:' + COL[cmp] + ';opacity:' + (clear ? 1 : .55) + '"></i>' +
        '<i class="gs-pt" style="left:' + P(e.est) + '%;' + (clear ? 'background:' + COL[cmp] : 'background:' + BG + ';border:1.5px solid ' + COL[cmp]) + '"></i>';
      hover(bar, '<b>' + esc(NICE[k]) + ', ' + cmp + '</b><br>' + fmt(e.est, DEC[k]) + esc(UNIT[k]) + ' vs level (95%: ' + fmt(e.lo, DEC[k]) + ' to ' + fmt(e.hi, DEC[k]) + ')<br>' + e.same + ' of ' + e.n + ' teams point this way');
    });
  });

  /* team by team */
  var STRIPS = [['leading', 'line', 'Back line when leading', 'm vs level · left = deeper'], ['trailing', 'length', 'Block length when trailing', 'm vs level · right = more stretched'],
                ['trailing', 'width', 'Block width when trailing', 'm vs level · left = narrower'], ['trailing', 'free', 'Free opponents when trailing', 'players vs level · right = more']];
  var sbox = $('gsStrips');
  if (sbox) STRIPS.forEach(function (s) {
    var cmp = s[0], k = s[1], pts = D.per_team[cmp + '|' + k], e = E(cmp, k);
    var div = document.createElement('div'); div.className = 'gs-strip';
    div.innerHTML = '<h3>' + esc(s[2]) + '</h3><div class="gs-sub">' + esc(s[3]) + ' · ' + e.same + ' of ' + e.n + ' teams ' + (e.est < 0 ? 'below' : 'above') + ' zero</div>';
    var svg = el('svg', { viewBox: '0 0 480 84', role: 'img', 'aria-label': s[2] }); div.appendChild(svg); sbox.appendChild(div);
    var lim = Math.max.apply(null, pts.map(function (p) { return Math.abs(p.diff); })) * 1.1;
    var step = lim > 8 ? 5 : lim > 3 ? 1 : lim > 1 ? .5 : .1;
    var stride = Math.max(1, Math.round(lim / step / 3));
    lim = Math.ceil(lim / (step * stride)) * step * stride;
    var X = function (v) { return 240 + v / lim * 220; };
    for (var t = -lim; t <= lim + 1e-9; t += step * stride) {
      el('line', { x1: X(t), x2: X(t), y1: 6, y2: 60, stroke: Math.abs(t) < 1e-9 ? LINE2 : GRID }, svg);
      var tv = Math.abs(t) < 1e-9 ? 0 : t;
      txt(svg, X(t), 78, (tv > 0 ? '+' : tv < 0 ? '−' : '') + (step < 1 ? Math.abs(tv).toFixed(1) : Math.abs(Math.round(tv))));
    }
    pts.slice().sort(function (a, b) { return a.diff - b.diff; }).forEach(function (p, i) {
      var x = X(p.diff), y = 33 + ((i % 3) - 1) * 12;
      el('circle', { cx: x, cy: y, r: 5.5, fill: COL[cmp], stroke: BG, 'stroke-width': 2 }, svg);
      var hit = el('circle', { cx: x, cy: y, r: 11, fill: 'transparent' }, svg);
      hover(hit, '<b>' + esc(p.team) + '</b> · ' + esc(p.slug) + '<br>' + fmt(p.diff, DEC[k]) + esc(UNIT[k]) + '<br>' + p.mc + ' min ' + cmp + ' · ' + p.ml + ' min level');
    });
  });

  /* when each state happens: stacked bars */
  (function () {
    var svg = $('gsTiming'); if (!svg) return;
    var T = D.timing, W = 480, H = 250, L = 36, R = 8, Tp = 10, Bt = 30;
    var max = Math.max.apply(null, T.map(function (b) { return b.leading + b.level + b.trailing; })), top = Math.ceil(max / 25) * 25;
    var Y = function (v) { return Tp + (H - Tp - Bt) * (1 - v / top); };
    for (var g = 0; g <= top; g += 25) { el('line', { x1: L, x2: W - R, y1: Y(g), y2: Y(g), stroke: GRID }, svg); txt(svg, L - 6, Y(g) + 4, g, { anchor: 'end' }); }
    var bw = (W - L - R) / T.length;
    T.forEach(function (b, i) {
      var x = L + i * bw + 7, w = bw - 14, acc = 0;
      ['leading', 'level', 'trailing'].forEach(function (s) {
        var v = b[s]; if (!v) return;
        var r = el('rect', { x: x, y: Y(acc + v), width: w, height: Math.max(0, Y(acc) - Y(acc + v) - 2), fill: COL[s], rx: 2 }, svg);
        hover(r, '<b>' + b.label + '</b><br>' + s + ': ' + v.toFixed(0) + ' min of live defending');
        acc += v;
      });
      txt(svg, x + w / 2, H - 10, b.label, { fill: DIM });
    });
  })();

  /* back-line height by band */
  (function () {
    var svg = $('gsLine'); if (!svg) return;
    var T = D.timing, W = 480, H = 250, L = 36, R = 14, Tp = 18, Bt = 30;
    var v = T.map(function (b) { return b.line; }), lo = Math.floor(Math.min.apply(null, v) - 1), hi = Math.ceil(Math.max.apply(null, v) + 1);
    var Y = function (x) { return Tp + (H - Tp - Bt) * (1 - (x - lo) / (hi - lo)); }, bw = (W - L - R) / T.length, X = function (i) { return L + bw * (i + .5); };
    for (var g = lo; g <= hi; g += 2) { el('line', { x1: L, x2: W - R, y1: Y(g), y2: Y(g), stroke: GRID }, svg); txt(svg, L - 6, Y(g) + 4, g, { anchor: 'end' }); }
    el('path', { d: T.map(function (b, i) { return (i ? 'L' : 'M') + X(i) + ' ' + Y(b.line); }).join(' '), fill: 'none', stroke: INK, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
    T.forEach(function (b, i) {
      var end = i === 0 || i === T.length - 1;
      el('circle', { cx: X(i), cy: Y(b.line), r: end ? 5 : 3.5, fill: INK, stroke: BG, 'stroke-width': 2 }, svg);
      hover(el('circle', { cx: X(i), cy: Y(b.line), r: 14, fill: 'transparent' }, svg), '<b>' + b.label + '</b><br>back line ' + b.line.toFixed(1) + ' m from own goal');
      txt(svg, X(i), H - 10, b.label, { fill: DIM });
      if (end) txt(svg, X(i) + (i ? -10 : 8), Y(b.line) + (i ? 18 : -10), b.line.toFixed(1) + ' m', { fill: INK, size: 12, anchor: i ? 'end' : 'start' });
    });
  })();

  /* Nürnberg example */
  (function () {
    var svg = $('gsEx'); if (!svg) return;
    var ex = D.example, P = ex.points;
    set('gsExFirst', ex.first15.toFixed(0) + ' m'); set('gsExLast', ex.last15.toFixed(0) + ' m');
    var W = 960, H = 240, L = 40, R = 14, Tp = 18, Bt = 26;
    var v = P.map(function (p) { return p[1]; }), lo = Math.floor(Math.min.apply(null, v) / 5) * 5, hi = Math.ceil(Math.max.apply(null, v) / 5) * 5;
    var xmax = Math.max(95, P[P.length - 1][0]);
    var X = function (m) { return L + (W - L - R) * m / xmax; }, Y = function (y) { return Tp + (H - Tp - Bt) * (1 - (y - lo) / (hi - lo)); };
    for (var g = lo; g <= hi; g += 5) { el('line', { x1: L, x2: W - R, y1: Y(g), y2: Y(g), stroke: GRID }, svg); txt(svg, L - 6, Y(g) + 4, g, { anchor: 'end' }); }
    for (var m = 0; m <= 90; m += 15) txt(svg, X(m), H - 6, m + "'");
    el('rect', { x: X(ex.goal), y: Tp, width: X(xmax) - X(ex.goal), height: H - Tp - Bt, fill: 'rgba(61,138,230,0.08)' }, svg);
    el('line', { x1: X(ex.goal), x2: X(ex.goal), y1: Tp, y2: H - Bt, stroke: COL.leading, 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }, svg);
    txt(svg, X(ex.goal) + 8, Tp + 14, ex.goal_label + ' ' + ex.team + ' score · leading from here', { fill: INK, size: 12, anchor: 'start' });
    var seg = [], cur = [];
    P.forEach(function (p, i) { if (i && p[0] < P[i - 1][0]) { seg.push(cur); cur = []; } cur.push(p); });
    seg.push(cur);
    seg.forEach(function (s) { el('path', { d: s.map(function (p, i) { return (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1); }).join(' '), fill: 'none', stroke: INK, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg); });
    var cross = el('line', { x1: 0, x2: 0, y1: Tp, y2: H - Bt, stroke: LINE2, opacity: 0 }, svg);
    var area = el('rect', { x: L, y: Tp, width: W - L - R, height: H - Tp - Bt, fill: 'transparent' }, svg);
    area.addEventListener('mousemove', function (ev) {
      var r = svg.getBoundingClientRect(), mm = ((ev.clientX - r.left) / r.width * W - L) / (W - L - R) * xmax;
      var best = P.reduce(function (a, p) { return Math.abs(p[0] - mm) < Math.abs(a[0] - mm) ? p : a; }, P[0]);
      cross.setAttribute('x1', X(best[0])); cross.setAttribute('x2', X(best[0])); cross.setAttribute('opacity', 1);
      showTip(ev, '<b>' + Math.floor(best[0]) + "'</b><br>back line " + best[1].toFixed(1) + ' m from goal');
    });
    area.addEventListener('mouseleave', function () { cross.setAttribute('opacity', 0); hideTip(); });
  })();

  /* robustness table */
  var RB = [['leading', 'line'], ['leading', 'possession'], ['leading', 'free'], ['trailing', 'press_rate'], ['trailing', 'length'], ['trailing', 'width'], ['trailing', 'free'], ['trailing', 'possession'], ['trailing', 'holes']];
  var body = $('gsRobust');
  if (body) body.innerHTML = RB.map(function (r) {
    var cmp = r[0], k = r[1], e = E(cmp, k), b = D.robust[cmp + '|' + k], dk = DEC[k], u = UNIT[k];
    var cell = function (x) { return x ? fmt(x.est, dk) + u + '<br><small>' + fmt(x.lo, dk) + ' to ' + fmt(x.hi, dk) + ' · ' + x.n + ' teams</small>' : '—'; };
    return '<tr><td><i class="gs-dot" style="background:' + COL[cmp] + '"></i>' + esc(NICE[k]) + ' <span class="gs-faint">· ' + cmp + '</span></td><td>' + cell(e) + '</td><td>' + cell(b.dfl) + '</td><td>' + cell(b.pff) +
      '</td><td>' + fmt(b.loo[0], dk) + ' to ' + fmt(b.loo[1], dk) + u + '</td><td>' + fmt(b.raw, dk) + u + '</td></tr>';
  }).join('');
})();

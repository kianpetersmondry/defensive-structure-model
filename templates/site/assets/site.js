/* Defensive Structure Model: shared page behaviour. Each block only runs on pages that have its elements.
   Page data arrives as window.SITE = {matches, kinds, reports, board, clip}. */
(function () {
  'use strict';
  var SITE = window.SITE || {};

  function esc(s) { var d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }
  function qs(name) { return new URLSearchParams(window.location.search).get(name); }

  /* ---------- mobile menu ---------- */
  var menuBtn = document.querySelector('.menu-btn');
  if (menuBtn) {
    menuBtn.addEventListener('click', function () {
      var links = document.querySelector('.nav-links');
      var open = links.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', String(open));
    });
  }

  /* ---------- hero: a short clip of real tracking ---------- */
  var heroCanvas = document.getElementById('heroPitch');
  if (heroCanvas && SITE.clip) {
    var PHASE = { 'High Press': '#e8323a', 'Mid Block': '#d99a2b', 'Low Block': '#3f8f5c', 'Defensive Transition': '#b5563a', 'In Possession': '#3c5148' };
    var LABEL = { 'High Press': 'High press', 'Mid Block': 'Mid block', 'Low Block': 'Low block', 'Defensive Transition': 'Transition', 'In Possession': 'On the ball' };
    var M = SITE.clip.meta, FR = SITE.clip.frames;
    var ctx = heroCanvas.getContext('2d'), W = heroCanvas.width, H = heroCanvas.height, S = W / 111, ox = 3 * S, oy = (H - 68 * S) / 2;
    var X = function (x) { return ox + (x + 52.5) * S; }, Y = function (y) { return oy + (34 - y) * S; };
    var isGk = function (id) { var p = M.players[id]; return p && p.pos === 'GK'; };
    var hexRgb = function (h) { h = h.replace('#', ''); return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16); }).join(','); };
    var D = M.defending || 'home', DP = D === 'home' ? 'H' : 'A';
    var DEF_RGB = hexRgb(M[D + 'Color']);
    var st = function (f, k) { return f[k] !== undefined ? f[k] : f[D + k.charAt(0).toUpperCase() + k.slice(1)]; };
    var ORGANISED = { 'High Press': 1, 'Mid Block': 1, 'Low Block': 1 };
    var hull = function (pts) {
      var p = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
      var cross = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
      var lo = [], up = [], i;
      for (i = 0; i < p.length; i++) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], p[i]) <= 0) lo.pop(); lo.push(p[i]); }
      for (i = p.length - 1; i >= 0; i--) { while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], p[i]) <= 0) up.pop(); up.push(p[i]); }
      up.pop(); lo.pop(); return lo.concat(up);
    };
    var draw = function (f) {
      ctx.fillStyle = '#163a28'; ctx.fillRect(0, 0, W, H);
      for (var i = 0; i < 14; i += 2) { ctx.fillStyle = '#183f2c'; ctx.fillRect(X(-52.5 + (i + 1) * 7.5), Y(34), 7.5 * S, 68 * S); }
      ctx.strokeStyle = 'rgba(235,245,238,0.55)'; ctx.lineWidth = 1.6;
      ctx.strokeRect(X(-52.5), Y(34), 105 * S, 68 * S);
      ctx.beginPath(); ctx.moveTo(X(0), Y(34)); ctx.lineTo(X(0), Y(-34)); ctx.stroke();
      ctx.beginPath(); ctx.arc(X(0), Y(0), 9.15 * S, 0, 7); ctx.stroke();
      ctx.strokeRect(X(-52.5), Y(20.16), 16.5 * S, 40.32 * S); ctx.strokeRect(X(36), Y(20.16), 16.5 * S, 40.32 * S);
      ctx.strokeRect(X(-52.5), Y(9.16), 5.5 * S, 18.32 * S); ctx.strokeRect(X(47), Y(9.16), 5.5 * S, 18.32 * S);
      var defending = f.possessionTeam && f.possessionTeam !== D && ORGANISED[st(f, 'structure')];
      var homeOut = f.players.filter(function (p) { return p[0][0] === DP && !isGk(p[0]); });
      if (defending && homeOut.length) {
        // the inter-line band: the defending team's deepest outfielder to its most advanced one
        var xs = homeOut.map(function (p) { return p[1]; });
        var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
        ctx.fillStyle = 'rgba(' + DEF_RGB + ',0.16)'; ctx.fillRect(X(x0), Y(34), (x1 - x0) * S, 68 * S);
        ctx.strokeStyle = 'rgba(' + DEF_RGB + ',0.75)'; ctx.lineWidth = 2;
        [x0, x1].forEach(function (x) { ctx.beginPath(); ctx.moveTo(X(x), Y(34)); ctx.lineTo(X(x), Y(-34)); ctx.stroke(); });
        // the marking zone: the five defenders nearest the ball, solid when tight, hatched when loose
        var mk = st(f, 'marking');
        if (f.ball && (mk === 'Tight' || mk === 'Loose')) {
          var near = homeOut.map(function (p) { return { p: p, d: Math.hypot(p[1] - f.ball[0], p[2] - f.ball[1]) }; })
            .sort(function (a, b) { return a.d - b.d; }).slice(0, 5);
          var hp = hull(near.map(function (n) { return [X(n.p[1]), Y(n.p[2])]; }));
          if (hp.length >= 3) {
            ctx.save(); ctx.beginPath();
            hp.forEach(function (pt, k) { if (k) ctx.lineTo(pt[0], pt[1]); else ctx.moveTo(pt[0], pt[1]); });
            ctx.closePath();
            var tight = mk === 'Tight';
            if (tight) { ctx.fillStyle = 'rgba(91,141,238,0.18)'; ctx.fill(); }
            else {
              ctx.save(); ctx.clip();
              var hx = hp.map(function (pt) { return pt[0]; }), hy = hp.map(function (pt) { return pt[1]; });
              var mnX = Math.min.apply(null, hx) - 16, mxX = Math.max.apply(null, hx) + 16, mnY = Math.min.apply(null, hy) - 16, mxY = Math.max.apply(null, hy) + 16, sp = mxY - mnY;
              ctx.strokeStyle = 'rgba(147,168,156,0.55)'; ctx.lineWidth = 1;
              for (var q = mnX - sp; q < mxX + sp; q += 7) { ctx.beginPath(); ctx.moveTo(q, mnY); ctx.lineTo(q + sp, mxY); ctx.stroke(); }
              ctx.restore();
            }
            ctx.lineWidth = tight ? 1.75 : 1.25;
            ctx.strokeStyle = tight ? 'rgba(91,141,238,0.85)' : 'rgba(147,168,156,0.6)';
            if (!tight) ctx.setLineDash([4, 3]);
            ctx.stroke(); ctx.setLineDash([]); ctx.restore();
          }
        }
      }
      if (f.ball && f.pressureScore != null && f.primaryPresser) {
        // pressure ring on the carrier: solid and glowing when engaged, dashed and dimmer when passive
        var passive = st(f, 'engaged') === 'Passive', ps = f.pressureScore, bx = X(f.ball[0]), by = Y(f.ball[1]), r = (14 + ps * 22) * S / 10;
        var grad = ctx.createRadialGradient(bx, by, 2, bx, by, r);
        grad.addColorStop(0, 'rgba(255,176,32,' + (0.03 + ps * (passive ? 0.10 : 0.22)) + ')'); grad.addColorStop(1, 'rgba(255,176,32,0)');
        ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(bx, by, r, 0, 7); ctx.fill();
        if (passive) ctx.setLineDash([5, 4]);
        ctx.strokeStyle = 'rgba(255,176,32,' + (passive ? 0.16 + ps * 0.28 : 0.35 + ps * 0.55) + ')';
        ctx.lineWidth = (passive ? 1 : 1.5) + ps * (passive ? 1.5 : 2.5);
        ctx.beginPath(); ctx.arc(bx, by, r, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      }
      f.players.forEach(function (p) {
        var px = X(p[1]), py = Y(p[2]);
        if (p[0] === f.primaryPresser) { ctx.beginPath(); ctx.arc(px, py, 1.25 * S + 4, 0, 7); ctx.strokeStyle = 'rgba(255,176,32,0.9)'; ctx.lineWidth = 1.5; ctx.stroke(); }
        ctx.beginPath(); ctx.arc(px, py, 1.25 * S, 0, 7);
        ctx.fillStyle = isGk(p[0]) ? '#e9eef0' : (p[0][0] === 'H' ? M.homeColor : M.awayColor); ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(8,14,11,0.9)'; ctx.stroke();
      });
      if (f.ball) { ctx.beginPath(); ctx.arc(X(f.ball[0]), Y(f.ball[1]), 0.62 * S, 0, 7); ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 1.2; ctx.strokeStyle = '#0b1210'; ctx.stroke(); }
    };
    var phaseEl = document.getElementById('heroPhase'), tagEl = document.getElementById('heroTag'), clockEl = document.getElementById('heroClock');
    var show = function (i) {
      var f = FR[i];
      draw(f);
      var ph = st(f, 'structure');
      phaseEl.textContent = LABEL[ph] || '—'; phaseEl.style.background = PHASE[ph] || '#3c5148';
      tagEl.textContent = st(f, 'engaged') || 'Passive';
      var s = Math.floor(f.clockS); clockEl.textContent = Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2);
    };
    var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var t = 0, last = null;
    var loop = function (now) {
      if (last != null) t = (t + Math.max(0, now - last) / 1000 * 10) % FR.length;
      last = now;
      show(Math.min(FR.length - 1, Math.floor(t)));
      requestAnimationFrame(loop);
    };
    show(0);
    if (!reduce) requestAnimationFrame(loop);
  }

  /* ---------- analysis viewer ---------- */
  var viewer = document.getElementById('viewer');
  if (viewer && SITE.kinds) {
    var K = SITE.kinds, slugs = K[0].data.map(function (d) { return d.slug; });
    var mi = Math.max(0, slugs.indexOf(qs('m'))), ki = 0;
    var want = qs('v') || decodeURIComponent((window.location.hash || '').slice(1));
    K.forEach(function (k, i) { if (k.name === want || k.key === want) ki = i; });
    var mt = document.getElementById('viewerMatches'), kt = document.getElementById('viewerKinds');
    var render = function () {
      mt.innerHTML = K[0].data.map(function (d, i) {
        var m = SITE.matches[i];
        return '<button type="button" class="tab" role="tab" aria-selected="' + (i === mi) + '" data-i="' + i + '"><span class="dot" style="background:' + m.hc + '"></span>' +
          esc(m.hcode) + ' <span class="sc">' + esc(m.hs + '–' + m.as) + '</span> ' + esc(m.acode) + '<span class="dot" style="background:' + m.ac + '"></span></button>';
      }).join('');
      kt.innerHTML = K.map(function (k, i) { return '<button type="button" class="subtab" role="tab" aria-selected="' + (i === ki) + '" data-k="' + i + '">' + esc(k.name) + '</button>'; }).join('');
      var e = K[ki].data[mi];
      document.getElementById('viewerDesc').textContent = K[ki].desc;
      var img = document.getElementById('viewerImg');
      img.src = e.src; img.alt = e.alt; img.width = e.w; img.height = e.h;
      document.getElementById('viewerCap').textContent = e.label + ' · ' + e.cap;
      document.getElementById('viewerFull').href = e.src;
      Array.prototype.forEach.call(mt.querySelectorAll('.tab'), function (b) { b.onclick = function () { mi = +b.dataset.i; render(); sync(); }; });
      Array.prototype.forEach.call(kt.querySelectorAll('.subtab'), function (b) { b.onclick = function () { ki = +b.dataset.k; render(); sync(); }; });
    };
    var sync = function () { history.replaceState(null, '', '?m=' + slugs[mi] + '&v=' + encodeURIComponent(K[ki].key)); };
    render();
  }

  /* ---------- scouting reports ---------- */
  var rep = document.getElementById('reportView');
  if (rep && SITE.reports) {
    var RP = SITE.reports, ri = 0;
    RP.forEach(function (r, i) { if (r.slug + '-' + r.team.toLowerCase().replace(/\s+/g, '-') === qs('t')) ri = i; });
    var tabs = document.getElementById('reportTabs');
    var kindKey = {};
    (SITE.kinds || []).forEach(function (k) { kindKey[k.name] = k.key; });
    var renderReport = function () {
      tabs.innerHTML = RP.map(function (r, i) {
        return '<button type="button" class="tab" role="tab" aria-selected="' + (i === ri) + '" data-i="' + i + '"><span class="dot" style="background:' + r.color + '"></span>' + esc(r.team) + ' <span class="sc">v ' + esc(r.opp) + '</span></button>';
      }).join('');
      var r = RP[ri];
      var chips = r.chips.map(function (c) {
        var pos = (c.rank - 1) / 11 * 100;
        return '<div class="chip"><span class="k">' + esc(c.label.charAt(0).toUpperCase() + c.label.slice(1)) + '</span><span class="v">' + esc(c.value) + '</span>' +
          '<span class="r"><span class="rankbar"><i style="left:' + pos + '%"></i></span>' + esc(c.rank_text) + '</span></div>';
      }).join('');
      var li = function (xs) { return xs.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join(''); };
      var views = r.views.map(function (v) { return '<a class="btn small" href="analysis.html?m=' + r.slug + '&v=' + encodeURIComponent(kindKey[v] || v) + '">' + esc(v) + ' →</a>'; }).join('');
      rep.innerHTML = '<div class="left"><div class="who"><span class="dot lg" style="background:' + r.color + '"></span>' + esc(r.team) + ' · vs ' + esc(r.opp) + ' · ' + esc(r.sub) + '</div>' +
        '<blockquote>' + esc(r.headline) + '</blockquote>' +
        '<h3>What it did well</h3><ul>' + li(r.good) + '</ul><h3>How to get at it</h3><ul>' + li(r.exploit) + '</ul>' +
        '<div class="views-links">' + views + '</div></div><div class="right"><div class="eyebrow" style="margin-bottom:6px">Ranked against all 12 teams</div>' + chips + '</div>';
      Array.prototype.forEach.call(tabs.querySelectorAll('.tab'), function (b) { b.onclick = function () { ri = +b.dataset.i; renderReport(); }; });
    };
    renderReport();
  }

  /* ---------- comparison table ---------- */
  var cmp = document.getElementById('cmpBody');
  if (cmp && SITE.board) {
    var B = SITE.board;
    var COLS = ['possession', 'back_line', 'length', 'width', 'holes', 'free', 'wins', 'wins_high', 'fast', 'shots_for', 'shots_against'];
    var FMT = { possession: function (v) { return v + '%'; }, wins_high: function (v) { return v + '%'; }, free: function (v) { return v.toFixed(1); },
      back_line: function (v) { return v.toFixed(0) + ' m'; }, length: function (v) { return v.toFixed(0) + ' m'; }, width: function (v) { return v.toFixed(0) + ' m'; },
      holes: function (v) { return v + ' m²'; } };
    var rng = {};
    COLS.forEach(function (c) { var v = B.map(function (r) { return r[c]; }); rng[c] = [Math.min.apply(null, v), Math.max.apply(null, v)]; });
    var order = B.map(function (_, i) { return i; }), sortCol = null, dir = -1;
    var draw = function () {
      cmp.innerHTML = order.map(function (i) {
        var r = B[i];
        return '<tr><td><span class="cmp-team"><span class="dot" style="background:' + r.color + '"></span><b>' + esc(r.team) + '</b><small>v ' + esc(r.opp) + '</small></span></td>' +
          COLS.map(function (c) {
            var v = r[c], lo = rng[c][0], hi = rng[c][1], w = hi > lo ? (v - lo) / (hi - lo) * 100 : 0;
            return '<td><span class="cell">' + (FMT[c] ? FMT[c](v) : v) + '<i><b style="width:' + w + '%"></b></i></span></td>';
          }).join('') + '</tr>';
      }).join('');
    };
    Array.prototype.forEach.call(document.querySelectorAll('table.cmp thead button'), function (b) {
      b.addEventListener('click', function () {
        var c = b.dataset.col;
        dir = sortCol === c ? -dir : -1; sortCol = c;
        order.sort(function (a, z) { return (B[a][c] - B[z][c]) * dir; });
        Array.prototype.forEach.call(document.querySelectorAll('table.cmp thead button'), function (x) { x.classList.toggle('on', x === b); });
        draw();
      });
    });
    draw();
  }
})();

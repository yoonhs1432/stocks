// 퀀트 대시보드 — 비교 · 분석 · 포트폴리오 · 설정.
// 안드로이드 앱(android-toss)을 옮긴 것. 계산은 전부 서버에서 끝내고 여기서는 그리기만 한다.

const PALETTE = ['#E0A24A', '#D9694E', '#CF5D7F', '#8A6FD0', '#4D8DF0', '#37A48C'];
const UP = '#EF6066', DOWN = '#5B9BF2', GOLD = '#E0A24A', TEAL = '#37B6C4', VIOLET = '#9B8CFF';

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

const S = {
  tab: localStorage.getItem('tab') || 'compare',
  market: localStorage.getItem('market') || 'US',
  usdMode: localStorage.getItem('cur') === 'usd',
  sortKey: 'day', sortDesc: true,
  ticker: localStorage.getItem('ticker') || null,
  bar: localStorage.getItem('bar') || '1d',
  account: null, rows: null, analysis: null, minutes: null, settings: null,
};

let tickTimer = null;
const live = {};          // symbol → 실시간 현재가

// ── 공통 포맷 ──
const num = (v, d = 2) => (v == null || Number.isNaN(v)) ? '–'
  : v.toLocaleString('ko-KR', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (r, d = 2) => (r == null || Number.isNaN(r)) ? '–'
  : (r >= 0 ? '+' : '') + r.toFixed(d) + '%';
const cls = v => (v == null || Number.isNaN(v)) ? 'muted' : v > 0 ? 'up' : v < 0 ? 'down' : 'muted';

/** 종목 가격 표기 — 국내는 원화 정수, 미장은 달러 소수 2자리. */
const price = (krw, v) => v == null ? '–'
  : krw ? '₩' + Math.round(v).toLocaleString('ko-KR') : '$' + num(v, 2);

/** 백분위(0~100) → 색. 저=매수 빨강 / 고=매도 파랑. */
function pctColor(p) {
  if (p == null) return 'var(--muted)';
  if (p < 20) return UP;
  if (p < 40) return '#EF8A8E';
  if (p < 60) return 'var(--text2)';
  if (p < 80) return '#8FB8F5';
  return DOWN;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const o = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(o.error || `요청 실패 (${res.status})`);
  return o;
}

function fail(e) {
  $('#body').innerHTML = '';
  $('#body').appendChild(el('p', 'err pad', '⚠️ ' + e.message));
}

// ══════════════════════════ 비교 ══════════════════════════

/** 표 왼쪽 미니 캔들 — 당일 시/고/저/종을 7×20 으로. */
function miniCandle(r) {
  const c = document.createElement('canvas');
  const w = 7, h = 20, dpr = window.devicePixelRatio || 1;
  c.width = w * dpr; c.height = h * dpr;
  c.style.width = w + 'px'; c.style.height = h + 'px';
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  const { open: o, high, low, price: cl } = r;
  if (o == null || high == null || low == null || high <= low) return c;
  const y = v => h - (v - low) / (high - low) * h;
  const up = cl >= o;
  g.strokeStyle = g.fillStyle = up ? UP : DOWN;
  g.beginPath(); g.moveTo(w / 2, y(high)); g.lineTo(w / 2, y(low)); g.stroke();
  const top = y(Math.max(o, cl)), bot = y(Math.min(o, cl));
  g.fillRect(1, top, w - 2, Math.max(bot - top, 1));
  return c;
}

function sortRows(rows) {
  const k = S.sortKey;
  const val = r => k === 'name' ? (r.name || r.ticker)
    : k === 'price' ? shownPrice(r) : k === 'day' ? shownDay(r) : r[k];
  return [...rows].sort((a, b) => {
    const x = val(a), y = val(b);
    if (typeof x === 'string') return S.sortDesc ? y.localeCompare(x) : x.localeCompare(y);
    const xn = x == null ? -Infinity : x, yn = y == null ? -Infinity : y;
    return S.sortDesc ? yn - xn : xn - yn;
  });
}

// 표시값과 정렬값이 어긋나지 않게 **같은 함수**를 쓴다 (안드로이드에서 겪은 버그)
const shownPrice = r => live[r.ticker] ?? r.price;
const shownDay = r => {
  const p = live[r.ticker];
  return (p != null && r.prevClose) ? (p / r.prevClose - 1) * 100 : r.day;
};

function renderCompare() {
  const body = $('#body');
  body.innerHTML = '';
  if (!S.rows) { body.appendChild(el('p', 'muted pad', '불러오는 중… (첫 조회는 20~30초)')); return; }

  const wrap = el('div', 'pad');
  wrap.style.padding = '0 var(--pad) 8px';
  const t = el('table', 'cmp');
  const head = el('tr');
  [['name', '종목', 'l'], ['price', '현재가', ''], ['day', '일', ''],
   ['zPct', 'Z', ''], ['mPct', 'M', '']].forEach(([k, label, c]) => {
    const th = el('th', c + (S.sortKey === k ? ' on' : ''),
      label + (S.sortKey === k ? (S.sortDesc ? ' ▼' : ' ▲') : ''));
    th.onclick = () => {
      if (S.sortKey === k) S.sortDesc = !S.sortDesc;
      else { S.sortKey = k; S.sortDesc = true; }
      renderCompare();
    };
    head.appendChild(th);
  });
  t.appendChild(head);

  sortRows(S.rows).forEach(r => {
    const tr = el('tr');
    tr.onclick = () => { S.ticker = r.ticker; localStorage.setItem('ticker', r.ticker); go('analysis'); };

    const nameTd = el('td', 'l');
    nameTd.appendChild(miniCandle({ ...r, price: shownPrice(r) }));
    nameTd.appendChild(el('span', r.holding ? 'hold-dot' : r.hasHistory ? 'hist-dot' : 'nodot'));
    nameTd.appendChild(el('span', 'nm', r.name || r.ticker));
    tr.appendChild(nameTd);

    const d = shownDay(r);
    const pTd = el('td', 'mono ' + cls(d), price(r.krw, shownPrice(r)));
    tr.appendChild(pTd);
    tr.appendChild(el('td', 'mono ' + cls(d), pct(d, 1)));

    [r.zPct, r.mPct].forEach(v => {
      const td = el('td', 'mono', v == null ? '–' : Math.round(v));
      td.style.color = pctColor(v);
      td.style.fontWeight = '700';
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });

  wrap.appendChild(t);
  body.appendChild(wrap);
}

async function loadCompare(force) {
  try {
    const o = await api(`/api/compare?market=${S.market}` + (force ? '&force=true' : ''));
    S.rows = o.rows;
    renderCompare();
    startTicks();
  } catch (e) { fail(e); }
}

/** 실시간 현재가 — 화면에 보이는 종목만 주기적으로 갱신. */
function startTicks() {
  clearInterval(tickTimer);
  tickTimer = setInterval(async () => {
    const syms = S.tab === 'compare' ? (S.rows || []).map(r => r.ticker)
      : S.tab === 'analysis' && S.ticker ? [S.ticker] : [];
    if (!syms.length) return;
    try {
      const o = await api('/api/prices?symbols=' + syms.join(','));
      Object.entries(o).forEach(([k, v]) => { live[k] = v.price; });
      if (S.tab === 'compare') renderCompare();
    } catch (e) { /* 틱 실패는 조용히 넘긴다 — 다음 주기에 다시 시도 */ }
  }, 10000);
}

// ══════════════════════════ 분석 ══════════════════════════

const charts = [];
function clearCharts() { charts.forEach(c => { try { c.remove(); } catch (e) {} }); charts.length = 0; }

function mkChart(host, height, opts = {}) {
  const c = LightweightCharts.createChart(host, {
    width: host.clientWidth, height,
    layout: { background: { color: 'transparent' }, textColor: '#8B95A1', fontSize: 10 },
    grid: { vertLines: { visible: false }, horzLines: { color: '#ffffff0d' } },
    rightPriceScale: { borderColor: '#24242A', scaleMargins: { top: .12, bottom: .08 } },
    timeScale: { borderColor: '#24242A', timeVisible: S.bar === '1m', secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: '#8B95A1', width: 1, style: 2, labelBackgroundColor: '#3182F6' },
      horzLine: { color: '#8B95A1', width: 1, style: 2, labelBackgroundColor: '#3182F6' } },
    handleScale: { axisPressedMouseMove: false },
    ...opts,
  });
  charts.push(c);
  new ResizeObserver(() => c.applyOptions({ width: host.clientWidth })).observe(host);
  return c;
}

function chartBox(parent, title, valueEl) {
  const head = el('div', 'ch-title');
  head.appendChild(el('span', null, title));
  if (valueEl) head.appendChild(valueEl);
  parent.appendChild(head);
  const wrap = el('div', 'ch-wrap');
  const host = el('div', 'chart');
  wrap.appendChild(host);
  parent.appendChild(wrap);
  return { host, wrap };
}

function renderAnalysis() {
  const body = $('#body');
  body.innerHTML = '';
  clearCharts();

  const wrap = el('div');
  wrap.style.padding = '0 var(--pad) 12px';

  // 종목 칩
  const chips = el('div', 'chips');
  (S.rows || []).forEach(r => {
    const b = el('button', r.ticker === S.ticker ? 'on' : '', r.name || r.ticker);
    b.onclick = () => { S.ticker = r.ticker; localStorage.setItem('ticker', r.ticker); loadAnalysis(); };
    chips.appendChild(b);
  });
  wrap.appendChild(chips);
  body.appendChild(wrap);

  const a = S.analysis;
  if (!a) { wrap.appendChild(el('p', 'muted', '불러오는 중…')); return; }

  const r = a.result;
  const minMode = S.bar === '1m';
  const m = S.minutes;

  // 헤더 — 종목 · σ·β · 현재가 · 평단
  const head = el('div', 'anl-head');
  head.appendChild(el('span', 'tk', a.ticker));
  const px = live[a.ticker] ?? (r ? r.lastPrice : a.candles.at(-1).close);
  head.appendChild(el('span', 'mono', price(a.krw, px)));
  if (r) head.appendChild(el('span', 'sub', `σ±${r.sigmaPct.toFixed(0)}% · β ${r.beta.toFixed(1)}`));
  if (a.avgPrice) {
    const roi = (px / a.avgPrice - 1) * 100;
    const s = el('span', 'sub mono');
    s.textContent = `평단 ${price(a.krw, a.avgPrice)} `;
    const b = el('b', cls(roi), pct(roi, 1));
    s.appendChild(b);
    head.appendChild(s);
  }
  wrap.appendChild(head);

  // ── 가격 차트 ──
  const bars = minMode ? (m ? m.candles : []) : a.candles;
  if (!bars.length) {
    wrap.appendChild(el('p', 'muted', minMode ? '1분봉 불러오는 중…' : '일봉이 없습니다'));
  } else {
    const { host, wrap: cw } = chartBox(wrap, minMode ? '가격 · 1분' : '가격 · 일봉');
    const ch = mkChart(host, 260);
    const cs = ch.addCandlestickSeries({
      upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN,
      wickUpColor: UP, wickDownColor: DOWN,
    });
    cs.setData(bars.map(b => ({ time: b.t, open: b.open, high: b.high, low: b.low, close: b.close })));

    // 평단선 — 보유 중일 때만
    if (a.avgPrice) {
      cs.createPriceLine({
        price: a.avgPrice, color: GOLD, lineWidth: 1, lineStyle: 2,
        axisLabelVisible: true, title: '평단',
      });
    }
    // 매매 마커 (일봉에서만 — 분봉은 날짜가 안 맞는다)
    if (!minMode && a.trades.length) {
      cs.setMarkers(a.trades.map(t => ({
        time: Math.floor(new Date(t.date + 'T00:00:00Z').getTime() / 1000),
        position: t.type === 'buy' ? 'belowBar' : 'aboveBar',
        color: t.type === 'buy' ? UP : DOWN,
        shape: t.type === 'buy' ? 'arrowUp' : 'arrowDown',
      })).sort((x, y) => x.time - y.time));
    }

    // 꾹 누르면(모바일) / 올리면(PC) 시고저종 상자
    const tip = el('div', 'ohlc');
    tip.hidden = true;
    cw.appendChild(tip);
    const byTime = new Map(bars.map((b, i) => [b.t, i]));
    ch.subscribeCrosshairMove(p => {
      const d = p.seriesData && p.seriesData.get(cs);
      if (!d || !p.point) { tip.hidden = true; return; }
      const i = byTime.get(p.time);
      const prev = i > 0 ? bars[i - 1].close : d.open;
      const chg = prev ? (d.close / prev - 1) * 100 : 0;
      const dt = new Date(p.time * 1000);
      const label = minMode
        ? dt.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
        : dt.toISOString().slice(2, 10).replace(/-/g, '.');
      tip.innerHTML = '';
      tip.appendChild(el('div', null, `${label}  `));
      const c1 = el('span', cls(chg), pct(chg, 2));
      tip.firstChild.appendChild(c1);
      tip.appendChild(el('div', null, `시 ${num(d.open, 2)}   고 ${num(d.high, 2)}`));
      tip.appendChild(el('div', null, `저 ${num(d.low, 2)}   종 ${num(d.close, 2)}`));
      tip.hidden = false;
    });
  }

  // ── 지표 ──
  if (minMode && m && m.candles.length) {
    lineChart(wrap, 'MACD · 1분', m.candles.map(b => b.t),
      [[m.macd, VIOLET], [m.macdSignal, '#C9C5BB']]);
    lineChart(wrap, 'RSI · 1분', m.candles.map(b => b.t), [[m.rsi, TEAL]], [30, 70]);
  } else if (r) {
    lineChart(wrap, 'Z · M', r.dates, [[r.zPct, UP], [r.mPct, '#FFD24D']], [20, 40, 60, 80]);
    lineChart(wrap, 'MACD', r.dates, [[r.macd, VIOLET], [r.macdSignal, '#C9C5BB']]);
    lineChart(wrap, 'RSI', r.dates, [[r.rsi, TEAL]], [30, 70]);
  } else {
    wrap.appendChild(el('p', 'muted', '분석 데이터 부족 — 상장 후 기간이 짧은 종목입니다'));
  }
}

/** 선 차트 한 판. series = [[값배열, 색], …], guides = 임계 가로선. */
function lineChart(parent, title, times, series, guides = []) {
  const { host } = chartBox(parent, title);
  const ch = mkChart(host, 120);
  series.forEach(([vals, color], idx) => {
    const s = ch.addLineSeries({ color, lineWidth: 2, priceLineVisible: false, lastValueVisible: idx === 0 });
    s.setData(times.map((t, i) => ({ time: t, value: vals[i] }))
      .filter(p => p.value != null && !Number.isNaN(p.value)));
    if (idx === 0) guides.forEach(g => s.createPriceLine({
      price: g, color: '#ffffff22', lineWidth: 1, lineStyle: 2, axisLabelVisible: false,
    }));
  });
}

async function loadAnalysis() {
  if (!S.ticker && S.rows && S.rows.length) S.ticker = S.rows[0].ticker;
  if (!S.ticker) { $('#body').innerHTML = '<p class="muted pad">비교 탭에서 종목을 고르세요</p>'; return; }
  S.analysis = null;
  renderAnalysis();
  try {
    S.analysis = await api('/api/analysis?ticker=' + encodeURIComponent(S.ticker));
    if (S.bar === '1m') S.minutes = await api('/api/minutes?ticker=' + encodeURIComponent(S.ticker));
    renderAnalysis();
    startTicks();
  } catch (e) { fail(e); }
}

// ══════════════════════════ 포트폴리오 ══════════════════════════

const money = krw => S.usdMode
  ? '$' + (krw / S.account.rate).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  : Math.round(krw).toLocaleString('ko-KR') + '원';
const signedMoney = krw => (krw >= 0 ? '+' : '-') + money(Math.abs(krw));
const qtyLabel = q => (Number.isInteger(q) ? q.toLocaleString('ko-KR')
  : String(parseFloat(q.toFixed(4)))) + '주';

function renderPortfolio() {
  const body = $('#body');
  body.innerHTML = '';
  const a = S.account;
  if (!a) { body.appendChild(el('p', 'muted pad', '불러오는 중…')); return; }

  const wrap = el('div');
  wrap.style.padding = '0 var(--pad) 12px';

  const hero = el('section', 'hero');
  const row = el('div', 'row');
  row.appendChild(el('span', 'label', '총자산'));
  row.appendChild(el('span', 'acct', a.accountNo ? '•••••' + a.accountNo.slice(-4) : ''));
  hero.appendChild(row);
  hero.appendChild(el('div', 'total mono', money(a.totalKrw)));
  hero.appendChild(el('div', 'today mono ' + cls(a.dailyPnlKrw),
    `오늘 ${signedMoney(a.dailyPnlKrw)} (${pct(a.dailyPnlRate * 100)})`));
  hero.appendChild(el('div', 'pnl mono ' + cls(a.pnlKrw),
    `평가손익 ${signedMoney(a.pnlKrw)} (${pct(a.pnlRate * 100)})`));

  // 원금이 적혀 있으면 총손익(총자산 − 원금)까지
  if (S.settings && S.settings.principal > 0) {
    const p = S.settings.principal;
    const gain = a.totalKrw - p;
    hero.appendChild(el('div', 'pnl mono ' + cls(gain),
      `원금 ${money(p)} · ${signedMoney(gain)} (${pct(gain / p * 100)})`));
  }

  const chips = el('div', 'chips');
  chips.style.overflow = 'visible';
  [['평가금액', money(a.evalKrw)], ['예수금', money(a.cashKrw)],
   ['환율', num(a.rate, 1)]].forEach(([k, v]) => {
    const c = el('span', 'chip', k + ' ');
    c.appendChild(el('b', 'mono', v));
    chips.appendChild(c);
  });
  hero.appendChild(chips);
  wrap.appendChild(hero);

  // 비중 파이 — conic-gradient 한 줄
  const sum = a.items.reduce((x, h) => x + h.evalKrw, 0);
  if (sum > 0) {
    const pw = el('div', 'pie-wrap');
    const pie = el('div', 'pie');
    let acc = 0;
    pie.style.background = 'conic-gradient(' + a.items.map((h, i) => {
      const from = acc / sum * 360; acc += h.evalKrw;
      return `${PALETTE[i % PALETTE.length]} ${from}deg ${acc / sum * 360}deg`;
    }).join(',') + ')';
    pw.appendChild(pie);
    wrap.appendChild(pw);
  }

  a.items.forEach((h, i) => {
    const art = el('article', 'hold');
    const r1 = el('div', 'r1');
    const dot = el('span', 'dot');
    dot.style.background = PALETTE[i % PALETTE.length];
    r1.appendChild(dot);
    r1.appendChild(el('span', 'name', h.name || h.symbol));
    r1.appendChild(el('span', 'eval mono', money(h.evalKrw)));
    art.appendChild(r1);
    const r2 = el('div', 'r2');
    r2.appendChild(el('span', 'qty', qtyLabel(h.quantity)));
    r2.appendChild(el('span', 'gain mono ' + cls(h.pnlKrw), signedMoney(h.pnlKrw)));
    r2.appendChild(el('span', 'sep', '|'));
    r2.appendChild(el('span', 'rate mono ' + cls(h.pnlRate), pct(h.pnlRate * 100)));
    art.appendChild(r2);
    art.onclick = () => { S.ticker = h.symbol; localStorage.setItem('ticker', h.symbol); go('analysis'); };
    wrap.appendChild(art);
  });

  body.appendChild(wrap);
}

async function loadPortfolio(force) {
  try {
    S.account = await api('/api/account' + (force ? '?force=true' : ''));
    if (!S.settings) S.settings = await api('/api/settings');
    renderPortfolio();
  } catch (e) { fail(e); }
}

// ══════════════════════════ 설정 ══════════════════════════

function renderSettings() {
  const body = $('#body');
  body.innerHTML = '';
  const s = S.settings;
  if (!s) { body.appendChild(el('p', 'muted pad', '불러오는 중…')); return; }

  const wrap = el('div');
  wrap.style.padding = '0 var(--pad) 12px';
  const msg = el('p', 'msg');

  // ── 분석 ──
  wrap.appendChild(el('div', 'sec', '분석'));
  const r1 = el('div', 'row2');
  r1.appendChild(el('span', 'g', '분석 기간'));
  const mi = el('input', 'box num');
  mi.style.width = '60px'; mi.value = s.months; mi.inputMode = 'numeric';
  r1.appendChild(mi);
  r1.appendChild(el('span', null, '개월'));
  const mb = el('button', 'gh acc', '적용');
  mb.onclick = async () => {
    mb.disabled = true;
    try {
      await api('/api/settings/months', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ months: parseInt(mi.value, 10) || s.months }),
      });
      msg.textContent = '적용했습니다. 일봉을 다시 받습니다.';
      S.rows = null; S.settings = await api('/api/settings');
    } catch (e) { msg.textContent = '⚠️ ' + e.message; }
    mb.disabled = false;
  };
  r1.appendChild(mb);
  wrap.appendChild(r1);

  // ── 원금 ──
  wrap.appendChild(el('div', 'sec', '원금'));
  const pr = el('div', 'row2');
  pr.appendChild(el('span', 'g', '입금 합계'));
  pr.appendChild(el('span', 'mono', Math.round(s.principal).toLocaleString('ko-KR') + '원'));
  wrap.appendChild(pr);

  const dr = el('div', 'row2');
  const dd = el('input', 'box num');
  dd.style.width = '120px';
  dd.value = new Date().toISOString().slice(0, 10);
  const da = el('input', 'box num');
  da.placeholder = '금액'; da.inputMode = 'numeric';
  dr.appendChild(dd);
  const g = el('span', 'g'); g.appendChild(da); dr.appendChild(g);
  const db = el('button', 'gh acc', '추가');
  db.onclick = async () => {
    const v = parseFloat(da.value);
    if (!v) return;
    try {
      const o = await api('/api/deposits', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date: dd.value, krw: v }),
      });
      S.settings.deposits = o.deposits; S.settings.principal = o.principal;
      renderSettings();
    } catch (e) { msg.textContent = '⚠️ ' + e.message; }
  };
  dr.appendChild(db);
  wrap.appendChild(dr);

  s.deposits.forEach((d, i) => {
    const row = el('div', 'row2');
    row.appendChild(el('span', 'mono', d.date));
    const amt = el('span', 'g mono', (d.krw >= 0 ? '+' : '') +
      Math.round(d.krw).toLocaleString('ko-KR') + '원');
    amt.style.textAlign = 'right';
    if (d.krw < 0) amt.classList.add('down');
    row.appendChild(amt);
    const x = el('button', 'gh', '삭제');
    x.onclick = async () => {
      const o = await api('/api/deposits/' + i, { method: 'DELETE' });
      S.settings.deposits = o.deposits; S.settings.principal = o.principal;
      renderSettings();
    };
    row.appendChild(x);
    wrap.appendChild(row);
  });

  // ── 종목 관리 ──
  wrap.appendChild(el('div', 'sec', '종목 관리'));
  const ar = el('div', 'row2');
  const ai = el('input', 'box');
  ai.placeholder = '티커 또는 6자리 코드';
  const ag = el('span', 'g'); ag.appendChild(ai); ar.appendChild(ag);
  const ab = el('button', 'gh acc', '추가');
  ab.onclick = async () => {
    if (!ai.value.trim()) return;
    await api('/api/tickers', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker: ai.value.trim() }),
    });
    S.rows = null; S.settings = await api('/api/settings');
    renderSettings();
  };
  ar.appendChild(ab);
  wrap.appendChild(ar);

  s.tickers.forEach(t => {
    const row = el('div', 'row2');
    row.appendChild(el('span', 'g mono', t.ticker));
    const x = el('button', 'gh', '삭제');
    x.onclick = async () => {
      await api('/api/tickers/' + encodeURIComponent(t.ticker), { method: 'DELETE' });
      S.rows = null; S.settings = await api('/api/settings');
      renderSettings();
    };
    row.appendChild(x);
    wrap.appendChild(row);
  });

  // ── 데이터 ──
  wrap.appendChild(el('div', 'sec', '데이터'));
  const fb = el('button', 'pri', `체결내역 가져오기 (${s.trades}건 저장됨)`);
  fb.onclick = async () => {
    fb.disabled = true; msg.textContent = '가져오는 중…';
    try {
      const o = await api('/api/fills', { method: 'POST' });
      msg.textContent = `체결 ${o.fetched}건 조회, 누적 ${o.total}건 저장`;
      S.settings = await api('/api/settings');
    } catch (e) { msg.textContent = '⚠️ ' + e.message; }
    fb.disabled = false;
  };
  wrap.appendChild(fb);
  wrap.appendChild(msg);

  const cb = el('div', 'row2');
  cb.appendChild(el('span', 'g', '일봉 다시 받기'));
  const cbtn = el('button', 'gh', '실행');
  cbtn.onclick = async () => {
    await api('/api/cache/clear', { method: 'POST' });
    S.rows = null;
    msg.textContent = '캐시를 비웠습니다. 비교 탭에서 다시 받습니다.';
  };
  cb.appendChild(cbtn);
  wrap.appendChild(cb);

  body.appendChild(wrap);
}

// ══════════════════════════ 탭 ══════════════════════════

function header() {
  const seg = $('#hdr-seg'), btn = $('#hdr-btn');
  seg.innerHTML = ''; btn.hidden = true;
  const mkSeg = (opts, sel, on) => opts.forEach(([id, label]) => {
    const b = el('button', id === sel ? 'on' : '', label);
    b.onclick = () => on(id);
    seg.appendChild(b);
  });

  if (S.tab === 'compare') {
    $('#title').textContent = '비교';
    mkSeg([['US', '미국'], ['KR', '한국']], S.market, m => {
      S.market = m; localStorage.setItem('market', m); S.rows = null;
      renderCompare(); loadCompare(false);
    });
    btn.hidden = false; btn.textContent = '새로고침';
    btn.onclick = () => { S.rows = null; renderCompare(); loadCompare(true); };
  } else if (S.tab === 'analysis') {
    $('#title').textContent = '분석';
    mkSeg([['1d', '일봉'], ['1m', '1분']], S.bar, b => {
      S.bar = b; localStorage.setItem('bar', b); loadAnalysis();
    });
  } else if (S.tab === 'portfolio') {
    $('#title').textContent = '포트폴리오';
    mkSeg([['krw', '원'], ['usd', '$']], S.usdMode ? 'usd' : 'krw', c => {
      S.usdMode = c === 'usd'; localStorage.setItem('cur', c);
      renderPortfolio(); header();
    });
    btn.hidden = false; btn.textContent = '새로고침';
    btn.onclick = () => loadPortfolio(true);
  } else {
    $('#title').textContent = '설정';
  }
}

function go(tab) {
  S.tab = tab;
  localStorage.setItem('tab', tab);
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('on', b.dataset.tab === tab));
  clearCharts();
  header();
  if (tab === 'compare') { S.rows ? renderCompare() : loadCompare(false); if (S.rows) startTicks(); }
  else if (tab === 'analysis') loadAnalysis();
  else if (tab === 'portfolio') { S.account ? renderPortfolio() : loadPortfolio(false); }
  else { S.settings ? renderSettings() : api('/api/settings').then(o => { S.settings = o; renderSettings(); }).catch(fail); }
}

document.querySelectorAll('#tabs button').forEach(b =>
  b.onclick = () => go(b.dataset.tab));

// 설정은 포트폴리오의 원금 표시에도 필요하므로 처음에 한 번 받아 둔다
api('/api/settings').then(o => { S.settings = o; }).catch(() => {});
go(S.tab);

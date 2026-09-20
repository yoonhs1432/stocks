// 퀀트 대시보드 — 비교 · 분석 · 포트폴리오 · 설정.
// 안드로이드 앱(android-toss)을 옮긴 것. 계산은 전부 서버에서 끝내고 여기서는 그리기만 한다.

const PALETTE = ['#E0A24A', '#D9694E', '#CF5D7F', '#8A6FD0', '#4D8DF0', '#37A48C',
                 '#B5843A', '#7FA8D9', '#C98BB8', '#5FBF8F'];
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
  group: localStorage.getItem('group') || 'series',
  account: null, rows: null, analysis: null, minutes: null, settings: null,
  snaps: null, journal: null, journalOpen: false,
  hist: null, histOpen: false, histDate: null,
};

let tickTimer = null;
const live = {};          // symbol → 실시간 현재가

/**
 * 늦게 끝난 요청이 이미 바뀐 화면을 덮지 않게 하는 관문.
 * 탭을 빨리 누르면(분석→포폴→분석) 포폴이 뒤늦게 도착해 분석 화면 자리에 그려졌다.
 * 받아 온 값은 그대로 두고 **그리기만** 건너뛴다 — 다시 그 탭에 가면 바로 쓴다.
 */
const onTab = t => S.tab === t;

// 한 번 받은 분석은 들고 있는다. 서버도 미리 계산해 두지만, 오가는 시간(폰↔집)이 있어
// 두 번째부터는 아예 안 받는 편이 빠르다.
const anCache = new Map();      // ticker → {at, data}
const minCache = new Map();
const rowCache = new Map();     // 'US'/'KR' → {at, data} — 시장 전환을 기다리지 않게
const AN_FRESH = 5 * 60 * 1000;
const MIN_FRESH = 60 * 1000;

function cacheGet(m, k, fresh) {
  const v = m.get(k);
  return (v && Date.now() - v.at < fresh) ? v.data : null;
}
function cachePut(m, k, data) { m.set(k, { at: Date.now(), data }); }

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
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    // fetch 자체가 실패하면 브라우저는 "Failed to fetch" 만 던진다. 실제로는 PC 가
    // 꺼졌거나 터널 주소가 바뀐 경우라, 그대로 보여 주면 뭘 해야 할지 알 수 없다.
    throw new Error('PC 서버에 연결할 수 없습니다. PC 가 켜져 있는지, 터널 주소가 바뀌지 않았는지 확인하세요.');
  }
  const o = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(o.error || `요청 실패 (${res.status})`);
  return o;
}

/** 오류 화면. **항상 다시 시도 버튼을 같이 둔다** — 없으면 탭을 나갔다 와야 했다. */
function fail(e, retry) {
  const body = $('#body');
  body.innerHTML = '';
  const box = el('div', 'pad');
  box.appendChild(el('p', 'err', '⚠️ ' + e.message));
  const b = el('button', 'gh', '다시 시도');
  b.style.marginTop = '10px';
  b.onclick = () => (retry || (() => go(S.tab)))();
  box.appendChild(b);
  body.appendChild(box);
}

// ══════════════════════════ 비교 ══════════════════════════

/** 표 왼쪽 미니 캔들 — 당일 시/고/저/종을 7×20 으로. */
function miniCandle(r) {
  const c = document.createElement('canvas');
  const w = 7, h = 20, dpr = window.devicePixelRatio || 1;
  c.width = w * dpr; c.height = h * dpr;
  c.style.width = w + 'px'; c.style.height = h + 'px';
  drawCandle(c, r);
  return c;
}

/** 이미 있는 캔버스에 다시 그린다 (틱마다 캔버스를 새로 만들지 않게). */
function drawCandle(c, r) {
  const w = 7, h = 20, dpr = window.devicePixelRatio || 1;
  const g = c.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, c.width, c.height);
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
  // 열 폭을 못 박는다 — 종목명이 길어도 숫자 열(특히 맨 끝 M)이 밀려나지 않게
  const cg = el('colgroup');
  ['c-name', 'c-price', 'c-day', 'c-zm', 'c-zm'].forEach(c => cg.appendChild(el('col', c)));
  t.appendChild(cg);
  const head = el('tr');
  [['name', '종목', 'l'], ['price', '현재가', ''], ['day', '일', ''],
   ['zPct', 'Z', ''], ['mPct', 'M', '']].forEach(([k, label, c]) => {
    const th = el('th', c + (S.sortKey === k ? ' on' : ''),
      label + (S.sortKey === k ? (S.sortDesc ? ' ▼' : ' ▲') : ''));
    th.onclick = () => {
      if (S.sortKey === k) S.sortDesc = !S.sortDesc;
      // 숫자는 큰 값부터가 자연스럽지만 이름은 ㄱ→ㅎ 이 자연스럽다
      else { S.sortKey = k; S.sortDesc = k !== 'name'; }
      renderCompare();
    };
    head.appendChild(th);
  });
  t.appendChild(head);

  // 틱마다 표를 통째로 다시 만들면 미니 캔들 19개를 매번 새로 그리게 되고,
  // 다시 정렬까지 하면 **행이 위아래로 튄다.** 그려 둔 칸을 기억해 두고 숫자만 고친다.
  cmpRefs = [];
  sortRows(S.rows).forEach(r => {
    const tr = el('tr', 'row');       // colgroup·머리글과 구분되게 표시해 둔다
    tr.onclick = () => { S.ticker = r.ticker; localStorage.setItem('ticker', r.ticker); go('analysis'); };

    const nameTd = el('td', 'l');
    const cv = miniCandle({ ...r, price: shownPrice(r) });
    nameTd.appendChild(cv);
    nameTd.appendChild(el('span', r.holding ? 'hold-dot' : r.hasHistory ? 'hist-dot' : 'nodot'));
    nameTd.appendChild(el('span', 'nm', r.name || r.ticker));
    tr.appendChild(nameTd);

    const d = shownDay(r);
    const pTd = el('td', 'mono ' + cls(d), price(r.krw, shownPrice(r)));
    tr.appendChild(pTd);
    const dTd = el('td', 'mono ' + cls(d), pct(d, 1));
    tr.appendChild(dTd);

    [r.zPct, r.mPct].forEach(v => {
      const td = el('td', 'mono', v == null ? '–' : Math.round(v));
      td.style.color = pctColor(v);
      td.style.fontWeight = '700';
      tr.appendChild(td);
    });
    t.appendChild(tr);
    cmpRefs.push({ r, cv, pTd, dTd, nameTd });
  });

  wrap.appendChild(t);

  // 언제 기준 숫자인지 — 일봉은 최대 6시간 캐시라 "지금"이 아닐 수 있다
  const hhmm = ms => new Date(ms).toLocaleTimeString('ko-KR',
    { hour: '2-digit', minute: '2-digit' });
  const stamp = el('p', 'muted stamp');
  cmpStamp = stamp;
  stampText(stamp);
  wrap.appendChild(stamp);

  body.appendChild(wrap);
}

let cmpRefs = null, cmpStamp = null;
const hhmm2 = ms => new Date(ms).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
function stampText(e) {
  e.textContent = (S.rowsAt ? `조회 ${hhmm2(S.rowsAt)}` : '') +
    (S.tickAt ? ` · 현재가 ${hhmm2(S.tickAt)}` : '') + ' · 일봉은 최대 6시간 캐시';
}

/** 틱에서 쓰는 가벼운 갱신 — 값이 바뀐 칸만 고친다. 정렬은 건드리지 않는다. */
function tickCompare() {
  if (!cmpRefs || !document.querySelector('#body table.cmp')) { renderCompare(); return; }
  cmpRefs.forEach(({ r, cv, pTd, dTd }) => {
    const d = shownDay(r), p = shownPrice(r);
    const c = 'mono ' + cls(d);
    pTd.className = c; dTd.className = c;
    pTd.textContent = price(r.krw, p);
    dTd.textContent = pct(d, 1);
    drawCandle(cv, { ...r, price: p });
  });
  if (cmpStamp) stampText(cmpStamp);
}

const ROWS_FRESH = 3 * 60 * 1000;

async function loadCompare(force) {
  stopPrefetch();           // 미리받기가 회선을 물고 있으면 화면이 늦게 뜬다
  const mk = S.market;
  const hit = force ? null : cacheGet(rowCache, mk, ROWS_FRESH);
  if (hit) {                // 한 번 본 시장은 곧바로 그린다 (현재가는 틱이 갱신한다)
    S.rows = hit.rows;
    S.rowsAt = hit.at;
    if (!onTab('compare')) return;
    renderCompare();
    startTicks();
    prefetchAll();
    return;
  }
  try {
    const o = await api(`/api/compare?market=${mk}` + (force ? '&force=true' : ''));
    if (mk !== S.market) return;        // 그 사이 다른 시장을 눌렀다
    S.rows = o.rows;
    S.rowsAt = (o.asOf ? o.asOf * 1000 : Date.now());
    cachePut(rowCache, mk, { rows: o.rows, at: S.rowsAt });
    if (!onTab('compare')) return;      // 그 사이 다른 탭으로 갔다
    renderCompare();
    startTicks();
    prefetchAll();          // 분석을 미리 받아 둔다 — 종목을 눌렀을 때 기다리지 않게
  } catch (e) { if (onTab('compare')) fail(e, () => loadCompare(force)); }
}

/** 지금 보는 종목의 1분봉을 미리 받아 둔다 — 일봉↔1분 전환을 기다리지 않게. */
function prefetchMinutes() {
  const t = S.ticker;
  if (!t || cacheGet(minCache, t, MIN_FRESH)) return;
  api('/api/minutes?ticker=' + encodeURIComponent(t))
    .then(m => cachePut(minCache, t, m))
    .catch(() => {});
}

/** 반대쪽 시장 표를 미리 받아 둔다 — 미국↔한국 전환을 기다리지 않게. */
function prefetchMarket() {
  const other = S.market === 'US' ? 'KR' : 'US';
  if (cacheGet(rowCache, other, ROWS_FRESH)) return;
  api(`/api/compare?market=${other}`)
    .then(o => cachePut(rowCache, other, { rows: o.rows, at: (o.asOf ? o.asOf * 1000 : Date.now()) }))
    .catch(() => {});
}

/** 실시간 현재가 — 화면에 보이는 종목만 주기적으로 갱신. */
function startTicks() {
  clearInterval(tickTimer);
  const sec = S.settings ? (S.settings.tickSeconds ?? 10) : 10;
  if (!sec) return;                  // 0 = 끔
  tickTimer = setInterval(async () => {
    const syms = S.tab === 'compare' ? (S.rows || []).map(r => r.ticker)
      : S.tab === 'analysis' && S.ticker ? [S.ticker]
      : S.tab === 'portfolio' && S.account ? S.account.items.map(h => h.symbol) : [];
    if (!syms.length) return;
    try {
      const o = await api('/api/prices?symbols=' + syms.join(','));
      Object.entries(o).forEach(([k, v]) => { live[k] = v.price; });
      S.tickAt = Date.now();
      if (S.tab === 'compare') tickCompare();
      else if (S.tab === 'portfolio') tickPortfolio();
    } catch (e) { /* 틱 실패는 조용히 넘긴다 — 다음 주기에 다시 시도 */ }
  }, sec * 1000);
}

// ══════════════════════════ 분석 ══════════════════════════

const charts = [];
function clearCharts() { charts.forEach(c => { try { c.remove(); } catch (e) {} }); charts.length = 0; }

function mkChart(host, height, opts = {}) {
  const c = LightweightCharts.createChart(host, {
    // ⚠️ autoSize 가 필요하다. 차트를 만드는 시점에 host 가 아직 문서에 붙기 전이면
    // clientWidth 가 0 이고, 그 0 을 기준으로 잡힌 구간이 그대로 남아 데이터가 한 줄로
    // 뭉쳐 버린다(포트폴리오 차트가 실제로 그랬다). autoSize 는 붙은 뒤 스스로 맞춘다.
    autoSize: true, height,
    layout: { background: { color: 'transparent' }, textColor: '#8B95A1', fontSize: 10 },
    grid: { vertLines: { visible: false }, horzLines: { color: '#ffffff0d' } },
    // 위아래 여백이 좁으면 축의 맨 위·맨 아래 숫자가 화면 경계에서 잘린다(작은 폰에서 확인)
    rightPriceScale: { borderColor: '#24242A', scaleMargins: { top: .16, bottom: .13 } },
    timeScale: { borderColor: '#24242A', timeVisible: S.bar === '1m', secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: { color: '#8B95A1', width: 1, style: 2, labelBackgroundColor: '#3182F6' },
      horzLine: { color: '#8B95A1', width: 1, style: 2, labelBackgroundColor: '#3182F6' } },
    handleScale: { axisPressedMouseMove: false },
    // ⚠️ 세로로 끄는 것은 차트가 먹지 않게 한다. 안 그러면 그래프 위에서 위아래로 쓸 때
    // 화면이 안 내려가고 차트만 움직인다(폰에서 실제로 걸렸다).
    // 가로로 끄는 것은 그대로 차트 이동 — 봉을 훑어보는 동작은 살아 있다.
    handleScroll: { mouseWheel: true, pressedMouseMove: true,
                    horzTouchDrag: true, vertTouchDrag: false },
    localization: { locale: 'ko-KR' },
    ...opts,
  });
  charts.push(c);
  return c;
}

/**
 * 처음 보여 줄 구간. 점이 적으면 꽉 채우고, 많으면 **최근 N개**만.
 *
 * 그냥 두면 lightweight-charts 가 기본 간격으로 오른쪽에 붙여 그려서, 스냅샷처럼
 * 점이 20개쯤이면 화면 왼쪽 절반이 텅 빈다. 반대로 일봉 500개를 다 채우면 너무 촘촘하다
 * (안드로이드도 기본 2개월만 보여 줬다).
 */
const rangeKey = () => 'range-' + S.bar;      // 일봉과 1분봉은 따로 기억한다
function savedRange() {
  try {
    const v = JSON.parse(localStorage.getItem(rangeKey()) || 'null');
    return (v && v.span > 2) ? v : null;
  } catch (e) { return null; }
}

/**
 * @param mode null=기억 안 씀(포트폴리오) · 'apply'=기억한 구간 적용 · 'remember'=적용+저장
 *
 * 봉 개수는 종목마다 다르므로 절대 위치를 저장하면 엉뚱한 곳이 열린다. 그래서
 * **오른쪽 끝에서 몇 칸 떨어졌는지(fromEnd) + 몇 칸을 보고 있는지(span)** 를 남긴다.
 * 그러면 종목을 바꿔도, 앱을 껐다 켜도 같은 배율·같은 위치로 열린다.
 */
function fitRange(chart, count, recent = 45, mode = null) {
  // 폭이 잡힌 다음 프레임에 적용한다 — 붙기 전에 계산하면 0 폭 기준이 된다
  requestAnimationFrame(() => {
    const ts = chart.timeScale();
    const saved = mode ? savedRange() : null;
    if (saved && count > 3) {
      const span = Math.min(saved.span, count - 1);
      const to = count - 1 - Math.max(0, Math.min(saved.fromEnd, count - span - 1));
      ts.setVisibleLogicalRange({ from: to - span, to });
    } else if (count <= recent) {
      ts.fitContent();
    } else {
      ts.setVisibleLogicalRange({ from: count - recent, to: count - 1 });
    }
    if (mode === 'remember') {
      ts.subscribeVisibleLogicalRangeChange(r => {
        if (!r) return;
        localStorage.setItem(rangeKey(), JSON.stringify({
          span: Math.round(r.to - r.from), fromEnd: Math.round(count - 1 - r.to),
        }));
      });
    }
  });
}

/**
 * 여러 차트의 시간축을 묶는다 — 하나를 확대·이동하면 나머지도 따라간다.
 *
 * ⚠️ **날짜로 묶는다(칸 번호가 아니라).** 지표 차트(Z·M·MACD·RSI)는 SPY 와 날짜가
 * 겹치는 날만 쓰므로 가격 차트보다 점이 적을 수 있다. 칸 번호로 묶으면 같은 번호가
 * 서로 다른 날을 가리켜, 가격은 9월인데 Z·M 은 5월을 보여 주는 식으로 어긋난다
 * (국내 종목에서 실제로 그랬다).
 */
function linkTime(list) {
  // 맞춰 준 결과가 이벤트로 **되돌아온다.** 데이터가 짧은 차트는 구간을 잘라서 알려 주는데
  // 그걸 다시 모두에게 퍼뜨리면 구간이 점점 쪼그라든다(하루까지 줄어드는 걸 봤다).
  // 그래서 우리가 맞춘 직후 잠깐은 들어오는 이벤트를 무시한다.
  let quiet = 0;
  const apply = (from, r) => {
    if (!r || performance.now() < quiet) return;
    quiet = performance.now() + 150;
    list.forEach(o => { if (o !== from) { try { o.timeScale().setVisibleRange(r); } catch (e) {} } });
  };
  list.forEach(c => c.timeScale().subscribeVisibleTimeRangeChange(r => apply(c, r)));
  // 처음 한 번, 가격 차트가 잡은 구간으로 나머지를 맞춘다
  requestAnimationFrame(() => {
    try {
      quiet = 0;
      apply(list[0], list[0].timeScale().getVisibleRange());
    } catch (e) {}
  });
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

function renderAnalysis(err) {
  const body = $('#body');
  body.innerHTML = '';
  clearCharts();

  const wrap = el('div');
  wrap.style.padding = '0 var(--pad) 12px';

  // 종목 칩
  const chips = el('div', 'tchips');
  (S.rows || []).forEach(r => {
    const b = el('button', r.ticker === S.ticker ? 'on' : '', r.name || r.ticker);
    b.onclick = () => { S.ticker = r.ticker; localStorage.setItem('ticker', r.ticker); loadAnalysis(); };
    chips.appendChild(b);
  });
  wrap.appendChild(chips);
  body.appendChild(wrap);
  // 뒤쪽 종목을 고르면 칩이 화면 밖에 있어 뭘 보는지 알 수 없었다 → 가운데로 당겨 온다
  requestAnimationFrame(() => {
    const on = chips.querySelector('button.on');
    if (on) chips.scrollLeft = on.offsetLeft - chips.clientWidth / 2 + on.offsetWidth / 2;
  });

  // 오류가 나도 칩은 남겨 둔다 — 칩까지 지우면 다른 종목으로 갈 수도, 다시 받을 수도 없다
  if (err) {
    wrap.appendChild(el('p', 'err', '⚠️ ' + err.message));
    const rb = el('button', 'gh', '다시 시도');
    rb.style.marginTop = '10px';
    rb.onclick = () => loadAnalysis();
    wrap.appendChild(rb);
    return;
  }

  const a = S.analysis;
  if (!a) { wrap.appendChild(el('p', 'muted', '불러오는 중…')); return; }

  const r = a.result;
  const minMode = S.bar === '1m';
  const m = S.minutes;
  const linked = [];          // 시간축을 함께 움직일 차트들

  // 헤더 — 종목 · σ·β · 현재가 · 평단
  const head = el('div', 'anl-head');
  head.appendChild(el('span', 'tk', a.name || a.ticker));
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

  // ── 가격 차트 (산점도 묶음이면 건너뛴다) ──
  const bars = S.group === 'scatter' ? [] : (minMode ? (m ? m.candles : []) : a.candles);
  if (S.group === 'scatter') {
    // 아래 산점도 블록에서 그린다
  } else if (!bars.length) {
    wrap.appendChild(el('p', 'muted', minMode ? '1분봉 불러오는 중…' : '일봉이 없습니다'));
  } else {
    const { host, wrap: cw } = chartBox(wrap, minMode ? '가격 · 1분' : '가격 · 일봉');
    const ch = mkChart(host, 260);
    const cs = ch.addCandlestickSeries({
      upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN,
      wickUpColor: UP, wickDownColor: DOWN,
    });
    cs.setData(bars.map(b => ({ time: b.t, open: b.open, high: b.high, low: b.low, close: b.close })));
    fitRange(ch, bars.length, minMode ? 120 : 45, 'remember');
    linked.push(ch);

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

  // ── 산점도 묶음 ──
  if (S.group === 'scatter') {
    if (!r) {
      wrap.appendChild(el('p', 'muted', '분석 데이터 부족 — 상장 후 기간이 짧은 종목입니다'));
      return;
    }
    const a1 = chartBox(wrap, '회귀 산점도 (SPY 대비)');
    regressionScatter(a1.host, r);
    const a2 = chartBox(wrap, 'Z·M 궤적');
    zmScatter(a2.host, r);
    // 폭이 바뀌면(회전 등) 다시 그린다 — 캔버스는 알아서 늘어나지 않는다
    new ResizeObserver(() => { regressionScatter(a1.host, r); zmScatter(a2.host, r); })
      .observe(wrap);
    return;
  }

  // ── 지표 ──
  if (minMode && m && m.candles.length) {
    lineChart(wrap, 'MACD · 1분', m.candles.map(b => b.t),
      [[m.macd, VIOLET], [m.macdSignal, '#C9C5BB']], [], linked, 120);
    lineChart(wrap, 'RSI · 1분', m.candles.map(b => b.t), [[m.rsi, TEAL]], [30, 70], linked, 120);
  } else if (r) {
    lineChart(wrap, 'Z · M', r.dates, [[r.zPct, UP], [r.mPct, '#FFD24D']], [20, 40, 60, 80], linked);
    lineChart(wrap, 'MACD', r.dates, [[r.macd, VIOLET], [r.macdSignal, '#C9C5BB']], [], linked);
    lineChart(wrap, 'RSI', r.dates, [[r.rsi, TEAL]], [30, 70], linked);
  } else {
    wrap.appendChild(el('p', 'muted', '분석 데이터 부족 — 상장 후 기간이 짧은 종목입니다'));
  }
  if (linked.length > 1) linkTime(linked);
}

/** 선 차트 한 판. series = [[값배열, 색], …], guides = 임계 가로선. */
function lineChart(parent, title, times, series, guides = [], linked = null, recent = 45) {
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
  fitRange(ch, times.length, recent, 'apply');
  if (linked) linked.push(ch);
  return ch;
}

// 분석 요청 번호. 칩을 빠르게 연달아 누르면 먼저 부른 종목의 응답이 **나중에** 도착해
// 화면을 덮어쓸 수 있다(칩은 SOXL 인데 차트는 TQQQ). 마지막 요청이 아니면 버린다.
let analysisSeq = 0;

async function loadAnalysis() {
  const seq = ++analysisSeq;
  stopPrefetch();
  // 종목 칩과 기본 종목이 비교 데이터에서 나온다. 분석 탭을 열어 둔 채 새로고침하면
  // 그게 없어서 화면이 통째로 비었다 → 없으면 여기서 직접 받아 온다.
  if (!S.rows) {
    $('#body').innerHTML = '<p class="muted pad">종목 목록 불러오는 중…</p>';
    try {
      const o = await api(`/api/compare?market=${S.market}`);
      if (seq !== analysisSeq) return;
      S.rows = o.rows;
      S.rowsAt = (o.asOf ? o.asOf * 1000 : Date.now());
      if (!onTab('analysis')) return;
    } catch (e) { if (onTab('analysis')) fail(e, () => loadAnalysis()); return; }
  }
  if (!S.ticker || !S.rows.some(r => r.ticker === S.ticker)) {
    S.ticker = S.rows.length ? S.rows[0].ticker : null;
  }
  if (!S.ticker) { $('#body').innerHTML = '<p class="muted pad">설정에서 종목을 추가하세요</p>'; return; }
  // 받아 둔 게 있으면 기다리지 않고 바로 그린다
  const hit = cacheGet(anCache, S.ticker, AN_FRESH);
  S.analysis = hit;
  S.minutes = S.bar === '1m' ? cacheGet(minCache, S.ticker, MIN_FRESH) : null;
  renderAnalysis();
  if (hit && (S.bar !== '1m' || S.minutes)) { startTicks(); prefetchAll(); return; }
  try {
    let a = hit;
    if (!a) {
      a = await api('/api/analysis?ticker=' + encodeURIComponent(S.ticker));
      if (seq !== analysisSeq) return;        // 그 사이 다른 종목을 눌렀다
      cachePut(anCache, S.ticker, a);
    }
    if (!onTab('analysis')) return;
    S.analysis = a;
    if (S.bar === '1m' && !S.minutes) {
      const m = await api('/api/minutes?ticker=' + encodeURIComponent(S.ticker));
      if (seq !== analysisSeq) return;
      cachePut(minCache, S.ticker, m);
      S.minutes = m;
    }
    renderAnalysis();
    startTicks();
    prefetchAll();
  } catch (e) {
    if (seq !== analysisSeq || !onTab('analysis')) return;
    S.analysis = null; renderAnalysis(e);
  }
}

/**
 * **목록 전체를 미리 받아 둔다.** 서버는 이미 계산을 끝내 놓았으므로 남은 건 오가는
 * 시간뿐인데, 그것마저 미리 치러 두면 종목을 눌렀을 때 기다림이 없다.
 *
 * 비교를 받은 직후부터 시작해 지금 보는 종목 다음 것들부터 하나씩(겹치지 않게) 받는다.
 * 데이터 절약 모드이거나 2G 면 하지 않는다.
 */
let preTimer = null;
let preGen = 0;

/** 미리받기를 멈춘다. 사용자가 누른 요청이 먼저 가야 한다(회선은 하나다). */
function stopPrefetch() {
  preGen++;
  clearTimeout(preTimer);
}

function prefetchAll(delay = 600) {
  if (S.settings && S.settings.prefetch === false) return;   // 설정에서 끔
  const c = navigator.connection;
  if (c && (c.saveData || /(^|-)2g$/.test(c.effectiveType || ''))) return;
  clearTimeout(preTimer);
  const all = (S.rows || []).map(r => r.ticker);
  const i = Math.max(0, all.indexOf(S.ticker));
  const order = [...all.slice(i), ...all.slice(0, i)];
  const todo = order.filter(t => !cacheGet(anCache, t, AN_FRESH));
  const gen = ++preGen;
  let k = 0;
  const step = () => {
    if (gen !== preGen) return;         // 그 사이 사용자가 뭔가 눌렀다
    if (k >= todo.length) { prefetchMinutes(); prefetchMarket(); return; }
    const t = todo[k++];
    api('/api/analysis?ticker=' + encodeURIComponent(t))
      .then(a => cachePut(anCache, t, a))
      .catch(() => {})
      .finally(() => { if (gen === preGen) preTimer = setTimeout(step, 120); });
  };
  preTimer = setTimeout(step, delay);
}

// ══════════════════════════ 포트폴리오 ══════════════════════════

/**
 * 실시간 현재가를 덮어써 계좌를 다시 계산한다 (안드로이드 PortfolioScreen 과 같은 규칙).
 *
 * 계좌 전체를 다시 부르면 토스 호출이 4번(계좌·보유·예수금·환율)이라 무겁다.
 * 현재가 1번만 받아 **평가금액·손익을 여기서 다시 계산**하면 틱마다 갱신할 수 있다.
 * 기준은 API 와 동일 — 누적 손익률 = 현재가/평단 − 1. 안 맞추면 증권사 앱과 숫자가 어긋난다.
 *
 * ⚠️ 정렬은 서버가 준 순서(조회 시점 평가금액)를 그대로 쓴다. 틱마다 다시 정렬하면
 * 행이 위아래로 튀어서 읽을 수가 없다.
 */
function liveAccount() {
  const a = S.account;
  if (!a) return null;
  let ev = 0, pnl = 0, buy = 0, daily = 0, baseSum = 0;
  const items = a.items.map(h => {
    const k = h.currency === 'USD' ? a.rate : 1;
    const p = live[h.symbol];
    // 전일 기준가 — 당일 손익률에서 역산 (토스가 기준가를 따로 주지 않는다)
    const base = (h.dailyPnlRate > -1 && h.dailyPnlRate !== 0)
      ? h.lastPrice / (1 + h.dailyPnlRate) : h.lastPrice;
    buy += h.avgPrice * h.quantity * k;
    baseSum += base * h.quantity * k;
    if (p == null) {
      ev += h.evalKrw; pnl += h.pnlKrw; daily += h.dailyPnlAmount * k;
      return h;
    }
    const evalKrw = p * h.quantity * k;
    const pnlKrw = (p - h.avgPrice) * h.quantity * k;
    ev += evalKrw; pnl += pnlKrw; daily += (p - base) * h.quantity * k;
    return { ...h, evalKrw, pnlKrw,
             pnlRate: h.avgPrice > 0 ? p / h.avgPrice - 1 : h.pnlRate };
  });
  return { ...a, items,
    evalKrw: ev, totalKrw: ev + a.cashKrw, pnlKrw: pnl,
    pnlRate: buy > 0 ? pnl / buy : a.pnlRate,
    dailyPnlKrw: daily, dailyPnlRate: baseSum > 0 ? daily / baseSum : a.dailyPnlRate };
}

const money = krw => S.usdMode
  ? '$' + (krw / S.account.rate).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  : Math.round(krw).toLocaleString('ko-KR') + '원';
const signedMoney = krw => (krw >= 0 ? '+' : '-') + money(Math.abs(krw));

/** 큰 금액은 만원 단위로 — 2,506만원 / 작은 값은 소수 2자리 11.50만원. */
const manwon = krw => {
  const v = krw / 10000;
  return (Math.abs(v) < 100 ? v.toFixed(2) : Math.round(v).toLocaleString('ko-KR')) + '만원';
};
const short = krw => S.usdMode ? money(krw) : manwon(krw);
const signedShort = krw => (krw >= 0 ? '+' : '-') + short(Math.abs(krw));
const qtyLabel = q => (Number.isInteger(q) ? q.toLocaleString('ko-KR')
  : String(parseFloat(q.toFixed(4)))) + '주';

/** 비중 파이. 조각이 너무 얇으면 글자가 겹치므로 일정 비율 이상만 이름을 적는다. */
function pieSvg(items, sum, size = 186) {
  const NS = 'http://www.w3.org/2000/svg';
  const R = size / 2 - 3, C = size / 2;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  const xy = (ang, r) => [C + r * Math.cos(ang - Math.PI / 2), C + r * Math.sin(ang - Math.PI / 2)];

  let acc = 0;
  items.forEach((h, i) => {
    const frac = h.evalKrw / sum;
    const a0 = acc * Math.PI * 2, a1 = (acc + frac) * Math.PI * 2;
    acc += frac;
    const [x0, y0] = xy(a0, R), [x1, y1] = xy(a1, R);
    const path = document.createElementNS(NS, 'path');
    // 한 종목뿐이면 호로는 원이 안 닫힌다 → 원으로 그린다
    path.setAttribute('d', frac >= 0.999
      ? `M ${C} ${C - R} A ${R} ${R} 0 1 1 ${C - 0.01} ${C - R} Z`
      : `M ${C} ${C} L ${x0} ${y0} A ${R} ${R} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1} Z`);
    path.setAttribute('fill', PALETTE[i % PALETTE.length]);
    svg.appendChild(path);

    if (frac >= 0.06) {                       // 6% 미만은 글자가 서로 겹친다
      const [lx, ly] = xy((a0 + a1) / 2, R * 0.63);
      const g = document.createElementNS(NS, 'text');
      g.setAttribute('x', lx); g.setAttribute('y', ly);
      g.setAttribute('text-anchor', 'middle');
      g.setAttribute('fill', '#12121A');
      g.setAttribute('font-size', '11');
      g.setAttribute('font-weight', '800');
      const n1 = document.createElementNS(NS, 'tspan');
      n1.textContent = (h.name || h.symbol).slice(0, 6);
      n1.setAttribute('x', lx); n1.setAttribute('dy', '-2');
      const n2 = document.createElementNS(NS, 'tspan');
      n2.textContent = (frac * 100).toFixed(1) + '%';
      n2.setAttribute('x', lx); n2.setAttribute('dy', '12');
      n2.setAttribute('font-size', '10');
      g.append(n1, n2);
      svg.appendChild(g);
    }
  });
  return svg;
}

/** 차트 아래 한 줄 — 시작일 · 기록 일수 · 끝일. 차트 자체 시간축보다 읽기 쉽다. */
function axisRow(from, mid, to) {
  const r = el('div', 'axrow');
  r.appendChild(el('span', null, from));
  r.appendChild(el('span', 'muted', mid));
  r.appendChild(el('span', null, to));
  return r;
}

function renderPortfolio() {
  const body = $('#body');
  // ⚠️ 이전 차트를 반드시 버린다. 안 그러면 틱마다 새 차트가 쌓여 (30초에 60개)
  // 폰을 켜 둘수록 느려지다 결국 멈춘다. renderAnalysis 는 하고 있었는데 여기만 빠져 있었다.
  clearCharts();
  pfRefs = null;
  body.innerHTML = '';
  const a = liveAccount();
  if (!a) { body.appendChild(el('p', 'muted pad', '불러오는 중…')); return; }

  const wrap = el('div');
  wrap.style.padding = '0 var(--pad) 12px';

  // 왼쪽에 총자산, 오른쪽에 색 범례 — 아래 자산 그래프의 색과 짝이 맞는다
  const hero = el('section', 'hero');
  const top = el('div', 'hero-top');

  const left = el('div');
  const row = el('div', 'row');
  row.appendChild(el('span', 'label', '현재 총자산'));
  left.appendChild(row);
  const totalEl = el('div', 'total mono', short(a.totalKrw));
  left.appendChild(totalEl);
  const todayEl = el('div', 'today mono ' + cls(a.dailyPnlKrw),
    `오늘 ${signedShort(a.dailyPnlKrw)} (${pct(a.dailyPnlRate * 100)})`);
  left.appendChild(todayEl);
  const prin = S.settings ? S.settings.principal : 0;
  let prinEl = null;
  if (prin > 0) {
    const gain = a.totalKrw - prin;
    prinEl = el('div', 'pnl mono ' + cls(gain),
      `원금 ${short(prin)} · ${signedShort(gain)} (${pct(gain / prin * 100)})`);
    left.appendChild(prinEl);
  }
  top.appendChild(left);

  const leg = el('div', 'legend');
  const legend = [['평가금액', UP, a.evalKrw], ['예수금', '#5B9BF2', a.cashKrw]];
  if (prin > 0) legend.push(['원금', GOLD, prin]);
  const legEls = [];
  legend.forEach(([k, color, v]) => {
    const r = el('div', 'lg');
    const d = el('span', 'dot');
    d.style.background = color;
    r.appendChild(d);
    r.appendChild(el('span', 'k', k));
    const vEl = el('span', 'v mono', short(v));
    r.appendChild(vEl);
    legEls.push(vEl);
    leg.appendChild(r);
  });
  const fx = el('div', 'lg');
  fx.appendChild(el('span', 'k muted', '환율'));
  fx.appendChild(el('span', 'v mono muted', num(a.rate, 1)));
  leg.appendChild(fx);
  top.appendChild(leg);

  hero.appendChild(top);
  row.appendChild(el('span', 'acct', a.accountNo ? '•••••' + a.accountNo.slice(-4) : ''));
  wrap.appendChild(hero);
  pfHead = { totalEl, todayEl, prinEl, legEls };

  // 비중 파이 — 조각 위에 종목과 % 를 얹으려면 SVG 라야 한다(conic-gradient 는 글자를 못 얹는다)
  const sum = a.items.reduce((x, h) => x + h.evalKrw, 0);
  if (sum > 0) {
    const pw = el('div', 'pie-wrap');
    pw.appendChild(pieSvg(a.items, sum));
    wrap.appendChild(pw);
    const cap = el('p', 'muted', '보유 종목 비중 · 예수금은 빠져 있습니다');
    cap.style.textAlign = 'center';
    cap.style.margin = '2px 0 6px';
    wrap.appendChild(cap);
  }

  pfRefs = { total: null, today: null, prin: null, legend: [], rows: [],
             syms: a.items.map(h => h.symbol).join(',') };
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
    const w = sum > 0 ? (h.evalKrw / sum * 100) : null;
    const qtyEl = el('span', 'qty', qtyLabel(h.quantity) +
      (w == null ? '' : ` · ${w.toFixed(1)}%`));
    r2.appendChild(qtyEl);
    const gainEl = el('span', 'gain mono ' + cls(h.pnlKrw), signedMoney(h.pnlKrw));
    r2.appendChild(gainEl);
    r2.appendChild(el('span', 'sep', '|'));
    const rateEl = el('span', 'rate mono ' + cls(h.pnlRate), pct(h.pnlRate * 100));
    r2.appendChild(rateEl);
    art.appendChild(r2);
    pfRefs.rows.push({ sym: h.symbol, evalEl: r1.lastChild, qtyEl, gainEl, rateEl });
    art.onclick = () => { S.ticker = h.symbol; localStorage.setItem('ticker', h.symbol); go('analysis'); };
    wrap.appendChild(art);
  });

  // ── 자산 추이 ──
  // 토스에 과거 잔고 API 가 없어 서버가 날마다 남긴 스냅샷으로 그린다.
  // 기록이 시작된 날부터만 쌓이고, 앱을 안 연 날은 비어 있다.
  const sn = S.snaps;
  if (sn && sn.dates.length >= 2) {
    const div = S.usdMode ? 1 : 10000;      // 원화는 만원 단위라야 축이 읽힌다
    const t = sn.dates;

    // ── 자산 ── 평가금액(빨강) 위에 예수금을 쌓아 **윗면이 총자산**이 되게 그린다.
    // 총자산 영역(파랑)을 먼저 깔고 평가금액을 그 위에 덮으면 사이 띠가 예수금이 된다.
    const a1 = chartBox(wrap, '자산');
    const c1 = mkChart(a1.host, 190, { timeScale: { visible: false } });
    const totS = c1.addAreaSeries({ lineColor: '#EEF1F4', topColor: 'rgba(58,110,165,.75)',
      bottomColor: 'rgba(58,110,165,.35)', lineWidth: 2, priceLineVisible: false,
      lastValueVisible: false });
    totS.setData(t.map((d, i) => ({ time: d, value: sn.total[i] / div })));
    const evalS = c1.addAreaSeries({ lineColor: UP, topColor: 'rgba(138,42,48,.95)',
      bottomColor: 'rgba(138,42,48,.75)', lineWidth: 2, priceLineVisible: false,
      lastValueVisible: false });
    evalS.setData(t.map((d, i) => ({ time: d, value: sn.eval[i] / div })));
    if (sn.principal.some(v => v != null)) {
      const pS = c1.addLineSeries({ color: GOLD, lineWidth: 1, lineStyle: 2,
        priceLineVisible: false, lastValueVisible: false });
      pS.setData(t.map((d, i) => ({ time: d, value: sn.principal[i] == null ? undefined : sn.principal[i] / div }))
        .filter(x => x.value !== undefined));
    }
    fitRange(c1, t.length, t.length);
    wrap.appendChild(axisRow(t[0], `${t.length}일 기록`, t.at(-1)));

    // ── 평가손익 ── 지금 값과, 보고 있는 구간에서 얼마나 움직였는지
    const pnlNow = sn.pnl.at(-1), pnlFrom = sn.pnl[0];
    const head = el('div', 'ch-title');
    head.appendChild(el('span', null, '평가손익'));
    const rt = el('span', 'v');
    rt.appendChild(el('span', 'muted', '표시 구간 '));
    rt.appendChild(el('b', cls(pnlNow - pnlFrom), signedShort(pnlNow - pnlFrom)));
    head.appendChild(rt);
    wrap.appendChild(head);

    const big = el('div', 'pnl-big');
    big.appendChild(el('span', 'k muted', '현재'));
    big.appendChild(el('span', 'v mono ' + cls(pnlNow), short(pnlNow)));
    if (prin > 0) {
      big.appendChild(el('span', 'r mono ' + cls(pnlNow),
        `원금 대비 ${pct(pnlNow / prin * 100)}`));
    }
    wrap.appendChild(big);

    const a2 = chartBox(wrap, '');
    const c2 = mkChart(a2.host, 150, { timeScale: { visible: false } });
    const pnlS = c2.addAreaSeries({ lineColor: UP, topColor: 'rgba(239,96,102,.35)',
      bottomColor: 'rgba(239,96,102,0)', lineWidth: 2, priceLineVisible: false,
      lastValueVisible: false });
    pnlS.setData(t.map((d, i) => ({ time: d, value: sn.pnl[i] / div })));
    pnlS.createPriceLine({ price: 0, color: '#ffffff33', lineWidth: 1, lineStyle: 2,
      axisLabelVisible: false });
    fitRange(c2, t.length, t.length);
    wrap.appendChild(axisRow(t[0], `${t.length}일 기록`, t.at(-1)));
  } else if (sn) {
    wrap.appendChild(el('p', 'muted', `기록 ${sn.dates.length}일 — 2일 이상 쌓이면 자산 추이가 표시됩니다`));
  }

  // ── 기록 ── 그날 무엇을 얼마나 들고 있었는지. 시세는 다시 받을 수 있지만
  // 보유 구성은 그때 남겨 두지 않으면 어디에도 없다.
  const hw = el('div');
  const hh = el('div', 'ch-title');
  hh.style.cursor = 'pointer';
  hh.appendChild(el('span', null, `기록${S.hist ? ` (${S.hist.dates.length}일)` : ''}`));
  hh.appendChild(el('span', 'v muted', S.histOpen ? '▲' : '▼'));
  hh.onclick = async () => {
    S.histOpen = !S.histOpen;
    if (S.histOpen && !S.hist) {
      try { S.hist = await api('/api/history?days=180'); } catch (e) { /* 없으면 비워 둔다 */ }
      if (S.hist && S.hist.dates.length) S.histDate = S.hist.dates.at(-1);
    }
    if (!onTab('portfolio')) return;
    renderPortfolio();
  };
  hw.appendChild(hh);
  if (S.histOpen) renderHistory(hw);
  wrap.appendChild(hw);

  // ── 매매 일지 ──
  const jw = el('div');
  const jh = el('div', 'ch-title');
  jh.style.cursor = 'pointer';
  jh.appendChild(el('span', null, `매매 일지${S.journal ? ` (${S.journal.total}건)` : ''}`));
  jh.appendChild(el('span', 'v muted', S.journalOpen ? '▲' : '▼'));
  jh.onclick = async () => {
    S.journalOpen = !S.journalOpen;
    if (S.journalOpen && !S.journal) {
      try { S.journal = await api('/api/journal'); } catch (e) { /* 없으면 비워 둔다 */ }
    }
    if (!onTab('portfolio')) return;
    renderPortfolio();
  };
  jw.appendChild(jh);
  if (S.journalOpen && S.journal) {
    S.journal.trades.forEach(tr => {
      const row = el('div', 'row2');
      row.appendChild(el('span', 'mono', tr.date));
      const nm = el('span', 'g mono', tr.ticker);
      row.appendChild(nm);
      row.appendChild(el('span', 'mono ' + (tr.type === 'buy' ? 'up' : 'down'),
        (tr.type === 'buy' ? '매수 ' : '매도 ') + qtyLabel(tr.qty)));
      row.appendChild(el('span', 'mono', price(tr.krw, tr.price)));
      jw.appendChild(row);
    });
    if (!S.journal.trades.length) {
      jw.appendChild(el('p', 'muted', '설정에서 체결내역을 가져오면 표시됩니다'));
    }
  }
  wrap.appendChild(jw);

  body.appendChild(wrap);
}

/** 기록 섹션 본문 — 종목별 평가금액 추이(쌓은 그래프) + 고른 날짜의 보유 내역. */
function renderHistory(parent) {
  const h = S.hist;
  if (!h || !h.dates.length) {
    parent.appendChild(el('p', 'muted', '아직 기록이 없습니다. 하루에 한 번씩 쌓입니다.'));
    return;
  }
  const div = S.usdMode ? 1 : 10000;
  const rate = S.account ? S.account.rate : 1400;

  if (h.dates.length >= 2) {
    const { host } = chartBox(parent, '종목별 평가금액');
    const c = mkChart(host, 170, { timeScale: { visible: false } });
    // 쌓아 그린다 — 누적이 큰 것부터 깔고 작은 것을 위에 덮으면 띠가 종목별 몫이 된다
    const syms = h.symbols.map(x => x.symbol);
    const cum = syms.map((_, k) => h.dates.map((_, i) =>
      syms.slice(k).reduce((x, sm) => x + ((h.eval[sm] || [])[i] || 0), 0)));
    syms.forEach((sm, k) => {
      const color = PALETTE[k % PALETTE.length];
      const a = c.addAreaSeries({ lineColor: color, topColor: color, bottomColor: color,
        lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
      a.setData(h.dates.map((d, i) => ({ time: d, value: cum[k][i] / (S.usdMode ? rate : div) })));
    });
    fitRange(c, h.dates.length, h.dates.length);
    parent.appendChild(axisRow(h.dates[0], `${h.dates.length}일 기록`, h.dates.at(-1)));
  }

  // 날짜 고르기
  const pick = el('div', 'row2');
  pick.appendChild(el('span', 'g', '날짜'));
  const sel = el('select', 'box hist-date');
  sel.style.width = '150px';
  [...h.dates].reverse().forEach(d => {
    const o = el('option', null, d);
    o.value = d;
    if (d === S.histDate) o.selected = true;
    sel.appendChild(o);
  });
  sel.onchange = () => { S.histDate = sel.value; renderPortfolio(); };
  pick.appendChild(sel);
  parent.appendChild(pick);

  const items = h.byDate[S.histDate] || [];
  if (!items.length) {
    parent.appendChild(el('p', 'muted', '그날 기록이 없습니다.'));
    return;
  }
  const t = el('table', 'cmp hist');
  const head = el('tr', 'head');
  [['종목', 'l'], ['수량', ''], ['평단', ''], ['평가금액', ''], ['손익', '']]
    .forEach(([label, c]) => head.appendChild(el('th', c, label)));
  t.appendChild(head);
  items.forEach(it => {
    const tr = el('tr', 'row');
    tr.appendChild(el('td', 'l', it.n || it.s));
    tr.appendChild(el('td', 'mono', qtyLabel(it.q)));
    tr.appendChild(el('td', 'mono', num(it.a, it.a >= 1000 ? 0 : 2)));
    tr.appendChild(el('td', 'mono', short(it.e)));
    tr.appendChild(el('td', 'mono ' + cls(it.g), signedShort(it.g)));
    t.appendChild(tr);
  });
  parent.appendChild(t);
  const sum = items.reduce((x, it) => x + it.e, 0);
  const gain = items.reduce((x, it) => x + it.g, 0);
  const cap = el('p', 'muted');
  cap.textContent = `${S.histDate} 합계 ${short(sum)} · 평가손익 ${signedShort(gain)}`;
  parent.appendChild(cap);
}

let pfRefs = null, pfHead = null;

/**
 * 틱에서 쓰는 가벼운 갱신 — 숫자만 고친다.
 * 통째로 다시 그리면 차트 3개를 매번 새로 만들고 자산 그래프가 10초마다 리셋된다.
 */
function tickPortfolio() {
  const a = liveAccount();
  if (!a || !pfRefs || !pfHead || pfRefs.syms !== a.items.map(h => h.symbol).join(',')) {
    renderPortfolio();
    return;
  }
  const sum = a.items.reduce((x, h) => x + h.evalKrw, 0);
  pfHead.totalEl.textContent = short(a.totalKrw);
  pfHead.todayEl.className = 'today mono ' + cls(a.dailyPnlKrw);
  pfHead.todayEl.textContent = `오늘 ${signedShort(a.dailyPnlKrw)} (${pct(a.dailyPnlRate * 100)})`;
  const prin = S.settings ? S.settings.principal : 0;
  if (pfHead.prinEl && prin > 0) {
    const gain = a.totalKrw - prin;
    pfHead.prinEl.className = 'pnl mono ' + cls(gain);
    pfHead.prinEl.textContent =
      `원금 ${short(prin)} · ${signedShort(gain)} (${pct(gain / prin * 100)})`;
  }
  if (pfHead.legEls[0]) pfHead.legEls[0].textContent = short(a.evalKrw);
  if (pfHead.legEls[1]) pfHead.legEls[1].textContent = short(a.cashKrw);

  const by = new Map(a.items.map(h => [h.symbol, h]));
  pfRefs.rows.forEach(ref => {
    const h = by.get(ref.sym);
    if (!h) return;
    ref.evalEl.textContent = money(h.evalKrw);
    const w = sum > 0 ? (h.evalKrw / sum * 100) : null;
    ref.qtyEl.textContent = qtyLabel(h.quantity) + (w == null ? '' : ` · ${w.toFixed(1)}%`);
    ref.gainEl.className = 'gain mono ' + cls(h.pnlKrw);
    ref.gainEl.textContent = signedMoney(h.pnlKrw);
    ref.rateEl.className = 'rate mono ' + cls(h.pnlRate);
    ref.rateEl.textContent = pct(h.pnlRate * 100);
  });
}

let acctTimer = null;

async function loadPortfolio(force) {
  try {
    // 차례로 기다리면 집 밖에서는 왕복만 3번이다. 서로 필요 없으니 같이 보낸다.
    const usd = S.usdMode;
    const [acc, snaps, settings] = await Promise.all([
      api('/api/account' + (force ? '?force=true' : '')),
      api('/api/snapshots' + (usd ? '?usd=true' : '')),
      S.settings ? Promise.resolve(S.settings) : api('/api/settings'),
    ]);
    S.account = acc;
    if (usd === S.usdMode) S.snaps = snaps;   // 그 사이 통화를 바꿨으면 그 값이 맞다
    S.settings = settings;
    if (force) S.hist = null;
    if (!onTab('portfolio')) return;    // 그 사이 다른 탭으로 갔다
    renderPortfolio();
    startTicks();
    startAccountRefresh();
  } catch (e) { if (onTab('portfolio')) fail(e, () => loadPortfolio(force)); }
}

/**
 * 계좌 전체 재조회 — 60초. 현재가는 틱이 맡고, 여기서는 **예수금·환율·보유 종목 변동**
 * 처럼 현재가로 알 수 없는 것만 따라잡는다. 호출이 4번이라 자주 부를 수 없다.
 */
function startAccountRefresh() {
  clearInterval(acctTimer);
  const sec = S.settings ? (S.settings.tickSeconds ?? 10) : 10;
  if (!sec) return;                  // 갱신 끔이면 계좌도 자동으로 다시 받지 않는다
  acctTimer = setInterval(async () => {
    if (S.tab !== 'portfolio') return;
    try {
      S.account = await api('/api/account');
      if (!onTab('portfolio')) return;
      renderPortfolio();
    } catch (e) { /* 조용히 넘긴다 */ }
  }, 60000);
}

// ══════════════════════════ 설정 ══════════════════════════

/** 저장을 마친 뒤 다시 그리기 — **아직 설정 탭일 때만.**
    저장하고 곧바로 탭을 옮기면 설정 화면이 딴 탭 위에 그려졌다. */
function reSettings() { if (onTab('settings')) renderSettings(); }

function renderSettings() {
  const body = $('#body');
  body.innerHTML = '';
  const s = S.settings;
  if (!s) { body.appendChild(el('p', 'muted pad', '불러오는 중…')); return; }

  const wrap = el('div');
  wrap.style.padding = '0 var(--pad) 12px';
  // 안내 문구는 S.msg 에 담는다 — 저장 뒤 화면을 다시 그리면 지역 변수만으로는 사라진다
  const msg = el('p', 'msg');
  msg.textContent = S.msg || '';
  const say = t => { S.msg = t; msg.textContent = t; };

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
      say('적용했습니다. 일봉을 다시 받습니다.');
      S.rows = null; S.settings = await api('/api/settings');
    } catch (e) { say('⚠️ ' + e.message); }
    mb.disabled = false;
  };
  r1.appendChild(mb);
  wrap.appendChild(r1);

  const pr2 = el('div', 'row2');
  pr2.appendChild(el('span', 'g', '미리 받기'));
  const psel = el('select', 'box pre-sel');
  psel.style.width = '110px';
  [[1, '켬 (빠름)'], [0, '끔 (데이터 절약)']].forEach(([v, lab]) => {
    const o = el('option', null, lab);
    o.value = v;
    if (!!v === (s.prefetch !== false)) o.selected = true;
    psel.appendChild(o);
  });
  psel.onchange = async () => {
    const on = psel.value === '1';
    try {
      await api('/api/settings/prefetch', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ on }),
      });
      S.settings.prefetch = on;
      say(on ? '종목을 미리 받아 둡니다.' : '누를 때만 받습니다.');
    } catch (e) { say('⚠️ ' + e.message); }
  };
  pr2.appendChild(psel);
  wrap.appendChild(pr2);

  const tr2 = el('div', 'row2');
  tr2.appendChild(el('span', 'g', '실시간 갱신'));
  const tsel = el('select', 'box');
  tsel.style.width = '110px';
  [[0, '끔'], [5, '5초'], [10, '10초'], [30, '30초'], [60, '60초']].forEach(([v, lab]) => {
    const o = el('option', null, lab);
    o.value = v;
    if (v === (s.tickSeconds ?? 10)) o.selected = true;
    tsel.appendChild(o);
  });
  tsel.onchange = async () => {
    await api('/api/settings/tick', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seconds: parseInt(tsel.value, 10) }),
    });
    S.settings.tickSeconds = parseInt(tsel.value, 10);
    startTicks();
  };
  tr2.appendChild(tsel);
  wrap.appendChild(tr2);

  // ── 원금 ──
  wrap.appendChild(el('div', 'sec', '원금'));
  const pr = el('div', 'row2');
  pr.appendChild(el('span', 'g', '입금 합계'));
  pr.appendChild(el('span', 'mono', Math.round(s.principal).toLocaleString('ko-KR') + '원'));
  wrap.appendChild(pr);

  const dr = el('div', 'row2');
  const dd = el('input', 'box num');
  dd.type = 'date';           // 손으로 2026-09-19 를 치게 두지 않는다
  dd.style.width = '150px';
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
      reSettings();
    } catch (e) { say('⚠️ ' + e.message); }
  };
  dr.appendChild(db);
  wrap.appendChild(dr);

  // 여러 건 한 번에 — 다른 곳에 적어 둔 기록을 그대로 옮겨 붙일 수 있게
  const dta = el('textarea', 'box dep-in');
  dta.rows = 3;
  dta.placeholder = '2026-09-01 13,789,303\n2026-09-04 10,728,849   (한 줄에 한 건)';
  wrap.appendChild(dta);
  const dbr = el('div', 'row2 dep-row');
  dbr.appendChild(el('span', 'g muted', '한 줄에 “날짜 금액”'));
  const drep = el('button', 'gh', '통째로 바꾸기');
  const dadd = el('button', 'gh acc', '추가');
  const bulk = async (replace) => {
    if (!dta.value.trim()) return;
    if (replace && !confirm('지금 입금 기록을 모두 지우고 적힌 것으로 바꿉니다.')) return;
    dadd.disabled = drep.disabled = true;
    try {
      const o = await api('/api/deposits/bulk', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: dta.value, replace: !!replace }),
      });
      S.settings.deposits = o.deposits; S.settings.principal = o.principal;
      S.snaps = null;                       // 원금 선이 바뀐다
      say(`${o.added}건 적용했습니다.`);
      dta.value = '';
      reSettings();
    } catch (e) { say('⚠️ ' + e.message); }
    dadd.disabled = drep.disabled = false;
  };
  dadd.onclick = () => bulk(false);
  drep.onclick = () => bulk(true);
  dbr.appendChild(drep);
  dbr.appendChild(dadd);
  wrap.appendChild(dbr);

  s.deposits.forEach((d, i) => {
    const row = el('div', 'row2');
    row.appendChild(el('span', 'mono', d.date));
    const amt = el('span', 'g mono', (d.krw >= 0 ? '+' : '') +
      Math.round(d.krw).toLocaleString('ko-KR') + '원');
    amt.style.textAlign = 'right';
    if (d.krw < 0) amt.classList.add('down');
    row.appendChild(amt);
    const x = el('button', 'gh del', '삭제');
    x.onclick = async () => {
      // 입금 기록은 토스에서 다시 받아올 수 없다. 실수로 지우면 원금이 통째로 틀어진다.
      if (!confirm(`${d.date} ${Math.round(d.krw).toLocaleString('ko-KR')}원 기록을 지울까요?\n` +
                   '입금 기록은 되돌릴 수 없습니다.')) return;
      const o = await api('/api/deposits/' + i, { method: 'DELETE' });
      S.settings.deposits = o.deposits; S.settings.principal = o.principal;
      reSettings();
    };
    row.appendChild(x);
    wrap.appendChild(row);
  });

  // ── 접속 ──
  wrap.appendChild(el('div', 'sec', '접속'));
  const kr = el('div', 'row2');
  kr.appendChild(el('span', 'g', '접속 암호'));
  const kv = el('span', 'mono');
  kv.style.fontSize = '12px';
  const mask = () => { kv.textContent = '•'.repeat(12); };
  mask();
  const show = el('button', 'gh', '보기');
  show.onclick = () => {
    if (show.textContent === '보기') { kv.textContent = s.accessToken || '–'; show.textContent = '숨기기'; }
    else { mask(); show.textContent = '보기'; }
  };
  kr.appendChild(kv);
  kr.appendChild(show);
  wrap.appendChild(kr);

  const kr2 = el('div', 'row2');
  kr2.appendChild(el('span', 'g muted', '암호가 새어 나갔다면 새로 만드세요'));
  const kb = el('button', 'gh del', '새로 만들기');
  kb.onclick = async () => {
    if (!confirm('접속 암호를 새로 만들까요?\n' +
                 '지금 접속 중인 다른 기기(폰 등)는 새 암호로 다시 들어가야 합니다.')) return;
    kb.disabled = true;
    try {
      const o = await api('/api/auth/rotate', { method: 'POST' });
      S.settings.accessToken = o.token;
      reSettings();
      // 폰에서 다시 들어갈 주소를 바로 알려 준다
      alert('새 암호: ' + o.token + '\n\n폰에서는 주소 뒤에 ?key=' + o.token + ' 를 붙여 한 번 열면 됩니다.');
    } catch (e) { say('⚠️ ' + e.message); }
    kb.disabled = false;
  };
  kr2.appendChild(kb);
  wrap.appendChild(kr2);

  // ── 앱 ──
  // 고친 코드를 받으려고 PC 앞에 갈 필요가 없게. 서버가 직접 받아 와 다시 뜬다.
  wrap.appendChild(el('div', 'sec', '앱'));
  const vr = el('div', 'row2');
  vr.appendChild(el('span', 'g', '현재 버전'));
  vr.appendChild(el('span', 'mono', s.version || '?'));
  wrap.appendChild(vr);

  const ub = el('button', 'pri', '업데이트 받기');
  ub.onclick = () => applyUpdate(ub, say);
  wrap.appendChild(ub);

  // ── 데이터 ──
  wrap.appendChild(el('div', 'sec', '데이터'));

  const bk = el('div', 'row2');
  bk.appendChild(el('span', 'g', '기록 내려받기'));
  const bl = el('a', 'gh', '백업 파일');
  bl.href = '/api/backup';
  bl.setAttribute('download', '');
  bl.style.textDecoration = 'none';
  bk.appendChild(bl);
  wrap.appendChild(bk);
  const bn = el('p', 'muted');
  bn.textContent = '입금·매매·자산 추이 기록은 토스에서 다시 못 받습니다. ' +
    'PC 에도 하루 한 번 data/backup 에 복사본이 쌓입니다.';
  wrap.appendChild(bn);
  const fb = el('button', 'pri', `체결내역 가져오기 (${s.trades}건 저장됨)`);
  fb.onclick = async () => {
    fb.disabled = true; say('가져오는 중…');
    try {
      const o = await api('/api/fills', { method: 'POST' });
      say(`체결 ${o.fetched}건 조회, 누적 ${o.total}건 저장`);
      S.settings = await api('/api/settings');
    } catch (e) { say('⚠️ ' + e.message); }
    fb.disabled = false;
  };
  wrap.appendChild(fb);
  wrap.appendChild(msg);

  const cb = el('div', 'row2');
  cb.appendChild(el('span', 'g', '일봉 다시 받기'));
  const cbtn = el('button', 'gh', '실행');
  cbtn.onclick = async () => {
    await api('/api/cache/clear', { method: 'POST' });
    S.rows = null; anCache.clear(); rowCache.clear(); minCache.clear();
    say('캐시를 비웠습니다. 비교 탭에서 다시 받습니다.');
  };
  cb.appendChild(cbtn);
  wrap.appendChild(cb);

  // ── 종목 관리 ──
  wrap.appendChild(el('div', 'sec', `종목 관리 (${s.tickers.length})`));
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
    S.rows = null; anCache.clear(); rowCache.clear(); S.settings = await api('/api/settings');
    reSettings();
  };
  ar.appendChild(ab);
  wrap.appendChild(ar);

  // 한 줄에 하나씩 놓으면 20종목이면 20줄이라 아래 항목이 한참 밑으로 밀린다 → 칩으로
  const list = el('div', 'tklist');
  s.tickers.forEach(t => {
    const c = el('span', 'tk' + (t.krw ? ' kr' : ''));
    c.appendChild(el('b', null, t.ticker));
    if (t.name) c.appendChild(el('i', null, t.name));
    const x = el('button', null, '✕');
    x.onclick = async () => {
      if (!confirm(`${t.ticker} 를 목록에서 뺄까요?`)) return;
      await api('/api/tickers/' + encodeURIComponent(t.ticker), { method: 'DELETE' });
      S.rows = null; anCache.clear(); rowCache.clear(); S.settings = await api('/api/settings');
      reSettings();
    };
    c.appendChild(x);
    list.appendChild(c);
  });
  wrap.appendChild(list);

  // 목록 통째로 바꾸기 — 다른 앱에서 쓰던 목록을 한 번에 옮길 수 있게
  const bt = el('div', 'sec2', '목록 통째로 바꾸기');
  wrap.appendChild(bt);
  const ta = el('textarea', 'box tk-in');
  ta.rows = 4;
  ta.placeholder = 'FNGU, TQQQ, 005930=이름 …  (쉼표·줄바꿈으로 구분, 이름은 선택)';
  ta.value = s.tickers.map(t => t.ticker + (t.name ? '=' + t.name : '')).join(', ');
  wrap.appendChild(ta);
  const br = el('div', 'row2 tk-row');
  br.appendChild(el('span', 'g muted', '적힌 것만 남습니다 · 코드=이름 으로 이름도 지정'));
  const bb = el('button', 'gh acc', '통째로 저장');
  bb.onclick = async () => {
    const n = (ta.value.match(/[^\s,;]+/g) || []).length;
    if (!n || !confirm(`목록을 ${n}개로 바꿉니다. 지금 목록은 사라집니다.`)) return;
    bb.disabled = true;
    try {
      await api('/api/tickers/bulk', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: ta.value }),
      });
      S.rows = null; anCache.clear(); rowCache.clear(); S.settings = await api('/api/settings');
      say(`${n}개로 바꿨습니다.`);
      reSettings();
    } catch (e) { say('⚠️ ' + e.message); }
    bb.disabled = false;
  };
  br.appendChild(bb);
  wrap.appendChild(br);

  body.appendChild(wrap);
}

/**
 * 새 코드를 받아 적용한다. 설정 탭의 버튼과 위쪽 알림 띠가 같이 쓴다.
 * 받은 게 있으면 서버가 스스로 재시작하므로, 다시 뜰 때까지 기다렸다 새로고침한다.
 */
async function applyUpdate(btn, say = () => {}) {
  if (btn.disabled) return;
  btn.disabled = true;
  const label = btn.textContent;
  btn.textContent = '받는 중…';
  say('받는 중…');
  try {
    const o = await api('/api/update', { method: 'POST' });
    if (o.restarting) {
      btn.textContent = '다시 시작 중…';
      say('새 코드를 받았습니다. 다시 시작하는 중…');
      await waitForServer();
      location.reload();
      return;
    }
    say(o.changed ? `받았습니다. ${o.note}` : '이미 최신입니다.');
    S.upd = null;
    renderBanner();
    if (S.tab === 'settings') { S.settings = await api('/api/settings'); renderSettings(); }
  } catch (e) { say('⚠️ ' + e.message); }
  btn.disabled = false;
  btn.textContent = label;
}

// ── 새 버전 알림 띠 ──
// 서버가 5분마다 확인만 한다. 받는 시점은 사용자가 정한다 — 보고 있는데 화면이 갑자기
// 다시 뜨면 곤란하기 때문이다.
function renderBanner() {
  const o = S.upd;
  let b = $('#newver');
  if (!o || !o.available || localStorage.getItem('skipVer') === o.subject) {
    if (b) b.remove();
    return;
  }
  if (!b) {
    b = el('div');
    b.id = 'newver';
    document.body.insertBefore(b, $('#body'));
  }
  b.innerHTML = '';
  b.appendChild(el('span', 'nv-txt', `새 버전 — ${o.subject}`));
  const go = el('button', 'nv-go', '받기');
  go.onclick = () => applyUpdate(go);
  b.appendChild(go);
  const x = el('button', 'nv-x', '✕');
  x.title = '이 버전은 넘어가기';
  x.onclick = () => { localStorage.setItem('skipVer', o.subject); b.remove(); };
  b.appendChild(x);
}

async function checkUpdate() {
  try {
    S.upd = await api('/api/update/check');
    renderBanner();
  } catch (e) { /* 확인 실패는 조용히 — 인터넷이 잠깐 끊겼을 수 있다 */ }
}

/** 서버가 다시 뜰 때까지 기다린다 — 재시작은 보통 2~5초. */
async function waitForServer(sec = 60) {
  for (let i = 0; i < sec; i++) {
    await new Promise(r => setTimeout(r, 1000));
    try {
      const res = await fetch('/api/health', { cache: 'no-store' });
      if (res.ok) return true;
    } catch (e) { /* 아직 안 떴다 */ }
  }
  return false;
}

// ══════════════════════════ 탭 ══════════════════════════

/** 버튼을 잠그고 끝날 때까지 "받는 중…" 으로 바꾼다. */
async function busyBtn(btn, fn) {
  if (btn.disabled) return;
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = '받는 중…';
  try { await fn(); } finally { btn.disabled = false; btn.textContent = label; }
}

function header() {
  const seg = $('#hdr-seg'), btn = $('#hdr-btn');
  seg.innerHTML = ''; btn.hidden = true;
  const mkSeg = (opts, sel, on) => opts.forEach(([id, label]) => {
    const b = el('button', id === sel ? 'on' : '', label);
    b.onclick = () => {
      // 칠은 여기서 직접 옮긴다. 콜백이 header() 를 다시 불러 주기를 기대하면
      // 하나만 빠뜨려도 "눌렀는데 색이 그대로다"가 된다(비교 탭 미국/한국이 그랬다).
      seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
      on(id);
    };
    seg.appendChild(b);
  });

  if (S.tab === 'compare') {
    $('#title').textContent = '비교';
    mkSeg([['US', '미국'], ['KR', '한국']], S.market, m => {
      S.market = m; localStorage.setItem('market', m); S.rows = null;
      renderCompare(); loadCompare(false);
    });
    btn.hidden = false; btn.textContent = '새로고침';
    // 비교 새로고침은 20~30초짜리 작업이다. 잠그지 않으면 반응이 없어 또 누르게 되고
    // 그만큼 요청이 겹쳐 더 느려진다.
    btn.onclick = () => busyBtn(btn, () => { S.rows = null; renderCompare(); return loadCompare(true); });
  } else if (S.tab === 'analysis') {
    $('#title').textContent = '분석';
    if (S.group === 'series') {
      mkSeg([['1d', '일봉'], ['1m', '1분']], S.bar, b => {
        S.bar = b; localStorage.setItem('bar', b); loadAnalysis();
      });
    }
    btn.hidden = false;
    btn.textContent = S.group === 'series' ? '산점도' : '시계열';
    btn.onclick = () => {
      S.group = S.group === 'series' ? 'scatter' : 'series';
      localStorage.setItem('group', S.group);
      header(); renderAnalysis();
    };
  } else if (S.tab === 'portfolio') {
    $('#title').textContent = '포트폴리오';
    mkSeg([['krw', '원'], ['usd', '$']], S.usdMode ? 'usd' : 'krw', async c => {
      S.usdMode = c === 'usd'; localStorage.setItem('cur', c);
      header(); renderPortfolio();
      // 과거 금액은 **그날 환율**로 환산해야 해서 서버에서 다시 받는다.
      // 원/$ 를 연달아 누르면 늦게 온 쪽이 나중에 도착해 **화면은 원인데 그래프는 달러**가
      // 됐다. 요청을 보낼 때의 통화를 기억해 두고, 그 사이 바뀌었으면 버린다.
      const usd = S.usdMode;
      try {
        const sn = await api('/api/snapshots' + (usd ? '?usd=true' : ''));
        if (usd !== S.usdMode || !onTab('portfolio')) return;
        S.snaps = sn;
      } catch (e) { return; }
      renderPortfolio();
    });
    btn.hidden = false; btn.textContent = '새로고침';
    btn.onclick = () => busyBtn(btn, () => loadPortfolio(true));
  } else {
    $('#title').textContent = '설정';
  }
}

function go(tab, fromBack) {
  if (tab !== S.tab) S.msg = '';      // 지난 탭의 안내 문구를 들고 다니지 않는다
  // 폰의 뒤로가기가 앱을 닫아 버리지 않고 **직전 탭으로** 가게 한다.
  if (!fromBack && tab !== S.tab) history.pushState({ tab }, '');
  S.tab = tab;
  localStorage.setItem('tab', tab);
  window.scrollTo(0, 0);      // 탭을 바꿨는데 이전 탭의 스크롤 위치에서 시작하면 헷갈린다
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('on', b.dataset.tab === tab));
  clearCharts();
  if (tab !== 'portfolio') clearInterval(acctTimer);
  header();

  // 받아 둔 게 없으면 **먼저 비운다.** 안 그러면 이전 탭 화면이 그대로 남아 있어
  // 제목만 바뀐 것처럼 보인다(분석인데 비교 표가 깔려 있는 식).
  const ready = tab === 'compare' ? S.rows
    : tab === 'analysis' ? (S.rows && S.analysis)
    : tab === 'portfolio' ? S.account : S.settings;
  if (!ready) $('#body').innerHTML = '<p class="muted pad">불러오는 중…</p>';

  if (tab === 'compare') { S.rows ? renderCompare() : loadCompare(false); if (S.rows) startTicks(); }
  else if (tab === 'analysis') loadAnalysis();
  else if (tab === 'portfolio') { S.account ? renderPortfolio() : loadPortfolio(false); }
  else {
    if (S.settings) renderSettings();
    else api('/api/settings')
      .then(o => { S.settings = o; if (onTab('settings')) renderSettings(); })
      .catch(e => { if (onTab('settings')) fail(e); });
  }
}

document.querySelectorAll('#tabs button').forEach(b =>
  b.onclick = () => go(b.dataset.tab));

// 앱을 내려놓으면(화면 꺼짐·다른 앱) 시세 요청을 멈춘다 — 배터리·데이터를 아낀다.
// 돌아오면 곧바로 한 번 받아 최신으로 맞춘다.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    clearInterval(tickTimer); clearInterval(acctTimer); stopPrefetch();
  } else {
    startTicks();
    if (S.tab === 'portfolio') { loadPortfolio(false); startAccountRefresh(); }
    else if (S.tab === 'compare') loadCompare(false);
  }
});

history.replaceState({ tab: S.tab }, '');
window.addEventListener('popstate', e => {
  const t = (e.state && e.state.tab) || 'compare';
  if (t !== S.tab) go(t, true);
});

/**
 * 모바일 크롬은 주소창이 접혔다 펴질 때 **보이는 영역과 레이아웃 영역이 어긋난다.**
 * 그대로 두면 `bottom:0` 인 탭바가 화면 밖으로 조금 밀려나 누르기 어렵다.
 * 실제로 보이는 영역의 바닥에 맞춰 끌어올린다. PC 에서는 차이가 0 이라 아무 일도 없다.
 */
function pinTabs() {
  const vv = window.visualViewport;
  if (!vv) return;
  const gap = document.documentElement.clientHeight - (vv.height + vv.offsetTop);
  $('#tabs').style.transform = gap > 1 ? `translateY(${-Math.round(gap)}px)` : '';
}
if (window.visualViewport) {
  visualViewport.addEventListener('resize', pinTabs);
  visualViewport.addEventListener('scroll', pinTabs);
  window.addEventListener('scroll', pinTabs, { passive: true });
  pinTabs();
}

// 설정은 포트폴리오의 원금 표시에도 필요하므로 처음에 한 번 받아 둔다
api('/api/settings').then(o => { S.settings = o; }).catch(() => {});

// 새 버전이 올라왔는지 확인 — 열 때 한 번, 그 뒤 5분마다
checkUpdate();
setInterval(checkUpdate, 5 * 60 * 1000);
go(S.tab);

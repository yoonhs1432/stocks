// 산점도 2종 — `ui/Charts.kt` 의 RegressionScatter · ZmScatter 를 캔버스로 옮긴 것.
// lightweight-charts 는 산점도를 지원하지 않아 직접 그린다.

/** Turbo 컬러맵 근사 (파랑→청록→초록→노랑→빨강). 점 색 = 시간 순서. */
function turbo(t) {
  const stops = [[48, 18, 59], [40, 187, 236], [162, 252, 60], [251, 128, 34], [122, 4, 3]];
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** 고해상도 캔버스 준비 — 폰에서 점이 뭉개지지 않게 devicePixelRatio 를 반영한다.
 *
 * 확대·이동 중에는 1초에 수십 번 다시 그린다. 그때마다 캔버스를 새로 만들면
 * 손가락을 따라오지 못하므로, 크기가 같으면 있던 판을 지워서 다시 쓴다.
 */
function setup(host, height) {
  const w = host.clientWidth || 320;
  const dpr = window.devicePixelRatio || 1;
  let cv = host.firstElementChild;
  if (!(cv instanceof HTMLCanvasElement)) {
    host.innerHTML = '';
    cv = document.createElement('canvas');
    host.appendChild(cv);
  }
  const W = Math.round(w * dpr), H = Math.round(height * dpr);
  if (cv.width !== W || cv.height !== H) {
    cv.width = W; cv.height = H;
    cv.style.width = '100%'; cv.style.height = height + 'px';
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, height);
  wireZoom(host);
  return { g, w, h: height };
}

/* ─────────────────── 확대·이동 ───────────────────
 * 산점도는 점이 빽빽해서 한 덩어리로 보인다. 손가락 두 개로 벌리면 확대되고,
 * 확대된 상태에서는 한 손가락으로 끌어 옮길 수 있다. 두 번 누르면 원래대로.
 * 확대가 1배일 때는 세로 넘김을 브라우저에 넘겨 화면 스크롤을 막지 않는다.
 */
const ZMAX = 8;

const zview = host => host._v || (host._v = { k: 1, tx: 0, ty: 0 });

/** 배율·이동을 허용 범위 안으로. 화면 밖으로 끌고 나가지 못하게 묶는다. */
function clampView(host, w, h) {
  const v = zview(host);
  v.k = Math.min(ZMAX, Math.max(1, v.k));
  const mx = (v.k - 1) * w / 2, my = (v.k - 1) * h / 2;
  v.tx = Math.min(mx, Math.max(-mx, v.tx));
  v.ty = Math.min(my, Math.max(-my, v.ty));
  host.style.touchAction = v.k > 1 ? 'none' : 'pan-y';
  return v;
}

function wireZoom(host) {
  if (host._wired) return;
  host._wired = true;
  host.style.touchAction = 'pan-y';
  const again = () => { if (host.redraw) host.redraw(host.clientHeight || 240); };
  const at = t => {
    const r = host.getBoundingClientRect();
    return { x: t.clientX - r.left, y: t.clientY - r.top };
  };
  // 확대 중심(손가락 사이 지점)이 제자리에 있도록 이동값을 다시 잡는다
  const zoomAt = (mx, my, k, from) => {
    const w = host.clientWidth || 1, h = host.clientHeight || 1;
    const v = zview(host);
    const px = (from.cx - w / 2 - from.tx) / from.k + w / 2;
    const py = (from.cy - h / 2 - from.ty) / from.k + h / 2;
    v.k = k;
    v.tx = mx - w / 2 - (px - w / 2) * k;
    v.ty = my - h / 2 - (py - h / 2) * k;
    clampView(host, w, h);
    again();
  };
  let pinch = null, drag = null, lastTap = 0;

  host.addEventListener('touchstart', e => {
    const v = zview(host);
    if (e.touches.length === 2) {
      const a = at(e.touches[0]), b = at(e.touches[1]);
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), k: v.k, tx: v.tx, ty: v.ty,
                cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
      drag = null;
      e.preventDefault();
    } else if (e.touches.length === 1) {
      const now = performance.now();
      if (now - lastTap < 320) {              // 두 번 누르면 원래 크기로
        v.k = 1; v.tx = 0; v.ty = 0;
        clampView(host, host.clientWidth || 1, host.clientHeight || 1);
        again();
        e.preventDefault();
      }
      lastTap = now;
      // 확대 전에는 끌기를 가로채지 않는다 — 화면이 안 내려가면 답답하다
      drag = v.k > 1 ? { p: at(e.touches[0]), tx: v.tx, ty: v.ty } : null;
      pinch = null;
    }
  }, { passive: false });

  host.addEventListener('touchmove', e => {
    const w = host.clientWidth || 1, h = host.clientHeight || 1;
    const v = zview(host);
    if (e.touches.length === 2 && pinch) {
      const a = at(e.touches[0]), b = at(e.touches[1]);
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch.d > 8) {
        zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2,
               Math.min(ZMAX, Math.max(1, pinch.k * d / pinch.d)), pinch);
      }
      e.preventDefault();
    } else if (e.touches.length === 1 && drag) {
      const p = at(e.touches[0]);
      v.tx = drag.tx + (p.x - drag.p.x);
      v.ty = drag.ty + (p.y - drag.p.y);
      clampView(host, w, h);
      again();
      e.preventDefault();
    }
  }, { passive: false });

  const end = () => { pinch = null; drag = null; };
  host.addEventListener('touchend', end);
  host.addEventListener('touchcancel', end);

  // PC — 휠로 확대
  host.addEventListener('wheel', e => {
    const v = zview(host), r = host.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    zoomAt(mx, my, Math.min(ZMAX, Math.max(1, v.k * Math.exp(-e.deltaY / 300))),
           { cx: mx, cy: my, k: v.k, tx: v.tx, ty: v.ty });
    e.preventDefault();
  }, { passive: false });
}

/** 확대된 만큼 좌표를 옮겨 주는 함수를 만든다 (점 크기·글자는 그대로 둔다). */
const zoomX = (f, v, w) => val => (f(val) - w / 2) * v.k + w / 2 + v.tx;
const zoomY = (f, v, h) => val => (f(val) - h / 2) * v.k + h / 2 + v.ty;

/** 확대 중이면 배율을 오른쪽 위에 적어 둔다 — 두 번 누르면 원래대로. */
function zoomBadge(g, host, w, h, padR, padT) {
  const v = zview(host);
  if (v.k <= 1.02) return;
  axisLabel(g, `×${v.k.toFixed(1)}`, w - padR - 4, padT + 10, '#ffffffaa', 'right');
}

function axisLabel(g, text, x, y, color = '#8B95A1', align = 'left') {
  g.fillStyle = color;
  g.font = '10px ui-monospace, monospace';
  g.textAlign = align;
  g.fillText(text, x, y);
}

/**
 * ① 회귀 산점도 (로그-로그). X=SPY 정규화, Y=종목 정규화.
 * 시간순 Turbo 점 + 회귀선 + ±1.5σ 밴드 + 현재 위치 ★.
 */
function regressionScatter(host, r, height = 240, marks = []) {
  const { g, w, h } = setup(host, height);
  const n = r.spyNorm.length;
  const PAD_L = 4, PAD_R = 44, PAD_T = 8, PAD_B = 18;

  let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity;
  for (const v of r.spyNorm) if (v > 0) { xLo = Math.min(xLo, v); xHi = Math.max(xHi, v); }
  for (const arr of [r.tickerNorm, r.bandUpper, r.bandLower])
    for (const v of arr) if (v > 0) { yLo = Math.min(yLo, v); yHi = Math.max(yHi, v); }
  if (!(xHi > xLo) || !(yHi > yLo)) return;

  const lxLo = Math.log10(xLo * 0.98), lxHi = Math.log10(xHi * 1.02);
  const lyLo = Math.log10(yLo * 0.88), lyHi = Math.log10(yHi * 1.18);
  const bx = v => PAD_L + (w - PAD_L - PAD_R) * (Math.log10(v) - lxLo) / (lxHi - lxLo);
  const by = v => PAD_T + (h - PAD_T - PAD_B) * (1 - (Math.log10(v) - lyLo) / (lyHi - lyLo));
  const view = clampView(host, w, h);
  const sx = zoomX(bx, view, w), sy = zoomY(by, view, h);

  // 확대하면 점이 축 자리까지 삐져나온다 → 그림 영역 안으로 자른다
  g.save();
  g.beginPath();
  g.rect(PAD_L, PAD_T, w - PAD_L - PAD_R, h - PAD_T - PAD_B);
  g.clip();

  // 밴드 (±1.5σ)
  const order = [...Array(n).keys()].sort((a, b) => r.spyNorm[a] - r.spyNorm[b]);
  g.beginPath();
  order.forEach((i, k) => k ? g.lineTo(sx(r.spyNorm[i]), sy(r.bandUpper[i]))
                            : g.moveTo(sx(r.spyNorm[i]), sy(r.bandUpper[i])));
  for (let k = n - 1; k >= 0; k--) {
    const i = order[k];
    g.lineTo(sx(r.spyNorm[i]), sy(r.bandLower[i]));
  }
  g.closePath();
  g.fillStyle = 'rgba(150,150,150,.2)';
  g.fill();

  // 회귀선
  g.beginPath();
  order.forEach((i, k) => k ? g.lineTo(sx(r.spyNorm[i]), sy(r.predicted[i]))
                            : g.moveTo(sx(r.spyNorm[i]), sy(r.predicted[i])));
  g.strokeStyle = '#ADBAC7'; g.lineWidth = 2; g.stroke();

  // 시간순 점
  for (let i = 0; i < n; i++) {
    if (!(r.spyNorm[i] > 0) || !(r.tickerNorm[i] > 0)) continue;
    g.beginPath();
    g.arc(sx(r.spyNorm[i]), sy(r.tickerNorm[i]), 3, 0, Math.PI * 2);
    g.fillStyle = turbo(i / (n - 1));
    g.fill();
  }

  // 매매한 날
  marks.forEach(m => {
    const i = m.i;
    if (i >= 0 && i < n && r.spyNorm[i] > 0 && r.tickerNorm[i] > 0)
      marker(g, sx(r.spyNorm[i]), sy(r.tickerNorm[i]), m.buy);
  });

  // 현재 위치 ★
  const li = n - 1;
  if (r.spyNorm[li] > 0 && r.tickerNorm[li] > 0) {
    star(g, sx(r.spyNorm[li]), sy(r.tickerNorm[li]), 8);
  }

  g.restore();

  // 축 — 우측에 y 값 몇 개, 아래에 SPY 안내
  [lyLo, (lyLo + lyHi) / 2, lyHi].forEach(l => {
    const v = Math.pow(10, l);
    axisLabel(g, v.toFixed(2), w - PAD_R + 4, sy(v) + 3);
  });
  axisLabel(g, 'SPY →', PAD_L, h - 4, '#ffffff88');
  axisLabel(g, `β ${r.beta.toFixed(2)}`, PAD_L, PAD_T + 10, '#ADBAC7');
  zoomBadge(g, host, w, h, PAD_R, PAD_T);
}

/** ② Z·M 궤적. X=Z 백분위, Y=M 백분위, 둘 다 0~100. 임계 20/40/60/80. */
function zmScatter(host, r, height = 240, marks = []) {
  const { g, w, h } = setup(host, height);
  const n = r.zPct.length;
  const PAD_L = 4, PAD_R = 44, PAD_T = 8, PAD_B = 18;
  const bx = v => PAD_L + (w - PAD_L - PAD_R) * v / 100;
  const by = v => PAD_T + (h - PAD_T - PAD_B) * (1 - v / 100);
  const view = clampView(host, w, h);
  const sx = zoomX(bx, view, w), sy = zoomY(by, view, h);

  g.save();
  g.beginPath();
  g.rect(PAD_L, PAD_T, w - PAD_L - PAD_R, h - PAD_T - PAD_B);
  g.clip();

  // 임계 격자
  g.strokeStyle = '#ffffff14'; g.lineWidth = 1;
  [20, 40, 60, 80].forEach(t => {
    g.beginPath(); g.moveTo(sx(t), PAD_T); g.lineTo(sx(t), h - PAD_B); g.stroke();
    g.beginPath(); g.moveTo(PAD_L, sy(t)); g.lineTo(w - PAD_R, sy(t)); g.stroke();
  });
  // 매수권(좌하) / 매도권(우상) 은은한 표시
  g.fillStyle = 'rgba(240,68,82,.07)';
  g.fillRect(sx(0), sy(40), sx(40) - sx(0), sy(0) - sy(40));
  g.fillStyle = 'rgba(49,130,246,.07)';
  g.fillRect(sx(60), sy(100), sx(100) - sx(60), sy(60) - sy(100));

  let last = -1;
  for (let i = 0; i < n; i++) {
    const z = r.zPct[i], m = r.mPct[i];
    if (z == null || m == null || Number.isNaN(z) || Number.isNaN(m)) continue;
    g.beginPath();
    g.arc(sx(z), sy(m), 3, 0, Math.PI * 2);
    g.fillStyle = turbo(i / (n - 1));
    g.fill();
    last = i;
  }
  marks.forEach(m => {
    const i = m.i, z = r.zPct[i], mm = r.mPct[i];
    if (i >= 0 && i < n && z != null && mm != null && !Number.isNaN(z) && !Number.isNaN(mm))
      marker(g, sx(z), sy(mm), m.buy);
  });
  if (last >= 0) star(g, sx(r.zPct[last]), sy(r.mPct[last]), 8);

  g.restore();

  axisLabel(g, 'Z(저평가 ←) →', PAD_L, h - 4, '#ffffff88');
  axisLabel(g, 'M ↑', PAD_L, PAD_T + 10, '#8B95A1');
  [0, 50, 100].forEach(v => axisLabel(g, String(v), w - PAD_R + 4, sy(v) + 3));
  zoomBadge(g, host, w, h, PAD_R, PAD_T);
}

/**
 * 매매 마커 — 안드로이드와 같은 모양: 채운 원 + 얇은 흰 테두리 + 흰 화살표.
 * 산점도와 시계열 차트가 **같은 그림**을 쓴다(app.js 의 markLayer 가 이걸 부른다).
 * 비율은 안드로이드 화면 실측값 — 캡처(배율 3.5)에서 원 지름 28px,
 * 화살표 16x23px, 기둥 폭 6px 였다(= 반지름 4px). 크기만 키워서 쓴다.
 */
const MARK_R = 5.6;   // 안드로이드 실측은 4 — 더 크게 해 달라 해서 1.4 배
const MARK = { ring: 0.1, head: 0.57, stem: 0.21, height: 1.64, headRatio: 0.52 };

function marker(g, cx, cy, buy, r = MARK_R) {
  // 원 + 흰 테두리
  g.beginPath();
  g.arc(cx, cy, r, 0, Math.PI * 2);
  g.fillStyle = buy ? '#DC2626' : '#2563EB';
  g.fill();
  g.lineWidth = r * MARK.ring;
  g.strokeStyle = '#fff';
  g.stroke();

  // 화살표 — 폰트 글자가 아니라 도형으로 그린다 (안드로이드와 같은 굵기)
  const h = r * MARK.height, hw = r * MARK.head, sw = r * MARK.stem;
  const s = buy ? 1 : -1;                  // 매수 ↑ · 매도 ↓
  const tip = cy - s * h / 2;              // 머리 꼭짓점
  const base = cy + s * h / 2;             // 기둥 끝
  const mid = tip + s * h * MARK.headRatio;  // 머리와 기둥이 만나는 높이
  g.beginPath();
  g.moveTo(cx, tip);
  g.lineTo(cx + hw, mid);
  g.lineTo(cx + sw, mid);
  g.lineTo(cx + sw, base);
  g.lineTo(cx - sw, base);
  g.lineTo(cx - sw, mid);
  g.lineTo(cx - hw, mid);
  g.closePath();
  g.fillStyle = '#fff';
  g.fill();
}

/** 현재 위치 별표 — 마젠타. */
function star(g, cx, cy, r) {
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 ? r * 0.45 : r;
    const a = -Math.PI / 2 + i * Math.PI / 5;
    const x = cx + rad * Math.cos(a), y = cy + rad * Math.sin(a);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.closePath();
  g.fillStyle = '#FF5FD2';
  g.fill();
  g.strokeStyle = '#fff'; g.lineWidth = 1; g.stroke();
}

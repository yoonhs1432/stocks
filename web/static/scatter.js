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

/** 고해상도 캔버스 준비 — 폰에서 점이 뭉개지지 않게 devicePixelRatio 를 반영한다. */
function setup(host, height) {
  const w = host.clientWidth || 320;
  const dpr = window.devicePixelRatio || 1;
  const cv = document.createElement('canvas');
  cv.width = w * dpr; cv.height = height * dpr;
  cv.style.width = '100%'; cv.style.height = height + 'px';
  host.innerHTML = '';
  host.appendChild(cv);
  const g = cv.getContext('2d');
  g.scale(dpr, dpr);
  return { g, w, h: height };
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
  const sx = v => PAD_L + (w - PAD_L - PAD_R) * (Math.log10(v) - lxLo) / (lxHi - lxLo);
  const sy = v => PAD_T + (h - PAD_T - PAD_B) * (1 - (Math.log10(v) - lyLo) / (lyHi - lyLo));

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

  // 축 — 우측에 y 값 몇 개, 아래에 SPY 안내
  [lyLo, (lyLo + lyHi) / 2, lyHi].forEach(l => {
    const v = Math.pow(10, l);
    axisLabel(g, v.toFixed(2), w - PAD_R + 4, sy(v) + 3);
  });
  axisLabel(g, 'SPY →', PAD_L, h - 4, '#ffffff88');
  axisLabel(g, `β ${r.beta.toFixed(2)}`, PAD_L, PAD_T + 10, '#ADBAC7');
}

/** ② Z·M 궤적. X=Z 백분위, Y=M 백분위, 둘 다 0~100. 임계 20/40/60/80. */
function zmScatter(host, r, height = 240, marks = []) {
  const { g, w, h } = setup(host, height);
  const n = r.zPct.length;
  const PAD_L = 4, PAD_R = 44, PAD_T = 8, PAD_B = 18;
  const sx = v => PAD_L + (w - PAD_L - PAD_R) * v / 100;
  const sy = v => PAD_T + (h - PAD_T - PAD_B) * (1 - v / 100);

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

  axisLabel(g, 'Z(저평가 ←) →', PAD_L, h - 4, '#ffffff88');
  axisLabel(g, 'M ↑', PAD_L, PAD_T + 10, '#8B95A1');
  [0, 50, 100].forEach(v => axisLabel(g, String(v), w - PAD_R + 4, sy(v) + 3));
}

/** 매매 마커 — 매수는 빨강 ↑, 매도는 파랑 ↓ (안드로이드와 같은 모양). */
function marker(g, cx, cy, buy, r = 7) {
  g.beginPath();
  g.arc(cx, cy, r, 0, Math.PI * 2);
  g.fillStyle = buy ? '#DC2626' : '#2563EB';
  g.fill();
  g.strokeStyle = '#fff'; g.lineWidth = 1; g.stroke();
  g.fillStyle = '#fff';
  g.font = `bold ${Math.round(r * 1.9)}px ui-monospace, monospace`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(buy ? '↑' : '↓', cx, cy + 0.5);
  g.textBaseline = 'alphabetic';
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

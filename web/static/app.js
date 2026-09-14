// 포트폴리오 화면 — 서버가 준 계좌 한 덩어리를 그린다.
// 계산은 전부 서버(원화 환산·정렬)에서 끝내고, 여기서는 표시 통화만 바꾼다.

// 보유 색 팔레트 — 안드로이드 앱 WeightPalette 와 같은 6색.
const PALETTE = ['#E0A24A', '#D9694E', '#CF5D7F', '#8A6FD0', '#4D8DF0', '#37A48C'];

let usdMode = localStorage.getItem('cur') === 'usd';
let data = null;

const $ = (s, r = document) => r.querySelector(s);

/** 원화 금액 → 표시 문자열. 달러 모드면 그 시점 환율로 나눈다. */
function money(krw) {
  if (usdMode) return '$' + (krw / data.rate).toLocaleString('en-US',
    { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return Math.round(krw).toLocaleString('ko-KR') + '원';
}

/** 부호를 통화기호 **앞**에 붙인다 (`-$12.34` / `-1,688,798원`). */
function signed(krw) {
  return (krw >= 0 ? '+' : '-') + money(Math.abs(krw));
}

/** 소수비율(0.0141) → "+1.41%". 토스는 손익률을 전부 소수비율로 준다. */
function pct(r) {
  return (r >= 0 ? '+' : '') + (r * 100).toFixed(2) + '%';
}

function cls(v) { return v > 0 ? 'up' : v < 0 ? 'down' : 'muted'; }

function qtyLabel(q) {
  // 미국 소수점 매매가 있어 필요한 자리까지만 보여준다
  return (Number.isInteger(q) ? q.toLocaleString('ko-KR')
    : String(parseFloat(q.toFixed(4)))) + '주';
}

function render() {
  const body = $('#body');
  body.innerHTML = '';
  if (!data) return;

  const node = $('#tpl-account').content.cloneNode(true);
  $('.acct', node).textContent = data.accountNo
    ? '•••••' + data.accountNo.slice(-4) : '';
  $('.total', node).textContent = money(data.totalKrw);

  const today = $('.today', node);
  today.textContent = `오늘 ${signed(data.dailyPnlKrw)} (${pct(data.dailyPnlRate)})`;
  today.className = 'today mono ' + cls(data.dailyPnlKrw);

  const pnl = $('.pnl', node);
  pnl.textContent = `평가손익 ${signed(data.pnlKrw)} (${pct(data.pnlRate)})`;
  pnl.className = 'pnl mono ' + cls(data.pnlKrw);

  $('.ev', node).textContent = money(data.evalKrw);
  $('.cash', node).textContent = money(data.cashKrw);
  $('.rate', node).textContent = data.rate.toLocaleString('ko-KR',
    { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  // 비중 파이 — 금액 내림차순이라 12시부터 큰 순서로 돈다
  const sum = data.items.reduce((a, b) => a + b.evalKrw, 0);
  const pie = $('.pie', node);
  if (sum > 0 && data.items.length) {
    let acc = 0;
    const stops = data.items.map((h, i) => {
      const from = acc / sum * 360;
      acc += h.evalKrw;
      const to = acc / sum * 360;
      return `${PALETTE[i % PALETTE.length]} ${from}deg ${to}deg`;
    });
    pie.style.background = `conic-gradient(${stops.join(',')})`;
  } else {
    pie.remove();
  }

  const list = $('.holdings', node);
  data.items.forEach((h, i) => {
    const row = $('#tpl-holding').content.cloneNode(true);
    $('.dot', row).style.background = PALETTE[i % PALETTE.length];
    $('.name', row).textContent = h.name || h.symbol;
    $('.eval', row).textContent = money(h.evalKrw);
    $('.qty', row).textContent = qtyLabel(h.quantity);
    const g = $('.gain', row);
    g.textContent = signed(h.pnlKrw);
    g.className = 'gain mono ' + cls(h.pnlKrw);
    const r = $('.rate', row);
    r.textContent = pct(h.pnlRate);
    r.className = 'rate mono ' + cls(h.pnlRate);
    list.appendChild(row);
  });

  body.appendChild(node);
  $('#stamp').textContent = '갱신 ' + new Date(data.at * 1000)
    .toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function load(force) {
  const st = $('#status');
  if (st) st.textContent = '불러오는 중…';
  try {
    const res = await fetch('/api/account' + (force ? '?force=true' : ''));
    const o = await res.json();
    if (!res.ok) {
      // 값을 지어내지 않고 실패를 그대로 보여준다
      $('#body').innerHTML = `<p class="err">⚠️ ${o.error || '조회 실패'}</p>`;
      return;
    }
    data = o;
    render();
  } catch (e) {
    $('#body').innerHTML = `<p class="err">⚠️ 서버에 연결하지 못했습니다 (${e.message})</p>`;
  }
}

$('#cur').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  usdMode = b.dataset.cur === 'usd';
  localStorage.setItem('cur', usdMode ? 'usd' : 'krw');
  document.querySelectorAll('#cur button').forEach(x =>
    x.classList.toggle('on', x === b));
  if (data) render();
});

$('#reload').addEventListener('click', () => load(true));

// 처음 로드 + 60초마다 조용히 갱신 (서버가 20초 캐시라 토스에는 그만큼만 나간다)
document.querySelectorAll('#cur button').forEach(x =>
  x.classList.toggle('on', (x.dataset.cur === 'usd') === usdMode));
load(false);
setInterval(() => load(false), 60000);

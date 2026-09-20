"""화면을 실제로 눌러 보는 검사 — 가짜 시세로 서버를 띄우고 브라우저로 조작한다.

    python test_ui.py            # 실패한 항목이 있으면 종료코드 1

`shot.py` 는 탭마다 첫 화면을 캡처할 뿐이라 **아무것도 누르지 않는다.** 그래서
"한국을 눌렀는데 버튼 색은 미국에 남아 있고 데이터만 바뀌는" 문제를 못 잡았다.
여기서는 세그먼트·탭·행을 실제로 눌러 ① 눌린 버튼에 칠이 옮겨 갔는지 ② 화면 내용이
실제로 바뀌었는지 두 가지를 같이 본다. 둘 중 하나만 맞아도 실패로 본다.

진짜 기록을 건드리지 않도록 QUANT_DATA 로 임시 폴더를 쓴다.
"""

from __future__ import annotations

import glob
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import DEFAULT_TICKERS      # 되돌릴 기준 목록

HERE = os.path.dirname(os.path.abspath(__file__))
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(("  ✅ " if ok else "  ❌ ") + name + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(f"{name}: {detail}")
    return ok


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def chromium() -> str | None:
    """샌드박스에 미리 깔린 크로미움. 없으면 Playwright 기본 경로에 맡긴다."""
    hit = sorted(glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome"))
    return hit[-1] if hit else None


def seed(data: str) -> None:
    """자산 추이 차트가 그려지도록 스냅샷·입금을 심는다.

    빈 폴더로 띄우면 기록이 1일뿐이라 차트가 아예 안 그려지고, 그러면 "그래프가 탭바에
    겹친다" 같은 문제를 못 본다. 실제 PC 에는 기록이 쌓여 있다.
    """
    import json, math
    from datetime import date, timedelta

    d0 = date.today() - timedelta(days=60)
    def items(i):
        return [{"s": "GDXU", "n": "GDXU", "q": 17, "a": 148.0, "p": 140 + i * 0.2,
                 "e": (140 + i * 0.2) * 17 * 1350, "g": (140 + i * 0.2 - 148) * 17 * 1350},
                {"s": "KORU", "n": "KORU", "q": 90, "a": 19.1, "p": 19 + i * 0.02,
                 "e": (19 + i * 0.02) * 90 * 1350, "g": (19 + i * 0.02 - 19.1) * 90 * 1350}]

    rows = [{"date": str(d0 + timedelta(days=i)),
             "krwEval": 35000 + i * 50, "usdEval": 8000 + i * 40 + math.sin(i / 6) * 300,
             "krwCash": 9770.0, "usdCash": 1841.0,
             "rate": 1350 + math.sin(i / 9) * 20, "pnlKrw": (i - 30) * 90000,
             "items": items(i)}
            for i in range(60)]
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(data, "snapshots.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)
    with open(os.path.join(data, "deposits.json"), "w", encoding="utf-8") as f:
        json.dump([{"date": str(d0), "krw": 12_000_000}], f)

    seed_repo(os.path.join(data, "repo"))


def seed_repo(root: str) -> str:
    """업데이트 버튼용 가짜 저장소 한 쌍 — work 는 한 칸 뒤, upstream 에 새 커밋.

    진짜 저장소에 git pull 을 걸면 작업 중인 코드가 딸려 올라가거나 충돌한다.
    """
    up, work = os.path.join(root, "upstream"), os.path.join(root, "work")
    git = lambda d, *a: subprocess.run(["git", *a], cwd=d, capture_output=True, text=True)
    os.makedirs(up, exist_ok=True)
    git(up, "init", "-q", "-b", "main")
    git(up, "config", "user.email", "t@t")
    git(up, "config", "user.name", "t")
    for msg in ("v1", "v2"):
        with open(os.path.join(up, "note.txt"), "w") as f:
            f.write(msg)
        git(up, "add", "-A")
        git(up, "commit", "-qm", msg)
    subprocess.run(["git", "clone", "-q", up, work], capture_output=True)
    git(work, "reset", "-q", "--hard", "HEAD~1")
    return work


def start_server(port: int, data: str) -> subprocess.Popen:
    # QUANT_CONFIG 도 임시 파일로 — 암호 교체 검사가 진짜 config.json 을 갈아 치우면
    # 폰에서 갑자기 못 들어오게 된다.
    env = {**os.environ, "QUANT_MOCK": "1", "QUANT_DATA": data,
           "QUANT_CONFIG": os.path.join(data, "config.json"),
           # 업데이트 버튼이 진짜 저장소를 당기지 않게 (감시 루프도 없으니 재시작도 안 한다)
           "QUANT_REPO": os.path.join(data, "repo", "work")}
    env.pop("QUANT_SUPERVISED", None)
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app",
                          "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                         cwd=HERE, env=env)
    for _ in range(100):
        if p.poll() is not None:
            raise SystemExit("서버가 바로 죽었습니다")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.3)
    p.kill()
    raise SystemExit("서버가 뜨지 않았습니다")


def seg_on(pg) -> str | None:
    """헤더 세그먼트에서 현재 칠해진 버튼의 글자."""
    b = pg.query_selector("#hdr-seg button.on")
    return b.inner_text().strip() if b else None


def state(pg, expr: str):
    """app.js 의 전역 S 를 읽는다 (top-level const 도 평가식에서는 보인다)."""
    return pg.evaluate(f"(() => {{ try {{ return {expr}; }} catch (e) {{ return '__ERR__' + e; }} }})()")


def run(pg, base: str, errs: list[str]) -> None:
    pg.goto(base + "/", wait_until="networkidle")
    pg.wait_for_selector("#body table tr", timeout=15000)

    # ── 비교: 미국/한국 세그먼트 ──
    print("\n[비교] 시장 세그먼트")
    check("처음엔 미국이 칠해져 있다", seg_on(pg) == "미국", f"칠={seg_on(pg)}")
    pg.click("#hdr-seg button:has-text('한국')")
    pg.wait_for_timeout(1500)
    check("한국을 누르면 칠이 한국으로 옮겨간다", seg_on(pg) == "한국", f"칠={seg_on(pg)}")
    kr_list = state(pg, "(S.rows||[]).map(r=>r.ticker)")
    check("한국을 누르면 한국 종목이 뜬다",
          state(pg, "S.market") == "KR" and "005930" in kr_list
          and not any(t in kr_list for t in ("TQQQ", "SOXL", "FNGU")),
          f"market={state(pg, 'S.market')} tickers={kr_list}")
    check("6자리 코드가 아니어도 보유 시장을 보고 국내로 분류한다",
          "SOLKR" in kr_list, f"국내 목록={kr_list}")
    pg.click("#hdr-seg button:has-text('미국')")
    pg.wait_for_timeout(1500)
    check("미국으로 되돌리면 칠도 같이 돌아온다", seg_on(pg) == "미국", f"칠={seg_on(pg)}")
    us_list = state(pg, "(S.rows||[]).map(r=>r.ticker)")
    check("미국 종목이 다시 뜬다",
          len(us_list) > 5 and not any(t in us_list for t in ("005930", "SOLKR")),
          f"tickers={us_list}")

    # ── 비교 → 분석: 행을 누르면 그 종목으로 ──
    print("\n[비교→분석] 행 누르기")
    name = pg.inner_text("#body table tr.row td.l")
    pg.click("#body table tr.row")
    pg.wait_for_timeout(2500)
    check("분석 탭으로 넘어간다", pg.inner_text("#title").strip() == "분석",
          pg.inner_text("#title"))
    check("탭바 칠도 분석으로 간다",
          pg.eval_on_selector("#tabs button.on", "b => b.dataset.tab") == "analysis")
    check("누른 종목이 열린다", (state(pg, "S.ticker") or "") in name, f"{state(pg, 'S.ticker')} / {name}")

    # ── 분석: 일봉/1분 세그먼트 ──
    print("\n[분석] 봉 주기 세그먼트")
    check("처음엔 일봉이 칠해져 있다", seg_on(pg) == "일봉", f"칠={seg_on(pg)}")
    got_min = []
    pg.on("request", lambda r: got_min.append(r.url) if "/api/minutes" in r.url else None)
    pg.click("#hdr-seg button:has-text('1분')")
    pg.wait_for_timeout(3000)
    check("1분을 누르면 칠이 옮겨간다", seg_on(pg) == "1분", f"칠={seg_on(pg)}")
    # 미리 받아 뒀으면 이 순간 요청이 없을 수도 있다 — 중요한 건 분봉이 그려졌는가다
    check("1분을 누르면 분봉이 그려진다",
          state(pg, "S.bar") == "1m" and state(pg, "!!(S.minutes && S.minutes.candles.length)") is True,
          f"bar={state(pg, 'S.bar')} 분봉={state(pg, '(S.minutes||{}).candles ? S.minutes.candles.length : 0')}")
    pg.click("#hdr-seg button:has-text('일봉')")
    pg.wait_for_timeout(2500)
    check("일봉으로 되돌아온다", seg_on(pg) == "일봉" and state(pg, "S.bar") == "1d",
          f"칠={seg_on(pg)} bar={state(pg, 'S.bar')}")

    # ── 분석: 시계열 ↔ 산점도 ──
    print("\n[분석] 산점도 전환")
    pg.click("#hdr-btn")
    pg.wait_for_timeout(2500)
    check("산점도를 누르면 캔버스가 그려진다", len(pg.query_selector_all("#body canvas")) >= 2,
          f"canvas={len(pg.query_selector_all('#body canvas'))}")
    check("버튼 글자가 시계열로 바뀐다", pg.inner_text("#hdr-btn").strip() == "시계열",
          pg.inner_text("#hdr-btn"))
    pg.click("#hdr-btn")
    pg.wait_for_timeout(2500)
    check("시계열로 돌아오면 봉 주기 세그먼트가 다시 보인다", seg_on(pg) in ("일봉", "1분"),
          f"칠={seg_on(pg)}")

    # ── 포트폴리오: 원/$ 토글 ──
    print("\n[포트폴리오] 통화 토글")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(3000)
    check("포트폴리오가 뜬다", "포트폴리오" in pg.inner_text("#title"))
    check("처음엔 원이 칠해져 있다", seg_on(pg) == "원", f"칠={seg_on(pg)}")
    krw_text = pg.inner_text("#body")
    pg.click("#hdr-seg button:has-text('$')")
    pg.wait_for_timeout(2500)
    check("$ 를 누르면 칠이 옮겨간다", seg_on(pg) == "$", f"칠={seg_on(pg)}")
    usd_text = pg.inner_text("#body")
    check("$ 를 누르면 금액 표기가 바뀐다", "$" in usd_text and usd_text != krw_text)
    pg.click("#hdr-seg button:has-text('원')")
    pg.wait_for_timeout(2500)
    check("원으로 되돌아온다", seg_on(pg) == "원" and state(pg, "S.usdMode") is False,
          f"칠={seg_on(pg)} usdMode={state(pg, 'S.usdMode')}")

    # ── 설정 ──
    print("\n[설정]")
    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2000)
    check("설정이 뜬다", "설정" in pg.inner_text("#title"))
    txt = pg.inner_text("#body")
    check("종목 목록이 보인다", "종목 관리" in txt and "SOXL" in txt)

    # 종목 추가/삭제까지 눌러 본다 (임시 데이터 폴더라 진짜 목록은 그대로다)
    pg.fill("#body input[placeholder='티커 또는 6자리 코드']", "AAPL")
    pg.click("#body .row2:has(input[placeholder='티커 또는 6자리 코드']) button")
    pg.wait_for_timeout(1500)
    added = check("종목을 추가하면 목록에 들어간다", "AAPL" in pg.inner_text("#body"))
    if added:
        pg.once("dialog", lambda d: d.accept())
        pg.click("#body .tk:has(b:text-is('AAPL')) button")
        pg.wait_for_timeout(1500)
        check("종목을 삭제하면 목록에서 빠진다", "AAPL" not in pg.inner_text("#body"))

    # 목록 통째로 바꾸기 — 다른 앱에서 쓰던 목록을 한 번에 옮기는 길
    before_n = len(pg.query_selector_all("#body .tk"))
    pg.fill("#body .tk-in", "AAA, BBB\nCCC 005930")
    pg.once("dialog", lambda d: d.accept())
    pg.click("#body button:has-text('통째로 저장')")
    pg.wait_for_timeout(2500)
    # 칩에는 이름도 같이 붙으므로 코드(b)만 본다
    chips = pg.evaluate("[...document.querySelectorAll('#body .tk b')].map(e => e.textContent)")
    check("붙여넣은 목록으로 통째로 바뀐다", chips == ["AAA", "BBB", "CCC", "005930"],
          f"{before_n}개 → {chips}")
    check("한국 종목은 눈에 띄게 표시된다",
          pg.query_selector("#body .tk.kr b") is not None)

    # 코드=이름 으로 이름까지 한 번에
    pg.fill("#body .tk-in", "005930=내가붙인이름, TQQQ")
    pg.once("dialog", lambda d: d.accept())
    pg.click("#body button:has-text('통째로 저장')")
    pg.wait_for_timeout(2500)
    check("코드=이름 으로 이름도 같이 저장된다",
          "내가붙인이름" in pg.inner_text("#body"), pg.inner_text("#body .tklist"))
    # 뒤 검사들이 기본 종목을 쓰므로 목록을 되돌려 놓는다
    pg.fill("#body .tk-in", ", ".join(DEFAULT_TICKERS))
    pg.once("dialog", lambda d: d.accept())
    pg.click("#body button:has-text('통째로 저장')")
    pg.wait_for_timeout(2500)
    check("목록을 되돌려 놓았다", len(pg.query_selector_all("#body .tk")) == len(DEFAULT_TICKERS),
          f"{len(pg.query_selector_all('#body .tk'))}개")

    # ── 탭을 한 바퀴 돌아도 칠이 따라온다 ──
    print("\n[탭바]")
    for tab, title in [("compare", "비교"), ("analysis", "분석"),
                       ("portfolio", "포트폴리오"), ("settings", "설정")]:
        pg.click(f"#tabs button[data-tab='{tab}']")
        pg.wait_for_timeout(1200)
        check(f"{title} 탭",
              pg.eval_on_selector("#tabs button.on", "b => b.dataset.tab") == tab
              and title in pg.inner_text("#title"))

    # ── 고정 막대(헤더·탭바)를 그래프가 덮지 않는가 ──
    # 차트 라이브러리가 캔버스에 z-index 를 박아 둬서, 막대에 z-index 가 없으면 스크롤
    # 중인 그래프가 막대 **위에** 그려진다. 바닥까지 내리면 안 보이므로 중간에서 본다.
    # ── 화면 폭 안에 들어오는가 ──
    print("\n[폭] 가로로 넘치지 않는가")
    for tab, title in [("compare", "비교"), ("portfolio", "포트폴리오"), ("settings", "설정")]:
        pg.click(f"#tabs button[data-tab='{tab}']")
        pg.wait_for_timeout(2500)
        over = pg.evaluate("""() => {
          const bad = [];
          document.querySelectorAll('#body *').forEach(e => {
            const r = e.getBoundingClientRect();
            if (r.right > innerWidth + 1 && e.closest('.tchips') === null)
              bad.push(`${e.tagName}.${(e.className||'').toString().slice(0,14)}`);
          });
          return {bad: [...new Set(bad)].slice(0, 4),
                  page: document.scrollingElement.scrollWidth > innerWidth};
        }""")
        check(f"{title} 탭이 가로로 넘치지 않는다", not over["bad"] and not over["page"], str(over))

    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(2500)
    cols = pg.evaluate("""() => {
      const th = [...document.querySelectorAll('#body table th')];
      const last = th.at(-1).getBoundingClientRect();
      return {끝열: th.at(-1).textContent.trim(), 오른쪽: Math.round(last.right), 화면: innerWidth};
    }""")
    check("맨 끝 M 열이 화면 안에 다 보인다", cols["오른쪽"] <= cols["화면"], str(cols))

    print("\n[겹침] 스크롤 중 헤더·탭바를 덮는 것이 없는가")
    probe = """(sel) => {
      const r = document.querySelector(sel).getBoundingClientRect();
      const y = Math.round(r.top + r.height / 2);
      const bad = [];
      for (const x of [40, 140, 250, 360]) {
        const e = document.elementFromPoint(x, y);
        if (e && !e.closest(sel)) bad.push(`${x}px=${e.tagName}.${(e.className || '')}`.slice(0, 40));
      }
      return bad;
    }"""
    for tab, title in [("analysis", "분석"), ("portfolio", "포트폴리오")]:
        pg.click(f"#tabs button[data-tab='{tab}']")
        pg.wait_for_timeout(3500)
        top = pg.evaluate("document.scrollingElement.scrollHeight - innerHeight")
        for frac in (0.35, 0.7, 1.0):
            pg.evaluate(f"window.scrollTo(0, {int(top * frac)})")
            pg.wait_for_timeout(600)
            for sel, what in [("#tabs", "탭바"), ("header", "헤더")]:
                bad = pg.evaluate(probe, sel)
                check(f"{title} {int(frac * 100)}% 지점 — {what}를 덮는 것 없음",
                      not bad, " / ".join(bad))

    # ── 오류가 났을 때 되살아날 수 있는가 ──
    # ── 탭을 옮겼을 때 이전 탭 화면이 남지 않는가 ──
    # ── 국내 종목 이름이 저절로 붙는가 ──
    print("\n[이름] 국내 코드가 이름으로 보이는가")
    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(1500)
    pg.click("#hdr-seg button:has-text('한국')")
    pg.wait_for_timeout(3000)
    pg.click("#hdr-btn")                     # 목록을 받아 오면서 종목 이름도 받아 둔다
    pg.wait_for_timeout(6000)
    pg.click("#hdr-btn")
    pg.wait_for_timeout(5000)
    names = pg.evaluate("(S.rows||[]).map(r => r.name)")
    codes = pg.evaluate("(S.rows||[]).map(r => r.ticker)")
    import re as _re2
    bare = [n for n in names if _re2.fullmatch(r"\d{6}", n or "")]
    check("국내 종목이 코드가 아니라 이름으로 보인다",
          names and not bare and any("국내 ETF" in (n or "") for n in names),
          f"{list(zip(codes, names))}")
    pg.click("#hdr-seg button:has-text('미국')")
    pg.wait_for_timeout(3000)

    # ── 뒤로가기 ──
    print("\n[뒤로가기] 직전 탭으로 돌아가는가")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(2500)
    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2000)
    pg.go_back()
    pg.wait_for_timeout(2500)
    check("뒤로 가면 직전 탭(포트폴리오)으로", "포트폴리오" in pg.inner_text("#title"),
          pg.inner_text("#title"))
    pg.go_back()
    pg.wait_for_timeout(2500)
    check("한 번 더 뒤로 가면 비교로", "비교" in pg.inner_text("#title"),
          pg.inner_text("#title"))
    check("뒤로 가도 화면이 비지 않는다", pg.query_selector("#body table.cmp") is not None)

    print("\n[탭] 옮기면 이전 내용이 지워지는가")
    pg.evaluate("S.analysis = null; S.account = null; S.settings = null")
    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(2000)
    for tab, title in [("analysis", "분석"), ("portfolio", "포트폴리오"), ("settings", "설정")]:
        pg.evaluate("S.analysis = null; S.account = null; S.settings = null")
        pg.click("#tabs button[data-tab='compare']")
        pg.wait_for_timeout(1200)
        pg.click(f"#tabs button[data-tab='{tab}']")
        pg.wait_for_timeout(120)
        check(f"{title} 으로 옮기면 비교 표가 바로 사라진다",
              pg.query_selector("#body table.cmp") is None,
              pg.inner_text("#body")[:40])
        pg.wait_for_timeout(3000)

    # ── 분석을 미리 받아 두는가 ──
    # ── 탭을 빨리 누르면 늦게 온 화면이 덮어쓰지 않는가 ──
    print("\n[탭 연타] 늦게 온 탭 내용이 딴 탭 자리에 그려지지 않는가")
    pg.evaluate("""() => {
      const f = window.fetch;                      // 포폴만 2초 늦춘다
      window.fetch = (u, o) => {
        const slow = String(u).includes('/api/account') || String(u).includes('/api/snapshots');
        return new Promise(r => setTimeout(() => r(f(u, o)), slow ? 2000 : 0));
      };
      S.account = null; S.snaps = null;
    }""")
    pg.click("#tabs button[data-tab='analysis']")
    pg.wait_for_timeout(700)
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(300)
    pg.click("#tabs button[data-tab='analysis']")
    pg.wait_for_timeout(4000)
    body3 = pg.inner_text("#body")
    check("분석·포폴·분석을 빨리 눌러도 분석 화면이 남는다",
          pg.evaluate("document.querySelector('#tabs button.on').dataset.tab") == "analysis"
          and "총자산" not in body3,
          body3[:50].replace("\n", " "))
    pg.reload(wait_until="networkidle")             # fetch 를 원래대로
    pg.wait_for_timeout(3000)

    # ── 늦게 온 응답이 지금 화면과 어긋나지 않는가 (같은 유형 전수) ──
    print("\n[늦은 응답] 지금 화면과 어긋나지 않는가")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(3500)
    pg.evaluate("""() => {
      const f = window.__of || window.fetch; window.__of = f;    // 달러 쪽만 늦춘다
      window.fetch = (u, o) => new Promise(r =>
        setTimeout(() => r(f(u, o)), String(u).includes('usd=true') ? 1800 : 0));
    }""")
    pg.click("#hdr-seg button:has-text('$')")
    pg.wait_for_timeout(250)
    pg.click("#hdr-seg button:has-text('원')")
    pg.wait_for_timeout(3500)
    st = pg.evaluate("({usd: S.usdMode, first: S.snaps ? Math.round(S.snaps.total[0]) : 0})")
    check("원/$ 를 연달아 눌러도 그래프가 지금 통화와 맞는다",
          st["usd"] is False and st["first"] > 1_000_000, str(st))
    pg.evaluate("window.fetch = window.__of")

    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2500)
    pg.evaluate("""() => {
      const f = window.__of || window.fetch; window.__of = f;    // 설정 다시 읽기를 늦춘다
      window.fetch = (u, o) => new Promise(r =>
        setTimeout(() => r(f(u, o)), String(u).endsWith('/api/settings') ? 1800 : 0));
    }""")
    pg.once("dialog", lambda d: d.accept())
    pg.click("#body .tk button")                  # 종목 하나 삭제 → 설정 다시 읽기
    pg.wait_for_timeout(200)
    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(3500)
    check("설정에서 저장하고 바로 탭을 옮겨도 설정 화면이 덮지 않는다",
          "종목 관리" not in pg.inner_text("#body"),
          pg.inner_text("#body")[:40].replace("\n", " "))
    pg.evaluate("window.fetch = window.__of")
    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2000)

    print("\n[미리받기] 종목을 누르기 전에 받아 두는가")
    pg.evaluate("anCache.clear()")
    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(1500)
    pg.click("#hdr-btn")          # 새로고침 → 비교를 다시 받으면 미리받기가 돈다
    pg.wait_for_timeout(9000)
    n = pg.evaluate("anCache.size")
    total = pg.evaluate("(S.rows||[]).length")
    check("목록 종목 대부분을 미리 받아 둔다", n >= max(3, total - 1), f"{n}/{total}개")

    pg.click("#tabs button[data-tab='analysis']")
    pg.wait_for_timeout(2500)
    t0 = time.time()
    pg.evaluate("document.querySelectorAll('#body .tchips button')[4].click()")
    pg.wait_for_function("document.querySelectorAll('#body .ch-wrap canvas').length > 0", timeout=15000)
    dt = time.time() - t0
    check("미리 받아 둔 종목은 기다림 없이 뜬다(1초 이내)", dt < 1.0, f"{dt:.2f}초")

    # ── 시장 전환이 빠른가 ──
    # ── 켜 두면 무거워지지 않는가 ──
    print("\n[누수] 틱이 돌아도 차트가 쌓이지 않는가")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(3000)
    n0 = pg.evaluate("charts.length")
    pg.evaluate("S.settings.tickSeconds = 1; startTicks()")
    pg.wait_for_timeout(12000)
    n1 = pg.evaluate("charts.length")
    check("포트폴리오를 켜 둬도 차트가 안 쌓인다", n1 <= n0, f"{n0} → {n1}개")
    check("틱이 돌아도 숫자는 갱신된다", "총자산" in pg.inner_text("#body"))
    pg.evaluate("S.settings.tickSeconds = 10; startTicks()")

    # ── 화면을 내려놓으면 요청을 멈추는가 ──
    print("\n[절전] 화면을 내려놓으면 요청이 멈추는가")
    hits = []
    handler = lambda r: hits.append(r.url) if "/api/prices" in r.url else None
    pg.on("request", handler)
    pg.evaluate("S.settings.tickSeconds = 1; startTicks()")
    pg.wait_for_timeout(3000)
    before = len(hits)
    pg.evaluate("""() => {
      Object.defineProperty(document, 'hidden', { value: true, configurable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    }""")
    hits.clear()
    pg.wait_for_timeout(4000)
    check("내려놓으면 시세 요청이 멈춘다", before > 0 and len(hits) == 0,
          f"켜져 있을 때 {before}건 → 내려놓고 {len(hits)}건")
    pg.evaluate("""() => {
      Object.defineProperty(document, 'hidden', { value: false, configurable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    }""")
    pg.wait_for_timeout(3000)
    check("돌아오면 다시 받는다", len(hits) > 0, f"{len(hits)}건")
    pg.remove_listener("request", handler)
    pg.evaluate("S.settings.tickSeconds = 10; startTicks()")

    print("\n[전환] 미국↔한국이 기다림 없이 바뀌는가")
    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(2000)
    times = []
    for target in ("한국", "미국", "한국"):
        t0 = time.time()
        pg.click(f"#hdr-seg button:has-text('{target}')")
        pg.wait_for_function("(S.rows||[]).length > 0", timeout=20000)
        times.append(time.time() - t0)
        pg.wait_for_timeout(600)
    check("한 번 본 시장은 즉시 바뀐다(0.5초 이내)", max(times[1:]) < 0.5,
          " / ".join(f"{t:.2f}초" for t in times))
    check("전환 뒤 그 시장 종목이 보인다",
          pg.query_selector("#body table tr.row") is not None)
    pg.click("#hdr-seg button:has-text('미국')")      # 뒤 검사들은 미국 목록을 쓴다
    pg.wait_for_timeout(1500)

    print("\n[오류] 막히지 않고 다시 시도할 수 있는가")
    pg.route("**/api/analysis*", lambda r: r.fulfill(
        status=502, content_type="application/json",
        body='{"error": "허용되지 않은 IP 입니다 (access_denied)"}'))
    pg.evaluate("localStorage.setItem('tab','analysis')")
    pg.goto(base + "/", wait_until="networkidle")
    pg.wait_for_timeout(3000)
    check("분석 오류에 서버가 준 이유가 보인다", "허용되지 않은 IP" in pg.inner_text("#body"),
          pg.inner_text("#body")[:60])
    check("분석 오류에도 종목 칩이 남는다", len(pg.query_selector_all("#body .tchips button")) > 3,
          f"칩 {len(pg.query_selector_all('#body .tchips button'))}개")
    check("분석 오류에 다시 시도 버튼이 있다",
          bool(pg.query_selector("#body button:has-text('다시 시도')")))
    pg.unroute("**/api/analysis*")
    pg.click("#body button:has-text('다시 시도')")
    pg.wait_for_timeout(3500)
    check("다시 시도를 누르면 복구된다", bool(pg.query_selector("#body .ch-wrap canvas")),
          pg.inner_text("#body")[:60])

    pg.route("**/api/compare*", lambda r: r.abort())
    pg.evaluate("localStorage.setItem('tab','compare')")
    pg.goto(base + "/")
    pg.wait_for_timeout(2500)
    txt = pg.inner_text("#body")
    check("서버가 안 잡히면 한국어로 알려 준다", "연결할 수 없습니다" in txt and "Failed to fetch" not in txt,
          txt[:60])
    pg.unroute("**/api/compare*")

    errs.clear()        # 위에서 일부러 낸 502·연결 끊김은 콘솔에 남는 게 정상이다

    # ── 실수로 지워지지 않는가 ──
    print("\n[삭제] 확인 없이 지워지지 않는가")
    pg.goto(base + "/", wait_until="networkidle")
    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2500)
    before = pg.inner_text("#body")
    pg.once("dialog", lambda d: d.dismiss())          # 취소를 누른 상황
    pg.click("#body .row2:has-text('+12,000,000원') button:has-text('삭제')")
    pg.wait_for_timeout(1500)
    check("입금 삭제는 확인을 먼저 묻는다(취소하면 남는다)",
          "12,000,000원" in pg.inner_text("#body"))
    pg.once("dialog", lambda d: d.accept())
    pg.click("#body .row2:has-text('+12,000,000원') button:has-text('삭제')")
    pg.wait_for_timeout(1500)
    check("확인을 누르면 지워진다", "+12,000,000원" not in pg.inner_text("#body"))
    # 뒤 검사(원금 선·범례)가 입금 기록을 쓰므로 되돌려 둔다
    pg.fill("#body .dep-in", "2026-07-22 12,000,000")
    pg.click("#body .dep-row button:has-text('추가')")
    pg.wait_for_timeout(2000)
    check("되돌려 놓았다", "12,000,000" in pg.inner_text("#body"))

    # ── 화면이 기억하는 것들 ──
    print("\n[기억] 확대 구간·기준 시각·정렬")
    pg.evaluate("localStorage.setItem('tab','analysis'); localStorage.removeItem('range-1d')")
    pg.goto(base + "/", wait_until="networkidle")
    pg.wait_for_timeout(3500)
    pg.mouse.move(200, 400)
    pg.mouse.wheel(0, -400)
    pg.wait_for_timeout(1200)
    saved = pg.evaluate("localStorage.getItem('range-1d')")
    check("확대하면 구간이 저장된다", bool(saved), str(saved))
    pg.click("#body .tchips button:nth-child(3)")
    pg.wait_for_timeout(3500)
    after = pg.evaluate("localStorage.getItem('range-1d')")
    check("종목을 바꿔도 같은 구간으로 열린다", saved == after, f"{saved} → {after}")
    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(3500)
    check("앱을 껐다 켜도 구간이 남는다",
          pg.evaluate("localStorage.getItem('range-1d')") == saved)

    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(3000)
    check("비교에 언제 기준 숫자인지 적혀 있다", "조회" in pg.inner_text("#body .stamp"),
          pg.inner_text("#body").splitlines()[-1] if pg.inner_text("#body") else "")
    pg.evaluate("document.querySelectorAll('#body table th')[0].click()")
    pg.wait_for_timeout(600)
    first = pg.inner_text("#body table tr.row").split("\t")[0]
    check("이름 정렬은 첫 클릭에 ㄱ→ㅎ", first.startswith("AVXX") or first < "F",
          f"1등={first}")

    # ── 잘못 눌리지 않는가 ──
    print("\n[조작]")
    # 표 행은 일부러 촘촘하게 둔다(한 화면에 많이 보이게) — 버튼류만 손가락 크기로
    small = pg.evaluate("""() => [...document.querySelectorAll('#hdr-seg button, #hdr-btn, #body .tchips button, #tabs button')]
        .map(e => e.getBoundingClientRect())
        .filter(r => r.height > 2 && r.height < 38).length""")
    check("버튼류가 손가락 크기(38px 이상)", small == 0, f"작은 것 {small}개")
    rowh = pg.evaluate("""() => {
      const r = document.querySelectorAll('#body table tr.row');
      return r.length > 1 ? Math.round(r[1].getBoundingClientRect().top - r[0].getBoundingClientRect().top) : 0;
    }""")
    check("비교 표 행 간격이 촘촘하다(24~36px)", 24 <= rowh <= 36, f"{rowh}px")

    reqs = []
    # force=true 는 새로고침이 보내는 것뿐이다. 반대 시장 미리받기는 세지 않는다.
    pg.on("request", lambda r: reqs.append(r.url) if "/api/compare" in r.url and "force=true" in r.url else None)
    for _ in range(4):
        pg.click("#hdr-btn", force=True)
        pg.wait_for_timeout(120)
    pg.wait_for_timeout(3000)
    check("새로고침을 연타해도 요청은 한 번", len(reqs) <= 1, f"{len(reqs)}번")

    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2000)
    pg.evaluate("window.scrollTo(0, 600)")
    pg.wait_for_timeout(400)
    pg.click("#tabs button[data-tab='compare']")
    pg.wait_for_timeout(2000)
    check("탭을 바꾸면 맨 위부터 보인다", pg.evaluate("Math.round(scrollY)") == 0,
          f"scrollY={pg.evaluate('Math.round(scrollY)')}")

    pg.click("#tabs button[data-tab='analysis']")
    pg.wait_for_timeout(3000)
    chip = pg.evaluate("""() => {
      const on = document.querySelector('#body .tchips button.on');
      if (!on) return null;
      const box = on.closest('.tchips').getBoundingClientRect(), r = on.getBoundingClientRect();
      return r.left >= box.left - 2 && r.right <= box.right + 2;
    }""")
    check("고른 종목 칩이 화면 안에 보인다", chip is True, str(chip))

    # ── 칩을 연달아 누를 때 엉뚱한 종목이 그려지지 않는가 ──
    # ── 1분봉도 미리 받아 두는가 ──
    print("\n[분봉] 일봉↔1분 전환이 빠른가")
    pg.click("#tabs button[data-tab='analysis']")
    pg.wait_for_timeout(3500)
    pg.wait_for_timeout(6000)          # 미리받기가 분봉까지 받을 시간
    t0 = time.time()
    pg.click("#hdr-seg button:has-text('1분')")
    pg.wait_for_function("S.bar === '1m' && document.querySelectorAll('#body .ch-wrap canvas').length > 0",
                         timeout=20000)
    dt = time.time() - t0
    check("1분봉 전환이 1초 이내", dt < 1.0, f"{dt:.2f}초")
    pg.click("#hdr-seg button:has-text('일봉')")
    pg.wait_for_timeout(2500)

    print("\n[연타] 늦게 온 옛 응답이 화면을 덮지 않는가")
    pg.click("#tabs button[data-tab='analysis']")
    pg.wait_for_timeout(3000)
    pg.evaluate("""() => {
      const orig = window.fetch;                 // 먼저 누른 종목만 3초 늦춘다
      window.fetch = (u, o) => {
        const d = String(u).includes('TQQQ') ? 3000 : 0;
        return new Promise(r => setTimeout(() => r(orig(u, o)), d));
      };
    }""")
    pg.click("#body .tchips button:has-text('TQQQ')")
    pg.wait_for_timeout(400)
    pg.click("#body .tchips button:has-text('SOXL')")
    pg.wait_for_timeout(6000)
    shown = pg.evaluate("document.querySelector('#body .anl-head .tk')?.textContent")
    chip = pg.evaluate("document.querySelector('#body .tchips button.on')?.textContent")
    check("칩과 차트의 종목이 같다", shown == chip == "SOXL", f"칩={chip} 차트={shown}")
    pg.reload(wait_until="networkidle")          # fetch 를 원래대로
    pg.wait_for_timeout(3000)

    # ── 포트폴리오·설정에 새로 넣은 것들 ──
    # ── 지난 날의 보유 내역이 남아 있는가 ──
    print("\n[기록] 그날 무엇을 들고 있었는지 볼 수 있는가")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(3000)
    pg.click("#body .ch-title:has-text('기록')")
    pg.wait_for_timeout(2500)
    check("기록 표가 열린다", pg.query_selector("#body table.hist") is not None,
          pg.inner_text("#body")[-80:].replace("\n", " "))
    dates = pg.evaluate("[...document.querySelectorAll('#body .hist-date option')].map(o => o.value)")
    check("날짜를 고를 수 있다", len(dates) > 10, f"{len(dates)}일")
    first_rows = pg.evaluate("[...document.querySelectorAll('#body table.hist tr.row')].map(r => r.innerText.replace(/\\n/g,' '))")
    check("그날 보유 종목이 표에 나온다", len(first_rows) >= 2, str(first_rows[:2]))
    pg.select_option("#body .hist-date", dates[-1])      # 가장 오래된 날
    pg.wait_for_timeout(2000)
    old_rows = pg.evaluate("[...document.querySelectorAll('#body table.hist tr.row')].map(r => r.innerText.replace(/\\n/g,' '))")
    check("날짜를 바꾸면 그날 값으로 바뀐다", old_rows and old_rows != first_rows,
          f"{first_rows[:1]} vs {old_rows[:1]}")
    check("종목별 평가금액 추이 그래프가 있다",
          pg.evaluate("document.querySelectorAll('#body .ch-wrap canvas').length") >= 2)

    print("\n[추가된 정보]")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(3500)
    body = pg.inner_text("#body")
    check("보유 종목에 비중 %가 보인다", "%" in body and "비중" in body, body[:60])
    leg = pg.inner_text("#body .legend")
    check("총자산 옆에 색 범례가 있다",
          all(k in leg for k in ("평가금액", "예수금", "원금")), leg.replace("\n", " ")[:60])
    ax = pg.evaluate("[...document.querySelectorAll('#body .axrow')].map(e => e.innerText.replace(/\\n/g, ' '))")
    check("자산 그래프 아래 시작일·기록일수·끝일이 있다",
          len(ax) >= 2 and "기록" in ax[0], str(ax[:2]))
    pnl = pg.inner_text("#body .pnl-big")
    check("평가손익에 현재값과 원금 대비가 있다",
          "현재" in pnl and "원금 대비" in pnl, pnl.replace("\n", " "))
    pie = pg.evaluate("""() => {
      const t = [...document.querySelectorAll('#body .pie-wrap text')].map(e => e.textContent);
      return t;
    }""")
    import re as _re
    check("파이 조각에 종목명과 %가 적혀 있다",
          len(pie) >= 3 and all(_re.match(r"^\S+\d+\.\d%$", x) for x in pie),
          str(pie[:6]))

    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2500)
    # 입금 여러 건을 한 번에
    pg.fill("#body .dep-in", "2026-09-01 13,789,303\n2026-09-04 10,728,849")
    pg.click("#body .dep-row button:has-text('추가')")
    pg.wait_for_timeout(2500)
    body2 = pg.inner_text("#body")
    check("입금을 여러 건 한 번에 넣을 수 있다",
          "13,789,303" in body2 and "10,728,849" in body2, body2[:120].replace("\n", " "))
    sums = pg.evaluate("""() => {
      const d = S.settings.deposits || [];
      return {합계: Math.round(S.settings.principal),
              더한값: Math.round(d.reduce((x, r) => x + r.krw, 0)), 건수: d.length};
    }""")
    check("입금 합계가 기록의 합과 같다",
          sums["합계"] == sums["더한값"] and sums["건수"] >= 3, str(sums))
    pg.fill("#body .dep-in", "엉터리 줄")
    pg.click("#body .dep-row button:has-text('추가')")
    pg.wait_for_timeout(1500)
    check("형식이 틀리면 알려 준다", "이렇게 적어 주세요" in pg.inner_text("#body"),
          pg.inner_text("#body .msg"))

    check("입금 날짜가 달력 입력이다",
          pg.eval_on_selector("#body input[type='date']", "e => e.type") == "date")
    check("접속 암호 칸이 있다", "접속 암호" in pg.inner_text("#body"))
    check("미리 받기를 끌 수 있다", pg.query_selector("#body .pre-sel") is not None)
    pg.select_option("#body .pre-sel", "0")
    pg.wait_for_timeout(1500)
    check("끄면 설정에 남는다", pg.evaluate("S.settings.prefetch") is False,
          str(pg.evaluate("S.settings.prefetch")))
    pg.select_option("#body .pre-sel", "1")
    pg.wait_for_timeout(1500)
    pg.click("#body button:has-text('보기')")
    pg.wait_for_timeout(400)
    shown = pg.inner_text("#body")
    check("보기를 누르면 암호가 나온다", "•••" not in shown.split("접속 암호")[1][:40],
          shown.split("접속 암호")[1][:40])
    old_tok = pg.evaluate("S.settings.accessToken")
    ok_all = lambda d: d.accept()
    pg.on("dialog", ok_all)             # 확인 → 새 암호 알림, 두 번 뜬다
    pg.click("#body button:has-text('새로 만들기')")
    pg.wait_for_timeout(2500)
    pg.remove_listener("dialog", ok_all)
    check("암호를 새로 만들 수 있다", pg.evaluate("S.settings.accessToken") != old_tok,
          "그대로였다")
    check("현재 버전이 보인다", "현재 버전" in pg.inner_text("#body"))

    # ── 새 버전 알림 띠 ──
    print("\n[알림] 새 버전이 올라오면 띠로 알려 주는가")
    pg.evaluate("localStorage.removeItem('skipVer')")
    pg.evaluate("checkUpdate()")
    pg.wait_for_timeout(2500)
    bar = pg.query_selector("#newver")
    check("새 버전이 있으면 띠가 뜬다", bar is not None and "새 버전" in bar.inner_text(),
          bar.inner_text() if bar else "안 뜸")
    if bar:
        check("무엇이 바뀌는지 적혀 있다", "v2" in bar.inner_text(), bar.inner_text())
        pg.click("#newver .nv-x")
        pg.wait_for_timeout(500)
        check("✕ 를 누르면 그 버전은 다시 안 띄운다",
              pg.query_selector("#newver") is None)
        pg.evaluate("checkUpdate()")
        pg.wait_for_timeout(1500)
        check("넘어간 버전은 다시 확인해도 안 뜬다", pg.query_selector("#newver") is None)
        pg.evaluate("localStorage.removeItem('skipVer'); checkUpdate()")
        pg.wait_for_timeout(1500)
        check("넘어가기를 지우면 다시 뜬다", pg.query_selector("#newver") is not None)
    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2000)
    pg.click("#body button:has-text('업데이트 받기')")
    pg.wait_for_timeout(2500)
    got = pg.inner_text("#body")
    check("업데이트 버튼이 새 코드를 받아 온다", "받았습니다" in got,
          [l for l in got.splitlines() if "⚠" in l or "받" in l][:2])
    check("감시 루프가 없으면 직접 켜라고 알려 준다", "다시 켜야" in got,
          [l for l in got.splitlines() if "받" in l][:2])
    pg.click("#body button:has-text('업데이트 받기')")
    pg.wait_for_timeout(2500)
    check("두 번째는 이미 최신이라고 한다", "이미 최신" in pg.inner_text("#body"))

    dl = pg.evaluate("""async () => {
      const r = await fetch('/api/backup');
      const o = await r.json();
      return o && o.files ? Object.keys(o.files).length : 0;
    }""")
    check("백업에 되살릴 수 없는 기록이 다 들어 있다", dl >= 7, f"파일 {dl}개")

    # ── 터널을 통해 들어온 요청이 암호를 건너뛰지 않는가 ──
    # 터널은 PC 안에서 127.0.0.1 로 붙는다. "PC 자신은 통과" 규칙이 그대로 걸리면
    # 인터넷에서 온 사람이 암호 없이 계좌를 본다.
    print("\n[보안] 터널로 들어온 요청은 암호를 묻는가")
    # 앞선 검사에서 암호 쿠키가 심겼다. 그대로 두면 "통과"가 쿠키 덕인지 규칙 덕인지
    # 구분되지 않으므로 지우고 본다.
    pg.context.clear_cookies()
    probe_auth = """async (h) => {
      const r = await fetch('/api/health', { headers: h, cache: 'no-store' });
      return r.status;
    }"""
    for name, hdr in [("실제 IP 를 알려 주는 터널", {"X-Forwarded-For": "203.0.113.9"}),
                      ("IP 는 안 주고 https 만 알리는 터널", {"X-Forwarded-Proto": "https"}),
                      ("Tailscale Funnel 표시만 있는 요청", {"Tailscale-Funnel-Request": "?1"})]:
        st = pg.evaluate(probe_auth, hdr)
        check(f"{name} → 막힌다", st == 401, f"{st} 로 통과했다")
    check("PC 자신에서 여는 것은 그대로 통과한다", pg.evaluate(probe_auth, {}) == 200)
    errs.clear()        # 위 401 세 번은 일부러 낸 것이다

    # ── 그래프 위에서 화면이 스크롤되는가 ──
    # 진짜 터치라야 차트 라이브러리가 반응한다 → CDP 로 터치를 넣는다.
    print("\n[터치] 그래프 위에서 위아래로 쓸면 화면이 내려가는가")
    tp = pg.context.browser.new_context(viewport={"width": 412, "height": 780},
                                        has_touch=True, is_mobile=True)
    tpg = tp.new_page()
    cdp = tp.new_cdp_session(tpg)

    def swipe(box, dy, dx=0):
        x = int(box["x"] + box["width"] / 2)
        y = int(box["y"] + box["height"] * 0.6)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
        for i in range(1, 9):
            cdp.send("Input.dispatchTouchEvent", {"type": "touchMove",
                     "touchPoints": [{"x": x + int(dx * i / 8), "y": y + int(dy * i / 8)}]})
            time.sleep(0.02)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        time.sleep(0.6)

    try:
        tpg.goto(base + "/", wait_until="networkidle")
        for tab, title in [("analysis", "분석"), ("portfolio", "포트폴리오")]:
            tpg.click(f"#tabs button[data-tab='{tab}']")
            tpg.wait_for_timeout(4500)
            tpg.evaluate("window.scrollTo(0, 120)")
            tpg.wait_for_timeout(400)
            box = tpg.query_selector("#body .chart").bounding_box()
            y0 = tpg.evaluate("Math.round(scrollY)")
            swipe(box, -160)
            y1 = tpg.evaluate("Math.round(scrollY)")
            check(f"{title} 그래프 위에서 위아래로 쓸면 화면이 내려간다", abs(y1 - y0) > 20,
                  f"스크롤 {y0} → {y1}")
            if tab == "analysis":
                # 위에서 화면이 내려갔으므로 차트 위치를 다시 잡는다(옛 좌표는 빗나간다)
                box = tpg.query_selector("#body .chart").bounding_box()
                r0 = tpg.evaluate("charts[0].timeScale().getVisibleLogicalRange()")
                swipe(box, 0, 110)
                r1 = tpg.evaluate("charts[0].timeScale().getVisibleLogicalRange()")
                check("가로로 쓸면 차트는 그대로 움직인다",
                      bool(r0 and r1 and abs(r1["from"] - r0["from"]) > 0.5),
                      f"{r0} → {r1}")
    finally:
        tp.close()

    print("\n[콘솔]")
    check("자바스크립트 오류 없음", not errs, " / ".join(errs[:5]))


def main() -> int:
    port = free_port()
    data = tempfile.mkdtemp(prefix="quant-test-")
    seed(data)
    srv = start_server(port, data)
    base = f"http://127.0.0.1:{port}"
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=chromium())
            pg = b.new_page(viewport={"width": 412, "height": 915})
            errs: list[str] = []
            pg.on("console", lambda m: errs.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
            pg.on("pageerror", lambda e: errs.append(f"[pageerror] {e}"))
            try:
                run(pg, base, errs)
            finally:
                b.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()

    print()
    if fails:
        print(f"실패 {len(fails)}건")
        for f in fails:
            print("  -", f)
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())

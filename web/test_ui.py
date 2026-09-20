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
    rows = [{"date": str(d0 + timedelta(days=i)),
             "krwEval": 35000 + i * 50, "usdEval": 8000 + i * 40 + math.sin(i / 6) * 300,
             "krwCash": 9770.0, "usdCash": 1841.0,
             "rate": 1350 + math.sin(i / 9) * 20, "pnlKrw": (i - 30) * 90000}
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
    check("한국을 누르면 한국 종목이 뜬다",
          state(pg, "S.market") == "KR"
          and state(pg, "S.rows.length > 0 && S.rows.every(r => /^\\d{6}$/.test(r.ticker))") is True,
          f"market={state(pg, 'S.market')} tickers={state(pg, '(S.rows||[]).map(r=>r.ticker)')}")
    pg.click("#hdr-seg button:has-text('미국')")
    pg.wait_for_timeout(1500)
    check("미국으로 되돌리면 칠도 같이 돌아온다", seg_on(pg) == "미국", f"칠={seg_on(pg)}")
    check("미국 종목이 다시 뜬다",
          state(pg, "S.rows.length > 5 && S.rows.every(r => !/^\\d{6}$/.test(r.ticker))") is True,
          f"tickers={state(pg, '(S.rows||[]).map(r=>r.ticker)')}")

    # ── 비교 → 분석: 행을 누르면 그 종목으로 ──
    print("\n[비교→분석] 행 누르기")
    name = pg.inner_text("#body table tr:nth-child(2) td.l")
    pg.click("#body table tr:nth-child(2)")
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
    check("1분을 누르면 분봉을 받아 온다", bool(got_min) and state(pg, "S.bar") == "1m",
          f"요청={got_min} bar={state(pg, 'S.bar')}")
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
        pg.click("#body .row2:has(span:text-is('AAPL')) button:has-text('삭제')")
        pg.wait_for_timeout(1500)
        check("종목을 삭제하면 목록에서 빠진다", "AAPL" not in pg.inner_text("#body"))

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
    first = pg.inner_text("#body table tr:nth-child(2)").split("\t")[0]
    check("이름 정렬은 첫 클릭에 ㄱ→ㅎ", first.startswith("AVXX") or first < "F",
          f"1등={first}")

    # ── 잘못 눌리지 않는가 ──
    print("\n[조작]")
    small = pg.evaluate("""() => [...document.querySelectorAll('#hdr-seg button, #hdr-btn, #body table tr, #body .tchips button')]
        .map(e => e.getBoundingClientRect())
        .filter(r => r.height > 2 && r.height < 38).length""")
    check("누르는 것들이 손가락 크기(38px 이상)", small == 0, f"작은 것 {small}개")

    reqs = []
    pg.on("request", lambda r: reqs.append(r.url) if "/api/compare" in r.url else None)
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
    print("\n[추가된 정보]")
    pg.click("#tabs button[data-tab='portfolio']")
    pg.wait_for_timeout(3500)
    body = pg.inner_text("#body")
    check("보유 종목에 비중 %가 보인다", "%" in body and "비중" in body, body[:60])

    pg.click("#tabs button[data-tab='settings']")
    pg.wait_for_timeout(2500)
    check("입금 날짜가 달력 입력이다",
          pg.eval_on_selector("#body input[type='date']", "e => e.type") == "date")
    check("접속 암호 칸이 있다", "접속 암호" in pg.inner_text("#body"))
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
    check("백업 파일을 내려받을 수 있다", dl >= 5, f"파일 {dl}개")

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

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


def start_server(port: int, data: str) -> subprocess.Popen:
    env = {**os.environ, "QUANT_MOCK": "1", "QUANT_DATA": data}
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

    print("\n[콘솔]")
    check("자바스크립트 오류 없음", not errs, " / ".join(errs[:5]))


def main() -> int:
    port = free_port()
    data = tempfile.mkdtemp(prefix="quant-test-")
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

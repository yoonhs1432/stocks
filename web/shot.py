"""화면을 폰 크기로 띄워 탭별로 캡처한다 — 개발 확인용.

    QUANT_MOCK=1 python server.py &
    python shot.py
"""
import sys, time
from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
TABS = ["compare", "analysis", "portfolio", "settings"]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = b.new_page(viewport={"width": 412, "height": 915}, device_scale_factor=2)
    errs = []
    pg.on("console", lambda m: errs.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(f"[pageerror] {e}"))

    pg.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    for t in TABS:
        pg.evaluate(f"localStorage.setItem('tab','{t}')")
        pg.reload(wait_until="networkidle")
        time.sleep(3)
        pg.screenshot(path=f"{OUT}/{t}.png", full_page=True)
        print(f"  {t}: {pg.evaluate('document.body.scrollHeight')}px")

    # 분석 탭 산점도도 한 장
    pg.evaluate("localStorage.setItem('tab','analysis'); localStorage.setItem('group','scatter')")
    pg.reload(wait_until="networkidle"); time.sleep(3)
    pg.screenshot(path=f"{OUT}/scatter.png", full_page=True)
    pg.evaluate("localStorage.setItem('group','series')")
    b.close()

    print("\n콘솔 오류:", "없음" if not errs else "")
    for e in errs[:15]:
        print("  ", e)

"""집 PC에서 돌리는 포트폴리오 서버.

**왜 PC에서 돌리나** — 토스 Open API 는 허용 IP 목록 밖에서 호출하면 `access_denied` 로
막는다. 폰이 5G 를 쓰면 IP 가 매일 바뀌어 그때마다 WTS 에 다시 등록해야 했다. PC 는 IP 가
고정이라 **한 번만 등록하면 끝난다.** 폰은 이 서버가 만든 화면을 브라우저로 볼 뿐,
토스에 직접 붙지 않는다.

실행:
    pip install -r requirements.txt
    python server.py            (또는 uvicorn server:app --host 0.0.0.0 --port 8000)

앱 키는 `config.json` 또는 환경변수(`TOSS_APP_KEY`/`TOSS_APP_SECRET`)로 준다.
**config.json 은 절대 커밋하지 않는다** (.gitignore 에 넣어 뒀다).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from toss import Toss, TossError

HERE = Path(__file__).parent
CONFIG = HERE / "config.json"

# 계좌 조회 캐시. 화면을 새로 그릴 때마다 토스를 부르면 한도(429)에 걸린다.
CACHE_TTL = 20.0


def _load_creds() -> tuple[str, str]:
    """환경변수 우선, 없으면 config.json. 둘 다 없으면 빈 값."""
    key = os.environ.get("TOSS_APP_KEY", "")
    secret = os.environ.get("TOSS_APP_SECRET", "")
    if key and secret:
        return key, secret
    if CONFIG.exists():
        try:
            o = json.loads(CONFIG.read_text(encoding="utf-8"))
            return o.get("app_key", ""), o.get("app_secret", "")
        except (ValueError, OSError):
            pass
    return "", ""


app = FastAPI(title="Quant Portfolio")
_key, _secret = _load_creds()
_toss = Toss(_key, _secret)

_lock = threading.Lock()
_cache: dict | None = None
_cache_at = 0.0


def _fetch_account() -> dict:
    """계좌 한 덩어리 — 요약·보유·예수금·환율을 한 번에.

    순서가 중요하다. 계좌 목록을 먼저 얻어야 `accountSeq` 로 나머지를 부를 수 있다.
    """
    accounts = _toss.accounts()
    if not accounts:
        raise TossError("no-account", 0, "조회 가능한 계좌가 없습니다")
    acc = accounts[0]
    seq = acc["accountSeq"]

    h = _toss.holdings(seq)
    krw_cash = _toss.buying_power(seq, "KRW")
    usd_cash = _toss.buying_power(seq, "USD")
    rate = _toss.usd_krw()
    if rate != rate or rate <= 0:      # NaN 방어 — 환율이 없으면 원화 환산이 전부 깨진다
        rate = 1400.0

    eval_krw = h["krwEval"] + h["usdEval"] * rate
    cash_krw = krw_cash + usd_cash * rate
    pnl_krw = h["krwPnl"] + h["usdPnl"] * rate
    daily_krw = h["krwDailyPnl"] + h["usdDailyPnl"] * rate

    # 보유 목록은 **원화 환산 평가금액 내림차순**. 파이 차트와 목록이 같은 순서여야
    # 색과 순서가 맞는다(안드로이드 앱과 같은 규칙).
    items = []
    for it in h["items"]:
        k = rate if it["currency"] == "USD" else 1.0
        items.append({**it, "evalKrw": it["evalAmount"] * k, "pnlKrw": it["pnlAmount"] * k})
    items.sort(key=lambda x: x["evalKrw"], reverse=True)

    return {
        "accountNo": acc["accountNo"],
        "rate": rate,
        "evalKrw": eval_krw,
        "cashKrw": cash_krw,
        "totalKrw": eval_krw + cash_krw,
        "pnlKrw": pnl_krw,
        "pnlRate": h["pnlRate"],
        "dailyPnlKrw": daily_krw,
        "dailyPnlRate": h["dailyPnlRate"],
        "krwEval": h["krwEval"], "usdEval": h["usdEval"],
        "krwCash": krw_cash, "usdCash": usd_cash,
        "items": items,
        "at": time.time(),
    }


@app.get("/api/account")
def api_account(force: bool = False):
    """포트폴리오 화면이 쓰는 유일한 엔드포인트."""
    global _cache, _cache_at
    with _lock:
        if not force and _cache and time.time() - _cache_at < CACHE_TTL:
            return _cache
        try:
            _cache = _fetch_account()
            _cache_at = time.time()
            return _cache
        except TossError as e:
            # 값을 지어내지 않는다. 화면에 실패를 그대로 드러내는 편이 낫다
            return JSONResponse(
                {"error": e.message, "code": e.code}, status_code=502)


@app.get("/api/health")
def api_health():
    return {"ok": True, "hasCreds": bool(_key and _secret)}


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    if not (_key and _secret):
        print("⚠️  앱 키가 없습니다. web/config.json 을 만들거나 "
              "TOSS_APP_KEY/TOSS_APP_SECRET 환경변수를 설정하세요.")
    # 0.0.0.0 으로 열어야 같은 네트워크의 폰에서 붙을 수 있다
    uvicorn.run(app, host="0.0.0.0", port=8000)

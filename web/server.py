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

from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import auth
import quant
import repo
import snapshots
import store
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


def _clean(v):
    """NaN·Inf 를 None 으로.

    ⚠️ 파이썬 json 은 NaN 을 그대로 흘려보내는데 그건 **유효한 JSON 이 아니라서**
    브라우저의 JSON.parse 가 통째로 거부한다. Z·M·RSI 는 warmup 구간이 NaN 이므로
    분석 화면이 전부 안 뜨게 된다. 내보내기 직전에 반드시 거른다.
    """
    if isinstance(v, float):
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


app = FastAPI(title="Quant Portfolio")
_key, _secret = _load_creds()

# QUANT_MOCK=1 이면 가짜 시세로 띄운다 — 화면을 직접 열어 확인하기 위한 개발용.
# 실제 사용에는 영향이 없고, 켜져 있으면 콘솔에 분명히 찍는다.
if os.environ.get("QUANT_MOCK") == "1":
    from mock import MockToss
    _toss = MockToss()
    print("⚠️  가짜 시세로 실행 중입니다 (QUANT_MOCK=1). 숫자는 진짜가 아닙니다.")
else:
    _toss = Toss(_key, _secret)

# ── 접속 인증 ──
# 이 서버는 계좌를 그대로 보여준다. 외부에 열면 주소를 아는 사람은 누구나 볼 수 있으므로
# 암호가 먼저다. 토큰은 config.json 에 저장되고, 없으면 처음 실행할 때 만들어진다.
ACCESS_TOKEN = auth.load_or_create_token(CONFIG)
app.middleware("http")(auth.make_guard(ACCESS_TOKEN))


@app.post("/login")
def login(request: Request, token: str = Form("")):
    ip = request.client.host if request.client else ""
    if auth._too_many(ip):
        return auth.login_page("시도가 너무 잦습니다. 잠시 후 다시 해 주세요.")
    import secrets as _s
    if not _s.compare_digest(token, ACCESS_TOKEN):
        auth._note_fail(ip)
        return auth.login_page("암호가 맞지 않습니다.")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp = RedirectResponse("/", status_code=303)
    auth._set_cookie(resp, ACCESS_TOKEN, secure)
    return resp

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
            return _clean(_cache)
        try:
            _cache = _fetch_account()
            _cache_at = time.time()
            snapshots.record(_cache)     # 토스에 과거 잔고 API 가 없어 직접 쌓는다
            return _clean(_cache)
        except TossError as e:
            # 값을 지어내지 않는다. 화면에 실패를 그대로 드러내는 편이 낫다
            return JSONResponse(
                {"error": e.message, "code": e.code}, status_code=502)


# ── 비교 ──

_ov_lock = threading.Lock()
_ov_cache: dict[str, tuple[float, list]] = {}
OV_TTL = 300.0


@app.get("/api/compare")
def api_compare(market: str = "US", force: bool = False):
    """비교 화면 — 종목별 현재가·등락률·Z·M.

    첫 호출은 종목 수만큼 일봉을 받느라 20~30초 걸린다. 그 다음부터는 캐시가 받쳐 준다.
    """
    # 보유 종목은 설정 목록에 없어도 **항상** 표에 넣는다.
    # 안드로이드에서 이걸 안 하니 보유 중인 종목이 비교 탭에서 통째로 빠졌다.
    held: set[str] = set()
    try:
        acc = api_account()          # 20초 캐시라 부담이 없다
        if isinstance(acc, dict):
            held = {h["symbol"] for h in acc.get("items", [])}
    except Exception:
        pass

    def mine(t: str) -> bool:
        return store.is_krw(t) if market == "KR" else not store.is_krw(t)

    watch = store.tickers()
    tickers = [t for t in watch if mine(t)]
    tickers += [t for t in sorted(held) if mine(t) and t not in tickers]
    if not tickers:
        return {"rows": [], "market": market}

    key = f"{market}:{store.lookback_months()}:{len(tickers)}"
    with _ov_lock:
        hit = _ov_cache.get(key)
        if not force and hit and time.time() - hit[0] < OV_TTL:
            rows = hit[1]
        else:
            try:
                rows = repo.overview(_toss, tickers, force, held)
            except TossError as e:
                return JSONResponse({"error": e.message, "code": e.code}, status_code=502)
            except Exception as e:
                return JSONResponse({"error": str(e) or "조회 실패"}, status_code=502)
            _ov_cache[key] = (time.time(), rows)

    # 이름·보유 여부는 계좌에서 덧입힌다 (국내는 코드만 보면 무슨 종목인지 모른다)
    acc = _cache
    if acc:
        by = {h["symbol"]: h for h in acc["items"]}
        for r in rows:
            h = by.get(r["ticker"])
            if h:
                r["name"] = h["name"] or r["ticker"]
                r["holding"] = True
                r["avgPrice"] = h["avgPrice"]
    return _clean({"rows": rows, "market": market, "at": time.time()})


_px_cache: dict[str, tuple[float, dict]] = {}
PX_TTL = 3.0


@app.get("/api/prices")
def api_prices(symbols: str = ""):
    """실시간 현재가 — 비교/분석 화면이 주기적으로 부른다.

    기기마다 따로 부르므로(PC + 폰 + 탭 여러 개) 짧게라도 캐시를 둔다.
    같은 목록을 3초 안에 다시 물으면 토스를 또 부르지 않는다.
    """
    syms = [s for s in symbols.split(",") if s]
    if not syms:
        return {}
    key = ",".join(syms)
    hit = _px_cache.get(key)
    if hit and time.time() - hit[0] < PX_TTL:
        return hit[1]
    try:
        out = _toss.prices(syms)
        _px_cache[key] = (time.time(), out)
        return out
    except TossError as e:
        return JSONResponse({"error": e.message, "code": e.code}, status_code=502)


# ── 분석 ──

@app.get("/api/analysis")
def api_analysis(ticker: str, force: bool = False):
    """한 종목 분석 — 회귀·Z·M·MACD·RSI + 일봉 + 매매 마커."""
    try:
        r, bars = repo.analyze(_toss, ticker, force=force)
    except TossError as e:
        return JSONResponse({"error": e.message, "code": e.code}, status_code=502)
    if not bars:
        return JSONResponse({"error": f"{ticker} 시세를 가져오지 못했습니다"}, status_code=502)

    tr = store.trades().get(ticker, [])
    pos = store.position(tr)
    # 평단은 토스 보유 정보를 우선한다 — 체결내역 역산은 기록이 빠지면 어긋난다
    avg = None
    if _cache:
        h = next((x for x in _cache["items"] if x["symbol"] == ticker), None)
        if h and h["avgPrice"] > 0:
            avg = h["avgPrice"]
    if avg is None and pos:
        avg = pos["avg"]

    return _clean({
        "ticker": ticker,
        "krw": store.is_krw(ticker),
        "candles": bars,
        "avgPrice": avg,
        "qty": pos["qty"] if pos else None,
        "trades": tr,
        "result": None if r is None else {
            "dates": r.dates, "zPct": r.zPct, "mPct": r.mPct, "rsi": r.rsi,
            "macd": r.macd, "macdSignal": r.macdSignal,
            "predicted": r.predicted, "bandUpper": r.bandUpper, "bandLower": r.bandLower,
            "spyNorm": r.spyNorm, "tickerNorm": r.tickerNorm,
            "beta": r.beta, "sigmaPct": r.sigmaPct, "lastPrice": r.lastPrice,
            "lastZpct": r.lastZpct, "lastMpct": r.lastMpct, "signal": r.signal,
        },
    })


@app.get("/api/minutes")
def api_minutes(ticker: str, force: bool = False):
    """1분봉 — 토스가 받아 주는 주기는 1d·1m 뿐이다."""
    try:
        bars = repo.minutes(_toss, ticker, force)
    except TossError as e:
        return JSONResponse({"error": e.message, "code": e.code}, status_code=502)
    closes = [b["close"] for b in bars]
    macd, sig = quant.macd_of(closes) if len(closes) >= 2 else ([], [])
    return _clean({
        "ticker": ticker, "candles": bars,
        "macd": macd, "macdSignal": sig,
        "rsi": quant.rsi_of(closes) if len(closes) >= 2 else [],
    })


# ── 설정 ──

@app.get("/api/snapshots")
def api_snapshots(usd: bool = False):
    """자산 추이 — 평가금액·예수금·총자산·평가손익·원금."""
    return _clean(snapshots.series(store.deposits(), usd))


@app.get("/api/journal")
def api_journal(limit: int = 200):
    """매매 일지 — 전 종목 체결 기록을 최신순으로."""
    out = []
    for tk, lst in store.trades().items():
        for t in lst:
            out.append({**t, "ticker": tk, "krw": store.is_krw(tk)})
    out.sort(key=lambda x: x["date"], reverse=True)
    return _clean({"trades": out[:limit], "total": len(out)})


@app.get("/api/settings")
def api_settings():
    return {
        "months": store.lookback_months(),
        "maxMonths": store.MAX_MONTHS,
        "tickers": [{"ticker": t, "krw": store.is_krw(t)} for t in store.tickers()],
        "deposits": store.deposits(),
        "principal": store.principal_total(),
        "trades": sum(len(v) for v in store.trades().values()),
        "tickSeconds": store.settings().get("tick_seconds", 10),
    }


@app.post("/api/settings/tick")
def api_set_tick(body: dict):
    v = max(0, min(60, int(body.get("seconds", 10))))
    store.put("tick_seconds", v)
    return {"tickSeconds": v}


@app.post("/api/settings/months")
def api_set_months(body: dict):
    m = max(3, min(store.MAX_MONTHS, int(body.get("months", store.MAX_MONTHS))))
    store.put("lookback_months", m)
    repo.clear_cache()           # 기간이 바뀌면 받아 둔 일봉을 다시 받아야 한다
    _ov_cache.clear()
    return {"months": m}


@app.post("/api/tickers")
def api_add_ticker(body: dict):
    store.add_ticker(str(body.get("ticker", "")))
    _ov_cache.clear()
    return {"tickers": store.tickers()}


@app.delete("/api/tickers/{ticker}")
def api_remove_ticker(ticker: str):
    store.remove_ticker(ticker)
    _ov_cache.clear()
    return {"tickers": store.tickers()}


@app.post("/api/deposits")
def api_add_deposit(body: dict):
    store.add_deposit(str(body.get("date", "")), float(body.get("krw", 0)))
    return {"deposits": store.deposits(), "principal": store.principal_total()}


@app.delete("/api/deposits/{index}")
def api_remove_deposit(index: int):
    store.remove_deposit(index)
    return {"deposits": store.deposits(), "principal": store.principal_total()}


@app.post("/api/fills")
def api_fills():
    """체결내역 가져오기 — 매매 마커와 평단(대체값)의 출처."""
    try:
        accounts = _toss.accounts()
        if not accounts:
            raise TossError("no-account", 0, "조회 가능한 계좌가 없습니다")
        fills = _toss.fills(accounts[0]["accountSeq"])
    except TossError as e:
        return JSONResponse({"error": e.message, "code": e.code}, status_code=502)
    total = store.save_fills(fills)
    return {"fetched": len(fills), "total": total}


@app.post("/api/cache/clear")
def api_clear_cache():
    repo.clear_cache()
    _ov_cache.clear()
    return {"ok": True}


@app.get("/api/health")
def api_health():
    return {"ok": True, "hasCreds": bool(_key and _secret)}


class NoCacheStatic(StaticFiles):
    """화면 파일은 항상 서버에 한 번 물어보고 쓰게 한다.

    코드를 고쳐 서버를 다시 켰는데도 폰에는 예전 화면이 그대로 뜨는 일이 있었다.
    캐시 지시가 없으면 브라우저가 알아서 "이 정도면 신선하겠지" 하고 예전 파일을
    그냥 쓰기 때문이다. no-cache 는 **받아 두되 쓰기 전에 확인**하라는 뜻이라,
    안 바뀌었으면 304 로 끝나고 바뀌었으면 바로 새 파일이 온다.
    """

    async def get_response(self, path: str, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/", NoCacheStatic(directory=HERE / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    if not (_key and _secret):
        print("⚠️  앱 키가 없습니다. web/config.json 을 만들거나 "
              "TOSS_APP_KEY/TOSS_APP_SECRET 환경변수를 설정하세요.")
    print()
    print("  PC 에서:     http://localhost:8000   (암호 없이 열립니다)")
    print("  다른 기기에서: 접속 암호  " + ACCESS_TOKEN)
    print("  즐겨찾기용:   <주소>/?key=" + ACCESS_TOKEN)
    print()
    # 0.0.0.0 으로 열어야 같은 네트워크의 폰에서 붙을 수 있다
    uvicorn.run(app, host="0.0.0.0", port=8000)

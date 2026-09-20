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
import re
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

import auth
import quant
import repo
import snapshots
import store
import universe
from toss import Toss, TossError

HERE = Path(__file__).parent
# 앱 키와 접속 암호가 든 파일. QUANT_CONFIG 로 바꿀 수 있는 건 검사용이다 —
# 검사가 진짜 config.json 의 암호를 갈아 치우면 폰이 갑자기 안 들어가진다.
CONFIG = Path(os.environ.get("QUANT_CONFIG") or HERE / "config.json")

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
    """NaN·Inf 를 None 으로, 소수는 4자리까지만.

    ⚠️ 파이썬 json 은 NaN 을 그대로 흘려보내는데 그건 **유효한 JSON 이 아니라서**
    브라우저의 JSON.parse 가 통째로 거부한다. Z·M·RSI 는 warmup 구간이 NaN 이므로
    분석 화면이 전부 안 뜨게 된다. 내보내기 직전에 반드시 거른다.

    소수 자르기는 응답 크기 때문이다. `0.12345678901234` 같은 값이 배열 10개 × 500봉이면
    그것만으로 수십 KB 다. 화면은 소수 2자리까지만 쓰므로 4자리면 넘치게 충분하다.
    """
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return round(v, 4)
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


app = FastAPI(title="Quant Portfolio")

# 분석 응답은 500봉 × 배열 10개라 130KB 가까이 된다. 폰에서 종목을 누를 때마다 그걸
# 통째로 받으니 느렸다. 압축만으로 1/3 이 되고, 소수 자릿수 정리(_clean)까지 하면 1/5.
app.add_middleware(GZipMiddleware, minimum_size=1024)
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
    if not _s.compare_digest(token, auth.token()):
        auth._note_fail(ip)
        return auth.login_page("암호가 맞지 않습니다.")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp = RedirectResponse("/", status_code=303)
    auth._set_cookie(resp, auth.token(), secure)
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
            store.learn_markets(_cache["items"])   # 어느 시장 종목인지 배워 둔다
            store.backup_daily(snapshots.today())   # 기록은 되살릴 수 없으니 하루 한 벌 복사
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
        rows_at = time.time()
        if not force and hit and time.time() - hit[0] < OV_TTL:
            rows, rows_at = hit[1], hit[0]
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

    # 이름 우선순위: 내가 붙인 이름 > 보유(토스) > 받아 둔 종목 목록 > 코드
    saved = store.names()
    for r in rows:
        t = r["ticker"]
        if saved.get(t):
            r["name"] = saved[t]
        elif r.get("name") in (None, "", t):
            r["name"] = universe.name_of(t) or t

    # 국내 종목 목록을 하루 1회 받아 둔다 — 코드만 뜨던 국내 종목에 이름을 붙이려고
    threading.Thread(target=lambda: universe.ensure(_toss), daemon=True).start()
    # 반대쪽 시장도 미리 계산해 둔다 — 미국↔한국 전환에서 기다리지 않게
    threading.Thread(target=_warm_market, args=("KR" if market == "US" else "US",),
                     daemon=True).start()
    # 일봉이 캐시에 올라온 김에 분석도 미리 계산해 둔다. 폰에서 종목을 누르는 순간
    # 계산이 시작되는 게 아니라 이미 끝나 있게.
    threading.Thread(target=_warm_analysis, args=(tickers,), daemon=True).start()
    return _clean({"rows": rows, "market": market, "at": time.time(), "asOf": rows_at})


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

# 분석 결과 캐시. 계산 자체는 빠르지만 종목마다 매번 다시 하면 폰에서 누를 때마다 기다린다.
# 일봉 캐시와 같은 6시간을 쓴다 — 어차피 원본이 그 주기로 갱신된다.
AN_TTL = 6 * 3600
_an_lock = threading.Lock()
_an_cache: dict[str, tuple[float, dict]] = {}
_warming = False


def _analysis(ticker: str, force: bool = False) -> dict:
    """분석 응답 한 벌. 캐시에 있으면 그대로 준다. 실패는 TossError/ValueError 로 던진다."""
    key = f"{ticker}:{store.lookback_months()}"
    if not force:
        with _an_lock:
            hit = _an_cache.get(key)
        if hit and time.time() - hit[0] < AN_TTL:
            return hit[1]

    r, bars = repo.analyze(_toss, ticker, force=force)
    if not bars:
        raise ValueError(f"{ticker} 시세를 가져오지 못했습니다")

    payload = _clean({
        "ticker": ticker,
        "name": store.name_of(ticker) or universe.name_of(ticker) or ticker,
        "krw": store.is_krw(ticker),
        "candles": bars,
        "trades": store.trades().get(ticker, []),
        "result": None if r is None else {
            "dates": r.dates, "zPct": r.zPct, "mPct": r.mPct, "rsi": r.rsi,
            "macd": r.macd, "macdSignal": r.macdSignal,
            "predicted": r.predicted, "bandUpper": r.bandUpper, "bandLower": r.bandLower,
            "spyNorm": r.spyNorm, "tickerNorm": r.tickerNorm,
            "beta": r.beta, "sigmaPct": r.sigmaPct, "lastPrice": r.lastPrice,
            "lastZpct": r.lastZpct, "lastMpct": r.lastMpct, "signal": r.signal,
        },
    })
    with _an_lock:
        _an_cache[key] = (time.time(), payload)
    return payload


def _warm_analysis(tickers: list[str]) -> None:
    """비교를 받아 온 김에 **전 종목 분석을 미리 계산해 둔다.**

    폰에서 종목을 누르면 그때부터 계산하느라 기다려야 했다. 일봉은 이미 캐시에 있으니
    계산만 하면 되고, PC 는 놀고 있으므로 미리 해 두는 편이 낫다.
    """
    global _warming
    with _an_lock:
        if _warming:
            return
        _warming = True
    try:
        for t in tickers:
            try:
                _analysis(t)
            except Exception:
                pass          # 한 종목이 실패해도 나머지는 계속
    finally:
        with _an_lock:
            _warming = False


def _warm_market(market: str) -> None:
    """반대쪽 시장 표를 캐시에 올려 둔다. 이미 최신이면 아무것도 하지 않는다."""
    try:
        held: set[str] = set()
        acc = _cache
        if acc:
            held = {h["symbol"] for h in acc.get("items", [])}

        def mine(t: str) -> bool:
            return store.is_krw(t) if market == "KR" else not store.is_krw(t)

        tickers = [t for t in store.tickers() if mine(t)]
        tickers += [t for t in sorted(held) if mine(t) and t not in tickers]
        if not tickers:
            return
        key = f"{market}:{store.lookback_months()}:{len(tickers)}"
        with _ov_lock:
            hit = _ov_cache.get(key)
            if hit and time.time() - hit[0] < OV_TTL:
                return
        rows = repo.overview(_toss, tickers, False, held)
        with _ov_lock:
            _ov_cache[key] = (time.time(), rows)
    except Exception:
        pass          # 미리 해 두는 일이라 실패해도 그만이다


@app.get("/api/analysis")
def api_analysis(ticker: str, force: bool = False):
    """한 종목 분석 — 회귀·Z·M·MACD·RSI + 일봉 + 매매 마커."""
    try:
        base = _analysis(ticker, force)
    except TossError as e:
        return JSONResponse({"error": e.message, "code": e.code}, status_code=502)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=502)

    # 평단·수량은 캐시에 넣지 않는다 — 체결되면 바로 바뀌어야 한다
    tr = base["trades"]
    pos = store.position(tr)
    # 평단은 토스 보유 정보를 우선한다 — 체결내역 역산은 기록이 빠지면 어긋난다
    avg = None
    if _cache:
        h = next((x for x in _cache["items"] if x["symbol"] == ticker), None)
        if h and h["avgPrice"] > 0:
            avg = h["avgPrice"]
    if avg is None and pos:
        avg = pos["avg"]

    return {**base, "avgPrice": _clean(avg), "qty": pos["qty"] if pos else None}


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

@app.get("/api/history")
def api_history(days: int = 120):
    """지난 날들의 보유 내역 — 그날 무엇을 얼마나 들고 있었는지."""
    return _clean(snapshots.history(max(2, min(1000, days))))


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
        "tickers": [{"ticker": t, "krw": store.is_krw(t),
                     "name": store.name_of(t) or universe.name_of(t)}
                    for t in store.tickers()],
        "deposits": store.deposits(),
        "principal": store.principal_total(),
        "trades": sum(len(v) for v in store.trades().values()),
        "tickSeconds": store.settings().get("tick_seconds", 10),
        # 이 응답은 이미 인증을 통과한 사람만 받는다(미들웨어). 암호를 잊었을 때
        # config.json 을 열어 보지 않아도 되게 화면에서 확인·교체할 수 있게 한다.
        "accessToken": auth.token(),
        "version": version(),
    }


@app.post("/api/auth/rotate")
def api_rotate(request: Request):
    """접속 암호를 새로 만든다. 요청한 기기만 쿠키를 새로 받아 그대로 쓸 수 있다."""
    try:
        new = auth.rotate(CONFIG)
    except OSError as e:
        return JSONResponse({"error": f"config.json 에 쓰지 못했습니다 ({e})"}, status_code=500)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp = JSONResponse({"token": new})
    auth._set_cookie(resp, new, secure)
    return resp


# ── 업데이트 ──
# 폰에서 고친 코드를 받으려고 PC 앞에 가야 했다. 여기서 git pull 을 하고 스스로 종료하면
# run.ps1 의 감시 루프가 새 코드로 다시 띄운다. **cloudflared 는 건드리지 않으므로
# 터널 주소가 그대로**라 폰에서 그대로 새로고침하면 된다.
# 검사용으로만 바꾼다 — 진짜 저장소에 git pull 을 하지 않게.
REPO = Path(os.environ.get("QUANT_REPO") or HERE.parent)
RESTART_CODE = 3          # run.ps1 이 이 코드를 보면 다시 띄운다


def _git(*args: str, timeout: int = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                           text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()
    except FileNotFoundError:
        return 127, "git 을 찾을 수 없습니다"
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def version() -> str:
    code, out = _git("log", "-1", "--format=%h %cd", "--date=format:%m-%d %H:%M", timeout=10)
    return out if code == 0 else "?"


# 새 버전이 올라왔는지만 조용히 확인한다(받지는 않는다). 받는 시점은 사용자가 정한다.
UPDATE_TTL = 300.0
_upd_lock = threading.Lock()
_upd: dict = {}


def _check_update(force: bool = False) -> dict:
    with _upd_lock:
        if not force and _upd and time.time() - _upd.get("at", 0) < UPDATE_TTL:
            return dict(_upd)

    info = {"at": time.time(), "available": False, "behind": 0, "subject": "",
            "version": version()}
    code, _ = _git("fetch", "origin", "main", timeout=60)
    if code == 0:
        c2, out = _git("rev-list", "--count", "HEAD..origin/main", timeout=20)
        if c2 == 0 and out.strip().isdigit():
            info["behind"] = int(out.strip())
            info["available"] = info["behind"] > 0
            if info["available"]:
                _, subj = _git("log", "-1", "--format=%s", "origin/main", timeout=20)
                info["subject"] = subj.splitlines()[0] if subj else ""
    # 확인에 실패해도 조용히 넘긴다 — 인터넷이 잠깐 끊긴 것뿐일 수 있다
    with _upd_lock:
        _upd.clear()
        _upd.update(info)
    return dict(info)


@app.get("/api/update/check")
def api_update_check(force: bool = False):
    return _check_update(force)


@app.post("/api/update")
def api_update():
    """최신 코드를 받아 온다. 받은 게 있으면 스스로 종료해 새 코드로 다시 뜬다."""
    code, out = _git("pull", "--ff-only", "origin", "main")
    if code != 0:
        return JSONResponse({"error": f"받지 못했습니다\n{out}"}, status_code=502)

    with _upd_lock:
        _upd.clear()            # 받았으니 확인 결과를 새로 구한다
    changed = "Already up to date" not in out and "이미 업데이트" not in out
    supervised = os.environ.get("QUANT_SUPERVISED") == "1"
    if changed and supervised:
        # 응답을 먼저 보내고 종료한다 — 바로 죽으면 폰에는 "연결 실패"만 남는다
        threading.Timer(1.0, lambda: os._exit(RESTART_CODE)).start()
    return {"output": out, "changed": changed, "restarting": changed and supervised,
            "version": version(),
            "note": "" if supervised else "서버를 직접 다시 켜야 적용됩니다 (run.ps1 로 띄우면 자동)"}


@app.get("/api/backup")
def api_backup():
    """기록 전부를 파일 하나로 내려 준다 — 폰에서도 받을 수 있게.

    입금·매매·자산 추이는 **토스에서 다시 못 받는다.** PC 가 고장 나면 끝이라
    사용자가 직접 챙길 수단이 있어야 한다.
    """
    day = snapshots.today()
    return JSONResponse(
        store.backup_payload(),
        headers={"Content-Disposition": f'attachment; filename="quant-backup-{day}.json"',
                 "Cache-Control": "no-store"})


@app.post("/api/deposits/bulk")
def api_deposits_bulk(body: dict):
    """입금 기록을 한 번에 여러 건 — 한 줄에 `날짜 금액`.

    한 건씩 넣으려면 날짜를 고르고 금액을 치고 추가를 누르길 반복해야 한다.
    다른 곳에 적어 둔 기록을 그대로 옮겨 붙일 수 있게 한다.
    쉼표는 금액의 자릿점일 수 있으므로 **줄 단위**로만 나눈다.
    """
    text = str(body.get("text", ""))
    rows, bad = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+([+-]?[\d,]+(?:\.\d+)?)\s*원?$", line)
        if not m:
            bad.append(line[:24])
            continue
        rows.append({"date": m.group(1), "krw": float(m.group(2).replace(",", ""))})
    if bad:
        return JSONResponse(
            {"error": "이렇게 적어 주세요: 2026-09-01 13,789,303\n읽지 못한 줄: "
                      + " / ".join(bad[:3])}, status_code=400)
    if not rows:
        return JSONResponse({"error": "입금 기록이 하나도 없습니다"}, status_code=400)
    if body.get("replace"):
        store.set_deposits(rows)
    else:
        for r in rows:
            store.add_deposit(r["date"], r["krw"])
    return {"deposits": store.deposits(), "principal": store.principal_total(), "added": len(rows)}


@app.post("/api/tickers/bulk")
def api_tickers_bulk(body: dict):
    """목록 전체를 한 번에 저장한다 — 쉼표·줄바꿈·공백 아무거나로 구분.

    하나씩 추가하려면 20번을 눌러야 했다. 다른 앱에서 쓰던 목록을 그대로 붙여넣게 한다.
    """
    text = str(body.get("text", ""))
    seen, out, nm = set(), [], {}
    # 쉼표·줄바꿈으로 나눈다. "005930=삼성전자" 처럼 이름을 같이 적을 수 있고,
    # 이름이 없는 덩어리는 공백으로도 나눈다 — "AAA BBB CCC" 처럼 붙여넣는 경우.
    # (이름에는 공백이 들어갈 수 있으므로 = 가 있으면 통째로 둔다.)
    items: list[str] = []
    for raw in re.split(r"[,;\n]+", text):
        chunk = raw.strip()
        if not chunk:
            continue
        items += [chunk] if "=" in chunk else chunk.split()
    for item in items:
        code, _, name = item.partition("=")
        t = code.strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if name.strip():
            nm[t] = name.strip()
    if not out:
        return JSONResponse({"error": "종목이 하나도 없습니다"}, status_code=400)
    store.set_tickers(out)
    if nm:
        store.set_names(nm)
    _ov_cache.clear()
    return {"tickers": out}


@app.post("/api/tickers/name")
def api_set_name(body: dict):
    """종목에 이름 붙이기 — 코드만 보면 뭔지 모르는 국내 종목용."""
    t = str(body.get("ticker", "")).strip().upper()
    if not t:
        return JSONResponse({"error": "종목이 없습니다"}, status_code=400)
    store.set_name(t, str(body.get("name", "")))
    _ov_cache.clear()
    return {"ticker": t, "name": store.name_of(t)}


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
    with _an_lock:
        _an_cache.clear()
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
    with _an_lock:
        _an_cache.clear()
    return {"ok": True}


@app.get("/api/health")
def api_health():
    return {"ok": True, "hasCreds": bool(_key and _secret)}


def _static_version() -> str:
    """화면 파일들 중 가장 최근에 바뀐 시각. 주소 뒤에 붙일 꼬리표로 쓴다."""
    try:
        return str(int(max(f.stat().st_mtime for f in (HERE / "static").rglob("*") if f.is_file())))
    except (OSError, ValueError):
        return "0"


@app.get("/")
def index():
    """첫 화면만 서버가 직접 만들어 준다 — app.js 주소에 버전 꼬리표를 붙이려고.

    코드를 고쳐 서버를 다시 켰는데도 브라우저가 예전 app.js 를 계속 쓰는 일이 있었다.
    캐시 지시(아래 NoCacheStatic)만으로는 **이미 캐시에 들어앉은 예전 파일**을 확실히
    밀어내지 못한다. 주소 자체가 `app.js?v=...` 로 바뀌면 브라우저에는 처음 보는
    파일이라 무조건 새로 받는다. 이 문서 자체는 매번 새로 받게 no-store 를 준다.
    """
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'\b(src|href)="(?!https?:|data:)([^"]+\.(?:js|css))"',
                  rf'\1="\2?v={_static_version()}"', html)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


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


def _recorder() -> None:
    """앱을 안 열어도 기록이 빠지지 않게, 서버가 주기적으로 계좌를 조회해 남긴다.

    같은 날짜는 덮어쓰므로 그날의 **마지막 조회값**이 남는다. 30분에 한 번이라
    토스 호출량도 무시할 만하다(조회 4번). 실패는 조용히 넘긴다 — 장 마감이든
    인터넷이 끊겼든 다음 차례에 다시 한다.
    """
    while True:
        try:
            api_account()
        except Exception:
            pass
        time.sleep(1800)


threading.Thread(target=_recorder, daemon=True).start()

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

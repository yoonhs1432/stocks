"""시세 캐시 + 비교/분석 계산 — `data/Quotes.kt` · `data/OverviewRepo.kt` 대응.

⚠️ **일봉 캐시가 이 파일의 핵심이다.** 비교 화면은 20여 종목을 한꺼번에 분석하는데
종목당 200봉씩 3페이지를 받으므로, 캐시가 없으면 60여 건이 한꺼번에 몰려 429 가 난다.
안드로이드에서 실제로 겪은 문제이고, 그때 대책이 ① 동시요청 3개 제한 ② 429 백오프 재시도
③ 일봉 6시간 캐시였다. 여기서는 캐시를 **파일로** 둬서 서버를 껐다 켜도 살아남는다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import quant
import store
from toss import Toss, TossError

CACHE = Path(os.environ.get("QUANT_DATA") or Path(__file__).parent / "data") / "candles"
# 캐시 형식이 바뀌면 올린다 — 예전 파일은 무시하고 다시 받는다.
# v2: 일봉 시각을 거래일 00:00 UTC 로 맞춰 저장.
CACHE_VERSION = "v2"
DAILY_TTL = 6 * 3600        # 일봉은 하루 한 번만 바뀐다
MINUTE_TTL = 60             # 1분봉은 계속 바뀐다
MINUTE_BARS = 390           # 미국 정규장 하루

_gate = threading.Semaphore(3)   # 일봉 동시 요청 제한
_mem: dict[str, tuple[float, list[dict]]] = {}
_mem_lock = threading.Lock()


def _cache_file(symbol: str, interval: str) -> Path:
    safe = "".join(c if c.isalnum() else "_" for c in symbol)
    # 파일 이름에 형식 번호를 넣어 둔다 — 예전 파일은 저절로 무시되고 다시 받는다
    return CACHE / f"{safe}.{interval}.{CACHE_VERSION}.json"


# ── 일봉 시각 맞추기 ──
# 토스는 봉 시각을 **그 시장의 자정**으로 준다. SPY 는 04:00 UTC(미국 자정)이라 UTC 날짜가
# 곧 거래일이지만, 국내 종목은 15:00 UTC(= 다음날 한국 자정)이라 **UTC 날짜가 하루 밀린다.**
# 그대로 두면 ① 차트에 하루 앞당겨 그려지고 ② SPY 와 날짜를 맞출 때 한국 월요일이
# UTC 일요일이 되어 통째로 빠지며(543일 중 137일) ③ 남은 날도 한국 화요일이 미국 월요일과
# 짝지어져 회귀·β·Z·M 이 어긋난다. 받아 오는 자리에서 **거래일 00:00 UTC 로 맞춰 둔다.**
KST_OFFSET = 9 * 3600


def _normalize_daily(symbol: str, bars: list[dict]) -> list[dict]:
    off = KST_OFFSET if store.is_krw(symbol) else 0
    out = []
    seen = set()
    for b in bars:
        day = (int(b["t"]) + off) // 86400 * 86400
        if day in seen:
            continue          # 같은 날이 둘이면 앞의 것만 (있을 리 없지만 방어)
        seen.add(day)
        out.append({**b, "t": day})
    return out


def _load_cache(symbol: str, interval: str, ttl: float) -> list[dict] | None:
    key = f"{symbol}.{interval}"
    with _mem_lock:
        hit = _mem.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]

    f = _cache_file(symbol, interval)
    if not f.exists() or time.time() - f.stat().st_mtime >= ttl:
        return None
    try:
        bars = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    with _mem_lock:
        _mem[key] = (f.stat().st_mtime, bars)
    return bars


def _save_cache(symbol: str, interval: str, bars: list[dict]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        _cache_file(symbol, interval).write_text(
            json.dumps(bars, separators=(",", ":")), encoding="utf-8")
    except OSError:
        pass
    with _mem_lock:
        _mem[f"{symbol}.{interval}"] = (time.time(), bars)


def clear_cache() -> None:
    """일봉을 전부 버린다 (조회기간을 늘렸거나 손으로 다시 받을 때)."""
    with _mem_lock:
        _mem.clear()
    if CACHE.exists():
        for f in CACHE.glob("*.json"):
            try:
                f.unlink()
            except OSError:
                pass


def candles(toss: Toss, symbol: str, months: int | None = None,
            force: bool = False) -> list[dict]:
    """일봉. 캐시가 살아 있으면 그대로 쓴다. 실패하면 **빈 리스트**(값을 지어내지 않는다)."""
    months = months or store.lookback_months()
    if not force:
        hit = _load_cache(symbol, "1d", DAILY_TTL)
        if hit is not None:
            return hit

    with _gate:
        for attempt in range(3):
            try:
                bars = _normalize_daily(symbol, toss.ohlc(symbol, "1d", store.bar_count(months)))
                if len(bars) >= 2:
                    _save_cache(symbol, "1d", bars)
                return bars
            except TossError as e:
                if e.http == 429:          # 한도면 잠깐 쉬었다 재시도
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break
            except Exception:
                break
    # 새로 못 받았으면 오래된 캐시라도 준다 — 빈 화면보다 낫다
    return _load_cache(symbol, "1d", 10 ** 9) or []


TAIL_BARS = 5        # 장중 갱신에서 다시 받는 봉 수


def refresh_tail(toss: Toss, symbol: str) -> bool:
    """**끝 몇 봉만** 다시 받아 캐시에 덮어쓴다 — 장중 자동 갱신용.

    장이 열려 있는 동안 오늘 봉은 계속 자란다. 전체를 다시 받으면 종목당 3페이지라
    스무 종목이면 60번인데, 끝 몇 봉만 받으면 종목당 한 번이면 된다.
    캐시가 아예 없으면 아무것도 하지 않는다(그건 `candles()` 가 할 일이다).
    """
    old = _load_cache(symbol, "1d", 10 ** 9)
    if not old:
        return False
    with _gate:
        try:
            tail = _normalize_daily(symbol, toss.ohlc(symbol, "1d", TAIL_BARS))
        except Exception:
            return False
    if not tail:
        return False
    by = {b["t"]: b for b in old}
    before = by.get(tail[-1]["t"], {}).get("close")
    for b in tail:
        by[b["t"]] = b                      # 같은 날이면 새 값으로 갈아끼운다
    _save_cache(symbol, "1d", [by[t] for t in sorted(by)])
    return before != tail[-1]["close"]      # 값이 실제로 바뀌었는가


def minutes(toss: Toss, symbol: str, force: bool = False) -> list[dict]:
    """1분봉. 분석 화면에서 **보고 있는 종목만** 받는다."""
    if not force:
        hit = _load_cache(symbol, "1m", MINUTE_TTL)
        if hit is not None:
            return hit
    with _gate:
        try:
            bars = toss.ohlc(symbol, "1m", MINUTE_BARS, adjusted=False)
            if len(bars) >= 2:
                _save_cache(symbol, "1m", bars)
            return bars
        except Exception:
            return _load_cache(symbol, "1m", 10 ** 9) or []


def _closes(bars: list[dict]) -> list[tuple[int, float]]:
    return [(b["t"], b["close"]) for b in bars]


def analyze(toss: Toss, ticker: str, months: int | None = None,
            force: bool = False) -> tuple[quant.Result | None, list[dict]]:
    """한 종목 분석 — SPY 와 날짜를 맞춰 회귀한다. (결과, 일봉) 반환."""
    spy = candles(toss, store.BASE, months, force)
    tk = candles(toss, ticker, months, force)
    if len(spy) < 2 or len(tk) < 2:
        return None, tk
    return quant.analyze(_closes(spy), _closes(tk)), tk


def overview(toss: Toss, tickers: list[str], force: bool = False,
             held: set[str] | None = None) -> list[dict]:
    """비교 화면 한 줄씩. 종목을 **병렬로** 분석하되 동시 요청은 게이트가 막는다."""
    months = store.lookback_months()
    candles(toss, store.BASE, months, force)      # 기준 자산을 먼저 받아 캐시에 올린다
    all_trades = store.trades()
    held = held or set()

    errors: list[Exception] = []

    def one(tk: str) -> dict | None:
        try:
            r, bars = analyze(toss, tk, months, force)
        except Exception as e:      # 종목 하나가 실패해도 나머지는 보여준다
            errors.append(e)
            return None
        if not bars:
            # 행을 버리면 종목이 소리 없이 사라져 빠진 줄도 모른다 → 행은 남기고 값만 비운다
            errors.append(TossError("no-data", 0, f"{tk} 시세를 가져오지 못했습니다"))
            return {"ticker": tk, "name": tk, "price": None, "prevClose": None,
                    "day": None, "open": None, "high": None, "low": None,
                    "zPct": None, "mPct": None, "beta": None, "sigmaPct": None,
                    "signal": "hold", "holding": tk in held,
                    "hasHistory": bool(all_trades.get(tk)), "krw": store.is_krw(tk)}
        last = bars[-1]
        prev = bars[-2]["close"] if len(bars) >= 2 else last["close"]
        day = (last["close"] / prev - 1) * 100 if prev else 0.0
        tr = all_trades.get(tk, [])
        pos = store.position(tr)
        return {
            "ticker": tk,
            "name": tk,
            "price": last["close"],
            "prevClose": prev,
            "day": day,
            "open": last["open"], "high": last["high"], "low": last["low"],
            # 회귀가 안 되는 종목(상장 30거래일 미만)도 행을 버리지 않고 지표만 비운다
            "zPct": r.lastZpct if r else None,
            "mPct": r.lastMpct if r else None,
            "beta": r.beta if r else None,
            "sigmaPct": r.sigmaPct if r else None,
            "signal": r.signal if r else "hold",
            "holding": pos is not None or tk in held,
            "hasHistory": bool(tr),
            "krw": store.is_krw(tk),
        }

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for r in ex.map(one, tickers) if r]

    # 한 종목도 시세를 못 받았으면 대시만 늘어선 표 대신 이유를 올린다 —
    # 키 없음·허용 IP 밖·한도 초과가 전부 "값 없음"으로 보이면 손쓸 수가 없다
    if errors and not any(r["price"] is not None for r in rows):
        raise errors[0]
    return rows

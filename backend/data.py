"""시세 데이터 페치 — app.py에서 포팅 (Streamlit 캐시 → 자체 TTL 캐시).

FinanceDataReader 우선, 실패 시 Yahoo chart API 직접 fallback.
NYSE 휴장일/시간대(zoneinfo)도 동일 포팅.
"""
from __future__ import annotations

import datetime
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Optional

import FinanceDataReader as fdr
import pandas as pd
import requests

from analysis import X_ASSET_FIXED

log = logging.getLogger("quant.data")

HTTP_TIMEOUT_SEC = 6
MAX_PARALLEL_FETCH = 8
DATA_TTL_SEC = 300


# ────────────────────── TTL 메모 캐시 (st.cache_data 대체) ──────────────────────
def ttl_cache(ttl: int):
    """인자 해시 기반 TTL 캐시. 스레드 안전."""
    def deco(fn):
        store: dict = {}
        lock = threading.Lock()

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
                if hit and now - hit[0] < ttl:
                    return hit[1]
            val = fn(*args, **kwargs)
            with lock:
                store[key] = (now, val)
            return val

        wrapper.cache_clear = lambda: store.clear()
        return wrapper
    return deco


# ────────────────────── NYSE 휴장일 / 시장 상태 ──────────────────────
def _easter_date(year: int) -> datetime.date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    day = (h + l - 7 * m + 33 * month + 19) % 32
    return datetime.date(year, month, day)


def us_holidays(year: int) -> set:
    from datetime import date, timedelta

    def nth_weekday(y, mo, wd, n):
        count = 0
        for day in range(1, 32):
            try:
                dd = date(y, mo, day)
            except ValueError:
                break
            if dd.weekday() == wd:
                count += 1
                if count == n:
                    return dd
        return None

    def observed(dd, sat_to_fri=True):
        if dd.weekday() == 5:
            return dd - timedelta(days=1) if sat_to_fri else None
        if dd.weekday() == 6:
            return dd + timedelta(days=1)
        return dd

    holidays = set()
    for md, sat_ok in ((date(year, 1, 1), False), (date(year, 6, 19), True),
                       (date(year, 7, 4), True), (date(year, 12, 25), True)):
        if md.month == 6 and year < 2022:
            continue
        obs = observed(md, sat_to_fri=sat_ok)
        if obs:
            holidays.add(obs)
    for hh in (nth_weekday(year, 1, 0, 3), nth_weekday(year, 2, 0, 3),
               nth_weekday(year, 9, 0, 1), nth_weekday(year, 11, 3, 4)):
        if hh:
            holidays.add(hh)
    holidays.add(_easter_date(year) - timedelta(days=2))
    for day in range(31, 24, -1):
        try:
            dd = date(year, 5, day)
            if dd.weekday() == 0:
                holidays.add(dd)
                break
        except ValueError:
            pass
    return holidays


def get_market_status() -> dict:
    from zoneinfo import ZoneInfo
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    today = now_et.date()
    is_weekend = today.weekday() >= 5
    is_holiday = today in us_holidays(today.year)
    mo = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    mc = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    in_hours = mo <= now_et <= mc
    is_open = not is_weekend and not is_holiday and in_hours
    last_day = today
    if is_weekend or is_holiday or (not in_hours and now_et < mo):
        last_day = today - datetime.timedelta(days=1)
        while last_day.weekday() >= 5 or last_day in us_holidays(last_day.year):
            last_day -= datetime.timedelta(days=1)
    return {
        "is_open": is_open,
        "last_trading_date": last_day.isoformat(),
    }


# ────────────────────── Yahoo fallback ──────────────────────
def yahoo_closes(symbol: str, range_: str = "1y") -> Optional[pd.Series]:
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            resp = requests.get(
                f"https://{host}/v8/finance/chart/{symbol}",
                params={"range": range_, "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=HTTP_TIMEOUT_SEC,
            )
            if not resp.ok:
                continue
            result = resp.json()["chart"]["result"][0]
            ts = result.get("timestamp") or []
            closes = result["indicators"]["quote"][0].get("close") or []
            if not ts or not closes:
                continue
            s = pd.Series(closes, index=pd.to_datetime(ts, unit="s")).dropna()
            if not s.empty:
                return s
        except Exception as e:
            log.warning(f"yahoo fallback {symbol} ({host}): {e}")
            continue
    return None


# ────────────────────── 종가 / OHLC 페치 ──────────────────────
def _fetch_close_one(ticker: str, start: str) -> Optional[pd.DataFrame]:
    try:
        data = fdr.DataReader(ticker, start)
        if data is None or data.empty:
            raise ValueError("empty")
        data = data[~data.index.duplicated(keep="last")].sort_index()
        return data[["Close"]].rename(columns={"Close": f"{ticker}_Close"})
    except Exception:
        s = yahoo_closes(ticker, range_="2y")
        if s is None:
            log.warning(f"fetch failed (fdr+yahoo): {ticker}")
            return None
        s = s[s.index >= pd.Timestamp(start)]
        return s.to_frame(name=f"{ticker}_Close")


def _filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    spy_col = f"{X_ASSET_FIXED}_Close"
    if spy_col not in df.columns or df.empty:
        return df
    spy = df[spy_col]
    traded = (spy != spy.shift(1)) | (spy.index == spy.index[0])
    is_wkday = pd.Series(df.index.weekday < 5, index=df.index)
    return df[traded & is_wkday]


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df_w = df.resample("W-FRI").last().dropna(how="all")
    if not df_w.empty and df.index[-1] > df_w.index[-1]:
        df_w = pd.concat([df_w, df.iloc[[-1]]])
    return df_w


def _resample_weekly_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df_w = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna(how="all")
    last_day = df.index[-1]
    if not df_w.empty and last_day > df_w.index[-1]:
        week = df[df.index > df_w.index[-1]]
        if not week.empty:
            row = pd.DataFrame([{
                "Open": week["Open"].iloc[0], "High": week["High"].max(),
                "Low": week["Low"].min(), "Close": week["Close"].iloc[-1],
            }], index=[last_day])
            df_w = pd.concat([df_w, row])
    return df_w


@ttl_cache(DATA_TTL_SEC)
def fetch_all_closes(tickers: tuple, start: str, candle: str = "일봉") -> pd.DataFrame:
    all_tickers = [X_ASSET_FIXED] + [t for t in tickers if t != X_ASSET_FIXED]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCH) as ex:
        results = list(ex.map(lambda t: _fetch_close_one(t, start), all_tickers))
    frames = [f for f in results if f is not None]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).ffill()
    df = _filter_trading_days(df)
    return _resample_weekly(df) if candle == "주봉" else df


@ttl_cache(DATA_TTL_SEC)
def fetch_ohlc(ticker: str, start: str, candle: str = "일봉") -> pd.DataFrame:
    try:
        data = fdr.DataReader(ticker, start)
        if data is None or data.empty:
            return pd.DataFrame()
        data = data[~data.index.duplicated(keep="last")].sort_index()
        cols = [c for c in ["Open", "High", "Low", "Close"] if c in data.columns]
        if len(cols) < 4:
            return pd.DataFrame()
        df = data[cols][data.index.weekday < 5].copy()
        return _resample_weekly_ohlc(df) if candle == "주봉" else df
    except Exception as e:
        log.warning(f"fetch_ohlc {ticker}: {e}")
        return pd.DataFrame()


# ────────────────────── 거시 지표 / 시장 체제 ──────────────────────
@ttl_cache(3600)
def get_market_regime(end_date_str: Optional[str] = None) -> dict:
    end = pd.Timestamp(end_date_str) if end_date_str else pd.Timestamp.today()
    start = end - pd.Timedelta(days=500)
    spy_close = None
    try:
        spy = fdr.DataReader("SPY", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if spy is not None and not spy.empty and "Close" in spy.columns:
            spy_close = spy["Close"].dropna()
    except Exception:
        pass
    if spy_close is None or len(spy_close) < 200:
        s = yahoo_closes("SPY", range_="2y")
        if s is not None:
            spy_close = s[s.index <= end]
    if spy_close is None or len(spy_close) < 200:
        return {"regime": "unknown", "spy_ret_6m": None, "spy_ret_1d": None}
    close = float(spy_close.iloc[-1])
    sma200 = float(spy_close.rolling(200).mean().iloc[-1])
    ret_6m = close / float(spy_close.iloc[-126]) - 1 if len(spy_close) >= 126 else None
    ret_1d = (close / float(spy_close.iloc[-2]) - 1
              if len(spy_close) >= 2 and float(spy_close.iloc[-2]) > 0 else None)
    above = close > sma200
    if ret_6m is None:
        regime = "neutral"
    elif above and ret_6m > 0.05:
        regime = "bull"
    elif (not above) and ret_6m < -0.10:
        regime = "bear"
    elif (not above) and ret_6m <= 0.0:
        regime = "correction"
    else:
        regime = "neutral"
    return {
        "regime": regime, "spy_ret_6m": ret_6m, "spy_ret_1d": ret_1d,
        "spy_close": close, "spy_above_sma200": above,
    }


@ttl_cache(3600)
def get_macro_indicators(end_date_str: Optional[str] = None) -> dict:
    end = pd.Timestamp(end_date_str) if end_date_str else pd.Timestamp.today()
    start = (end - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    def _last(fdr_syms, yahoo_syms):
        for sym in fdr_syms:
            try:
                d = fdr.DataReader(sym, start, end_s)
                if d is not None and not d.empty and "Close" in d.columns:
                    v = float(d["Close"].iloc[-1])
                    if pd.notna(v):
                        return v
            except Exception:
                continue
        for sym in yahoo_syms:
            s = yahoo_closes(sym, range_="1y")
            if s is not None:
                s = s[s.index <= end]
                if not s.empty:
                    return float(s.iloc[-1])
        return None

    vix = _last(["VIX", "^VIX"], ["^VIX"])
    us10y = _last(["^TNX", "US10YT=X", "US10Y", "TNX"], ["^TNX"])
    if us10y is not None and us10y > 50:
        us10y /= 10.0
    return {"vix": vix, "us10y": us10y, "usdkrw": _last(["USD/KRW"], ["KRW=X"])}


@ttl_cache(86400)
def korean_stock_names() -> dict:
    try:
        df = fdr.StockListing("KRX")
        if df is None or df.empty:
            return {}
        code_col = next((c for c in ("Code", "Symbol", "code", "symbol") if c in df.columns), None)
        name_col = next((c for c in ("Name", "name", "종목명") if c in df.columns), None)
        if not code_col or not name_col:
            return {}
        return {str(c).zfill(6): str(n) for c, n in zip(df[code_col], df[name_col])
                if pd.notna(c) and pd.notna(n)}
    except Exception:
        return {}


def clear_all_caches() -> None:
    for fn in (fetch_all_closes, fetch_ohlc, get_market_regime,
               get_macro_indicators, korean_stock_names):
        fn.cache_clear()

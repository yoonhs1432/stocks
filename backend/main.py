"""FastAPI 백엔드 — 네이티브 Android 클라이언트용 REST API.

분석 로직은 analysis.py(순수), 시세는 data.py, 영속화는 store.py.
실행:  uvicorn main:app --host 0.0.0.0 --port 8000
개인용이라 인증 없음 — 신뢰된 네트워크/사이드로드 전제.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import analysis as A
import data as D
import store as S

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("quant.api")

app = FastAPI(title="Quant Dashboard API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

WARMUP_DAYS = 60
DEFAULT_LOOKBACK_DAYS = 730


# ────────────────────── 공통 헬퍼 ──────────────────────
def _resolve_dates(start: Optional[str]) -> tuple[str, str]:
    """(display_start, fetch_start). fetch는 warmup 60일 앞당김."""
    if start:
        try:
            ds = pd.Timestamp(start)
        except Exception:
            raise HTTPException(400, f"bad start date: {start}")
    else:
        ds = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=DEFAULT_LOOKBACK_DAYS))
    fetch = (ds - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    return ds.strftime("%Y-%m-%d"), fetch


def _clean(v):
    """NaN/inf → None (JSON 안전)."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _series_list(s: pd.Series) -> list:
    return [_clean(x) for x in s.values]


# ────────────────────── 메타 ──────────────────────
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat()}


@app.get("/market")
def market(asof: Optional[str] = None):
    """장 상태 + 시장 체제(SPY) + 거시 지표(VIX/10Y/KRW)."""
    return {
        "status": D.get_market_status(),
        "regime": D.get_market_regime(asof),
        "macro": D.get_macro_indicators(asof),
    }


@app.get("/tickers")
def tickers():
    trades = S.load_trades()
    overrides = S.load_settings().get("display_name_overrides", {})
    krx = D.korean_stock_names()
    tks = S.load_tickers()
    state = A.build_portfolio_state(trades)
    out = []
    for tk in tks:
        cyc = state.get(tk, {}).get("cycle", {})
        out.append({
            "ticker": tk,
            "name": S.display_name(tk, overrides, krx),
            "holding": bool(cyc.get("hold_qty", 0) > 0),
            "has_history": bool(trades.get(tk)),
        })
    return {"tickers": out}


# ────────────────────── 전 종목 요약 (비교 표 / 산점도) ──────────────────────
@app.get("/overview")
def overview(start: Optional[str] = None, candle: str = "일봉", asof: Optional[str] = None):
    display_start, fetch_start = _resolve_dates(start)
    tks = S.load_tickers()
    df = D.fetch_all_closes(tuple(tks), fetch_start, candle)
    if df.empty:
        raise HTTPException(503, "시세 데이터를 가져오지 못했습니다")
    if asof:
        df = df[df.index <= pd.Timestamp(asof)]

    betas = A.compute_spy_betas(df, tks)
    overrides = S.load_settings().get("display_name_overrides", {})
    krx = D.korean_stock_names()
    state = A.build_portfolio_state(S.load_trades())
    spy_col = f"{A.X_ASSET_FIXED}_Close"
    df_x = df[[spy_col]]

    rows = []
    for tk in tks:
        col = f"{tk}_Close"
        if col not in df.columns:
            continue
        closes = df[col].dropna()
        if len(closes) < 2:
            continue
        res = A.process_asset_data(df_x, df[[col]], A.X_ASSET_FIXED, tk)
        if res[0] is None:
            continue
        tdf = res[0]
        cur = float(closes.iloc[-1])
        prev_d = float(closes.iloc[-2])
        prev_w = float(closes.iloc[-6]) if len(closes) > 5 else prev_d
        high_v = float(closes.max())
        z_raw = _clean(tdf["Z_Score"].iloc[-1]) or 0.0
        ms, ds = A.last_m_stds(tdf)
        m_raw = A.compute_momentum_score_smooth(
            _clean(tdf["MACD_Pct"].iloc[-1]) or 0.0,
            _clean(tdf["dMACD_Pct"].iloc[-1]) or 0.0,
            _clean(tdf["RSI"].iloc[-1]) or 50.0, ms, ds,
        )
        cyc = state.get(tk, {}).get("cycle", {})
        rows.append({
            "ticker": tk,
            "name": S.display_name(tk, overrides, krx),
            "price": cur,
            "day_pct": (cur / prev_d - 1) * 100 if prev_d > 0 else 0,
            "week_pct": (cur / prev_w - 1) * 100 if prev_w > 0 else 0,
            "from_high": (cur / high_v - 1) * 100 if high_v > 0 else 0,
            "z_pct": A.z_to_pct(z_raw),
            "m_pct": A.z_to_pct(m_raw),
            "signal": A.pct_to_signal(A.z_to_pct(m_raw)),
            "beta_spy": _clean(betas.get(tk)),
            "holding": bool(cyc.get("hold_qty", 0) > 0),
        })
    return {"rows": rows, "as_of": (asof or df.index[-1].strftime("%Y-%m-%d"))}


# ────────────────────── 단일 종목 분석 (차트) ──────────────────────
@app.get("/analysis/{ticker}")
def analysis(ticker: str, start: Optional[str] = None, candle: str = "일봉",
             asof: Optional[str] = None):
    ticker = ticker.strip().upper()
    display_start, fetch_start = _resolve_dates(start)
    tks = S.load_tickers()
    req = tuple(dict.fromkeys(tks + [ticker]))
    df = D.fetch_all_closes(req, fetch_start, candle)
    if df.empty or f"{ticker}_Close" not in df.columns:
        raise HTTPException(404, f"{ticker} 데이터 없음")
    if asof:
        df = df[df.index <= pd.Timestamp(asof)]

    res = A.process_asset_data(
        df[[f"{A.X_ASSET_FIXED}_Close"]], df[[f"{ticker}_Close"]],
        A.X_ASSET_FIXED, ticker,
    )
    if res[0] is None:
        raise HTTPException(422, "분석 데이터 부족")
    tdf, beta, std_resid = res

    # 표시 구간 마스킹 (warmup 제외)
    disp = tdf[tdf.index >= pd.Timestamp(display_start)].copy()
    if disp.empty:
        disp = tdf

    z_pct = A.z_to_pct  # 스칼라
    z_series = ((disp["Z_Score"].fillna(0) + 2.5) / 5.0 * 100).clip(0, 100)
    m_raw = A.compute_momentum_series(disp)
    m_series = ((m_raw + 2.5) / 5.0 * 100).clip(0, 100)

    # 현재 시점 expanding σ (액션/위치 계산용)
    log_resid = (np.log(disp[f"{ticker}_Norm"]) - np.log(disp["Predicted"])).dropna()
    exp_std = log_resid.expanding(min_periods=A.CFG.EXPANDING_MIN_PERIODS).std().dropna()
    sigma_unit = float(exp_std.iloc[-1]) if len(exp_std) and exp_std.iloc[-1] > 0 else std_resid

    cz = _clean(disp["Z_Score"].iloc[-1]) or 0.0
    mhz = _clean(disp["MACD_Hist_Z"].iloc[-1]) or 0.0
    rsi_last = _clean(disp["RSI"].iloc[-1]) or 50.0
    ms, ds = A.last_m_stds(disp)
    m_last = A.compute_momentum_score_smooth(
        _clean(disp["MACD_Pct"].iloc[-1]) or 0.0,
        _clean(disp["dMACD_Pct"].iloc[-1]) or 0.0, rsi_last, ms, ds,
    )

    dates = [d.strftime("%Y-%m-%d") for d in disp.index]
    band_up = np.exp(np.log(disp["Predicted"]) + 1.5 * std_resid)
    band_lo = np.exp(np.log(disp["Predicted"]) - 1.5 * std_resid)

    # OHLC (캔들) — 같은 표시 구간으로 맞춤
    ohlc_payload = None
    odf = D.fetch_ohlc(ticker, fetch_start, candle)
    if odf is not None and not odf.empty:
        if asof:
            odf = odf[odf.index <= pd.Timestamp(asof)]
        odf = odf[odf.index >= pd.Timestamp(display_start)]
        if not odf.empty:
            ohlc_payload = {
                "dates": [d.strftime("%Y-%m-%d") for d in odf.index],
                "open": _series_list(odf["Open"]),
                "high": _series_list(odf["High"]),
                "low": _series_list(odf["Low"]),
                "close": _series_list(odf["Close"]),
            }

    return {
        "ticker": ticker,
        "summary": {
            "beta": _clean(beta),
            "sigma_pct": _clean((np.exp(sigma_unit) - 1) * 100) if sigma_unit else None,
            "price": _clean(disp[f"{ticker}_Close"].iloc[-1]),
            "z_pct": z_pct(cz),
            "m_pct": z_pct(m_last),
            "signal": A.pct_to_signal(z_pct(m_last)),
            "combined_signal": A.get_signal_combined(cz, mhz, rsi_last),
            "rsi": rsi_last,
        },
        "series": {
            "dates": dates,
            "ticker_norm": _series_list(disp[f"{ticker}_Norm"]),
            "spy_norm": _series_list(disp[f"{A.X_ASSET_FIXED}_Norm"]),
            "predicted": _series_list(disp["Predicted"]),
            "band_upper": _series_list(band_up),
            "band_lower": _series_list(band_lo),
            "price": _series_list(disp[f"{ticker}_Close"]),
            "z_pct": _series_list(z_series),
            "m_pct": _series_list(m_series),
            "macd": _series_list(disp["MACD"]),
            "macd_signal": _series_list(disp["MACD_Signal"]),
            "rsi": _series_list(disp["RSI"]),
        },
        "ohlc": ohlc_payload,
        "trades": S.load_trades().get(ticker, []),
    }


# ────────────────────── 포트폴리오 ──────────────────────
@app.get("/portfolio")
def portfolio(start: Optional[str] = None, candle: str = "일봉"):
    display_start, fetch_start = _resolve_dates(start)
    trades = S.load_trades()
    state = A.build_portfolio_state(trades)
    tks = list(dict.fromkeys(S.load_tickers() + list(trades.keys())))
    df = D.fetch_all_closes(tuple(tks), fetch_start, candle)
    if df.empty:
        raise HTTPException(503, "시세 데이터를 가져오지 못했습니다")
    df_disp = df[df.index >= pd.Timestamp(display_start)]
    last_close = df.iloc[-1].to_dict()

    seed = S.load_settings().get("seed_usd", A.CFG.SEED_USD)
    equity = A.compute_portfolio_equity(df_disp, trades)
    dd = A.compute_drawdown(equity, seed)
    total_pnl = A.calc_portfolio_total_pnl(state, last_close)

    overrides = S.load_settings().get("display_name_overrides", {})
    krx = D.korean_stock_names()

    holdings, realized = [], []
    for tk, ts in state.items():
        cyc = ts["cycle"]
        name = S.display_name(tk, overrides, krx)
        real = ts["cumulative_pnl"] + (cyc["current_pnl"] or 0.0)
        if real != 0.0:
            realized.append({"ticker": tk, "name": name, "realized": real})
        if cyc["hold_qty"] > 0 and cyc["buy_qty"] > 0:
            avg = cyc["buy_cost"] / cyc["buy_qty"]
            cur = last_close.get(f"{tk}_Close", avg)
            qty = cyc["hold_qty"]
            holdings.append({
                "ticker": tk, "name": name, "qty": qty,
                "avg_price": avg, "cur_price": float(cur),
                "eval": float(cur) * qty, "pnl": (float(cur) - avg) * qty,
                "ret_pct": (float(cur) / avg - 1) * 100 if avg > 0 else 0,
            })
    return {
        "seed_usd": seed,
        "total_pnl": total_pnl,
        "drawdown": dd,
        "holdings": sorted(holdings, key=lambda h: -h["eval"]),
        "realized": sorted(realized, key=lambda r: -r["realized"]),
        "equity": ([{"date": d.strftime("%Y-%m-%d"), "pnl": _clean(v)}
                    for d, v in equity.items()] if equity is not None else []),
    }


# ────────────────────── 매매 기록 CRUD ──────────────────────
class TradeIn(BaseModel):
    ticker: str
    date: str       # YYYY-MM-DD
    type: str       # buy | sell
    qty: int
    price: float
    memo: Optional[str] = None


@app.get("/trades")
def get_trades(ticker: Optional[str] = None):
    trades = S.load_trades()
    if ticker:
        return {"trades": trades.get(ticker.strip().upper(), [])}
    return {"trades": trades}


@app.post("/trades")
def add_trade(t: TradeIn):
    if t.type not in ("buy", "sell"):
        raise HTTPException(400, "type은 buy/sell")
    if t.qty <= 0 or t.price <= 0:
        raise HTTPException(400, "수량·단가는 0보다 커야 함")
    try:
        datetime.date.fromisoformat(t.date)
    except ValueError:
        raise HTTPException(400, "날짜 형식 YYYY-MM-DD")
    tk = t.ticker.strip().upper()
    rec = {"date": t.date, "type": t.type, "qty": t.qty, "price": t.price}
    if t.memo and t.memo.strip():
        rec["memo"] = t.memo.strip()
    trades = S.load_trades()
    trades.setdefault(tk, []).append(rec)
    S.save_trades(trades)
    return {"ok": True, "count": len(trades[tk])}


@app.delete("/trades/{ticker}/{idx}")
def delete_trade(ticker: str, idx: int):
    tk = ticker.strip().upper()
    trades = S.load_trades()
    recs = trades.get(tk, [])
    if not (0 <= idx < len(recs)):
        raise HTTPException(404, "기록 인덱스 범위 밖")
    recs.pop(idx)
    S.save_trades(trades)
    return {"ok": True, "count": len(recs)}


@app.post("/refresh")
def refresh():
    """시세 캐시 강제 비움 (장중 가격 갱신)."""
    D.clear_all_caches()
    return {"ok": True}

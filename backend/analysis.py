"""순수 퀀트 분석 코어 — app.py에서 포팅 (Streamlit/Plotly/HTML 의존 없음).

app.py는 CLAUDE.md 규칙상 단일 파일로 유지되므로, 백엔드는 동일 수식을
이 모듈에 별도 포팅한다. 수식을 바꾸면 app.py와 이 파일을 함께 수정할 것.
출처(app.py 기준): process_asset_data / compute_momentum_* / extract_cycles_avgs /
compute_cycle_stats / _resolve_all_cycles / compute_portfolio_equity /
compute_drawdown / compute_spy_betas (분석 캐시 _version=9).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

X_ASSET_FIXED = "SPY"


@dataclass(frozen=True)
class Config:
    """매직 넘버 — app.py Config의 분석 관련 필드 미러."""
    SEED_USD: float = 20_000.0
    USD_KRW_FALLBACK: float = 1400.0

    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    Z_HIGH: float = 1.5
    MACD_HIGH: float = 2.0
    SCORE_MAX: int = 4

    # 모멘텀 M
    M_W_HEIGHT: float = 0.30
    M_W_INFLECT: float = 0.15
    M_W_RSI: float = 0.55
    M_VOL_WINDOW: int = 120
    M_SIGMA_SCALE: float = 1.5
    M_RSI_SCALE: float = 30.0

    EXPANDING_MIN_PERIODS: int = 30


CFG = Config()


# ====================================================
# 신호 / 백분위
# ====================================================
def z_to_pct(z: float) -> float:
    """Z/M 점수 → 0~100 백분위 (Z=-2.5→0, 0→50, +2.5→100)."""
    pct = (z + 2.5) / 5.0 * 100
    return max(0.0, min(100.0, pct))


def compute_combined_score(cz: float, mhz: float, rsi: float) -> int:
    s = 0
    s += -2 if cz <= -CFG.Z_HIGH else -1 if cz < 0 else 2 if cz >= CFG.Z_HIGH else 1
    s += -2 if mhz <= -CFG.MACD_HIGH else -1 if mhz < 0 else 2 if mhz >= CFG.MACD_HIGH else 1
    s += -2 if rsi <= CFG.RSI_OVERSOLD else -1 if rsi < 50 else 2 if rsi >= CFG.RSI_OVERBOUGHT else 1
    return s


def score_to_signal(score: int) -> str:
    if score <= -5: return "FB2"
    if score <= -3: return "FB"
    if score <= -1: return "B"
    if score >= 5:  return "FS2"
    if score >= 3:  return "FS"
    if score >= 1:  return "S"
    return "H"


def get_signal_combined(cz: float, mhz: float, rsi: float) -> str:
    return score_to_signal(compute_combined_score(cz, mhz, rsi))


def pct_to_signal(m_pct: float) -> str:
    """백분위(0~100) → 5단계 신호 라벨 (색은 클라이언트가 매핑)."""
    if m_pct < 20: return "strong_buy"
    if m_pct < 40: return "buy"
    if m_pct < 60: return "hold"
    if m_pct < 80: return "sell"
    return "strong_sell"


# ====================================================
# 모멘텀 M (app.py와 동일 수식)
# ====================================================
def compute_momentum_score_smooth(
    macd_pct: float, dmacd_pct: float, rsi: float,
    macd_std: Optional[float] = None, dmacd_std: Optional[float] = None,
) -> float:
    if macd_std is not None and macd_std > 0:
        h = macd_pct / (CFG.M_SIGMA_SCALE * macd_std)
    else:
        h = macd_pct / 2.0
    if dmacd_std is not None and dmacd_std > 0:
        d = dmacd_pct / (CFG.M_SIGMA_SCALE * dmacd_std)
    else:
        d = dmacd_pct / 0.5
    h = max(-1.0, min(1.0, h))
    d = max(-1.0, min(1.0, d))
    r = max(-1.0, min(1.0, (rsi - 50) / CFG.M_RSI_SCALE))
    return 2.5 * (CFG.M_W_HEIGHT * h + CFG.M_W_INFLECT * d + CFG.M_W_RSI * r)


def compute_momentum_series(df: pd.DataFrame) -> pd.Series:
    macd_pct = df["MACD_Pct"].fillna(0)
    dmacd_pct = df["dMACD_Pct"].fillna(0)
    rsi = df["RSI"].fillna(50)
    if "MACD_Pct_Std" in df.columns:
        h = (macd_pct / (CFG.M_SIGMA_SCALE * df["MACD_Pct_Std"].replace(0, np.nan))).fillna(macd_pct / 2.0)
    else:
        h = macd_pct / 2.0
    if "dMACD_Pct_Std" in df.columns:
        d = (dmacd_pct / (CFG.M_SIGMA_SCALE * df["dMACD_Pct_Std"].replace(0, np.nan))).fillna(dmacd_pct / 0.5)
    else:
        d = dmacd_pct / 0.5
    h = h.clip(-1.0, 1.0)
    d = d.clip(-1.0, 1.0)
    r = ((rsi - 50) / CFG.M_RSI_SCALE).clip(-1.0, 1.0)
    return 2.5 * (CFG.M_W_HEIGHT * h + CFG.M_W_INFLECT * d + CFG.M_W_RSI * r)


def last_m_stds(df: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    ms = ds = None
    if "MACD_Pct_Std" in df.columns and pd.notna(df["MACD_Pct_Std"].iloc[-1]):
        ms = float(df["MACD_Pct_Std"].iloc[-1])
    if "dMACD_Pct_Std" in df.columns and pd.notna(df["dMACD_Pct_Std"].iloc[-1]):
        ds = float(df["dMACD_Pct_Std"].iloc[-1])
    return ms, ds


# ====================================================
# 종목 분석 (회귀 + Z + RSI + MACD + M)
# ====================================================
def process_asset_data(
    df_x: pd.DataFrame, df_y: pd.DataFrame, x_name: str, y_name: str
) -> tuple:
    """회귀(numpy.polyfit) + expanding-std Z-Score. app.py와 동일."""
    df = pd.merge(df_x, df_y, left_index=True, right_index=True).dropna().sort_index()
    if df.empty:
        return (None,) * 3

    base_x = df[f"{x_name}_Close"].iloc[0]
    base_y = df[f"{y_name}_Close"].iloc[0]
    df[f"{x_name}_Norm"] = df[f"{x_name}_Close"] / base_x
    df[f"{y_name}_Norm"] = df[f"{y_name}_Close"] / base_y

    log_x = np.log(df[f"{x_name}_Norm"].values)
    log_y = np.log(df[f"{y_name}_Norm"].values)
    beta, intercept = np.polyfit(log_x, log_y, 1)
    df["Predicted"] = np.exp(intercept) * df[f"{x_name}_Norm"] ** beta

    close = df[f"{y_name}_Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["EMA26"] = ema26
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    exp_std_hist = df["MACD_Hist"].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std()
    exp_mean_hist = df["MACD_Hist"].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).mean()
    df["MACD_Hist_Z"] = (df["MACD_Hist"] - exp_mean_hist) / exp_std_hist.replace(0, np.nan)

    ema26_safe = ema26.replace(0, np.nan)
    df["MACD_Pct"] = (df["MACD"] / ema26_safe) * 100
    df["MACD_Hist_Pct"] = (df["MACD_Hist"] / ema26_safe) * 100

    dmacd_smooth = df["MACD"].diff().ewm(span=3, adjust=False).mean()
    df["dMACD_Raw_Pct"] = (dmacd_smooth / ema26_safe) * 100
    df["dMACD_Pct"] = -df["dMACD_Raw_Pct"]

    df["MACD_Pct_Std"] = df["MACD_Pct"].rolling(
        CFG.M_VOL_WINDOW, min_periods=CFG.EXPANDING_MIN_PERIODS).std()
    df["dMACD_Pct_Std"] = df["dMACD_Pct"].rolling(
        CFG.M_VOL_WINDOW, min_periods=CFG.EXPANDING_MIN_PERIODS).std()

    log_resid = np.log(df[f"{y_name}_Norm"]) - np.log(df["Predicted"])
    std_resid = log_resid.std()
    df["Z_Score"] = (
        log_resid
        / log_resid.expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std().replace(0, np.nan)
    )
    return df, float(beta), float(std_resid)


# ====================================================
# 사이클 통계
# ====================================================
def compute_cycle_stats(records: list) -> Optional[dict]:
    """완료된 사이클(0주→매수→…→0주) 통계. app.py와 동일."""
    valid = [r for r in records if r.get("qty", 0) > 0 and r.get("price", 0) > 0]
    if not valid:
        return None
    sorted_recs = sorted(valid, key=lambda r: r["date"])
    cycles = []
    hold_qty = buy_qty = 0
    buy_cost = sell_proceeds = 0.0
    cycle_start: Optional[datetime.date] = None
    for r in sorted_recs:
        date = datetime.date.fromisoformat(r["date"])
        qty = int(r["qty"])
        if r["type"] == "buy":
            if hold_qty == 0:
                cycle_start = date
                buy_qty = 0
                buy_cost = sell_proceeds = 0.0
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * r["price"]
        elif r["type"] == "sell" and hold_qty > 0:
            sell_proceeds += qty * r["price"]
            hold_qty = max(hold_qty - qty, 0)
            if hold_qty == 0 and buy_qty > 0:
                pnl = sell_proceeds - buy_cost
                cycles.append({
                    "start": cycle_start.isoformat() if cycle_start else None,
                    "end": date.isoformat(),
                    "ret_pct": pnl / buy_cost * 100,
                    "pnl": pnl,
                    "hold_days": (date - cycle_start).days if cycle_start else 0,
                })
    if not cycles:
        return None
    wins = [c for c in cycles if c["ret_pct"] > 0]
    losses = [c for c in cycles if c["ret_pct"] <= 0]
    total_gain = sum(c["pnl"] for c in wins)
    total_loss = abs(sum(c["pnl"] for c in losses))
    best = max(cycles, key=lambda c: c["ret_pct"])
    worst = min(cycles, key=lambda c: c["ret_pct"])
    return {
        "count": len(cycles),
        "win_rate": len(wins) / len(cycles) * 100,
        "avg_ret_pct": sum(c["ret_pct"] for c in cycles) / len(cycles),
        "avg_hold_days": sum(c["hold_days"] for c in cycles) / len(cycles),
        "profit_factor": (total_gain / total_loss) if total_loss > 0 else None,
        "best_pct": best["ret_pct"], "worst_pct": worst["ret_pct"],
        "best_date": best["end"], "worst_date": worst["end"],
        "cycles": cycles,
    }


# ====================================================
# 포트폴리오 상태 (사이클 해석)
# ====================================================
def resolve_all_cycles(valid: list) -> tuple[dict, float]:
    """매매 기록 → 현재 사이클 + 누적 실현손익. app.py _resolve_all_cycles."""
    sorted_records = sorted(valid, key=lambda r: r["date"])
    cycle_start = cycle_end = None
    hold_qty = buy_qty = 0
    buy_cost = sell_proceeds = cumulative_pnl = realized_partial = 0.0
    has_any_sell = False
    for r in sorted_records:
        date = datetime.date.fromisoformat(r["date"])
        qty = int(r["qty"])
        if r["type"] == "buy":
            if hold_qty == 0:
                if cycle_start is not None and cycle_end is not None:
                    cumulative_pnl += sell_proceeds - buy_cost
                cycle_start = date
                cycle_end = None
                buy_qty = 0
                buy_cost = sell_proceeds = realized_partial = 0.0
                has_any_sell = False
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * r["price"]
        elif r["type"] == "sell" and hold_qty > 0:
            avg_buy = buy_cost / buy_qty if buy_qty > 0 else 0
            realized_partial += qty * (r["price"] - avg_buy)
            has_any_sell = True
            sell_proceeds += qty * r["price"]
            hold_qty = max(hold_qty - qty, 0)
            if hold_qty == 0:
                cycle_end = date
    if cycle_end is not None:
        current_pnl = sell_proceeds - buy_cost
    elif has_any_sell:
        current_pnl = realized_partial
    else:
        current_pnl = None
    cyc = {
        "cycle_start": cycle_start.isoformat() if cycle_start else None,
        "cycle_end": cycle_end.isoformat() if cycle_end else None,
        "hold_qty": hold_qty, "buy_qty": buy_qty,
        "buy_cost": buy_cost, "current_pnl": current_pnl,
    }
    return cyc, cumulative_pnl


def build_portfolio_state(trade_history: dict) -> dict:
    state = {}
    for ticker, records in trade_history.items():
        valid = [r for r in records if r.get("qty", 0) > 0 and r.get("price", 0) > 0]
        if not valid:
            continue
        cyc, cum = resolve_all_cycles(valid)
        state[ticker] = {"cycle": cyc, "cumulative_pnl": cum}
    return state


def calc_portfolio_total_pnl(portfolio_state: dict, last_close: dict) -> float:
    """전 종목 (누적실현 + 현재평가) 합계. last_close: {f'{tk}_Close': price}."""
    total = 0.0
    for ticker, ts in portfolio_state.items():
        cyc = ts["cycle"]
        if cyc["buy_qty"] == 0:
            continue
        realized = ts["cumulative_pnl"] + (cyc["current_pnl"] or 0.0)
        unrealized = 0.0
        if cyc["hold_qty"] > 0:
            cur = last_close.get(f"{ticker}_Close")
            if cur is not None:
                avg = cyc["buy_cost"] / cyc["buy_qty"]
                unrealized = (float(cur) - avg) * cyc["hold_qty"]
        total += realized + unrealized
    return total


# ====================================================
# 자산 시계열 / 드로다운
# ====================================================
def compute_portfolio_equity(
    df_close: pd.DataFrame, trade_history: dict
) -> Optional[pd.Series]:
    if df_close.empty:
        return None
    all_events = []
    for ticker, records in trade_history.items():
        for r in records:
            if r.get("qty", 0) > 0 and r.get("price", 0) > 0:
                all_events.append({
                    "date": pd.to_datetime(r["date"]), "ticker": ticker,
                    "type": r["type"], "qty": int(r["qty"]), "price": float(r["price"]),
                })
    if not all_events:
        return None
    all_events.sort(key=lambda e: e["date"])
    holdings: dict[str, int] = {}
    realized_total = 0.0
    avg_costs: dict[str, float] = {}
    equity = pd.Series(index=df_close.index, dtype=float)
    event_idx = 0
    for date in df_close.index:
        while event_idx < len(all_events) and all_events[event_idx]["date"] <= date:
            ev = all_events[event_idx]
            tk, q, p = ev["ticker"], ev["qty"], ev["price"]
            cur_q = holdings.get(tk, 0)
            cur_avg = avg_costs.get(tk, 0.0)
            if ev["type"] == "buy":
                new_q = cur_q + q
                avg_costs[tk] = ((cur_avg * cur_q) + (p * q)) / new_q if new_q > 0 else 0
                holdings[tk] = new_q
            elif ev["type"] == "sell" and cur_q > 0:
                sq = min(q, cur_q)
                realized_total += (p - cur_avg) * sq
                holdings[tk] = cur_q - sq
                if holdings[tk] == 0:
                    avg_costs[tk] = 0
            event_idx += 1
        unrealized = 0.0
        for tk, q in holdings.items():
            if q == 0:
                continue
            col = f"{tk}_Close"
            if col in df_close.columns:
                px = df_close.loc[date, col]
                if pd.notna(px):
                    unrealized += (px - avg_costs.get(tk, 0)) * q
        equity.loc[date] = realized_total + unrealized
    return equity.dropna()


def compute_drawdown(equity: Optional[pd.Series], seed: float) -> dict:
    if equity is None or equity.empty:
        return {"current_dd": 0.0, "mdd": 0.0, "mdd_date": None}
    portfolio_value = equity + seed
    running_max = portfolio_value.cummax()
    dd = (portfolio_value - running_max) / running_max * 100
    mdd_date = dd.idxmin()
    return {
        "current_dd": float(dd.iloc[-1]),
        "mdd": float(dd.min()),
        "mdd_date": mdd_date.date().isoformat() if pd.notna(mdd_date) else None,
    }


def compute_spy_betas(df_close: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    spy_betas: dict[str, float] = {}
    spy_col = "SPY_Close"
    if spy_col not in df_close.columns:
        return spy_betas
    spy_price = df_close[spy_col].dropna()
    if len(spy_price) <= 10:
        return spy_betas
    log_spy = np.log(spy_price)
    for tk in tickers:
        col = f"{tk}_Close"
        if col not in df_close.columns:
            continue
        tk_price = df_close[col].dropna()
        tk_price = tk_price[tk_price > 0]
        common = log_spy.index.intersection(tk_price.index)
        if len(common) < 10:
            continue
        try:
            slope, _ = np.polyfit(log_spy.loc[common].values, np.log(tk_price.loc[common]).values, 1)
            if np.isfinite(slope):
                spy_betas[tk] = float(slope)
        except Exception:
            continue
    return spy_betas

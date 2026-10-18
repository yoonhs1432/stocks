"""Z·M 매매 타이밍 검증 — 회귀(Newey-West) + walk-forward 백테스트.

사용: python -I zm_study.py <web 디렉터리>
"""
import json, math, sys
from pathlib import Path
import numpy as np

WEB = Path(sys.argv[1])
sys.path.insert(0, str(WEB))
import quant  # noqa: E402

CANDLES = WEB / "data" / "candles"
TICKERS = json.load(open(WEB / "data" / "tickers.json", encoding="utf-8"))
HS = [5, 10, 20]
WARM = 120          # walk-forward 회귀 최소 표본


def load(sym):
    f = CANDLES / f"{sym}.1d.v2.json"
    if not f.exists():
        return None
    return [(b["t"], b["close"]) for b in json.load(open(f))]


def walk_forward_z(logx, logy, warm=WARM):
    """t 시점까지만으로 OLS → t 의 잔차 / t 까지 잔차 표준편차. 미래 정보 없음."""
    n = len(logx)
    z = np.full(n, np.nan)
    sx = sy = sxx = sxy = syy = 0.0
    for t in range(n):
        x, y = logx[t], logy[t]
        sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y
        k = t + 1
        if k < warm:
            continue
        cxx = sxx - sx * sx / k
        cxy = sxy - sx * sy / k
        cyy = syy - sy * sy / k
        if cxx <= 0:
            continue
        b = cxy / cxx
        a = sy / k - b * sx / k
        ssr = max(cyy - b * cxy, 0.0)
        sd = math.sqrt(ssr / (k - 2))
        if sd > 0:
            z[t] = (y - (a + b * x)) / sd
    return z


def nw_ols(x, y, lag):
    """y = a + b x, Newey-West(HAC) t-stat. 겹치는 수익률 구간 보정."""
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 40:
        return np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return beta[1], beta[1] / math.sqrt(V[1, 1]), n


def rank_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 40:
        return np.nan
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def fwd_ret(close, h):
    """신호는 t 종가에 나오고 **t+1 종가에 진입** → t+1+h 종가까지 로그수익률."""
    lc = np.log(close)
    out = np.full(len(close), np.nan)
    out[: len(close) - 1 - h] = lc[1 + h:] - lc[1:len(close) - h]
    return out


def backtest(close, score, lo, hi, start, end):
    """score<lo 면 다음날 종가에 매수, score>hi 면 다음날 종가에 매도. 롱/현금."""
    pos = 0
    lr = np.diff(np.log(close))           # lr[t] = t→t+1
    ret = 0.0; held = 0; trades = 0
    for t in range(start, end - 1):
        s = score[t]
        # t 종가 신호 → t+1 진입이므로 t→t+1 수익은 이전 포지션
        r = lr[t] if pos else 0.0
        ret += r; held += pos
        if not np.isnan(s):
            if pos == 0 and s < lo:
                pos = 1; trades += 1
            elif pos == 1 and s > hi:
                pos = 0
    bh = math.log(close[end - 1] / close[start])
    days = end - 1 - start
    return ret, bh, held / max(days, 1), trades


def main():
    spy = load("SPY")
    rows = []
    pooled = {h: {"zf": [], "zw": [], "m": [], "r": []} for h in HS}
    for tk in TICKERS:
        bars = load(tk)
        if not bars:
            print(f"# {tk}: 캐시 없음"); continue
        res = quant.analyze(spy, bars)
        if res is None:
            continue
        n = len(res.price)
        close = np.array(res.price)
        zf = (np.array(res.zPct) - 50) / 20          # 앱의 Z (전체기간 회귀, look-ahead)
        m = (np.array(res.mPct) - 50) / 20           # 앱의 M (인과적)
        lx = np.log(np.array(res.spyNorm)); ly = np.log(np.array(res.tickerNorm))
        zw = walk_forward_z(lx, ly)                   # walk-forward Z
        zw_pct = np.clip((zw + 2.5) / 5 * 100, 0, 100)
        m_pct = np.array(res.mPct)

        row = {"tk": tk, "n": n, "beta": res.beta}
        for h in HS:
            r = fwd_ret(close, h)
            ok = np.arange(n) >= WARM
            for name, v in (("zf", zf), ("zw", zw), ("m", m)):
                vv = np.where(ok, v, np.nan)
                b, t, _ = nw_ols(vv, r, lag=h)
                row[f"{name}_t{h}"] = t
                row[f"{name}_ic{h}"] = rank_ic(vv, r)
                pooled[h][name].append(np.where(ok, (vv - np.nanmean(vv)) / np.nanstd(vv), np.nan))
            # 종목별로 수익률 표준화 후 풀링
            pooled[h]["r"].append((r - np.nanmean(r)) / np.nanstd(r))

        # ── 백테스트: 앞 절반에서 문턱 고르고 뒤 절반에서 검증 ──
        mid = WARM + (n - WARM) // 2
        grid = [(lo, hi) for lo in (10, 20, 30, 40) for hi in (50, 60, 70, 80, 90) if hi > lo]
        for name, sc in (("M", m_pct), ("Zw", zw_pct), ("ZM", (zw_pct + m_pct) / 2)):
            best = max(grid, key=lambda g: backtest(close, sc, g[0], g[1], WARM, mid)[0])
            ins = backtest(close, sc, *best, WARM, mid)
            oos = backtest(close, sc, *best, mid, n)
            row[f"{name}_best"] = best
            row[f"{name}_is"] = ins[0]; row[f"{name}_oos"] = oos[0]
            row[f"{name}_oos_bh"] = oos[1]; row[f"{name}_oos_exp"] = oos[2]
            row[f"{name}_oos_tr"] = oos[3]
            # 전체 구간(참고, 과최적화 주의)
            full = max(grid, key=lambda g: backtest(close, sc, g[0], g[1], WARM, n)[0])
            fr = backtest(close, sc, *full, WARM, n)
            row[f"{name}_full_best"] = full; row[f"{name}_full"] = fr[0]
        row["bh_full"] = math.log(close[-1] / close[WARM])
        row["last_zapp"] = res.lastZpct; row["last_zw"] = float(zw_pct[-1]); row["last_m"] = res.lastMpct
        rows.append(row)

    json.dump(rows, open(Path(__file__).with_name("rows.json"), "w"), indent=1, default=float)

    print("\n=== 풀링(종목별 표준화) 예측회귀: 미래 h일 수익률 ~ 신호, NW t ===")
    for h in HS:
        r = np.concatenate(pooled[h]["r"])
        out = []
        for name in ("zf", "zw", "m"):
            x = np.concatenate(pooled[h][name])
            b, t, nn = nw_ols(x, r, lag=h)
            out.append(f"{name}: b={b:+.3f} t={t:+.2f} ic={rank_ic(x, r):+.3f}")
        print(f"h={h:2d}  " + "  |  ".join(out) + f"  (n={nn})")

    print("\n=== 종목별 20일 예측력 (rank IC / NW t) — 음수면 '높을수록 이후 하락' ===")
    print(f"{'tk':7s} {'β':>5s} | {'Z앱 ic':>7s} {'t':>6s} | {'Zwf ic':>7s} {'t':>6s} | {'M ic':>7s} {'t':>6s}")
    for r in rows:
        print(f"{r['tk']:7s} {r['beta']:5.2f} | {r['zf_ic20']:+7.3f} {r['zf_t20']:+6.2f} | "
              f"{r['zw_ic20']:+7.3f} {r['zw_t20']:+6.2f} | {r['m_ic20']:+7.3f} {r['m_t20']:+6.2f}")

    print("\n=== 백테스트 (로그수익) 뒤 절반 OOS: 전략 vs 보유 ===")
    print(f"{'tk':7s} | {'M best':>9s} {'oos':>6s} {'bh':>6s} {'exp':>4s} | {'Zw best':>9s} {'oos':>6s} | {'ZM best':>9s} {'oos':>6s} | now Zapp Zwf  M")
    for r in rows:
        print(f"{r['tk']:7s} | {str(r['M_best']):>9s} {r['M_oos']:+6.2f} {r['M_oos_bh']:+6.2f} {r['M_oos_exp']:4.0%} | "
              f"{str(r['Zw_best']):>9s} {r['Zw_oos']:+6.2f} | {str(r['ZM_best']):>9s} {r['ZM_oos']:+6.2f} | "
              f"{r['last_zapp']:4.0f} {r['last_zw']:4.0f} {r['last_m']:4.0f}")
    for name in ("M", "Zw", "ZM"):
        wins = sum(1 for r in rows if r[f"{name}_oos"] > r[f"{name}_oos_bh"])
        avg = np.mean([r[f"{name}_oos"] - r[f"{name}_oos_bh"] for r in rows])
        print(f"{name}: OOS 보유 대비 승 {wins}/{len(rows)}, 평균 초과 {avg:+.3f}")


if __name__ == "__main__":
    main()

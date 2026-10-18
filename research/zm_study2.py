"""2차 — 공적분(ADF), 굴리는 창 Z, 추세 필터, 실제 매매 시점의 Z/M."""
import json, math, sys, datetime as dt
from pathlib import Path
import numpy as np

WEB = Path(sys.argv[1])
sys.path.insert(0, str(WEB)); sys.path.insert(0, str(Path(__file__).parent))
import quant  # noqa: E402
from zm_study import load, walk_forward_z, nw_ols, rank_ic, fwd_ret, WARM, TICKERS  # noqa: E402


def adf_t(e, lags=1):
    """Engle-Granger 잔차 ADF t (상수 포함). 5% 임계 ≈ -3.37, 10% ≈ -3.07."""
    de = np.diff(e)
    y = de[lags:]
    cols = [np.ones(len(y)), e[lags:-1]]
    for L in range(1, lags + 1):
        cols.append(de[lags - L:-L])
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    V = s2 * np.linalg.inv(X.T @ X)
    return b[1] / math.sqrt(V[1, 1]), -math.log(2) / math.log(1 + b[1]) if -1 < b[1] < 0 else np.inf


def rolling_z(lx, ly, win):
    n = len(lx); z = np.full(n, np.nan)
    for t in range(win - 1, n):
        x = lx[t - win + 1:t + 1]; y = ly[t - win + 1:t + 1]
        b, a = np.polyfit(x, y, 1)
        e = y - (a + b * x)
        z[t] = e[-1] / e.std(ddof=2)
    return z


def max_dd(lr):
    eq = np.cumsum(lr)
    return float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0


def trend_bt(close, sma, start):
    lr = np.diff(np.log(close))
    ma = np.convolve(close, np.ones(sma) / sma, "valid")
    ma = np.concatenate([np.full(sma - 1, np.nan), ma])
    pos = (close > ma).astype(float)            # t 종가 판단 → t+1 부터 보유
    strat = pos[start:-1] * lr[start:]
    # 매매 비용 0.1% 왕복 반쪽씩
    sw = np.abs(np.diff(pos[start - 1:-1]))
    strat = strat - sw * 0.001
    return strat.sum(), lr[start:].sum(), max_dd(strat), max_dd(lr[start:]), pos[start:-1].mean()


def main():
    spy = load("SPY")
    trades = json.load(open(WEB / "data" / "trades.json", encoding="utf-8"))
    print(f"{'tk':7s} {'β_lvl':>6s} {'β_ret':>6s} {'ADF t':>6s} {'반감기':>5s} | IC20: {'expand':>7s} {'roll250':>8s} {'roll120':>8s} | "
          f"SMA50 oos {'전략':>6s} {'보유':>6s} {'MDD전략':>7s} {'MDD보유':>7s} {'보유율':>5s}")
    summ = []
    for tk in TICKERS:
        bars = load(tk)
        if not bars:
            continue
        res = quant.analyze(spy, bars)
        n = len(res.price); close = np.array(res.price)
        lx = np.log(np.array(res.spyNorm)); ly = np.log(np.array(res.tickerNorm))
        e = ly - (np.polyfit(lx, ly, 1) @ np.vstack([lx, np.ones(n)]))
        at, hl = adf_t(e)
        rb = np.polyfit(np.diff(lx), np.diff(ly), 1)[0]
        r20 = fwd_ret(close, 20)
        ok = np.arange(n) >= WARM
        zw = np.where(ok, walk_forward_z(lx, ly), np.nan)
        z250 = np.where(ok, rolling_z(lx, ly, 250), np.nan)
        z120 = np.where(ok, rolling_z(lx, ly, 120), np.nan)
        ic = [rank_ic(v, r20) for v in (zw, z250, z120)]
        tt = [nw_ols(v, r20, 20)[1] for v in (zw, z250, z120)]
        mid = WARM + (n - WARM) // 2
        tb = trend_bt(close, 50, mid)
        summ.append((tk, at, ic, tt, tb))
        print(f"{tk:7s} {res.beta:6.2f} {rb:6.2f} {at:6.2f} {hl:5.0f} | {ic[0]:+.2f}({tt[0]:+.1f}) {ic[1]:+.2f}({tt[1]:+.1f}) {ic[2]:+.2f}({tt[2]:+.1f}) | "
              f"{tb[0]:+6.2f} {tb[1]:+6.2f} {tb[2]:7.2f} {tb[3]:7.2f} {tb[4]:5.0%}")

        # 실제 매매 시점
        days = {d // 86400: i for i, d in enumerate(res.dates)}
        zwp = np.clip((walk_forward_z(lx, ly) + 2.5) / 5 * 100, 0, 100)
        for tr in trades.get(tk, []):
            d = dt.date.fromisoformat(tr["date"])
            k = (dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()) // 86400
            i = days.get(int(k))
            if i is None:
                continue
            fr = (close[i + 20] / tr["price"] - 1) if i + 20 < n else (close[-1] / tr["price"] - 1)
            tag = "" if i + 20 < n else f"(현재까지 {n - 1 - i}일)"
            TR.append((tk, tr["date"], tr["type"], zwp[i], res.mPct[i], fr, tag))

    print("\n5% 기준 공적분(ADF t < -3.37):", [s[0] for s in summ if s[1] < -3.37])
    print("10% 기준(< -3.07):", [s[0] for s in summ if s[1] < -3.07])
    w = sum(1 for s in summ if s[4][0] > s[4][1]); dd = sum(1 for s in summ if s[4][2] < s[4][3])
    print(f"SMA50 추세필터 OOS: 수익 승 {w}/{len(summ)}, MDD 개선 {dd}/{len(summ)}, "
          f"평균 초과 {np.mean([s[4][0] - s[4][1] for s in summ]):+.3f}")

    print("\n=== 실제 매매 시점의 Z(walk-forward)·M 과 그 뒤 20거래일 수익 ===")
    for t in sorted(TR, key=lambda x: x[1]):
        print(f"{t[1]} {t[0]:7s} {t[2]:4s} Z={t[3]:4.0f} M={t[4]:4.0f}  +20일 {t[5]:+6.1%} {t[6]}")
    for side in ("buy", "sell"):
        s = [t for t in TR if t[2] == side]
        if s:
            print(f"{side}: {len(s)}건, 평균 Z={np.mean([t[3] for t in s]):.0f} M={np.mean([t[4] for t in s]):.0f}, "
                  f"이후 20일 평균 {np.mean([t[5] for t in s]):+.1%}")


TR = []
if __name__ == "__main__":
    main()

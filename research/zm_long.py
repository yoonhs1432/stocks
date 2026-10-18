"""장기(3·6·12개월) Z·M 검증 — Yahoo 전체 기간 일봉(수정종가).

Z 는 앱과 같은 **2년(504봉) 창** 로그-로그 회귀를 그날까지 데이터로만 굴린다(미래 정보 없음).
사용: python -I zm_long.py <web 디렉터리> <yahoo 디렉터리>
"""
import json, math, sys
from pathlib import Path
import numpy as np

WEB = Path(sys.argv[1]); YH = Path(sys.argv[2])
sys.path.insert(0, str(WEB))
import quant  # noqa: E402

HS = [60, 120, 250]
WIN = 504
ACTUAL = ["SOXL", "TARK", "HIBL", "TQQQ", "DFEN", "DPST", "GDXU", "KORU", "LABU", "NAIL", "000660", "005930"]
PROXY = {"BITU": "BTC-USD", "SOXL": "SOXX", "TARK": "ARKK", "HIBL": "SPHB", "TQQQ": "QQQ",
         "DFEN": "ITA", "DPST": "KRE", "GDXU": "GDX", "WTIU": "USO", "KORU": "EWY",
         "BNKU": "KBE", "NRGU": "XLE", "LABU": "XBI", "NAIL": "ITB", "URAA": "URA"}


def load(sym):
    rows = json.load(open(YH / f"{sym}.json"))
    out = {}
    for t, c in rows:
        if c and c > 0:
            out[(int(t) + 9 * 3600) // 86400] = float(c)   # 한국 자정 봉도 같은 거래일로
    return out


def align(base, tk):
    days = sorted(set(base) & set(tk))
    return np.array(days), np.array([base[d] for d in days]), np.array([tk[d] for d in days])


def rolling_z(lx, ly, win):
    """창 안 OLS — 누적합으로 O(n)."""
    n = len(lx); z = np.full(n, np.nan)
    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    Sx, Sy, Sxx, Sxy, Syy = cs(lx), cs(ly), cs(lx * lx), cs(lx * ly), cs(ly * ly)
    for t in range(win - 1, n):
        a0, a1 = t + 1 - win, t + 1
        k = win
        sx, sy = Sx[a1] - Sx[a0], Sy[a1] - Sy[a0]
        cxx = Sxx[a1] - Sxx[a0] - sx * sx / k
        cxy = Sxy[a1] - Sxy[a0] - sx * sy / k
        cyy = Syy[a1] - Syy[a0] - sy * sy / k
        if cxx <= 1e-12:
            continue
        b = cxy / cxx; a = sy / k - b * sx / k
        sd = math.sqrt(max(cyy - b * cxy, 0) / (k - 2))
        if sd > 0:
            z[t] = (ly[t] - a - b * lx[t]) / sd
    return z


def m_score(y):
    """quant.analyze 의 M 과 같은 식(인과적). Z 계산을 빼서 빠르게."""
    n = len(y); y = list(y)
    rsi = quant._rsi(y)
    e12, e26 = quant._ema(y, 12), quant._ema(y, 26)
    macd = [e12[i] - e26[i] for i in range(n)]
    mp = [macd[i] / e26[i] * 100 for i in range(n)]
    dm = quant._ema([0.0] + [macd[i] - macd[i - 1] for i in range(1, n)], 3)
    dp = [-(dm[i] / e26[i] * 100) for i in range(n)]
    ms = quant._rolling_std(mp, quant.M_VOL_WINDOW, quant.EXPANDING_MIN)
    ds = quant._rolling_std(dp, quant.M_VOL_WINDOW, quant.EXPANDING_MIN)
    return np.array([quant._momentum(mp[i], dp[i], rsi[i], ms[i], ds[i]) for i in range(n)])


def fwd(close, h):
    lc = np.log(close); out = np.full(len(close), np.nan)
    out[: len(close) - 1 - h] = lc[1 + h:] - lc[1:len(close) - h]
    return out


def nw_t(x, y, lag):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; n = len(x)
    if n < 3 * lag:
        return np.nan
    X = np.column_stack([np.ones(n), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; Xe = X * e[:, None]; S = Xe.T @ Xe
    for L in range(1, lag + 1):
        G = Xe[L:].T @ Xe[:-L]; S += (1 - L / (lag + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X); V = Q @ S @ Q
    return b[1] / math.sqrt(V[1, 1])


def ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 100:
        return np.nan
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def pct(z):
    return np.clip((z + 2.5) / 5 * 100, 0, 100)


def strat(close, pos, cost=0.001):
    """pos[t] 는 t 종가 판단 → t+1 부터 보유. CAGR·MDD."""
    lr = np.diff(np.log(close))
    p = np.nan_to_num(pos[:-1])
    r = p * lr - np.abs(np.diff(np.concatenate([[0], p]))) * cost
    eq = np.cumsum(r); yrs = len(r) / 252
    mdd = 1 - math.exp(-float(np.max(np.maximum.accumulate(eq) - eq)))
    return math.exp(eq[-1] / yrs) - 1, mdd, p.mean()


def band_pos(score_pct, lo=20, hi=80):
    pos = np.zeros(len(score_pct)); cur = 0
    for t, s in enumerate(score_pct):
        if not np.isnan(s):
            if cur == 0 and s < lo: cur = 1
            elif cur == 1 and s > hi: cur = 0
        pos[t] = cur
    return pos


def study(name, base, tk):
    days, x, y = align(base, tk)
    n = len(y)
    if n < WIN + 300:
        return None
    lx, ly = np.log(x / x[0]), np.log(y / y[0])
    z = rolling_z(lx, ly, WIN)
    m = m_score(y)
    mom = np.full(n, np.nan); mom[250:] = ly[250:] - ly[:-250]
    ma200 = np.convolve(y, np.ones(200) / 200, "valid"); d200 = np.full(n, np.nan); d200[199:] = np.log(y[199:] / ma200)
    ok = np.arange(n) >= WIN
    sig = {k: np.where(ok, v, np.nan) for k, v in (("Z", z), ("M", m), ("MOM12", mom), ("D200", d200))}
    row = {"name": name, "years": (n - WIN) / 252, "start": int(days[WIN])}
    for h in HS:
        r = fwd(y, h)
        for k, v in sig.items():
            row[f"{k}_ic{h}"] = ic(v, r); row[f"{k}_t{h}"] = nw_t(v, r, h)
        # 앞/뒤 절반 부호 일관성
        idx = np.where(ok & ~np.isnan(r))[0]
        half = idx[len(idx) // 2] if len(idx) else 0
        for k in ("Z", "M"):
            a = ic(np.where(np.arange(n) < half, sig[k], np.nan), r)
            b = ic(np.where(np.arange(n) >= half, sig[k], np.nan), r)
            row[f"{k}_h1_{h}"] = a; row[f"{k}_h2_{h}"] = b
        # 구간별 조건부 평균 (연율 아님, h일 로그수익 → %)
        for k in ("Z", "M"):
            p = pct(sig[k]); base_mean = np.nanmean(r[ok])
            for lo, hi, lab in ((0, 20, "lo"), (80, 101, "hi")):
                msk = ok & (p >= lo) & (p < hi) & ~np.isnan(r)
                row[f"{k}_{lab}{h}"] = (math.exp(np.mean(r[msk])) - 1) if msk.sum() > 20 else np.nan
                row[f"{k}_{lab}{h}_n"] = int(msk.sum())
            row[f"all{h}"] = math.exp(base_mean) - 1
    # 전략 (문턱 고정 = 앱의 20/80, 학습 없음)
    sl = slice(WIN, n)
    yy = y[sl]
    row["bh"] = strat(yy, np.ones(len(yy)))
    row["zband"] = strat(yy, band_pos(pct(z[sl])))
    row["mband"] = strat(yy, band_pos(pct(m[sl])))
    row["sma200"] = strat(yy, (d200[sl] > 0).astype(float))
    row["last"] = {"Z": float(pct(z[-1:])[0]), "M": float(pct(m[-1:])[0]), "D200": float(d200[-1]),
                   "MOM12": float(mom[-1])}
    return row, sig, y, ok


def main():
    spy = load("SPY"); k200 = load("069500")
    rows = []; pool = {h: {k: [] for k in ("Z", "M", "MOM12", "D200", "r")} for h in HS}
    jobs = [(f"{a}", spy, a) for a in ACTUAL] + [(f"{p}({e})", spy, p) for e, p in PROXY.items()]
    jobs += [("000660/K200", k200, "000660"), ("005930/K200", k200, "005930")]
    seen_proxy = set()
    for label, base, sym in jobs:
        out = study(label, base, load(sym))
        if out is None:
            print(f"# {label}: 기간 부족"); continue
        row, sig, y, ok = out; rows.append(row)
        # 풀링은 실제 ETF + 기초자산 중복 없이(기초자산만) — 레버리지판과 겹치므로 기초자산·국내주식만
        if label in ACTUAL and not label[0].isdigit():
            continue
        if "/K200" in label:
            continue
        for h in HS:
            r = fwd(y, h); rr = np.where(ok, r, np.nan)
            pool[h]["r"].append((rr - np.nanmean(rr)) / np.nanstd(rr))
            for k in ("Z", "M", "MOM12", "D200"):
                v = sig[k]; pool[h][k].append((v - np.nanmean(v)) / np.nanstd(v))
    json.dump(rows, open(Path(__file__).with_name("long_rows.json"), "w"), default=float, indent=1)

    print("=== 풀링(기초자산 15 + 삼성·하이닉스, 종목별 표준화): 미래 h일 수익 ~ 신호 [IC, NW t] ===")
    for h in HS:
        r = np.concatenate(pool[h]["r"])
        print(f"h={h:3d}  " + "  ".join(f"{k}: {ic(np.concatenate(pool[h][k]), r):+.3f} ({nw_t(np.concatenate(pool[h][k]), r, h):+.1f})"
                                       for k in ("Z", "M", "MOM12", "D200")))

    print("\n=== 종목별 IC [6개월 / 12개월], ( ) 는 NW t ===")
    print(f"{'종목':16s} {'년':>4s} | {'Z 6m':>11s} {'Z 12m':>11s} | {'M 6m':>11s} {'M 12m':>11s} | {'MOM12 12m':>11s} | Z12m 앞/뒤절반")
    for r in rows:
        f = lambda k, h: f"{r[f'{k}_ic{h}']:+.2f}({r[f'{k}_t{h}']:+.1f})"
        print(f"{r['name']:16s} {r['years']:4.1f} | {f('Z',120):>11s} {f('Z',250):>11s} | {f('M',120):>11s} {f('M',250):>11s} | "
              f"{f('MOM12',250):>11s} | {r['Z_h1_250']:+.2f}/{r['Z_h2_250']:+.2f}")

    print("\n=== Z 구간별 이후 수익 (평균, 12개월 / 6개월) — 저Z(<20) vs 고Z(>80) vs 전체 ===")
    for r in rows:
        g = lambda k: "  n/a " if r[k] != r[k] else f"{r[k]:+6.0%}"
        print(f"{r['name']:16s} 12m 저Z {g('Z_lo250')} 고Z {g('Z_hi250')} 전체 {g('all250')} | "
              f"6m 저Z {g('Z_lo120')} 고Z {g('Z_hi120')} 전체 {g('all120')} | 저M12m {g('M_lo250')} 고M12m {g('M_hi250')}")

    print("\n=== 고정 규칙 전략 (CAGR / MDD / 보유율) ===")
    for r in rows:
        f = lambda k: f"{r[k][0]:+6.1%} {r[k][1]:5.0%} {r[k][2]:4.0%}"
        print(f"{r['name']:16s} 보유 {f('bh')} | Z20/80 {f('zband')} | M20/80 {f('mband')} | SMA200 {f('sma200')}")

    print("\n=== 현재 ===")
    for r in rows:
        L = r["last"]
        print(f"{r['name']:16s} Z {L['Z']:4.0f}  M {L['M']:4.0f}  200일선 {math.exp(L["D200"])-1:+.1%}  12개월 {math.exp(L['MOM12'])-1:+.1%}")


if __name__ == "__main__":
    main()

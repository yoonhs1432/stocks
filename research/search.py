"""매매 타이밍 신호 탐색 — 미리 정한 후보만, 발견(앞 절반)·검증(뒤 절반·레버리지) 분리.

평가: 롱/현금 전략의 샤프지수 − 그냥 보유 샤프. 샤프는 비중에 무관해서 '덜 들고 있어서 이긴' 효과가 빠진다.
비용 0.1%/회전. 신호는 t 종가로 판단 → t+1 수익에 적용.
사용: python -I search.py <yahoo dir> <web dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

YH = Path(sys.argv[1]); WEB = Path(sys.argv[2])
sys.path.insert(0, str(WEB))
import quant  # noqa: E402

UNDER = ["SPY", "QQQ", "SOXX", "SPHB", "ITA", "KRE", "KBE", "GDX", "USO", "EWY", "XLE", "XBI",
         "ITB", "URA", "ARKK", "BTC-USD", "005930", "000660", "069500"]
LEV = ["SOXL", "TQQQ", "HIBL", "DFEN", "DPST", "GDXU", "KORU", "LABU", "NAIL", "TARK"]
COST = 0.001
WARM = 504


def load(sym):
    rows = json.load(open(YH / f"{sym}.json"))
    s = {}
    for t, c in rows:
        if c and c > 0:
            s[(int(t) + 9 * 3600) // 86400] = float(c)
    return pd.Series(s).sort_index()


SPY = load("SPY")


def frame(sym):
    s = load(sym)
    idx = s.index.intersection(SPY.index)     # 코인 주말·국내 휴장 차이 정리
    return pd.DataFrame({"c": s[idx], "spy": SPY[idx]})


# ── 신호 재료 ──
def rsi(c, n):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)


def rolling_z(lx, ly, win):
    x, y = lx.values, ly.values
    n = len(x); z = np.full(n, np.nan)
    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    Sx, Sy, Sxx, Sxy, Syy = cs(x), cs(y), cs(x * x), cs(x * y), cs(y * y)
    for t in range(win - 1, n):
        a0, a1 = t + 1 - win, t + 1
        sx, sy = Sx[a1] - Sx[a0], Sy[a1] - Sy[a0]
        cxx = Sxx[a1] - Sxx[a0] - sx * sx / win; cxy = Sxy[a1] - Sxy[a0] - sx * sy / win
        cyy = Syy[a1] - Syy[a0] - sy * sy / win
        if cxx <= 1e-12: continue
        b = cxy / cxx; a = sy / win - b * sx / win
        sd = math.sqrt(max(cyy - b * cxy, 0) / (win - 2))
        if sd > 0: z[t] = (y[t] - a - b * x[t]) / sd
    return pd.Series(z, index=lx.index)


def m_score(c):
    v = [float(x) for x in c.values]; n = len(v)
    r = quant._rsi(v); e12, e26 = quant._ema(v, 12), quant._ema(v, 26)
    macd = [e12[i] - e26[i] for i in range(n)]
    mp = [macd[i] / e26[i] * 100 for i in range(n)]
    dm = quant._ema([0.0] + [macd[i] - macd[i - 1] for i in range(1, n)], 3)
    dp = [-(dm[i] / e26[i] * 100) for i in range(n)]
    ms = quant._rolling_std(mp, 120, 30); ds = quant._rolling_std(dp, 120, 30)
    return pd.Series([quant._momentum(mp[i], dp[i], r[i], ms[i], ds[i]) for i in range(n)], index=c.index)


def band(score, lo, hi):
    out = np.zeros(len(score)); cur = 0
    for i, s in enumerate(score.values):
        if s == s:
            if cur == 0 and s < lo: cur = 1
            elif cur == 1 and s > hi: cur = 0
        out[i] = cur
    return pd.Series(out, index=score.index)


def signals(df):
    c, spy = df["c"], df["spy"]
    r = c.pct_change()
    lc = np.log(c)
    sma = lambda n: c.rolling(n).mean()
    vol20 = r.rolling(20).std() * math.sqrt(252)
    vol60 = r.rolling(60).std() * math.sqrt(252)
    # 목표 변동성 = 그 시점까지의 장기 중앙값(미래 정보 없음)
    vt = vol20.expanding(252).median()
    S = {}
    S["보유"] = pd.Series(1.0, index=c.index)
    # ── 앱의 기존 신호 ──
    z504 = rolling_z(np.log(spy / spy.iloc[0]), np.log(c / c.iloc[0]), 504)
    m = m_score(c)
    S["앱 Z 20/80"] = band((z504 + 2.5) / 5 * 100, 20, 80)
    S["앱 M 20/80"] = band((m + 2.5) / 5 * 100, 20, 80)
    S["M 추세(M>0 보유)"] = (m > 0).astype(float)
    # ── Z 변형 ──
    z252 = rolling_z(np.log(spy / spy.iloc[0]), np.log(c / c.iloc[0]), 252)
    S["Z 1년창 20/80"] = band((z252 + 2.5) / 5 * 100, 20, 80)
    # 수익률 β 로 뺀 잔차 수익률 누적(잔차 모멘텀) 6개월
    rs = spy.pct_change()
    beta = r.rolling(252).cov(rs) / rs.rolling(252).var()
    resid = r - beta.shift(1) * rs
    S["잔차모멘텀 6개월>0"] = (resid.rolling(126).sum() > 0).astype(float)
    S["잔차역발상 6개월<0"] = (resid.rolling(126).sum() < 0).astype(float)
    # 자기 200일선 대비 이탈(σ) 역발상
    ownz = (lc - np.log(sma(200))) / (r.rolling(200).std() * math.sqrt(200))
    S["자기Z 역발상 20/80"] = band((ownz + 2.5) / 5 * 100, 20, 80)
    # ── 추세 ──
    S["200일선 위"] = (c > sma(200)).astype(float)
    S["100일선 위"] = (c > sma(100)).astype(float)
    S["50일선 위"] = (c > sma(50)).astype(float)
    S["골든크로스 50>200"] = (sma(50) > sma(200)).astype(float)
    S["12개월 수익>0"] = (c / c.shift(252) > 1).astype(float)
    S["6개월 수익>0"] = (c / c.shift(126) > 1).astype(float)
    S["1·3·6·12개월 다수결"] = sum((c / c.shift(n) > 1).astype(float) for n in (21, 63, 126, 252)) / 4
    S["MACD>시그널"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
                       > (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()).ewm(span=9, adjust=False).mean()).astype(float)
    # ── 변동성 ──
    S["변동성 조절"] = (vt / vol20).clip(upper=1.0)
    S["변동성 조절+200일선"] = (vt / vol20).clip(upper=1.0) * (c > sma(200))
    S["고변동 회피(60일 vol<중앙값×1.5)"] = (vol60 < vt * 1.5).astype(float)
    # ── 역발상 ──
    rsi2 = rsi(c, 2)
    pos = np.zeros(len(c)); cur = 0
    s5 = sma(5).values; s200 = sma(200).values; r2 = rsi2.values; cv = c.values
    for i in range(len(c)):
        if cur == 0 and r2[i] < 10 and cv[i] > s200[i]: cur = 1
        elif cur == 1 and cv[i] > s5[i]: cur = 0
        pos[i] = cur
    S["RSI2 눌림목(200일선 위)"] = pd.Series(pos, index=c.index)
    dd = c / c.rolling(252).max() - 1
    S["고점대비 -30% 이하면 보유"] = (dd < -0.30).astype(float)
    S["12개월 수익 하위권(<0)"] = (c / c.shift(252) < 1).astype(float)
    return S


def evaluate(df, pos, lev=1.0):
    r = df["c"].pct_change()
    if lev != 1.0:                       # 일간 리밸런싱 레버리지 합성(연 1.5% 비용)
        r = lev * r - 0.015 / 252
    p = pos.shift(1).fillna(0)
    sr = p * r - (p.diff().abs().fillna(0)) * COST
    return sr


def stats(sr):
    sr = sr.dropna()
    if len(sr) < 250 or sr.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, mdd=np.nan)
    eq = (1 + sr).cumprod()
    yrs = len(sr) / 252
    return dict(sharpe=sr.mean() / sr.std() * math.sqrt(252),
                cagr=eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1,
                mdd=float((1 - eq / eq.cummax()).max()))


def run(universe, lev=1.0, label=""):
    rows = []
    for sym in universe:
        df = frame(sym)
        if len(df) < WARM + 500:
            continue
        S = signals(df)
        idx = df.index[WARM:]
        mid = idx[len(idx) // 2]
        for name, pos in S.items():
            sr = evaluate(df, pos, lev).loc[idx]
            for part, seg in (("전체", sr), ("앞", sr.loc[:mid]), ("뒤", sr.loc[mid:])):
                st = stats(seg)
                rows.append(dict(sym=sym, sig=name, part=part, expo=float(pos.loc[idx].mean()), **st))
    d = pd.DataFrame(rows)
    bh = d[d.sig == "보유"].set_index(["sym", "part"])[["sharpe", "mdd", "cagr"]]
    d = d.join(bh, on=["sym", "part"], rsuffix="_bh")
    d["dS"] = d.sharpe - d.sharpe_bh
    d["dMDD"] = d.mdd_bh - d.mdd
    return d


def summarize(d, title):
    print(f"\n=== {title} — 샤프 개선(전략−보유): 중앙값, 개선 종목 비율 / 최대낙폭 개선 중앙값 ===")
    g = []
    for sig, x in d[d.sig != "보유"].groupby("sig"):
        row = {"신호": sig}
        for part in ("앞", "뒤", "전체"):
            y = x[x.part == part]
            row[f"{part} dS"] = y.dS.median(); row[f"{part} 승"] = (y.dS > 0).mean()
        row["MDD개선"] = x[x.part == "전체"].dMDD.median()
        row["보유율"] = x[x.part == "전체"].expo.median()
        g.append(row)
    t = pd.DataFrame(g).sort_values("앞 dS", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:+.2f}".format):
        print(t.to_string(index=False))
    return t


if __name__ == "__main__":
    d_u = run(UNDER)
    t_u = summarize(d_u, f"발견: 일반 자산 {len(UNDER)}개 (1배)")
    d_s = run(UNDER, lev=3.0)
    t_s = summarize(d_s, "검증①: 같은 자산을 3배 레버리지로 합성")
    d_l = run(LEV)
    t_l = summarize(d_l, f"검증②: 실제 레버리지 ETF {len(LEV)}개")
    pd.concat([d_u.assign(set="under"), d_s.assign(set="syn3x"), d_l.assign(set="lev")]).to_csv(
        Path(__file__).with_name("search_rows.csv"), index=False, encoding="utf-8-sig")

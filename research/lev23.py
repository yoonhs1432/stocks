"""2배 vs 3배 — ① 실제 ETF 쌍(같은 지수) ② 긴 기록 지수로 합성(1·2·3배, 2000·2008 포함)."""
import json, math, sys
import numpy as np, pandas as pd
YH = sys.argv[1]
def load(s):
    d = {}
    for t, c in json.load(open(f"{YH}/{s}.json")):
        if c and c > 0: d[pd.Timestamp((int(t) + 9 * 3600) // 86400 * 86400, unit="s")] = c
    return pd.Series(d).sort_index()
def st(r):
    eq = (1 + r).cumprod(); yrs = len(r) / 252
    return eq.iloc[-1] ** (1 / yrs) - 1, float((1 - eq / eq.cummax()).max()), r.mean() / r.std() * math.sqrt(252)
print("① 실제 ETF 쌍 — 둘 다 있는 기간")
for name, a, b in (("S&P500", "SSO", "UPRO"), ("나스닥100", "QLD", "TQQQ"), ("반도체", "USD", "SOXL"),
                   ("러셀2000", "UWM", "TNA"), ("다우", "DDM", "UDOW"), ("미국 장기국채", "UBT", "TMF")):
    try:
        x, y = load(a), load(b)
    except Exception as e:
        print(name, "자료 없음", e); continue
    idx = x.index.intersection(y.index)
    rx, ry = x[idx].pct_change().dropna(), y[idx].pct_change().dropna()
    (ca, ma, sa), (cb, mb, sb) = st(rx), st(ry)
    print(f"  {name:8s} {idx[0].date()}~  2배 {a}: 연 {ca:+6.1%} 낙폭 {ma:4.0%} 샤프 {sa:.2f} | 3배 {b}: 연 {cb:+6.1%} 낙폭 {mb:4.0%} 샤프 {sb:.2f}")
print("\n② 지수로 합성 (매일 배수 유지, 연 비용 1배 0.1%·2배 0.9%·3배 1.0% + 빌린 돈 이자 연 2%)")
for name, s in (("S&P500", "SPY"), ("다우", "DIA"), ("러셀2000", "IWM"), ("나스닥100", "QQQ"), ("반도체", "SOXX")):
    p = load(s); r = p.pct_change().dropna()
    vol = r.std() * math.sqrt(252)
    out = []
    for L, fee in ((1, 0.001), (2, 0.009), (3, 0.010)):
        rl = (L * r - (fee + (L - 1) * 0.02) / 252).clip(lower=-0.999)
        out.append(st(rl))
    mu = r.mean() * 252
    kelly = (mu - 0.02) / (vol ** 2)
    print(f"  {name:8s} {p.index[0].year}~ 변동성 {vol:4.0%} | " + " | ".join(f"{L}배 연 {c:+6.1%} 낙폭 {m:4.0%}" for L, (c, m, _) in zip((1, 2, 3), out))
          + f" | 수익 최대 배수≈{kelly:.1f}")

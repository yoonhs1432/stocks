"""바구니 13개 — 기초자산별 '수익이 가장 큰 배수' + 2·3배 합성(긴 기록)."""
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
    return eq.iloc[-1] ** (1 / yrs) - 1, float((1 - eq / eq.cummax()).max())
BASK = [("SSO", "2배", "S&P500", "SPY"), ("UDOW", "3배", "다우", "DIA"), ("UGL", "2배", "금", "GLD"),
        ("TMF", "3배", "미국 장기국채", "TLT"), ("KORU", "3배", "한국", "EWY"), ("BITX", "2배", "비트코인", "BTC-USD"),
        ("ERX", "2배", "에너지", "XLE"), ("LABU", "3배", "바이오", "XBI"), ("NAIL", "3배", "주택건설", "ITB"),
        ("YINN", "3배", "중국", "FXI"), ("DRN", "3배", "리츠", "IYR"), ("QPUX", "2배", "양자", "QTUM"), ("BRZU", "2배", "브라질", "EWZ")]
print(f"{'종목':6s} {'지금':4s} {'기초':10s} {'기간':6s} {'변동성':>5s} {'최적배수':>6s} | {'1배':>14s} | {'2배':>14s} | {'3배':>14s}")
for tk, now, nm, u in BASK:
    p = load(u)
    if u == "BTC-USD":
        p = p[p.index.dayofweek < 5]
    r = p.pct_change().dropna()
    vol = r.std() * math.sqrt(252); mu = r.mean() * 252
    k = (mu - 0.02) / vol ** 2
    res = [st((L * r - (fee + (L - 1) * 0.02) / 252).clip(lower=-0.999)) for L, fee in ((1, .001), (2, .009), (3, .010))]
    print(f"{tk:6s} {now:4s} {nm:10s} {p.index[0].year}~ {vol:5.0%} {k:6.1f} | " +
          " | ".join(f"{c:+6.1%} ({m:3.0%})" for c, m in res))

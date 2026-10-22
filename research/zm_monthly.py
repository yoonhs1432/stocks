"""매월 1회 Z·M 20/80 매매 vs 지금 방식(동일 비중·분기·현금 20%) — 같은 13종목, 세후.

Z = 그날까지 최대 2년 창 로그-로그 회귀 잔차(미래 정보 없음), M = 앱과 같은 식. 둘 다 0~100 백분위.
Z·M 규칙: 종목마다 1/13 자리. 월말에 조건이 맞으면 들어가고(다음 날 매수) 반대 조건이면 나온다. 빈 자리는 현금.
사용: python -I zm_monthly.py <yahoo> <web> <out>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS
WEB = Path(sys.argv[2]); sys.path.insert(0, str(WEB))
import insight as I
from backtest_portfolio import expanding_rolling_z  # noqa: E402

B13 = ["UPRO", "UGL", "DDM", "KORU", "BITX", "UBT", "ERX", "BIB", "NAIL", "YINN", "DRN", "QPUX", "BRZU"]
spy = RS.load("SPY")
ZP, MP = {}, {}
for t in B13:
    s = RS.load(t)
    days = s.index.intersection(spy.index)
    x, y = spy[days].values, s[days].values
    z = expanding_rolling_z(np.log(x / x[0]), np.log(y / y[0]))
    ZP[t] = pd.Series(np.clip((z + 2.5) / 5 * 100, 0, 100), index=days)
    m = I._m_score(s.values)
    MP[t] = pd.Series(np.clip((m + 2.5) / 5 * 100, 0, 100), index=s.index)

def last(sr, d):
    v = sr[sr.index <= d]
    return float(v.iloc[-1]) if len(v) and v.iloc[-1] == v.iloc[-1] else np.nan

def rule(kind):
    state = {t: 0 for t in B13}
    checks = set()
    def f(i, live):
        d = CAL[i]
        if d in CHECKS:
            for t in live:
                z, m = last(ZP[t], d), last(MP[t], d)
                if z != z or m != m:
                    continue
                lo = {"both": z < 20 and m < 20, "z": z < 20, "m": m < 20}[kind]
                hi = {"both": z > 80 and m > 80, "z": z > 80, "m": m > 80}[kind]
                if state[t] == 0 and lo: state[t] = 1
                elif state[t] == 1 and hi: state[t] = 0
        return [t for t in live if state[t]]
    return f

day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
RS.TICKERS = B13
for st in ("2016-01-04", "2024-10-01"):
    cal, krw, bars = RS.prices(day(st))
    CAL = cal; CHECKS = RS.schedule(cal, "monthly")
    rows = [("지금 방식 (분기·현금 20%)", RS.simulate(cal, krw, bars, "quarterly", 0.20)),
            ("지금 방식 (분기·현금 0%)", RS.simulate(cal, krw, bars, "quarterly", 0.0))]
    for k, name in (("both", "Z·M 둘 다 20/80 (매월)"), ("z", "Z만 20/80 (매월)"), ("m", "M만 20/80 (매월)")):
        r = RS.simulate(cal, krw, bars, "monthly", 0.0, pick=rule(k), slot=True)
        rows.append((name, r))
    RS.table(rows, f"13종목 · 세후 — {st}~")
    # 평균 투자 비중
    for name, r in rows:
        eq = r["eq"]
    print("  (Z·M 방식은 조건이 맞을 때만 들어가므로 대부분 현금일 수 있다)")

# 평균 투자 비중 — 월말마다 자리가 몇 개 차 있었나
print("\n평균 투자 비중(월말 기준, 13자리 중 찬 자리)")
for st in ("2016-01-04", "2024-10-01"):
    cal, krw, bars = RS.prices(day(st)); CAL = cal; CHECKS = RS.schedule(cal, "monthly")
    for k, name in (("both", "Z·M 둘 다"), ("z", "Z만"), ("m", "M만")):
        f = rule(k); occ = []
        for i, d in enumerate(cal):
            if d in CHECKS:
                live = {t for t in B13 if bars.loc[d, t] >= RS.MIN_BARS}
                if live: occ.append(len(f(i, live)) / len(live))
        print(f"  {st[:4]}~ {name:8s} 평균 {np.mean(occ):.0%}  (투자 0% 인 달 {np.mean([o == 0 for o in occ]):.0%})")

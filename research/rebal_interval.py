"""리밸런싱 간격 1~6개월 비교 — 지금 13종목·현금 20%·비용 0.2%·해외 양도세, 세후.

간격 N개월이면 '어느 달에 맞추느냐'(위상)에 따라 결과가 달라진다(운). 그래서 N 가지 위상을 모두 돌려
평균·최저·최고를 같이 본다. 예: 3개월 = 1·4·7·10월말 / 2·5·8·11월말 / 3·6·9·12월말(지금 분기).
사용: python -I rebal_interval.py <yahoo> <web> <out>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS

B13 = ["UPRO", "UGL", "DDM", "KORU", "BITX", "UBT", "ERX", "BIB", "NAIL", "YINN", "DRN", "QPUX", "BRZU"]
RS.TICKERS = B13
_orig = RS.schedule
CUR = {}


def sched(cal, kind):
    # N개월 간격·위상 k — 월 번호(연*12+월) % N == k 인 달의 마지막 거래일
    if "n" not in CUR:
        return _orig(cal, kind)
    n, k = CUR["n"], CUR["k"]
    d = pd.to_datetime(pd.Series(cal) * 86400, unit="s")
    mi = d.dt.year * 12 + d.dt.month - 1
    last = (mi != mi.shift(-1)).fillna(True).astype(bool).values & ((mi % n) == k).values
    return set(np.array(cal)[last])


RS.schedule = sched
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400

for cash in (0.20, 0.0):
    for st in ("2016-01-04", "2020-01-02"):
        cal, krw, bars = RS.prices(day(st))
        print(f"\n=== 현금 {int(cash*100)}% · {st[:4]}~ (세후) ===")
        print(f"{'간격':6s} {'세후 연수익 평균':>12s} {'최저~최고':>16s} {'최대낙폭':>7s} {'샤프':>5s} {'매매/년':>7s} {'비용+세금/년':>12s}")
        for n in range(1, 7):
            rs = []
            for k in range(n):
                CUR.update(n=n, k=k)
                rs.append(RS.simulate(cal, krw, bars, "monthly", cash))
            c = np.array([r["cagr_after"] for r in rs])
            yrs = (rs[0]["eq"].index[-1] - rs[0]["eq"].index[0]).days / 365.25
            print(f"{n}개월   {c.mean():+12.1%} {c.min():+8.1%}~{c.max():+.1%} "
                  f"{np.mean([r['mdd'] for r in rs]):7.0%} {np.mean([r['sharpe'] for r in rs]):5.2f} "
                  f"{np.mean([r['trades'] for r in rs]) / yrs:7.0f} "
                  f"{np.mean([r['cost'] + r['tax'] for r in rs]) / yrs / 1e4:10,.0f}만")
        CUR.clear()

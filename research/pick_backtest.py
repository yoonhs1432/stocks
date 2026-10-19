"""추린 바구니·묶음별 비중 vs 관심종목 전체 — 동일 비중·분기·세후 (rebalance_study 재사용)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS

# correlation_pick.py 의 묶음(평균 상관 ≥ 0.7)
CLUSTERS = [["SOXL", "HIBL", "TQQQ", "SPYU", "FNGU"], ["TARK", "QPUX"], ["DPST", "BNKU"], ["WTIU", "NRGU"],
            ["000660", "005930"], ["BITU"], ["DFEN"], ["GDXU"], ["KORU"], ["LABU"], ["NAIL"], ["URAA"], ["473460"]]
PICK10 = ["TQQQ", "005930", "DPST", "WTIU", "BITU", "DFEN", "GDXU", "LABU", "NAIL", "URAA"]
PICK13 = ["TQQQ", "TARK", "DPST", "WTIU", "005930", "BITU", "DFEN", "GDXU", "KORU", "LABU", "NAIL", "URAA", "473460"]
# 묶음별 동일: 묶음마다 1/K, 묶음 안에서 다시 나눈다 (상장 전 종목은 그날 다시 나눌 때 빠진다 — 근사)
CW = {t: 1.0 / len(g) for g in CLUSTERS for t in g}

day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
full = list(RS.TICKERS)
for label, start in (("2016~", "2016-01-04"), ("최근 2년", "2024-10-01")):
    rows = []
    for name, tks, w in (("관심종목 21개 동일", full, None), ("묶음별 동일(21개 유지)", full, CW),
                         ("추린 13개 (상관<0.7)", PICK13, None), ("추린 10개 (상관<0.6)", PICK10, None)):
        RS.TICKERS = tks
        cal, krw, bars = RS.prices(day(start))
        rows.append((name, RS.simulate(cal, krw, bars, "quarterly", 0.0, wmap=w)))
    RS.table(rows, f"분기 · 현금 0% · 세후 — {label}")

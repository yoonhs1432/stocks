"""떨어질 때 쓰려고 남겨 두는 현금(비상금)이 득인가 — 분기 리밸런싱·세후."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS

BASKET10 = ["TQQQ", "005930", "DPST", "WTIU", "BITU", "DFEN", "GDXU", "LABU", "NAIL", "URAA"]
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
full = list(RS.TICKERS)
CASES = [
    ("현금 0%", 0.0, None),
    ("현금 20% 고정", 0.20, None),
    ("비상금 20%: −20%에 반, −35%에 전부", 0.20, [(-0.20, 0.10), (-0.35, 0.0)]),
    ("비상금 20%: −25%에 전부", 0.20, [(-0.25, 0.0)]),
    ("현금 30% 고정", 0.30, None),
    ("비상금 30%: −20%에 반, −35%에 전부", 0.30, [(-0.20, 0.15), (-0.35, 0.0)]),
]
for bname, tks in (("추린 10개", BASKET10), ("관심종목 21개", full)):
    RS.TICKERS = tks
    for label, start in (("2016~", "2016-01-04"), ("최근 2년", "2024-10-01")):
        cal, krw, bars = RS.prices(day(start))
        rows = [(n, RS.simulate(cal, krw, bars, "quarterly", c, dip=dp)) for n, c, dp in CASES]
        RS.table(rows, f"{bname} · 분기 · 세후 — {label}")

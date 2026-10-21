"""기본 9개 + KORU vs + 하이닉스 vs 9개 그대로 — 분기·세후."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS
B9 = ["TQQQ", "DPST", "WTIU", "BITU", "DFEN", "GDXU", "LABU", "NAIL", "URAA"]
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
for st in ("2016-01-04", "2024-10-01"):
    rows = []
    for name, tks in (("9개 그대로", B9), ("9개 + KORU", B9 + ["KORU"]), ("9개 + 하이닉스", B9 + ["000660"])):
        RS.TICKERS = tks
        cal, krw, bars = RS.prices(day(st))
        for c in (0.0, 0.2):
            rows.append((f"{name} · 현금 {int(c*100)}%", RS.simulate(cal, krw, bars, "quarterly", c)))
    RS.table(rows, f"{st}~ 분기·세후")

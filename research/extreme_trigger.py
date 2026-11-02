"""매월 리밸런싱 + '목표의 2배/절반이면 그날 바로' 를 더하면 나아지나 — 13종목·현금 20%·세후.
사용: python -I extreme_trigger.py <yahoo> <web> <out>
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS

RS.TICKERS = ["UPRO", "UGL", "DDM", "KORU", "BITX", "UBT", "ERX", "BIB", "NAIL", "YINN", "DRN", "QPUX", "BRZU"]
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
for st in ("2016-01-04", "2020-01-02"):
    cal, krw, bars = RS.prices(day(st))
    rows = [("매월만", RS.simulate(cal, krw, bars, "monthly", 0.20))]
    for lo, hi in ((0.5, 2.0), (0.67, 1.5), (0.75, 1.33)):
        rows.append((f"매월 + {hi:g}배/{lo:g}배 즉시", RS.simulate(cal, krw, bars, "monthly", 0.20, extreme=(lo, hi))))
    RS.table(rows, f"현금 20% · {st[:4]}~")

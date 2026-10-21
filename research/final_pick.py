"""추천안 확인 — 새 기본 14개 vs MSOX 뺀 13개, 현금 20·30%."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS
B14 = ["SSO", "UGL", "UDOW", "KORU", "BITX", "TMF", "ERX", "LABU", "NAIL", "YINN", "DRN", "QPUX", "MSOX", "BRZU"]
B13 = [t for t in B14 if t != "MSOX"]
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
for st in ("2016-01-04", "2024-10-01"):
    rows = []
    for name, tks in (("14개", B14), ("13개(MSOX 제외)", B13)):
        RS.TICKERS = tks
        cal, krw, bars = RS.prices(day(st))
        for c in (0.0, 0.2, 0.3):
            rows.append((f"{name} · 현금 {int(c*100)}%", RS.simulate(cal, krw, bars, "quarterly", c)))
    RS.table(rows, f"{st}~ 분기·세후")

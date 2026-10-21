"""바구니 수준: 지금 13개 vs '3배 있으면 3배' vs '2배 있으면 2배(S&P 만 3배)' — 2016~·최근 2년, 분기·세후."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS
A = ["SSO", "UGL", "UDOW", "KORU", "BITX", "TMF", "ERX", "LABU", "NAIL", "YINN", "DRN", "QPUX", "BRZU"]
B = ["UPRO" if t == "SSO" else t for t in A]
SWAP = {"SSO": "UPRO", "UDOW": "DDM", "TMF": "UBT", "LABU": "BIB", "YINN": "XPP", "DRN": "URE"}
C = [SWAP.get(t, t) for t in A]
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
for st in ("2016-01-04", "2024-10-01"):
    rows = []
    for name, tks in (("A 지금 13개", A), ("B 3배 있으면 3배", B), ("C 2배 있으면 2배(S&P만 3배)", C)):
        RS.TICKERS = tks
        cal, krw, bars = RS.prices(day(st))
        for c in (0.0, 0.2):
            rows.append((f"{name} · 현금 {int(c*100)}%", RS.simulate(cal, krw, bars, "quarterly", c)))
    RS.table(rows, f"{st}~ 분기·세후")

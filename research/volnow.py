"""지금 변동성 조절 비중 = min(1, 장기 중앙값 변동성 / 최근 20일 변동성)."""
import json, math, sys
from pathlib import Path
import pandas as pd
D = Path(sys.argv[1])
for sym in json.load(open(sys.argv[2], encoding="utf-8")):
    f = D / f"{sym}.json"
    if not f.exists(): continue
    o = json.load(open(f)); rows = o["rows"] if isinstance(o, dict) else o
    c = pd.Series([x[1] for x in rows])
    r = c.pct_change(); v = r.rolling(20).std() * math.sqrt(252)
    med = v.expanding(252).median().iloc[-1]
    if med != med: med = v.median()
    print(f"{sym:7s} 최근20일 변동성 {v.iloc[-1]:5.0%}  장기 중앙값 {med:5.0%}  → 권장 비중 {min(1, med / v.iloc[-1]):4.0%}")

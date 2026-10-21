"""새 기본(여러 종목 ETF 대표 14개) vs 지금 기본 9개 — 분기·세후, 최근 2년."""
import json, sys
from pathlib import Path
import pandas as pd
YH, WEB, OUTD, LEVD = (Path(a) for a in sys.argv[1:5])
sys.argv = sys.argv[:4]
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS
NEW = json.load(open(Path(__file__).with_name("lev_multi.json"), encoding="utf-8"))["basket"]
for s in NEW:
    f = YH / f"{s}.json"
    if not f.exists():
        f.write_text(json.dumps([[r[0], r[1]] for r in json.loads((LEVD / f"{s}.json").read_text())]))
OLD = ["TQQQ", "DPST", "WTIU", "BITU", "DFEN", "GDXU", "LABU", "NAIL", "URAA"]
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
for st in ("2016-01-04", "2024-10-01"):
    rows = []
    for name, tks in ((f"새 기본 {len(NEW)}개", NEW), ("지금 기본 9개", OLD), ("지금 9개 + KORU", OLD + ["KORU"])):
        RS.TICKERS = tks
        cal, krw, bars = RS.prices(day(st))
        for c in (0.0, 0.2):
            rows.append((f"{name} · 현금 {int(c*100)}%", RS.simulate(cal, krw, bars, "quarterly", c)))
    RS.table(rows, f"{st}~ 분기·세후")

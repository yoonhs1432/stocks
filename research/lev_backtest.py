"""레버리지 전체에서 만든 바구니 vs 지금 바구니들 — 동일 비중·분기·세후 (rebalance_study 재사용).

새 바구니 종목은 상장이 짧은 것이 많아 **최근 2년**만 공정하게 비교할 수 있다.
사용: python -I lev_backtest.py <yahoo 디렉터리> <web 디렉터리> <출력> <묶음 json> <레버리지 시세 디렉터리> [바구니 크기]
"""
import json, sys
from pathlib import Path
import pandas as pd

YH, WEB, OUTD, GROUPS, LEVD = (Path(a) for a in sys.argv[1:6])
TOP = int(sys.argv[6]) if len(sys.argv) > 6 else 20
sys.argv = sys.argv[:4]
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS  # noqa: E402

import re
bk = json.load(open(GROUPS, encoding="utf-8"))["basket"]
reps = [b["pick"] for b in bk]
SINGLE = re.compile(r"Long [A-Z]{2,5}|Daily [A-Z]{2,5} Bull|[A-Z]{2,5} Bull 2X")
index_only = [b["pick"] for b in bk if not SINGLE.search(b["name"])]
# 레버리지 시세를 rebalance_study 가 읽는 모양([t, 종가])으로 YH 에 맞춰 둔다
for s in reps:
    f = YH / f"{s}.json"
    if not f.exists():
        rows = json.loads((LEVD / f"{s}.json").read_text())
        f.write_text(json.dumps([[r[0], r[1]] for r in rows]))

BASKET10 = ["TQQQ", "005930", "DPST", "WTIU", "BITU", "DFEN", "GDXU", "LABU", "NAIL", "URAA"]
full = list(RS.TICKERS)
day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
for start in ("2024-10-01", "2025-04-01"):
    rows = []
    for name, tks in ((f"레버리지 전체 바구니 {len(reps)}개", reps), (f"지수·업종만 {len(index_only)}개", index_only),
                      ("지금 기본 10개", BASKET10), ("관심종목 21개", full)):
        RS.TICKERS = tks
        cal, krw, bars = RS.prices(day(start))
        for c in (0.0, 0.2):
            rows.append((f"{name} · 현금 {int(c*100)}%", RS.simulate(cal, krw, bars, "quarterly", c)))
    RS.table(rows, f"동일 비중 · 분기 · 세후 — {start}~")

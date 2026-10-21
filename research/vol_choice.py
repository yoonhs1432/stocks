"""같은 묶음에서 변동성 낮은 것 vs 높은 것 — 분기마다 그 시점까지 1년 변동성으로 다시 고른다(미래 정보 없음).

① 관심종목 묶음(삼성전자 제외) — 2016~ / 최근 2년
② 미국 레버리지 ETF 전체 묶음(거래대금 $20M+, VIX·천연가스 제외) — 최근 2년, 무작위 고르기 5번 평균도
사용: python -I vol_choice.py <yahoo> <web> <out> <lev_groups.json> <levdata>
"""
import json, random, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

YH, WEB, OUTD, GROUPS, LEVD = (Path(a) for a in sys.argv[1:6])
sys.argv = sys.argv[:4]
sys.path.insert(0, str(Path(__file__).parent))
import rebalance_study as RS  # noqa: E402

day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400


def run(clusters, start, label, seeds=0):
    tks = sorted({t for g in clusters for t in g})
    RS.TICKERS = tks
    cal, krw, bars = RS.prices(day(start))
    vol = krw.pct_change(fill_method=None).rolling(252, min_periods=120).std()
    def chooser(mode, rng=None):
        def f(i, live):
            out = []
            for g in clusters:
                m = [t for t in g if t in live]
                if not m:
                    continue
                v = {t: vol.iloc[i][t] for t in m}
                v = {t: x for t, x in v.items() if x == x} or {t: 0.0 for t in m}
                if mode == "low": out.append(min(v, key=v.get))
                elif mode == "high": out.append(max(v, key=v.get))
                else: out.append(rng.choice(sorted(v)))
            return out
        return f
    rows = [("묶음마다 변동성 낮은 것", RS.simulate(cal, krw, bars, "quarterly", 0.0, pick=chooser("low"))),
            ("묶음마다 변동성 높은 것", RS.simulate(cal, krw, bars, "quarterly", 0.0, pick=chooser("high")))]
    if seeds:
        rs = [RS.simulate(cal, krw, bars, "quarterly", 0.0, pick=chooser("rand", random.Random(k))) for k in range(seeds)]
        avg = {k: float(np.mean([r[k] for r in rs])) for k in ("after", "cagr_after", "mdd", "sharpe", "cost", "tax")}
        avg["trades"] = int(np.mean([r["trades"] for r in rs]))
        rows.append((f"묶음마다 무작위({seeds}번 평균)", avg))
    RS.table(rows, f"{label} — {start}~ (분기·현금 0%·세후)")


WATCH = [["SOXL", "HIBL", "TQQQ", "SPYU", "FNGU", "TARK", "QPUX", "473460"], ["KORU", "000660"],
         ["DPST", "BNKU"], ["WTIU", "NRGU"], ["BITU"], ["DFEN"], ["GDXU"], ["LABU"], ["NAIL"], ["URAA"]]
for st in ("2016-01-04", "2024-10-01"):
    run(WATCH, st, "① 관심종목 묶음")

g = json.load(open(GROUPS, encoding="utf-8"))
info = {}
for grp in g["groups"]:
    pass
DECAY = re.compile(r"(VIX|Natural Gas)", re.I)
names = {c["symbol"]: c["name"] for c in json.load(open(Path(GROUPS).parent / "lev_candidates.json", encoding="utf-8"))}
clusters = []
for grp in g["groups"]:
    m = []
    for t in grp["members"]:
        rows = json.loads((LEVD / f"{t}.json").read_text())
        df = pd.DataFrame(rows, columns=["t", "adj", "vol", "close"]).dropna(subset=["adj"])
        dv = (df.close * df.vol).tail(60).mean()
        if dv >= 20e6 and not DECAY.search(names.get(t, "")):
            m.append(t)
            f = YH / f"{t}.json"
            if not f.exists():
                f.write_text(json.dumps([[r[0], r[1]] for r in rows]))
    if m:
        clusters.append(m)
multi = [c for c in clusters if len(c) > 1]
print(f"\n레버리지 묶음 {len(clusters)}개 (그중 고를 게 2개 이상인 묶음 {len(multi)}개)")
run(clusters, "2024-10-01", "② 레버리지 ETF 전체 묶음", seeds=5)

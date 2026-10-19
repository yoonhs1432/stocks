"""관심종목끼리 얼마나 같이 움직이나 — 주간 수익률(원화) 상관 → 묶음 → 묶음마다 하나.

주간으로 보는 이유: 한국은 미국보다 먼저 닫혀 일간 상관이 낮게 나온다(같은 날 소식이 다음 날 반영).
원화로 보는 이유: 실제 계좌가 원화라 환율도 같이 움직이는 몫이다.
사용: python -I correlation_pick.py <yahoo 디렉터리> <web 디렉터리>
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

YH, WEB = Path(sys.argv[1]), Path(sys.argv[2])
TICKERS = json.load(open(WEB / "data" / "tickers.json", encoding="utf-8"))


def load(sym):
    s = {}
    for t, c in json.load(open(YH / f"{sym}.json")):
        if c and c > 0:
            s[pd.Timestamp((int(t) + 9 * 3600) // 86400 * 86400, unit="s")] = float(c)
    return pd.Series(s).sort_index()


fx = load("KRW=X")
px = {}
for t in TICKERS:
    s = load(t)
    if not t[:1].isdigit():
        s = s * fx.reindex(s.index.union(fx.index)).ffill().reindex(s.index)
    px[t] = s
P = pd.DataFrame(px)
W = P.resample("W-FRI").last()
R = np.log(W).diff()
R = R[R.index >= "2024-10-01"]
C = R.corr(min_periods=30)          # 짧은 종목은 겹치는 주만으로
vol = R.std() * math.sqrt(52)
n_obs = R.notna().sum()

# 평균 연결 계층 묶음 (거리 = 1 − 상관)
def cluster(th):
    groups = [[t] for t in TICKERS]
    def link(a, b):
        v = [C.loc[x, y] for x in a for y in b if pd.notna(C.loc[x, y])]
        return np.mean(v) if v else 0.0
    while True:
        best, pair = -1, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                l = link(groups[i], groups[j])
                if l > best: best, pair = l, (i, j)
        if best < th: break
        i, j = pair
        groups[i] += groups[j]; del groups[j]
    return groups

print("=== 서로 많이 닮은 쌍 (주간 상관 ≥ 0.75) ===")
pairs = []
for i, a in enumerate(TICKERS):
    for b in TICKERS[i + 1:]:
        if pd.notna(C.loc[a, b]) and C.loc[a, b] >= 0.75:
            pairs.append((C.loc[a, b], a, b))
for c, a, b in sorted(pairs, reverse=True):
    print(f"  {a:7s} ~ {b:7s} {c:.2f}")

for th in (0.7, 0.6):
    print(f"\n=== 묶음 (평균 상관 ≥ {th} 이면 같은 묶음) ===")
    for g in sorted(cluster(th), key=len, reverse=True):
        inside = np.mean([C.loc[x, y] for x in g for y in g if x < y]) if len(g) > 1 else float("nan")
        desc = ", ".join(f"{t}(변동성 {vol[t]:.0%}, {n_obs[t]}주)" for t in g)
        print(f"  [{len(g)}] 안쪽 평균 {inside:.2f} — {desc}")

print("\n=== 종목별: 다른 종목들과의 평균 상관 (낮을수록 분산 효과 큼) ===")
avg = C.apply(lambda col: col.drop(col.name).mean())
for t in avg.sort_values().index:
    print(f"  {t:7s} 평균 {avg[t]:+.2f}  변동성 {vol[t]:.0%}")

C.round(2).to_csv(Path(__file__).with_name("correlation_weekly.csv"), encoding="utf-8-sig")

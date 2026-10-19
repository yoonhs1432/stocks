"""미국 상장 레버리지 롱 ETF 전체에서 서로 덜 닮은 바구니 만들기.

1) 나스닥 ETF 목록(5,564개)에서 이름으로 레버리지 롱만(인버스 제외) → 토스에서 살 수 있는 615개
2) 최근 1년 이상 기록 + 하루 평균 거래대금 $3M 이상(호가가 얇으면 사고팔 때 손해)
3) 최근 2년 주간 수익률 상관 → 평균 연결 묶음(1 − 상관 거리)
4) 묶음마다 거래대금이 가장 큰 것 하나

사용: python -I lev_universe.py <후보 json> <시세 디렉터리> <출력 json>
"""
import json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

CANDS, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MIN_DAYS = 252
MIN_DV = 3_000_000
TH = 0.6                         # 같은 묶음 = 평균 상관 0.6 이상

names = {c["symbol"]: c["name"] for c in json.load(open(CANDS, encoding="utf-8"))}


def lev_of(name):
    m = re.search(r"(\d(?:\.\d+)?)\s?[xX]\b", name)
    if m:
        return float(m.group(1))
    if re.search(r"UltraPro", name): return 3.0
    if re.search(r"\bUltra\b", name): return 2.0
    return float("nan")


px, info = {}, {}
for sym in names:
    f = DATA / f"{sym}.json"
    if not f.exists():
        continue
    rows = json.loads(f.read_text() or "[]")
    if len(rows) < MIN_DAYS:
        continue
    df = pd.DataFrame(rows, columns=["t", "adj", "vol", "close"])
    df["d"] = pd.to_datetime((df.t // 86400) * 86400, unit="s")
    df = df.dropna(subset=["adj"]).set_index("d")
    dv = (df.close * df.vol).tail(60).mean()
    if not dv or dv != dv or dv < MIN_DV:
        continue
    r = np.log(df.adj).diff()
    px[sym] = df.adj
    info[sym] = {"name": names[sym], "dv": float(dv), "days": len(df),
                 "vol": float(r.tail(252).std() * math.sqrt(252)), "lev": lev_of(names[sym]),
                 "ret1y": float(df.adj.iloc[-1] / df.adj.iloc[-253] - 1) if len(df) > 253 else float("nan")}

print(f"기록 1년+·거래대금 ${MIN_DV/1e6:.0f}M+ : {len(px)}개")
W = pd.DataFrame(px).resample("W-FRI").last()
R = np.log(W).diff()
R = R[R.index >= R.index[-1] - pd.Timedelta(weeks=104)]
C = R.corr(min_periods=40)
keep = [s for s in C.columns if C[s].notna().sum() > 1]
C = C.loc[keep, keep].fillna(0.0)          # 겹치는 주가 모자란 쌍은 '관계 없음'으로
D = 1 - C.values
np.fill_diagonal(D, 0)
D = np.clip((D + D.T) / 2, 0, 2)
Z = linkage(squareform(D, checks=False), method="average")
lab = fcluster(Z, t=1 - TH, criterion="distance")

groups = {}
for s, g in zip(keep, lab):
    groups.setdefault(g, []).append(s)
out = []
for g, mem in groups.items():
    mem.sort(key=lambda s: -info[s]["dv"])
    rep = mem[0]
    inside = float(np.mean([C.loc[a, b] for a in mem for b in mem if a < b])) if len(mem) > 1 else float("nan")
    out.append({"rep": rep, "size": len(mem), "inside": inside, "members": mem,
                "repInfo": info[rep], "dvSum": sum(info[s]["dv"] for s in mem)})
out.sort(key=lambda x: -x["dvSum"])

print(f"\n=== 묶음 {len(out)}개 (평균 상관 ≥ {TH}) — 거래대금 큰 묶음부터 ===")
for i, o in enumerate(out, 1):
    ri = o["repInfo"]
    others = ", ".join(o["members"][1:7]) + (" …" if o["size"] > 7 else "")
    print(f"{i:2d}. [{o['size']:3d}] {o['rep']:6s} {ri['lev']:.1f}x  ${ri['dv']/1e6:7.0f}M/일  변동성 {ri['vol']:4.0%}  "
          f"1년 {ri['ret1y']:+5.0%}  {ri['name'][:48]}")
    if others:
        print(f"         같은 묶음: {others}")
json.dump({"groups": out, "corr_symbols": keep}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
C.round(3).to_csv(OUT.with_suffix(".corr.csv"), encoding="utf-8-sig")

# ── 바구니: 거래 충분 + 구조적 감쇠 상품 제외 + 묶음마다 '거래 충분한 것 중 변동성 가장 낮은 것' ──
BASKET_DV = 20_000_000
# 선물 롤오버로 오래 들수록 녹는 상품 — VIX 선물, 천연가스 선물
DECAY = re.compile(r"(VIX|Natural Gas)", re.I)
basket = []
for o in out:
    pool = [s for s in o["members"] if info[s]["dv"] >= BASKET_DV and not DECAY.search(info[s]["name"])]
    if not pool:
        continue
    pick = min(pool, key=lambda s: (info[s]["vol"], -info[s]["dv"]))
    basket.append({"pick": pick, **info[pick], "group": o["members"][:6], "size": o["size"]})
print(f"\n=== 바구니 {len(basket)}개 (거래대금 ${BASKET_DV/1e6:.0f}M+, VIX·천연가스 제외, 묶음마다 변동성 최저) ===")
for b in basket:
    single = bool(re.search(r"Long [A-Z]{2,5}\b|Daily [A-Z]{2,5} Bull|\b[A-Z]{2,5} Bull 2X", b["name"]))
    print(f"  {b['pick']:6s} {b['lev']:.1f}x  변동성 {b['vol']:4.0%}  ${b['dv']/1e6:6.0f}M  {'단일종목' if single else '지수/업종'}  "
          f"{b['name'][:46]}   (묶음 {b['size']}: {', '.join(b['group'][:4])})")
json.dump({"groups": out, "basket": basket}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

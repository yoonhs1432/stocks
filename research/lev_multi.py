"""단일 종목 레버리지 ETF 를 빼고 — 여러 종목을 담은 ETF(+원자재·코인)만으로 묶고 대표 고르기 (2026-10-21).

분류: lev_universe.py 결과 표에 이름 규칙으로 multi / asset(원자재·코인 한 자산) / single 을 붙인 것(`lev_kind.csv`).
제외: single 전부, 레버리지가 아닌 RDIV, 구조적 감쇠(VIX·천연가스 선물).
대표: 묶음마다 하루 거래대금 $20M 이상 중 변동성 최저. 없으면 $3M 이상 중 최저(거래 적음 표시).
한국(KORU 묶음)은 사용자 지정으로 KORU.

사용: python -I lev_multi.py <lev_kind.csv> <levdata> <출력 md> <출력 json>
"""
import json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

KIND, DATA, OUT_MD, OUT_JSON = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
TH = 0.6
t = pd.read_csv(KIND, encoding="utf-8-sig")
t.loc[t.symbol == "QPUX", "kind"] = "multi"          # 'Pure Quantum' 지수 — 이름 규칙이 놓쳤다
t = t[(t.kind != "single") & ~t.symbol.isin(["RDIV"]) & ~t.name.str.contains("VIX|Natural Gas", regex=True)]
info = t.set_index("symbol").to_dict("index")

px = {}
for s in info:
    rows = json.loads((DATA / f"{s}.json").read_text())
    df = pd.DataFrame(rows, columns=["t", "adj", "vol", "close"]).dropna(subset=["adj"])
    df.index = pd.to_datetime((df.t // 86400) * 86400, unit="s")
    px[s] = df.adj
W = pd.DataFrame(px).resample("W-FRI").last()
R = np.log(W).diff()
R = R[R.index >= R.index[-1] - pd.Timedelta(weeks=104)]
C = R.corr(min_periods=20).fillna(0.0)
D = 1 - C.values
np.fill_diagonal(D, 0)
Z = linkage(squareform(np.clip((D + D.T) / 2, 0, 2), checks=False), method="average")
lab = fcluster(Z, t=1 - TH, criterion="distance")

groups = {}
for s, g in zip(C.columns, lab):
    if s == "KORU":
        continue                       # 한국은 따로 한 자리(사용자 지정) — 어느 묶음에 섞이든 빼서 세운다
    groups.setdefault(g, []).append(s)
groups["korea"] = ["KORU"]
out = []
for mem in groups.values():
    mem.sort(key=lambda s: info[s]["vol"])
    if mem == ["KORU"]:
        rep, note = "KORU", "사용자 지정 — 단일 종목을 빼면 미국 기술 묶음과 상관이 높아 합쳐진다"
    else:
        liq = [s for s in mem if info[s]["dv"] >= 20e6]
        ok = [s for s in mem if info[s]["dv"] >= 3e6]
        rep, note = (liq[0], "") if liq else ((ok[0], "거래 적음") if ok else (None, "거래 부족"))
    inside = float(np.mean([C.loc[a, b] for a in mem for b in mem if a < b])) if len(mem) > 1 else float("nan")
    out.append({"members": mem, "rep": rep, "note": note, "inside": inside,
                "dvSum": sum(info[s]["dv"] for s in mem)})
out.sort(key=lambda o: -o["dvSum"])

# 구체적인 것부터 — 'S&P Oil & Gas'·'S&P Biotech' 가 미국 대형주로 잡히지 않게
THEME_HINT = [("Treasury", "미국 장기국채"), ("Gold|Silver", "금·은·금광"), ("Bitcoin|Ether|Solana|XRP", "암호화폐"),
              ("Energy|Oil|Crude", "원유·에너지"), ("Biotech|Healthcare", "바이오·헬스케어"), ("China|CSI", "중국"),
              ("Homebuilders", "주택건설"), ("South Korea", "한국"), ("Quantum|Uranium|Space", "양자·우라늄"),
              ("MSOS", "미국 대마초"), ("Brazil", "브라질"), ("Real Estate|Utilities", "리츠·유틸리티(금리 민감)"),
              ("Dow|Russell|Small|Mid|Financial|Banks|Retail|Industrials|Defense", "미국 중소형·다우·금융·은행·방산"),
              ("S&P|QQQ|Technology|Semiconductor|FANG|Magnificent|Internet|Innovation|High Beta|Top 5|4X", "미국 대형·기술·반도체")]


def theme(mem):
    """구성 종목이 가장 많이 해당하는 주제. 같으면 목록 앞(구체적인) 쪽."""
    best, best_n = "", 0
    for pat, th in THEME_HINT:
        n = sum(1 for s in mem if re.search(pat, info[s]["name"]))
        if n > best_n:
            best, best_n = th, n
    return best


lines = ["| # | 주제 | 대표 (기본값) | 구성 — 티커(배수·변동성·하루 거래대금), 변동성 낮은 순 |", "|---|---|---|---|"]
basket = []
for i, o in enumerate(out, 1):
    mem = ", ".join(f"{s}({'?' if info[s]['lev'] != info[s]['lev'] else '%g' % info[s]['lev']}x·{info[s]['vol']:.0%}·${info[s]['dv']/1e6:.0f}M)" for s in o["members"])
    rep = f"**{o['rep']}**" + (f" ({o['note']})" if o["note"] else "") if o["rep"] else f"— ({o['note']})"
    lines.append(f"| {i} | {theme(o['members'])} | {rep} | {mem} |")
    if o["rep"]:
        basket.append(o["rep"])
OUT_MD.write_text("# 여러 종목 레버리지 ETF(+원자재·코인) 묶음 — 단일 종목 ETF 제외 (2026-10-21)\n\n"
                  "6개월 이상·하루 거래대금 $1M 이상, 최근 2년 주간 상관 평균 연결(≥0.6). 대표 = 거래 $20M 이상 중 변동성 최저"
                  "(없으면 $3M 이상), 한국은 KORU(사용자 지정).\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
json.dump({"basket": basket, "groups": out}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(len(info), "개 →", len(out), "묶음, 기본 바구니", basket)
print("\n".join(lines))

"""지난 12개월 수익률 5분위별 다음 12개월 수익 — 종목 안에서 분위를 나눔."""
import sys, math, numpy as np
sys.path.insert(0, ".")
import zm_long as L
names = ["SOXL", "TQQQ", "HIBL", "DFEN", "DPST", "KORU", "LABU", "NAIL", "005930", "000660",
         "BTC-USD", "SOXX", "ARKK", "SPHB", "QQQ", "ITA", "KRE", "GDX", "USO", "EWY", "KBE", "XLE", "XBI", "ITB", "URA"]
allq = [[] for _ in range(5)]
for s in names:
    d = L.load(s); days = sorted(d); y = np.array([d[k] for k in days])
    ly = np.log(y); n = len(y)
    mom = np.full(n, np.nan); mom[250:] = ly[250:] - ly[:-250]
    f = L.fwd(y, 250)
    ok = ~np.isnan(mom) & ~np.isnan(f)
    if ok.sum() < 500: continue
    qs = np.nanpercentile(mom[ok], [20, 40, 60, 80])
    b = np.digitize(mom, qs)
    row = []
    for q in range(5):
        msk = ok & (b == q)
        v = math.exp(np.mean(f[msk])) - 1; row.append(v)
        if not s[0].isdigit() and s not in ("SOXL","TQQQ","HIBL","DFEN","DPST","KORU","LABU","NAIL"):
            allq[q].append(v)
    cur = (math.exp(mom[-1]) - 1); cq = int(np.digitize([mom[-1]], qs)[0]) + 1
    print(f"{s:8s} 하위20% {row[0]:+6.0%} | {row[1]:+6.0%} {row[2]:+6.0%} {row[3]:+6.0%} | 상위20% {row[4]:+6.0%}   지금 {cur:+6.0%} → {cq}분위")
print("기초자산 평균", " ".join(f"{np.mean(q):+.0%}" for q in allq))

"""M20/80·Z20/80 이 '보유율이 낮아서' 이긴 건지 — 같은 보유율·평균 보유 기간의 무작위 진입과 비교."""
import sys, math, numpy as np
sys.path.insert(0, ".")
sys.argv = sys.argv[:1] + [sys.argv[1], sys.argv[2]]
import zm_long as L
rng = np.random.default_rng(0)
spy = L.load("SPY")
names = ["SOXL", "TARK", "HIBL", "TQQQ", "DFEN", "DPST", "GDXU", "KORU", "LABU", "NAIL", "005930", "000660", "KRE", "KBE", "SOXX", "QQQ"]
print(f"{'종목':8s} | {'M CAGR':>7s} {'무작위중앙':>8s} {'백분위':>5s} | {'Z CAGR':>7s} {'무작위중앙':>8s} {'백분위':>5s}")
for s in names:
    days, x, y = L.align(spy, L.load(s))
    lx, ly = np.log(x / x[0]), np.log(y / y[0])
    z = L.rolling_z(lx, ly, L.WIN); m = L.m_score(y)
    sl = slice(L.WIN, len(y)); yy = y[sl]
    out = []
    for sc in (m, z):
        pos = L.band_pos(L.pct(sc[sl]))
        cagr = L.strat(yy, pos)[0]
        p = pos.mean(); ent = max(1, int(np.sum(np.diff(pos) > 0)))
        hold = p * len(pos) / ent; gap = (1 - p) * len(pos) / ent
        sims = []
        for _ in range(500):
            q = np.zeros(len(pos)); t = 0; cur = rng.random() < p
            while t < len(pos):
                d = max(1, int(rng.exponential(hold if cur else gap)))
                q[t:t + d] = cur; t += d; cur = not cur
            sims.append(L.strat(yy, q)[0])
        sims = np.array(sims)
        out.append((cagr, np.median(sims), (sims < cagr).mean()))
    print(f"{s:8s} | {out[0][0]:+7.1%} {out[0][1]:+8.1%} {out[0][2]:5.0%} | {out[1][0]:+7.1%} {out[1][1]:+8.1%} {out[1][2]:5.0%}")

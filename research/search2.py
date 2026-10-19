"""2차 탐색 — ① 시장 국면·계절성으로 타이밍 ② 종목 간 순위(로테이션).

① 은 search.py 와 같은 틀(샤프 개선, 앞/뒤 절반, 1배·3배 합성·실제 레버리지).
② 는 매달 말 순위로 상위 3개를 동일 비중 보유 → 전체 동일 비중과 비교.
"""
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.argv = sys.argv[:3]
sys.path.insert(0, str(Path(__file__).parent))
import search as S  # noqa: E402

VIX = S.load("^VIX"); HYG = S.load("HYG"); IEF = S.load("IEF")


def macro(df):
    """종목이 아니라 시장 국면 신호 — 모든 종목에 같은 날 같은 값."""
    idx = df.index
    spy = df["spy"]
    vix = VIX.reindex(idx).ffill()
    cr = (HYG / IEF).reindex(idx).ffill()
    out = {"보유": pd.Series(1.0, index=idx)}
    out["SPY 200일선 위"] = (spy > spy.rolling(200).mean()).astype(float)
    out["VIX<1년 중앙값×1.2"] = (vix < vix.rolling(252).median() * 1.2).astype(float)
    out["VIX<30"] = (vix < 30).astype(float)
    out["VIX 급등 후 진정(VIX<5일 전)"] = (vix < vix.shift(5)).astype(float)
    out["신용(HYG/IEF) 100일선 위"] = (cr > cr.rolling(100).mean()).astype(float).where(cr.notna(), 1.0)
    m = pd.Series(idx.map(lambda d: pd.Timestamp(d * 86400, unit="s").month), index=idx)
    out["11~4월만 보유"] = m.isin([11, 12, 1, 2, 3, 4]).astype(float)
    r = df["c"].pct_change(); rs = spy.pct_change()
    beta = r.rolling(252).cov(rs) / rs.rolling(252).var()
    resid = r - beta.shift(1) * rs
    out["잔차역발상 1개월(<0)"] = (resid.rolling(21).sum() < 0).astype(float)
    out["잔차역발상 1주(<0)"] = (resid.rolling(5).sum() < 0).astype(float)
    return out


def run_macro(universe, lev=1.0):
    rows = []
    for sym in universe:
        df = S.frame(sym)
        if len(df) < S.WARM + 500:
            continue
        sig = macro(df)
        idx = df.index[S.WARM:]
        mid = idx[len(idx) // 2]
        for name, pos in sig.items():
            sr = S.evaluate(df, pos, lev).loc[idx]
            for part, seg in (("전체", sr), ("앞", sr.loc[:mid]), ("뒤", sr.loc[mid:])):
                rows.append(dict(sym=sym, sig=name, part=part, expo=float(pos.loc[idx].mean()), **S.stats(seg)))
    d = pd.DataFrame(rows)
    bh = d[d.sig == "보유"].set_index(["sym", "part"])[["sharpe", "mdd", "cagr"]]
    d = d.join(bh, on=["sym", "part"], rsuffix="_bh")
    d["dS"] = d.sharpe - d.sharpe_bh; d["dMDD"] = d.mdd_bh - d.mdd
    return d


# ── ② 로테이션 ──
ROT = ["QQQ", "SOXX", "SPHB", "ITA", "KRE", "KBE", "GDX", "USO", "EWY", "XLE", "XBI", "ITB", "URA", "ARKK"]


def rotation(lev=1.0, k=3):
    px = pd.DataFrame({s: S.load(s) for s in ROT}).reindex(S.SPY.index).ffill()
    spy = S.SPY
    rets = px.pct_change()
    if lev != 1.0:
        rets = lev * rets - 0.015 / 252
    # 월말 인덱스
    dates = pd.to_datetime(px.index * 86400, unit="s")
    mend = pd.Series(dates, index=px.index).groupby([dates.year, dates.month]).tail(1).index
    lp = np.log(px)
    scores = {
        "6개월 모멘텀 상위": lp - lp.shift(126),
        "12-1개월 모멘텀 상위": lp.shift(21) - lp.shift(252),
        "1개월 하락 상위(단기 역발상)": -(lp - lp.shift(21)),
        "12개월 하락 상위(장기 역발상)": -(lp - lp.shift(252)),
        "변동성 낮은 순": -rets.rolling(60).std(),
    }
    # Z 순위 — 각 종목의 2년창 Z (낮은 = SPY 대비 싸다)
    zs = {}
    for s in ROT:
        sp = spy.reindex(px.index)
        ok = px[s].notna()
        z = pd.Series(np.nan, index=px.index)
        if ok.sum() > 600:
            sub = px[s][ok]; spsub = sp[ok]
            z[ok] = S.rolling_z(np.log(spsub / spsub.iloc[0]), np.log(sub / sub.iloc[0]), 504).values
        zs[s] = z
    Z = pd.DataFrame(zs)
    scores["Z 낮은 순(SPY 대비 쌈)"] = -Z
    scores["Z 높은 순"] = Z
    res = {}
    avail = px.notna().sum(axis=1)
    start = px.index[(avail >= 10).values.argmax() + 504]
    for name, sc in list(scores.items()) + [("전체 동일비중", None)]:
        w = pd.DataFrame(0.0, index=px.index, columns=ROT)
        cur = pd.Series(0.0, index=ROT)
        for d in px.index[px.index >= start]:
            if d in mend:
                live = px.loc[d].notna()
                if sc is None:
                    cur = live.astype(float) / live.sum()
                else:
                    s = sc.loc[d][live].dropna()
                    top = s.nlargest(k).index
                    cur = pd.Series(0.0, index=ROT); cur[top] = 1.0 / len(top)
            w.loc[d] = cur
        pw = w.shift(1).fillna(0)
        sr = (pw * rets.fillna(0)).sum(axis=1) - (pw.diff().abs().sum(axis=1).fillna(0)) * S.COST
        sr = sr.loc[start:]
        mid = sr.index[len(sr) // 2]
        res[name] = {p: S.stats(seg) for p, seg in (("전체", sr), ("앞", sr.loc[:mid]), ("뒤", sr.loc[mid:]))}
    print(f"\n=== 로테이션 (매달 상위 {k}개, {'3배 합성' if lev != 1 else '1배'}) — 시작 {pd.Timestamp(start * 86400, unit='s').date()} ===")
    print(f"{'방식':24s} | {'샤프 앞':>7s} {'뒤':>6s} {'전체':>6s} | {'연수익':>7s} {'최대낙폭':>7s}")
    for name, v in res.items():
        print(f"{name:24s} | {v['앞']['sharpe']:+7.2f} {v['뒤']['sharpe']:+6.2f} {v['전체']['sharpe']:+6.2f} | "
              f"{v['전체']['cagr']:+7.1%} {v['전체']['mdd']:7.0%}")


if __name__ == "__main__":
    S.summarize(run_macro(S.UNDER), "시장 국면·계절성: 일반 자산 (1배)")
    S.summarize(run_macro(S.UNDER, 3.0), "시장 국면·계절성: 3배 합성")
    S.summarize(run_macro(S.LEV), "시장 국면·계절성: 실제 레버리지 ETF")
    rotation(1.0)
    rotation(3.0)

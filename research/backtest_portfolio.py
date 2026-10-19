"""관심종목 전체 모의 백테스트 — 가상 1억, 주 1회, 네 계좌 비교.

  A 동일 비중      관심종목을 똑같이 나눠 담고 매주 맞춘다(비교 기준)
  B 장기 제안      앱 카드(web/insight.py advice) 그대로. 통계는 **그 시점까지 데이터로** 매년 1월 다시 낸다
  C 변동성 조절    동일 비중 × min(1, 장기 중앙값 변동성 / 최근 20일 변동성), 나머지 현금
  D Z 20/80        Z<20 이면 들어가고 Z>80 이면 나온다(앱의 원래 방식)
  E 85% 고정       A 를 늘 85% 만 투자 — C 의 평균 투자 비율과 같게 맞춘 비교 기준

- 금요일 종가로 판단 → 다음 거래일 종가에 매매. 비용 0.2%(매매 금액), 현금 이자 0.
- 원화 계산. 미국 종목은 KRW=X 환율.
- 종목은 상장 후 120봉이 쌓이면 편입. Z 는 그날까지의 창(최대 504봉)으로만 회귀.

사용: python -I backtest_portfolio.py <yahoo 디렉터리(목록형 json)> <web 디렉터리> <출력 디렉터리>
"""
import json, math, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd

YH, WEB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(WEB))
import insight as I  # noqa: E402  — 앱과 같은 판단 로직을 쓴다

TICKERS = json.load(open(WEB / "data" / "tickers.json", encoding="utf-8"))
CAPITAL = 100_000_000
COST = 0.002
MIN_BARS = 120
FACTOR = {"매수 고려": 1.5, "분할 매수": 1.25, "관망": 1.0, "타이밍 근거 없음": 1.0, "비중 축소": 0.5, "매도 고려": 0.0}


def load(sym):
    f = YH / f"{sym}.json"
    if not f.exists():
        return None
    s = {}
    for t, c in json.load(open(f)):
        if c and c > 0:
            s[(int(t) + 9 * 3600) // 86400] = float(c)
    return pd.Series(s).sort_index()


def expanding_rolling_z(lx, ly, win=504, minp=MIN_BARS):
    """그날까지 최대 win 봉으로 회귀한 Z (처음엔 쌓인 만큼만)."""
    n = len(lx); z = np.full(n, np.nan)
    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    Sx, Sy, Sxx, Sxy, Syy = cs(lx), cs(ly), cs(lx * lx), cs(lx * ly), cs(ly * ly)
    for t in range(minp - 1, n):
        a1 = t + 1; a0 = max(0, a1 - win); k = a1 - a0
        sx, sy = Sx[a1] - Sx[a0], Sy[a1] - Sy[a0]
        cxx = Sxx[a1] - Sxx[a0] - sx * sx / k; cxy = Sxy[a1] - Sxy[a0] - sx * sy / k
        cyy = Syy[a1] - Syy[a0] - sy * sy / k
        if cxx <= 1e-12: continue
        b = cxy / cxx; a = sy / k - b * sx / k
        sd = math.sqrt(max(cyy - b * cxy, 0) / (k - 2))
        if sd > 0: z[t] = (ly[t] - a - b * lx[t]) / sd
    return z


class Asset:
    """한 종목(또는 기초자산)의 시계열과 미리 계산한 인과적 신호."""

    def __init__(self, sym, spy):
        s = load(sym)
        self.sym = sym
        self.ok = s is not None and len(s) >= MIN_BARS
        if not self.ok:
            return
        if sym.endswith("-USD"):
            s = s[s.index.isin(spy.index)]
        self.s = s
        days = s.index.intersection(spy.index)
        self.days = days
        x, y = spy[days].values, s[days].values
        self.y = y
        self.z = pd.Series(expanding_rolling_z(np.log(x / x[0]), np.log(y / y[0])), index=days)
        self.m = pd.Series(I._m_score(y), index=days)
        r = s.pct_change()
        self.vol20 = r.rolling(20).std() * math.sqrt(252)
        self.volmed = self.vol20.expanding(60).median()

    def stats_asof(self, cut):
        """insight._series_stats 와 같은 모양 — 단 cut 날까지의 데이터만."""
        own = self.s[self.s.index <= cut].values
        if len(own) < 300:
            return None
        out = {"sym": self.sym, "years": round(len(own) / 252, 1), "mom": I._mom_stats(own)}
        if len(own) >= 452:
            ma = np.convolve(own, np.ones(200) / 200, "valid")
            above = np.concatenate([np.zeros(199), (own[199:] > ma).astype(float)])
            bh = I._cagr_mdd(own[199:], np.ones(len(own) - 199)); tf = I._cagr_mdd(own[199:], above[199:])
            out["trend"] = {"gap": own[-1] / ma[-1] - 1, "mddHold": bh[1], "mddFilter": tf[1]}
        k = int((self.days <= cut).sum())
        if k >= I.CFG.win + I.CFG.min_test:
            y = self.y[:k]; z = self.z.values[:k]; m = self.m.values[:k]
            sl = slice(I.CFG.win, None); seed = zlib.crc32(self.sym.encode())
            out["zRank"], out["zTrades"] = I._vs_random(y[sl], I._band(I._pct(z[sl])), seed)
            out["mRank"], out["mTrades"] = I._vs_random(y[sl], I._band(I._pct(m[sl])), seed + 1)
        return out


def run(start, label):
    spy = load("SPY"); fx = load("KRW=X")
    assets = {t: Asset(t, spy) for t in TICKERS}
    proxies = {t: Asset(I.PROXY[t], spy) for t in TICKERS if t in I.PROXY}
    assets = {t: a for t, a in assets.items() if a.ok}

    # 달력: 모든 종목 거래일의 합집합, 쉬는 날은 직전 가격
    cal = sorted(set().union(*[set(a.s.index) for a in assets.values()]))
    cal = [d for d in cal if d >= start]
    px = pd.DataFrame({t: a.s for t, a in assets.items()}).reindex(cal).ffill()
    fxs = fx.reindex(sorted(set(fx.index) | set(cal))).ffill().reindex(cal)
    krw = px.copy()
    for t in px.columns:
        if not t[:1].isdigit():
            krw[t] = px[t] * fxs          # 원화 가격
    dates = pd.to_datetime(pd.Series(cal) * 86400, unit="s")
    week = dates.dt.isocalendar()
    fri = set(pd.Series(cal)[(week.week != week.week.shift(-1)).values].values)   # 그 주 마지막 거래일

    accts = {k: {"cash": float(CAPITAL), "qty": {t: 0.0 for t in px.columns}, "eq": [], "cost": 0.0,
                 "invested": []} for k in "ABCDE"}
    zin = {t: 0 for t in px.columns}     # D: 들어가 있나
    stats_year = None
    stats = {}
    pending = None                       # 금요일에 정한 목표 비중 → 다음 거래일에 집행

    for i, d in enumerate(cal):
        p = krw.loc[d]
        # ① 지난 금요일 결정 집행
        if pending is not None:
            for k, target in pending.items():
                a = accts[k]
                val = a["cash"] + sum(a["qty"][t] * p[t] for t in px.columns if p[t] == p[t])
                for t in px.columns:
                    if p[t] != p[t]:
                        continue
                    want = val * target.get(t, 0.0) / p[t]
                    trade = (want - a["qty"][t]) * p[t]
                    if abs(trade) < val * 0.002:          # 0.2% 미만 차이는 건너뛴다
                        continue
                    c = abs(trade) * COST
                    a["cash"] -= trade + c; a["cost"] += c; a["qty"][t] = want
            pending = None
        # ② 평가
        for k, a in accts.items():
            pos = sum(a["qty"][t] * p[t] for t in px.columns if p[t] == p[t])
            a["eq"].append(a["cash"] + pos)
            a["invested"].append(pos / (a["cash"] + pos))
        # ③ 금요일이면 다음 거래일 목표 비중 결정
        if d in fri and i < len(cal) - 1:
            live = [t for t, a in assets.items() if (a.s.index <= d).sum() >= MIN_BARS and p[t] == p[t]]
            if not live:
                continue
            w = 1.0 / len(live)
            yr = pd.Timestamp(d * 86400, unit="s").year
            if yr != stats_year:          # 매년(처음 포함) 그 시점까지의 통계로
                stats_year = yr
                stats = {}
                for t in TICKERS:
                    if t not in assets:
                        continue
                    own = assets[t].stats_asof(d)
                    pr = proxies.get(t)
                    stats[t] = {"own": own, "proxy": pr.stats_asof(d) if pr and pr.ok else None}
                I._data = {"day": str(yr), "tickers": stats}
            tg = {k: {} for k in "ABCDE"}
            for t in live:
                a = assets[t]
                zt = a.z.get(d, np.nan) if d in a.z.index else a.z[a.z.index <= d].iloc[-1]
                mt = a.m[a.m.index <= d].iloc[-1]
                zp = None if zt != zt else float(np.clip((zt + 2.5) / 5 * 100, 0, 100))
                mp = float(np.clip((mt + 2.5) / 5 * 100, 0, 100))
                tg["A"][t] = w
                tg["E"][t] = w * 0.85
                adv = I.advice(t, zp, mp)
                tg["B"][t] = w * FACTOR.get(adv["action"], 1.0) if adv else w
                v20 = a.vol20[a.vol20.index <= d].iloc[-1]; vm = a.volmed[a.volmed.index <= d].iloc[-1]
                tg["C"][t] = w * (min(1.0, vm / v20) if v20 == v20 and vm == vm and v20 > 0 else 1.0)
                if zp is not None:
                    if zin[t] == 0 and zp < 20: zin[t] = 1
                    elif zin[t] == 1 and zp > 80: zin[t] = 0
                tg["D"][t] = w * zin[t]
            s = sum(tg["B"].values())
            if s > 1:                     # 매수 가중으로 100% 를 넘으면 맞춰 줄인다
                tg["B"] = {t: v / s for t, v in tg["B"].items()}
            pending = tg

    idx = pd.to_datetime(pd.Series(cal) * 86400, unit="s")
    res = {}
    for k, a in accts.items():
        eq = pd.Series(a["eq"], index=idx)
        r = eq.pct_change().dropna()
        yrs = (idx.iloc[-1] - idx.iloc[0]).days / 365.25
        res[k] = dict(final=eq.iloc[-1], cagr=(eq.iloc[-1] / CAPITAL) ** (1 / yrs) - 1,
                      mdd=float((1 - eq / eq.cummax()).max()),
                      sharpe=r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else np.nan,
                      invested=float(np.mean(a["invested"])), cost=a["cost"], eq=eq)
    return res, idx


NAMES = {"A": "A Equal weight", "B": "B Long-term advice", "C": "C Vol targeting", "D": "D Z 20/80",
         "E": "E Fixed 85%"}
KNAMES = {"A": "A 동일 비중", "B": "B 장기 제안", "C": "C 변동성 조절", "D": "D Z 20/80", "E": "E 85% 고정"}


def report(res, idx, label, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print(f"\n=== {label}  ({idx.iloc[0].date()} ~ {idx.iloc[-1].date()}) ===")
    print(f"{'계좌':14s} {'최종 금액':>14s} {'연수익':>7s} {'최대낙폭':>7s} {'샤프':>5s} {'평균투자':>7s} {'비용':>11s}")
    for k, v in res.items():
        print(f"{KNAMES[k]:14s} {v['final']:14,.0f} {v['cagr']:+7.1%} {v['mdd']:7.0%} {v['sharpe']:5.2f} "
              f"{v['invested']:7.0%} {v['cost']:11,.0f}")
    # 연도별
    yr = {k: v["eq"].resample("YE").last().pct_change() for k, v in res.items()}
    first = {k: v["eq"].resample("YE").last().iloc[0] / CAPITAL - 1 for k, v in res.items()}
    years = res["A"]["eq"].resample("YE").last().index
    print("연도별 수익: " + " | ".join(KNAMES[k][:1] for k in res))
    for j, y in enumerate(years):
        vals = [first[k] if j == 0 else yr[k].iloc[j] for k in res]
        print(f"  {y.year}: " + "  ".join(f"{v:+6.1%}" for v in vals))
    # 세로형 차트 — 위: 자산 추이(로그), 아래: 낙폭
    fig, ax = plt.subplots(2, 1, figsize=(6, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"A": "#888888", "B": "#d62728", "C": "#1f77b4", "D": "#ff7f0e", "E": "#2ca02c"}
    for k, v in res.items():
        eq = v["eq"] / 1e8
        ax[0].plot(eq.index, eq.values, color=colors[k], lw=1.6 if k not in "AE" else 1.1,
                   ls="--" if k == "E" else "-", label=f"{NAMES[k]}  {v['cagr']:+.0%}/yr, MDD {v['mdd']:.0%}")
        dd = v["eq"] / v["eq"].cummax() - 1
        ax[1].plot(dd.index, dd.values * 100, color=colors[k], lw=1.0)
    ax[0].set_yscale("log"); ax[0].set_ylabel("Equity (x100M KRW, log)"); ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(label, fontsize=11); ax[0].grid(alpha=0.3)
    ax[1].set_ylabel("Drawdown (%)"); ax[1].grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=130)


if __name__ == "__main__":
    day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
    r1, i1 = run(day("2024-10-01"), "recent")
    report(r1, i1, "Watchlist backtest - recent 2 years", "bt_recent.png")
    r2, i2 = run(day("2016-01-04"), "long")
    report(r2, i2, "Watchlist backtest - since 2016", "bt_long.png")

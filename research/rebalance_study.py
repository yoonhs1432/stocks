"""동일 비중 — 리밸런싱 주기 × 현금 비중, 비용·해외주식 양도세까지.

- 관심종목 전체, 120봉 쌓이면 편입. 판단은 그날 종가 → 다음 거래일 종가에 매매. 비용 0.2%.
- 해외주식 양도세: 원화 기준 실현손익(평균단가)을 해마다 합산, 250만 원 공제 후 22%, 다음 해 5월 납부.
  현금이 모자라면 비율대로 팔아 낸다. 국내 종목은 양도세 없음(소액주주).
- 마지막 비교는 '전량 매도 후 세후' — 세금을 미루는 전략이 부당하게 유리하지 않게.
- 현금 이자 연 2.5%.

사용: python -I rebalance_study.py <yahoo 디렉터리> <web 디렉터리> <출력 디렉터리>
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

YH, WEB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TICKERS = json.load(open(WEB / "data" / "tickers.json", encoding="utf-8"))
CAPITAL = 100_000_000
COST = 0.002
MIN_BARS = 120
TAX, EXEMPT = 0.22, 2_500_000
CASH_RATE = 0.025


def load(sym):
    f = YH / f"{sym}.json"
    s = {}
    for t, c in json.load(open(f)):
        if c and c > 0:
            s[(int(t) + 9 * 3600) // 86400] = float(c)
    return pd.Series(s).sort_index()


def prices(start):
    raw = {t: load(t) for t in TICKERS if (YH / f"{t}.json").exists()}
    fx = load("KRW=X")
    cal = sorted(set().union(*[set(s.index) for s in raw.values()]))
    cal = [d for d in cal if d >= start]
    px = pd.DataFrame(raw).reindex(sorted(set().union(*[set(s.index) for s in raw.values()]))).ffill()
    bars = pd.DataFrame({t: s.reindex(px.index).notna().cumsum() for t, s in raw.items()})
    px, bars = px.reindex(cal), bars.reindex(cal)
    fxs = fx.reindex(sorted(set(fx.index) | set(cal))).ffill().reindex(cal)
    krw = px.copy()
    for t in px.columns:
        if not t[:1].isdigit():
            krw[t] = px[t] * fxs
    return cal, krw, bars


def schedule(cal, kind):
    d = pd.to_datetime(pd.Series(cal) * 86400, unit="s")
    if kind == "weekly":
        w = d.dt.isocalendar().week
        last = (w != w.shift(-1)).values
    elif kind == "monthly":
        last = (d.dt.month != d.dt.month.shift(-1)).values
    elif kind == "quarterly":
        last = (d.dt.quarter != d.dt.quarter.shift(-1)).values
    elif kind == "yearly":
        last = (d.dt.year != d.dt.year.shift(-1)).values
    else:                                    # 밴드·편입 때만: 매주 점검만
        w = d.dt.isocalendar().week
        last = (w != w.shift(-1)).values
    last = pd.Series(last).fillna(True).astype(bool).values   # 마지막 날은 다음이 없어 NA
    return set(np.array(cal)[last])


def simulate(cal, krw, bars, kind, cash_w=0.0, band=None, wmap=None, dip=None, reset=-0.05):
    """kind: weekly/monthly/quarterly/yearly/band/none. band = 목표 대비 상대 이탈 한도(0.2 = ±20%).
    wmap: 종목별 상대 비중(없으면 동일). 그날 살아 있는 종목끼리 다시 나눈다.
    dip: 비상금 규칙 [(낙폭, 현금비중), …] — 계좌가 고점 대비 그만큼 빠지면 현금을 그 비중까지 투입.
         고점 근처(reset, 기본 −5%)로 돌아오면 다시 cash_w 로 채운다. 더 깊어질 때만 내려간다."""
    cols = list(krw.columns)
    us = {t: not t[:1].isdigit() for t in cols}
    qty = {t: 0.0 for t in cols}; basis = {t: 0.0 for t in cols}     # basis = 평균단가 × 수량 (원화)
    cash = float(CAPITAL)
    realized = {}; owed = {}; cost_total = 0.0; tax_total = 0.0; trades = 0
    checks = schedule(cal, kind)
    eq = []; live_prev = set(); pending = None
    peak = float(CAPITAL); dip_cw = cash_w; cw_prev = cash_w
    dts = pd.to_datetime(pd.Series(cal) * 86400, unit="s")

    def sell(t, q, p, yr):
        nonlocal cash, cost_total
        if q <= 0: return
        avg = basis[t] / qty[t] if qty[t] > 0 else 0.0
        if us[t]:
            realized[yr] = realized.get(yr, 0.0) + q * (p - avg)
        basis[t] -= q * avg; qty[t] -= q
        c = q * p * COST; cash += q * p - c; cost_total += c

    def buy(t, q, p):
        nonlocal cash, cost_total
        if q <= 0: return
        c = q * p * COST; cash -= q * p + c; cost_total += c
        qty[t] += q; basis[t] += q * p

    for i, d in enumerate(cal):
        p = krw.loc[d]; yr = dts.iloc[i].year
        if i > 0:
            cash *= (1 + CASH_RATE) ** (1 / 252) if cash > 0 else 1.0
        # 세금 납부 — 다음 해 5월 첫 거래일
        if dts.iloc[i].month == 5 and (i == 0 or dts.iloc[i - 1].month != 5):
            due = owed.pop(yr - 1, 0.0)
            if due > 0:
                if cash < due:                 # 모자라면 비율대로 판다
                    val = sum(qty[t] * p[t] for t in cols if p[t] == p[t])
                    need = (due - cash) * 1.01
                    for t in cols:
                        if qty[t] > 0 and p[t] == p[t]:
                            sell(t, qty[t] * need / val, p[t], yr)
                cash -= due; tax_total += due
        if dts.iloc[i].month == 1 and i > 0 and dts.iloc[i - 1].month == 12:
            g = realized.get(yr - 1, 0.0)
            owed[yr - 1] = max(0.0, g - EXEMPT) * TAX
        # 집행
        if pending is not None:
            val = cash + sum(qty[t] * p[t] for t in cols if p[t] == p[t])
            for t in cols:                    # 매도 먼저, 그다음 매수
                if p[t] != p[t]: continue
                want = val * pending.get(t, 0.0) / p[t]
                if want < qty[t] and (qty[t] - want) * p[t] > val * 0.002:
                    sell(t, qty[t] - want, p[t], yr); trades += 1
            for t in cols:
                if p[t] != p[t]: continue
                want = val * pending.get(t, 0.0) / p[t]
                if want > qty[t] and (want - qty[t]) * p[t] > val * 0.002:
                    buy(t, want - qty[t], p[t]); trades += 1
            pending = None
        val = cash + sum(qty[t] * p[t] for t in cols if p[t] == p[t])
        eq.append(val)
        # 판단
        live = {t for t in cols if bars.loc[d, t] >= MIN_BARS and p[t] == p[t]}
        if not live or i == len(cal) - 1:
            continue
        cw = cash_w
        if dip:
            peak = max(peak, val)
            dd = val / peak - 1
            lvl = cash_w
            for th, w in dip:
                if dd <= th:
                    lvl = min(lvl, w)
            dip_cw = cash_w if dd > reset else min(dip_cw, lvl)
            cw = dip_cw
        if wmap:
            tot = sum(wmap.get(t, 0.0) for t in live) or 1.0
            target = {t: (1 - cw) * wmap.get(t, 0.0) / tot for t in live}
        else:
            target = {t: (1 - cw) / len(live) for t in live}
        do = False
        if i == 0 or live != live_prev or cw != cw_prev:
            do = True                          # 처음·종목 편입·비상금 단계가 바뀔 때는 항상
        elif d in checks:
            if kind in ("weekly", "monthly", "quarterly", "yearly"):
                do = True
            elif kind == "band":
                for t in live:
                    w = qty[t] * p[t] / val
                    if abs(w / target[t] - 1) > band:
                        do = True; break
                if abs(cash / val - cash_w) > max(0.05, cash_w * band):
                    do = True
        if do:
            pending = target
        live_prev = live
        cw_prev = cw

    idx = pd.to_datetime(pd.Series(cal) * 86400, unit="s")
    eqs = pd.Series(eq, index=idx)
    # 전량 매도 후 세후 — 남은 미실현 해외 이익 + 아직 안 낸 세금
    p = krw.loc[cal[-1]]
    unreal = sum(qty[t] * p[t] - basis[t] for t in cols if us[t] and qty[t] > 0)
    last_yr = idx.iloc[-1].year
    g = realized.get(last_yr, 0.0) + unreal
    pending_tax = sum(owed.values()) + max(0.0, g - EXEMPT) * TAX
    after = eqs.iloc[-1] - pending_tax - sum(qty[t] * p[t] for t in cols if qty[t] > 0) * COST
    yrs = (idx.iloc[-1] - idx.iloc[0]).days / 365.25
    r = eqs.pct_change().dropna()
    return dict(final=eqs.iloc[-1], after=after, cagr_after=(after / CAPITAL) ** (1 / yrs) - 1,
                mdd=float((1 - eqs / eqs.cummax()).max()), sharpe=r.mean() / r.std() * math.sqrt(252),
                cost=cost_total, tax=tax_total + pending_tax, trades=trades, eq=eqs)


def table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'방식':22s} {'세후 최종':>13s} {'세후 연수익':>9s} {'최대낙폭':>7s} {'샤프':>5s} {'매매':>6s} {'비용':>11s} {'세금':>12s}")
    for name, v in rows:
        print(f"{name:22s} {v['after']:13,.0f} {v['cagr_after']:+9.1%} {v['mdd']:7.0%} {v['sharpe']:5.2f} "
              f"{v['trades']:6d} {v['cost']:11,.0f} {v['tax']:12,.0f}")


if __name__ == "__main__":
    day = lambda s: int(pd.Timestamp(s).timestamp()) // 86400
    results = {}
    for label, start in (("2016~", "2016-01-04"), ("최근 2년", "2024-10-01")):
        cal, krw, bars = prices(day(start))
        freq = [("매주", "weekly", None), ("매월", "monthly", None), ("분기", "quarterly", None),
                ("매년", "yearly", None), ("±20% 벗어날 때", "band", 0.20), ("±35% 벗어날 때", "band", 0.35),
                ("±50% 벗어날 때", "band", 0.50), ("편입 때만(거의 안 함)", "none", None)]
        rows = [(n, simulate(cal, krw, bars, k, 0.0, b)) for n, k, b in freq]
        table(rows, f"리밸런싱 주기 — {label} (현금 0%)")
        results[label] = rows
    # 현금 비중 — 가장 나은 주기 두 개로
    for label, start in (("2016~", "2016-01-04"), ("최근 2년", "2024-10-01")):
        cal, krw, bars = prices(day(start))
        for fname, kind, b in (("매월", "monthly", None), ("±35%", "band", 0.35)):
            rows = [(f"현금 {int(c * 100)}% · {fname}", simulate(cal, krw, bars, kind, c, b))
                    for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
            table(rows, f"현금 비중 — {label}")
            results[f"cash {label} {fname}"] = rows
    pd.to_pickle({k: [(n, {kk: vv for kk, vv in v.items()}) for n, v in rows] for k, rows in results.items()},
                 OUT / "rebal.pkl")

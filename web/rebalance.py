"""리밸런싱 도우미 — 바구니를 동일 비중으로 맞추려면 무엇을 얼마나 사고팔지.

2026-10-20 검증(`research/zm_study.md`):
  - 동일 비중 리밸런싱은 **분기마다**가 세후로 가장 나았다(매주는 비용·양도세만 늘었다).
  - 현금 비중은 공짜 점심이 아니다 — 10%p 마다 연수익 −1.3~1.9%p, 최대 낙폭 −5~7%p.
    그래서 고를 때 보라고 **현금 비중별 과거 성과**를 바구니마다 다시 계산해 보여 준다.
  - 기본 바구니는 여러 종목 레버리지 ETF 의 상관 묶음 대표(아래 DEFAULT_BASKET). 설정에서 바꿀 수 있다.

계좌 밖으로 돈이 나가지 않는다. 주문은 내지 않고 **무엇을 얼마나**만 계산한다.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

import insight
import store

KST = timezone(timedelta(hours=9))

# 기본 바구니 (2026-10-21, research/lev_multi.py · lev23_basket.py):
# 토스에서 살 수 있는 미국 레버리지 롱 ETF 중 **단일 종목 ETF 를 뺀** 여러 종목 ETF(+원자재·코인)를 주간 상관으로
# 묶고 묶음마다 대표 하나. 배수는 **기초자산이 감당하는 만큼** — 수익이 최대가 되는 배수 ≈ (수익−이자)/분산이
# S&P 만 3배 근처라 S&P 는 3배(UPRO), 나머지는 2배가 있으면 2배(DDM·UBT·BIB). 중국·리츠는 2배(XPP·URE)가
# 하루 거래 $0.1~0.2M 뿐이라 3배(YINN·DRN) 유지. 한국은 사용자 지정 KORU, MSOX 는 사용자 동의로 제외.
# 2016~ 분기·세후: 현금 0% 연 +18.5%·최대 낙폭 55% (이전 13개 +16.6%·63%).
# 앱은 이 목록을 응답(defaultBasket)으로 받아 '기본' 단추에 쓴다 — 바꿔도 앱을 다시 빌드할 필요 없다.
DEFAULT_BASKET = ["UPRO", "UGL", "DDM", "KORU", "BITX", "UBT", "ERX", "BIB", "NAIL", "YINN",
                  "DRN", "QPUX", "BRZU"]

# 종목 설명 — 앱의 주문 목록·설정에 그대로 보인다(서버만 고치면 된다)
DESC = {
    "UPRO": "S&P500 3배 · 미국 대형주 500개 (ProShares UltraPro S&P500)",
    "SSO": "S&P500 2배 (ProShares Ultra S&P500)",
    "UGL": "금 가격 2배 · 주식과 따로 움직이는 편 (ProShares Ultra Gold)",
    "DDM": "다우존스30 2배 · 미국 우량 대형주 30개 (ProShares Ultra Dow30)",
    "UDOW": "다우존스30 3배 (ProShares UltraPro Dow30)",
    "KORU": "MSCI 한국 3배 · 삼성전자·SK하이닉스 비중이 크다 (Direxion)",
    "BITX": "비트코인 2배 · 선물로 추종 (Volatility Shares 2x Bitcoin)",
    "BITU": "비트코인 2배 (ProShares Ultra Bitcoin)",
    "UBT": "미국 20년+ 장기국채 2배 · 금리가 내리면 오른다, 주식 급락 때 완충(2022 처럼 같이 빠질 때도 있다)",
    "TMF": "미국 20년+ 장기국채 3배 (Direxion)",
    "ERX": "미국 에너지 업종 2배 · 엑슨모빌·셰브론 등 (Direxion)",
    "BIB": "나스닥 바이오 2배 · 대형 바이오 중심 (ProShares Ultra Nasdaq Biotechnology)",
    "LABU": "S&P 바이오 3배 · 중소형 바이오 균등 비중 (Direxion)",
    "NAIL": "미국 주택건설·건자재 3배 · 금리에 민감 (Direxion)",
    "YINN": "중국 대형주 50 3배 · 홍콩 상장 (Direxion FTSE China)",
    "DRN": "미국 리츠(부동산) 3배 · 금리에 민감 (Direxion)",
    "QPUX": "양자컴퓨팅 기업 지수 2배 · 아이온큐·리게티 등 (Defiance)",
    "BRZU": "브라질 대형주 2배 · 페트로브라스·발레 등 (Direxion MSCI Brazil)",
    "TQQQ": "나스닥100 3배 (ProShares UltraPro QQQ)",
    "SOXL": "미국 반도체 3배 (Direxion)",
    "DPST": "미국 지역은행 3배 (Direxion)",
    "DFEN": "미국 항공·방산 3배 (Direxion)",
    "GDXU": "금광주 3배 (MicroSectors ETN)",
    "WTIU": "미국 에너지 3배 (MicroSectors ETN) · 거래가 매우 적다",
    "BNKU": "미국 대형은행 3배 (MicroSectors ETN) · 거래가 매우 적다",
}
DEFAULTS = {"basket": DEFAULT_BASKET, "cashPct": 20, "period": "quarterly", "lastDone": None}
PERIODS = ("monthly", "quarterly", "yearly")

COST = 0.002              # 매매 금액 대비 비용(수수료·환전·거래세 대략)
MIN_TRADE = 200_000       # 이보다 작은 차이는 건너뛴다
TAX, EXEMPT = 0.22, 2_500_000
CASH_RATE = 0.025
SIM_START = "2016-01-04"
SIM_MIN_BARS = 120
CASH_LEVELS = (0, 10, 20, 30, 40, 50)


# ── 설정 ──

def settings() -> dict:
    cur = store.settings().get("rebalance") or {}
    out = {**DEFAULTS, **cur}
    if out["period"] not in PERIODS:
        out["period"] = "quarterly"
    out["cashPct"] = int(max(0, min(80, out.get("cashPct", 20))))
    return out


def save(patch: dict) -> dict:
    cur = settings()
    for k in ("basket", "cashPct", "period", "lastDone"):
        if k in patch:
            cur[k] = patch[k]
    store.put("rebalance", cur)
    kick_table()
    return settings()


# ── 기간 ──

def _period_end(d: date, period: str) -> date:
    """d 가 속한 기간의 마지막 날."""
    if period == "monthly":
        m = d.month
    elif period == "quarterly":
        m = ((d.month - 1) // 3 + 1) * 3
    else:
        m = 12
    nxt = date(d.year + (m // 12), m % 12 + 1, 1)
    return nxt - timedelta(days=1)


def status(s: dict, today: date | None = None) -> dict:
    """지금 맞출 때인가 — 마지막으로 맞춘 날이 직전 기간 끝보다 앞이면 '맞출 때'."""
    today = today or datetime.now(KST).date()
    cur_end = _period_end(today, s["period"])
    # 이번 기간의 첫날 → 그 하루 전이 직전 기간의 끝
    if s["period"] == "monthly":
        start = date(today.year, today.month, 1)
    elif s["period"] == "quarterly":
        start = date(today.year, (today.month - 1) // 3 * 3 + 1, 1)
    else:
        start = date(today.year, 1, 1)
    prev_end = start - timedelta(days=1)
    done = date.fromisoformat(s["lastDone"]) if s.get("lastDone") else None
    due = done is None or done <= prev_end
    return {"due": due, "periodEnd": prev_end.isoformat(), "nextEnd": cur_end.isoformat(),
            "daysLeft": (cur_end - today).days, "lastDone": s.get("lastDone")}


# ── 해외 실현이익(올해) — 체결내역으로 평균단가를 따라가며 ──

def _realized_ytd(rate: float) -> float:
    """올해 해외 종목 매도로 실현한 이익(원화, 추정). 환율은 지금 값을 쓴다."""
    year = datetime.now(KST).year
    total = 0.0
    for tk, fills in store.trades().items():
        if store.is_krw(tk):
            continue
        qty = cost = 0.0
        for f in sorted(fills, key=lambda x: x.get("date", "")):
            q, p = float(f.get("qty") or 0), float(f.get("price") or 0)
            if f.get("type") == "buy":
                qty += q; cost += q * p
            elif qty > 0:
                avg = cost / qty
                q = min(q, qty)
                if str(f.get("date", "")).startswith(str(year)):
                    total += q * (p - avg) * rate
                qty -= q; cost -= q * avg
    return total


# ── 계획 ──

def _floor_qty(q: float, krw: bool) -> float:
    """국내는 정수 주, 미국은 소수점 매매가 되므로 0.01주까지 — 0 쪽으로 자른다."""
    if krw:
        return float(math.trunc(q))
    return math.trunc(q * 100) / 100


def plan(acc: dict, quote) -> dict:
    """acc: /api/account 결과. quote(symbols) → {sym: price(현지 통화)}."""
    s = settings()
    rate = acc.get("rate") or 1400.0
    held = {it["symbol"]: it for it in acc.get("items", [])}
    basket = [t for t in s["basket"] if t]
    missing = [t for t in basket if t not in held]
    q = quote(missing) if missing else {}

    def price_krw(t):
        if t in held and held[t].get("lastPrice"):
            p = held[t]["lastPrice"]
        else:
            p = q.get(t)
        if not p or p != p:
            return None
        return p * (1.0 if store.is_krw(t) else rate)

    priced = [t for t in basket if price_krw(t)]
    outside = [{"ticker": t, "name": it.get("name") or t, "evalKrw": it.get("evalKrw", 0.0)}
               for t, it in held.items() if t not in basket]
    cash = acc.get("cashKrw", 0.0)
    invested = sum(held[t].get("evalKrw", 0.0) for t in priced if t in held)
    total = cash + invested               # 바구니 밖 보유는 계산에서 뺀다(그대로 둔다)
    cash_w = s["cashPct"] / 100
    per = total * (1 - cash_w) / len(priced) if priced else 0.0

    rows = []
    for t in priced:
        pk = price_krw(t)
        now = held[t].get("evalKrw", 0.0) if t in held else 0.0
        diff = per - now
        krw = store.is_krw(t)
        qty = _floor_qty(abs(diff) / pk, krw)
        act = "none"
        if abs(diff) >= MIN_TRADE and qty > 0:
            act = "buy" if diff > 0 else "sell"
        native = pk / (1.0 if krw else rate)
        rows.append({
            "ticker": t, "name": (held.get(t) or {}).get("name") or store.name_of(t) or t, "krw": krw,
            "valueKrw": now, "targetKrw": per, "diffKrw": diff,
            "devPct": (now / per - 1) * 100 if per else 0.0,
            "action": act, "qty": qty if act != "none" else 0.0, "amountKrw": qty * pk if act != "none" else 0.0,
            "price": native, "held": t in held,
            # 아직 안 산 종목은 '새로 담기'일 뿐 이탈이 아니다 — 알림이 매번 울리지 않게
            "extreme": t in held and per > 0 and (now >= 2 * per or now <= 0.5 * per),
        })
    rows.sort(key=lambda r: -r["devPct"])

    sells = [r for r in rows if r["action"] == "sell"]
    buys = [r for r in rows if r["action"] == "buy"]
    sell_amt = sum(r["amountKrw"] for r in sells)
    buy_amt = sum(r["amountKrw"] for r in buys)

    # 양도세 — 이번 해외 매도의 이익(지금 평단·지금 환율, 추정)
    planned = 0.0
    for r in sells:
        it = held.get(r["ticker"])
        if it and not r["krw"] and it.get("avgPrice"):
            planned += r["qty"] * (r["price"] - it["avgPrice"]) * rate
    ytd = _realized_ytd(rate)
    tax_before = max(0.0, ytd - EXEMPT) * TAX
    tax_after = max(0.0, ytd + planned - EXEMPT) * TAX

    return {
        "settings": s, "status": status(s),
        "totalKrw": total, "cashKrw": cash, "cashPctNow": cash / total * 100 if total else 0.0,
        "cashPctTarget": s["cashPct"], "perTargetKrw": per,
        "rows": rows, "outside": outside,
        "sells": {"count": len(sells), "amountKrw": sell_amt},
        "buys": {"count": len(buys), "amountKrw": buy_amt},
        "costKrw": (sell_amt + buy_amt) * COST,
        "tax": {"realizedYtd": ytd, "afterSells": ytd + planned, "exempt": EXEMPT,
                "thisTrade": tax_after - tax_before, "yearTotal": tax_after},
        "alerts": [r["ticker"] for r in rows if r["extreme"]],
        "table": table_for(s),
        "watchlist": store.tickers(),
        "defaultBasket": DEFAULT_BASKET,
        "desc": {t: DESC[t] for t in set(basket) | set(DEFAULT_BASKET) | set(store.tickers()) if DESC.get(t)},
    }


# ── 현금 비중별 과거 성과 — 바구니가 바뀌면 백그라운드로 다시 ──

_tab_lock = threading.Lock()
_tab: dict[tuple, dict] = {}
_busy: set[tuple] = set()


def _key(s: dict) -> tuple:
    return tuple(sorted(s["basket"])), s["period"], datetime.now(KST).date().isoformat()


def table_for(s: dict) -> dict | None:
    with _tab_lock:
        hit = _tab.get(_key(s))
    if hit is None:
        kick_table()
    return hit


def kick_table() -> None:
    s = settings()
    k = _key(s)
    with _tab_lock:
        if k in _tab or k in _busy:
            return
        _busy.add(k)

    def go():
        try:
            res = _compute_table(list(s["basket"]), s["period"])
            with _tab_lock:
                _tab[k] = res
        except Exception as e:      # 표가 없어도 계획은 보여 준다
            with _tab_lock:
                _tab[k] = {"error": str(e)}
        finally:
            with _tab_lock:
                _busy.discard(k)

    threading.Thread(target=go, daemon=True).start()


def _compute_table(basket: list[str], period: str) -> dict:
    series = {t: insight._fetch(t) for t in basket}
    series = {t: v for t, v in series.items() if len(v) > SIM_MIN_BARS}
    fx = insight._fetch("KRW=X")
    if not series or not fx:
        return {"error": "시세를 받지 못했습니다"}
    start = int(datetime.fromisoformat(SIM_START).timestamp()) // 86400
    all_days = sorted(set().union(*[set(v) for v in series.values()]))
    days = [d for d in all_days if d >= start]
    cols = list(series)
    us = np.array([not t[:1].isdigit() for t in cols])
    P = np.full((len(days), len(cols)), np.nan)
    B = np.zeros((len(days), len(cols)))
    fx_days = sorted(fx)
    for j, t in enumerate(cols):
        v = series[t]
        last = np.nan
        cnt = sum(1 for d in v if d < start)
        for i, d in enumerate(days):
            if d in v:
                last = v[d]; cnt += 1
            P[i, j] = last
            B[i, j] = cnt
    # 환율(직전 값)
    fxa = np.full(len(days), np.nan)
    k, cur = 0, np.nan
    for i, d in enumerate(days):
        while k < len(fx_days) and fx_days[k] <= d:
            cur = fx[fx_days[k]]; k += 1
        fxa[i] = cur
    P[:, us] = P[:, us] * fxa[:, None]
    live_ok = (B >= SIM_MIN_BARS) & ~np.isnan(P)
    dts = [date(1970, 1, 1) + timedelta(days=d) for d in days]

    def run(per, cash_w):
        return _simulate(P, live_ok, us, dts, per, cash_w)

    cash_rows = []
    for c in CASH_LEVELS:
        r = run(period, c / 100)
        cash_rows.append({"cashPct": c, **r})
    per_rows = [{"period": p, **run(p, settings()["cashPct"] / 100)} for p in PERIODS]
    return {"from": dts[0].isoformat(), "to": dts[-1].isoformat(), "tickers": cols,
            "cash": cash_rows, "periods": per_rows}


def _simulate(P, live_ok, us, dts, period, cash_w) -> dict:
    """research/rebalance_study.simulate 와 같은 규칙(분기 등 기간 말 판단 → 다음 날 매매, 세후)."""
    n = P.shape[1]
    qty = np.zeros(n); basis = np.zeros(n)
    cap = 100_000_000.0
    cash = cap
    realized: dict[int, float] = {}; owed: dict[int, float] = {}
    eq = []; pending = None; prev = None
    ends = [i == len(dts) - 1 or _period_end(dts[i], period) != _period_end(dts[i + 1], period)
            for i in range(len(dts))]
    for i, d in enumerate(dts):
        p = P[i]; valid = ~np.isnan(p)
        pv = np.where(valid, p, 0.0)
        if i > 0 and cash > 0:
            cash *= (1 + CASH_RATE) ** (1 / 252)
        if i > 0 and d.month == 1 and dts[i - 1].month == 12:
            owed[d.year - 1] = max(0.0, realized.get(d.year - 1, 0.0) - EXEMPT) * TAX
        if i > 0 and d.month == 5 and dts[i - 1].month != 5:
            due = owed.pop(d.year - 1, 0.0)
            if due > 0 and cash < due:
                val = float(np.sum(qty * pv))
                if val > 0:
                    frac = min(1.0, (due - cash) * 1.01 / val)
                    sq = qty * frac
                    avg = np.where(qty > 0, basis / np.maximum(qty, 1e-12), 0.0)
                    realized[d.year] = realized.get(d.year, 0.0) + float(np.sum(sq * (pv - avg) * us))
                    basis -= sq * avg; qty -= sq
                    cash += float(np.sum(sq * pv)) * (1 - COST)
            cash -= due
        if pending is not None:
            val = cash + float(np.sum(qty * pv))
            want = np.where(valid, val * pending / np.where(valid, p, 1.0), qty)
            dq = want - qty
            big = valid & (np.abs(dq) * pv > val * 0.002)
            s = big & (dq < 0)
            avg = np.where(qty > 0, basis / np.maximum(qty, 1e-12), 0.0)
            sq = np.where(s, -dq, 0.0)
            realized[d.year] = realized.get(d.year, 0.0) + float(np.sum(sq * (pv - avg) * us))
            basis -= sq * avg; qty -= sq
            cash += float(np.sum(sq * pv)) * (1 - COST)
            b = big & (dq > 0)
            bq = np.where(b, dq, 0.0)
            cash -= float(np.sum(bq * pv)) * (1 + COST)
            qty += bq; basis += bq * pv
            pending = None
        val = cash + float(np.sum(qty * pv))
        eq.append(val)
        live = live_ok[i]
        if not live.any() or i == len(dts) - 1:
            continue
        if prev is None or (live != prev).any() or ends[i]:
            pending = np.where(live, (1 - cash_w) / live.sum(), 0.0)
        prev = live
    eqa = np.array(eq)
    pv = np.where(np.isnan(P[-1]), 0.0, P[-1])
    unreal = float(np.sum((qty * pv - basis) * us))
    last = dts[-1].year
    pend_tax = sum(owed.values()) + max(0.0, realized.get(last, 0.0) + unreal - EXEMPT) * TAX
    after = eqa[-1] - pend_tax - float(np.sum(qty * pv)) * COST
    yrs = (dts[-1] - dts[0]).days / 365.25
    mdd = float(np.max(1 - eqa / np.maximum.accumulate(eqa)))
    r = np.diff(eqa) / eqa[:-1]
    return {"cagr": (after / cap) ** (1 / yrs) - 1 if after > 0 else -1.0, "mdd": mdd,
            "worst": cap * (1 - mdd), "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0}

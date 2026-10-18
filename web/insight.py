"""종목별 장기 제안 — **매일 아침 한 번** 긴 일봉으로 통계를 다시 내고, 분석 화면에 붙인다.

2026-10-18 검증(`research/zm_study.md`)에서 나온 결론을 앱으로 옮긴 것:
  ① Z(SPY 대비 회귀 잔차)는 대부분 종목에서 3·6·12개월 예측력이 없다. 은행처럼 통하는 종목만 쓴다.
  ② M 20/80(M<20 매수·M>80 매도)은 일부 레버리지 ETF 에서 같은 보유율의 무작위 매매보다 나았다.
     ⚠️ 이 규칙들은 **1~2년에 한 번** 매매한다(한 번 사면 보통 수개월). 근거가 매매 5~10번에서 나오므로
     강한 증거가 아니다 — 카드에 매매 횟수를 같이 보여 주고, 5번이 안 되면 판정하지 않는다.
  ③ 년 단위로는 '지난 12개월 수익률'이 자기 역사 상위 20% 면 다음 1년이 약했다.
  ④ 200일선 필터는 수익보다 낙폭을 줄인다.

**종목 분류를 손으로 박지 않는다.** 규칙이 통하는지는 매일 데이터로 다시 판정한다 —
은행이 내년에 안 통하게 되면 카드도 저절로 '근거 없음'으로 바뀐다.

데이터는 토스가 아니라 Yahoo 수정종가다. 토스 일봉은 24개월이라 년 단위 검증에 모자라고,
레버리지 ETF 는 상장이 짧아 **추종하는 기초자산**(SOXL→SOXX 등)의 긴 기록으로 함께 본다.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import quant
import store


@dataclass(frozen=True)
class Config:
    win: int = 504              # Z 회귀 창 — 앱의 24개월과 같게
    min_test: int = 756         # 규칙 검증에 쓸 최소 기간(3년). 이보다 짧으면 그 종목 자신은 판정에서 뺀다
    sims: int = 200             # 무작위 비교 횟수
    min_trades: int = 5         # 규칙 매매가 이보다 적으면 판정하지 않는다
    lo: float = 20.0            # 앱의 매수 문턱
    hi: float = 80.0            # 앱의 매도 문턱
    works: float = 85.0         # 무작위보다 이만큼(백분위) 나으면 '통함'
    harmful: float = 25.0       # 이 아래면 '역효과'
    mom_edge: float = 0.10      # 12개월 구간의 다음 1년 평균이 전체보다 10%p 이상 차이 나야 반영
    cost: float = 0.001         # 매매 한 번 비용
    refresh_hour_kst: int = 7   # 미국장이 끝난 뒤(KST 05~06시) 여유를 두고


CFG = Config()
KST = timezone(timedelta(hours=9))
DIR = store.DATA / "yahoo"
OUT = store.DATA / "insight.json"

# 레버리지 ETF → 추종하는 기초자산. 상장이 짧은 종목은 이쪽 기록만으로 판정한다.
PROXY = {
    "BITU": "BTC-USD", "SOXL": "SOXX", "TARK": "ARKK", "HIBL": "SPHB", "TQQQ": "QQQ",
    "DFEN": "ITA", "DPST": "KRE", "GDXU": "GDX", "WTIU": "USO", "KORU": "EWY",
    "BNKU": "KBE", "NRGU": "XLE", "LABU": "XBI", "NAIL": "ITB", "URAA": "URA",
    "FNGU": "QQQ", "YINN": "FXI",
}

_lock = threading.Lock()
_data: dict = {}


# ── 데이터 ──

def _yahoo_symbols(sym: str) -> list[str]:
    if sym[:1].isdigit():
        return [sym + ".KS", sym + ".KQ"]       # 코스피에 없으면 코스닥
    return [sym]


def _fetch(sym: str) -> dict[int, float]:
    """전체 기간 일봉(수정종가) → {거래일: 종가}. 하루 한 번만 받고 나머지는 파일에서."""
    DIR.mkdir(parents=True, exist_ok=True)
    f = DIR / f"{''.join(c if c.isalnum() else '_' for c in sym)}.json"
    today = datetime.now(KST).date().isoformat()
    rows = None
    if f.exists():
        try:
            o = json.loads(f.read_text())
            if o.get("day") == today:
                rows = o["rows"]
        except (ValueError, OSError, KeyError):
            pass
    if rows is None:
        for y in _yahoo_symbols(sym):
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{y}"
                   f"?period1=0&period2={int(time.time())}&interval=1d&events=split")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                d = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
                adj = d["indicators"]["adjclose"][0]["adjclose"]
                rows = [[t, c] for t, c in zip(d["timestamp"], adj) if c and c > 0]
                if rows:
                    f.write_text(json.dumps({"day": today, "rows": rows}))
                    break
            except Exception:
                rows = None
            time.sleep(0.3)
        if rows is None and f.exists():       # 못 받으면 지난 파일이라도
            try:
                rows = json.loads(f.read_text())["rows"]
            except (ValueError, OSError, KeyError):
                rows = None
    # 한국 봉은 한국 자정(UTC 전날 15시)이라 9시간 더해야 같은 거래일이 된다
    return {(int(t) + 9 * 3600) // 86400: float(c) for t, c in (rows or [])}


# ── 계산 ──

def _rolling_z(lx: np.ndarray, ly: np.ndarray, win: int) -> np.ndarray:
    """그날까지의 창으로만 회귀한 Z — 미래 정보가 섞이지 않는다. 누적합으로 O(n)."""
    n = len(lx)
    z = np.full(n, np.nan)
    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    Sx, Sy, Sxx, Sxy, Syy = cs(lx), cs(ly), cs(lx * lx), cs(lx * ly), cs(ly * ly)
    for t in range(win - 1, n):
        a0, a1 = t + 1 - win, t + 1
        sx, sy = Sx[a1] - Sx[a0], Sy[a1] - Sy[a0]
        cxx = Sxx[a1] - Sxx[a0] - sx * sx / win
        cxy = Sxy[a1] - Sxy[a0] - sx * sy / win
        cyy = Syy[a1] - Syy[a0] - sy * sy / win
        if cxx <= 1e-12:
            continue
        b = cxy / cxx
        a = sy / win - b * sx / win
        sd = math.sqrt(max(cyy - b * cxy, 0.0) / (win - 2))
        if sd > 0:
            z[t] = (ly[t] - a - b * lx[t]) / sd
    return z


def _m_score(y: np.ndarray) -> np.ndarray:
    """quant.analyze 의 M 과 같은 식 — Z 쪽 계산을 빼서 긴 기록에도 빠르게."""
    v = [float(x) for x in y]
    n = len(v)
    rsi = quant._rsi(v)
    e12, e26 = quant._ema(v, 12), quant._ema(v, 26)
    macd = [e12[i] - e26[i] for i in range(n)]
    mp = [macd[i] / e26[i] * 100 for i in range(n)]
    dm = quant._ema([0.0] + [macd[i] - macd[i - 1] for i in range(1, n)], 3)
    dp = [-(dm[i] / e26[i] * 100) for i in range(n)]
    ms = quant._rolling_std(mp, quant.M_VOL_WINDOW, quant.EXPANDING_MIN)
    ds = quant._rolling_std(dp, quant.M_VOL_WINDOW, quant.EXPANDING_MIN)
    return np.array([quant._momentum(mp[i], dp[i], rsi[i], ms[i], ds[i]) for i in range(n)])


def _pct(z: np.ndarray) -> np.ndarray:
    return np.clip((z + 2.5) / 5 * 100, 0, 100)


def _band(score: np.ndarray) -> np.ndarray:
    """문턱 아래에서 사고 위에서 판다. pos[t] 는 t 종가 판단 → t+1 부터 보유."""
    pos = np.zeros(len(score))
    cur = 0
    for t, s in enumerate(score):
        if s == s:
            if cur == 0 and s < CFG.lo:
                cur = 1
            elif cur == 1 and s > CFG.hi:
                cur = 0
        pos[t] = cur
    return pos


def _cagr_mdd(close: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    lr = np.diff(np.log(close))
    p = pos[:-1]
    r = p * lr - np.abs(np.diff(np.concatenate([[0.0], p]))) * CFG.cost
    eq = np.cumsum(r)
    yrs = max(len(r) / 252, 1e-9)
    mdd = 1 - math.exp(-float(np.max(np.maximum.accumulate(eq) - eq))) if len(eq) else 0.0
    return math.exp(eq[-1] / yrs) - 1, mdd


def _vs_random(close: np.ndarray, pos: np.ndarray, seed: int) -> tuple[float | None, int]:
    """같은 보유율·평균 보유 기간의 무작위 매매 중 몇 %보다 나았나.

    그냥 보유와 비교하면 안 된다 — 레버리지 ETF 는 **덜 들고 있기만 해도** 감쇠를 덜 맞아 이긴다.
    """
    p = pos.mean()
    entries = int(np.sum(np.diff(pos) > 0)) + int(pos[0] > 0)
    if entries < CFG.min_trades or p <= 0 or p >= 1:
        return None, entries
    mine = _cagr_mdd(close, pos)[0]
    hold, gap = p * len(pos) / entries, (1 - p) * len(pos) / entries
    rng = np.random.default_rng(seed)
    beat = 0
    for _ in range(CFG.sims):
        q = np.zeros(len(pos))
        t, cur = 0, rng.random() < p
        while t < len(pos):
            d = max(1, int(rng.exponential(hold if cur else gap)))
            q[t:t + d] = cur
            t += d
            cur = not cur
        if _cagr_mdd(close, q)[0] < mine:
            beat += 1
    return beat / CFG.sims * 100, entries


def _mom_stats(y: np.ndarray) -> dict | None:
    """지난 12개월 수익률 5분위별 다음 12개월 평균 + 지금 어느 분위인지."""
    if len(y) < 250 * 4:
        return None
    ly = np.log(y)
    mom = np.full(len(y), np.nan)
    mom[250:] = ly[250:] - ly[:-250]
    fwd = np.full(len(y), np.nan)
    fwd[:-251] = ly[251:] - ly[1:-250]
    ok = ~np.isnan(mom) & ~np.isnan(fwd)
    if ok.sum() < 500:
        return None
    qs = np.percentile(mom[ok], [20, 40, 60, 80])
    b = np.digitize(mom, qs)
    by_q = [math.exp(float(np.mean(fwd[ok & (b == q)]))) - 1 for q in range(5)]
    return {
        "now": math.exp(float(mom[-1])) - 1,
        "quintile": int(np.digitize([mom[-1]], qs)[0]) + 1,
        "fwd": by_q,
        "all": math.exp(float(np.mean(fwd[ok]))) - 1,
        "years": round(len(y) / 252, 1),
    }


def _series_stats(sym: str, spy: dict[int, float]) -> dict | None:
    tk = _fetch(sym)
    if sym.endswith("-USD"):
        # 코인은 주말에도 봉이 있어 250봉이 12개월이 아니다 — 미국 거래일로 맞춘다
        tk = {d: c for d, c in tk.items() if d in spy}
    if len(tk) < 300:
        return None
    seed = zlib.crc32(sym.encode())
    own = np.array([tk[d] for d in sorted(tk)])
    out: dict = {"sym": sym, "years": round(len(own) / 252, 1), "mom": _mom_stats(own)}

    # 200일선 — 지금 위치 + 그 필터가 낙폭을 얼마나 줄였나
    if len(own) >= 200 + 252:
        ma = np.convolve(own, np.ones(200) / 200, "valid")
        above = np.concatenate([np.zeros(199), (own[199:] > ma).astype(float)])
        sl = slice(199, None)
        bh = _cagr_mdd(own[sl], np.ones(len(own) - 199))
        tf = _cagr_mdd(own[sl], above[sl])
        out["trend"] = {"gap": own[-1] / ma[-1] - 1, "mddHold": bh[1], "mddFilter": tf[1],
                        "cagrHold": bh[0], "cagrFilter": tf[0]}
    elif len(own) >= 200:
        out["trend"] = {"gap": own[-1] / own[-200:].mean() - 1}

    # Z·M 규칙 검증 — SPY 와 날짜를 맞춰서
    days = sorted(set(spy) & set(tk))
    if len(days) >= CFG.win + CFG.min_test:
        x = np.array([spy[d] for d in days])
        y = np.array([tk[d] for d in days])
        z = _rolling_z(np.log(x / x[0]), np.log(y / y[0]), CFG.win)
        m = _m_score(y)
        sl = slice(CFG.win, None)
        yy = y[sl]
        out["testYears"] = round(len(yy) / 252, 1)
        out["zRank"], out["zTrades"] = _vs_random(yy, _band(_pct(z[sl])), seed)
        out["mRank"], out["mTrades"] = _vs_random(yy, _band(_pct(m[sl])), seed + 1)
    return out


def _compute(tickers: list[str]) -> dict:
    spy = _fetch(store.BASE)
    stats: dict[str, dict] = {}
    for tk in tickers:
        try:
            own = _series_stats(tk, spy)
            px = PROXY.get(tk)
            stats[tk] = {"own": own, "proxy": _series_stats(px, spy) if px else None}
        except Exception as e:          # 한 종목 실패로 전체를 버리지 않는다
            stats[tk] = {"error": str(e)}
    return {"day": datetime.now(KST).date().isoformat(), "at": time.time(), "tickers": stats}


# ── 공개 API ──

def load() -> dict:
    global _data
    with _lock:
        if not _data and OUT.exists():
            try:
                _data = json.loads(OUT.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                _data = {}
        return _data


def stale(tickers: list[str]) -> bool:
    d = load()
    now = datetime.now(KST)
    if not d:
        return True
    if set(tickers) - set(d.get("tickers", {})):
        return True                      # 새로 넣은 종목이 있다
    return d.get("day") != now.date().isoformat() and now.hour >= CFG.refresh_hour_kst


def refresh(tickers: list[str]) -> None:
    global _data
    d = _compute(tickers)
    with _lock:
        _data = d
        OUT.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def _evidence(own: dict | None, proxy: dict | None, key: str) -> tuple[float | None, list[str], bool]:
    """그 종목 자신과 기초자산의 백분위를 평균. 하나뿐이면 그것만.

    반환: (평균 백분위, 설명 조각, 검증은 했는데 매매가 너무 적었는지)
    설명 조각의 'SOXL 99%·8회' 는 '무작위 매매 99%보다 나았다, 매매 8번 기준' 이라는 뜻.
    """
    vals, parts, few = [], [], False
    tkey = key.replace("Rank", "Trades")
    for s in (own, proxy):
        if not s:
            continue
        v, n = s.get(key), s.get(tkey)
        if v is not None:
            vals.append(v)
            parts.append(f"{s['sym']} {v:.0f}%·{n}회")
        elif n is not None:
            few = True
    return (sum(vals) / len(vals) if vals else None), parts, few


def _pc(v: float) -> str:
    return f"{v * 100:+.0f}%"


def advice(ticker: str, z_now: float | None, m_now: float | None) -> dict | None:
    """지금 Z·M(앱 화면과 같은 값) + 매일 낸 통계 → 제안 한 벌. 통계가 아직 없으면 None."""
    d = load()
    st = d.get("tickers", {}).get(ticker)
    if not st or "error" in st:
        return None
    own, px = st.get("own"), st.get("proxy")
    if not own and not px:
        return None
    z_ev, z_parts, z_few = _evidence(own, px, "zRank")
    m_ev, m_parts, m_few = _evidence(own, px, "mRank")
    z_ok = z_ev is not None and z_ev >= CFG.works
    z_bad = z_ev is not None and z_ev <= CFG.harmful
    m_ok = m_ev is not None and m_ev >= CFG.works

    score = 0
    lines: list[str] = []
    drivers: list[str] = []          # 점수를 움직인 줄 — 제목으로 쓴다

    # ① Z
    if z_ok:
        if z_now is not None and z_now < CFG.lo:
            score += 2
            lines.append(f"Z {z_now:.0f} — 매수 구간. 이 종목은 Z 역발상이 통했다 ({' · '.join(z_parts)})")
            drivers.append(f"Z {z_now:.0f} 매수 구간")
        elif z_now is not None and z_now > CFG.hi:
            score -= 2
            lines.append(f"Z {z_now:.0f} — 매도 구간. 이 종목은 Z 역발상이 통했다 ({' · '.join(z_parts)})")
            drivers.append(f"Z {z_now:.0f} 매도 구간")
        else:
            lines.append(f"Z {z_now:.0f} — 중간. Z 20 아래 매수·80 위 매도가 통하는 종목 ({' · '.join(z_parts)})"
                         if z_now is not None else "Z 역발상이 통하는 종목")
    elif z_bad:
        lines.append(f"Z 가 낮다고 사면 손해였다 — 이 종목은 강할 때 계속 강했다 ({' · '.join(z_parts)})")
    elif z_ev is not None:
        lines.append(f"Z 로 타이밍 잡을 근거 없음 ({' · '.join(z_parts)})")
    elif z_few:
        lines.append("Z 20/80 매매가 너무 드물어 판단 불가")

    # ② M — 한 번 사면 보통 수개월 간다
    if m_ok:
        if m_now is not None and m_now < CFG.lo:
            score += 1
            lines.append(f"M {m_now:.0f} — 과매도. M 20 아래 매수·80 위 매도가 통한 종목 ({' · '.join(m_parts)})")
            drivers.append(f"M {m_now:.0f} 과매도")
        elif m_now is not None and m_now > CFG.hi:
            score -= 1
            lines.append(f"M {m_now:.0f} — 과매수, 이익 실현 자리. M 20/80 이 통한 종목 ({' · '.join(m_parts)})")
            drivers.append(f"M {m_now:.0f} 과매수")
        else:
            lines.append(f"M {m_now:.0f} — 중간. M 20 아래 매수·80 위 매도가 통한 종목 ({' · '.join(m_parts)})"
                         if m_now is not None else "M 20/80 이 통한 종목")
    elif m_ev is not None:
        lines.append(f"M 20/80 도 무작위 매매와 차이 없음 ({' · '.join(m_parts)})")
    elif m_few:
        lines.append("M 20/80 매매가 너무 드물어 판단 불가")

    # ③ 지난 12개월 — 자기 기록이 짧으면 기초자산 것으로
    mom_src = own if own and own.get("mom") else px
    mom = mom_src.get("mom") if mom_src else None
    if mom:
        q = mom["quintile"]
        fq, fa = mom["fwd"][q - 1], mom["all"]
        tag = "" if mom_src is own else f" ({mom_src['sym']} 기준)"
        pos_txt = {1: "하위 20%", 2: "하위 40%", 3: "중간", 4: "상위 40%", 5: "상위 20%"}[q]
        lines.append(f"지난 12개월 {_pc(mom['now'])} — {mom['years']:.0f}년 기록 중 {pos_txt}{tag}. "
                     f"이 구간 다음 1년 평균 {_pc(fq)} (전체 {_pc(fa)})")
        if q == 5 and fq - fa <= -CFG.mom_edge:
            score -= 1
            drivers.append("1년 과열")
        elif q == 1 and fq - fa >= CFG.mom_edge:
            score += 1
            drivers.append("1년 침체")

    # ④ 200일선
    tr_src = own if own and own.get("trend") else px
    tr = tr_src.get("trend") if tr_src else None
    if tr:
        side = "위" if tr["gap"] >= 0 else "아래"
        txt = f"200일선 {side} ({_pc(tr['gap'])})"
        if "mddHold" in tr:
            txt += f" — 200일선 아래에서 쉬면 최대 낙폭 {tr['mddHold'] * 100:.0f}% → {tr['mddFilter'] * 100:.0f}%"
        lines.append(txt)

    usable = z_ok or m_ok or (mom is not None)
    if score >= 2:
        action, tone = "매수 고려", "buy"
    elif score == 1:
        action, tone = "분할 매수", "buy"
    elif score == -1:
        action, tone = "비중 축소", "sell"
    elif score <= -2:
        action, tone = "매도 고려", "sell"
    elif not (z_ok or m_ok):
        action, tone = "타이밍 근거 없음", "none"
    else:
        action, tone = "관망", "hold"

    if tone == "none":
        head = "Z·M 으로 사고팔 근거가 없다 — 비중과 200일선으로 관리"
    elif tone == "hold":
        head = "신호 구간 밖 — " + ("Z 20 아래를 기다린다" if z_ok else "M 20 아래를 기다린다")
    else:
        head = " · ".join(drivers) or action

    return {
        "action": action, "tone": tone, "score": score, "headline": head, "lines": lines,
        "zEvidence": z_ev, "mEvidence": m_ev, "usable": usable,
        "day": d.get("day"),
    }


def loop(tickers_fn) -> None:
    """서버 쓰레드 — 30분마다 '오늘 것을 냈나' 만 본다. 계산은 하루 한 번(종목이 늘면 그때 또)."""
    time.sleep(20)                      # 서버가 뜨자마자 무거운 일을 하지 않는다
    while True:
        try:
            tks = list(tickers_fn())
            if stale(tks):
                refresh(tks)
        except Exception:
            pass
        time.sleep(1800)

"""로컬 저장소 — 설정·종목·매매기록·원금. `data/Store.kt` 대응.

안드로이드는 앱 내부 저장소에 JSON 을 뒀는데, 여기서는 `web/data/` 아래 파일로 둔다.
앱 키만은 여기 두지 않는다 — `config.json` 또는 환경변수에서 읽는다.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

# 기본은 web/data. QUANT_DATA 로 바꿀 수 있는데, 이건 테스트가 진짜 기록(입금·매매·
# 스냅샷)을 건드리지 않게 하려는 것이다. 평소에는 설정하지 않는다.
DATA = Path(os.environ.get("QUANT_DATA") or Path(__file__).parent / "data")
_lock = threading.Lock()

# 기본 종목 — Tickers.kt DEFAULT 와 같은 목록
DEFAULT_TICKERS = [
    "FNGU", "TQQQ", "SOXL", "HIBL", "QPUX", "LABU", "DFEN", "DPST",
    "GDXU", "KORU", "005930", "AVXX", "SPYU", "TARK", "URTY", "TNA",
    "BNKU", "GLD",
]
BASE = "SPY"          # 회귀 기준 자산
MAX_MONTHS = 24


def _read(name: str, default):
    f = DATA / name
    if not f.exists():
        return default
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def _write(name: str, value) -> None:
    with _lock:
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


# ── 설정 ──

def settings() -> dict:
    return _read("settings.json", {})


def put(key: str, value) -> None:
    s = settings()
    s[key] = value
    _write("settings.json", s)


def lookback_months() -> int:
    v = settings().get("lookback_months", MAX_MONTHS)
    try:
        return max(3, min(MAX_MONTHS, int(v)))
    except (TypeError, ValueError):
        return MAX_MONTHS


def bar_count(months: int) -> int:
    """토스는 봉 수로 요청한다 — 개월당 약 22 거래일 + 경계 여유."""
    return max(40, months * 22 + 15)


# ── 종목 ──

def tickers() -> list[str]:
    v = _read("tickers.json", None)
    return list(v) if isinstance(v, list) and v else list(DEFAULT_TICKERS)


def set_tickers(v: list[str]) -> None:
    _write("tickers.json", v)


def add_ticker(t: str) -> list[str]:
    t = t.strip().upper()
    cur = tickers()
    if t and t not in cur:
        cur.append(t)
        set_tickers(cur)
    return cur


def remove_ticker(t: str) -> list[str]:
    cur = [x for x in tickers() if x != t]
    set_tickers(cur)
    return cur


def is_krw(ticker: str) -> bool:
    """원화로 표시할 종목인가 — 6자리 숫자 코드면 국내."""
    t = ticker.split(".")[0]
    return len(t) == 6 and t.isdigit()


# ── 매매 기록 (체결내역에서 가져온 것) ──

def trades() -> dict[str, list[dict]]:
    """티커 → 체결 목록(날짜 오름차순)."""
    return _read("trades.json", {})


def set_trades(v: dict[str, list[dict]]) -> None:
    _write("trades.json", v)


def save_fills(fills: list[dict]) -> int:
    """체결내역을 티커별로 묶어 저장. 같은 주문(orderId)은 덮어쓴다."""
    by: dict[str, dict[str, dict]] = {}
    for t, lst in trades().items():
        by[t] = {x.get("orderId", f"{x['date']}-{x['price']}"): x for x in lst}
    for f in fills:
        sym = f["symbol"]
        by.setdefault(sym, {})[f["orderId"]] = {
            "orderId": f["orderId"], "date": f["date"], "qty": f["quantity"],
            "price": f["price"], "type": "buy" if f["buy"] else "sell",
        }
    out = {t: sorted(d.values(), key=lambda x: x["date"]) for t, d in by.items()}
    set_trades(out)
    return sum(len(v) for v in out.values())


def position(tk_trades: list[dict]) -> dict | None:
    """현재 사이클의 보유 수량·평단 (`Portfolio.position` 대응).

    전량 매도하면 사이클이 끝나 평단이 초기화된다 — 그래야 재진입 후 평단이 맞는다.
    """
    hold = 0.0
    buy_qty = 0.0
    buy_cost = 0.0
    for t in sorted([x for x in tk_trades if x.get("qty", 0) > 0 and x.get("price", 0) > 0],
                    key=lambda x: x["date"]):
        if t["type"] == "buy":
            hold += t["qty"]
            buy_qty += t["qty"]
            buy_cost += t["qty"] * t["price"]
        else:
            hold -= t["qty"]
            if hold <= 1e-9:          # 전량 매도 → 사이클 종료
                hold = 0.0
                buy_qty = 0.0
                buy_cost = 0.0
    if hold > 0 and buy_qty > 0:
        return {"qty": hold, "avg": buy_cost / buy_qty}
    return None


# ── 원금 (입금 장부) ──

def deposits() -> list[dict]:
    v = _read("deposits.json", [])
    return sorted(v, key=lambda x: x.get("date", "")) if isinstance(v, list) else []


def add_deposit(date: str, krw: float) -> list[dict]:
    v = deposits() + [{"date": date, "krw": krw}]
    _write("deposits.json", sorted(v, key=lambda x: x["date"]))
    return deposits()


def remove_deposit(index: int) -> list[dict]:
    v = deposits()
    if 0 <= index < len(v):
        del v[index]
        _write("deposits.json", v)
    return deposits()


def principal_total() -> float:
    return sum(d.get("krw", 0.0) for d in deposits())

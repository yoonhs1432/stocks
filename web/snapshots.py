"""일별 잔고 스냅샷 — `data/Snapshots.kt` 대응.

토스에는 **과거 잔고 API 가 없다.** 그래서 계좌를 조회할 때마다 그날 값을 남겨 두고,
그걸 이어 자산 추이를 그린다. 하루 1회 덮어쓰기라 앱을 여러 번 열어도 하루 한 점이다.

⚠️ 기록이 시작된 날부터만 쌓인다. 앱을 안 연 날은 비어 있다.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

FILE = Path(os.environ.get("QUANT_DATA") or Path(__file__).parent / "data") / "snapshots.json"
MAX = 1500          # 약 4년치
_lock = threading.Lock()

KST = timezone(timedelta(hours=9))


def today() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def load() -> list[dict]:
    if not FILE.exists():
        return []
    try:
        v = json.loads(FILE.read_text(encoding="utf-8"))
        return sorted(v, key=lambda x: x.get("date", "")) if isinstance(v, list) else []
    except (ValueError, OSError):
        return []


def record(acc: dict) -> None:
    """계좌 조회에 성공했을 때 호출. 같은 날짜가 있으면 덮어쓴다.

    ⚠️ **보유 내역도 같이 남긴다.** 시세는 나중에 토스에서 다시 받을 수 있지만
    "그날 무엇을 얼마나, 평단 얼마에 들고 있었는가" 는 지나가면 어디에도 없다.
    하루 한 줄에 종목당 40바이트 남짓이라 4년을 모아도 1MB 가 안 된다.
    """
    row = {
        "date": today(),
        "krwEval": acc["krwEval"], "usdEval": acc["usdEval"],
        "krwCash": acc["krwCash"], "usdCash": acc["usdCash"],
        "rate": acc["rate"],
        # 평가손익은 통화별로 받아 두면 나중에 그날 환율로 되돌릴 수 있다
        "pnlKrw": acc["pnlKrw"],
        # s=종목 n=이름 q=수량 a=평단 p=현재가 e=평가금액(원) g=평가손익(원)
        "items": [{"s": h["symbol"], "n": h.get("name") or h["symbol"],
                   "q": h["quantity"], "a": h["avgPrice"], "p": h["lastPrice"],
                   "e": h["evalKrw"], "g": h["pnlKrw"]}
                  for h in acc.get("items", [])],
    }
    with _lock:
        rows = [r for r in load() if r.get("date") != row["date"]] + [row]
        rows = sorted(rows, key=lambda x: x["date"])[-MAX:]
        try:
            FILE.parent.mkdir(parents=True, exist_ok=True)
            FILE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def series(deposits: list[dict], usd: bool = False) -> dict:
    """자산 추이 — 평가금액·예수금·총자산·평가손익·원금.

    ⚠️ 과거 금액을 **오늘 환율**로 바꾸면 환율 변동이 자산 변동처럼 보인다.
    그래서 달러 환산은 그날 저장해 둔 rate 를 쓴다.

    원금은 그 날짜까지의 입금 누적. 기록이 없는 구간은 None(선을 긋지 않는다).
    """
    rows = load()
    deps = sorted(deposits, key=lambda d: d.get("date", ""))

    dates, ev, cash, total, pnl, prin = [], [], [], [], [], []
    k = 0
    acc = 0.0
    for r in rows:
        rate = r.get("rate") or 1400.0
        conv = (lambda v: v / rate) if usd else (lambda v: v)
        e = r["krwEval"] + r["usdEval"] * rate
        c = r["krwCash"] + r["usdCash"] * rate
        while k < len(deps) and deps[k]["date"] <= r["date"]:
            acc += deps[k]["krw"]
            k += 1
        dates.append(r["date"])
        ev.append(conv(e))
        cash.append(conv(c))
        total.append(conv(e + c))
        pnl.append(conv(r.get("pnlKrw", 0.0)))
        prin.append(conv(acc) if k > 0 else None)

    return {"dates": dates, "eval": ev, "cash": cash, "total": total,
            "pnl": pnl, "principal": prin}


def history(days: int = 120) -> dict:
    """최근 N일의 보유 내역. 화면에서 날짜를 고르면 그날 표를 보여 주려고.

    종목별 평가금액 추이(쌓은 그래프)와 날짜별 표를 한 번에 그릴 수 있게 함께 낸다.
    """
    rows = [r for r in load() if r.get("items")][-max(1, days):]
    dates = [r["date"] for r in rows]

    names: dict[str, str] = {}
    series: dict[str, list] = {}
    by_date: dict[str, list] = {}
    for i, r in enumerate(rows):
        by_date[r["date"]] = r["items"]
        for it in r["items"]:
            sym = it["s"]
            names.setdefault(sym, it.get("n") or sym)
            series.setdefault(sym, [None] * len(rows))
            series[sym][i] = it["e"]

    # 마지막 날 평가금액이 큰 종목부터 — 그래프를 쌓는 순서이기도 하다
    order = sorted(series, key=lambda k: -(series[k][-1] or 0))
    return {"dates": dates,
            "symbols": [{"symbol": k, "name": names[k]} for k in order],
            "eval": {k: series[k] for k in order},
            "byDate": by_date}

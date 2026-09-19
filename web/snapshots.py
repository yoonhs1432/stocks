"""일별 잔고 스냅샷 — `data/Snapshots.kt` 대응.

토스에는 **과거 잔고 API 가 없다.** 그래서 계좌를 조회할 때마다 그날 값을 남겨 두고,
그걸 이어 자산 추이를 그린다. 하루 1회 덮어쓰기라 앱을 여러 번 열어도 하루 한 점이다.

⚠️ 기록이 시작된 날부터만 쌓인다. 앱을 안 연 날은 비어 있다.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

FILE = Path(__file__).parent / "data" / "snapshots.json"
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
    """계좌 조회에 성공했을 때 호출. 같은 날짜가 있으면 덮어쓴다."""
    row = {
        "date": today(),
        "krwEval": acc["krwEval"], "usdEval": acc["usdEval"],
        "krwCash": acc["krwCash"], "usdCash": acc["usdCash"],
        "rate": acc["rate"],
        # 평가손익은 통화별로 받아 두면 나중에 그날 환율로 되돌릴 수 있다
        "pnlKrw": acc["pnlKrw"],
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

"""국내 종목 이름 — `GET /api/v1/stocks/all` 을 하루 1회 받아 둔다.

왜 필요한가 — 국내 종목은 코드(005930)만 보면 뭔지 알 수 없다. 보유 중이면 토스가
이름을 주지만 관심종목으로만 담아 둔 종목은 이름이 없어 숫자만 떴다.

국내(KOSPI·KOSDAQ·KR_ETC)만 받는다. 미국 종목도 토스는 한글 이름을 주지만, 비교 표가
갑자기 한글로 바뀌면 오히려 알아보기 어렵다(안드로이드에서도 같은 이유로 안 썼다).

하루 1회면 충분하다 — 상장·폐지는 그보다 자주 일어나지 않는다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

FILE = Path(os.environ.get("QUANT_DATA") or Path(__file__).parent / "data") / "universe.json"
MARKETS = ("KOSPI", "KOSDAQ", "KR_ETC")
KST = timezone(timedelta(hours=9))

_lock = threading.Lock()
_mem: dict = {}          # {"date": "YYYY-MM-DD", "names": {symbol: name}}
_busy = False


def _today() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _load() -> dict:
    global _mem
    if _mem:
        return _mem
    try:
        v = json.loads(FILE.read_text(encoding="utf-8"))
        if isinstance(v, dict) and isinstance(v.get("names"), dict):
            _mem = v
    except (OSError, ValueError):
        _mem = {}
    return _mem


def names() -> dict[str, str]:
    """코드 → 이름. 없으면 빈 사전 — 화면은 코드를 그대로 보여 주면 된다."""
    return _load().get("names", {})


def name_of(symbol: str) -> str:
    return names().get(symbol.strip().upper(), "")


def is_kr(symbol: str) -> bool:
    """받아 둔 국내 목록에 있는가. 없으면 모른다는 뜻이라 False."""
    return symbol.strip().upper() in names()


def ensure(toss, force: bool = False) -> int:
    """오늘 받은 게 없으면 받아 온다. 실패해도 예전 것을 그대로 쓴다."""
    global _busy, _mem
    cur = _load()
    if not force and cur.get("date") == _today() and cur.get("names"):
        return len(cur["names"])
    with _lock:
        if _busy:
            return len(cur.get("names", {}))
        _busy = True
    try:
        got, ok = {}, 0
        for m in MARKETS:
            try:
                for it in toss.list_stocks(m):
                    if it["name"]:
                        got[it["symbol"].upper()] = it["name"]
                ok += 1
            except Exception:
                pass          # 한 마켓이 실패해도 나머지는 쓴다
        if not got:
            return len(cur.get("names", {}))
        # 일부 마켓만 받아졌으면 날짜를 비워 다음에 다시 시도한다
        merged = {**cur.get("names", {}), **got}
        _mem = {"date": _today() if ok == len(MARKETS) else "", "names": merged,
                "at": time.time()}
        try:
            FILE.parent.mkdir(parents=True, exist_ok=True)
            FILE.write_text(json.dumps(_mem, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return len(merged)
    finally:
        with _lock:
            _busy = False

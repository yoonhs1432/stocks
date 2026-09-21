"""장 운영시간 — 실시간 조회를 **언제 돌릴지** 판정한다.

토스 `/market-calendar` 를 하루 한 번 받아 한국(프리·정규·애프터)과
미국(데이·프리·정규·애프터) 세션을 전부 들고 있는다. 시간을 코드에 박아 두면
휴장일·서머타임·시간외를 못 맞춘다.

왜 필요한가 — 이게 없으면 장이 닫힌 새벽에도 1초마다 조회한다. 데이터도 아깝고,
화면은 숫자가 안 바뀌는 이유를 알려 주지 못해 고장과 구분이 안 된다.

토스를 못 부르면 대략 판정으로 물러난다(평일 + 한국 09:00~15:30 / 미국 04:00~20:00 ET).
휴장일은 못 걸러내지만 밤새 조회하는 것보다는 낫다.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

_lock = threading.Lock()
_sessions: list[dict] = []
_day = ""            # 세션을 받아 둔 날짜 (UTC 기준 YYYY-MM-DD)
_note = ""           # 못 받았으면 그 이유


def ensure(toss, force: bool = False) -> None:
    """하루 한 번 세션표를 받아 둔다. 실패해도 예외를 밖으로 내보내지 않는다."""
    global _sessions, _day, _note
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _lock:
        if not force and _day == today and _sessions:
            return
    out: list[dict] = []
    err = ""
    for country in ("KR", "US"):
        try:
            out += toss.market_sessions(country)
        except Exception as e:                      # 연동 전·한도·네트워크 — 폴백으로 간다
            err = str(e)
    with _lock:
        if out:
            _sessions, _day, _note = out, today, ""
        else:
            _note = err or "장 시간표를 못 받았습니다"


def _now_open(now: float) -> list[dict]:
    with _lock:
        return [s for s in _sessions if s["start"] <= now <= s["end"]]


def _fallback_open(now: float) -> bool:
    """세션표가 없을 때 — 평일 + 대략의 시간대. 휴장일은 못 걸러낸다."""
    kst = datetime.fromtimestamp(now, ZoneInfo("Asia/Seoul"))
    et = datetime.fromtimestamp(now, ZoneInfo("America/New_York"))
    kr = kst.weekday() < 5 and 9 * 60 <= kst.hour * 60 + kst.minute <= 15 * 60 + 30
    us = et.weekday() < 5 and 4 <= et.hour < 20
    return kr or us


def is_open(now: float | None = None) -> bool:
    now = time.time() if now is None else now
    with _lock:
        have = bool(_sessions)
    return bool(_now_open(now)) if have else _fallback_open(now)


def label(now: float | None = None) -> str:
    """'US 정규장' · 'KR 정규장 · US 프리마켓' · '장 마감'."""
    now = time.time() if now is None else now
    open_now = _now_open(now)
    if open_now:
        return " · ".join(f"{s['market']} {s['name']}" for s in open_now)
    with _lock:
        have = bool(_sessions)
    if not have:
        return "장중(추정)" if _fallback_open(now) else "장 마감(추정)"
    return "장 마감"


def session_start(market: str, now: float | None = None) -> int | None:
    """그 시장에서 **지금 또는 마지막으로** 시작된 세션의 시작 시각.

    장 마감 중에는 마지막 세션의 종가가 정상값이므로, 이 시각보다 앞선 체결만
    '이번 장에 체결 없음'으로 본다.
    """
    now = time.time() if now is None else now
    with _lock:
        started = [s["start"] for s in _sessions if s["market"] == market and s["start"] <= now]
    return max(started) if started else None


def is_stale(market: str, at, now: float | None = None) -> bool:
    """이번(또는 마지막) 세션에 체결이 없었는가. 판정할 수 없으면 False."""
    start = session_start(market, now)
    if start is None:
        return False
    if at is None:
        return True
    try:
        return float(at) < start
    except (TypeError, ValueError):
        return False


def status(toss, now: float | None = None) -> dict:
    """화면에 뿌릴 한 덩어리."""
    ensure(toss)
    now = time.time() if now is None else now
    with _lock:
        have, note = bool(_sessions), _note
    return {
        "open": is_open(now),
        "label": label(now),
        "exact": have,                  # False = 대략 판정 중
        "note": note or None,
        "at": int(now),
    }

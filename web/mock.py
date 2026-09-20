"""가짜 토스 — **개발 중 화면을 직접 띄워 보기 위한 것.** 실제 사용과는 무관하다.

왜 필요한가 — 토스 API 는 허용 IP 밖에서 막히고 앱 키도 개발 환경에는 없다. 그래서
화면이 실제로 어떻게 그려지는지 확인할 방법이 없었고, 칩이 3줄로 쌓이거나 차트가
탭바에 가리는 것 같은 문제를 사용자가 폰에서 보고 알려 줘야만 했다.
이 모듈이 `Toss` 와 같은 인터페이스로 그럴듯한 값을 만들어 주면 브라우저로 직접 열어
눈으로 확인할 수 있다.

    QUANT_MOCK=1 python server.py

⚠️ 진짜 시세가 아니다. 숫자의 정확성은 이걸로 검증할 수 없다 — 레이아웃 확인용이다.
"""

from __future__ import annotations

import math
import random
import time

NAMES = {
    "SPY": "SPDR S&P 500", "FNGU": "FNGU", "TQQQ": "TQQQ", "SOXL": "SOXL",
    "005930": "삼성전자", "BITU": "BITU", "NAIL": "NAIL",
    "473460": "국내 ETF A", "SOLKR": "국내 ETF B",
}


def _walk(symbol: str, n: int, start: float, step: float, seed_extra: int = 0):
    """종목마다 재현 가능한 가격 시계열. 같은 종목은 늘 같은 그림이 나온다."""
    rnd = random.Random(hash(symbol) % 10_000 + seed_extra)
    px = start
    out = []
    t0 = int(time.time()) - n * step
    for i in range(n):
        drift = math.sin(i / 18.0) * 0.004
        px = max(0.5, px * (1 + drift + rnd.gauss(0, 0.018)))
        o = px * (1 + rnd.gauss(0, 0.004))
        h = max(o, px) * (1 + abs(rnd.gauss(0, 0.005)))
        l = min(o, px) * (1 - abs(rnd.gauss(0, 0.005)))
        out.append({"t": int(t0 + i * step), "open": round(o, 2), "high": round(h, 2),
                    "low": round(l, 2), "close": round(px, 2)})
    return out


class MockToss:
    """`Toss` 와 같은 메서드만 흉내 낸다. 주문 관련은 진짜와 마찬가지로 없다."""

    def __init__(self, *_a, **_k):
        self._base = {}

    def clear_token(self) -> None:
        pass

    def accounts(self):
        return [{"accountNo": "12345677487", "accountSeq": 1, "accountType": "BROKERAGE"}]

    def holdings(self, account_seq: int):
        spec = [
            ("GDXU", "USD", 17, 140.0, 148.0), ("KORU", "USD", 90, 19.9, 19.1),
            ("BITU", "USD", 99, 14.1, 13.0), ("QPUX", "USD", 112, 12.7, 13.5),
            ("SOXL", "USD", 8, 123.0, 103.0), ("DFEN", "USD", 11, 52.3, 57.0),
            ("NAIL", "USD", 8, 28.7, 28.8), ("473460", "KRW", 2, 17660.0, 16900.0),
            # 6자리 숫자가 아닌 국내 종목 — 코드 모양만 보고 미장으로 분류하던 문제 재현용
            ("SOLKR", "KRW", 3, 17660.0, 16900.0),
        ]
        items, ke, ue, kp, up, kl, ul = [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for sym, cur, qty, last, avg in spec:
            ev, buy = last * qty, avg * qty
            pnl = ev - buy
            daily = ev * 0.004
            items.append({
                "symbol": sym, "name": NAMES.get(sym, sym),
                "marketCountry": "KR" if cur == "KRW" else "US", "currency": cur,
                "quantity": float(qty), "lastPrice": last, "avgPrice": avg,
                "purchaseAmount": buy, "evalAmount": ev,
                "pnlAmount": pnl, "pnlRate": last / avg - 1,
                "dailyPnlAmount": daily, "dailyPnlRate": 0.004,
            })
            if cur == "KRW":
                ke += ev; kp += buy; kl += pnl
            else:
                ue += ev; up += buy; ul += pnl
        return {"krwPurchase": kp, "usdPurchase": up, "krwEval": ke, "usdEval": ue,
                "krwPnl": kl, "usdPnl": ul,
                "pnlRate": (kl + ul * 1400) / max(1.0, kp + up * 1400),
                "krwDailyPnl": ke * 0.004, "usdDailyPnl": ue * 0.004,
                "dailyPnlRate": 0.004, "items": items}

    def buying_power(self, account_seq: int, currency: str) -> float:
        return 13_220_610.0 if currency == "KRW" else 1_841.52

    def usd_krw(self) -> float:
        return 1353.1

    def prices(self, symbols: list[str]):
        # 초마다 조금씩 흔들어 실시간 갱신이 화면에 반영되는지 볼 수 있게
        out = {}
        for s in symbols:
            base = self._base.setdefault(s, _walk(s, 2, 50, 86400)[-1]["close"])
            out[s] = {"price": round(base * (1 + math.sin(time.time() / 7) * 0.003), 2),
                      "at": None}
        return out

    def ohlc(self, symbol: str, interval: str = "1d", count: int = 520,
             adjusted: bool = True):
        if symbol == "AVXX":
            return []          # 시세를 못 받는 종목 — 행이 남는지 보려고 일부러 비운다
        step = 60 if interval == "1m" else 86400
        n = min(count, 390 if interval == "1m" else 500)
        bars = _walk(symbol, n, 30 if symbol != "GLD" else 400, step)
        self._base[symbol] = bars[-1]["close"]
        return bars

    def list_stocks(self, market: str, status: str = "ACTIVE"):
        if market == "KOSPI":
            return [{"symbol": "005930", "name": "가나전자", "type": "STOCK"},
                    {"symbol": "000660", "name": "다라반도체", "type": "STOCK"},
                    {"symbol": "473460", "name": "국내 ETF A", "type": "ETF"}]
        if market == "KOSDAQ":
            return [{"symbol": "SOLKR", "name": "국내 ETF B", "type": "ETF"}]
        return []

    def fills(self, account_seq: int, max_pages: int = 20):
        return [
            {"orderId": "o1", "symbol": "SOXL", "buy": True, "date": "2026-08-14",
             "quantity": 8, "price": 103.0, "currency": "USD"},
            {"orderId": "o2", "symbol": "KORU", "buy": True, "date": "2026-09-02",
             "quantity": 90, "price": 19.1, "currency": "USD"},
            {"orderId": "o3", "symbol": "GDXU", "buy": True, "date": "2026-08-28",
             "quantity": 17, "price": 148.0, "currency": "USD"},
        ]

"""토스증권 Open API 클라이언트 — **조회 전용**.

`android-toss` 의 `data/TossApi.kt` 를 파이썬으로 옮긴 것. 규칙은 그대로다.

- base: ``https://openapi.tossinvest.com``
- 인증: OAuth2 Client Credentials — ``POST /oauth2/token`` (form-urlencoded).
  refresh token 없음. client 당 유효 토큰은 1개이며 재발급하면 이전 토큰이 즉시 무효화된다.
- 계좌 컨텍스트가 필요한 API 는 ``X-Tossinvest-Account: {accountSeq}`` 헤더 필요.
- 응답 봉투: 성공 ``{"result": ...}`` / 실패 ``{"error": {code, message}}``.
  단 ``/oauth2/token`` 만 OAuth2 표준 형식이다.
- **모든 수치는 문자열(decimal)로 내려온다.** 반드시 float 로 바꿔 쓸 것.

⚠️ 주문·정정·취소는 **의도적으로 구현하지 않는다.** 이 프로그램은 계좌를 읽기만 한다.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

BASE = "https://openapi.tossinvest.com"
TIMEOUT = 10


def _epoch(iso) -> int | None:
    """``2026-09-21T09:00:00+09:00`` → epoch 초. 못 읽으면 None."""
    if not isinstance(iso, str) or not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


class TossError(Exception):
    """토스 API 호출 실패. code 는 스펙의 에러 코드(`invalid-token`, `access_denied` 등)."""

    def __init__(self, code: str, http: int, message: str):
        super().__init__(message)
        self.code = code
        self.http = http
        self.message = message


def _dec(obj: dict | None, key: str, default: float = 0.0) -> float:
    """문자열 decimal → float. null/빈값/파싱실패는 기본값."""
    if not obj:
        return default
    v = obj.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _epoch(iso: str | None) -> int | None:
    """ISO 8601 offset 문자열 → epoch 초. 실패하면 None."""
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


def _request(method: str, url: str, *, body: bytes | None = None,
             headers: dict[str, str] | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:  # 4xx·5xx 는 본문에 에러 코드가 들어 있다
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise TossError("network", 0, f"연결 실패: {e.reason}") from e


_TOKEN_MSG = {
    "invalid_client": "App Key 또는 Secret이 올바르지 않습니다",
    # 허용 IP 목록 밖에서 호출한 경우. 이 프로그램을 PC에서 돌리는 이유가 바로 이것이다.
    "access_denied": "허용되지 않은 IP입니다. 토스증권 WTS → 설정 → Open API → "
                     "허용 IP 관리에서 이 PC의 공인 IP를 등록하세요",
    "unsupported_grant_type": "지원하지 않는 인증 방식입니다",
}


class Toss:
    """앱 키 한 쌍에 대한 클라이언트. 토큰은 메모리에만 둔다(파일로 남기지 않는다)."""

    def __init__(self, app_key: str, app_secret: str):
        self._key = app_key
        self._secret = app_secret
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    # ── 인증 ──

    def _access_token(self) -> str:
        """유효한 access token. 만료 60초 전이면 재발급."""
        with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            if not self._key or not self._secret:
                raise TossError("no-credentials", 0, "App Key/Secret이 설정되지 않았습니다")

            body = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "client_id": self._key,
                "client_secret": self._secret,
            }).encode()
            status, text = _request("POST", f"{BASE}/oauth2/token", body=body, headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            })
            try:
                o = json.loads(text)
            except ValueError:
                o = {}
            if status < 200 or status >= 300:
                code = o.get("error") or f"http-{status}"
                msg = _TOKEN_MSG.get(code) or o.get("error_description") or f"인증 실패 ({code})"
                raise TossError(code, status, msg)

            self._token = o["access_token"]
            self._expires_at = time.time() + float(o.get("expires_in", 86400))
            return self._token

    def clear_token(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    # ── 공통 GET ──

    def _get(self, path: str, query: dict[str, str] | None = None,
             account_seq: int | None = None):
        """성공하면 봉투의 ``result`` 를 반환. 실패하면 TossError."""
        qs = ("?" + urllib.parse.urlencode(query)) if query else ""
        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "Accept": "application/json",
        }
        if account_seq is not None:
            headers["X-Tossinvest-Account"] = str(account_seq)

        status, text = _request("GET", f"{BASE}{path}{qs}", headers=headers)
        try:
            o = json.loads(text)
        except ValueError:
            o = {}
        if status < 200 or status >= 300:
            err = o.get("error") or {}
            code = err.get("code") or f"http-{status}"
            if status == 401:  # 토큰 만료/무효 → 다음 호출에서 재발급되게
                self.clear_token()
            msg = err.get("message") or (
                "요청 한도를 초과했습니다" if status == 429 else f"요청 실패 ({code})")
            raise TossError(code, status, msg)
        return o.get("result")

    # ── 계좌 ──

    def accounts(self) -> list[dict]:
        """``GET /api/v1/accounts`` — 정상 상태 계좌 목록.

        여기서 얻은 ``accountSeq`` 가 다른 계좌 API 의 헤더 값이다.
        """
        arr = self._get("/api/v1/accounts") or []
        return [{
            "accountNo": a.get("accountNo", ""),
            "accountSeq": int(a.get("accountSeq", 0)),
            "accountType": a.get("accountType", ""),
        } for a in arr]

    def holdings(self, account_seq: int) -> dict:
        """``GET /api/v1/holdings`` — 계좌 요약 + 보유 종목.

        금액은 전부 **거래 통화 기준**(KR=KRW, US=USD)이고, 합계도 통화별로만 온다.
        원화 환산은 호출한 쪽에서 환율을 곱해 한다.
        """
        r = self._get("/api/v1/holdings", account_seq=account_seq) or {}

        def pair(o):
            return _dec(o, "krw"), _dec(o, "usd")

        kp, up = pair(r.get("totalPurchaseAmount"))
        ke, ue = pair((r.get("marketValue") or {}).get("amount"))
        pl = r.get("profitLoss") or {}
        kl, ul = pair(pl.get("amount"))
        dpl = r.get("dailyProfitLoss") or {}
        kd, ud = pair(dpl.get("amount"))

        items = []
        for o in r.get("items") or []:
            mv = o.get("marketValue") or {}
            p = o.get("profitLoss") or {}
            d = o.get("dailyProfitLoss") or {}
            items.append({
                "symbol": o.get("symbol", ""),
                "name": o.get("name", ""),
                "marketCountry": o.get("marketCountry", ""),
                "currency": o.get("currency", "KRW"),
                "quantity": _dec(o, "quantity"),
                "lastPrice": _dec(o, "lastPrice"),
                "avgPrice": _dec(o, "averagePurchasePrice"),
                "purchaseAmount": _dec(mv, "purchaseAmount"),
                "evalAmount": _dec(mv, "amount"),
                "pnlAmount": _dec(p, "amount"),
                # 소수비율(0.1077 = 10.77%). = lastPrice/avgPrice - 1
                "pnlRate": _dec(p, "rate"),
                "dailyPnlAmount": _dec(d, "amount"),
                "dailyPnlRate": _dec(d, "rate"),
            })

        return {
            "krwPurchase": kp, "usdPurchase": up,
            "krwEval": ke, "usdEval": ue,
            "krwPnl": kl, "usdPnl": ul,
            "pnlRate": _dec(pl, "rate"),
            "krwDailyPnl": kd, "usdDailyPnl": ud,
            "dailyPnlRate": _dec(dpl, "rate"),
            "items": items,
        }

    def buying_power(self, account_seq: int, currency: str) -> float:
        """``GET /api/v1/buying-power`` — 통화별 매수 가능 금액.

        ⚠️ 엄밀한 "예수금"이 아니라 매수 가능 금액이다. 미결제 대금 등이 반영되면 실제
        예수금과 다를 수 있는데, 토스가 주는 유일한 현금 지표라 총자산 계산에 이 값을 쓴다.
        """
        r = self._get("/api/v1/buying-power", {"currency": currency}, account_seq) or {}
        return _dec(r, "cashBuyingPower")

    # ── 시세 ──

    def prices(self, symbols: list[str]) -> dict[str, dict]:
        """``GET /api/v1/prices`` — 현재가. 조회 실패 종목은 결과에서 빠진다.

        ``at`` 은 **체결 시각**이라 None 이면 이번 세션에 아직 체결이 없다는 뜻
        (= price 가 직전 종가다).
        """
        out: dict[str, dict] = {}
        for i in range(0, len(symbols), 200):
            chunk = symbols[i:i + 200]
            arr = self._get("/api/v1/prices", {"symbols": ",".join(chunk)}) or []
            for o in arr:
                v = _dec(o, "lastPrice", float("nan"))
                if v != v:
                    continue
                # ⚠️ 체결 시각은 **epoch 초로 바꿔서** 내보낸다. 토스가 주는 ISO 문자열을
                # 그대로 흘리면 받는 쪽에서 숫자로 못 읽고 0 으로 떨어져, 전 종목이
                # '이번 장 체결 없음'으로 판정된다(안드로이드 앱에서 실제로 그랬다).
                out[o.get("symbol", "")] = {"price": v, "at": _epoch(o.get("timestamp"))}
        return out

    # ── 장 운영시간 ──

    def market_sessions(self, country: str) -> list[dict]:
        """``GET /api/v1/market-calendar/{KR|US}`` — 전·당·익 3영업일의 세션 시각.

        미국 정규장은 한국 시각 22:30 에 시작해 **다음 날 05:00 에 끝난다.** 그래서
        새벽에는 '오늘'이 아니라 전 영업일 세션이 열려 있다 — 3일치를 모두 펼쳐 준다.
        """
        r = self._get(f"/api/v1/market-calendar/{country}") or {}
        names = ([("preMarket", "프리마켓"), ("regularMarket", "정규장"),
                  ("afterMarket", "애프터마켓")] if country == "KR" else
                 [("dayMarket", "데이마켓"), ("preMarket", "프리마켓"),
                  ("regularMarket", "정규장"), ("afterMarket", "애프터마켓")])
        out: list[dict] = []
        for day in ("previousBusinessDay", "today", "nextBusinessDay"):
            d = r.get(day) if isinstance(r, dict) else None
            if country == "KR" and isinstance(d, dict):
                d = d.get("integrated")          # 국내는 통합 세션 한 벌
            if not isinstance(d, dict):
                continue
            for key, name in names:
                o = d.get(key)
                if not isinstance(o, dict):
                    continue
                a, b = _epoch(o.get("startTime")), _epoch(o.get("endTime"))
                if a and b and b > a:
                    out.append({"market": country, "name": name, "start": a, "end": b})
        return out

    def list_stocks(self, market: str, status: str = "ACTIVE") -> list[dict]:
        """마켓별 전체 종목 — `GET /api/v1/stocks/all`.

        국내 종목은 코드(005930)만으로는 뭔지 알 수 없어서 이름을 얻으려고 쓴다.
        마켓당 수천 건이라 **하루 1회** 받아 캐시한다(`universe.py`).
        market: KOSPI · KOSDAQ · KR_ETC · NYSE · NASDAQ · AMEX · US_ETC
        """
        arr = self._get("/api/v1/stocks/all", {"market": market, "status": status})
        out = []
        for o in arr if isinstance(arr, list) else []:
            sym = str(o.get("symbol") or "").strip()
            if sym:
                out.append({"symbol": sym, "name": str(o.get("name") or "").strip(),
                            "type": str(o.get("securityType") or "")})
        return out

    def ohlc(self, symbol: str, interval: str = "1d", count: int = 520,
             adjusted: bool = True) -> list[dict]:
        """``GET /api/v1/candles`` — 요청당 200봉 상한이라 ``nextBefore`` 로 이어 받는다.

        응답이 최신순이므로 마지막에 **오래된→최신** 으로 뒤집는다.
        서버가 받아 주는 주기는 ``1d`` 와 ``1m`` 뿐이다(2026-09 전수 확인).
        """
        out: list[dict] = []
        before: str | None = None
        guard = 0
        while len(out) < count and guard < 10:
            q = {
                "symbol": symbol,
                "interval": interval,
                "count": str(max(1, min(200, count - len(out)))),
                "adjusted": str(adjusted).lower(),
            }
            if before:
                q["before"] = before
            r = self._get("/api/v1/candles", q) or {}
            arr = r.get("candles") or []
            if not arr:
                break
            for o in arr:
                t = _epoch(o.get("timestamp"))
                c = _dec(o, "closePrice", float("nan"))
                if t is None or c != c:
                    continue
                out.append({
                    "t": t,
                    "open": _dec(o, "openPrice", c),
                    "high": _dec(o, "highPrice", c),
                    "low": _dec(o, "lowPrice", c),
                    "close": c,
                })
            before = r.get("nextBefore") or None
            if not before:
                break
            guard += 1

        # 페이지 경계가 inclusive 라 겹칠 수 있다 → timestamp 중복 제거
        seen = set()
        uniq = []
        for c in reversed(out):
            if c["t"] in seen:
                continue
            seen.add(c["t"])
            uniq.append(c)
        return uniq

    # ── 체결 내역 ──

    def fills(self, account_seq: int, max_pages: int = 20) -> list[dict]:
        """``GET /api/v1/orders?status=CLOSED`` — 커서 페이징으로 모아 **실제 체결분만**.

        취소·거부된 주문도 부분 체결이 있었다면 그 수량은 실제 매매라 포함한다.
        """
        out: list[dict] = []
        cursor: str | None = None
        for _ in range(max_pages):
            q = {"status": "CLOSED", "limit": "100"}
            if cursor:
                q["cursor"] = cursor
            r = self._get("/api/v1/orders", q, account_seq) or {}
            for o in r.get("orders") or []:
                ex = o.get("execution") or {}
                qty = _dec(ex, "filledQuantity")
                price = _dec(ex, "averageFilledPrice", float("nan"))
                if qty <= 0 or price != price or price <= 0:
                    continue
                stamp = ex.get("filledAt") or o.get("orderedAt") or ""
                date = stamp[:10]
                if len(date) != 10:
                    continue
                out.append({
                    "orderId": o.get("orderId", ""),
                    "symbol": o.get("symbol", ""),
                    "buy": o.get("side") == "BUY",
                    "date": date,
                    "quantity": qty,
                    "price": price,
                    "currency": o.get("currency", "KRW"),
                })
            if not r.get("hasNext"):
                break
            cursor = r.get("nextCursor") or None
            if not cursor:
                break
        return out

    def usd_krw(self) -> float:
        """``GET /api/v1/exchange-rate`` — USD→KRW. 1분 주기로 갱신되는 표시용 환율."""
        r = self._get("/api/v1/exchange-rate",
                      {"baseCurrency": "USD", "quoteCurrency": "KRW"}) or {}
        return _dec(r, "rate", float("nan"))

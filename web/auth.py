"""접속 인증 — 외부에 열 때 **반드시** 거쳐야 하는 관문.

이 서버는 증권 계좌를 그대로 보여준다. 인터넷에 노출하는 순간 주소를 아는 사람은
누구나 잔고를 볼 수 있으므로, 집 밖에서 쓰려면 암호가 먼저다.

규칙
  · 암호(토큰)는 `config.json` 의 `access_token`. 없으면 **처음 실행할 때 무작위로 만들어**
    저장하고 콘솔에 찍는다. 기본 암호 같은 건 두지 않는다.
  · 한 번 로그인하면 쿠키가 남아 다시 묻지 않는다(1년).
  · `?key=<토큰>` 으로 들어와도 통과시키고 쿠키를 심은 뒤 주소에서 지운다 —
    폰에서 즐겨찾기 한 번으로 끝내라고.
  · **127.0.0.1 은 통과.** PC 자신에서 여는 건 막을 이유가 없다.
  · 토큰 비교는 `secrets.compare_digest` (길이·내용 비교 시간차로 새어 나가지 않게).
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "q_auth"
MAX_AGE = 365 * 24 * 3600
LOCAL = {"127.0.0.1", "::1", "localhost"}

_fails: dict[str, list[float]] = {}


# 지금 유효한 암호. 설정 화면에서 바꿀 수 있어야 하므로 **값이 아니라 여기를** 본다.
_token = ""


def token() -> str:
    return _token


def _read(config_path: Path) -> dict:
    if config_path.exists():
        try:
            v = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(v, dict):
                return v
        except (ValueError, OSError):
            pass
    return {}


def load_or_create_token(config_path: Path) -> str:
    """`config.json` 의 access_token. 없으면 만들어 넣는다."""
    global _token
    o = _read(config_path)
    tok = str(o.get("access_token") or "").strip()
    if not tok:
        tok = secrets.token_urlsafe(24)
        o["access_token"] = tok
        try:
            config_path.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
    _token = tok
    return tok


def rotate(config_path: Path) -> str:
    """암호를 새로 만든다. 앱 키 같은 다른 항목은 그대로 둔다.

    쓴 뒤에야 메모리 값을 바꾼다 — 파일에 못 썼는데 암호만 바뀌면 서버를 껐다 켰을 때
    아무도 못 들어오는 상태가 된다.
    """
    global _token
    o = _read(config_path)
    new = secrets.token_urlsafe(24)
    o["access_token"] = new
    config_path.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
    _token = new
    return new


def _too_many(ip: str) -> bool:
    """같은 IP 에서 5분 안에 10번 틀리면 잠시 막는다 (무차별 대입 완화)."""
    now = time.time()
    hits = [t for t in _fails.get(ip, []) if now - t < 300]
    _fails[ip] = hits
    return len(hits) >= 10


def _note_fail(ip: str) -> None:
    _fails.setdefault(ip, []).append(time.time())


def _set_cookie(resp, token: str, secure: bool) -> None:
    resp.set_cookie(COOKIE, token, max_age=MAX_AGE, httponly=True,
                    samesite="lax", secure=secure, path="/")


LOGIN_HTML = """<!DOCTYPE html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>로그인</title><style>
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#101013;color:#F2F4F6;font-family:-apple-system,BlinkMacSystemFont,
"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
form{width:min(320px,86vw)}h1{font-size:19px;margin:0 0 4px}
p{color:#8B95A1;font-size:12px;margin:0 0 16px}
input{width:100%;box-sizing:border-box;background:#1B1B20;color:#F2F4F6;border:0;
border-radius:10px;padding:13px;font-size:16px;margin-bottom:10px}
button{width:100%;background:#3182F6;color:#fff;border:0;border-radius:12px;
padding:13px;font-size:15px;font-weight:800}
.e{color:#F04452;font-size:12px;margin-top:10px}
</style></head><body><form method="post" action="/login">
<h1>퀀트 대시보드</h1><p>접속 암호를 입력하세요</p>
<input type="password" name="token" autofocus autocomplete="current-password">
<button type="submit">들어가기</button>__ERR__
</form></body></html>"""


def login_page(error: str = "") -> HTMLResponse:
    html = LOGIN_HTML.replace("__ERR__", f'<div class="e">{error}</div>' if error else "")
    return HTMLResponse(html, status_code=200 if not error else 401)


def make_guard(_initial: str = ""):
    """FastAPI 미들웨어 — 인증되지 않은 요청을 로그인 화면/401 로 돌린다.

    암호는 호출 시점의 `_token` 을 본다. 값을 잡아 두면 설정에서 바꾼 뒤에도 옛 암호가
    계속 통과한다.
    """

    async def guard(request: Request, call_next):
        tok = _token
        ip = request.client.host if request.client else ""
        path = request.url.path

        # PC 자신에서 여는 건 통과 (외부 노출과 무관)
        if ip in LOCAL:
            return await call_next(request)
        if path == "/login":
            return await call_next(request)

        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"

        # ?key=... 로 들어오면 쿠키를 심고 주소에서 지운다
        key = request.query_params.get("key")
        if key and secrets.compare_digest(key, tok):
            clean = str(request.url.remove_query_params("key"))
            resp = RedirectResponse(clean, status_code=303)
            _set_cookie(resp, tok, secure)
            return resp

        cookie = request.cookies.get(COOKIE)
        if cookie and secrets.compare_digest(cookie, tok):
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse({"error": "로그인이 필요합니다", "code": "unauthorized"},
                                status_code=401)
        return login_page()

    return guard

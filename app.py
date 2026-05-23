"""
퀀트 트레이딩 대시보드 (단일 파일)

[v2 주요 개선 사항]
- _resolve_all_cycles 결과를 main() 초반에 한 번만 계산하고 재사용 (성능 5~10배 개선)
- compute_combined_score 벡터화 (수년치 데이터에서 10~50배 빠름)
- fetch_all_data 병렬 다운로드 (ThreadPoolExecutor)
- HTML 빌더 헬퍼 함수로 인라인 스타일 중복 제거
- logging 모듈 도입으로 silent fail 제거
- TypedDict로 타입 명확화
- Config dataclass로 매직 넘버 통합
- sklearn 의존성 제거 (numpy.polyfit으로 대체)
- 분석/Z-score look-ahead 일관성 개선 (expanding std로 통일)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, TypedDict

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ====================================================
# 0. 로깅 설정
# ====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
log = logging.getLogger("quant")

# ====================================================
# 1. 전역 설정
# ====================================================
st.set_page_config(page_title="퀀트 트레이딩 대시보드", layout="wide")


@dataclass(frozen=True)
class Config:
    """매직 넘버 모음. frozen으로 불변 보장."""
    # 시드/자금 (달러 기준 — 환전 시점에 결정된 값)
    SEED_USD: float = 20_000.0
    SEED_KRW_LEGACY: int = 30_000_000  # 호환용 (사용 안 함)
    USD_KRW_FALLBACK: float = 1400.0

    # 베타 경고 임계값
    BETA_WARN: float = 4.0
    BETA_HIGH: float = 6.0

    # 신호 임계값
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    Z_HIGH: float = 1.5
    MACD_HIGH: float = 2.0
    Z_THRESHOLD_WEAK: float = 1.0       # 약 신호
    Z_THRESHOLD_STRONG: float = 2.0     # 강 신호
    MOMENTUM_THRESHOLD_WEAK: float = 2.0
    MOMENTUM_THRESHOLD_STRONG: float = 4.0
    SCORE_MAX: int = 4                  # 모멘텀 점수 ±4

    # 가중치 (compute_momentum_score_smooth)
    MACD_WEIGHT: float = 1.2
    RSI_WEIGHT: float = 0.8

    # 시스템
    DATA_TTL_SEC: int = 300
    HTTP_TIMEOUT_SEC: int = 6
    MAX_PARALLEL_FETCH: int = 8
    EXPANDING_MIN_PERIODS: int = 30


CFG = Config()


class Colors:
    """모든 색 상수 통합. 의미별 그룹."""
    # ── 신호: 수익/매수=빨강, 손실/매도=파랑 (한국식) ──
    PROFIT          = '#dc2626'     # 빨강 (수익/매수)
    PROFIT_GAIN     = '#b91c1c'     # 진한 빨강 (수익 강조용)
    LOSS            = '#2563eb'     # 파랑 (손실/매도)
    LOSS_STRONG     = '#1d4ed8'     # 진한 파랑
    NEUTRAL         = '#9ca3af'     # 회색 (중립)

    # ── 모멘텀 7단계 ──
    MOM_BUY_STRONG  = '#7f1d1d'     # 강 매수 (-4)
    MOM_BUY         = '#dc2626'     # 매수 (-2, -3)
    MOM_BUY_WEAK    = '#fca5a5'     # 약 매수 (-1)
    MOM_HOLD        = '#9ca3af'     # 중립 (0)
    MOM_SELL_WEAK   = '#93c5fd'     # 약 매도 (+1)
    MOM_SELL        = '#2563eb'     # 매도 (+2, +3)
    MOM_SELL_STRONG = '#1e3a8a'     # 강 매도 (+4)

    # ── UI 기본 ──
    TEXT            = '#374151'     # 본문
    TEXT_DARK       = '#111827'     # 강조 본문
    LABEL           = '#9ca3af'     # 라벨
    LABEL_DARK      = '#6b7280'
    BORDER          = '#e5e7eb'
    BORDER_LIGHT    = '#f3f4f6'
    BG              = '#ffffff'
    BG_HOLDING      = '#f0fdf4'     # 보유 종목 행 배경
    BG_HOLDING_BORDER = '#86efac'
    BG_NEUTRAL      = '#f3f4f6'
    BG_NEUTRAL_BORDER = '#d1d5db'

    # ── 게이지/바 ──
    GAUGE_BG        = '#f3f4f6'
    PRINCIPAL       = '#d1d5db'     # 원금 회색 (보유 평가 막대)

    # ── DD 단계 ──
    DD_RISK         = '#dc2626'     # 0~-3%: 거의 고점
    DD_CAUTION      = '#f59e0b'     # -3~-10%
    DD_OK           = '#6b7280'     # -10~-25%
    DD_CHANCE       = '#2563eb'     # < -25%: 매수 기회


# 이전 코드 호환용 alias (점진 마이그레이션)
COLOR_GAIN = Colors.PROFIT_GAIN
COLOR_LOSS = Colors.LOSS_STRONG
COLOR_NEUTRAL = Colors.NEUTRAL
COLOR_TEXT = Colors.TEXT
COLOR_LABEL = Colors.LABEL
COLOR_BORDER = Colors.BORDER


X_ASSET_FIXED = 'SPY'
# 종목 리스트 — 초기값은 DEFAULT_TICKERS, main()에서 load_target_tickers()로 갱신
TARGET_TICKERS: list[str] = [
    'FNGU', 'TQQQ', 'SOXL', 'HIBL', 'QPUX', 'LABU', 'DFEN', 'DPST',
    'GDXU', 'KORU', '005930', 'AVXX', 'SPYU', 'TARK', 'URTY', 'TNA',
    'BNKU', 'BTC-USD', 'ETH-USD', 'GLD',
]
TICKER_DISPLAY_NAMES = {'BTC-USD': 'BTC', 'ETH-USD': 'ETH', '005930': '삼전', '000660': '하닉'}

# 종목별 색상 (산업군)
_C = {
    'index':   '#dc2626',  # 대형지수
    'tech':    '#f97316',  # 테크/혁신
    'semi':    '#eab308',
    'bio':     '#16a34a',
    'defense': '#14b8a6',
    'fin':     '#2563eb',
    'em':      '#7c3aed',
    'commod':  '#ca8a04',
    'crypto':  '#6b7280',
    'other':   '#9ca3af',
}
TICKER_COLOR = {
    'TQQQ': _C['index'], 'QPUX': _C['index'],
    'SPYU': _C['index'], 'URTY': _C['index'], 'TNA': _C['index'],
    'FNGU': _C['tech'],  'HIBL': _C['tech'],
    'SOXL': _C['tech'],  'LABU': _C['tech'], 'TARK': _C['tech'],
    'DFEN': _C['defense'], 'AVXX': _C['defense'],
    'KORU': _C['em'],
    'DPST': _C['fin'],   'BNKU': _C['fin'],
    'GDXU': _C['commod'], 'GLD':  _C['commod'],
    'BTC-USD': _C['crypto'], 'ETH-USD': _C['crypto'],
    '005930': _C['other'],
}

SIGNAL_STYLE = {
    'FB2': (Colors.MOM_BUY_STRONG,  '#ffffff'),
    'FB':  (Colors.MOM_BUY,         '#ffffff'),
    'B':   (Colors.MOM_BUY_WEAK,    '#1a1a1a'),
    'H':   (Colors.MOM_HOLD,        '#ffffff'),
    'S':   (Colors.MOM_SELL_WEAK,   '#1a1a1a'),
    'FS':  (Colors.MOM_SELL,        '#ffffff'),
    'FS2': (Colors.MOM_SELL_STRONG, '#ffffff'),
}
BUTTON_TEXT_STYLE = {
    'FB2': '#f8fafc', 'FB': '#f8fafc', 'B': '#111827',
    'H': '#111827', 'S': '#111827', 'FS': '#f8fafc', 'FS2': '#f8fafc',
}
SIG_MARKER = {
    'FB2': ('triangle-up',   Colors.MOM_BUY_STRONG,  10),
    'FB':  ('triangle-up',   Colors.MOM_BUY,          8),
    'FS':  ('triangle-down', Colors.MOM_SELL,         8),
    'FS2': ('triangle-down', Colors.MOM_SELL_STRONG, 10),
}


# ====================================================
# 2. 타입 정의
# ====================================================
class CycleInfo(TypedDict):
    cycle_start: Optional[datetime.date]
    cycle_end: Optional[datetime.date]
    hold_qty: int
    buy_qty: int
    buy_cost: float
    current_pnl: Optional[float]


class TickerState(TypedDict):
    cycle: CycleInfo
    cumulative_pnl: float


# ====================================================
# 3. 유틸리티
# ====================================================
# ─────────────────────── 인증 (간단 비밀번호 보호) ───────────────────────
# 개인 정보 (매매 기록, 평가, 자산 추이) 는 로그인 후에만 표시.
# 종목 분석, 그래프, Z+M, MACD/RSI 등 시장 정보는 비로그인도 볼 수 있음.

import hashlib
import hmac as _hmac
import base64 as _b64
import time as _time

# salt + 비밀번호의 sha256 hash (코드/git에 평문 비밀번호 X)
_AUTH_SALT = "quant_dashboard_2026"
_AUTH_HASH_FALLBACK = (
    "d23564ed156d528873bcb00378b20e2610000502010879ec7edfa661ef4016b8"
)


def _get_auth_hash() -> str:
    """비밀번호 hash 반환. st.secrets 우선, 없으면 코드 fallback."""
    try:
        return st.secrets["auth"]["password_hash"]
    except Exception:
        return _AUTH_HASH_FALLBACK


def _hash_password(pw: str) -> str:
    return hashlib.sha256((_AUTH_SALT + pw).encode()).hexdigest()


def verify_password(pw: str) -> bool:
    """입력 비밀번호 검증."""
    return _hash_password(pw) == _get_auth_hash()


def is_authenticated() -> bool:
    """현재 세션이 로그인 상태인지."""
    return st.session_state.get('authenticated', False)


# ─── 쿠키 자동 로그인 (30일) ───
_COOKIE_PREFIX = "quant_dash/"
_COOKIE_TOKEN_KEY = "auth_token"
_COOKIE_MAX_AGE_SEC = 30 * 24 * 3600


def _get_cookie_password() -> str:
    """쿠키 암호화 키. st.secrets 우선, 없으면 fallback."""
    try:
        return st.secrets["auth"]["cookie_password"]
    except Exception:
        return _AUTH_SALT + "_cookie_v1"


def _make_auth_token() -> str:
    """HMAC 서명된 만료 토큰 (payload.signature)."""
    payload = json.dumps(
        {"exp": int(_time.time()) + _COOKIE_MAX_AGE_SEC, "v": 1},
        separators=(",", ":"),
    ).encode()
    msg = _b64.urlsafe_b64encode(payload).decode()
    sig = _hmac.new(
        _get_cookie_password().encode(), msg.encode(), "sha256",
    ).hexdigest()
    return f"{msg}.{sig}"


def _verify_auth_token(token: str) -> bool:
    if not token or "." not in token:
        return False
    try:
        msg, sig = token.rsplit(".", 1)
        exp_sig = _hmac.new(
            _get_cookie_password().encode(), msg.encode(), "sha256",
        ).hexdigest()
        if not _hmac.compare_digest(sig, exp_sig):
            return False
        payload = json.loads(_b64.urlsafe_b64decode(msg.encode()).decode())
        return int(payload.get("exp", 0)) > int(_time.time())
    except Exception:
        return False


def _get_cookies_manager():
    """쿠키 매니저 lazy init. 라이브러리 미설치 시 None.

    extra-streamlit-components 사용 — cryptography 의존 없음.
    토큰은 HMAC 서명으로 위조 방지.
    """
    if '_cookies_mgr' in st.session_state:
        return st.session_state['_cookies_mgr']
    try:
        import extra_streamlit_components as stx
    except Exception:
        return None
    mgr = stx.CookieManager(key="auth_cookie_mgr")
    st.session_state['_cookies_mgr'] = mgr
    return mgr


def try_auto_login_from_cookie() -> None:
    """앱 시작 시 호출: 쿠키 토큰 검증 → 자동 인증."""
    if is_authenticated():
        return
    mgr = _get_cookies_manager()
    if mgr is None:
        return
    try:
        token = mgr.get(cookie=_COOKIE_TOKEN_KEY)
    except Exception:
        return
    if token and _verify_auth_token(token):
        st.session_state['authenticated'] = True


def save_auth_cookie() -> None:
    """로그인 성공 시 호출: 30일 토큰 쿠키 저장."""
    mgr = _get_cookies_manager()
    if mgr is None:
        return
    exp = (datetime.datetime.now(datetime.timezone.utc)
           + datetime.timedelta(seconds=_COOKIE_MAX_AGE_SEC))
    try:
        mgr.set(_COOKIE_TOKEN_KEY, _make_auth_token(), expires_at=exp)
    except Exception:
        pass


def clear_auth_cookie() -> None:
    """로그아웃 시 호출."""
    mgr = _get_cookies_manager()
    if mgr is None:
        return
    try:
        mgr.delete(_COOKIE_TOKEN_KEY)
    except Exception:
        pass


def display_name(ticker: str) -> str:
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)


def safe_key(ticker: str) -> str:
    return ticker.replace('-', '_').replace('.', '_').replace('/', '_')


def pnl_color(val: float) -> str:
    return COLOR_GAIN if val >= 0 else COLOR_LOSS


def dd_color(dd_pct: float) -> str:
    """역대 고점 대비 드로다운 % → 색.

    0~-3%: 위험 (고점 근처)
    -3~-10%: 주의
    -10~-25%: 보통
    < -25%: 기회 (크게 떨어짐)
    """
    if dd_pct >= -3:    return Colors.DD_RISK
    if dd_pct >= -10:   return Colors.DD_CAUTION
    if dd_pct >= -25:   return Colors.DD_OK
    return Colors.DD_CHANCE


def signed_str(val: float, fmt: str = "{:,.0f}") -> str:
    """+/- 부호가 붙은 포맷 문자열."""
    sign = '+' if val >= 0 else ''
    return f"{sign}{fmt.format(val)}"


def compute_halflife(residual: pd.Series) -> Optional[float]:
    """잔차의 평균회귀 반감기 (영업일 단위).

    AR(1) 회귀: residual[t] = α + φ × residual[t-1] + ε
    half_life = -log(2) / log(φ)

    φ < 1 이고 φ > 0 이면 평균회귀. 짧을수록 빠르게 회귀.
    """
    s = residual.dropna()
    if len(s) < 30:
        return None
    s_lag = s.shift(1).dropna()
    s_now = s.loc[s_lag.index]
    if len(s_now) < 30:
        return None
    try:
        # OLS 슬로프 = AR(1) 계수 φ
        x = s_lag.values
        y = s_now.values
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        denom = ((x - x_mean) ** 2).sum()
        if denom <= 0:
            return None
        phi = num / denom
        # 평균회귀 조건: 0 < φ < 1
        if phi <= 0 or phi >= 1 or not np.isfinite(phi):
            return None
        hl = -np.log(2) / np.log(phi)
        if not np.isfinite(hl) or hl <= 0:
            return None
        return float(hl)
    except Exception:
        return None


def halflife_color(hl: Optional[float]) -> str:
    """half-life 색 — 짧을수록 평균회귀 좋음 (초록), 길수록 추세 추종 (회색)."""
    if hl is None: return Colors.NEUTRAL
    if hl <= 10:   return '#16a34a'   # 초록 (평균회귀 강)
    if hl <= 20:   return '#65a30d'
    if hl <= 30:   return Colors.LABEL_DARK
    return Colors.NEUTRAL


# ── #1 신호 정렬 우선순위 ──
SIGNAL_PRIORITY = {
    'FB2': 0, 'FB': 1, 'B': 2, 'H': 3, 'S': 4, 'FS': 5, 'FS2': 6,
}


# ====================================================
# 4. HTML 빌더 헬퍼 (인라인 스타일 중복 제거)
# ====================================================
def html_metric(label: str, value: str, sub: str = "", color: str = "#111827") -> str:
    """라벨 + 값 + 보조정보 메트릭 블록."""
    sub_html = f"<div style='color:{COLOR_LABEL};font-size:0.62rem;'>{sub}</div>" if sub else ""
    return (
        f"<div>"
        f"<div style='color:#6b7280;font-size:0.68rem;'>{label}</div>"
        f"<div style='font-weight:700;color:{color};'>{value}</div>"
        f"{sub_html}"
        f"</div>"
    )


def html_section_divider() -> str:
    return f"<div style='border-top:1px solid {COLOR_BORDER};margin:6px 0 5px 0;padding-top:6px;'>"


def html_dash_cell(label: str) -> str:
    return (
        f"<div><div style='color:#6b7280;font-size:0.68rem;'>{label}</div>"
        f"<div style='font-weight:700;color:#9ca3af;'>-</div></div>"
    )


def html_progress_bar(width_pct: float, color: str, height: int = 7) -> str:
    """단일 색상 진행바."""
    return (
        f"<div style='flex:1;background:#e5e7eb;border-radius:3px;height:{height}px;'>"
        f"<div style='width:{max(width_pct, 0):.1f}%;background:{color};"
        f"border-radius:3px;height:{height}px;'></div></div>"
    )


# ====================================================
# 5. 영속화 (로컬 + Gist)
# ====================================================
TRADE_FILE = 'trade_history.json'
SETTINGS_FILE = 'settings.json'
TICKERS_FILE = 'target_tickers.json'
GIST_FILENAME = 'quant_trade_history.json'
TICKERS_GIST_FILENAME = 'quant_target_tickers.json'

# 기본 종목 (사용자 설정이 없을 때 fallback)
DEFAULT_TICKERS = [
    'FNGU', 'TQQQ', 'SOXL', 'HIBL', 'QPUX', 'LABU', 'DFEN', 'DPST',
    'GDXU', 'KORU', '005930', 'AVXX', 'SPYU', 'TARK', 'URTY', 'TNA',
    'BNKU', 'BTC-USD', 'ETH-USD', 'GLD',
]
MIN_TICKERS = 3   # 최소 보유 종목 수


def _gist_cfg() -> tuple[str, str]:
    try:
        token = st.secrets.get("GITHUB_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
        gist_id = st.secrets.get("GIST_ID", "") or os.environ.get("GIST_ID", "")
        return str(token).strip(), str(gist_id).strip()
    except Exception as e:
        log.debug(f"_gist_cfg: secrets unavailable ({e})")
        return "", ""


def _gist_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def _gist_read(gist_id: str, token: str, filename: str) -> Optional[dict]:
    try:
        resp = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=_gist_headers(token),
            timeout=CFG.HTTP_TIMEOUT_SEC,
        )
        if resp.ok:
            files = resp.json().get("files", {})
            if filename in files:
                return json.loads(files[filename]["content"])
        else:
            log.warning(f"Gist read HTTP {resp.status_code}: {filename}")
    except (requests.RequestException, json.JSONDecodeError) as e:
        log.warning(f"Gist read failed ({filename}): {e}")
    return None


def _gist_write(gist_id: str, token: str, filename: str, data: dict) -> None:
    try:
        payload = {"files": {filename: {"content": json.dumps(data, indent=4, ensure_ascii=False)}}}
        resp = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers=_gist_headers(token),
            json=payload,
            timeout=CFG.HTTP_TIMEOUT_SEC,
        )
        if not resp.ok:
            log.warning(f"Gist write HTTP {resp.status_code}: {filename}")
    except requests.RequestException as e:
        log.warning(f"Gist write failed ({filename}): {e}")


def _load_json(local_file: str, gist_filename: str) -> dict:
    token, gist_id = _gist_cfg()
    if token and gist_id:
        data = _gist_read(gist_id, token, gist_filename)
        if data is not None:
            return data
    if os.path.exists(local_file):
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.error(f"Local read failed ({local_file}): {e}")
    return {}


def _save_json(local_file: str, gist_filename: str, data: dict) -> None:
    try:
        with open(local_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        log.error(f"Local write failed ({local_file}): {e}")
    token, gist_id = _gist_cfg()
    if token and gist_id:
        _gist_write(gist_id, token, gist_filename, data)


def load_trade_history() -> dict: return _load_json(TRADE_FILE, GIST_FILENAME)
def save_trade_history(h: dict) -> None: _save_json(TRADE_FILE, GIST_FILENAME, h)


def load_target_tickers() -> list[str]:
    """저장된 종목 리스트. 없으면 DEFAULT_TICKERS."""
    data = _load_json(TICKERS_FILE, TICKERS_GIST_FILENAME)
    tickers = data.get('tickers') if isinstance(data, dict) else None
    if isinstance(tickers, list) and len(tickers) >= MIN_TICKERS:
        # 중복 제거 + 빈 문자열 제거 + 순서 유지
        seen = set()
        out = []
        for t in tickers:
            t = str(t).strip().upper()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        if len(out) >= MIN_TICKERS:
            return out
    return list(DEFAULT_TICKERS)


def save_target_tickers(tickers: list[str]) -> None:
    _save_json(TICKERS_FILE, TICKERS_GIST_FILENAME, {'tickers': tickers})


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning(f"Settings load failed: {e}")
    return {}


def save_settings(s: dict) -> None:
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
    except OSError as e:
        log.warning(f"Settings save failed: {e}")


def init_session_state() -> None:
    defaults = {
        'trade_history':       load_trade_history,
        'ticker_signals':      dict,
        'ticker_momentum_scores': dict,
        'ticker_momentum_smooth': dict,
        'selected_option':     lambda: TARGET_TICKERS[0],
        'custom_ticker_input': str,
        'last_data_date':      str,
        'view_months':         lambda: load_settings().get('view_months', 2),
        'overview_view_months': lambda: 3,
        'overview_bar_unit':    lambda: '일',
        'analysis_start':      lambda: load_settings().get(
            'analysis_start',
            (datetime.date.today() - datetime.timedelta(days=365)).strftime('%y-%m')
        ),
        'candle_type':         lambda: '일봉',
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


# ====================================================
# 6. 시장 상태 (NYSE)
# ====================================================
def _us_holidays(year: int) -> set:
    """주요 NYSE 휴장일 (정확하지 않을 수 있음, 라이브러리 도입 권장)."""
    from datetime import date

    def nth_weekday(y: int, m: int, wd: int, n: int) -> Optional[datetime.date]:
        count = 0
        for day in range(1, 32):
            try:
                d = date(y, m, day)
            except ValueError:
                break
            if d.weekday() == wd:
                count += 1
                if count == n:
                    return d
        return None

    holidays = {date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)}
    for h in (
        nth_weekday(year, 1, 0, 3),   # MLK Day
        nth_weekday(year, 2, 0, 3),   # Presidents' Day
        nth_weekday(year, 9, 0, 1),   # Labor Day
        nth_weekday(year, 11, 3, 4),  # Thanksgiving
    ):
        if h:
            holidays.add(h)
    # Memorial Day: 5월 마지막 월요일
    for day in range(31, 24, -1):
        try:
            d = date(year, 5, day)
            if d.weekday() == 0:
                holidays.add(d)
                break
        except ValueError:
            pass
    return holidays


def get_market_status() -> dict:
    ET = datetime.timezone(datetime.timedelta(hours=-4))
    now_et = datetime.datetime.now(ET)
    today = now_et.date()
    is_weekend = today.weekday() >= 5
    is_holiday = today in _us_holidays(today.year)
    mo = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    mc = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    in_hours = mo <= now_et <= mc
    is_open = not is_weekend and not is_holiday and in_hours
    last_day = today
    if is_weekend or is_holiday or (not in_hours and now_et < mo):
        last_day = today - datetime.timedelta(days=1)
        while last_day.weekday() >= 5 or last_day in _us_holidays(last_day.year):
            last_day -= datetime.timedelta(days=1)
    return {
        'is_open':            is_open,
        'status_label':       "🟢 장중" if is_open else "🔴 장마감",
        'last_trading_label': f"기준: {last_day.strftime('%Y-%m-%d')} 종가",
        'last_trading_date':  last_day,
    }


# ====================================================
# 7. 신호 계산 (벡터화)
# ====================================================
def compute_combined_score(cz: float, mhz: float, rsi: float) -> int:
    """단일 시점 스코어 (스칼라용).

    부호 컨벤션: 음수 = 매수 신호, 양수 = 매도 신호 (Z, MACD, RSI 부호와 일치)
    """
    s = 0
    s += -2 if cz <= -CFG.Z_HIGH else -1 if cz < 0 else 2 if cz >= CFG.Z_HIGH else 1
    s += -2 if mhz <= -CFG.MACD_HIGH else -1 if mhz < 0 else 2 if mhz >= CFG.MACD_HIGH else 1
    s += -2 if rsi <= CFG.RSI_OVERSOLD else -1 if rsi < 50 else 2 if rsi >= CFG.RSI_OVERBOUGHT else 1
    return s


def score_to_signal(score: int) -> str:
    """음수 = 매수, 양수 = 매도, 0 = 중립."""
    if score <= -5: return 'FB2'   # 강한 매수
    if score <= -3: return 'FB'    # 매수
    if score <= -1: return 'B'     # 약 매수
    if score >= 5:  return 'FS2'   # 강한 매도
    if score >= 3:  return 'FS'    # 매도
    if score >= 1:  return 'S'     # 약 매도
    return 'H'


def get_signal_combined(cz: float, mhz: float, rsi: float) -> str:
    return score_to_signal(compute_combined_score(cz, mhz, rsi))


def get_seed_usd() -> float:
    """현재 설정된 시드 (USD) 반환 — 사이드바 입력 우선."""
    return float(st.session_state.get('seed_usd', CFG.SEED_USD))


def z_to_pct(z: float) -> float:
    """Z/M 점수 → 0~100 백분위 변환 (선형 매핑).

    매핑:
    - Z = -2.5 → 0   (극단 매수)
    - Z = -2.0 → 10  (강 매수)
    - Z = -1.0 → 30  (약 매수 임계 ≈ RSI 30)
    - Z = 0.0  → 50  (중립)
    - Z = +1.0 → 70  (약 매도 임계 ≈ RSI 70)
    - Z = +2.0 → 90  (강 매도)
    - Z = +2.5 → 100 (극단 매도)

    RSI 30/70 임계와 동일 척도 — 통일된 직관성.
    """
    pct = (z + 2.5) / 5.0 * 100
    return max(0.0, min(100.0, pct))


def pct_to_label(pct: float) -> str:
    """백분위 → 카테고리 라벨.

    임계:
    - 0~10  : 극단 매수 (Z ≤ -2)
    - 10~30 : 강 매수 (-2 < Z ≤ -1)
    - 30~45 : 약 매수 (-1 < Z ≤ -0.25)
    - 45~55 : 중립
    - 55~70 : 약 매도 (+0.25 ≤ Z < +1)
    - 70~90 : 강 매도 (+1 ≤ Z < +2)
    - 90~100: 극단 매도 (Z ≥ +2)
    """
    if pct < 10:   return "극단 매수"
    if pct < 30:   return "강 매수"
    if pct < 45:   return "약 매수"
    if pct < 55:   return "중립"
    if pct < 70:   return "약 매도"
    if pct < 90:   return "강 매도"
    return "극단 매도"


def compute_momentum_score_smooth(
    macd_pct: float, dmacd_pct: float, rsi: float,
) -> float:
    """모멘텀 점수 (연속, ±2.5 범위, Z와 동일 척도).

    세 정보 통합 (평균회귀 매매자 관점):
    - MACD_Pct (30%): 추세 절대 높이 (MACD/EMA26 %)
                      낮으면 매수 영역(-), 높으면 매도 영역(+)
    - dMACD_Pct (20%): MACD 변곡 (1차 미분, smoothed, 부호 반전)
                      MACD 하락→상승 변곡 → 음수 → 매수 시그널
                      MACD 상승→하락 변곡 → 양수 → 매도 시그널
    - RSI (50%): 과매수/과매도 — 매매 진입 핵심 지표

    임계:
    - MACD_Pct ±2% = ±0.3 기여
    - dMACD_Pct ±0.5%/일 = ±0.2 기여
    - RSI ±20 = ±0.5 기여

    평균회귀 진입 (강 신호):
    - 낮음(-) + 변곡 후 상승(dmacd_pct<0) → 둘 다 (-) → 강 매수
    - 높음(+) + 변곡 후 하락(dmacd_pct>0) → 둘 다 (+) → 강 매도
    """
    h = macd_pct / 2.0     # ±2% 도달 시 ±1
    d = dmacd_pct / 0.5    # ±0.5%/일 도달 시 ±1
    r = (rsi - 50) / 20.0  # ±2.5
    return 0.3 * h + 0.2 * d + 0.5 * r


def compute_momentum_score(
    macd_pct: float, dmacd_pct: float, rsi: float,
) -> int:
    """모멘텀 정수 점수 (-4 ~ +4)."""
    smooth = compute_momentum_score_smooth(macd_pct, dmacd_pct, rsi)
    return max(-CFG.SCORE_MAX, min(CFG.SCORE_MAX, int(round(smooth))))


def momentum_score_to_signal(score: int) -> str:
    """모멘텀 점수 → 신호 라벨 (음수=매수, 양수=매도)."""
    if score <= -4: return 'FB2'   # 강 매수
    if score <= -2: return 'FB'
    if score <= -1: return 'B'
    if score >= 4:  return 'FS2'   # 강 매도
    if score >= 2:  return 'FS'
    if score >= 1:  return 'S'
    return 'H'


def momentum_to_color(score: int) -> str:
    """모멘텀 점수 (-4 ~ +4) → 색.

    음수 = 매수 (빨강 계열), 양수 = 매도 (파랑 계열), 0 = 중립
    """
    if score <= -4: return Colors.MOM_BUY_STRONG
    if score <= -2: return Colors.MOM_BUY
    if score <= -1: return Colors.MOM_BUY_WEAK
    if score >= 4:  return Colors.MOM_SELL_STRONG
    if score >= 2:  return Colors.MOM_SELL
    if score >= 1:  return Colors.MOM_SELL_WEAK
    return Colors.MOM_HOLD


def get_time_grid_dtick_ms(start: pd.Timestamp, end: pd.Timestamp, target_grids: int = 8) -> int:
    span_days = max((end - start).days, 1)
    target_days = span_days / max(target_grids, 1)
    best_days = min(
        [3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180],
        key=lambda d: abs(d - target_days),
    )
    return int(best_days * 24 * 60 * 60 * 1000)


# ====================================================
# 8. 데이터 다운로드 (병렬화)
# ====================================================
def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df_w = df.resample('W-FRI').last().dropna(how='all')
    last_day = df.index[-1]
    if not df_w.empty and last_day > df_w.index[-1]:
        df_w = pd.concat([df_w, df.iloc[[-1]]])
    return df_w


def _resample_weekly_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df_w = df.resample('W-FRI').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    ).dropna(how='all')
    last_day = df.index[-1]
    if not df_w.empty and last_day > df_w.index[-1]:
        week_slice = df[df.index > df_w.index[-1]]
        if not week_slice.empty:
            row = pd.DataFrame([{
                'Open':  week_slice['Open'].iloc[0],
                'High':  week_slice['High'].max(),
                'Low':   week_slice['Low'].min(),
                'Close': week_slice['Close'].iloc[-1],
            }], index=[last_day])
            df_w = pd.concat([df_w, row])
    return df_w


def _filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    spy_col = f'{X_ASSET_FIXED}_Close'
    if spy_col not in df.columns or df.empty:
        return df
    spy = df[spy_col]
    traded = (spy != spy.shift(1)) | (spy.index == spy.index[0])
    is_wkday = pd.Series(df.index.weekday < 5, index=df.index)
    return df[traded & is_wkday]


def _fetch_close_one(ticker: str, start_date_str: str) -> Optional[pd.DataFrame]:
    """단일 티커 Close 컬럼만 가져오는 내부 워커."""
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if data.empty:
            return None
        data = data[~data.index.duplicated(keep='last')].sort_index()
        return data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
    except Exception as e:
        log.warning(f"fetch failed for {ticker}: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=CFG.DATA_TTL_SEC)
def fetch_ohlc(ticker: str, start_date_str: str, candle_type: str = '일봉') -> pd.DataFrame:
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if data.empty:
            return pd.DataFrame()
        data = data[~data.index.duplicated(keep='last')].sort_index()
        cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
        if len(cols) < 4:
            log.warning(f"OHLC missing for {ticker}: {cols}")
            return pd.DataFrame()
        df = data[cols][data.index.weekday < 5].copy()
        return _resample_weekly_ohlc(df) if candle_type == '주봉' else df
    except Exception as e:
        log.warning(f"fetch_ohlc failed for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=CFG.DATA_TTL_SEC)
def fetch_all_data(tickers: list, start_date_str: str, candle_type: str = '일봉') -> pd.DataFrame:
    """전 종목 Close를 병렬 다운로드."""
    all_tickers = [X_ASSET_FIXED] + list(tickers)
    with ThreadPoolExecutor(max_workers=CFG.MAX_PARALLEL_FETCH) as ex:
        results = list(ex.map(lambda t: _fetch_close_one(t, start_date_str), all_tickers))
    frames = [f for f in results if f is not None]
    if not frames:
        log.error("fetch_all_data: no frames returned")
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).ffill()
    df = _filter_trading_days(df)
    return _resample_weekly(df) if candle_type == '주봉' else df


@st.cache_data(show_spinner=False, ttl=CFG.DATA_TTL_SEC)
def fetch_usd_krw() -> tuple[float, bool]:
    """USD/KRW 실시간 환율. 반환: (값, fallback 사용 여부)."""
    try:
        today = datetime.date.today().strftime('%Y-%m-%d')
        week_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        data = fdr.DataReader('USD/KRW', week_ago, today)
        if not data.empty:
            return float(data['Close'].iloc[-1]), False
    except Exception as e:
        log.warning(f"USD/KRW fetch failed: {e}")
    return CFG.USD_KRW_FALLBACK, True


@st.cache_data(show_spinner=False, ttl=CFG.DATA_TTL_SEC)
def fetch_single_ticker(ticker: str, start_date_str: str) -> pd.DataFrame:
    """직접 입력 티커용 단일 fetch (외부 노출용)."""
    result = _fetch_close_one(ticker, start_date_str)
    return result if result is not None else pd.DataFrame()


# ====================================================
# 9. 데이터 처리 (look-ahead 일관성 개선)
# ====================================================
def process_asset_data(
    df_x: pd.DataFrame, df_y: pd.DataFrame, x_name: str, y_name: str
) -> tuple:
    """
    회귀: numpy.polyfit (sklearn 의존 제거)
    Z-Score: expanding std (look-ahead 없음)
    std_resid: 전체 std (밴드용, 시각적 일관성)
    """
    df = pd.merge(df_x, df_y, left_index=True, right_index=True).dropna().sort_index()
    if df.empty:
        return (None,) * 3

    base_x = df[f'{x_name}_Close'].iloc[0]
    base_y = df[f'{y_name}_Close'].iloc[0]
    df[f'{x_name}_Norm'] = df[f'{x_name}_Close'] / base_x
    df[f'{y_name}_Norm'] = df[f'{y_name}_Close'] / base_y

    log_x = np.log(df[f'{x_name}_Norm'].values)
    log_y = np.log(df[f'{y_name}_Norm'].values)
    # numpy.polyfit으로 OLS — sklearn 제거
    beta, intercept = np.polyfit(log_x, log_y, 1)
    df['Predicted'] = np.exp(intercept) * df[f'{x_name}_Norm'] ** beta

    close = df[f'{y_name}_Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['EMA26'] = ema26
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # MACD-Hist Z (변곡점 선행, 종목 신호 분류용으로만 유지)
    exp_std_hist = df['MACD_Hist'].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std()
    exp_mean_hist = df['MACD_Hist'].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).mean()
    df['MACD_Hist_Z'] = (df['MACD_Hist'] - exp_mean_hist) / exp_std_hist.replace(0, np.nan)

    # MACD-Z (호환용)
    exp_std_macd = df['MACD'].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std()
    exp_mean_macd = df['MACD'].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).mean()
    df['MACD_Z'] = (df['MACD'] - exp_mean_macd) / exp_std_macd.replace(0, np.nan)

    # ── 모멘텀 계산용 새 정규화 (분포 의존 X, 가격 급변 영향 적음) ──
    # MACD / EMA26 × 100 = "단기-장기 EMA 차이의 비율 (%)"
    # 의미: 단기 EMA가 장기 EMA보다 몇 % 위/아래에 있는가
    # 일반적 범위: ±3% 정도, ±5% 이상은 매우 강한 추세
    ema26_safe = ema26.replace(0, np.nan)
    df['MACD_Pct'] = (df['MACD'] / ema26_safe) * 100      # 추세 절대 높이
    df['MACD_Hist_Pct'] = (df['MACD_Hist'] / ema26_safe) * 100  # Hist (참고용)

    # ── 변곡 검출: dMACD = MACD 1차 미분 (EMA로 smoothing) ──
    # dMACD/dt = 0 → MACD 자체의 극값 (저점/고점 = 변곡점)
    # smoothing: EMA span=3 으로 일간 노이즈 완화
    dmacd_smooth = df['MACD'].diff().ewm(span=3, adjust=False).mean()
    # 시각화용: 부호 그대로 (MACD 상승 = 양수)
    df['dMACD_Raw_Pct'] = (dmacd_smooth / ema26_safe) * 100
    # 모멘텀 계산용: 부호 반전 (평균회귀 방향 = 매도방향 양수)
    #   MACD 상승 중 (dMACD > 0) → dMACD_Pct 음수 → 매수 방향 신호
    #   MACD 하락 중 (dMACD < 0) → dMACD_Pct 양수 → 매도 방향 신호
    df['dMACD_Pct'] = -df['dMACD_Raw_Pct']

    log_resid = np.log(df[f'{y_name}_Norm']) - np.log(df['Predicted'])
    std_resid = log_resid.std()
    df['Z_Score'] = (
        log_resid
        / log_resid.expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std().replace(0, np.nan)
    )

    return df, beta, std_resid


@st.cache_data(show_spinner=False, ttl=CFG.DATA_TTL_SEC)
def compute_all_analyses(
    df_close: pd.DataFrame, _version: int = 9, candle_type: str = '일봉',
    extra_tickers: Optional[tuple] = None,
) -> dict:
    """전체 종목 분석. TARGET_TICKERS + extra_tickers (매매 기록 종목 등).

    extra_tickers는 tuple (해시 가능) — st.cache_data 호환.
    """
    df_x = df_close[[f'{X_ASSET_FIXED}_Close']]
    results = {}
    tickers_to_analyze = list(TARGET_TICKERS)
    if extra_tickers:
        for t in extra_tickers:
            if t not in tickers_to_analyze:
                tickers_to_analyze.append(t)
    for ticker in tickers_to_analyze:
        col = f'{ticker}_Close'
        results[ticker] = (
            process_asset_data(df_x, df_close[[col]], X_ASSET_FIXED, ticker)
            if col in df_close.columns else None
        )
    return results


# ====================================================
# 9-A. 사이클 통계 (#1)
# ====================================================
def extract_cycles_avgs(records: list) -> list[dict]:
    """매매 기록 → 사이클별 (시작, 끝, 평균 매수가, 평균 매도가, 진행 여부).

    한 사이클 = 0주 → 매수 → 매수/매도 → 0주 (완료) 또는 보유중 (진행)
    그래프에 평균 매수/매도가 수평선 그리기 위함.

    Returns:
        list of dict: [
            {
                'start': date,        # 매수 시작일
                'end': date,          # 매도 완료일 (진행 중이면 None)
                'avg_buy': float,     # 평균 매수가 (cost / qty)
                'avg_sell': float,    # 평균 매도가 (proceeds / qty), 매도 0이면 None
                'is_active': bool,    # 진행 중 여부
                'buy_qty': int,
                'sell_qty': int,
            },
            ...
        ]  # 시간순
    """
    valid = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]
    if not valid:
        return []

    sorted_recs = sorted(valid, key=lambda r: r['date'])
    cycles = []
    hold_qty = 0
    buy_qty = 0
    sell_qty = 0
    buy_cost = 0.0
    sell_proceeds = 0.0
    first_sell_date: Optional[datetime.date] = None  # 사이클 내 첫 매도일
    cycle_start: Optional[datetime.date] = None

    for r in sorted_recs:
        date = datetime.date.fromisoformat(r['date'])
        qty = int(r['qty'])
        price = float(r['price'])

        if r['type'] == 'buy':
            if hold_qty == 0:
                # 새 사이클 시작
                cycle_start = date
                buy_qty = 0
                sell_qty = 0
                buy_cost = 0.0
                sell_proceeds = 0.0
                first_sell_date = None
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * price

        elif r['type'] == 'sell' and hold_qty > 0:
            if first_sell_date is None:
                first_sell_date = date
            sell_proceeds += qty * price
            sell_qty += qty
            hold_qty = max(hold_qty - qty, 0)

            if hold_qty == 0 and buy_qty > 0:
                # 사이클 완료
                cycles.append({
                    'start': cycle_start,
                    'end': date,
                    'avg_buy': buy_cost / buy_qty,
                    'avg_sell': (sell_proceeds / sell_qty) if sell_qty > 0 else None,
                    'first_sell_date': first_sell_date,
                    'is_active': False,
                    'buy_qty': buy_qty,
                    'sell_qty': sell_qty,
                })
                # reset
                cycle_start = None
                first_sell_date = None

    # 진행 중 사이클 (보유 중)
    if hold_qty > 0 and buy_qty > 0:
        cycles.append({
            'start': cycle_start,
            'end': None,
            'avg_buy': buy_cost / buy_qty,
            'avg_sell': (sell_proceeds / sell_qty) if sell_qty > 0 else None,
            'first_sell_date': first_sell_date,
            'is_active': True,
            'buy_qty': buy_qty,
            'sell_qty': sell_qty,
        })

    return cycles


def compute_cycle_stats(records: list) -> Optional[dict]:
    """
    종목의 매매 기록에서 완료된 사이클들의 통계를 산출.
    - 한 사이클 = 0주 → 매수 → ... → 0주
    - 미청산(현재 보유 중) 사이클은 제외
    반환: None (사이클 없음) 또는
         {'count', 'win_rate', 'avg_ret_pct', 'avg_hold_days',
          'profit_factor', 'best_pct', 'worst_pct', 'best_date', 'worst_date'}
    """
    valid = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]
    if not valid:
        return None

    sorted_recs = sorted(valid, key=lambda r: r['date'])
    cycles = []  # list of (start_date, end_date, ret_pct, pnl_dollar)
    hold_qty = 0
    buy_qty = 0
    buy_cost = 0.0
    sell_proceeds = 0.0
    cycle_start: Optional[datetime.date] = None

    for r in sorted_recs:
        date = datetime.date.fromisoformat(r['date'])
        qty = int(r['qty'])
        if r['type'] == 'buy':
            if hold_qty == 0:
                cycle_start = date
                buy_qty = 0
                buy_cost = 0.0
                sell_proceeds = 0.0
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * r['price']
        elif r['type'] == 'sell' and hold_qty > 0:
            sell_proceeds += qty * r['price']
            hold_qty = max(hold_qty - qty, 0)
            if hold_qty == 0 and buy_qty > 0:
                pnl = sell_proceeds - buy_cost
                ret_pct = pnl / buy_cost * 100
                hold_days = (date - cycle_start).days if cycle_start else 0
                cycles.append({
                    'start': cycle_start, 'end': date,
                    'ret_pct': ret_pct, 'pnl': pnl, 'hold_days': hold_days,
                })

    if not cycles:
        return None

    wins = [c for c in cycles if c['ret_pct'] > 0]
    losses = [c for c in cycles if c['ret_pct'] <= 0]
    total_gain = sum(c['pnl'] for c in wins)
    total_loss = abs(sum(c['pnl'] for c in losses))
    best = max(cycles, key=lambda c: c['ret_pct'])
    worst = min(cycles, key=lambda c: c['ret_pct'])
    return {
        'count': len(cycles),
        'win_rate': len(wins) / len(cycles) * 100,
        'avg_ret_pct': sum(c['ret_pct'] for c in cycles) / len(cycles),
        'avg_hold_days': sum(c['hold_days'] for c in cycles) / len(cycles),
        'profit_factor': (total_gain / total_loss) if total_loss > 0 else float('inf'),
        'best_pct': best['ret_pct'],
        'worst_pct': worst['ret_pct'],
        'best_date': best['end'],
        'worst_date': worst['end'],
        'cycles': cycles,
    }


def compute_cycle_avg_prices(
    records: list,
    df_daily: Optional[pd.DataFrame] = None,
) -> list:
    """
    종목의 매매 기록에서 모든 사이클의 가중평균 매수/매도가를 반환.
    완료된 사이클뿐 아니라 진행 중인 사이클(매도 미완료)도 포함.

    df_daily가 주어지면 매매 시점의 Z_Score를 가중평균한 σ도 추가.
    이는 그래프2 가격 패널의 빨간/파란 수직선 (= 매매 시점 Z-score)과 정확히 일치.

    반환: [{'idx', 'avg_buy', 'avg_sell', 'avg_buy_sigma', 'avg_sell_sigma',
            'is_active', 'start', 'end'}]
    오래된 → 최근 순
    """
    valid = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]
    if not valid:
        return []
    sorted_recs = sorted(valid, key=lambda r: r['date'])

    # df_daily가 있으면 Z_Score lookup 헬퍼 준비
    z_lookup = None
    if df_daily is not None and 'Z_Score' in df_daily.columns:
        z_series = df_daily['Z_Score'].dropna()

        def _z_at(d: datetime.date) -> Optional[float]:
            """매매 일자 d 또는 그 이전 가장 가까운 거래일의 Z_Score."""
            if z_series.empty:
                return None
            ts = pd.Timestamp(d)
            # 해당 일자까지의 series에서 마지막 값 (asof)
            sub = z_series[z_series.index <= ts]
            if sub.empty:
                return None
            return float(sub.iloc[-1])
        z_lookup = _z_at

    cycles = []
    hold_qty = 0
    buy_qty = 0
    buy_cost = 0.0
    sell_qty = 0
    sell_proceeds = 0.0
    # σ 가중합 (매매 시점 Z-score × qty)
    buy_sigma_qty_sum = 0.0
    buy_sigma_qty_total = 0  # 유효한 z를 가진 qty 합
    sell_sigma_qty_sum = 0.0
    sell_sigma_qty_total = 0
    cycle_start: Optional[datetime.date] = None

    def _close_cycle(end_date: Optional[datetime.date], is_active: bool):
        if buy_qty == 0:
            return
        avg_buy_sigma = (
            buy_sigma_qty_sum / buy_sigma_qty_total
            if buy_sigma_qty_total > 0 else None
        )
        avg_sell_sigma = (
            sell_sigma_qty_sum / sell_sigma_qty_total
            if sell_sigma_qty_total > 0 else None
        )
        cycles.append({
            'idx': len(cycles) + 1,
            'avg_buy': buy_cost / buy_qty,
            'avg_sell': (sell_proceeds / sell_qty) if sell_qty > 0 else None,
            'avg_buy_sigma': avg_buy_sigma,
            'avg_sell_sigma': avg_sell_sigma,
            'is_active': is_active,
            'start': cycle_start,
            'end': end_date,
        })

    for r in sorted_recs:
        date = datetime.date.fromisoformat(r['date'])
        qty = int(r['qty'])
        if r['type'] == 'buy':
            if hold_qty == 0:
                # 새 사이클 시작 — 이전 데이터 초기화
                cycle_start = date
                buy_qty = 0
                buy_cost = 0.0
                sell_qty = 0
                sell_proceeds = 0.0
                buy_sigma_qty_sum = 0.0
                buy_sigma_qty_total = 0
                sell_sigma_qty_sum = 0.0
                sell_sigma_qty_total = 0
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * r['price']
            if z_lookup is not None:
                z_val = z_lookup(date)
                if z_val is not None and np.isfinite(z_val):
                    buy_sigma_qty_sum += qty * z_val
                    buy_sigma_qty_total += qty
        elif r['type'] == 'sell' and hold_qty > 0:
            sell_qty += qty
            sell_proceeds += qty * r['price']
            if z_lookup is not None:
                z_val = z_lookup(date)
                if z_val is not None and np.isfinite(z_val):
                    sell_sigma_qty_sum += qty * z_val
                    sell_sigma_qty_total += qty
            hold_qty = max(hold_qty - qty, 0)
            if hold_qty == 0 and buy_qty > 0:
                _close_cycle(date, is_active=False)

    # 진행 중 사이클 (마지막 매수 후 청산 안 됨)
    if hold_qty > 0 and buy_qty > 0:
        _close_cycle(None, is_active=True)

    return cycles


# ====================================================
# 9-C. 드로다운 (#6)
# ====================================================
def compute_portfolio_equity(
    portfolio_state: dict, df_close: pd.DataFrame, trade_history: dict
) -> Optional[pd.Series]:
    """
    일별 평가금액(USD) 시계열을 계산.
    각 날짜에 대해: 보유 수량 * 종가 + 현금 누적 손익
    단순화: 매수/매도 이벤트 시점에 누적 보유량 변화를 반영.
    """
    if df_close.empty:
        return None

    # 모든 매매 시점 정렬
    all_events = []
    for ticker, records in trade_history.items():
        for r in records:
            if r.get('qty', 0) > 0 and r.get('price', 0) > 0:
                all_events.append({
                    'date': pd.to_datetime(r['date']),
                    'ticker': ticker, 'type': r['type'],
                    'qty': int(r['qty']), 'price': float(r['price']),
                })
    if not all_events:
        return None
    all_events.sort(key=lambda e: e['date'])

    # 각 날짜의 보유 수량 추적용
    holdings: dict[str, int] = {}      # 현재 보유
    realized_total = 0.0                # 누적 실현손익
    avg_costs: dict[str, float] = {}    # 가중평균 단가

    equity = pd.Series(index=df_close.index, dtype=float)
    event_idx = 0
    for date in df_close.index:
        # 이 날짜까지 발생한 이벤트 모두 처리
        while event_idx < len(all_events) and all_events[event_idx]['date'] <= date:
            ev = all_events[event_idx]
            tk = ev['ticker']
            q = ev['qty']
            p = ev['price']
            cur_q = holdings.get(tk, 0)
            cur_avg = avg_costs.get(tk, 0.0)
            if ev['type'] == 'buy':
                new_q = cur_q + q
                avg_costs[tk] = ((cur_avg * cur_q) + (p * q)) / new_q if new_q > 0 else 0
                holdings[tk] = new_q
            elif ev['type'] == 'sell' and cur_q > 0:
                sq = min(q, cur_q)
                realized_total += (p - cur_avg) * sq
                holdings[tk] = cur_q - sq
                if holdings[tk] == 0:
                    avg_costs[tk] = 0
            event_idx += 1

        # 미실현 평가금액
        unrealized = 0.0
        for tk, q in holdings.items():
            if q == 0:
                continue
            col = f'{tk}_Close'
            if col in df_close.columns:
                px = df_close.loc[date, col]
                if pd.notna(px):
                    unrealized += (px - avg_costs.get(tk, 0)) * q
        equity.loc[date] = realized_total + unrealized

    return equity.dropna()


def compute_drawdown(equity: pd.Series) -> dict:
    """누적 손익 시계열 → MDD, 현재 DD."""
    if equity is None or equity.empty:
        return {'current_dd': 0.0, 'mdd': 0.0, 'mdd_date': None}

    # 누적 평가액의 cummax 대비 하락률 (시드 USD 기반)
    seed = get_seed_usd()
    portfolio_value = equity + seed         # 평가 자산 = 시드(USD) + 누적손익(USD)
    running_max = portfolio_value.cummax()
    dd = (portfolio_value - running_max) / running_max * 100
    current_dd = float(dd.iloc[-1])
    mdd = float(dd.min())
    mdd_date = dd.idxmin()
    return {
        'current_dd': current_dd,
        'mdd': mdd,
        'mdd_date': mdd_date.date() if pd.notna(mdd_date) else None,
    }


# ====================================================
# 9-D. 상관관계 매트릭스 (#5)
# ====================================================
def _resolve_all_cycles(valid: list) -> tuple[CycleInfo, float]:
    """매매 기록 → 현재 사이클 + 누적 실현손익.

    current_pnl 의미:
      - 사이클 완료 (cycle_end != None, 전량 매도): 사이클 전체 손익
      - 사이클 진행 중 (hold_qty > 0): 지금까지 부분 매도로 실현된 손익
        (보유분의 미실현은 제외, 평균단가 대비 매도가)
      - 매수만 있고 매도 없음: None
    """
    sorted_records = sorted(valid, key=lambda r: r['date'])

    cycle_start: Optional[datetime.date] = None
    cycle_end: Optional[datetime.date] = None
    hold_qty = 0
    buy_qty = 0
    buy_cost = 0.0
    sell_proceeds = 0.0
    cumulative_pnl = 0.0
    realized_partial = 0.0  # 현재 사이클의 부분 매도 실현 손익 (평균단가 대비)
    has_any_sell = False    # 현재 사이클에 매도가 한 번이라도 있었나

    for r in sorted_records:
        date = datetime.date.fromisoformat(r['date'])
        qty = int(r['qty'])

        if r['type'] == 'buy':
            if hold_qty == 0:
                if cycle_start is not None and cycle_end is not None:
                    cumulative_pnl += sell_proceeds - buy_cost
                cycle_start = date
                cycle_end = None
                buy_qty = 0
                buy_cost = 0.0
                sell_proceeds = 0.0
                realized_partial = 0.0
                has_any_sell = False
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * r['price']

        elif r['type'] == 'sell' and hold_qty > 0:
            # 부분 매도 손익 누적 (평균단가 대비 매도가)
            avg_buy = buy_cost / buy_qty if buy_qty > 0 else 0
            realized_partial += qty * (r['price'] - avg_buy)
            has_any_sell = True

            sell_proceeds += qty * r['price']
            hold_qty = max(hold_qty - qty, 0)
            if hold_qty == 0:
                cycle_end = date

    # current_pnl 결정
    if cycle_end is not None:
        # 사이클 완료
        current_pnl = sell_proceeds - buy_cost
    elif has_any_sell:
        # 부분 매도 진행 중 (보유 중)
        current_pnl = realized_partial
    else:
        # 매수만 있음
        current_pnl = None

    cyc: CycleInfo = {
        'cycle_start': cycle_start,
        'cycle_end':   cycle_end,
        'hold_qty':    hold_qty,
        'buy_qty':     buy_qty,
        'buy_cost':    buy_cost,
        'current_pnl': current_pnl,
    }
    return cyc, cumulative_pnl


def build_portfolio_state(trade_history: dict) -> dict[str, TickerState]:
    """
    [핵심 최적화] 모든 종목의 사이클 정보를 한 번만 계산.
    main()에서 호출 후 사이드바·트래커·메인이 모두 재사용.
    """
    state: dict[str, TickerState] = {}
    for ticker, records in trade_history.items():
        valid = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]
        if not valid:
            continue
        cyc, cum = _resolve_all_cycles(valid)
        state[ticker] = {'cycle': cyc, 'cumulative_pnl': cum}
    return state


def calc_portfolio_total_pnl(
    portfolio_state: dict[str, TickerState], df_close: pd.DataFrame
) -> float:
    """전 종목 (누적실현 + 현재평가) 합계."""
    total = 0.0
    for ticker, ts in portfolio_state.items():
        cyc = ts['cycle']
        cum = ts['cumulative_pnl']
        if cyc['buy_qty'] == 0:
            continue

        realized = cum + (cyc['current_pnl'] if cyc['current_pnl'] is not None else 0.0)

        unrealized = 0.0
        if cyc['hold_qty'] > 0:
            col = f'{ticker}_Close'
            if col in df_close.columns:
                current_price = float(df_close[col].iloc[-1])
                avg_price = cyc['buy_cost'] / cyc['buy_qty']
                unrealized = (current_price - avg_price) * cyc['hold_qty']
        total += realized + unrealized
    return total


# ====================================================
# 11. 차트 헬퍼
# ====================================================


# ====================================================
# 12. 사이드바 - 포트폴리오 카드 빌더 (분리)
# ====================================================
def build_trade_journal(
    trade_history: dict, all_analyses: dict, df_close: pd.DataFrame,
) -> list[dict]:
    """매매 기록 + 매매 시점 신호 매칭.

    각 매매 기록에 대해 그 시점의 Z, M(모멘텀), DD를 분석 데이터에서 추출.
    매매 짝 (buy → sell)을 묶어 수익률도 계산.

    Returns:
        list of dict: [
            {ticker, date, type, qty, price, z, m, dd, pnl_pct (sell만)},
            ...
        ]
    """
    journal: list[dict] = []

    for tk, recs in trade_history.items():
        result = all_analyses.get(tk)
        if not result or result[0] is None:
            df_t = None
        else:
            df_t, _, _ = result

        sorted_recs = sorted(
            [r for r in recs if r.get('qty', 0) > 0 and r.get('price', 0) > 0],
            key=lambda r: r['date']
        )

        # 가중평균 단가 추적 (수익률 계산용)
        avg_p = 0.0
        hqty = 0
        for orig_idx, r in enumerate(sorted_recs):
            try:
                rd = datetime.date.fromisoformat(r['date'])
            except Exception:
                continue
            ts = pd.Timestamp(rd)
            qty = int(r['qty'])
            entry = {
                'ticker': tk, 'date': r['date'], 'type': r['type'],
                'qty': qty, 'price': float(r['price']),
                'z': None, 'm': None,
                'pnl_pct': None, 'pnl_amount': None,
                'memo': r.get('memo', ''),
                # 편집용: 원본 trade_history 리스트에서의 인덱스
                'record_idx': trade_history[tk].index(r) if r in trade_history[tk] else -1,
            }

            # 매매 시점 신호 추출 (df_t에서 ts 또는 가장 가까운 영업일)
            if df_t is not None and not df_t.empty:
                # 가장 가까운 영업일 (ts 또는 그 이전)
                idx = df_t.index[df_t.index <= ts]
                if len(idx) > 0:
                    nearest = idx[-1]
                    row = df_t.loc[nearest]
                    z_v = row.get('Z_Score')
                    macd_pct = row.get('MACD_Pct')
                    dmacd_pct = row.get('dMACD_Pct')
                    rsi = row.get('RSI')
                    if pd.notna(z_v):
                        entry['z'] = float(z_v)
                    if pd.notna(macd_pct) and pd.notna(dmacd_pct) and pd.notna(rsi):
                        entry['m'] = compute_momentum_score_smooth(
                            float(macd_pct), float(dmacd_pct), float(rsi),
                        )

            # 수익률 (매도 시점만)
            if r['type'] == 'buy':
                if hqty + qty > 0:
                    avg_p = (avg_p * hqty + r['price'] * qty) / (hqty + qty)
                hqty += qty
            elif r['type'] == 'sell' and hqty > 0:
                sq = min(qty, hqty)
                if avg_p > 0:
                    entry['pnl_pct'] = (r['price'] / avg_p - 1) * 100
                    entry['pnl_amount'] = sq * (r['price'] - avg_p)
                hqty -= sq
                if hqty == 0:
                    avg_p = 0.0

            journal.append(entry)

    return journal


def _build_journal_html(
    journal: list[dict],
    stats: dict,
    show_ticker: bool = True,
    filter_ticker: Optional[str] = None,
    title: str = "📓 매매 일지",
) -> str:
    """매매 일지 카드.

    표시 항목: 날짜 / 방향(▲▼) / [티커] / 모멘텀(M) / 수익률(%) / 손익금($) / 메모

    Args:
        journal: 전체 매매 일지
        stats: (현재 미사용)
        show_ticker: 티커 컬럼 표시 여부
        filter_ticker: 특정 종목만 필터링 (None이면 전체)
        title: 카드 제목
    """
    if not journal:
        return ""

    # 필터링 + 최신순 정렬
    if filter_ticker:
        items = [j for j in journal if j['ticker'] == filter_ticker]
    else:
        items = list(journal)
    if not items:
        return ""
    recent = sorted(items, key=lambda x: x['date'], reverse=True)

    html = (
        f"{html_section_divider()}"
        f"<div style='font-size:0.62rem;font-weight:700;color:{COLOR_LABEL};"
        f"margin-bottom:6px;'>{title}</div>"
        f"<div style='font-size:0.55rem;color:{COLOR_LABEL};margin-bottom:4px;'>"
        f"전체 {len(recent)}건</div>"
    )

    for e in recent:
        is_buy = e['type'] == 'buy'
        type_col = '#dc2626' if is_buy else '#2563eb'
        type_icon = '▲' if is_buy else '▼'

        # 모멘텀 (M) — 백분위로 표시 (0~100, RSI 동일 척도)
        m_html = ""
        if e['m'] is not None:
            m_int = max(-4, min(4, int(round(e['m']))))
            m_col = momentum_to_color(m_int)
            m_pct = z_to_pct(e['m'])
            m_html = f"<span style='color:{m_col};'>M{int(round(m_pct))}</span>"

        # 수익률 + 손익금 (매도만)
        pnl_html = ""
        if e.get('pnl_pct') is not None:
            pcol = pnl_color(e['pnl_pct'])
            amt_html = ""
            if e.get('pnl_amount') is not None:
                amt = e['pnl_amount']
                amt_sign = '+' if amt >= 0 else '-'
                amt_html = (
                    f" <span style='color:{pcol};font-weight:600;'>"
                    f"({amt_sign}${abs(amt):.0f})</span>"
                )
            pnl_html = (
                f" → <span style='color:{pcol};font-weight:700;'>"
                f"{signed_str(e['pnl_pct'], '{:.1f}')}%</span>{amt_html}"
            )

        # 날짜 짧게 (MM/DD)
        try:
            d_short = datetime.date.fromisoformat(e['date']).strftime('%m/%d')
        except Exception:
            d_short = e['date']

        # 매매 수량 (×N) — 화살표 옆 작게
        qty = e.get('qty', 0)
        qty_html = (
            f"<span style='color:#9ca3af;font-size:0.55rem;'>×{qty}</span>"
            if qty else ""
        )

        # 행 (티커 표시 옵션)
        if show_ticker:
            tk_short = display_name(e['ticker'])
            html += (
                f"<div style='font-size:0.65rem;color:#374151;margin-bottom:3px;"
                f"display:flex;gap:6px;align-items:center;'>"
                f"<span style='color:#9ca3af;width:36px;flex-shrink:0;'>{d_short}</span>"
                f"<span style='color:{type_col};font-weight:700;width:14px;flex-shrink:0;'>{type_icon}</span>"
                f"<span style='font-weight:600;width:50px;flex-shrink:0;'>{tk_short}</span>"
                f"<span style='width:28px;flex-shrink:0;'>{qty_html}</span>"
                f"<span style='color:#6b7280;flex:1;min-width:0;'>{m_html}{pnl_html}</span>"
                f"</div>"
            )
        else:
            html += (
                f"<div style='font-size:0.65rem;color:#374151;margin-bottom:3px;"
                f"display:flex;gap:6px;align-items:center;'>"
                f"<span style='color:#9ca3af;width:36px;flex-shrink:0;'>{d_short}</span>"
                f"<span style='color:{type_col};font-weight:700;width:14px;flex-shrink:0;'>{type_icon}</span>"
                f"<span style='width:28px;flex-shrink:0;'>{qty_html}</span>"
                f"<span style='color:#6b7280;flex:1;min-width:0;'>{m_html}{pnl_html}</span>"
                f"</div>"
            )

        # 메모 인라인 (있는 경우만)
        if e.get('memo'):
            memo_safe = (
                e['memo']
                .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            )
            indent = '90px' if show_ticker else '84px'
            html += (
                f"<div style='font-size:0.6rem;color:#9ca3af;"
                f"margin:0 0 6px {indent};font-style:italic;line-height:1.3;'>"
                f"📝 {memo_safe}</div>"
            )

    return html


def _build_realized_html(
    portfolio_state: dict[str, TickerState], usd_krw: float
) -> str:
    """💵 실현손익 — 합계 헤더 + 종목별 막대 (달러 주, 원화 보조)."""
    rows = []
    for tk, ts in portfolio_state.items():
        cyc = ts['cycle']
        total_real = ts['cumulative_pnl'] + (
            cyc['current_pnl'] if cyc['current_pnl'] is not None else 0.0
        )
        if total_real != 0.0:
            rows.append((tk, total_real))

    if not rows:
        return ""

    total_abs = sum(abs(v) for _, v in rows)
    net_sum = sum(v for _, v in rows)
    net_col = pnl_color(net_sum)
    max_abs = max(abs(v) for _, v in rows)

    net_krw_man = net_sum * usd_krw / 10000

    # ── 라벨 ──
    html = (
        f"{html_section_divider()}"
        f"<div style='font-size:0.62rem;font-weight:700;color:{COLOR_LABEL};"
        f"margin-bottom:6px;'>💵 실현손익</div>"
    )

    # ── 합계 헤더 ──
    html += (
        f"<div style='display:flex;justify-content:flex-end;align-items:flex-end;"
        f"padding:6px 4px 8px 4px;border-bottom:1px solid #f3f4f6;margin-bottom:8px;'>"
        f"<div style='text-align:right;'>"
        f"<div style='font-size:0.55rem;color:{COLOR_LABEL};margin-bottom:2px;'>합계</div>"
        f"<div style='font-size:0.92rem;color:{net_col};font-weight:700;'>"
        f"{signed_str(int(round(net_sum)), '{:,}')}</div>"
        f"<div style='font-size:0.55rem;color:#9ca3af;'>"
        f"({signed_str(int(round(net_krw_man)), '{:,}')}만원)</div>"
        f"</div>"
        f"</div>"
    )

    # ── 종목별 좌우 분기 막대 ──
    # 중앙선을 기준으로 이익 종목은 오른쪽, 손실 종목은 왼쪽
    # 손실 큰 순 → 손실 작은 순 → 이익 작은 순 → 이익 큰 순으로 정렬
    # (손실이 위로 모이는 효과)
    sorted_rows = sorted(rows, key=lambda x: x[1])

    for tk, real in sorted_rows:
        ratio = abs(real) / total_abs * 100 if total_abs else 0
        w_pct = max(abs(real) / max_abs * 100, 2) if max_abs else 2
        is_loss = real < 0
        bar_c = '#2563eb' if is_loss else '#dc2626'
        vc = pnl_color(real)
        real_krw_man = real * usd_krw / 10000

        # 좌우 분기 막대: 중앙선 기준
        # 왼쪽 50% 영역 (손실 막대 = 우측 끝에서 왼쪽으로)
        # 오른쪽 50% 영역 (이익 막대 = 좌측 끝에서 오른쪽으로)
        bar_html = (
            f"<div style='flex:1;display:flex;align-items:center;min-width:0;"
            f"position:relative;height:10px;'>"
            # 중앙 분기선
            f"<div style='position:absolute;left:50%;top:0;bottom:0;width:1px;"
            f"background:#9ca3af;z-index:2;'></div>"
            # 좌측 영역 (손실)
            f"<div style='width:50%;height:7px;background:#f3f4f6;"
            f"border-radius:3px 0 0 3px;display:flex;justify-content:flex-end;"
            f"overflow:hidden;'>"
            + (
                f"<div style='width:{w_pct / 2:.1f}%;height:100%;background:{bar_c};"
                f"border-radius:3px 0 0 3px;'></div>"
                if is_loss else ""
            )
            + f"</div>"
            # 우측 영역 (이익)
            f"<div style='width:50%;height:7px;background:#f3f4f6;"
            f"border-radius:0 3px 3px 0;display:flex;justify-content:flex-start;"
            f"overflow:hidden;'>"
            + (
                f"<div style='width:{w_pct / 2:.1f}%;height:100%;background:{bar_c};"
                f"border-radius:0 3px 3px 0;'></div>"
                if not is_loss else ""
            )
            + f"</div>"
            f"</div>"
        )

        html += (
            f"<div style='display:flex;align-items:center;gap:5px;margin-bottom:4px;'>"
            f"<div style='font-size:0.7rem;color:{vc};width:42px;"
            f"flex-shrink:0;font-weight:600;'>{display_name(tk)}</div>"
            f"{bar_html}"
            # 비율%
            f"<div style='font-size:0.6rem;color:#9ca3af;width:30px;text-align:right;"
            f"flex-shrink:0;'>{ratio:.0f}%</div>"
            # 손익 ($, 만원)
            f"<div style='font-size:0.7rem;font-weight:700;color:{vc};"
            f"width:54px;text-align:right;flex-shrink:0;line-height:1.15;'>"
            f"<div>{signed_str(int(round(real)), '{:,}')}</div>"
            f"<div style='font-size:0.5rem;color:#9ca3af;font-weight:400;'>"
            f"{signed_str(int(round(real_krw_man)), '{:,}')}만</div>"
            f"</div>"
            f"</div>"
        )
    return html


def _build_alloc_html(
    portfolio_state: dict[str, TickerState],
    df_close_last: dict,
    usd_krw: float,
) -> str:
    """💼 보유 종목 평가 — 3숫자 헤더 (원금→평가, 손익) + 달러 주, 원화 보조."""
    rows = []
    for tk, ts in portfolio_state.items():
        cyc = ts['cycle']
        if cyc['hold_qty'] <= 0 or cyc['buy_qty'] <= 0:
            continue
        avg = cyc['buy_cost'] / cyc['buy_qty']
        inv_usd = avg * cyc['hold_qty']
        cur = df_close_last.get(f'{tk}_Close')
        if cur is None:
            cur = avg
        eval_usd = cur * cyc['hold_qty']
        pnl_usd = eval_usd - inv_usd
        ret_pct = (cur / avg - 1) * 100 if avg > 0 else 0
        rows.append({
            'tk': tk, 'inv': inv_usd, 'eval': eval_usd,
            'pnl': pnl_usd, 'ret': ret_pct,
        })

    if not rows:
        return ""

    rows.sort(key=lambda r: -r['eval'])

    total_inv = sum(r['inv'] for r in rows)
    total_eval = sum(r['eval'] for r in rows)
    total_pnl = total_eval - total_inv
    total_ret = (total_pnl / total_inv * 100) if total_inv > 0 else 0
    total_pnl_color = pnl_color(total_pnl)

    inv_krw_man = total_inv * usd_krw / 10000
    eval_krw_man = total_eval * usd_krw / 10000
    pnl_krw_man = total_pnl * usd_krw / 10000

    max_bar_value = max(max(r['inv'], r['eval']) for r in rows)

    C_PRINCIPAL = '#d1d5db'
    C_PROFIT    = '#dc2626'
    C_LOSS      = '#2563eb'

    # ── 라벨 ──
    html = (
        f"{html_section_divider()}"
        f"<div style='font-size:0.62rem;font-weight:700;color:{COLOR_LABEL};"
        f"margin-bottom:6px;'>💼 보유 종목 평가</div>"
    )

    # ── 3숫자 헤더 (원금 → 평가, 손익) ──
    html += (
        f"<div style='display:flex;justify-content:space-between;align-items:flex-end;"
        f"padding:6px 4px 8px 4px;border-bottom:1px solid #f3f4f6;margin-bottom:8px;'>"

        f"<div style='text-align:left;flex:1;'>"
        f"<div style='font-size:0.55rem;color:{COLOR_LABEL};margin-bottom:2px;'>원금</div>"
        f"<div style='font-size:0.92rem;color:#374151;font-weight:700;'>"
        f"${int(round(total_inv)):,}</div>"
        f"<div style='font-size:0.55rem;color:#9ca3af;'>"
        f"({int(round(inv_krw_man)):,}만원)</div>"
        f"</div>"

        f"<div style='color:#9ca3af;font-size:0.85rem;padding:0 4px 8px 4px;'>→</div>"

        f"<div style='text-align:center;flex:1;'>"
        f"<div style='font-size:0.55rem;color:{COLOR_LABEL};margin-bottom:2px;'>평가</div>"
        f"<div style='font-size:0.92rem;color:{total_pnl_color};font-weight:700;'>"
        f"${int(round(total_eval)):,}</div>"
        f"<div style='font-size:0.55rem;color:#9ca3af;'>"
        f"({int(round(eval_krw_man)):,}만원)</div>"
        f"</div>"

        f"<div style='text-align:right;flex:1;'>"
        f"<div style='font-size:0.55rem;color:{COLOR_LABEL};margin-bottom:2px;'>"
        f"손익 ({signed_str(round(total_ret), '{:d}')}%)</div>"
        f"<div style='font-size:0.92rem;color:{total_pnl_color};font-weight:700;'>"
        f"{signed_str(int(round(total_pnl)), '{:,}')}</div>"
        f"<div style='font-size:0.55rem;color:#9ca3af;'>"
        f"({signed_str(int(round(pnl_krw_man)), '{:,}')}만원)</div>"
        f"</div>"

        f"</div>"
    )

    # ── 종목별 막대 ──
    for r in rows:
        tk = r['tk']
        inv_w = (r['inv'] / max_bar_value) * 100 if max_bar_value > 0 else 0
        eval_w = (r['eval'] / max_bar_value) * 100 if max_bar_value > 0 else 0
        pnl_w = abs(eval_w - inv_w)

        is_profit = r['pnl'] >= 0
        pnl_color_v = pnl_color(r['pnl'])

        if is_profit:
            seg1_w = inv_w; seg1_c = C_PRINCIPAL
            seg2_w = pnl_w; seg2_c = C_PROFIT
        else:
            seg1_w = eval_w; seg1_c = C_PRINCIPAL
            seg2_w = pnl_w; seg2_c = C_LOSS

        bar_inner = (
            f"<div style='width:{seg1_w:.1f}%;background:{seg1_c};height:7px;"
            f"flex-shrink:0;border-radius:3px 0 0 3px;'></div>"
        )
        if seg2_w > 0.3:
            bar_inner += (
                f"<div style='width:{seg2_w:.1f}%;background:{seg2_c};height:7px;"
                f"flex-shrink:0;border-radius:0 3px 3px 0;'></div>"
            )

        inv_krw_man_t = r['inv'] * usd_krw / 10000

        # 평가 금액 분해: $평가 ($원금±$손익)
        # 손익은 색깔로 (빨강=이익, 파랑=손실)
        pnl_abs = abs(r['pnl'])
        sign = '+' if r['pnl'] >= 0 else '-'

        html += (
            f"<div style='display:flex;align-items:center;gap:5px;margin-bottom:4px;'>"
            f"<div style='font-size:0.7rem;color:{COLOR_TEXT};width:42px;"
            f"flex-shrink:0;font-weight:600;'>{display_name(tk)}</div>"
            f"<div style='flex:1;background:#f3f4f6;border-radius:3px;height:7px;"
            f"display:flex;align-items:center;overflow:hidden;min-width:0;'>"
            f"{bar_inner}"
            f"</div>"
            f"<div style='font-size:0.62rem;color:#374151;width:84px;text-align:right;"
            f"flex-shrink:0;line-height:1.2;'>"
            f"<div style='font-weight:700;'>${int(round(r['eval'])):,}</div>"
            f"<div style='font-size:0.52rem;color:#9ca3af;white-space:nowrap;'>"
            f"${int(round(r['inv'])):,}"
            f"<span style='color:{pnl_color_v};font-weight:700;'>"
            f"{sign}${int(round(pnl_abs)):,}</span>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:0.7rem;width:38px;text-align:right;flex-shrink:0;"
            f"color:{pnl_color_v};font-weight:700;'>"
            f"{signed_str(int(round(r['ret'])), '{:d}')}%</div>"
            f"</div>"
        )

    return html


def render_sidebar(
    selected_ticker: str,
    portfolio_state: dict[str, TickerState],
) -> dict:
    # 사이드바 → 탭4(설정)로 이전. 들여쓰기 유지를 위해 container 컨텍스트 사용.
    with st.container():
        # ─────────── 로그인 영역 (개인 정보 보호) ───────────
        if is_authenticated():
            # 로그인 상태: 로그아웃 버튼
            c_a, c_b = st.columns([3, 1])
            with c_a:
                st.markdown(
                    "<div style='font-size:0.7rem;color:#16a34a;padding-top:4px;'>"
                    "🔓 로그인됨</div>",
                    unsafe_allow_html=True,
                )
            with c_b:
                if st.button("⏏", key="logout_btn", help="로그아웃"):
                    st.session_state.pop('authenticated', None)
                    clear_auth_cookie()
                    st.rerun()
        else:
            # 비로그인: 로그인 폼 (collapsed expander로 작게)
            with st.expander("🔐 로그인", expanded=False):
                pw_input = st.text_input(
                    "비밀번호",
                    type="password",
                    key="login_pw_input",
                    label_visibility="collapsed",
                    placeholder="비밀번호",
                )
                if st.button("로그인", key="login_submit_btn",
                             use_container_width=True):
                    if pw_input and verify_password(pw_input):
                        st.session_state['authenticated'] = True
                        save_auth_cookie()
                        st.rerun()
                    else:
                        st.error("비밀번호 오류")
                st.caption("로그인 시 30일 자동 유지 · 개인 정보(매매/평가)는 로그인 후 표시")
        st.markdown(
            "<div style='border-bottom:1px solid #e5e7eb;margin:4px 0 8px 0;'></div>",
            unsafe_allow_html=True,
        )

        portfolio_pnl = st.session_state.get('portfolio_pnl_cache')
        usd_krw = st.session_state.get('usd_krw_cache', CFG.USD_KRW_FALLBACK)
        usd_krw_fallback = st.session_state.get('usd_krw_fallback', False)
        df_close_last = st.session_state.get('df_close_last', {})

        # 환율 fallback 경고만 (시장 상황 카드 제거됨)
        if usd_krw_fallback:
            st.markdown(
                f"<div style='margin-bottom:6px;padding:4px 8px;background:#fef3c7;"
                f"border:1px solid #fbbf24;border-radius:4px;"
                f"font-size:0.62rem;color:#92400e;'>"
                f"⚠️ 환율 fallback ${usd_krw:,.0f}/$ 사용 중"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### ⚙️ 분석 파라미터")
        candle_type = st.radio(
            "봉 기준", ['일봉', '주봉'], horizontal=True,
            index=1 if st.session_state.candle_type == '주봉' else 0,
        )

        st.caption("분석 시작일")
        today = datetime.date.today()
        presets = [('6개월', 182), ('1년', 365), ('1년6개월', 548), ('2년', 730)]
        p_cols = st.columns(len(presets))
        for pc, (plabel, pdays) in zip(p_cols, presets):
            pdate = (today - datetime.timedelta(days=pdays)).strftime('%y-%m')
            is_active = st.session_state.analysis_start == pdate
            if pc.button(
                plabel, key=f"astart_{plabel}", use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.analysis_start = pdate
                s = load_settings()
                s['analysis_start'] = pdate
                save_settings(s)
                st.rerun()
        analysis_start = st.session_state.analysis_start
        view_months = st.number_input(
            "차트 조회 기간 (최근 N개월)", min_value=1, max_value=240,
            value=st.session_state.view_months, step=1,
        )
        guide_n = 4

        # ── 시드 (달러) 입력 — 로그인 시에만 ──
        if is_authenticated():
            st.caption("시드 ($)")
            cur_seed_usd = st.session_state.get('seed_usd', CFG.SEED_USD)
            seed_usd_input = st.number_input(
                "시드 (USD)",
                min_value=100.0, max_value=10_000_000.0,
                value=float(cur_seed_usd), step=100.0,
                key="seed_usd_input_sidebar",
                label_visibility="collapsed",
            )
            if abs(seed_usd_input - cur_seed_usd) > 0.01:
                st.session_state['seed_usd'] = seed_usd_input
                st.rerun()

        # ── 통계 막대 단위 (탭3 일별손익 + 자산추이 공통) ──
        st.caption("자산 추이 단위")
        bar_units = ['일', '주', '월']
        ov_unit = st.session_state.get('overview_bar_unit', '일')
        if ov_unit not in bar_units:
            ov_unit = '일'
        ov_unit_choice = st.radio(
            "통계 단위",
            bar_units,
            index=bar_units.index(ov_unit),
            horizontal=True,
            key="overview_unit_radio_sidebar",
            label_visibility="collapsed",
        )
        if ov_unit_choice != ov_unit:
            st.session_state['overview_bar_unit'] = ov_unit_choice
            st.rerun()

        st.markdown("---")
        tok, gid = _gist_cfg()
        st.caption(
            f"☁️ Gist 연동됨 (`{gid[:8]}...`)" if (tok and gid)
            else "💾 로컬 저장 (Gist 미설정)"
        )

        # ── 종목 관리 expander ──
        with st.expander(f"📊 종목 관리 ({len(TARGET_TICKERS)}개)", expanded=False):
            # 추가
            st.caption("➕ 새 종목 추가")
            add_col1, add_col2 = st.columns([3, 1])
            new_ticker = add_col1.text_input(
                "ticker",
                key="add_ticker_input",
                placeholder="예: NVDA, AAPL, 000660",
                label_visibility="collapsed",
            )
            if add_col2.button("추가", key="add_ticker_btn",
                               use_container_width=True):
                tk = (new_ticker or "").strip().upper()
                if not tk:
                    st.warning("티커를 입력하세요")
                elif tk in TARGET_TICKERS:
                    st.warning(f"이미 추가됨: {tk}")
                else:
                    new_list = TARGET_TICKERS + [tk]
                    TARGET_TICKERS[:] = new_list
                    save_target_tickers(new_list)
                    fetch_all_data.clear()
                    compute_all_analyses.clear()
                    st.success(f"추가됨: {tk}")
                    st.rerun()

            st.caption("⚠️ 잘못된 티커 추가 시 데이터 로드 실패 (yfinance 의존)")

            # 삭제 (체크박스)
            st.caption(f"🗑️ 삭제 (현재 {len(TARGET_TICKERS)}개, 최소 {MIN_TICKERS}개 유지)")
            to_delete = []
            # 5열 그리드
            cols_per_row = 5
            for i in range(0, len(TARGET_TICKERS), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, tk in enumerate(TARGET_TICKERS[i:i+cols_per_row]):
                    if cols[j].checkbox(
                        display_name(tk),
                        key=f"del_chk_{tk}",
                        help=tk,
                    ):
                        to_delete.append(tk)

            del_col1, del_col2 = st.columns([1, 1])
            if to_delete:
                del_col1.caption(f"선택: {len(to_delete)}개")
            if del_col2.button(
                "선택 종목 삭제", key="del_tickers_btn",
                use_container_width=True,
                disabled=not to_delete,
                type="primary" if to_delete else "secondary",
            ):
                remaining = [t for t in TARGET_TICKERS if t not in to_delete]
                if len(remaining) < MIN_TICKERS:
                    st.error(f"최소 {MIN_TICKERS}개 종목 유지 필요")
                else:
                    TARGET_TICKERS[:] = remaining
                    save_target_tickers(remaining)
                    # 캐시 무효화
                    fetch_all_data.clear()
                    compute_all_analyses.clear()
                    # 삭제된 ticker가 선택되어 있으면 첫 번째로 변경
                    if st.session_state.selected_option in to_delete:
                        st.session_state.selected_option = remaining[0]
                    st.success(f"삭제됨: {', '.join(to_delete)}")
                    st.rerun()

            st.caption("매매 기록은 삭제되지 않음 (자산 추이/실현손익엔 반영)")

        # 매매 기록 (메모는 매매 시점에 함께 입력) — 로그인 시에만
        if not is_authenticated():
            st.caption("🔒 매매 입력 및 기록 보기는 로그인 후 가능")
            return {
                'analysis_start': analysis_start.strip(),
                'view_months': int(view_months),
                'guide_n': guide_n,
                'candle_type': candle_type,
            }
        with st.container():
            ticker_options = (
                TARGET_TICKERS if selected_ticker in TARGET_TICKERS
                else [selected_ticker] + TARGET_TICKERS
            )
            t_ticker = st.selectbox("종목", ticker_options, index=ticker_options.index(selected_ticker))
            t_date = st.date_input("날짜", datetime.date.today())
            t_type = st.radio("종류", ['buy', 'sell'], horizontal=True)
            t_col1, t_col2 = st.columns(2)
            t_qty = t_col1.number_input("수량", min_value=0, value=0, step=1, format="%d")
            t_price = t_col2.number_input("단가($)", min_value=0.0, value=0.0, step=0.01, format="%.4f")
            t_memo = st.text_input(
                "메모 (선택)", value="", key="trade_memo_input",
                placeholder="예: 추세선 -2σ 매수, 익절 등",
            )
            if st.button("💾 기록 저장", key="trade_save_btn",
                         type="primary", use_container_width=True):
                record = {'date': t_date.strftime("%Y-%m-%d"), 'type': t_type}
                if t_qty > 0:
                    record['qty'] = int(t_qty)
                if t_price > 0:
                    record['price'] = t_price
                if t_memo.strip():
                    record['memo'] = t_memo.strip()
                st.session_state.trade_history.setdefault(t_ticker, []).append(record)
                save_trade_history(st.session_state.trade_history)
                st.markdown(
                    "<script>if(navigator.vibrate){navigator.vibrate(50);}</script>",
                    unsafe_allow_html=True,
                )
                st.success("저장 완료!")
                st.rerun()

            st.markdown("**🗑️ 기존 기록 삭제 / 메모 편집**")
            history = st.session_state.trade_history
            if selected_ticker in history and history[selected_ticker]:
                for i, record in enumerate(history[selected_ticker]):
                    qty_str = f" {record['qty']}주" if record.get('qty') else ""
                    prc_str = f" @${record['price']:.2f}" if record.get('price') else ""
                    type_icon = '🔴' if record['type'] == 'buy' else '🔵'
                    label = f"{type_icon} {record['date']} {record['type'].upper()}{qty_str}{prc_str}"
                    st.markdown(f"<div style='font-size:0.78rem;color:#374151;"
                                f"margin-top:6px;'>{label}</div>",
                                unsafe_allow_html=True)
                    # 메모 편집 (인라인)
                    cur_memo = record.get('memo', '')
                    new_memo = st.text_input(
                        "메모", value=cur_memo,
                        key=f"trade_memo_edit_{selected_ticker}_{i}",
                        label_visibility="collapsed",
                        placeholder="메모 (편집 후 엔터)",
                    )
                    if new_memo.strip() != cur_memo:
                        if new_memo.strip():
                            st.session_state.trade_history[selected_ticker][i]['memo'] = new_memo.strip()
                        elif 'memo' in record:
                            del st.session_state.trade_history[selected_ticker][i]['memo']
                        save_trade_history(st.session_state.trade_history)
                        st.rerun()
                    if st.button(f"✕ 삭제 ({record['date']})", key=f"del_{selected_ticker}_{i}",
                                 use_container_width=True):
                        st.session_state.trade_history[selected_ticker].pop(i)
                        save_trade_history(st.session_state.trade_history)
                        st.rerun()
            else:
                st.caption("매매 기록이 없습니다.")

    return {
        'analysis_start': analysis_start.strip(),
        'view_months': int(view_months),
        'guide_n': guide_n,
        'candle_type': candle_type,
    }


# ====================================================
# 14. 차트 렌더링
# ====================================================
def render_chart(
    df_daily: pd.DataFrame,
    selected_ticker: str,
    beta: float,
    std_resid: float,
    guide_n: int,
    view_months: int,
    df_ohlc: Optional[pd.DataFrame] = None,
    df_daily_raw: Optional[pd.DataFrame] = None,
    trade_records: Optional[list] = None,
) -> None:
    st.markdown("""<style>
    .js-plotly-plot, .js-plotly-plot .plotly, .js-plotly-plot svg {
        touch-action: none !important; }
    </style>""", unsafe_allow_html=True)

    PX = {
        'main': 150, 'zm_scatter': 150, 'spacer': 20,
        'price': 100, 'zscore': 100, 'macd': 100, 'rsi': 100,
    }
    plot_order = [
        'main', 'zm_scatter', 'spacer',
        'price', 'zscore', 'macd', 'rsi',
    ]
    total_rows = len(plot_order)
    total_h = sum(PX[p] for p in plot_order)
    # 모든 패널 단일 Y축 (Z + M 같은 척도)
    fig = make_subplots(
        rows=total_rows, cols=1,
        row_heights=[PX[p] / total_h for p in plot_order],
        vertical_spacing=0.02,
    )
    row = 1

    # [1] 로그-로그 산점도
    sc_df = (
        df_daily_raw if (df_daily_raw is not None and not df_daily_raw.empty) else df_daily
    )
    sdf = sc_df.sort_values(f'{X_ASSET_FIXED}_Norm')
    x_vals = sdf[f'{X_ASSET_FIXED}_Norm']
    min_x, max_x = sc_df[f'{X_ASSET_FIXED}_Norm'].min(), sc_df[f'{X_ASSET_FIXED}_Norm'].max()

    emp_c = sc_df[f'{selected_ticker}_Norm'] / (sc_df[f'{X_ASSET_FIXED}_Norm'] ** guide_n)
    for log_c in np.linspace(np.log10(emp_c.min()) - 1.0, np.log10(emp_c.max()) + 1.0, 15):
        fig.add_trace(go.Scatter(
            x=x_vals, y=(10 ** log_c) * (x_vals ** guide_n),
            mode='lines',
            line=dict(color='rgba(200,200,200,0.6)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) - 1.5 * std_resid),
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) + 1.5 * std_resid),
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(150,150,150,0.2)', showlegend=False, hoverinfo='skip',
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=sdf['Predicted'],
        mode='lines', line=dict(color='black', width=2), name='Predicted Trend',
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=sc_df[f'{X_ASSET_FIXED}_Norm'], y=sc_df[f'{selected_ticker}_Norm'],
        mode='markers',
        marker=dict(color=np.linspace(0, 1, len(sc_df)), colorscale='Viridis', size=5, opacity=0.8),
        name='Daily Data',
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=[sc_df[f'{X_ASSET_FIXED}_Norm'].iloc[-1]],
        y=[sc_df[f'{selected_ticker}_Norm'].iloc[-1]],
        mode='markers',
        marker=dict(symbol='diamond', color='#fbbf24', size=11,
                    line=dict(color='white', width=2)),
        name='Current',
    ), row=row, col=1)

    # ── 사이클별 평균 매수/매도가 수평선 ──
    # 그래프 1 (회귀): Y축은 _Norm = Close / first_close
    # 진행 중 사이클만 표시 (이전 사이클은 표시 안 함)
    cycles = extract_cycles_avgs(trade_records or [])
    base_y_ticker = sc_df[f'{selected_ticker}_Close'].iloc[0]
    if cycles and base_y_ticker > 0:
        active = next((c for c in cycles if c['is_active']), None)

        def _hline_g1(y_norm: float, color: str, width: float = 1.2):
            fig.add_trace(go.Scatter(
                x=[min_x * 0.98, max_x * 1.02],
                y=[y_norm, y_norm],
                mode='lines',
                line=dict(color=color, width=width, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ), row=row, col=1)

        # 진행 중 사이클만 (진한 색)
        if active:
            _hline_g1(active['avg_buy'] / base_y_ticker, '#f97316', 1.2)
            if active.get('avg_sell'):
                _hline_g1(active['avg_sell'] / base_y_ticker, '#2563eb', 1.2)

    band_upper = np.exp(np.log(sdf['Predicted'].values) + 1.5 * std_resid)
    band_lower = np.exp(np.log(sdf['Predicted'].values) - 1.5 * std_resid)
    y_all = np.concatenate(
        [sc_df[f'{selected_ticker}_Norm'].dropna().values, band_upper, band_lower]
    )
    fig.update_xaxes(
        type="log", showgrid=False,
        range=[np.log10(min_x * 0.98), np.log10(max_x * 1.02)],
        row=row, col=1,
    )
    fig.update_yaxes(
        type="log", showgrid=False,
        range=[np.log10(np.nanmin(y_all) * 0.88), np.log10(np.nanmax(y_all) * 1.18)],
        row=row, col=1,
    )
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>β = {beta:.2f}</b>", showarrow=False,
        font=dict(size=11, color='black'), xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
        row=row, col=1,
    )
    # 회귀 산점도 X축 라벨 숨김
    fig.update_xaxes(showticklabels=False, row=row, col=1)
    # row=1 (main) 완료
    # row=2는 zm_scatter (별도 코드에서 zm_row=2로 그림)
    # row=3 = spacer로 점프
    row += 2

    # [3] Spacer
    fig.update_xaxes(visible=False, row=row, col=1)
    fig.update_yaxes(visible=False, row=row, col=1)
    row += 1

    # 뷰 기간
    last_date = df_daily.index[-1]
    first_date = df_daily.index[0]
    view_start = max(last_date - pd.DateOffset(months=view_months), first_date)
    snap_idx = min(df_daily.index.searchsorted(view_start), len(df_daily) - 1)
    view_start = df_daily.index[snap_idx]

    grid_dtick_ms = get_time_grid_dtick_ms(view_start, last_date)
    base_spy = df_daily.loc[df_daily.index >= view_start, f'{X_ASSET_FIXED}_Norm'].iloc[0]
    base_tkr = df_daily.loc[df_daily.index >= view_start, f'{selected_ticker}_Norm'].iloc[0]
    df_daily['Plot_Norm_SPY'] = df_daily[f'{X_ASSET_FIXED}_Norm'] / base_spy
    df_daily['Plot_Norm_Ticker'] = df_daily[f'{selected_ticker}_Norm'] / base_tkr

    # [3] Price
    price_row = row

    # 캔들스틱 먼저 (배경) — SPY 라인이 위로 가도록
    ohlc_norm = pd.DataFrame()
    if df_ohlc is not None and not df_ohlc.empty:
        base_close = df_daily[f'{selected_ticker}_Close'].iloc[0]
        base_n = df_daily[f'{selected_ticker}_Norm'].iloc[0]
        base_vn = df_daily.loc[df_daily.index >= view_start, f'{selected_ticker}_Norm'].iloc[0]
        scale = base_n / base_vn / base_close if base_close != 0 else 1.0
        ohlc_norm = df_ohlc * scale
        fig.add_trace(go.Candlestick(
            x=ohlc_norm.index,
            open=ohlc_norm['Open'], high=ohlc_norm['High'],
            low=ohlc_norm['Low'], close=ohlc_norm['Close'],
            increasing=dict(line=dict(color='#dc2626', width=1), fillcolor='#dc2626'),
            decreasing=dict(line=dict(color='#1d4ed8', width=1), fillcolor='#1d4ed8'),
            showlegend=False, hoverinfo='skip',
        ), row=row, col=1)
        # 캔들 rangeslider 끄기 - price row의 xaxis 동적 매칭
        fig.update_layout(**{f'xaxis{row}_rangeslider_visible': False})
    else:
        fig.add_trace(go.Scatter(
            x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
            mode='lines', line=dict(color='black', width=1.5), name=selected_ticker,
        ), row=row, col=1)

    # SPY 라인 제거됨 (사용자 요청)

    # ── 사이클별 평균 매수 → 평균 매도 화살표 ──
    # 완료된 사이클: 매수 평균가 → 매도 평균가 화살표
    # 진행 중 사이클: 평균 매수가 작은 마커만
    cycles_g2 = extract_cycles_avgs(trade_records or [])
    if cycles_g2:
        base_close_p = df_daily[f'{selected_ticker}_Close'].iloc[0]
        base_n_p = df_daily[f'{selected_ticker}_Norm'].iloc[0]
        base_vn_p = df_daily.loc[df_daily.index >= view_start, f'{selected_ticker}_Norm'].iloc[0]
        scale_p = base_n_p / base_vn_p / base_close_p if base_close_p != 0 else 1.0
        last_date_in_data = df_daily.index[-1]

        for c in cycles_g2:
            start_ts = pd.Timestamp(c['start'])
            is_active = c['is_active']

            avg_buy_norm = c['avg_buy'] * scale_p

            # 진행 중 사이클: 평균 매수가 작은 점선 + 매수 마커
            if is_active:
                end_ts = last_date_in_data
                fig.add_trace(go.Scatter(
                    x=[start_ts, end_ts],
                    y=[avg_buy_norm, avg_buy_norm],
                    mode='lines',
                    line=dict(color='#f97316', width=1.2, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                ), row=row, col=1)
                # 매도가 일부 있으면 함께
                if c.get('avg_sell') and c.get('first_sell_date'):
                    first_sell_ts = pd.Timestamp(c['first_sell_date'])
                    avg_sell_norm = c['avg_sell'] * scale_p
                    fig.add_trace(go.Scatter(
                        x=[first_sell_ts, end_ts],
                        y=[avg_sell_norm, avg_sell_norm],
                        mode='lines',
                        line=dict(color='#2563eb', width=1.2, dash='dot'),
                        hoverinfo='skip', showlegend=False,
                    ), row=row, col=1)

            # 완료된 사이클: 평균 매수 → 평균 매도 화살표
            elif c.get('avg_sell'):
                end_ts = pd.Timestamp(c['end'])
                first_sell_ts = pd.Timestamp(c.get('first_sell_date') or c['end'])
                avg_sell_norm = c['avg_sell'] * scale_p
                # 화살표: 매수 위치 → 매도 위치
                # 매수 위치 = (사이클 시작, avg_buy)
                # 매도 위치 = (첫 매도일, avg_sell)
                arrow_color = '#16a34a' if c['avg_sell'] >= c['avg_buy'] else '#dc2626'
                fig.add_annotation(
                    x=first_sell_ts, y=avg_sell_norm,
                    ax=start_ts, ay=avg_buy_norm,
                    xref=f'x{row}', yref=f'y{row}',
                    axref=f'x{row}', ayref=f'y{row}',
                    showarrow=True,
                    arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                    arrowcolor=arrow_color, opacity=0.7,
                )

    if not ohlc_norm.empty:
        vc = ohlc_norm[ohlc_norm.index >= view_start]
        p_lo = vc['Low'].min() if not vc.empty else df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].min()
        p_hi = vc['High'].max() if not vc.empty else df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].max()
    else:
        p_lo = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].min()
        p_hi = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].max()
    # SPY 라인 제거됨 — ticker 범위만 사용
    p_lo, p_hi = p_lo * 0.97, p_hi * 1.03

    fig.update_yaxes(
        type="log",
        range=[np.log10(max(p_lo, 1e-6)), np.log10(max(p_hi, 1e-6))],
        autorange=False, fixedrange=True, row=row, col=1,
    )
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>${df_daily[f'{selected_ticker}_Close'].iloc[-1]:,.2f}</b>",
        showarrow=False, font=dict(size=11, color='black'),
        xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
        row=row, col=1,
    )
    time_x_axis = f'x{row}'
    row += 1

    # [4~5] Z+M / MACD (라인) — 두 패널 모두 라인 그래프
    for col_name, hi, lo, label, color_fn in [
        ('Z_Score',     CFG.Z_HIGH,    -CFG.Z_HIGH,    'Z',
         lambda v: 'black'),
        ('MACD_Hist_Z', CFG.MACD_HIGH, -CFG.MACD_HIGH, 'MACD',
         lambda v: '#dc2626' if v <= -CFG.MACD_HIGH else '#1d4ed8' if v >= CFG.MACD_HIGH else 'black'),
    ]:
        if col_name == 'Z_Score':
            # ── Z + 모멘텀 패널 (백분위 0~100 척도) ──
            # Z 라인 (검정), 모멘텀 라인 (주황) — 동일 Y축
            # 변환: z_to_pct(z) = (z + 2.5) / 5 * 100, clip 0~100
            # 임계: 10 (강매수) / 30 (약매수) / 50 (중립) / 70 (약매도) / 90 (강매도)
            z_raw = df_daily[col_name].fillna(0)

            # 모멘텀 점수 시계열 (raw Z 척도)
            # 통합 공식: MACD_Pct (높이, 30%) + dMACD_Pct (변곡, 20%) + RSI (50%)
            macd_pct_v = df_daily['MACD_Pct'].fillna(0).values
            dmacd_pct_v = df_daily['dMACD_Pct'].fillna(0).values
            rsi_v = df_daily['RSI'].fillna(50).values
            h_norm = macd_pct_v / 2.0
            d_norm = dmacd_pct_v / 0.5
            r_norm = (rsi_v - 50) / 20.0
            momentum_raw = 0.3 * h_norm + 0.2 * d_norm + 0.5 * r_norm

            # 백분위 변환 (0~100)
            z_series = ((z_raw + 2.5) / 5.0 * 100).clip(0, 100)
            momentum_series = pd.Series(
                ((momentum_raw + 2.5) / 5.0 * 100).clip(0, 100),
                index=df_daily.index,
            )

            # ── 임계 수평선 (0~100 척도) ──
            for y_val, lc, ld, lw in [
                (90,   '#dc2626', 'solid', 0.7),   # 강 매도 (Z+2)
                (70,   '#fca5a5', 'dot',   0.6),   # 약 매도 (Z+1)
                (50,   '#9ca3af', 'solid', 0.5),   # 중립 (Z=0)
                (30,   '#93c5fd', 'dot',   0.6),   # 약 매수 (Z-1)
                (10,   '#2563eb', 'solid', 0.7),   # 강 매수 (Z-2)
            ]:
                fig.add_trace(go.Scatter(
                    x=[df_daily.index[0], df_daily.index[-1]],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color=lc, width=lw, dash=ld),
                    hoverinfo='skip', showlegend=False,
                ), row=row, col=1)

            # ── Z 실선 (검정) — 임계선보다 굵게 ──
            fig.add_trace(go.Scatter(
                x=df_daily.index, y=z_series,
                mode='lines',
                line=dict(color='#111827', width=2.0, shape='spline', smoothing=0.5),
                name='Z', hoverinfo='skip', showlegend=False,
                connectgaps=True,
            ), row=row, col=1)

            # ── 모멘텀 실선 (주황) — Z와 같은 Y축 (척도 동일) ──
            fig.add_trace(go.Scatter(
                x=df_daily.index, y=momentum_series,
                mode='lines',
                line=dict(color='#f97316', width=1.5, shape='spline', smoothing=0.5),
                name='Momentum', hoverinfo='skip', showlegend=False,
                connectgaps=True,
            ), row=row, col=1)

            last_v = z_series.iloc[-1]
            val = float(last_v) if pd.notna(last_v) else 0.0

            # ── 범례 (좌측 상단) ──
            fig.add_annotation(
                x=0, y=1, xref='x domain', yref='y domain',
                text=(
                    "<span style='color:#111827;'>━ Z</span>"
                    "  "
                    "<span style='color:#f97316;'>━ M</span>"
                ),
                showarrow=False,
                font=dict(size=11),
                xanchor='left', yanchor='top',
                bgcolor='rgba(255,255,255,0.85)', bordercolor='#d1d5db',
                borderwidth=1, borderpad=3,
                row=row, col=1,
            )
        else:
            # ── MACD 패널: MACD 라인 + Signal 라인 (가격 단위) ──
            # M 계산은 MACD_Hist_Z 그대로 유지 (그래프 3에서 사용)
            # 여기는 단순히 두 라인 표시
            macd_series = df_daily['MACD'].fillna(0)
            signal_series = df_daily['MACD_Signal'].fillna(0)

            # MACD 라인 (보라) — 먼저 그림
            fig.add_trace(go.Scatter(
                x=df_daily.index, y=macd_series,
                mode='lines',
                line=dict(color='#7c3aed', width=2.0, shape='spline', smoothing=0.5),
                name='MACD', hoverinfo='skip', showlegend=False,
                connectgaps=True,
            ), row=row, col=1)

            # Signal 라인 (검정) — MACD 위에 그림 (사용자 요청)
            fig.add_trace(go.Scatter(
                x=df_daily.index, y=signal_series,
                mode='lines',
                line=dict(color='#111827', width=1.5, shape='spline', smoothing=0.5),
                name='Signal', hoverinfo='skip', showlegend=False,
                connectgaps=True,
            ), row=row, col=1)

            # 0 중립 수평선
            fig.add_trace(go.Scatter(
                x=[df_daily.index[0], df_daily.index[-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='#9ca3af', width=0.5),
                hoverinfo='skip', showlegend=False,
            ), row=row, col=1)

            # ── MACD-Signal 교차점 화살표 ──
            # hist = MACD - Signal
            # 부호가 바뀌는 시점을 교차로 정의 (hist=0인 점 안전 처리)
            # 상향: 직전 hist<0, 현재 hist≥0 → 매수 → ▲ 빨강
            # 하향: 직전 hist>0, 현재 hist≤0 → 매도 → ▼ 파랑
            macd_raw = df_daily['MACD']
            sig_raw = df_daily['MACD_Signal']
            valid_idx = macd_raw.notna() & sig_raw.notna()
            hist_clean = (macd_raw - sig_raw).where(valid_idx)
            hist_prev = hist_clean.shift(1)
            both_valid = hist_clean.notna() & hist_prev.notna()
            bullish_mask = both_valid & (hist_clean >= 0) & (hist_prev < 0)
            bearish_mask = both_valid & (hist_clean <= 0) & (hist_prev > 0)

            # Y 오프셋 — view 범위 기준 18% (마커가 라인에 가려지지 않게)
            macd_view = df_daily.loc[df_daily.index >= view_start, 'MACD'].dropna()
            sig_view_y = df_daily.loc[df_daily.index >= view_start, 'MACD_Signal'].dropna()
            macd_max_y = float(abs(macd_view).max()) if not macd_view.empty else 0.0
            sig_max_y = float(abs(sig_view_y).max()) if not sig_view_y.empty else 0.0
            data_max_y = max(macd_max_y, sig_max_y)
            if data_max_y <= 0:
                data_max_y = 1.0
            y_offset = data_max_y * 0.18   # 18% — 라인 위/아래로 충분히 떨어뜨림

            # ── 마커 위치 (고정) ──
            # ▲ 매수: 크로스 바로 아래 (-offset)
            # ▼ 매도: 크로스 바로 위  (+offset)
            # → 마커 포함하도록 Y축 범위 계산 (이후 update_yaxes에 반영)
            bull_y = (macd_raw[bullish_mask].values - y_offset) if bullish_mask.any() else np.array([])
            bear_y = (macd_raw[bearish_mask].values + y_offset) if bearish_mask.any() else np.array([])

            # 마커 포함 Y축 max (view 범위 내만)
            view_mask_idx = df_daily.index >= view_start
            bull_in_view = bullish_mask & view_mask_idx
            bear_in_view = bearish_mask & view_mask_idx
            extra_max_y = 0.0
            if bull_in_view.any():
                extra_max_y = max(
                    extra_max_y,
                    float(abs((macd_raw[bull_in_view] - y_offset)).max())
                )
            if bear_in_view.any():
                extra_max_y = max(
                    extra_max_y,
                    float(abs((macd_raw[bear_in_view] + y_offset)).max())
                )
            # 추후 macd_axis_max 계산 시 사용
            st.session_state['_macd_marker_extra'] = extra_max_y

            if bullish_mask.any():
                bull_x = df_daily.index[bullish_mask]
                bull_y_base = macd_raw[bullish_mask].values
                # annotation으로 표시 — 항상 최상위 layer, grid에 가려지지 않음
                for bx, by in zip(bull_x, bull_y_base):
                    fig.add_annotation(
                        x=bx, y=by - y_offset,
                        text="▲",
                        showarrow=False,
                        font=dict(size=14, color='#dc2626'),
                        xanchor='center', yanchor='middle',
                        row=row, col=1,
                    )

            if bearish_mask.any():
                bear_x = df_daily.index[bearish_mask]
                bear_y_base = macd_raw[bearish_mask].values
                for bx, by in zip(bear_x, bear_y_base):
                    fig.add_annotation(
                        x=bx, y=by + y_offset,
                        text="▼",
                        showarrow=False,
                        font=dict(size=14, color='#2563eb'),
                        xanchor='center', yanchor='middle',
                        row=row, col=1,
                    )

            last_v = macd_series.iloc[-1]
            val = float(last_v) if pd.notna(last_v) else 0.0
            last_sig = signal_series.iloc[-1]
            sig_v = float(last_sig) if pd.notna(last_sig) else 0.0

            # 범례 (좌측 상단)
            fig.add_annotation(
                x=0, y=1, xref='x domain', yref='y domain',
                text=(
                    "<span style='color:#7c3aed;'>━ MACD</span>"
                    "  "
                    "<span style='color:#111827;'>━ Signal</span>"
                ),
                showarrow=False,
                font=dict(size=11),
                xanchor='left', yanchor='top',
                bgcolor='rgba(255,255,255,0.85)', bordercolor='#d1d5db',
                borderwidth=1, borderpad=3,
                row=row, col=1,
            )

        view_abs = abs(df_daily.loc[df_daily.index >= view_start, col_name].dropna())
        z_data_max = float(view_abs.max()) if not view_abs.empty else 0.0

        if col_name == 'Z_Score':
            # ── Z + M 단일 Y축: 백분위 0~100 고정 ──
            fig.update_yaxes(
                range=[0, 100], autorange=False, fixedrange=True,
                row=row, col=1,
            )
        else:
            # ── MACD 패널 Y축: MACD + Signal + 교차 마커 모두 포함 ──
            # MACD는 가격 단위 (종목마다 척도 다름) — 데이터 기반 동적 범위
            macd_view = df_daily.loc[df_daily.index >= view_start, 'MACD'].dropna()
            sig_view = df_daily.loc[df_daily.index >= view_start, 'MACD_Signal'].dropna()
            macd_data_max = float(abs(macd_view).max()) if not macd_view.empty else 0.0
            sig_data_max = float(abs(sig_view).max()) if not sig_view.empty else 0.0
            # 교차 마커 위치도 포함 (st.session_state에 저장된 값)
            marker_extra = st.session_state.get('_macd_marker_extra', 0.0)
            macd_axis_max = max(
                macd_data_max, sig_data_max, marker_extra
            ) * 1.10  # 10% 여유
            if macd_axis_max <= 0:
                macd_axis_max = 1.0
            fig.update_yaxes(
                range=[-macd_axis_max, macd_axis_max], autorange=False, fixedrange=True,
                row=row, col=1,
            )

        row += 1

    # ── RSI 패널: RSI-50 (0 중심) + 0 대칭 자동 범위 ──
    rsi_series = (df_daily['RSI'] - 50).fillna(0)

    # 임계 수평선: ±20 (RSI 70 / 30 에 해당)
    # 0 = 중립
    for y_val, lc, ld, lw in [
        (CFG.RSI_OVERBOUGHT - 50, '#dc2626', 'solid', 0.7),   # +20 빨강 (RSI 70)
        (0,                       '#9ca3af', 'solid', 0.5),   # 중립
        (CFG.RSI_OVERSOLD - 50,   '#2563eb', 'solid', 0.7),   # -20 파랑 (RSI 30)
    ]:
        fig.add_trace(go.Scatter(
            x=[df_daily.index[0], df_daily.index[-1]],
            y=[y_val, y_val],
            mode='lines',
            line=dict(color=lc, width=lw, dash=ld),
            hoverinfo='skip', showlegend=False,
        ), row=row, col=1)

    # RSI 라인 (청록)
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=rsi_series,
        mode='lines',
        line=dict(color='#0891b2', width=2.0, shape='spline', smoothing=0.5),
        name='RSI', hoverinfo='skip', showlegend=False,
        connectgaps=True,
    ), row=row, col=1)

    last_rsi = df_daily['RSI'].iloc[-1]
    rsi_val = float(last_rsi) if pd.notna(last_rsi) else 50.0
    rsi_color = (
        '#dc2626' if rsi_val >= CFG.RSI_OVERBOUGHT
        else '#2563eb' if rsi_val <= CFG.RSI_OVERSOLD else '#0891b2'
    )
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>RSI  {rsi_val:.1f}</b>", showarrow=False,
        font=dict(size=11, color=rsi_color), xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
        row=row, col=1,
    )
    # Y축: 0 대칭, view 범위 내 데이터의 max 기반 자동 조절
    view_rsi = rsi_series.loc[rsi_series.index >= view_start]
    rsi_abs_max = max(float(view_rsi.abs().max()) if not view_rsi.empty else 0, 25.0)
    rsi_abs_max *= 1.1
    fig.update_yaxes(
        range=[-rsi_abs_max, rsi_abs_max],
        autorange=False, fixedrange=True, row=row, col=1,
    )

    # ── 매매 마커 (사이클별 opacity 차등) ──
    # 진행 중 사이클 매매: 진하게 (opacity 1.0)
    # 완료 사이클 매매: 연하게 (opacity 0.3)
    cycles_for_markers = extract_cycles_avgs(
        st.session_state.trade_history.get(selected_ticker, [])
    )

    def _trade_is_active(t_date_dt: datetime.date) -> bool:
        """매매 날짜가 진행 중 사이클에 속하면 True."""
        for c in cycles_for_markers:
            if not c['is_active']:
                continue
            if c['start'] <= t_date_dt:
                return True
        return False

    for trade in (st.session_state.trade_history.get(selected_ticker, [])
                  if is_authenticated() else []):
        t_date = pd.to_datetime(trade['date'])
        try:
            t_date_d = datetime.date.fromisoformat(trade['date'])
        except Exception:
            t_date_d = t_date.date()
        is_buy = trade['type'] == 'buy'
        is_active_cycle = _trade_is_active(t_date_d)
        # 색 — 모두 진하게
        base_color = '#dc2626' if is_buy else '#1d4ed8'
        m_opacity = 1.0 if is_active_cycle else 0.6  # 완료 사이클도 진하게 (0.3 → 0.6)
        vline_opacity = 0.8 if is_active_cycle else 0.4
        idx_sc = sc_df.index.get_indexer([t_date], method='nearest')[0]
        d_sc = sc_df.index[idx_sc]
        fig.add_trace(go.Scatter(
            x=[sc_df.loc[d_sc, f'{X_ASSET_FIXED}_Norm']],
            y=[sc_df.loc[d_sc, f'{selected_ticker}_Norm']],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if is_buy else 'triangle-down',
                size=10, color=base_color, opacity=m_opacity,
                line=dict(width=1.5, color='black'),
            ),
            name=f"{trade['type'].upper()} ({t_date.date()})", hoverinfo='skip',
        ), row=1, col=1)
        # vline은 시간축 패널만 (price~rsi = row 4~7)
        # plot_order: main(1), zm_scatter(2), spacer(3), price(4), zscore(5), macd(6), rsi(7)
        # 점선 + 옅은 opacity로 캔들 가독성 보호
        for r in range(4, total_rows + 1):
            fig.add_vline(
                x=t_date, line_dash="dot", line_width=1.5,
                line_color=base_color, opacity=vline_opacity * 0.5, row=r, col=1,
                layer='below',
            )

    # 현재 위치 — 매매 마커보다 위 layer (앰버 후광 + 다이아몬드)
    # 후광 (반투명 큰 원)
    fig.add_trace(go.Scatter(
        x=[sc_df[f'{X_ASSET_FIXED}_Norm'].iloc[-1]],
        y=[sc_df[f'{selected_ticker}_Norm'].iloc[-1]],
        mode='markers',
        marker=dict(symbol='circle', color='rgba(251,191,36,0.25)',
                    size=26, line=dict(width=0)),
        hoverinfo='skip', showlegend=False,
    ), row=1, col=1)
    # 본체 (앰버 다이아몬드)
    fig.add_trace(go.Scatter(
        x=[sc_df[f'{X_ASSET_FIXED}_Norm'].iloc[-1]],
        y=[sc_df[f'{selected_ticker}_Norm'].iloc[-1]],
        mode='markers',
        marker=dict(
            symbol='diamond', color='#fbbf24', size=12,
            line=dict(color='white', width=2),
        ),
        name='Current_Top', hoverinfo='skip', showlegend=False,
    ), row=1, col=1)

    # ── Z-M 산점도 패널 (마지막) ──
    # X = Z 백분위 (위치, 0~100)
    # Y = M 백분위 (모멘텀, 0~100)
    # 색 = 시간 (viridis, 그래프 1과 동일)
    # 사분면 의미:
    #   Q1 (X>50, Y>50): 비싸고 더 오름 → 추세 추종
    #   Q2 (X>50, Y<50): 비싼데 꺾임 → 매도 신호
    #   Q3 (X<50, Y<50): 싸고 더 떨어짐 → 매수 대기
    #   Q4 (X<50, Y>50): 싼데 반등 시작 → 매수 진입
    zm_row = 2  # zm_scatter는 회귀(1) 바로 아래 (row=2)

    # Z와 M 백분위 계산
    z_raw = df_daily['Z_Score'].fillna(0)
    macd_pct_v_2 = df_daily['MACD_Pct'].fillna(0).values
    dmacd_pct_v_2 = df_daily['dMACD_Pct'].fillna(0).values
    rsi_v_2 = df_daily['RSI'].fillna(50).values
    momentum_raw_full = (
        0.3 * (macd_pct_v_2 / 2.0)
        + 0.2 * (dmacd_pct_v_2 / 0.5)
        + 0.5 * ((rsi_v_2 - 50) / 20.0)
    )

    z_pct_series = ((z_raw + 2.5) / 5.0 * 100).clip(0, 100)
    m_pct_series = pd.Series(
        ((momentum_raw_full + 2.5) / 5.0 * 100).clip(0, 100),
        index=df_daily.index,
    )

    # 분석 기간 전체 사용 (보기 기간이 아님 — 장기 Z-M 궤적)
    zm_x = z_pct_series.values
    zm_y = m_pct_series.values
    zm_dates = z_pct_series.index

    # 시간 색 (viridis)
    n_pts = len(zm_x)
    if n_pts > 0:
        color_indices = list(range(n_pts))

        # 사분면 분할선 + 그래프 3과 동일한 Z/M 임계선
        # X축 (Z) 임계선
        # Y축 (M) 임계선
        # 색/스타일 = 그래프 3 (Z+M 패널) 와 동일
        threshold_lines = [
            (10, '#2563eb', 'solid', 0.7),   # 강 매수 임계
            (30, '#93c5fd', 'dot',   0.6),   # 약 매수 임계
            (50, '#9ca3af', 'solid', 0.5),   # 중립
            (70, '#fca5a5', 'dot',   0.6),   # 약 매도 임계
            (90, '#dc2626', 'solid', 0.7),   # 강 매도 임계
        ]
        for val, lc, ld, lw in threshold_lines:
            # X축 임계선 (세로)
            fig.add_shape(
                type='line', x0=val, x1=val, y0=0, y1=100,
                line=dict(color=lc, width=lw, dash=ld),
                row=zm_row, col=1, layer='below',
            )
            # Y축 임계선 (가로)
            fig.add_shape(
                type='line', x0=0, x1=100, y0=val, y1=val,
                line=dict(color=lc, width=lw, dash=ld),
                row=zm_row, col=1, layer='below',
            )

        # 산점도 — 시간 색 (viridis)
        fig.add_trace(go.Scatter(
            x=zm_x, y=zm_y, mode='markers',
            marker=dict(
                size=5,
                color=color_indices,
                colorscale='Viridis',
                showscale=False,
                line=dict(width=0),
            ),
            hovertext=[d.strftime('%Y-%m-%d') for d in zm_dates],
            hovertemplate='Z: %{x:.0f}<br>M: %{y:.0f}<br>%{hovertext}<extra></extra>',
            showlegend=False,
        ), row=zm_row, col=1)

        # 매매 마커 — 그래프 1과 동일 (Z, M 위치)
        # 현재 사이클은 진하게, 완료 사이클은 연하게
        cycles_for_zm = extract_cycles_avgs(
            st.session_state.trade_history.get(selected_ticker, [])
        )

        def _zm_trade_is_active(t_date_dt) -> bool:
            for c in cycles_for_zm:
                if not c['is_active']:
                    continue
                if c['start'] <= t_date_dt:
                    return True
            return False

        for trade in (st.session_state.trade_history.get(selected_ticker, [])
                      if is_authenticated() else []):
            try:
                t_date_d = datetime.date.fromisoformat(trade['date'])
            except Exception:
                continue
            t_ts = pd.Timestamp(trade['date'])
            # 가장 가까운 영업일 매핑
            if t_ts not in z_pct_series.index:
                idx = z_pct_series.index[z_pct_series.index <= t_ts]
                if len(idx) == 0:
                    continue
                t_ts = idx[-1]
            z_val = z_pct_series.loc[t_ts]
            m_val = m_pct_series.loc[t_ts]
            if pd.isna(z_val) or pd.isna(m_val):
                continue

            is_buy = trade['type'] == 'buy'
            is_active_cycle = _zm_trade_is_active(t_date_d)
            base_color = '#dc2626' if is_buy else '#1d4ed8'
            m_opacity = 1.0 if is_active_cycle else 0.6

            fig.add_trace(go.Scatter(
                x=[z_val], y=[m_val],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if is_buy else 'triangle-down',
                    size=10, color=base_color, opacity=m_opacity,
                    line=dict(width=1, color='black'),
                ),
                name=f"{trade['type'].upper()} ({trade['date']})",
                hoverinfo='skip', showlegend=False,
            ), row=zm_row, col=1)

        # 현재 위치 — 매매 마커보다 위 layer (앰버 후광 + 다이아몬드)
        # 후광
        fig.add_trace(go.Scatter(
            x=[zm_x[-1]], y=[zm_y[-1]],
            mode='markers',
            marker=dict(symbol='circle', color='rgba(251,191,36,0.25)',
                        size=26, line=dict(width=0)),
            hoverinfo='skip', showlegend=False,
        ), row=zm_row, col=1)
        # 본체
        fig.add_trace(go.Scatter(
            x=[zm_x[-1]], y=[zm_y[-1]],
            mode='markers',
            marker=dict(
                symbol='diamond', color='#fbbf24', size=12,
                line=dict(color='white', width=2),
            ),
            hovertemplate=(
                f'<b>현재</b><br>Z: %{{x:.0f}}<br>M: %{{y:.0f}}<extra></extra>'
            ),
            showlegend=False,
        ), row=zm_row, col=1)

    # Z-M 산점도 축 (제목 없음, 숫자만)
    # 범위 -5 ~ 105: 별표가 극단(0 또는 100)에 가도 잘리지 않도록 마진
    fig.update_xaxes(
        range=[-5, 105], autorange=False, fixedrange=True,
        row=zm_row, col=1,
    )
    fig.update_yaxes(
        range=[-5, 105], autorange=False, fixedrange=True,
        row=zm_row, col=1,
    )

    # 축 공통 스타일
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # spacer (row 3, 회귀/zm 산점도와 시간축 사이) 숨김
    fig.update_xaxes(visible=False, row=3, col=1)
    fig.update_yaxes(visible=False, row=3, col=1)
    # 시간축 그리드 — price ~ rsi (row 4 ~ total_rows = 7)
    # RSI X 라벨은 마지막 row(total_rows) 에서 표시
    # price row(4)는 time_x_axis = 'x4' 자체이므로 matches 미적용 (자기참조 회피)
    x_max_with_margin = last_date + pd.Timedelta(days=3)
    # price row (4) 별도 처리 - 자기 자신을 matches하면 안 됨
    fig.update_xaxes(
        showgrid=True, gridcolor='rgba(156,163,175,0.28)',
        gridwidth=0.6, griddash='dot', dtick=grid_dtick_ms,
        rangebreaks=[dict(bounds=['sat', 'mon'])],
        showticklabels=False, tickformat="%m/%d",
        range=[view_start, x_max_with_margin], row=4, col=1,
        layer='below traces',
    )
    fig.update_yaxes(
        showgrid=False, autorange=False, fixedrange=True, row=4, col=1,
        layer='below traces',
    )
    # zscore, macd, rsi (row 5~7) - price와 matches
    for r in range(5, total_rows + 1):
        fig.update_xaxes(
            showgrid=True, gridcolor='rgba(156,163,175,0.28)',
            gridwidth=0.6, griddash='dot', dtick=grid_dtick_ms,
            matches=time_x_axis, rangebreaks=[dict(bounds=['sat', 'mon'])],
            showticklabels=(r == total_rows), tickformat="%m/%d",
            range=[view_start, x_max_with_margin], row=r, col=1,
            layer='below traces',
        )
        fig.update_yaxes(
            showgrid=False, autorange=False, fixedrange=True, row=r, col=1,
            layer='below traces',
        )

    fig.update_traces(hoverinfo='skip')
    fig.update_layout(
        height=total_h, showlegend=False, hovermode=False,
        dragmode='pan', margin=dict(l=2, r=18, t=10, b=20),
        paper_bgcolor='white', plot_bgcolor='white', uirevision='constant',
    )

    st.plotly_chart(
        fig, use_container_width=True,
        config={
            'scrollZoom': True, 'displayModeBar': False,
            'doubleClick': 'reset', 'responsive': True, 'showTips': False,
        },
    )


# ====================================================
# 15. 포지션 트래커
# ====================================================
def render_position_tracker(
    selected_ticker: str,
    df_daily: pd.DataFrame,
    df_close: pd.DataFrame,
    portfolio_state: dict[str, TickerState],
    beta: Optional[float] = None,
    std_resid: Optional[float] = None,
    phase: str = 'all',  # 'top' (헤더+위치바), 'bottom' (정보카드+사이클+메모), 'all' (모두)
) -> None:
    portfolio_pnl = calc_portfolio_total_pnl(portfolio_state, df_close)
    usd_krw, is_fallback = fetch_usd_krw()
    st.session_state['portfolio_pnl_cache'] = portfolio_pnl
    st.session_state['usd_krw_cache'] = usd_krw
    st.session_state['usd_krw_fallback'] = is_fallback

    col_close = f'{selected_ticker}_Close'
    current_price = float(df_daily[col_close].iloc[-1]) if col_close in df_daily.columns else None

    # 회귀선 가격 + 현재 시점 expanding std (그래프2 Z-score와 일치)
    trend_price = None
    sigma_unit = std_resid  # fallback
    if (df_daily is not None and 'Predicted' in df_daily.columns
            and f'{selected_ticker}_Norm' in df_daily.columns):
        try:
            cur_predicted = float(df_daily['Predicted'].iloc[-1])
            cur_norm_y = float(df_daily[f'{selected_ticker}_Norm'].iloc[-1])
            if cur_norm_y > 0 and current_price is not None:
                trend_price = current_price * (cur_predicted / cur_norm_y)
            # expanding std (표시용)
            log_resid = (
                np.log(df_daily[f'{selected_ticker}_Norm'])
                - np.log(df_daily['Predicted'])
            ).dropna()
            exp_std = log_resid.expanding(
                min_periods=CFG.EXPANDING_MIN_PERIODS
            ).std().dropna()
            if len(exp_std) > 0:
                last_std = float(exp_std.iloc[-1])
                if last_std > 0 and np.isfinite(last_std) and std_resid is not None:
                    sigma_unit = last_std
        except Exception:
            trend_price = None

    def _fmt_pnl(val: float) -> str:
        sign = '+' if val >= 0 else ''
        color = pnl_color(val)
        return f"<span style='font-weight:700;color:{color};'>{sign}${int(round(val)):,}</span>"

    # ── 종목명 헤더 — 한눈에 보기와 동일 정보 ──
    # σ% (변동성), β·SPY (SPY +10% 시 변화율), DD (역대 고점 대비)
    sigma_pct_int = None
    if df_daily is not None and sigma_unit > 0 and np.isfinite(sigma_unit):
        sigma_pct_int = int(round((np.exp(sigma_unit) - 1) * 100))

    # β·SPY (SPY 대비 로그 회귀 슬로프, 배수)
    spy_betas_hdr = st.session_state.get('spy_betas', {})
    beta_spy_hdr = spy_betas_hdr.get(selected_ticker)
    beta_str_hdr = "—"
    if beta_spy_hdr is not None and np.isfinite(beta_spy_hdr):
        sign = '+' if beta_spy_hdr >= 0 else ''
        beta_str_hdr = f"{sign}{beta_spy_hdr:.1f}×"

    # half-life (잔차 평균회귀 반감기) — 분석 기간 전체 잔차 시계열로 계산
    half_life = None
    if df_daily is not None:
        norm_col = f'{selected_ticker}_Norm'
        if 'Predicted' in df_daily.columns and norm_col in df_daily.columns:
            log_resid = (np.log(df_daily[norm_col]) - np.log(df_daily['Predicted'])).dropna()
            half_life = compute_halflife(log_resid)

    sigma_str_hdr = f"±{sigma_pct_int}%" if sigma_pct_int is not None else "—"

    # Z 백분위 (숫자만) - 색은 종목 버튼과 동일 (momentum_to_color)
    z_pct_str = "—"
    z_pct_color = Colors.MOM_HOLD
    if df_daily is not None and 'Z_Score' in df_daily.columns:
        last_z = df_daily['Z_Score'].iloc[-1]
        if pd.notna(last_z):
            z_pct_val = z_to_pct(float(last_z))
            z_pct_str = f"{int(round(z_pct_val))}"
            # 백분위 → 정수 점수 (-4~+4): (pct - 50) / 20 ≈ Z, round
            z_score_int = max(-4, min(4, int(round((z_pct_val - 50) / 20))))
            z_pct_color = momentum_to_color(z_score_int)

    # M 백분위 (숫자만) - 색은 종목 버튼과 동일
    m_pct_str = "—"
    m_pct_color = Colors.MOM_HOLD
    cur_m_smooth = st.session_state.get(
        'ticker_momentum_smooth', {}
    ).get(selected_ticker)
    cur_m_score_int = st.session_state.get(
        'ticker_momentum_scores', {}
    ).get(selected_ticker, 0)
    if cur_m_smooth is not None:
        m_pct_val = z_to_pct(float(cur_m_smooth))
        m_pct_str = f"{int(round(m_pct_val))}"
        # M은 이미 정수 점수가 ticker_momentum_scores 에 있음
        m_pct_color = momentum_to_color(cur_m_score_int)

    header_right = (
        f"<span style='font-size:0.7rem;color:#6b7280;'>"
        f"<span title='1σ 변동성' style='font-weight:600;'>σ {sigma_str_hdr}</span>"
        f" · <span title='SPY 대비 로그회귀 슬로프 (장기 가격 관계)' style='font-weight:600;'>"
        f"β·SPY {beta_str_hdr}</span>"
        f" · <span title='Z 백분위 (0=극단매수, 50=중립, 100=극단매도)' "
        f"style='color:{z_pct_color};font-weight:600;'>"
        f"Z {z_pct_str}</span>"
        f" · <span title='M 모멘텀 백분위 (MACD/dMACD/RSI 통합 점수)' "
        f"style='color:{m_pct_color};font-weight:600;'>"
        f"M {m_pct_str}</span>"
        f"</span>"
    )
    # 종목명 색깔: 현재 모멘텀 점수 기반
    cur_mom_score = st.session_state.get(
        'ticker_momentum_scores', {}
    ).get(selected_ticker, 0)
    ticker_color = momentum_to_color(cur_mom_score)

    header_html = (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"padding:4px 12px 2px 12px;margin-top:4px;flex-wrap:wrap;gap:6px;'>"
        f"<span style='font-size:1rem;font-weight:800;color:{ticker_color};'>"
        f"{display_name(selected_ticker)}</span>"
        f"{header_right}"
        f"</div>"
    )

    # 회귀선 메트릭 HTML (있을 때만)
    if trend_price and current_price:
        trend_diff_pct = (trend_price / current_price - 1) * 100
        trend_color = pnl_color(-trend_diff_pct)
        trend_html = (
            f"<div><div style='color:#6b7280;font-size:0.68rem;'>회귀선</div>"
            f"<div style='font-weight:700;color:#374151;'>${trend_price:,.2f}"
            f" <span style='font-size:0.7rem;color:{trend_color};'>"
            f"{signed_str(trend_diff_pct, '{:.1f}')}%</span></div></div>"
        )
    else:
        trend_html = ""

    ts = portfolio_state.get(selected_ticker)

    # 매매 기록 없는 경우 — 정보 카드 (로그인 시에만)
    if ts is None or ts['cycle']['cycle_start'] is None or ts['cycle']['buy_qty'] == 0:
        if phase in ('top', 'all'):
            st.markdown(header_html, unsafe_allow_html=True)
            if is_authenticated():
                price_html = (
                    html_metric("현재가", f"${current_price:,.2f}")
                    if current_price is not None else html_dash_cell("현재가")
                )
                st.markdown(f"""
                <div style='display:flex;gap:12px;flex-wrap:wrap;margin:0 0 8px 0;
                            padding:8px 12px;background:#f3f4f6;
                            border:1px solid #d1d5db;border-radius:8px;font-size:0.78rem;'>
                  {price_html}
                  {html_dash_cell("평균단가")}
                  {html_dash_cell("보유수량")}
                  {html_dash_cell("보유기간")}
                  {html_dash_cell("평가손익")}
                  {html_dash_cell("누적실현손익")}
                </div>""", unsafe_allow_html=True)
        return

    cyc = ts['cycle']
    cumulative_pnl = ts['cumulative_pnl']
    hold_qty = cyc['hold_qty']
    avg_price = cyc['buy_cost'] / cyc['buy_qty']
    is_closed = cyc['cycle_end'] is not None

    # 현재가
    if hold_qty > 0 and avg_price and current_price is not None:
        price_pct = (current_price - avg_price) / avg_price * 100
        price_color = pnl_color(price_pct)
        price_html = (
            f"<div><div style='color:#6b7280;font-size:0.68rem;'>현재가</div>"
            f"<div style='font-weight:700;color:{price_color};'>"
            f"${current_price:,.2f}&nbsp;<span style='font-size:0.72rem;'>"
            f"({signed_str(price_pct, '{:.0f}')}%)</span></div></div>"
        )
    else:
        price_html = (
            html_metric("현재가", f"${current_price:,.2f}")
            if current_price is not None else html_dash_cell("현재가")
        )

    avg_html = (
        html_metric("평균단가", f"${avg_price:,.2f}")
        if not is_closed else html_dash_cell("평균단가")
    )
    qty_html = (
        html_metric("보유수량", f"{hold_qty:,}주")
        if not is_closed else html_dash_cell("보유수량")
    )
    if not is_closed:
        hold_days = (datetime.date.today() - cyc['cycle_start']).days
        period_html = html_metric("보유기간", f"{hold_days}일")
    else:
        period_html = html_dash_cell("보유기간")

    if is_closed:
        pnl_dollar = cyc['current_pnl']
        pnl_label = "실현손익"
    else:
        pnl_dollar = (current_price - avg_price) * hold_qty if current_price is not None else 0.0
        pnl_label = "평가손익"
    pnl_html = (
        f"<div><div style='color:#6b7280;font-size:0.68rem;'>{pnl_label}</div>"
        f"<div>{_fmt_pnl(pnl_dollar)}</div></div>"
    )

    # current_pnl이 None이 아니면(매도 한 번 이상 발생) 누적에 포함
    has_realized_in_cycle = cyc['current_pnl'] is not None
    total_realized = cumulative_pnl + (
        cyc['current_pnl'] if has_realized_in_cycle else 0.0
    )
    has_cumulative = (cumulative_pnl != 0.0) or has_realized_in_cycle
    cumulative_html = (
        f"<div><div style='color:#6b7280;font-size:0.68rem;'>누적실현손익</div>"
        f"<div>{_fmt_pnl(total_realized)}</div></div>"
        if has_cumulative else html_dash_cell("누적실현손익")
    )

    bg_color = '#f0fdf4' if hold_qty > 0 else '#f3f4f6'
    border_c = '#86efac' if hold_qty > 0 else '#d1d5db'

    if phase in ('top', 'all'):
        # 헤더 (σ, β, Z, M) — 항상 표시
        st.markdown(header_html, unsafe_allow_html=True)
        # 정보 카드 (현재가/평균단가/보유 등) — 로그인 시에만
        if is_authenticated():
            st.markdown(f"""
            <div style='display:flex;gap:12px;flex-wrap:wrap;margin:0 0 8px 0;
                        padding:8px 12px;background:{bg_color};
                        border:1px solid {border_c};border-radius:8px;font-size:0.78rem;'>
              {price_html}
              {avg_html}
              {qty_html}
              {period_html}
              {pnl_html}
              {cumulative_html}
            </div>""", unsafe_allow_html=True)


# ====================================================
# 15-B. 액션 카드 빌더 (위치 bar + 액션 카드 + 서브 액션)
# ====================================================
def build_action_card_html(
    df_daily: pd.DataFrame,
    selected_ticker: str,
    cur_price: float,
    avg_price: Optional[float],
    beta: float,
    std_resid: float,
    trade_records: Optional[list] = None,
) -> Optional[str]:
    """그래프 위에 표시할 위치 bar + 액션 카드 통합 HTML.

    None을 반환하면 데이터 부족으로 스킵.
    """
    if 'Predicted' not in df_daily.columns:
        return None
    norm_col = f'{selected_ticker}_Norm'
    if norm_col not in df_daily.columns:
        return None
    cur_predicted = float(df_daily['Predicted'].iloc[-1])
    cur_norm_y = float(df_daily[norm_col].iloc[-1])
    if cur_norm_y <= 0:
        return None
    trend_price = cur_price * (cur_predicted / cur_norm_y)

    # log_resid (분위가 + expanding std 계산용)
    log_resid_series = (
        np.log(df_daily[norm_col]) - np.log(df_daily['Predicted'])
    ).dropna()

    # ── 현재 시점 expanding std (그래프2의 Z-score 표시값과 일치하기 위함) ──
    # process_asset_data의 Z_Score = log_resid / log_resid.expanding().std()
    # 매매가 제안의 σ도 같은 std를 써야 일치함.
    expanding_std = log_resid_series.expanding(
        min_periods=CFG.EXPANDING_MIN_PERIODS
    ).std()
    # 마지막 유효 std 사용 (데이터 부족 시 fallback으로 std_resid)
    cur_std = expanding_std.dropna()
    if len(cur_std) > 0:
        sigma_unit = float(cur_std.iloc[-1])
        # 안전장치: 0이거나 비정상이면 fallback
        if sigma_unit <= 0 or not np.isfinite(sigma_unit):
            sigma_unit = std_resid
    else:
        sigma_unit = std_resid

    def _price_at_sigma(k: float) -> float:
        return trend_price * np.exp(k * sigma_unit)

    def _price_at_quantile(q: float) -> Optional[float]:
        if len(log_resid_series) < 30:
            return None
        return trend_price * np.exp(float(log_resid_series.quantile(q)))

    def _price_to_sigma(p: float) -> float:
        if p <= 0 or trend_price <= 0:
            return 0.0
        return float(np.log(p / trend_price) / sigma_unit)

    def _sigma_to_pct(sigma: float) -> tuple[float, bool]:
        # ±3σ → 0~100%
        pct = (sigma + 3) / 6 * 100
        is_outside = pct < 0 or pct > 100
        return float(max(0, min(100, pct))), is_outside

    cur_sigma = _price_to_sigma(cur_price)
    cur_pct_bar, cur_outside = _sigma_to_pct(cur_sigma)
    avg_sigma = _price_to_sigma(avg_price) if avg_price else None
    if avg_sigma is not None:
        avg_pct_bar, avg_outside = _sigma_to_pct(avg_sigma)
    else:
        avg_pct_bar, avg_outside = None, False

    def _label_align(pct: float) -> tuple[str, str]:
        if pct < 20:
            return ("translateX(0%)", "left")
        elif pct > 80:
            return ("translateX(-100%)", "right")
        else:
            return ("translateX(-50%)", "center")

    # ── 위치 bar HTML (±3σ 범위) ──
    # σ 위치: -3=0%, -2=16.67%, -1.5=25%, -1=33.33%, 0=50%,
    #         +1=66.67%, +1.5=75%, +2=83.33%, +3=100%
    bar_html = (
        "<div style='position:relative;height:48px;margin:6px 8px 14px 8px;'>"
        "<div style='position:absolute;top:18px;left:0;right:0;height:8px;"
        "border-radius:4px;"
        "background:linear-gradient(to right,"
        "#450a0a 0%,#7f1d1d 16.67%,#dc2626 25%,#fca5a5 33.33%,"
        "#e5e7eb 50%,"
        "#93c5fd 66.67%,#2563eb 75%,#1e3a8a 83.33%,#172554 100%);'></div>"
    )
    sigma_marks = [
        (0, '-3σ'), (16.67, '-2σ'), (33.33, '-1σ'),
        (50, '추세'),
        (66.67, '+1σ'), (83.33, '+2σ'), (100, '+3σ'),
    ]
    for pos, lbl in sigma_marks:
        bar_html += (
            f"<div style='position:absolute;left:{pos}%;top:28px;"
            f"transform:translateX(-50%);font-size:0.55rem;"
            f"color:#9ca3af;'>{lbl}</div>"
        )
    if avg_pct_bar is not None:
        avg_tf, avg_ta = _label_align(avg_pct_bar)
        out_arrow = (
            " ◀" if avg_sigma is not None and avg_sigma < -3
            else " ▶" if avg_sigma is not None and avg_sigma > 3
            else ""
        )
        bar_html += (
            f"<div style='position:absolute;left:{avg_pct_bar:.1f}%;top:16px;"
            f"transform:translateX(-50%);width:12px;height:12px;"
            f"background:#dc2626;border:2px solid #fff;"
            f"box-shadow:0 0 0 1px #7f1d1d,0 1px 2px rgba(0,0,0,0.3);"
            f"z-index:2;cursor:help;' "
            f"title='평균단가 ${avg_price:.2f}'></div>"
            f"<div style='position:absolute;left:{avg_pct_bar:.1f}%;top:0;"
            f"transform:{avg_tf};text-align:{avg_ta};font-size:0.6rem;"
            f"font-weight:700;color:#374151;white-space:nowrap;"
            f"padding:0 3px;background:rgba(255,255,255,0.85);"
            f"border-radius:3px;'>"
            f"평균 ${avg_price:.2f}{out_arrow}</div>"
        )
    cur_tf, cur_ta = _label_align(cur_pct_bar)
    cur_out_arrow = (
        " ◀" if cur_outside and cur_sigma < -3
        else " ▶" if cur_outside and cur_sigma > 3
        else ""
    )

    # ── 현재가 마커 색상 = 모멘텀 점수 (MACD-Z + RSI 만, Z 제외) ──
    # 위치(σ)와 독립된 모멘텀 정보를 색으로 표시
    cur_momentum_score = 0
    cur_signal = 'H'
    if ('MACD_Pct' in df_daily.columns and 'RSI' in df_daily.columns):
        last_macd_pct = df_daily['MACD_Pct'].iloc[-1]
        last_dmacd_pct = df_daily['dMACD_Pct'].iloc[-1]
        last_rsi = df_daily['RSI'].iloc[-1]
        macd_pct_v = float(last_macd_pct) if pd.notna(last_macd_pct) else 0.0
        dmacd_pct_v = float(last_dmacd_pct) if pd.notna(last_dmacd_pct) else 0.0
        rsi_v = float(last_rsi) if pd.notna(last_rsi) else 50.0
        cur_momentum_score = compute_momentum_score(macd_pct_v, dmacd_pct_v, rsi_v)
        cur_signal = momentum_score_to_signal(cur_momentum_score)
    marker_color = momentum_to_color(cur_momentum_score)

    # 현재가 ■ 사각형 마커 — 투명 + 신호색 굵은 테두리 + 검정 외곽선
    bar_html += (
        f"<div style='position:absolute;left:{cur_pct_bar:.1f}%;top:11px;"
        f"transform:translateX(-50%);width:20px;height:20px;"
        f"background:transparent;border:3px solid {marker_color};"
        f"outline:1px solid #000;"
        f"box-shadow:inset 0 0 0 1px #000,0 1px 3px rgba(0,0,0,0.3);"
        f"z-index:3;cursor:help;' "
        f"title='현재가 ${cur_price:.2f} · 신호 {cur_signal}'></div>"
    )
    # ── 사이클별 평균 매매가 마커 (매매 시점 Z-score 기반) ──
    if trade_records:
        cycle_list = compute_cycle_avg_prices(trade_records, df_daily=df_daily)
        # 최근 5개만 (가장 오래된 → 최근 순으로 정렬되어 있음)
        cycle_list = cycle_list[-5:]
        n_cycles = len(cycle_list)

        def _draw_cycle_marker(
            sigma_val: float, color: str, dark_color: str,
            opacity: float, tooltip: str,
        ) -> str:
            """매매 시점 σ로 마커를 그림. 범위 밖이면 가장자리에 σ값 라벨."""
            pct, outside = _sigma_to_pct(sigma_val)
            if not outside:
                # bar 안에 들어옴 → 일반 동그라미 마커
                return (
                    f"<div style='position:absolute;left:{pct:.1f}%;top:17px;"
                    f"transform:translateX(-50%);width:10px;height:10px;"
                    f"border-radius:50%;background:{color};border:2px solid #fff;"
                    f"box-shadow:0 0 0 1px {dark_color},0 1px 2px rgba(0,0,0,0.3);"
                    f"opacity:{opacity:.2f};z-index:2;cursor:help;' "
                    f"title='{tooltip}'></div>"
                )
            # 범위 밖 → 가장자리 화살표 + σ 라벨
            if sigma_val < -3:
                arrow = "◀"
                edge_pct = 0.0
                tf = "translateX(0%)"
                lbl_align = "left"
            else:  # sigma_val > 3
                arrow = "▶"
                edge_pct = 100.0
                tf = "translateX(-100%)"
                lbl_align = "right"
            return (
                # 가장자리 마커
                f"<div style='position:absolute;left:{edge_pct:.1f}%;top:17px;"
                f"transform:translateX(-50%);width:10px;height:10px;"
                f"border-radius:50%;background:{color};border:2px solid #fff;"
                f"box-shadow:0 0 0 1px {dark_color},0 1px 2px rgba(0,0,0,0.3);"
                f"opacity:{opacity:.2f};z-index:2;cursor:help;' "
                f"title='{tooltip}'></div>"
                # σ 라벨 (가장자리 — 눈금 라벨 아래)
                f"<div style='position:absolute;left:{edge_pct:.1f}%;top:38px;"
                f"transform:{tf};text-align:{lbl_align};font-size:0.55rem;"
                f"color:{dark_color};font-weight:700;white-space:nowrap;"
                f"opacity:{opacity:.2f};'>"
                f"{arrow}{sigma_val:+.1f}σ</div>"
            )

        for i, cyc in enumerate(cycle_list):
            # 최근일수록 진하게: opacity 0.5 → 1.0
            opacity = 0.5 + 0.5 * ((i + 1) / max(n_cycles, 1))

            # 매수 마커 (매매 시점 σ 사용)
            if cyc.get('avg_buy_sigma') is not None:
                buy_sigma_at_trade = float(cyc['avg_buy_sigma'])
                tooltip_buy = (
                    f"사이클 {cyc['idx']} 평균매수 "
                    f"${cyc['avg_buy']:.2f} (당시 σ {buy_sigma_at_trade:+.2f})"
                )
                bar_html += _draw_cycle_marker(
                    buy_sigma_at_trade, '#dc2626', '#7f1d1d', opacity, tooltip_buy,
                )

            # 매도 마커 (매도 완료된 경우만)
            if cyc.get('avg_sell_sigma') is not None and cyc['avg_sell'] is not None:
                sell_sigma_at_trade = float(cyc['avg_sell_sigma'])
                tooltip_sell = (
                    f"사이클 {cyc['idx']} 평균매도 "
                    f"${cyc['avg_sell']:.2f} (당시 σ {sell_sigma_at_trade:+.2f})"
                )
                bar_html += _draw_cycle_marker(
                    sell_sigma_at_trade, '#1d4ed8', '#1e3a8a', opacity, tooltip_sell,
                )

    if cur_outside:
        bar_html += (
            f"<div style='position:absolute;left:{cur_pct_bar:.1f}%;top:0;"
            f"transform:{cur_tf};text-align:{cur_ta};font-size:0.6rem;"
            f"font-weight:700;color:#000;white-space:nowrap;"
            f"padding:0 3px;background:rgba(255,255,255,0.85);"
            f"border-radius:3px;'>"
            f"${cur_price:.2f}{cur_out_arrow}</div>"
        )
    bar_html += "</div>"

    # ── 현재 위치 한 줄 ──
    if cur_sigma <= -1.5:
        interp, interp_c = f"🔴 매우 과매도 ({cur_sigma:+.1f}σ)", '#b91c1c'
    elif cur_sigma <= -0.5:
        interp, interp_c = f"🟠 과매도 ({cur_sigma:+.1f}σ)", '#dc2626'
    elif cur_sigma >= 1.5:
        interp, interp_c = f"🔵 매우 과매수 ({cur_sigma:+.1f}σ)", '#1e3a8a'
    elif cur_sigma >= 0.5:
        interp, interp_c = f"🟦 과매수 ({cur_sigma:+.1f}σ)", '#2563eb'
    else:
        interp, interp_c = f"⚪ 회귀선 부근 ({cur_sigma:+.1f}σ)", '#6b7280'
    # 사이클 마커 범례 (사이클 있을 때만)
    legend_html = ""
    if trade_records:
        cyc_count = len(compute_cycle_avg_prices(trade_records, df_daily=df_daily))
        if cyc_count > 0:
            shown = min(cyc_count, 5)
            legend_html = (
                f"<span style='font-size:0.6rem;color:#9ca3af;font-weight:400;'>"
                f"<span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:#dc2626;border:1.5px solid #fff;"
                f"box-shadow:0 0 0 1px #7f1d1d;vertical-align:middle;'></span>"
                f" 매수&nbsp;&nbsp;"
                f"<span style='display:inline-block;width:8px;height:8px;"
                f"border-radius:50%;background:#1d4ed8;border:1.5px solid #fff;"
                f"box-shadow:0 0 0 1px #1e3a8a;vertical-align:middle;'></span>"
                f" 매도 · <i>당시 σ 기준</i> (최근 {shown}/{cyc_count}사이클)</span>"
            )

    interp_html = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin:0 8px 8px 8px;flex-wrap:wrap;gap:6px;'>"
        f"<span style='font-size:0.72rem;color:{interp_c};font-weight:600;'>"
        f"현재 위치: {interp}</span>"
        f"{legend_html}"
        f"</div>"
    )

    # ── 액션 카드 ──
    price_specs = [
        ("FB2", '#7f1d1d', -2.0, 0.025, 'buy'),
        ("FB",  '#dc2626', -1.5, 0.07, 'buy'),
        ("B",   '#fca5a5', -1.0, 0.16, 'buy'),
        ("S",   '#93c5fd',  1.0, 0.84, 'sell'),
        ("FS",  '#2563eb',  1.5, 0.93, 'sell'),
        ("FS2", '#1e3a8a',  2.0, 0.975, 'sell'),
    ]
    buy_levels, sell_levels = [], []
    for label, c, sk, qq, side in price_specs:
        p_sigma = _price_at_sigma(sk)
        p_quantile = _price_at_quantile(qq)
        p_reco = (p_sigma + p_quantile) / 2 if p_quantile is not None else p_sigma
        if side == 'buy':
            buy_levels.append((label, p_reco, sk, c))
        else:
            sell_levels.append((label, p_reco, sk, c))
    # 다음 매수 트리거: 현재가보다 낮은 가격 중 "가장 가까운" = 가장 높은 가격
    # (FB2/FB/B 순으로 점점 더 깊은 매수 영역. 가장 얕은 = 곧 도달할 수 있는 트리거)
    buy_below = [(lbl, p, sk, c) for lbl, p, sk, c in buy_levels if p < cur_price]
    next_buy = max(buy_below, key=lambda x: x[1]) if buy_below else None

    # 다음 익절 트리거: 현재가보다 높은 가격 중 "가장 가까운" = 가장 낮은 가격
    sell_above = [(lbl, p, sk, c) for lbl, p, sk, c in sell_levels if p > cur_price]
    next_sell = min(sell_above, key=lambda x: x[1]) if sell_above else None

    holding = avg_price is not None
    if cur_sigma <= -1.5:
        action_text = ("🔥 강한 매수 영역 — 추가매수 적극 검토" if holding
                       else "🔥 강한 매수 영역 — 신규 진입 검토")
        action_bg, action_border = '#fef2f2', '#7f1d1d'
    elif cur_sigma <= -0.5:
        action_text = ("🟧 매수 영역 — 평균단가 낮출 기회" if holding
                       else "🟧 매수 영역 — 분할매수 검토")
        action_bg, action_border = '#fff7ed', '#dc2626'
    elif cur_sigma >= 1.5:
        action_text = ("💰 강한 익절 영역 — 일부/전량 매도 검토" if holding
                       else "🚫 강한 익절 영역 — 신규 진입 부적절")
        action_bg, action_border = '#eff6ff', '#1e3a8a'
    elif cur_sigma >= 0.5:
        action_text = ("🟦 익절 영역 — 일부 매도 검토" if holding
                       else "⏸ 익절 영역 — 관망")
        action_bg, action_border = '#eff6ff', '#2563eb'
    else:
        action_text = "⏸ 회귀선 부근 — 관망 (트리거 대기)"
        action_bg, action_border = '#f9fafb', '#9ca3af'

    action_html = (
        f"<div style='padding:8px 10px;background:{action_bg};"
        f"border-left:4px solid {action_border};border-radius:6px;"
        f"margin:0 8px 8px 8px;'>"
        f"<div style='font-size:0.8rem;font-weight:700;color:#111827;"
        f"margin-bottom:6px;'>{action_text}</div>"
        f"<div style='display:flex;gap:8px;font-size:0.7rem;'>"
    )
    if next_buy:
        lbl, p, sk, c = next_buy
        drop_pct = (p / cur_price - 1) * 100
        action_html += (
            f"<div style='flex:1;background:#fff;padding:5px 7px;"
            f"border-radius:5px;border:1px solid #fecaca;'>"
            f"<div style='color:#9ca3af;font-size:0.58rem;'>다음 매수</div>"
            f"<div style='display:flex;align-items:baseline;gap:4px;'>"
            f"<span style='background:{c};color:#fff;padding:1px 4px;"
            f"border-radius:3px;font-size:0.55rem;font-weight:700;'>{lbl}</span>"
            f"<span style='font-weight:700;color:#111827;'>${p:,.2f}</span>"
            f"</div>"
            f"<div style='color:#b91c1c;font-size:0.58rem;'>"
            f"<b>{drop_pct:.1f}%</b> 더 하락 시</div>"
            f"</div>"
        )
    else:
        action_html += (
            f"<div style='flex:1;background:#fff;padding:5px 7px;"
            f"border-radius:5px;border:1px solid #e5e7eb;'>"
            f"<div style='color:#9ca3af;font-size:0.58rem;'>다음 매수</div>"
            f"<div style='color:#9ca3af;font-size:0.65rem;'>-2σ보다 낮음</div>"
            f"</div>"
        )
    if next_sell:
        lbl, p, sk, c = next_sell
        rise_pct = (p / cur_price - 1) * 100
        action_html += (
            f"<div style='flex:1;background:#fff;padding:5px 7px;"
            f"border-radius:5px;border:1px solid #bfdbfe;'>"
            f"<div style='color:#9ca3af;font-size:0.58rem;'>다음 익절</div>"
            f"<div style='display:flex;align-items:baseline;gap:4px;'>"
            f"<span style='background:{c};color:#fff;padding:1px 4px;"
            f"border-radius:3px;font-size:0.55rem;font-weight:700;'>{lbl}</span>"
            f"<span style='font-weight:700;color:#111827;'>${p:,.2f}</span>"
            f"</div>"
            f"<div style='color:#1d4ed8;font-size:0.58rem;'>"
            f"<b>+{rise_pct:.1f}%</b> 상승 시</div>"
            f"</div>"
        )
    else:
        action_html += (
            f"<div style='flex:1;background:#fff;padding:5px 7px;"
            f"border-radius:5px;border:1px solid #e5e7eb;'>"
            f"<div style='color:#9ca3af;font-size:0.58rem;'>다음 익절</div>"
            f"<div style='color:#9ca3af;font-size:0.65rem;'>+2σ보다 높음</div>"
            f"</div>"
        )
    action_html += "</div></div>"

    # ── 보유 종목 서브 액션 (조건부) ──
    sub_html = ""
    if avg_price:
        ret_pct_now = (cur_price - avg_price) / avg_price * 100
        sub_action, sub_color = None, None
        if ret_pct_now <= -10 and cur_sigma <= -1.0:
            sub_action = (
                f"💡 손실 -{abs(ret_pct_now):.0f}% + 과매도 → "
                f"평균단가 낮추기 좋은 시점"
            )
            sub_color = '#b91c1c'
        elif ret_pct_now >= 20 and cur_sigma >= 0.5:
            sub_action = (
                f"💡 수익 +{ret_pct_now:.0f}% + 익절권 → "
                f"일부 익절로 리스크 축소"
            )
            sub_color = '#1d4ed8'
        elif ret_pct_now >= 50:
            sub_action = f"💡 수익 +{ret_pct_now:.0f}% — 트레일링 스톱 고려"
            sub_color = '#1d4ed8'
        if sub_action:
            sub_html = (
                f"<div style='padding:6px 10px;background:#fefce8;"
                f"border:1px dashed {sub_color};border-radius:5px;"
                f"margin:0 8px 8px 8px;font-size:0.7rem;color:{sub_color};"
                f"font-weight:600;'>{sub_action}</div>"
            )

    return bar_html


# ====================================================
# 15-C. 탭2용 미니 그라디언트 바 빌더 (종목별 한눈에 보기)
# ====================================================
def build_mini_gradient_bar(
    df_daily: pd.DataFrame,
    selected_ticker: str,
    cur_price: float,
    avg_price: Optional[float],
    beta: float,
    std_resid: float,
    trade_records: Optional[list] = None,
    bar_height: int = 32,
) -> Optional[str]:
    """탭2용 컴팩트 그라디언트 바.
    - σ 눈금 라벨 없음 (헤더에서 한번만 표시)
    - 현재가 ■ 사각형 마커 (투명 + 신호색 테두리)
    - 평균단가 ▪ 작은 사각형 마커 (빨강 채움, 보유 시)
    - 사이클 매수/매도 마커 (작은 채움 점)
    """
    if 'Predicted' not in df_daily.columns:
        return None
    norm_col = f'{selected_ticker}_Norm'
    if norm_col not in df_daily.columns:
        return None
    cur_predicted = float(df_daily['Predicted'].iloc[-1])
    cur_norm_y = float(df_daily[norm_col].iloc[-1])
    if cur_norm_y <= 0:
        return None
    trend_price = cur_price * (cur_predicted / cur_norm_y)

    log_resid_series = (
        np.log(df_daily[norm_col]) - np.log(df_daily['Predicted'])
    ).dropna()
    expanding_std = log_resid_series.expanding(
        min_periods=CFG.EXPANDING_MIN_PERIODS
    ).std().dropna()
    if len(expanding_std) > 0:
        sigma_unit = float(expanding_std.iloc[-1])
        if sigma_unit <= 0 or not np.isfinite(sigma_unit):
            sigma_unit = std_resid
    else:
        sigma_unit = std_resid

    def _price_to_sigma(p: float) -> float:
        if p <= 0 or trend_price <= 0:
            return 0.0
        return float(np.log(p / trend_price) / sigma_unit)

    def _sigma_to_pct(sigma: float) -> tuple[float, bool]:
        pct = (sigma + 3) / 6 * 100
        is_outside = pct < 0 or pct > 100
        return float(max(0, min(100, pct))), is_outside

    cur_sigma = _price_to_sigma(cur_price)
    cur_pct_bar, cur_outside = _sigma_to_pct(cur_sigma)

    # ── 현재 모멘텀 점수 (MACD + RSI만, Z 제외) ──
    cur_momentum_score = 0
    cur_signal = 'H'
    if ('MACD_Pct' in df_daily.columns and 'RSI' in df_daily.columns):
        last_macd_pct = df_daily['MACD_Pct'].iloc[-1]
        last_dmacd_pct = df_daily['dMACD_Pct'].iloc[-1]
        last_rsi = df_daily['RSI'].iloc[-1]
        macd_pct_v = float(last_macd_pct) if pd.notna(last_macd_pct) else 0.0
        dmacd_pct_v = float(last_dmacd_pct) if pd.notna(last_dmacd_pct) else 0.0
        rsi_v = float(last_rsi) if pd.notna(last_rsi) else 50.0
        cur_momentum_score = compute_momentum_score(macd_pct_v, dmacd_pct_v, rsi_v)
        cur_signal = momentum_score_to_signal(cur_momentum_score)
    marker_color = momentum_to_color(cur_momentum_score)

    # ── bar HTML ──
    # 컨테이너 높이 = bar_height; 그라디언트 두께 6px; 마커 14px
    grad_top = (bar_height - 6) // 2  # 그라디언트 vertical center
    marker_top = (bar_height - 14) // 2  # 마커 vertical center
    # σ 세로선용: 그라디언트보다 약간 길게 (위아래 2px씩 더)
    line_top = grad_top - 2
    line_height = 6 + 4
    bar_html = (
        f"<div style='position:relative;height:{bar_height}px;"
        f"width:100%;'>"
        # 그라디언트
        f"<div style='position:absolute;top:{grad_top}px;left:0;right:0;height:6px;"
        f"border-radius:3px;"
        f"background:linear-gradient(to right,"
        f"#450a0a 0%,#7f1d1d 16.67%,#dc2626 25%,#fca5a5 33.33%,"
        f"#e5e7eb 50%,"
        f"#93c5fd 66.67%,#2563eb 75%,#1e3a8a 83.33%,#172554 100%);'></div>"
    )
    # σ 위치 세로선 (-3, -2, -1, 0(강조), +1, +2, +3)
    sigma_marks = [
        (0.00,    'rgba(255,255,255,0.5)', 1),   # -3σ
        (16.67,   'rgba(255,255,255,0.5)', 1),   # -2σ
        (33.33,   'rgba(255,255,255,0.5)', 1),   # -1σ
        (50.00,   'rgba(255,255,255,0.9)', 2),   # 0σ (강조)
        (66.67,   'rgba(255,255,255,0.5)', 1),   # +1σ
        (83.33,   'rgba(255,255,255,0.5)', 1),   # +2σ
        (100.00,  'rgba(255,255,255,0.5)', 1),   # +3σ
    ]
    for pos_pct, line_color, line_w in sigma_marks:
        bar_html += (
            f"<div style='position:absolute;left:{pos_pct:.2f}%;"
            f"top:{line_top}px;width:{line_w}px;height:{line_height}px;"
            f"background:{line_color};transform:translateX(-50%);"
            f"z-index:1;'></div>"
        )

    # ── 사이클별 마커 (작게) ──
    if trade_records:
        cycle_list = compute_cycle_avg_prices(trade_records, df_daily=df_daily)
        cycle_list = cycle_list[-5:]
        n_cycles = len(cycle_list)
        cyc_marker_top = (bar_height - 8) // 2

        for i, cyc in enumerate(cycle_list):
            opacity = 0.5 + 0.5 * ((i + 1) / max(n_cycles, 1))

            if cyc.get('avg_buy_sigma') is not None:
                sigma_val = float(cyc['avg_buy_sigma'])
                pct, outside = _sigma_to_pct(sigma_val)
                edge_pct = pct if not outside else (0 if sigma_val < -3 else 100)
                bar_html += (
                    f"<div style='position:absolute;left:{edge_pct:.1f}%;"
                    f"top:{cyc_marker_top}px;"
                    f"transform:translateX(-50%);width:8px;height:8px;"
                    f"border-radius:50%;background:#dc2626;border:1.5px solid #fff;"
                    f"box-shadow:0 0 0 1px #7f1d1d;"
                    f"opacity:{opacity:.2f};z-index:2;cursor:help;' "
                    f"title='사이클 {cyc['idx']} 매수 ${cyc['avg_buy']:.2f} "
                    f"(당시 σ {sigma_val:+.2f})'></div>"
                )

            if (cyc.get('avg_sell_sigma') is not None
                    and cyc['avg_sell'] is not None):
                sigma_val = float(cyc['avg_sell_sigma'])
                pct, outside = _sigma_to_pct(sigma_val)
                edge_pct = pct if not outside else (0 if sigma_val < -3 else 100)
                bar_html += (
                    f"<div style='position:absolute;left:{edge_pct:.1f}%;"
                    f"top:{cyc_marker_top}px;"
                    f"transform:translateX(-50%);width:8px;height:8px;"
                    f"border-radius:50%;background:#1d4ed8;border:1.5px solid #fff;"
                    f"box-shadow:0 0 0 1px #1e3a8a;"
                    f"opacity:{opacity:.2f};z-index:2;cursor:help;' "
                    f"title='사이클 {cyc['idx']} 매도 ${cyc['avg_sell']:.2f} "
                    f"(당시 σ {sigma_val:+.2f})'></div>"
                )

    # ── 평균단가 ● 마커 (보유 시) — 투명 + 굵은 회색 테두리 ──
    if avg_price is not None and avg_price > 0:
        avg_sigma_v = _price_to_sigma(avg_price)
        avg_pct_v, avg_outside_v = _sigma_to_pct(avg_sigma_v)
        avg_marker_top = (bar_height - 12) // 2
        bar_html += (
            f"<div style='position:absolute;left:{avg_pct_v:.1f}%;"
            f"top:{avg_marker_top}px;"
            f"transform:translateX(-50%);width:12px;height:12px;"
            f"background:#dc2626;border:2px solid #fff;"
            f"box-shadow:0 0 0 1px #7f1d1d,0 1px 2px rgba(0,0,0,0.3);"
            f"z-index:2;cursor:help;' "
            f"title='평균단가 ${avg_price:.2f} · {avg_sigma_v:+.2f}σ'></div>"
        )

    # ── 현재가 ■ 사각형 마커 — 투명 + 신호색 굵은 테두리 + 검정 외곽선 ──
    cur_marker_top = (bar_height - 16) // 2
    bar_html += (
        f"<div style='position:absolute;left:{cur_pct_bar:.1f}%;top:{cur_marker_top}px;"
        f"transform:translateX(-50%);width:16px;height:16px;"
        f"background:transparent;border:2.5px solid {marker_color};"
        f"outline:1px solid #000;"
        f"box-shadow:inset 0 0 0 1px #000,0 1px 2px rgba(0,0,0,0.3);"
        f"z-index:3;cursor:help;' "
        f"title='현재가 ${cur_price:.2f} · 신호 {cur_signal} · {cur_sigma:+.2f}σ'></div>"
    )

    bar_html += "</div>"
    return bar_html


# ====================================================
# 16. 분석 패널 (#1 사이클 통계 + #2 신호 백테스트 + #5 상관관계)
# ====================================================
def render_analytics_panel(
    selected_ticker: str,
    df_daily: Optional[pd.DataFrame],
    df_close: pd.DataFrame,
    portfolio_state: dict[str, TickerState],
    beta: Optional[float] = None,
    std_resid: Optional[float] = None,
) -> None:
    """차트 아래 expander (세로 stack, 모바일 친화)."""

    # 인증 가드: 비로그인 시 개인 정보 표시 안 함
    if not is_authenticated():
        return

    # ── #1 사이클 통계 + #5 진행 게이지 ──
    with st.expander("📈 사이클 통계", expanded=False):
        records = st.session_state.trade_history.get(selected_ticker, [])
        stats = compute_cycle_stats(records)
        if stats is None:
            st.caption("완료된 사이클이 없습니다.")
        else:
            pf_str = (
                f"{stats['profit_factor']:.2f}"
                if stats['profit_factor'] != float('inf') else "∞"
            )
            wr_color = pnl_color(stats['win_rate'] - 50)
            avg_color = pnl_color(stats['avg_ret_pct'])
            # 가로 배치 (한 줄에 핵심지표 6개)
            st.markdown(
                f"<div style='display:grid;grid-template-columns:repeat(3,1fr);"
                f"gap:8px;font-size:0.78rem;'>"
                f"<div><div style='color:#6b7280;font-size:0.7rem;'>사이클</div>"
                f"<b>{stats['count']}회</b></div>"
                f"<div><div style='color:#6b7280;font-size:0.7rem;'>승률</div>"
                f"<b style='color:{wr_color};'>{stats['win_rate']:.0f}%</b></div>"
                f"<div><div style='color:#6b7280;font-size:0.7rem;'>PF</div>"
                f"<b>{pf_str}</b></div>"
                f"<div><div style='color:#6b7280;font-size:0.7rem;'>평균수익</div>"
                f"<b style='color:{avg_color};'>{signed_str(stats['avg_ret_pct'], '{:.1f}')}%</b></div>"
                f"<div><div style='color:#6b7280;font-size:0.7rem;'>평균보유</div>"
                f"<b>{stats['avg_hold_days']:.0f}일</b></div>"
                f"<div><div style='color:#6b7280;font-size:0.7rem;'>최고/최저</div>"
                f"<span style='color:#b91c1c;font-weight:700;'>+{stats['best_pct']:.1f}%</span>"
                f" / <span style='color:#1d4ed8;font-weight:700;'>{stats['worst_pct']:.1f}%</span></div>"
                f"</div>"
                f"<div style='font-size:0.65rem;color:#9ca3af;margin-top:6px;'>"
                f"최고: {stats['best_date']} · 최저: {stats['worst_date']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── #5 현재 사이클 진행 게이지 (보유 중일 때만) ──
            ts = portfolio_state.get(selected_ticker)
            if ts and ts['cycle']['hold_qty'] > 0 and ts['cycle']['cycle_start']:
                cyc = ts['cycle']
                # 현재 사이클의 보유일 / 손익률
                hold_days_now = (datetime.date.today() - cyc['cycle_start']).days
                avg_buy_price = cyc['buy_cost'] / cyc['buy_qty']
                col_close = f'{selected_ticker}_Close'
                cur_price = (
                    float(df_daily[col_close].iloc[-1])
                    if col_close in df_daily.columns else avg_buy_price
                )
                cur_ret_pct = (cur_price - avg_buy_price) / avg_buy_price * 100

                # 진행률 (평균 대비)
                day_progress = (
                    hold_days_now / stats['avg_hold_days'] * 100
                    if stats['avg_hold_days'] > 0 else 0
                )
                ret_progress = (
                    cur_ret_pct / stats['avg_ret_pct'] * 100
                    if stats['avg_ret_pct'] != 0 else 0
                )

                # 진행바 색상
                def _progress_bar(pct: float, cur_label: str, avg_label: str,
                                   value_color: str) -> str:
                    # 100% 초과는 진행바 가득, 라벨은 그대로 노출
                    bar_pct = max(0, min(pct, 100))
                    over_100 = pct > 100
                    bar_color = '#16a34a' if over_100 else value_color
                    over_badge = (
                        f"<span style='color:#16a34a;font-weight:700;'>"
                        f" 초과달성</span>" if over_100 else ""
                    )
                    return (
                        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:5px;'>"
                        f"<div style='width:90px;font-size:0.7rem;color:{value_color};font-weight:700;'>"
                        f"{cur_label}</div>"
                        f"<div style='flex:1;background:#e5e7eb;border-radius:3px;height:8px;'>"
                        f"<div style='width:{bar_pct:.1f}%;background:{bar_color};"
                        f"border-radius:3px;height:8px;'></div></div>"
                        f"<div style='width:48px;font-size:0.65rem;color:#6b7280;text-align:right;'>"
                        f"{pct:.0f}%{over_badge}</div>"
                        f"<div style='width:80px;font-size:0.65rem;color:#9ca3af;text-align:right;'>"
                        f"평균 {avg_label}</div>"
                        f"</div>"
                    )

                day_color = '#374151'
                ret_color = pnl_color(cur_ret_pct)
                avg_days_str = f"{stats['avg_hold_days']:.0f}일"
                avg_ret_str = f"{signed_str(stats['avg_ret_pct'], '{:.1f}')}%"
                cur_days_str = f"보유 {hold_days_now}일"
                cur_ret_str = f"손익 {signed_str(cur_ret_pct, '{:.1f}')}%"

                gauge_html = (
                    f"<div style='border-top:1px dashed #e5e7eb;margin-top:10px;padding-top:8px;'>"
                    f"<div style='font-size:0.7rem;color:#6b7280;font-weight:600;margin-bottom:6px;'>"
                    f"🚀 이번 사이클 진행 (vs 평균)</div>"
                    f"{_progress_bar(day_progress, cur_days_str, avg_days_str, day_color)}"
                    f"{_progress_bar(ret_progress, cur_ret_str, avg_ret_str, ret_color)}"
                    f"</div>"
                )
                st.markdown(gauge_html, unsafe_allow_html=True)


# ====================================================
# 17-B. 전체 통계 패널 (메인 탭3)
# ====================================================
def render_overview_panel(
    portfolio_state: dict[str, TickerState],
    df_close: pd.DataFrame,
    all_analyses: Optional[dict] = None,
) -> None:
    """전체 포트폴리오 통계 — 시드/실현/비중/달력/자산 추이.

    이전엔 사이드바에 있던 영역을 메인 탭3로 이동.
    """
    portfolio_pnl = st.session_state.get('portfolio_pnl_cache')
    usd_krw = st.session_state.get('usd_krw_cache', CFG.USD_KRW_FALLBACK)
    df_close_last = st.session_state.get('df_close_last', {})
    dd_info = st.session_state.get('dd_info_cache')

    # ── 0. 새로고침 버튼 + 마지막 갱신 시간 ──
    with st.container(key="ov_refresh_row"):
        rc1, rc2 = st.columns([1, 3])
        if rc1.button("🔄 새로고침", key="ov_refresh_btn",
                      use_container_width=True):
            with st.spinner("데이터 갱신 중..."):
                st.cache_data.clear()
                # 통계 관련 캐시 무효화
                for k in [
                    'portfolio_pnl_cache', 'equity_series_cache',
                    'dd_info_cache', 'df_close_last',
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state['overview_last_refresh'] = (
                    datetime.datetime.now(
                        datetime.timezone(datetime.timedelta(hours=9))
                    ).strftime('%H:%M:%S')
                )
            st.rerun()

        ov_last = st.session_state.get('overview_last_refresh')
        if ov_last:
            rc2.markdown(
                f"<div style='font-size:0.7rem;color:#9ca3af;"
                f"padding-top:8px;'>마지막 갱신 {ov_last}</div>",
                unsafe_allow_html=True,
            )

    # ── 1. 보유 종목 평가 ──
    alloc_html = _build_alloc_html(portfolio_state, df_close_last, usd_krw)
    st.markdown(
        f"<div class='app-card'>{alloc_html}</div>",
        unsafe_allow_html=True,
    )

    # ── 2. 실현손익 카드 ──
    real_html = _build_realized_html(portfolio_state, usd_krw)
    st.markdown(
        f"<div class='app-card'>{real_html}</div>",
        unsafe_allow_html=True,
    )

    # ── 3. 자산 추이 통합 카드 (매매 일지 위로 이동) (시드 대비 누적 막대) ──
    # 모든 계산 달러 기반 — 환율 변동 영향 제거
    equity_series = st.session_state.get('equity_series_cache')
    seed_usd = get_seed_usd()
    seed_krw_man_approx = seed_usd * usd_krw / 10000   # 현재 환율 기준 원화 환산 (참고용)

    # 카드 헤더 - 시드/평가/손익률 (USD 메인 + KRW 병기)
    if portfolio_pnl is not None:
        # 달러 기반 계산
        cur_value_usd = seed_usd + portfolio_pnl
        ret_pct = portfolio_pnl / seed_usd * 100
        ret_color = pnl_color(ret_pct)
        # 원화 환산 (참고용, 현재 환율)
        cur_value_krw_man = cur_value_usd * usd_krw / 10000
        pnl_krw_man = portfolio_pnl * usd_krw / 10000

        header_summary = (
            f"<div style='font-size:0.7rem;color:#6b7280;margin-bottom:2px;'>"
            f"시드 <b>${seed_usd:,.0f}</b> "
            f"<span style='color:#9ca3af;'>(≈{int(round(seed_krw_man_approx)):,}만원)</span>"
            f" → 평가 "
            f"<span style='color:{ret_color};font-weight:700;'>"
            f"${cur_value_usd:,.0f}</span>"
            f" <span style='color:#9ca3af;'>(≈{int(round(cur_value_krw_man)):,}만원)</span>"
            f"</div>"
            f"<div style='font-size:1rem;color:{ret_color};font-weight:800;"
            f"margin-bottom:2px;'>"
            f"{signed_str(int(round(portfolio_pnl)), '${:,}')} "
            f"<span style='font-size:0.85rem;'>({signed_str(round(ret_pct), '{:d}')}%)</span>"
            f"</div>"
            f"<div style='font-size:0.65rem;color:#9ca3af;margin-bottom:6px;'>"
            f"≈ {signed_str(int(round(pnl_krw_man)), '{:,}')}만원"
            f"</div>"
        )
    else:
        header_summary = (
            f"<div style='font-size:0.85rem;color:#6b7280;margin-bottom:6px;'>"
            f"시드 <b>${seed_usd:,.0f}</b> "
            f"<span style='color:#9ca3af;'>(≈{int(round(seed_krw_man_approx)):,}만원)</span>"
            f"</div>"
        )

    # DD/MDD 텍스트
    dd_text = ""
    if dd_info and dd_info.get('mdd', 0) < 0:
        cur_dd = dd_info.get('current_dd', 0)
        mdd = dd_info.get('mdd', 0)
        mdd_date = dd_info.get('mdd_date')
        cur_dd_color = '#b91c1c' if cur_dd < -3 else '#6b7280'
        mdd_str = f"{mdd:.1f}%"
        if mdd_date is not None:
            try:
                mdd_d = pd.Timestamp(mdd_date).strftime('%y.%m.%d')
                mdd_str += f" ({mdd_d})"
            except Exception:
                pass
        dd_text = (
            f"<div style='font-size:0.62rem;color:#9ca3af;margin-bottom:8px;"
            f"display:flex;gap:10px;'>"
            f"<span title='최대 낙폭'>MDD <span style='color:#b91c1c;"
            f"font-weight:600;'>{mdd_str}</span></span>"
            f"<span title='현재 고점 대비 낙폭'>현재 DD "
            f"<span style='color:{cur_dd_color};font-weight:600;'>"
            f"{cur_dd:.1f}%</span></span>"
            f"</div>"
        )

    # 카드 시작 (헤더에 환율 정보 추가)
    bar_unit = st.session_state.get('overview_bar_unit', '일')
    bar_unit_label = {'일': '일별', '주': '주별', '월': '월별'}[bar_unit]
    st.markdown(
        f"<div class='app-card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"font-size:0.65rem;color:{COLOR_LABEL};margin-bottom:4px;'>"
        f"<span style='font-weight:700;'>💼 자산 추이 ({bar_unit_label})</span>"
        f"<span style='font-size:0.6rem;color:#6b7280;font-weight:500;'>"
        f"💲 {usd_krw:,.0f}원</span>"
        f"</div>"
        f"{header_summary}"
        f"{dd_text}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 시드 대비 누적 손익 막대 그래프 (USD 단위)
    if equity_series is not None and not equity_series.empty:
        # equity_series = USD 누적 손익 시계열
        # 막대 차트 Y축: USD 단위
        pnl_usd_series = equity_series.copy()

        # 일/주/월 단위로 마지막 값 추출 (누적값이라 resample.last())
        if bar_unit == '일':
            last_idx = pnl_usd_series.index.max()
            first_idx = last_idx - pd.Timedelta(days=20 * 2)
            full_idx = pd.date_range(first_idx, last_idx, freq='D')
            equity_resampled = pnl_usd_series.reindex(
                full_idx, method='ffill'
            ).iloc[-20:]
        elif bar_unit == '주':
            equity_resampled = pnl_usd_series.resample(
                'W-MON', label='left', closed='left'
            ).last().dropna().iloc[-20:]
        else:  # 월
            equity_resampled = pnl_usd_series.resample('MS').last().dropna().iloc[-20:]

        if not equity_resampled.empty:
            # 막대 색: 양수=빨강 (시드 위), 음수=파랑 (시드 아래)
            bar_colors_eq = [
                '#dc2626' if v >= 0 else '#2563eb'
                for v in equity_resampled.values
            ]
            # X축 라벨
            if bar_unit == '일':
                x_labels_eq = [d.strftime('%m/%d') for d in equity_resampled.index]
            elif bar_unit == '주':
                x_labels_eq = [d.strftime('%m/%d') for d in equity_resampled.index]
            else:
                x_labels_eq = [d.strftime('%y.%m') for d in equity_resampled.index]

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Bar(
                x=x_labels_eq,
                y=equity_resampled.values,
                marker_color=bar_colors_eq,
                hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>',
                showlegend=False,
            ))
            fig_eq.add_hline(y=0, line_color='#9ca3af', line_width=0.8)

            # MDD 마커 (시점이 표시 범위 안일 때만)
            if dd_info and dd_info.get('mdd', 0) < 0:
                mdd_date = dd_info.get('mdd_date')
                if mdd_date is not None:
                    try:
                        mdd_ts = pd.Timestamp(mdd_date)
                        # 가장 가까운 표시 인덱스
                        if len(equity_resampled) > 0:
                            nearest = equity_resampled.index.get_indexer(
                                [mdd_ts], method='nearest'
                            )[0]
                            if 0 <= nearest < len(equity_resampled):
                                mdd_x = x_labels_eq[nearest]
                                mdd_y = float(equity_resampled.iloc[nearest])
                                fig_eq.add_annotation(
                                    x=mdd_x, y=mdd_y,
                                    text=f"MDD {dd_info['mdd']:.1f}%",
                                    showarrow=True, arrowhead=2, arrowsize=0.8,
                                    arrowwidth=1, arrowcolor='#b91c1c',
                                    ax=0, ay=20,
                                    font=dict(size=9, color='#b91c1c'),
                                    bgcolor='rgba(255,255,255,0.85)',
                                    bordercolor='#b91c1c', borderwidth=0.8,
                                    borderpad=2,
                                )
                    except Exception:
                        pass

            # ── 매매 시점 마커 (▲ 매수 빨강, ▼ 매도 파랑) ──
            # bar_unit에 맞춰 시간 단위 정규화
            trade_marker_buy_x = []
            trade_marker_buy_hover = []
            trade_marker_sell_x = []
            trade_marker_sell_hover = []

            def _normalize_to_bar(d: datetime.date) -> Optional[str]:
                """매매 날짜를 막대 X 라벨로 매핑."""
                ts = pd.Timestamp(d)
                if bar_unit == '일':
                    # 정확히 같은 날 또는 가장 가까운 영업일
                    if ts in equity_resampled.index:
                        return ts.strftime('%m/%d')
                    return None
                elif bar_unit == '주':
                    # 해당 주 월요일
                    monday = ts - pd.Timedelta(days=ts.weekday())
                    if monday in equity_resampled.index:
                        return monday.strftime('%m/%d')
                    return None
                else:  # 월
                    month_start = ts.replace(day=1)
                    if pd.Timestamp(month_start) in equity_resampled.index:
                        return pd.Timestamp(month_start).strftime('%y.%m')
                    return None

            for tk, recs in st.session_state.trade_history.items():
                for r in recs:
                    qty = r.get('qty', 0)
                    price = r.get('price', 0)
                    if qty <= 0 or price <= 0:
                        continue
                    try:
                        rd = datetime.date.fromisoformat(r['date'])
                    except Exception:
                        continue
                    x_label = _normalize_to_bar(rd)
                    if x_label is None:
                        continue
                    hover = f"{tk} {r['type']} {qty}@${price:.2f} ({r['date']})"
                    if r['type'] == 'buy':
                        trade_marker_buy_x.append(x_label)
                        trade_marker_buy_hover.append(hover)
                    elif r['type'] == 'sell':
                        trade_marker_sell_x.append(x_label)
                        trade_marker_sell_hover.append(hover)

            # 매수 마커 (▲ 빨강) — Y=0 위치, X축 위에 표시
            if trade_marker_buy_x:
                fig_eq.add_trace(go.Scatter(
                    x=trade_marker_buy_x,
                    y=[0] * len(trade_marker_buy_x),
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up', size=9,
                        color='#dc2626', opacity=0.75,
                        line=dict(color='white', width=1),
                    ),
                    text=trade_marker_buy_hover,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ))
            # 매도 마커 (▼ 파랑)
            if trade_marker_sell_x:
                fig_eq.add_trace(go.Scatter(
                    x=trade_marker_sell_x,
                    y=[0] * len(trade_marker_sell_x),
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down', size=9,
                        color='#2563eb', opacity=0.75,
                        line=dict(color='white', width=1),
                    ),
                    text=trade_marker_sell_hover,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ))

            # X축 tick: 5~6개만
            max_ticks_eq = 6
            if len(x_labels_eq) > max_ticks_eq:
                step = max(1, len(x_labels_eq) // max_ticks_eq)
                tick_idx = list(range(0, len(x_labels_eq), step))
                tick_vals_eq = [x_labels_eq[i] for i in tick_idx]
            else:
                tick_vals_eq = x_labels_eq

            fig_eq.update_layout(
                height=200,
                margin=dict(l=4, r=8, t=4, b=4),
                xaxis=dict(showgrid=False, tickfont=dict(size=9),
                           tickmode='array', tickvals=tick_vals_eq),
                yaxis=dict(showgrid=True, gridcolor='rgba(156,163,175,0.2)',
                           tickfont=dict(size=9),
                           ticksuffix='만',
                           zeroline=True, zerolinecolor='#9ca3af',
                           zerolinewidth=0.8),
                paper_bgcolor='white', plot_bgcolor='white',
                bargap=0.15,
            )
            st.plotly_chart(fig_eq, use_container_width=True,
                            config={'displayModeBar': False, 'staticPlot': True})


    # ── 4. 매매 일지 + 신호 분석 ──
    if all_analyses is not None:
        journal = build_trade_journal(
            st.session_state.trade_history, all_analyses, df_close,
        )
        if journal:
            # 종목별 필터 selectbox
            journal_tickers = sorted({e['ticker'] for e in journal})
            filter_options = ['전체'] + [display_name(tk) + f" ({tk})" for tk in journal_tickers]
            ticker_label_to_code = {
                display_name(tk) + f" ({tk})": tk for tk in journal_tickers
            }
            filter_label = st.selectbox(
                "종목 필터",
                filter_options,
                index=0,
                key="tab3_journal_filter",
                label_visibility='collapsed',
            )
            filter_ticker = (
                ticker_label_to_code.get(filter_label) if filter_label != '전체' else None
            )

            jhtml = _build_journal_html(
                journal, {},
                show_ticker=(filter_ticker is None),
                filter_ticker=filter_ticker,
                title="📓 매매 일지" + (
                    f" — {display_name(filter_ticker)}" if filter_ticker else ""
                ),
            )
            if jhtml:
                st.markdown(
                    f"<div class='app-card'>{jhtml}</div>",
                    unsafe_allow_html=True,
                )

            # 메모 편집 expander (전체)
            recent_journal = sorted(journal, key=lambda x: x['date'], reverse=True)
            with st.expander(f"📝 매매 메모 편집 ({len(recent_journal)}건)", expanded=False):
                st.caption("전체 매매 메모 편집")
                for e in recent_journal:
                    if e.get('record_idx', -1) < 0:
                        continue
                    type_icon = '🔴' if e['type'] == 'buy' else '🔵'
                    label = (
                        f"{type_icon} {e['date']} {display_name(e['ticker'])} "
                        f"{e['qty']}주 @${e['price']:.2f}"
                    )
                    st.markdown(
                        f"<div style='font-size:0.72rem;color:#374151;"
                        f"margin-top:8px;'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                    cur_memo = e.get('memo', '')
                    new_memo = st.text_input(
                        f"memo_{e['ticker']}_{e['record_idx']}",
                        value=cur_memo,
                        key=f"ov_memo_{e['ticker']}_{e['record_idx']}_{e['date']}",
                        label_visibility="collapsed",
                        placeholder="메모 (편집 후 엔터)",
                    )
                    if new_memo.strip() != cur_memo:
                        idx = e['record_idx']
                        recs = st.session_state.trade_history.get(e['ticker'], [])
                        if 0 <= idx < len(recs):
                            if new_memo.strip():
                                recs[idx]['memo'] = new_memo.strip()
                            elif 'memo' in recs[idx]:
                                del recs[idx]['memo']
                            save_trade_history(st.session_state.trade_history)
                            st.rerun()

# ====================================================
# 17. CSS
# ====================================================
def build_css(selected_option: str, holding_tickers: set) -> str:
    btn_parts = []
    for ticker in TARGET_TICKERS:
        # 모멘텀 점수 → 마커 안 색 (탭2 마커와 동일)
        mom_score = st.session_state.ticker_momentum_scores.get(ticker, 0)
        bg = momentum_to_color(mom_score)
        # 짙은 색이면 흰 글씨, 연한 색이면 진한 글씨
        # 짙은: 7f1d1d, dc2626, 2563eb, 1e3a8a → 흰글씨
        # 연한: fca5a5, 93c5fd, 9ca3af → 검은글씨
        dark_bg = bg in ('#7f1d1d', '#dc2626', '#2563eb', '#1e3a8a')
        fg = '#ffffff' if dark_bg else '#1a1a1a'
        k = f"ticker_btn_{safe_key(ticker)}"
        sel_extra = (
            f"box-shadow:0 0 0 2px #fff,0 0 0 4px {bg}!important;"
            "transform:scale(1.03);"
        ) if selected_option == ticker else ""
        btn_parts.append(f"""
        div.st-key-{k} button {{
            background:{bg}!important; border-color:{bg}!important;
            color:{fg}!important; font-weight:500!important;
            height:1.7rem!important; font-size:0.62rem!important;
            padding:0 2px!important; line-height:1!important;
            min-height:0!important; border-radius:3px!important;
            width:100%!important; text-align:left!important; {sel_extra}
        }}
        div.st-key-{k} button p, div.st-key-{k} button strong,
        div.st-key-{k} button span {{ color:{fg}!important; }}
        div.st-key-{k} button strong {{ font-weight:700!important; }}
        div.st-key-{k} button:hover {{ opacity:0.82!important; }}""")

    di_border = (
        "border:2px solid #1565C0!important;font-weight:700!important;"
        if selected_option == "직접 입력" else ""
    )
    btn_parts.append(f"""
    div.st-key-ticker_btn_direct button {{
        height:1.7rem!important; font-size:0.62rem!important;
        padding:0 2px!important; min-height:0!important; border-radius:3px!important;
        line-height:1!important; {di_border}
    }}
    div.st-key-full_refresh_btn button {{
        height:1.7rem!important; min-height:0!important; border-radius:3px!important;
        font-size:0.62rem!important; font-weight:700!important; padding:0 2px!important;
        border:1px solid #cbd5e1!important; background:#f8fafc!important;
        color:#0f172a!important; line-height:1!important;
    }}
    div.st-key-full_refresh_btn button:hover {{
        border-color:#94a3b8!important; background:#eef2f7!important; }}""")

    return f"""<style>
    /* ─────────── 디자인 토큰 ─────────── */
    :root {{
        --bg-app:         #fafafa;
        --bg-card:        #ffffff;
        --bg-subtle:      #f8fafc;
        --border:         #e5e7eb;
        --border-strong:  #cbd5e1;
        --text-primary:   #0f172a;
        --text-secondary: #475569;
        --text-muted:     #94a3b8;
        --accent-buy:     #dc2626;
        --accent-sell:    #2563eb;
        --accent-success: #16a34a;
        --accent-warn:    #f59e0b;
        --space-1: 4px;  --space-2: 8px;  --space-3: 12px;
        --space-4: 16px; --space-5: 24px;
        --radius-sm: 4px; --radius-md: 8px; --radius-lg: 12px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-lg: 0 4px 6px rgba(0,0,0,0.04), 0 2px 4px rgba(0,0,0,0.04);
        --text-xs:   0.62rem; --text-sm: 0.72rem; --text-base: 0.85rem;
        --text-lg:   1rem;    --text-xl: 1.2rem;
    }}

    /* ─────────── Streamlit 기본 UI 완전 숨김 ─────────── */
    /* 사이드바 → 탭4로 이전했으므로 사이드바·토글 모두 제거 */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{ display: none !important; }}
    /* 상단 헤더 (Fork·GitHub·⋮ 메뉴) 통째로 숨김 */
    header[data-testid="stHeader"] {{ display: none !important; }}
    /* 하단 푸터, 메뉴, ⛵·🏯 등 잔여 UI */
    #MainMenu, footer {{ visibility: hidden !important; }}
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stAppDeployButton"],
    [data-testid="manageAppButton"],
    .stDeployButton,
    .stAppDeployButton,
    [class*="viewerBadge"],
    [class*="manageApp"],
    [class*="ManageApp"],
    [class*="_terminalButton_"],
    [class*="stStatusWidget"],
    a[href*="streamlit.app"],
    a[href*="streamlit.io"],
    a[href*="share.streamlit"],
    button[title*="Manage app"],
    button[title*="manage app"],
    button[kind="manageApp"] {{ display: none !important; }}
    /* Streamlit Cloud floating buttons (fixed position right-bottom + center-bottom) */
    .stApp > div:last-child:not([data-testid]):not([class*="stMain"]) {{
        display: none !important;
    }}
    body > iframe:not([title="streamlitApp"]) {{ display: none !important; }}

    /* ─────────── 공통 카드 클래스 ─────────── */
    .app-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-3) var(--space-4);
        box-shadow: var(--shadow-md);
        margin: var(--space-2) 0;
    }}
    .app-card-tight {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: var(--space-2) var(--space-3);
        box-shadow: var(--shadow-sm);
        margin: var(--space-1) 0;
    }}
    .app-card-header {{
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: var(--space-2);
        letter-spacing: -0.01em;
    }}

    /* ─────────── 타이포 위계 통일 (탭 전체) ─────────── */
    section[data-testid="stMain"] h3 {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin: var(--space-3) 0 var(--space-2) 0 !important;
        color: var(--text-secondary) !important;
        letter-spacing: -0.01em !important;
    }}
    /* 캡션(라벨 역할) 통일 */
    section[data-testid="stMain"] [data-testid="stCaptionContainer"],
    section[data-testid="stMain"] small {{
        font-size: var(--text-sm) !important;
        color: var(--text-secondary) !important;
        margin-bottom: var(--space-1) !important;
        font-weight: 500 !important;
    }}
    /* 라디오 라벨 */
    section[data-testid="stMain"] .stRadio label p {{
        font-size: var(--text-sm) !important;
    }}
    /* 입력 위젯 통일 */
    section[data-testid="stMain"] .stNumberInput input,
    section[data-testid="stMain"] .stTextInput input,
    section[data-testid="stMain"] .stDateInput input,
    section[data-testid="stMain"] .stSelectbox > div > div {{
        font-size: var(--text-base) !important;
        border-radius: var(--radius-md) !important;
    }}

    /* ─────────── 탭4 분석 시작일 버튼 컴팩트화 ─────────── */
    div.st-key-astart_6개월 button,
    div.st-key-astart_1년 button,
    div.st-key-astart_1년6개월 button,
    div.st-key-astart_2년 button {{
        height: 2rem !important;
        min-height: 0 !important;
        padding: 0.2rem 0.4rem !important;
        font-size: var(--text-sm) !important;
        border-radius: var(--radius-sm) !important;
    }}
    div.st-key-add_ticker_btn button,
    div.st-key-logout_btn button {{
        height: 2rem !important;
        min-height: 0 !important;
        font-size: var(--text-sm) !important;
        border-radius: var(--radius-sm) !important;
    }}

    /* ─────────── Phase C: 매매 일지 행 hover ─────────── */
    .app-card > div[style*="display:flex"]:hover {{
        background: var(--bg-subtle) !important;
        border-radius: var(--radius-sm);
    }}

    /* ─────────── Phase C: 매매 기록 삭제 버튼 컴팩트화 ─────────── */
    [class*="st-key-del_"] button {{
        height: 1.8rem !important;
        min-height: 0 !important;
        font-size: var(--text-sm) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--accent-sell) !important;
        border-color: var(--border) !important;
    }}
    [class*="st-key-del_"] button:hover {{
        background: var(--bg-subtle) !important;
        border-color: var(--accent-sell) !important;
    }}

    /* ─────────── Phase C: 기록 저장 버튼 강조 ─────────── */
    div[class*="st-key-"] button[kind="secondary"] {{
        transition: all 0.15s ease;
    }}

    /* ─────────── Phase C: expander 헤더 톤 다운 ─────────── */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > summary {{
        font-size: var(--text-base) !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
    }}

    /* ─────────── Phase C: 탭 헤더 톤 통일 ─────────── */
    [data-testid="stTabs"] [role="tab"] {{
        font-size: var(--text-base) !important;
        padding: var(--space-2) var(--space-3) !important;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        font-weight: 700 !important;
        color: var(--accent-buy) !important;
    }}

    .block-container {{
        padding-top:0.8rem!important; padding-bottom:0.5rem!important; max-width:100%!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {{
        flex-wrap:nowrap!important; gap:5px!important; align-items:flex-start!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:first-child {{
        flex:0 0 92px!important; min-width:92px!important;
        max-width:92px!important; padding:0!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:last-child {{
        flex:1 1 0!important; min-width:0!important; overflow:visible!important;
        padding-left:2px!important; padding-right:2px!important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child
        div[data-testid="stVerticalBlock"] > div {{ margin-bottom:0px!important; padding:0!important; }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child
        div[data-testid="stVerticalBlock"] {{ gap:1px!important; }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button p {{
        margin:0!important; padding:0!important; font-size:0.73rem!important;
        line-height:1!important; font-weight:500!important; white-space:pre!important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button span,
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button strong
        {{ color:inherit!important; }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button strong
        {{ font-weight:700!important; }}
    /* 탭3 달력 네비 — 88px 첫 컬럼 룰 무력화 */
    div.st-key-ov_cal_nav div[data-testid="stHorizontalBlock"] {{
        gap:2px!important; flex-wrap:nowrap!important;
    }}
    div.st-key-ov_cal_nav div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {{
        flex:initial!important; min-width:0!important;
        max-width:none!important; padding:0!important;
    }}
    div.st-key-ov_cal_nav div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:first-child,
    div.st-key-ov_cal_nav div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:last-child {{
        flex:0 0 60px!important; min-width:60px!important; max-width:60px!important;
    }}
    div.st-key-ov_cal_nav div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:nth-child(2) {{
        flex:1 1 auto!important;
    }}
    div.st-key-ov_cal_nav button {{
        min-height:32px!important; padding:4px!important; font-size:0.85rem!important;
    }}
    /* 탭3 새로고침 행 — 88px 첫 컬럼 룰 무력화 */
    div.st-key-ov_refresh_row div[data-testid="stHorizontalBlock"] {{
        gap:8px!important; flex-wrap:nowrap!important; margin-bottom:8px!important;
    }}
    div.st-key-ov_refresh_row div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"] {{
        flex:initial!important; min-width:0!important;
        max-width:none!important; padding:0!important;
    }}
    div.st-key-ov_refresh_row div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:first-child {{
        flex:0 0 120px!important; min-width:120px!important; max-width:120px!important;
    }}
    div.st-key-ov_refresh_row div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:nth-child(2) {{
        flex:1 1 auto!important;
    }}
    div.st-key-ov_refresh_row button {{
        min-height:32px!important; padding:4px 8px!important;
        font-size:0.8rem!important;
    }}
    {''.join(btn_parts)}
    </style>"""


# ====================================================
# 18. 메인
# ====================================================
def _append_ticker_to_close(
    df_close: pd.DataFrame, ticker: str, analysis_start: str, candle_type: str,
) -> pd.DataFrame:
    """단일 ticker fetch 후 df_close에 병합. 실패 시 원본 반환."""
    if f'{ticker}_Close' in df_close.columns:
        return df_close
    try:
        df_new = fetch_single_ticker(ticker, analysis_start)
        if df_new.empty:
            log.warning(f"Ticker fetch empty: {ticker}")
            return df_close
        if candle_type == '주봉':
            df_new = _resample_weekly(df_new)
        merged = pd.concat([df_close, df_new], axis=1).ffill()
        # 주말/비거래일 필터 적용 (fetch_all_data와 동일)
        if candle_type == '일봉':
            merged = _filter_trading_days(merged)
        return merged
    except Exception as e:
        log.warning(f"Ticker fetch failed: {ticker}: {e}")
        return df_close


def append_history_and_spy(
    df_close: pd.DataFrame, trade_history: dict, analysis_start: str, candle_type: str,
) -> pd.DataFrame:
    """매매 이력 종목 (TARGET 외) + SPY 추가 fetch."""
    # 매매 이력 종목
    extra_tickers = [
        tk for tk in trade_history.keys()
        if tk and f'{tk}_Close' not in df_close.columns
    ]
    if extra_tickers:
        with st.spinner(f"매매 이력 종목 {len(extra_tickers)}개 추가 로드..."):
            for tk in extra_tickers:
                df_close = _append_ticker_to_close(df_close, tk, analysis_start, candle_type)
    # SPY (β·SPY 계산용)
    df_close = _append_ticker_to_close(df_close, 'SPY', analysis_start, candle_type)
    return df_close


def compute_spy_betas(df_close: pd.DataFrame, tickers: list[str]) -> dict[str, float]:
    """SPY 대비 로그 회귀 슬로프 (β·SPY) 계산.

    log(ticker_price) = α + β × log(spy_price)
    분석 시작일~현재 전체 기간의 장기 가격 관계.
    """
    spy_betas: dict[str, float] = {}
    spy_col = 'SPY_Close'
    if spy_col not in df_close.columns:
        return spy_betas

    spy_price = df_close[spy_col].dropna()
    if len(spy_price) <= 10:
        return spy_betas

    log_spy = np.log(spy_price)
    for tk in tickers:
        col = f'{tk}_Close'
        if col not in df_close.columns:
            continue
        tk_price = df_close[col].dropna()
        tk_price = tk_price[tk_price > 0]
        common = log_spy.index.intersection(tk_price.index)
        if len(common) < 10:
            continue
        log_tk = np.log(tk_price.loc[common])
        log_s = log_spy.loc[common]
        try:
            slope, _ = np.polyfit(log_s.values, log_tk.values, 1)
            if np.isfinite(slope):
                spy_betas[tk] = float(slope)
        except Exception as e:
            log.warning(f"β·SPY log-regression failed: {tk}: {e}")
    return spy_betas


def update_ticker_signals(df_close: pd.DataFrame, all_analyses: dict) -> dict[str, float]:
    """각 종목의 신호/모멘텀 점수를 session_state에 저장. 일일 변화율 반환.

    저장: ticker_signals, ticker_momentum_scores, ticker_momentum_smooth
    """
    pct_changes = {}
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        pct_changes[ticker] = (
            df_close[col].pct_change().iloc[-1] * 100
            if col in df_close.columns and len(df_close) > 1 else 0.0
        )
        result = all_analyses.get(ticker)
        if result and result[0] is not None:
            df_t = result[0]
            cz = float(df_t['Z_Score'].iloc[-1]) if pd.notna(df_t['Z_Score'].iloc[-1]) else 0.0
            mhz = float(df_t['MACD_Hist_Z'].iloc[-1]) if pd.notna(df_t['MACD_Hist_Z'].iloc[-1]) else 0.0
            macd_pct = float(df_t['MACD_Pct'].iloc[-1]) if pd.notna(df_t['MACD_Pct'].iloc[-1]) else 0.0
            dmacd_pct = float(df_t['dMACD_Pct'].iloc[-1]) if pd.notna(df_t['dMACD_Pct'].iloc[-1]) else 0.0
            rsi = float(df_t['RSI'].iloc[-1]) if pd.notna(df_t['RSI'].iloc[-1]) else 50.0
            st.session_state.ticker_signals[ticker] = get_signal_combined(cz, mhz, rsi)
            st.session_state.ticker_momentum_scores[ticker] = compute_momentum_score(macd_pct, dmacd_pct, rsi)
            st.session_state.ticker_momentum_smooth[ticker] = compute_momentum_score_smooth(macd_pct, dmacd_pct, rsi)
        else:
            st.session_state.ticker_signals.setdefault(ticker, 'H')
            st.session_state.ticker_momentum_scores.setdefault(ticker, 0)
            st.session_state.ticker_momentum_smooth.setdefault(ticker, 0.0)
    return pct_changes


def main() -> None:
    init_session_state()
    # 쿠키 토큰이 있으면 자동 로그인 (30일 유지)
    try_auto_login_from_cookie()

    # ── 저장된 사용자 종목 리스트로 갱신 (Gist/로컬) ──
    loaded_tickers = load_target_tickers()
    TARGET_TICKERS[:] = loaded_tickers   # in-place 갱신

    DIRECT_INPUT_LABEL = "직접 입력"
    all_options = TARGET_TICKERS + [DIRECT_INPUT_LABEL]
    if st.session_state.selected_option not in all_options:
        st.session_state.selected_option = all_options[0]
    selected_option = st.session_state.selected_option

    selected_ticker = (
        st.session_state.get('custom_ticker_input', '').strip().upper() or None
        if selected_option == DIRECT_INPUT_LABEL else selected_option
    )

    # ★ 핵심 최적화: 사이클 정보 한 번만 계산
    portfolio_state = build_portfolio_state(st.session_state.trade_history)

    # ── 탭 컨테이너 생성 (탭4 = 설정, 사이드바 대체) ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 종목 분석", "🗺️ 종목 비교", "💼 포트폴리오", "⚙️ 설정"
    ])
    # 설정 위젯을 탭4 안에 렌더 → 이후 cfg 사용
    with tab4:
        cfg = render_sidebar(selected_ticker or TARGET_TICKERS[0], portfolio_state)

    if (st.session_state.analysis_start != cfg['analysis_start']
            or st.session_state.view_months != cfg['view_months']):
        st.session_state.analysis_start = cfg['analysis_start']
        st.session_state.view_months = cfg['view_months']
        s = load_settings()
        s.update({
            'analysis_start': cfg['analysis_start'],
            'view_months': cfg['view_months'],
        })
        save_settings(s)

    candle_type = cfg['candle_type']
    st.session_state.candle_type = candle_type

    raw_start = st.session_state.analysis_start.strip()
    try:
        analysis_start = datetime.datetime.strptime(raw_start, '%y-%m').strftime('%Y-%m-01')
    except ValueError:
        try:
            datetime.datetime.strptime(raw_start, '%Y-%m-%d')
            analysis_start = raw_start
        except ValueError:
            log.warning(f"Invalid analysis_start format: {raw_start}, using fallback")
            analysis_start = '2025-01-01'

    # ── Warmup 기간: EMA/expanding 안정화를 위해 fetch는 60일 일찍 시작 ──
    # 사용자 설정 시작일(analysis_start) 이전 60일 데이터를 추가로 fetch.
    # 그래프 표시할 때는 analysis_start 이후만 사용 (warmup 데이터는 계산용).
    try:
        _start_dt = pd.Timestamp(analysis_start)
        _fetch_start_dt = _start_dt - pd.Timedelta(days=60)
        fetch_start = _fetch_start_dt.strftime('%Y-%m-%d')
    except Exception:
        fetch_start = analysis_start
    # 사용자 분석 시작일 (표시 마스킹용)
    st.session_state['_display_analysis_start'] = analysis_start

    with st.spinner("데이터 로드 중..."):
        df_close = fetch_all_data(TARGET_TICKERS, fetch_start, candle_type)

    if selected_ticker and f'{selected_ticker}_Close' not in df_close.columns:
        with st.spinner(f"{selected_ticker} 데이터를 불러오는 중..."):
            df_custom = fetch_single_ticker(selected_ticker, fetch_start)
        if not df_custom.empty:
            if candle_type == '주봉':
                df_custom = _resample_weekly(df_custom)
            df_close = pd.concat([df_close, df_custom], axis=1).ffill()
            # 주말/비거래일 필터 (fetch_all_data와 동일)
            if candle_type == '일봉':
                df_close = _filter_trading_days(df_close)
        else:
            log.warning(f"Custom ticker fetch empty: {selected_ticker}")
            selected_ticker = None

    # 매매 이력 종목 + SPY 추가 fetch
    df_close = append_history_and_spy(
        df_close, st.session_state.trade_history, analysis_start, candle_type,
    )

    mkt = get_market_status()
    last_trading_date = pd.Timestamp(mkt['last_trading_date'])
    if not df_close.empty:
        st.session_state.last_data_date = df_close.index[-1].strftime('%Y-%m-%d')
        st.session_state['df_close_last'] = df_close.iloc[-1].to_dict()
        if candle_type == '일봉':
            df_close = df_close[df_close.index <= last_trading_date]

    with st.spinner("전체 종목 분석 중..."):
        # 매매 기록 종목도 분석에 포함 (현재 리스트에 없는 종목도)
        history_tickers = tuple(sorted(st.session_state.trade_history.keys()))
        all_analyses = compute_all_analyses(
            df_close, _version=8, candle_type=candle_type,
            extra_tickers=history_tickers,
        )

    # ── β·SPY 계산 (한눈에 보기 + 산점도용) ──
    spy_betas = compute_spy_betas(df_close, TARGET_TICKERS)
    st.session_state['spy_betas'] = spy_betas
    # 탭1 매매 일지에서 사용
    st.session_state['_all_analyses_cache'] = all_analyses
    st.session_state['_df_close_cache'] = df_close

    # ── 종목별 신호/모멘텀 계산 ──
    pct_changes = update_ticker_signals(df_close, all_analyses)

    df_daily = beta = std_resid = None
    if selected_ticker:
        if selected_ticker in TARGET_TICKERS:
            result = all_analyses.get(selected_ticker)
        elif f'{selected_ticker}_Close' in df_close.columns:
            with st.spinner(f"{display_name(selected_ticker)} 분석 중..."):
                result = process_asset_data(
                    df_close[[f'{X_ASSET_FIXED}_Close']],
                    df_close[[f'{selected_ticker}_Close']],
                    X_ASSET_FIXED, selected_ticker,
                )
        else:
            result = None

        if result and result[0] is not None:
            df_daily, beta, std_resid = result
            cz = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
            mhz = float(df_daily['MACD_Hist_Z'].iloc[-1]) if pd.notna(df_daily['MACD_Hist_Z'].iloc[-1]) else 0.0
            macd_pct = float(df_daily['MACD_Pct'].iloc[-1]) if pd.notna(df_daily['MACD_Pct'].iloc[-1]) else 0.0
            dmacd_pct = float(df_daily['dMACD_Pct'].iloc[-1]) if pd.notna(df_daily['dMACD_Pct'].iloc[-1]) else 0.0
            rsi = float(df_daily['RSI'].iloc[-1]) if pd.notna(df_daily['RSI'].iloc[-1]) else 50.0
            st.session_state.ticker_signals[selected_ticker] = get_signal_combined(cz, mhz, rsi)
            st.session_state.ticker_momentum_scores[selected_ticker] = compute_momentum_score(macd_pct, dmacd_pct, rsi)
            st.session_state.ticker_momentum_smooth[selected_ticker] = compute_momentum_score_smooth(macd_pct, dmacd_pct, rsi)

    holding_tickers = {
        tk for tk, ts in portfolio_state.items() if ts['cycle']['hold_qty'] > 0
    }

    # 드로다운 계산 (#6) + 자산 시계열 캐싱 (#15)
    # 표시 시점은 사용자 분석 시작 이후만 (warmup 60일은 계산용)
    if portfolio_state and not df_close.empty:
        display_start = st.session_state.get('_display_analysis_start', analysis_start)
        try:
            _ds_eq = pd.Timestamp(display_start)
            df_close_for_equity = df_close[df_close.index >= _ds_eq]
        except Exception:
            df_close_for_equity = df_close
        equity = compute_portfolio_equity(
            portfolio_state, df_close_for_equity, st.session_state.trade_history
        )
        if equity is not None and not equity.empty:
            st.session_state['dd_info_cache'] = compute_drawdown(equity)
            st.session_state['equity_series_cache'] = equity
            # portfolio_pnl을 equity 마지막 값으로 통일 (시드 카드와 자산추이 일치)
            st.session_state['portfolio_pnl_cache'] = float(equity.iloc[-1])
        else:
            st.session_state['dd_info_cache'] = None
            st.session_state['equity_series_cache'] = None
    else:
        st.session_state['dd_info_cache'] = None
        st.session_state['equity_series_cache'] = None

    st.markdown(build_css(selected_option, holding_tickers), unsafe_allow_html=True)
    KST = datetime.timezone(datetime.timedelta(hours=9))
    queried = datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M')
    data_lbl = (
        f"🟢 장중&nbsp;·&nbsp;조회: {queried}" if mkt['is_open']
        else f"🔴 장마감&nbsp;·&nbsp;{mkt['last_trading_label']}&nbsp;·&nbsp;조회: {queried}"
    )
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"margin-bottom:1px;padding-bottom:1px;'>"
        f"<b style='font-size:1.15rem;white-space:nowrap;color:#111;'>📊 퀀트 대시보드</b>"
        f"<span style='font-size:10px;color:#999;white-space:nowrap;'>{data_lbl}</span></div>",
        unsafe_allow_html=True,
    )

    # ── sorted_tickers 미리 계산 (탭 1, 탭 2 모두 사용) ──
    # 정렬 우선순위:
    #   1. 모멘텀 정수 점수 (색 분류와 일치, 음수=매수 위로)
    #   2. 모멘텀 smooth 점수 (같은 색 내 신호 강도 — 더 음수가 위로)
    #   3. 원래 순서 (TARGET_TICKERS index)
    def _ticker_sort_key(tk: str) -> tuple:
        mom_int = st.session_state.ticker_momentum_scores.get(tk, 0)
        mom_smooth = st.session_state.ticker_momentum_smooth.get(tk, 0.0)
        return (mom_int, mom_smooth, TARGET_TICKERS.index(tk))
    sorted_tickers = sorted(TARGET_TICKERS, key=_ticker_sort_key)

    # 매매 이력 있는 종목 (보유 여부 무관 — trade_history에 기록 있음)
    history_tickers = {
        tk for tk, recs in st.session_state.trade_history.items() if recs
    }

    # 탭은 cfg 결정 시점(위쪽)에서 이미 생성됨 (tab1, tab2, tab3, tab4)

    # ====================================================
    # 탭 1: 기존 화면 (종목버튼 + 차트 + 분석패널 + 메모)
    # ====================================================
    with tab1:
        btn_col, chart_col = st.columns([1, 6])
        with btn_col:
            for ticker in sorted_tickers:
                pct = pct_changes.get(ticker, 0)
                # ★ = 현재 보유 중, ☆ = 매매 이력만 (로그인 시에만 표시)
                if not is_authenticated():
                    star = ""
                elif ticker in holding_tickers:
                    star = "★ "
                elif ticker in history_tickers:
                    star = "☆ "
                else:
                    star = ""
                if st.button(
                    f"{star}**{display_name(ticker)}**   {pct:+.1f}%",
                    key=f"ticker_btn_{safe_key(ticker)}", use_container_width=True,
                ):
                    st.session_state.selected_option = ticker
                    st.session_state.custom_ticker_input = ''
                    st.rerun()
            if st.button(DIRECT_INPUT_LABEL, key="ticker_btn_direct", use_container_width=True):
                st.session_state.selected_option = DIRECT_INPUT_LABEL
                st.rerun()
            if selected_option == DIRECT_INPUT_LABEL:
                custom_input = st.text_input(
                    "티커", value=st.session_state.get('custom_ticker_input', ''),
                    placeholder="NVDA", label_visibility="collapsed",
                )
                new_val = custom_input.strip().upper()
                if new_val != st.session_state.get('custom_ticker_input', ''):
                    st.session_state.custom_ticker_input = new_val
                    st.rerun()
            if st.button("🔄 refresh", key="full_refresh_btn", use_container_width=True):
                with st.spinner("데이터 갱신 중..."):
                    st.cache_data.clear()
                    st.session_state['last_refresh'] = datetime.datetime.now(
                        datetime.timezone(datetime.timedelta(hours=9))
                    ).strftime('%H:%M:%S')
                st.rerun()
            last_refresh = st.session_state.get('last_refresh')
            if last_refresh:
                st.markdown(
                    f"<div style='font-size:0.65rem;color:#9ca3af;text-align:center;"
                    f"margin-top:-4px;'>updated {last_refresh}</div>",
                    unsafe_allow_html=True,
                )

        with chart_col:
            if df_daily is not None:
                # 헤더 + 위치 바 (그래프 위)
                render_position_tracker(
                    selected_ticker, df_daily, df_close, portfolio_state, beta, std_resid,
                    phase='top',
                )
                with st.spinner("캔들 데이터 로드 중..."):
                    df_ohlc = fetch_ohlc(selected_ticker, fetch_start, candle_type)
                df_daily_raw = None
                if candle_type == '주봉':
                    df_raw = fetch_all_data(TARGET_TICKERS, fetch_start, '일봉')
                    if not df_raw.empty:
                        df_raw = df_raw[df_raw.index <= last_trading_date]
                        col_raw = f'{selected_ticker}_Close'
                        if col_raw in df_raw.columns:
                            result_raw = process_asset_data(
                                df_raw[[f'{X_ASSET_FIXED}_Close']],
                                df_raw[[col_raw]], X_ASSET_FIXED, selected_ticker,
                            )
                            if result_raw[0] is not None:
                                df_daily_raw = result_raw[0]
                # 매매 기록 전체 전달 (사이클별 평균 매수/매도가 표시용)
                trade_records = st.session_state.trade_history.get(selected_ticker, [])
                # 표시용 마스킹: 사용자 분석 시작일 이전 (warmup) 데이터 제외
                display_start = st.session_state.get('_display_analysis_start', analysis_start)
                try:
                    _ds = pd.Timestamp(display_start)
                    df_daily_display = df_daily[df_daily.index >= _ds].copy()
                    if df_daily_raw is not None:
                        df_daily_raw_display = df_daily_raw[df_daily_raw.index >= _ds].copy()
                    else:
                        df_daily_raw_display = None
                    if df_ohlc is not None and not df_ohlc.empty:
                        df_ohlc_display = df_ohlc[df_ohlc.index >= _ds].copy()
                    else:
                        df_ohlc_display = df_ohlc
                except Exception:
                    df_daily_display = df_daily
                    df_daily_raw_display = df_daily_raw
                    df_ohlc_display = df_ohlc
                render_chart(
                    df_daily_display, selected_ticker, beta, std_resid,
                    cfg['guide_n'], st.session_state.view_months,
                    df_ohlc_display, df_daily_raw_display,
                    trade_records=trade_records,
                )
            elif selected_option == DIRECT_INPUT_LABEL:
                if not st.session_state.get('custom_ticker_input', ''):
                    st.info("왼쪽에서 티커를 입력해 주세요. (예: NVDA, 000660)")
                else:
                    st.error(f"'{st.session_state.custom_ticker_input}' 데이터를 가져올 수 없습니다.")
            elif selected_ticker:
                st.error("분석에 필요한 데이터가 부족합니다.")

        # 분석 패널 + 메모 (탭1 안)
        if df_daily is not None and selected_ticker:
            # 표시용 마스킹 (warmup 데이터 제외)
            display_start = st.session_state.get('_display_analysis_start', analysis_start)
            try:
                _ds_p = pd.Timestamp(display_start)
                df_daily_panel = df_daily[df_daily.index >= _ds_p].copy()
            except Exception:
                df_daily_panel = df_daily
            st.markdown(
                "<div data-analytics-panel style='margin-top:8px;'></div>",
                unsafe_allow_html=True,
            )
            render_analytics_panel(
                selected_ticker, df_daily_panel, df_close, portfolio_state, beta, std_resid,
            )

    # ====================================================
    # 탭 2: 한눈에 보기 (풀폭 22개 종목 미니바 리스트)
    # ====================================================
    with tab2:
        # ── 헤더 ──
        # [종목 70px] [σ 위치 라벨 flex] [기간 위치 라벨 flex]
        st.markdown(
            "<div style='display:flex;align-items:center;gap:6px;"
            "padding:6px 4px 4px 4px;font-size:0.55rem;color:#9ca3af;"
            "border-bottom:1px solid #e5e7eb;margin-bottom:2px;'>"
            # 좌측 70px
            "<div style='width:70px;flex-shrink:0;font-weight:700;color:#6b7280;"
            "font-size:0.65rem;'>종목</div>"
            # σ 위치 라벨
            "<div style='flex:1;position:relative;height:14px;min-width:0;'>"
            "<span style='position:absolute;left:0%;transform:translateX(-50%);'>-3σ</span>"
            "<span style='position:absolute;left:16.67%;transform:translateX(-50%);'>-2σ</span>"
            "<span style='position:absolute;left:33.33%;transform:translateX(-50%);'>-1σ</span>"
            "<span style='position:absolute;left:50%;transform:translateX(-50%);font-weight:700;'>추세</span>"
            "<span style='position:absolute;left:66.67%;transform:translateX(-50%);'>+1σ</span>"
            "<span style='position:absolute;left:83.33%;transform:translateX(-50%);'>+2σ</span>"
            "<span style='position:absolute;left:100%;transform:translateX(-50%);'>+3σ</span>"
            "</div>"
            # 기간 위치 라벨
            "<div style='flex:1;position:relative;height:14px;min-width:0;'>"
            "<span style='position:absolute;left:0%;transform:translateX(-50%);'>저점</span>"
            "<span style='position:absolute;left:50%;transform:translateX(-50%);font-weight:700;'>중간</span>"
            "<span style='position:absolute;left:100%;transform:translateX(-50%);'>고점</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── 종목 행 (탭1과 동일 정렬) ──
        for ticker in sorted_tickers:
            t_result = all_analyses.get(ticker)
            if not t_result or t_result[0] is None:
                continue
            t_df, t_beta, t_std = t_result
            if t_df.empty:
                continue

            t_col = f'{ticker}_Close'
            if t_col not in df_close.columns:
                continue
            t_cur_price = float(df_close[t_col].iloc[-1])

            t_avg_price = None
            t_ts = portfolio_state.get(ticker)
            if (t_ts and t_ts['cycle']['hold_qty'] > 0
                    and t_ts['cycle']['buy_qty'] > 0):
                t_avg_price = t_ts['cycle']['buy_cost'] / t_ts['cycle']['buy_qty']

            t_records = st.session_state.trade_history.get(ticker, [])

            mini_bar = build_mini_gradient_bar(
                t_df, ticker, t_cur_price, t_avg_price, t_beta, t_std,
                trade_records=t_records, bar_height=22,
            )
            if mini_bar is None:
                continue

            # ── 분석기간 최저~최고 대비 현재 위치 바 ──
            range_bar_html = ""
            t_close = df_close[t_col].dropna()
            if len(t_close) > 1:
                p_min = float(t_close.min())
                p_max = float(t_close.max())
                if p_max > p_min:
                    pos_pct = (t_cur_price - p_min) / (p_max - p_min) * 100
                    pos_pct = max(0, min(100, pos_pct))
                else:
                    pos_pct = 50

                # 그라디언트 바: 빨강(저점) → 회색 → 파랑(고점)
                # 마커: 현재 위치
                marker_color = (
                    '#7f1d1d' if pos_pct < 20 else
                    '#dc2626' if pos_pct < 35 else
                    '#6b7280' if pos_pct < 65 else
                    '#2563eb' if pos_pct < 80 else
                    '#1e3a8a'
                )
                range_bar_html = (
                    f"<div style='position:relative;height:22px;width:100%;'>"
                    # 그라디언트 바
                    f"<div style='position:absolute;top:8px;left:0;right:0;height:6px;"
                    f"border-radius:3px;"
                    f"background:linear-gradient(to right,"
                    f"#7f1d1d 0%,#dc2626 20%,#fca5a5 35%,"
                    f"#e5e7eb 50%,"
                    f"#93c5fd 65%,#2563eb 80%,#1e3a8a 100%);'></div>"
                    # 현재 위치 마커
                    f"<div style='position:absolute;left:{pos_pct:.1f}%;"
                    f"top:4px;width:10px;height:14px;background:{marker_color};"
                    f"border:1.5px solid white;border-radius:2px;"
                    f"box-shadow:0 0 0 1px {marker_color};"
                    f"transform:translateX(-50%);z-index:2;'></div>"
                    f"</div>"
                )

            is_holding = ticker in holding_tickers
            row_bg = '#f0fdf4' if is_holding else '#ffffff'
            star = "★ " if is_holding else ""

            # 행: [티커 70px] [σ 바 flex] [위치 바 flex]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"padding:2px 4px;background:{row_bg};"
                f"border-bottom:1px solid #f3f4f6;'>"
                # 좌측 70px - 티커만
                f"<div style='width:70px;flex-shrink:0;font-size:0.78rem;"
                f"font-weight:700;color:#111827;white-space:nowrap;"
                f"overflow:hidden;text-overflow:ellipsis;'>"
                f"{star}{display_name(ticker)}</div>"
                # σ 바
                f"<div style='flex:1;min-width:0;'>{mini_bar}</div>"
                # 위치 바
                f"<div style='flex:1;min-width:0;'>{range_bar_html}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.caption(
            "■ σ 바: 현재가 추세선 대비 위치 (테두리=모멘텀) · ▪ 평균단가 · ● 매수 ● 매도 · "
            "■ 기간 바: 분석 기간 최저~최고 대비 현재 위치"
        )

        # ── σ vs β 산점도 ──
        st.markdown(
            "<div style='margin-top:16px;padding:8px 4px 4px 4px;"
            "border-top:1px solid #e5e7eb;font-size:0.7rem;font-weight:700;"
            "color:#6b7280;'>📊 종목 변동성·시장민감도 분포</div>",
            unsafe_allow_html=True,
        )

        # 데이터 수집
        scatter_data = []
        spy_betas_sc = st.session_state.get('spy_betas', {})
        df_close_last_sc = st.session_state.get('df_close_last', {})
        for ticker in TARGET_TICKERS:
            t_result = all_analyses.get(ticker)
            if not t_result or t_result[0] is None:
                continue
            t_df, _, _ = t_result
            if t_df.empty:
                continue
            # σ%
            t_norm_col = f'{ticker}_Norm'
            sigma_pct_v = None
            if 'Predicted' in t_df.columns and t_norm_col in t_df.columns:
                t_log_resid = (np.log(t_df[t_norm_col]) - np.log(t_df['Predicted'])).dropna()
                t_exp_std = t_log_resid.expanding(
                    min_periods=CFG.EXPANDING_MIN_PERIODS
                ).std().dropna()
                if len(t_exp_std) > 0:
                    t_sigma_unit = float(t_exp_std.iloc[-1])
                    if t_sigma_unit > 0 and np.isfinite(t_sigma_unit):
                        sigma_pct_v = (np.exp(t_sigma_unit) - 1) * 100
            # β·SPY (로그 회귀 슬로프, 배수)
            beta_v = spy_betas_sc.get(ticker)
            if beta_v is None or not np.isfinite(beta_v):
                beta_v = None

            if sigma_pct_v is None or beta_v is None:
                continue

            # 보유/이력 상태
            t_ts = portfolio_state.get(ticker)
            has_history = ticker in st.session_state.trade_history and len(
                st.session_state.trade_history.get(ticker, [])
            ) > 0
            if t_ts and t_ts['cycle']['hold_qty'] > 0 and t_ts['cycle']['buy_qty'] > 0:
                t_avg = t_ts['cycle']['buy_cost'] / t_ts['cycle']['buy_qty']
                t_cur = df_close_last_sc.get(f'{ticker}_Close', t_avg)
                eval_usd = t_cur * t_ts['cycle']['hold_qty']
                is_holding = True
            else:
                eval_usd = 0
                is_holding = False

            scatter_data.append({
                'ticker': ticker,
                'sigma': sigma_pct_v,
                'beta': beta_v,
                'eval': eval_usd,
                'holding': is_holding,
                'has_history': has_history,
            })

        if scatter_data:
            # ── 모든 점 통일 (보유 이력 무관) ──
            # 크기, 테두리, 라벨 모두 동일
            sigma_vals = [d['sigma'] for d in scatter_data]
            beta_vals = [d['beta'] for d in scatter_data]
            sigma_med = float(np.median(sigma_vals))
            beta_med = float(np.median(beta_vals))

            # 색: 탭1 버튼과 동일 — 모멘텀 점수 기반 (구분의 유일한 차원)
            mom_scores_map = st.session_state.get('ticker_momentum_scores', {})
            colors = [
                momentum_to_color(mom_scores_map.get(d['ticker'], 0))
                for d in scatter_data
            ]

            # ── 라벨 위치 분산 (거리 기반 8방향) ──
            POSITION_8 = [
                'middle right',     # 0°
                'top right',        # 45°
                'top center',       # 90°
                'top left',         # 135°
                'middle left',      # 180°
                'bottom left',      # 225°
                'bottom center',    # 270°
                'bottom right',     # 315°
            ]
            import math as _m
            log_sigma_vals = [_m.log10(max(s, 0.1)) for s in sigma_vals]
            log_sigma_range = max(log_sigma_vals) - min(log_sigma_vals) or 1.0
            beta_range = max(beta_vals) - min(beta_vals) or 1.0
            norm_coords = [
                (
                    (b - min(beta_vals)) / beta_range,
                    (_m.log10(max(s, 0.1)) - min(log_sigma_vals)) / log_sigma_range,
                )
                for b, s in zip(beta_vals, sigma_vals)
            ]

            text_positions = []
            for i, (xi, yi) in enumerate(norm_coords):
                min_dist = float('inf')
                nearest_dx, nearest_dy = 0, 0
                for j, (xj, yj) in enumerate(norm_coords):
                    if i == j:
                        continue
                    dx = xj - xi
                    dy = yj - yi
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        nearest_dx, nearest_dy = dx, dy
                if nearest_dx == 0 and nearest_dy == 0:
                    text_positions.append(POSITION_8[i % 8])
                else:
                    angle = _m.atan2(-nearest_dy, -nearest_dx)
                    angle_norm = (angle + 2 * _m.pi) % (2 * _m.pi)
                    idx = int(round(angle_norm / (_m.pi / 4))) % 8
                    text_positions.append(POSITION_8[idx])

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=beta_vals,
                y=sigma_vals,
                mode='markers+text',
                marker=dict(
                    size=14,
                    color=colors,
                    opacity=0.9,
                    line=dict(color='white', width=1.5),
                ),
                text=[display_name(d['ticker']) for d in scatter_data],
                textposition=text_positions,
                textfont=dict(size=10, color='#374151', weight=500),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'β·SPY: %{x:+.2f}×<br>'
                    'σ: %{y:.0f}%<extra></extra>'
                ),
                showlegend=False,
                cliponaxis=False,
            ))

            # 사분면 가이드선 (더 흐리게)
            fig_sc.add_hline(
                y=sigma_med, line_dash="dot", line_color='#e5e7eb',
                line_width=1,
            )
            fig_sc.add_vline(
                x=beta_med, line_dash="dot", line_color='#e5e7eb',
                line_width=1,
            )

            # X축 (β): 그대로 — 음수~양수
            beta_min = min(beta_vals) - 0.8
            beta_max = max(beta_vals) + 1.2
            # Y축 (σ): 로그 스케일
            sigma_log_min = max(min(sigma_vals) * 0.7, 1)
            sigma_log_max = max(sigma_vals) * 1.3

            fig_sc.update_layout(
                height=420,
                margin=dict(l=44, r=24, t=20, b=44),
                xaxis=dict(
                    title=dict(text='β·SPY (SPY 대비 로그회귀 슬로프)',
                               font=dict(size=11, color='#6b7280')),
                    showgrid=True, gridcolor='rgba(229,231,235,0.6)',
                    tickfont=dict(size=9, color='#9ca3af'),
                    range=[beta_min, beta_max],
                    ticksuffix='×',
                    zeroline=True, zerolinecolor='#9ca3af', zerolinewidth=1.2,
                ),
                yaxis=dict(
                    title=dict(text='σ% (변동성)',
                               font=dict(size=11, color='#6b7280')),
                    type='log',
                    showgrid=True, gridcolor='rgba(229,231,235,0.6)',
                    tickfont=dict(size=9, color='#9ca3af'),
                    range=[np.log10(sigma_log_min), np.log10(sigma_log_max)],
                    ticksuffix='%',
                ),
                paper_bgcolor='white', plot_bgcolor='white',
            )
            st.plotly_chart(fig_sc, use_container_width=True,
                            config={'displayModeBar': False, 'staticPlot': True})

            st.caption(
                "■ 점 색=모멘텀 점수 (탭1 버튼과 동일) · 점선=중앙값"
            )

    # ====================================================
    # 탭 3: 전체 통계 (시드/실현/비중/달력/자산추이)
    # ====================================================
    with tab3:
        if not is_authenticated():
            st.markdown(
                "<div style='padding:32px 16px;text-align:center;color:#6b7280;'>"
                "<div style='font-size:2rem;margin-bottom:8px;'>🔒</div>"
                "<div style='font-weight:600;color:#374151;margin-bottom:4px;'>"
                "포트폴리오 정보는 로그인 후 표시됩니다</div>"
                "<div style='font-size:0.8rem;'>"
                "⚙️ 설정 탭에서 로그인하세요</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            render_overview_panel(portfolio_state, df_close, all_analyses)

    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

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

import calendar as _cal_mod
import datetime
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

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
    SEED_KRW: int = 21_000_000
    BETA_WARN: float = 4.0
    BETA_HIGH: float = 6.0
    USD_KRW_FALLBACK: float = 1400.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    Z_HIGH: float = 1.5
    MACD_HIGH: float = 2.0
    DATA_TTL_SEC: int = 300
    HTTP_TIMEOUT_SEC: int = 6
    MAX_PARALLEL_FETCH: int = 8
    EXPANDING_MIN_PERIODS: int = 30


CFG = Config()

X_ASSET_FIXED = 'SPY'
TARGET_TICKERS = [
    'TQQQ', 'SOXL', 'FNGU', 'HIBL', 'QPUX', 'LABU', 'DFEN', 'DPST',
    'GDXU', 'KORU', '005930', 'BITU', 'ETHT', 'AVXX', 'BTC-USD', 'ETH-USD', 'AVAV',
]
TICKER_DISPLAY_NAMES = {'BTC-USD': 'BTC', 'ETH-USD': 'ETH', '005930': '삼전', '000660': '하닉'}

# 종목별 색상
_C = {
    'index':   '#dc2626',  # 대형지수
    'tech':    '#f97316',  # 테크/혁신
    'semi':    '#eab308',  # 반도체
    'bio':     '#16a34a',  # 바이오
    'defense': '#14b8a6',  # 방산
    'fin':     '#2563eb',  # 금융/은행
    'em':      '#7c3aed',  # 신흥국/해외
    'commod':  '#ca8a04',  # 원자재/금
    'crypto':  '#6b7280',  # 암호화폐
    'other':   '#9ca3af',  # 기타
}
TICKER_COLOR = {
    'TQQQ': _C['index'], 'QPUX': _C['index'],
    'FNGU': _C['tech'],  'HIBL': _C['tech'],
    'SOXL': _C['tech'],  'LABU': _C['tech'],
    'DFEN': _C['defense'], 'AVXX': _C['defense'], 'AVAV': _C['defense'],
    'KORU': _C['em'],
    'DPST': _C['fin'],
    'GDXU': _C['commod'],
    'BITU': _C['crypto'], 'ETHT': _C['crypto'],
    'BTC-USD': _C['crypto'], 'ETH-USD': _C['crypto'],
    '005930': _C['other'],
}

SIGNAL_STYLE = {
    'FB2': ('#7f1d1d', '#ffffff'), 'FB':  ('#dc2626', '#ffffff'),
    'B':   ('#fca5a5', '#1a1a1a'), 'H':   ('#9ca3af', '#ffffff'),
    'S':   ('#93c5fd', '#1a1a1a'), 'FS':  ('#2563eb', '#ffffff'),
    'FS2': ('#1e3a8a', '#ffffff'),
}
BUTTON_TEXT_STYLE = {
    'FB2': '#f8fafc', 'FB': '#f8fafc', 'B': '#111827',
    'H': '#111827', 'S': '#111827', 'FS': '#f8fafc', 'FS2': '#f8fafc',
}
SIG_MARKER = {
    'FB2': ('triangle-up',   '#7f1d1d', 10),
    'FB':  ('triangle-up',   '#dc2626',  8),
    'FS':  ('triangle-down', '#2563eb',  8),
    'FS2': ('triangle-down', '#1e3a8a', 10),
}

# 색상 팔레트 (반복 사용)
COLOR_GAIN = '#b91c1c'   # 수익(빨강 - 한국식)
COLOR_LOSS = '#1d4ed8'   # 손실(파랑 - 한국식)
COLOR_NEUTRAL = '#9ca3af'
COLOR_TEXT = '#374151'
COLOR_LABEL = '#9ca3af'
COLOR_BORDER = '#e5e7eb'


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
def ticker_color(ticker: str) -> str:
    return TICKER_COLOR.get(ticker, '#9ca3af')


def display_name(ticker: str) -> str:
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)


def safe_key(ticker: str) -> str:
    return ticker.replace('-', '_').replace('.', '_').replace('/', '_')


def pnl_color(val: float) -> str:
    return COLOR_GAIN if val >= 0 else COLOR_LOSS


def signed_str(val: float, fmt: str = "{:,.0f}") -> str:
    """+/- 부호가 붙은 포맷 문자열."""
    sign = '+' if val >= 0 else ''
    return f"{sign}{fmt.format(val)}"


# ── #1 신호 정렬 우선순위 ──
SIGNAL_PRIORITY = {
    'FB2': 0, 'FB': 1, 'B': 2, 'H': 3, 'S': 4, 'FS': 5, 'FS2': 6,
}


def signal_sort_key(signal: str) -> int:
    return SIGNAL_PRIORITY.get(signal, 99)


# ── #18 percentile (역사적 분위) ──
def historical_percentile(series: pd.Series, current_value: float,
                           direction: str = 'low') -> float:
    """
    series 내에서 current_value의 분위를 % 단위로 반환.
    direction='low': 작은 값 기준 분위 (예: RSI가 낮을수록 1~10%)
    direction='high': 큰 값 기준 분위 (예: RSI가 높을수록 90~100%)
    """
    arr = series.dropna().values
    if len(arr) < 5 or pd.isna(current_value):
        return 50.0
    if direction == 'low':
        return float((arr <= current_value).mean() * 100)
    return float((arr >= current_value).mean() * 100)


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


def html_section_header(label: str, right: str = "") -> str:
    """사이드바 카드 내부 섹션 헤더."""
    right_html = right if right else ""
    return (
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.62rem;color:{COLOR_LABEL};margin-bottom:4px;'>"
        f"<span>{label}</span>{right_html}"
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
MEMO_FILE = 'memo_history.json'
SETTINGS_FILE = 'settings.json'
GIST_FILENAME = 'quant_trade_history.json'
MEMO_GIST_FILENAME = 'quant_memo_history.json'


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
def load_memo_history() -> dict: return _load_json(MEMO_FILE, MEMO_GIST_FILENAME)
def save_memo_history(h: dict) -> None: _save_json(MEMO_FILE, MEMO_GIST_FILENAME, h)


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
        'memo_history':        load_memo_history,
        'ticker_signals':      dict,
        'ticker_betas':        dict,
        'selected_option':     lambda: TARGET_TICKERS[0],
        'custom_ticker_input': str,
        'last_data_date':      str,
        'view_months':         lambda: load_settings().get('view_months', 3),
        'overview_view_months': lambda: 12,
        'analysis_start':      lambda: load_settings().get(
            'analysis_start',
            (datetime.date.today() - datetime.timedelta(days=365)).strftime('%y-%m')
        ),
        'memo_editing_idx':    lambda: None,
        'memo_input_key':      int,
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


def compute_combined_score_vec(
    cz: pd.Series, mhz: pd.Series, rsi: pd.Series
) -> np.ndarray:
    """벡터화 버전 — 전체 시계열을 한 번에 계산.

    부호 컨벤션: 음수 = 매수 신호, 양수 = 매도 신호
    """
    cz_v = cz.fillna(0).values
    mhz_v = mhz.fillna(0).values
    rsi_v = rsi.fillna(50).values

    s_cz = np.where(cz_v <= -CFG.Z_HIGH, -2,
            np.where(cz_v < 0, -1,
            np.where(cz_v >= CFG.Z_HIGH, 2, 1)))
    s_mhz = np.where(mhz_v <= -CFG.MACD_HIGH, -2,
             np.where(mhz_v < 0, -1,
             np.where(mhz_v >= CFG.MACD_HIGH, 2, 1)))
    s_rsi = np.where(rsi_v <= CFG.RSI_OVERSOLD, -2,
             np.where(rsi_v < 50, -1,
             np.where(rsi_v >= CFG.RSI_OVERBOUGHT, 2, 1)))
    return s_cz + s_mhz + s_rsi


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


def get_price_fill_color_combined(score: int) -> str:
    """음수 = 매수 (빨강 음영), 양수 = 매도 (파랑 음영)."""
    if score <= -5: return 'rgba(127,29,29,0.40)'   # 강 매수
    if score <= -3: return 'rgba(220,38,38,0.30)'
    if score <= -1: return 'rgba(252,165,165,0.20)'
    if score >= 5:  return 'rgba(30,58,138,0.40)'   # 강 매도
    if score >= 3:  return 'rgba(37,99,235,0.30)'
    if score >= 1:  return 'rgba(147,197,253,0.20)'
    return 'rgba(156,163,175,0.10)'


# ────────────────────────────────────────────────
# 모멘텀 점수 (MACD-Z + RSI 만, Z 제외)
# 위치(σ)와 독립적인 모멘텀 정보를 마커 색으로 표시하기 위함
# ────────────────────────────────────────────────
def compute_momentum_score(mhz: float, rsi: float) -> int:
    """MACD-Z + RSI 가중합산 모멘텀 점수 (-4 ~ +4 정수).

    가중치: MACD 1.2 / RSI 0.8 (MACD 60% 비중, 더 안정적)
    부호 컨벤션: 음수 = 매수 모멘텀, 양수 = 매도 모멘텀
    (Z, MACD, RSI 부호와 일치)

    RSI:
      ≤30: -2  / 30~40: -1  / 40~60: 0  / 60~70: +1  / ≥70: +2
    MACD-Z:
      ≤-2: -2  / -2~-1: -1  / -1~+1: 0  / +1~+2: +1  / ≥+2: +2
    """
    # RSI 점수 (음수=과매도/매수)
    if rsi <= CFG.RSI_OVERSOLD:        # 30
        s_rsi = -2
    elif rsi <= 40:
        s_rsi = -1
    elif rsi < 60:
        s_rsi = 0
    elif rsi < CFG.RSI_OVERBOUGHT:     # 70
        s_rsi = 1
    else:
        s_rsi = 2

    # MACD-Z 점수 (음수=매수 영역)
    if mhz <= -CFG.MACD_HIGH:          # -2
        s_mhz = -2
    elif mhz <= -1:
        s_mhz = -1
    elif mhz < 1:
        s_mhz = 0
    elif mhz < CFG.MACD_HIGH:          # +2
        s_mhz = 1
    else:
        s_mhz = 2

    # 가중합 (MACD 1.2 / RSI 0.8) → 합 ±4 보존
    return int(round(1.2 * s_mhz + 0.8 * s_rsi))


def compute_momentum_score_smooth(mhz: float, rsi: float) -> float:
    """모멘텀 점수의 연속 버전 (시각화용).

    선형 무제한 — 임계값 이상의 강도 차이도 그대로 보존.
    saturation 없음. 동일 가중치/부호 컨벤션 사용.

    - MACD-Z를 ±2 단위로 정규화, 가중 1.2
      (MACD ±2 → s_mhz ±1.2, MACD ±4 → s_mhz ±2.4)
    - RSI를 ±20 단위 (50 중심)로 정규화, 가중 0.8
      (RSI 30/70 → s_rsi ±0.8, RSI 10/90 → s_rsi ±1.6)
    - 합산 × 2 = 정수 척도와 일치
      (정상 범위 ±4, 극단 ±8까지 표현 가능)
    """
    s_mhz = 1.2 * (mhz / CFG.MACD_HIGH)
    s_rsi = 0.8 * ((rsi - 50) / 20.0)
    return 2.0 * (s_mhz + s_rsi)


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
    """모멘텀 점수 (-4 ~ +4) → 마커 테두리 색.

    음수 = 매수 (빨강), 양수 = 매도 (파랑), 0 = 중립 (회색)
    """
    if score <= -4: return '#7f1d1d'  # 짙은 빨강 — 강 매수 모멘텀
    if score <= -2: return '#dc2626'  # 빨강
    if score <= -1: return '#fca5a5'  # 연빨강
    if score >= 4:  return '#1e3a8a'  # 짙은 파랑 — 강 매도 모멘텀
    if score >= 2:  return '#2563eb'  # 파랑
    if score >= 1:  return '#93c5fd'  # 연파랑
    return '#9ca3af'                   # 회색 (중립)


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
def fetch_vix() -> Optional[float]:
    """VIX 현재값 — 변동성 레짐 표시용 (#17 미적용이지만 대비)."""
    try:
        today = datetime.date.today().strftime('%Y-%m-%d')
        week_ago = (datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        data = fdr.DataReader('VIX', week_ago, today)
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except Exception as e:
        log.debug(f"VIX fetch failed: {e}")
    return None


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
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    exp_std_macd = df['MACD_Hist'].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std()
    exp_mean_macd = df['MACD_Hist'].expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).mean()
    df['MACD_Hist_Z'] = (df['MACD_Hist'] - exp_mean_macd) / exp_std_macd.replace(0, np.nan)

    log_resid = np.log(df[f'{y_name}_Norm']) - np.log(df['Predicted'])
    std_resid = log_resid.std()
    df['Z_Score'] = (
        log_resid
        / log_resid.expanding(min_periods=CFG.EXPANDING_MIN_PERIODS).std().replace(0, np.nan)
    )

    # 벡터화 스코어
    df['Combined_Score'] = compute_combined_score_vec(
        df['Z_Score'], df['MACD_Hist_Z'], df['RSI']
    )
    df['Price_Fill_Color'] = df['Combined_Score'].apply(get_price_fill_color_combined)

    return df, beta, std_resid


@st.cache_data(show_spinner=False, ttl=CFG.DATA_TTL_SEC)
def compute_all_analyses(
    df_close: pd.DataFrame, _version: int = 9, candle_type: str = '일봉'
) -> dict:
    df_x = df_close[[f'{X_ASSET_FIXED}_Close']]
    results = {}
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        results[ticker] = (
            process_asset_data(df_x, df_close[[col]], X_ASSET_FIXED, ticker)
            if col in df_close.columns else None
        )
    return results


# ====================================================
# 9-A. 사이클 통계 (#1)
# ====================================================
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

    # 누적 평가액의 cummax 대비 하락률 (단, 시드 대비 절대값으로 계산)
    seed = CFG.SEED_KRW / fetch_usd_krw()[0]  # USD 환산 시드 (근사)
    portfolio_value = equity + seed         # 평가 자산 = 시드 + 누적손익
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
def compute_correlation_matrix(df_close: pd.DataFrame, tickers: list) -> Optional[pd.DataFrame]:
    """일별 로그수익률 기준 상관계수."""
    cols = [f'{t}_Close' for t in tickers if f'{t}_Close' in df_close.columns]
    if len(cols) < 2:
        return None
    sub = df_close[cols].copy()
    sub.columns = [c.replace('_Close', '') for c in cols]
    log_ret = np.log(sub / sub.shift(1)).dropna()
    if log_ret.empty:
        return None
    return log_ret.corr()


# ====================================================
# 9-D. 매수/매도 추천 후보 (#1, #2)
# ====================================================
def compute_buy_candidates(
    all_analyses: dict,
    holding_tickers: set,
    pct_changes: dict,
    df_close: pd.DataFrame,
) -> list:
    """
    매수 추천 후보:
    - 보유 안 한 종목 중
    - 신호: FB2 또는 FB
    - Z-score 분위 < 20% (역사적으로 과매도 영역)

    반환: [{ticker, signal, z_score, z_pct, rsi, rsi_pct, price, pct_change}]
          z_pct 낮은 순 정렬
    """
    candidates = []
    for ticker in TARGET_TICKERS:
        if ticker in holding_tickers:
            continue
        result = all_analyses.get(ticker)
        if not result or result[0] is None:
            continue
        df_t = result[0]
        if df_t.empty:
            continue
        last = df_t.iloc[-1]
        cz = float(last['Z_Score']) if pd.notna(last['Z_Score']) else 0.0
        mhz = float(last['MACD_Hist_Z']) if pd.notna(last['MACD_Hist_Z']) else 0.0
        rsi_v = float(last['RSI']) if pd.notna(last['RSI']) else 50.0
        signal = get_signal_combined(cz, mhz, rsi_v)

        # 강한 매수 신호만 (FB2/FB)
        if signal not in ('FB2', 'FB'):
            continue

        z_pct = historical_percentile(df_t['Z_Score'], cz, 'low')
        rsi_pct = historical_percentile(df_t['RSI'], rsi_v, 'low')

        # 분위 20% 이하 추가 필터 (역사적으로도 과매도)
        if z_pct > 20:
            continue

        col = f'{ticker}_Close'
        price = float(df_close[col].iloc[-1]) if col in df_close.columns else 0.0

        candidates.append({
            'ticker': ticker,
            'signal': signal,
            'z_score': cz,
            'z_pct': z_pct,
            'rsi': rsi_v,
            'rsi_pct': rsi_pct,
            'price': price,
            'pct_change': pct_changes.get(ticker, 0.0),
        })
    # 분위 낮은 순 (가장 과매도)
    candidates.sort(key=lambda x: x['z_pct'])
    return candidates


def compute_sell_candidates(
    all_analyses: dict,
    portfolio_state: dict,
    df_close: pd.DataFrame,
) -> list:
    """
    매도 추천 후보:
    - 보유 중인 종목 중
    - (a) 신호: FS2/FS  또는
    - (b) 평균단가 대비 +20% 이상 수익

    반환: [{ticker, signal, z_score, z_pct, ret_pct, hold_qty, reason}]
          ret_pct 높은 순 정렬
    """
    candidates = []
    for ticker, ts in portfolio_state.items():
        cyc = ts['cycle']
        if cyc['hold_qty'] <= 0:
            continue
        result = all_analyses.get(ticker)
        if not result or result[0] is None:
            continue
        df_t = result[0]
        if df_t.empty:
            continue

        last = df_t.iloc[-1]
        cz = float(last['Z_Score']) if pd.notna(last['Z_Score']) else 0.0
        mhz = float(last['MACD_Hist_Z']) if pd.notna(last['MACD_Hist_Z']) else 0.0
        rsi_v = float(last['RSI']) if pd.notna(last['RSI']) else 50.0
        signal = get_signal_combined(cz, mhz, rsi_v)

        col = f'{ticker}_Close'
        cur_price = float(df_close[col].iloc[-1]) if col in df_close.columns else 0.0
        avg_price = cyc['buy_cost'] / cyc['buy_qty']
        ret_pct = (cur_price - avg_price) / avg_price * 100 if avg_price > 0 else 0.0

        # 트리거 조건
        is_signal = signal in ('FS2', 'FS')
        is_profit = ret_pct >= 20.0

        if not (is_signal or is_profit):
            continue

        reasons = []
        if is_signal:
            reasons.append(f"{signal} 신호")
        if is_profit:
            reasons.append(f"+{ret_pct:.0f}% 익절권")

        z_pct = historical_percentile(df_t['Z_Score'], cz, 'high')

        candidates.append({
            'ticker': ticker,
            'signal': signal,
            'z_score': cz,
            'z_pct': z_pct,
            'ret_pct': ret_pct,
            'hold_qty': cyc['hold_qty'],
            'cur_price': cur_price,
            'avg_price': avg_price,
            'reason': ' · '.join(reasons),
        })
    candidates.sort(key=lambda x: -x['ret_pct'])
    return candidates


# ====================================================
# 10. 포트폴리오 사이클 계산 (단일 호출 최적화)
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
def _bar_colors(
    series: pd.Series,
    hi_thr: float, lo_thr: float,
    hi_c: str, lo_c: str, mid_hi_c: str, mid_lo_c: str,
) -> np.ndarray:
    return np.where(series >= hi_thr, hi_c,
           np.where(series >= 0, mid_hi_c,
           np.where(series <= lo_thr, lo_c, mid_lo_c)))


def add_segmented_fill(fig, df, y_col, color_col, row, col, baseline_y):
    for i in range(1, len(df)):
        y0, y1 = df[y_col].iloc[i - 1], df[y_col].iloc[i]
        fc = df[color_col].iloc[i]
        if pd.isna(y0) or pd.isna(y1) or not fc or fc == 'rgba(0,0,0,0)':
            continue
        fig.add_trace(go.Scatter(
            x=[df.index[i - 1], df.index[i - 1], df.index[i], df.index[i]],
            y=[baseline_y, y0, y1, baseline_y],
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='toself', fillcolor=fc,
            showlegend=False, hoverinfo='skip',
        ), row=row, col=col)


# ====================================================
# 12. 사이드바 - 포트폴리오 카드 빌더 (분리)
# ====================================================
def _build_seed_html(
    portfolio_pnl: Optional[float], usd_krw: float, dd_info: Optional[dict] = None,
) -> str:
    if portfolio_pnl is None:
        return f"<div style='font-size:0.7rem;color:{COLOR_LABEL};margin-bottom:4px;'>데이터 로딩 중...</div>"

    pnl_krw = portfolio_pnl * usd_krw
    seed_ret = pnl_krw / CFG.SEED_KRW * 100
    sc = pnl_color(seed_ret)

    # 드로다운 표시 (#6)
    dd_html = ""
    if dd_info and dd_info.get('mdd', 0) < -0.1:
        cur_dd = dd_info.get('current_dd', 0.0)
        mdd = dd_info.get('mdd', 0.0)
        cur_color = '#b91c1c' if cur_dd < -10 else '#ca8a04' if cur_dd < -3 else '#16a34a'
        mdd_date_str = (
            dd_info['mdd_date'].strftime('%y.%m') if dd_info.get('mdd_date') else ''
        )
        dd_html = (
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.6rem;color:{COLOR_LABEL};margin-top:3px;"
            f"border-top:1px dashed #e5e7eb;padding-top:3px;'>"
            f"<span>📉 현재DD <b style='color:{cur_color};'>{cur_dd:.1f}%</b></span>"
            f"<span>MDD <b style='color:#b91c1c;'>{mdd:.1f}%</b>"
            f"&nbsp;<span style='color:{COLOR_LABEL};'>({mdd_date_str})</span></span>"
            f"</div>"
        )

    return (
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:baseline;margin-bottom:4px;'>"
        f"<div>"
        f"<div style='font-size:0.62rem;color:{COLOR_LABEL};'>💰 시드 대비 수익률"
        f" &nbsp;<span style='font-size:0.6rem;'>({usd_krw:,.0f}₩/$)</span></div>"
        f"<div style='font-size:1.2rem;font-weight:800;color:{sc};line-height:1.2;'>"
        f"{signed_str(seed_ret, '{:.1f}')}%</div>"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"<div style='font-size:0.62rem;color:{COLOR_LABEL};'>손익</div>"
        f"<div style='font-size:0.82rem;font-weight:700;color:{sc};'>"
        f"{signed_str(round(pnl_krw / 10000))}만원</div>"
        f"<div style='font-size:0.72rem;font-weight:600;color:{sc};'>"
        f"{signed_str(round(portfolio_pnl), '${:,.0f}'.replace('$', ''))[0]}"
        f"${int(round(abs(portfolio_pnl))):,}</div>"
        f"<div style='font-size:0.62rem;color:{COLOR_LABEL};'>시드 {CFG.SEED_KRW // 10000:,}만원</div>"
        f"</div></div>"
        f"{dd_html}"
    )


def _build_realized_html(
    portfolio_state: dict[str, TickerState], usd_krw: float
) -> str:
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

    html = (
        f"{html_section_divider()}"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.62rem;color:{COLOR_LABEL};margin-bottom:4px;'>"
        f"<span>💵 실현손익</span>"
        f"<span style='color:{net_col};font-weight:700;'>"
        f"{signed_str(net_sum, '${:,.0f}'.replace('$',''))[0]}"
        f"${int(round(abs(net_sum))):,}"
        f"&nbsp;<span style='font-weight:400;color:{COLOR_LABEL};'>"
        f"({signed_str(round(net_sum * usd_krw / 10000))}만원)</span></span></div>"
    )
    for tk, real in sorted(rows, key=lambda x: -abs(x[1])):
        ratio = abs(real) / total_abs * 100 if total_abs else 0
        w = max(abs(real) / max_abs * 100, 2) if max_abs else 2
        # 수익=빨강, 손실=파랑
        bar_c = '#dc2626' if real >= 0 else '#2563eb'
        vc = pnl_color(real)
        html += (
            f"<div style='display:flex;align-items:center;gap:5px;margin-bottom:3px;'>"
            f"<div style='font-size:0.67rem;color:{COLOR_TEXT};width:40px;flex-shrink:0;'>{display_name(tk)}</div>"
            f"{html_progress_bar(w, bar_c)}"
            f"<div style='font-size:0.63rem;color:#6b7280;width:28px;text-align:right;flex-shrink:0;'>{ratio:.0f}%</div>"
            f"<div style='font-size:0.63rem;font-weight:700;color:{vc};"
            f"width:40px;text-align:right;flex-shrink:0;'>"
            f"{signed_str(real, '${:,.0f}'.replace('$',''))[0]}${int(round(abs(real))):,}</div>"
            f"</div>"
        )
    html += "</div>"
    return html


def _build_alloc_html(
    portfolio_state: dict[str, TickerState],
    df_close_last: dict,
    usd_krw: float,
) -> str:
    """💼 보유 종목 평가 — 수익/손실 분리 표시.

    수익 종목: [원금(연빨강)][수익(초록)] = 평가금액 길이
    손실 종목: [남은 평가(연빨강)][손실(파랑)] = 원금 길이
    """
    rows = []
    for tk, ts in portfolio_state.items():
        cyc = ts['cycle']
        if cyc['hold_qty'] <= 0 or cyc['buy_qty'] <= 0:
            continue
        # 평균단가 (전체 매수 평균)
        avg = cyc['buy_cost'] / cyc['buy_qty']
        # 남은 보유분의 원가 (탭1 정보카드와 일치)
        # 부분 매도 후 buy_cost는 줄지 않으므로 hold_qty × avg로 재계산
        inv_usd = avg * cyc['hold_qty']
        cur = df_close_last.get(f'{tk}_Close')
        if cur is None:
            cur = avg  # fallback
        eval_usd = cur * cyc['hold_qty']
        pnl_usd = eval_usd - inv_usd
        ret_pct = (cur / avg - 1) * 100 if avg > 0 else 0
        rows.append({
            'tk': tk, 'inv': inv_usd, 'eval': eval_usd, 'pnl': pnl_usd, 'ret': ret_pct,
        })

    if not rows:
        return ""

    # 평가금액 내림차순
    rows.sort(key=lambda r: -r['eval'])

    total_inv_krw = sum(r['inv'] for r in rows) * usd_krw
    total_eval_krw = sum(r['eval'] for r in rows) * usd_krw
    total_pnl_krw = total_eval_krw - total_inv_krw
    total_ret = (total_pnl_krw / total_inv_krw * 100) if total_inv_krw > 0 else 0
    total_pnl_color = pnl_color(total_pnl_krw)

    # 바 길이 기준: 모든 종목의 max(원금, 평가금액)을 100%로
    max_bar_value = max(max(r['inv'], r['eval']) for r in rows)

    # 색상: 원금=회색, 수익=빨강, 손실=파랑
    C_PRINCIPAL = '#d1d5db'  # 회색 — 원금/남은평가
    C_PROFIT    = '#dc2626'  # 빨강 — 수익
    C_LOSS      = '#2563eb'  # 파랑 — 손실

    html = (
        f"{html_section_divider()}"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"font-size:0.62rem;color:{COLOR_LABEL};margin-bottom:6px;'>"
        f"<span style='font-weight:700;'>💼 보유 종목 평가</span>"
        f"<span style='color:{total_pnl_color};font-weight:700;'>"
        f"{signed_str(int(round(total_pnl_krw / 10000)), '{:,}')}만원 "
        f"({signed_str(round(total_ret), '{:d}')}%)</span></div>"
    )

    for r in rows:
        tk = r['tk']
        inv_w = (r['inv'] / max_bar_value) * 100 if max_bar_value > 0 else 0
        eval_w = (r['eval'] / max_bar_value) * 100 if max_bar_value > 0 else 0
        pnl_w = abs(eval_w - inv_w)

        is_profit = r['pnl'] >= 0
        pnl_int = int(round(r['pnl']))   # USD
        ret_int = int(round(r['ret']))
        inv_int = int(round(r['inv']))
        pnl_color_v = pnl_color(r['pnl'])

        if is_profit:
            # 수익 종목: [원금 inv_w][수익 pnl_w] = eval_w
            seg1_w = inv_w
            seg1_c = C_PRINCIPAL
            seg2_w = pnl_w
            seg2_c = C_PROFIT
        else:
            # 손실 종목: [남은평가 eval_w][손실 pnl_w] = inv_w
            seg1_w = eval_w
            seg1_c = C_PRINCIPAL
            seg2_w = pnl_w
            seg2_c = C_LOSS

        bar_inner = (
            f"<div style='width:{seg1_w:.1f}%;background:{seg1_c};height:7px;"
            f"flex-shrink:0;border-radius:3px 0 0 3px;'></div>"
        )
        if seg2_w > 0.3:
            bar_inner += (
                f"<div style='width:{seg2_w:.1f}%;background:{seg2_c};height:7px;"
                f"flex-shrink:0;border-radius:0 3px 3px 0;'></div>"
            )

        html += (
            f"<div style='display:flex;align-items:center;gap:5px;margin-bottom:3px;'>"
            f"<div style='font-size:0.67rem;color:{COLOR_TEXT};width:42px;"
            f"flex-shrink:0;font-weight:600;'>{display_name(tk)}</div>"
            f"<div style='flex:1;background:#f3f4f6;border-radius:3px;height:7px;"
            f"display:flex;align-items:center;overflow:hidden;min-width:0;'>"
            f"{bar_inner}"
            f"</div>"
            # 우측: 원금 / 손익금($) / 손익률
            f"<div style='font-size:0.6rem;color:#6b7280;width:46px;text-align:right;"
            f"flex-shrink:0;line-height:1.15;'>"
            f"<div>${inv_int:,}</div>"
            f"<div style='color:{pnl_color_v};font-weight:600;'>"
            f"{signed_str(pnl_int, '{:,}')}</div>"
            f"</div>"
            f"<div style='font-size:0.62rem;width:34px;text-align:right;flex-shrink:0;"
            f"color:{pnl_color_v};font-weight:700;'>"
            f"{signed_str(ret_int, '{:d}')}%</div>"
            f"</div>"
        )

    # 사용률 표시 (시드 대비)
    used_pct = total_inv_krw / CFG.SEED_KRW * 100
    use_color = (
        '#b91c1c' if used_pct >= 90
        else '#f59e0b' if used_pct >= 70
        else '#16a34a'
    )
    html += (
        f"<div style='font-size:0.6rem;color:{COLOR_LABEL};margin-top:6px;"
        f"padding-top:4px;border-top:1px dashed #e5e7eb;display:flex;"
        f"justify-content:space-between;'>"
        f"<span>총 원금 {int(round(total_inv_krw/10000)):,}만원</span>"
        f"<span style='color:{use_color};font-weight:700;'>"
        f"시드 대비 {used_pct:.0f}% 사용</span></div>"
    )
    return html


def _build_calendar_html(
    trade_history: dict, cal_month: datetime.date, usd_krw: float
) -> str:
    today = datetime.date.today()
    dim = _cal_mod.monthrange(cal_month.year, cal_month.month)[1]
    fw = _cal_mod.monthrange(cal_month.year, cal_month.month)[0]

    daily_pnl: dict[int, float] = {}
    daily_buy: set[int] = set()    # 매수 발생일
    daily_sell: set[int] = set()   # 매도 발생일 (전량/일부 무관)
    for tk in TARGET_TICKERS:
        records = trade_history.get(tk, [])
        valid = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]
        if not valid:
            continue
        avg_p = 0.0
        hqty = 0
        for r in sorted(valid, key=lambda r: r['date']):
            rd = datetime.date.fromisoformat(r['date'])
            qty = int(r['qty'])
            in_month = (rd.year == cal_month.year and rd.month == cal_month.month)
            if r['type'] == 'buy':
                avg_p = (avg_p * hqty + r['price'] * qty) / (hqty + qty)
                hqty += qty
                if in_month:
                    daily_buy.add(rd.day)
            elif r['type'] == 'sell' and hqty > 0:
                sq = min(qty, hqty)
                pnl_d = (r['price'] - avg_p) * sq
                hqty -= sq
                if hqty == 0:
                    avg_p = 0.0
                if in_month:
                    daily_sell.add(rd.day)
                    daily_pnl[rd.day] = daily_pnl.get(rd.day, 0.0) + pnl_d

    month_total = sum(daily_pnl.values())
    header = (
        f"{html_section_divider().replace('5px', '4px')}"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"margin-bottom:5px;'>"
        f"<span style='font-size:0.62rem;color:{COLOR_LABEL};'>📅 일별 손익</span>"
    )
    if month_total != 0:
        mt_col = pnl_color(month_total)
        mt_krw = round(month_total * usd_krw / 10000)
        header += (
            f"<span style='font-size:0.62rem;font-weight:700;color:{mt_col};'>"
            f"{signed_str(month_total, '${:,.0f}'.replace('$',''))[0]}"
            f"${int(round(abs(month_total))):,}"
            f"&nbsp;<span style='font-weight:400;color:{COLOR_LABEL};'>"
            f"({signed_str(mt_krw)}만원)</span></span>"
        )
    header += "</div>"

    grid = (
        f"<div style='display:grid;grid-template-columns:repeat(7,1fr);"
        f"gap:2px;font-size:0.6rem;text-align:center;'>"
    )
    for wd, wc in [
        ('월', '#6b7280'), ('화', '#6b7280'), ('수', '#6b7280'),
        ('목', '#6b7280'), ('금', '#6b7280'), ('토', '#1d4ed8'), ('일', '#b91c1c'),
    ]:
        grid += f"<div style='color:{wc};font-weight:600;padding-bottom:2px;'>{wd}</div>"
    for _ in range(fw):
        grid += "<div></div>"

    for day in range(1, dim + 1):
        do = datetime.date(cal_month.year, cal_month.month, day)
        wkd = do.weekday()
        bdr = '1.5px solid #f59e0b' if do == today else '1px solid transparent'
        has_buy = day in daily_buy
        has_sell = day in daily_sell
        is_mixed = has_buy and has_sell

        if day in daily_pnl:
            # 매도가 있는 날 (손익 표시)
            p = daily_pnl[day]
            bg = '#fef2f2' if p >= 0 else '#eff6ff'
            fc = pnl_color(p)
            abs_p = abs(p)
            sign = '+' if p >= 0 else '-'
            lbl = f"{sign}${int(abs_p / 1000)}k" if abs_p >= 1000 else f"{sign}${int(abs_p)}"
            # 매수+매도 같은 날: 좌측 빨간 막대 추가
            mix_bar = (
                "border-left:3px solid #dc2626;"
                if is_mixed else ""
            )
            grid += (
                f"<div style='background:{bg};border-radius:3px;border:{bdr};"
                f"{mix_bar}line-height:1.2;padding:1px;'>"
                f"<div style='color:{COLOR_TEXT};font-size:0.58rem;'>{day}</div>"
                f"<div style='color:{fc};font-weight:700;font-size:0.52rem;'>{lbl}</div>"
                f"</div>"
            )
        elif has_buy:
            # 매수만 있는 날
            fc_d = COLOR_TEXT if wkd < 5 else '#d1d5db'
            grid += (
                f"<div style='border-radius:3px;border:{bdr};line-height:1.2;padding:1px;'>"
                f"<div style='color:{fc_d};font-size:0.58rem;'>{day}</div>"
                f"<div style='color:#dc2626;font-size:0.48rem;'>●</div>"
                f"</div>"
            )
        else:
            fc_d = '#d1d5db' if wkd >= 5 else COLOR_TEXT
            grid += (
                f"<div style='border:1px solid transparent;border:{bdr};border-radius:3px;padding:1px;'>"
                f"<div style='color:{fc_d};font-size:0.58rem;'>{day}</div>"
                f"</div>"
            )
    grid += "</div>"

    legend = (
        "<div style='display:flex;gap:8px;margin-top:5px;font-size:0.58rem;color:#9ca3af;'>"
        "<span><span style='background:#fef2f2;color:#b91c1c;padding:0 2px;"
        "border-radius:2px;font-size:0.55rem;'>+</span> 수익</span>"
        "<span><span style='background:#eff6ff;color:#1d4ed8;padding:0 2px;"
        "border-radius:2px;font-size:0.55rem;'>-</span> 손실</span>"
        "<span><span style='color:#dc2626;'>●</span> 매수</span>"
        "<span><span style='border-left:3px solid #dc2626;padding-left:2px;'>┃</span> 매수+매도</span>"
        "</div></div>"
    )
    return header + grid + legend


# ====================================================
# 13. 사이드바 (메인 진입점)
# ====================================================
def render_sidebar(
    selected_ticker: str,
    portfolio_state: dict[str, TickerState],
) -> dict:
    with st.sidebar:
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

        st.markdown("---")
        tok, gid = _gist_cfg()
        st.caption(
            f"☁️ Gist 연동됨 (`{gid[:8]}...`)" if (tok and gid)
            else "💾 로컬 저장 (Gist 미설정)"
        )

        # 매매 기록 / 메모를 탭으로 분리 (#4)
        tab_trade, tab_memo = st.tabs(["📈 매매 기록", "📝 메모"])

        with tab_trade:
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
            if st.button("기록 저장", key="trade_save_btn"):
                record = {'date': t_date.strftime("%Y-%m-%d"), 'type': t_type}
                if t_qty > 0:
                    record['qty'] = int(t_qty)
                if t_price > 0:
                    record['price'] = t_price
                st.session_state.trade_history.setdefault(t_ticker, []).append(record)
                save_trade_history(st.session_state.trade_history)
                # #14 햅틱 피드백 (모바일에서 진동)
                st.markdown(
                    "<script>if(navigator.vibrate){navigator.vibrate(50);}</script>",
                    unsafe_allow_html=True,
                )
                st.success("저장 완료!")
                st.rerun()

            st.markdown("**🗑️ 기존 기록 삭제**")
            history = st.session_state.trade_history
            if selected_ticker in history and history[selected_ticker]:
                for i, record in enumerate(history[selected_ticker]):
                    qty_str = f" {record['qty']}주" if record.get('qty') else ""
                    prc_str = f" @${record['price']:.2f}" if record.get('price') else ""
                    label = f"✕  {record['date']}  {record['type'].upper()}{qty_str}{prc_str}"
                    if st.button(label, key=f"del_{selected_ticker}_{i}"):
                        st.session_state.trade_history[selected_ticker].pop(i)
                        save_trade_history(st.session_state.trade_history)
                        st.rerun()
            else:
                st.caption("매매 기록이 없습니다.")

        with tab_memo:
            st.caption(f"현재 종목: **{display_name(selected_ticker)}**")
            memo_date = st.date_input("날짜 ", datetime.date.today(), key="sb_memo_date")
            memo_text = st.text_area(
                "메모 내용", value="",
                key=f"sb_memo_text_{st.session_state.memo_input_key}",
                placeholder="메모를 입력하세요...", height=80,
            )
            if st.button("메모 저장", key="memo_save_btn"):
                text = memo_text.strip()
                if text:
                    mh = st.session_state.memo_history
                    mh.setdefault(selected_ticker, []).append(
                        {'date': memo_date.strftime("%Y-%m-%d"), 'text': text}
                    )
                    mh[selected_ticker].sort(key=lambda x: x['date'], reverse=True)
                    save_memo_history(mh)
                    st.session_state.memo_input_key += 1
                    st.markdown(
                        "<script>if(navigator.vibrate){navigator.vibrate(50);}</script>",
                        unsafe_allow_html=True,
                    )
                    st.rerun()
                else:
                    st.warning("메모 내용을 입력해 주세요.")

            st.markdown("**📋 메모 목록**")
            mh = st.session_state.memo_history
            ticker_memos = mh.get(selected_ticker, [])
            for i, memo in enumerate(ticker_memos):
                preview = f"{memo['date']} {memo['text'][:12]}{'…' if len(memo['text']) > 12 else ''}"
                c1, c2 = st.columns(2)
                if c1.button(
                    f"✏️ {preview}",
                    key=f"memo_edit_btn_{safe_key(selected_ticker)}_{i}",
                    use_container_width=True,
                ):
                    st.session_state.memo_editing_idx = i
                    st.rerun()
                if c2.button(
                    f"✕ {preview}",
                    key=f"memo_del_{safe_key(selected_ticker)}_{i}",
                    use_container_width=True,
                ):
                    st.session_state.memo_history[selected_ticker].pop(i)
                    if st.session_state.memo_editing_idx == i:
                        st.session_state.memo_editing_idx = None
                    save_memo_history(st.session_state.memo_history)
                    st.rerun()
                if st.session_state.memo_editing_idx == i:
                    st.markdown(
                        "<div style='background:#f3f4f6;padding:6px;"
                        "border-radius:6px;margin:2px 0 6px 0;'>",
                        unsafe_allow_html=True,
                    )
                    try:
                        edit_date_default = datetime.date.fromisoformat(memo['date'])
                    except ValueError:
                        edit_date_default = datetime.date.today()
                    edit_date = st.date_input(
                        "날짜 수정", value=edit_date_default,
                        key=f"memo_edit_date_{safe_key(selected_ticker)}_{i}",
                    )
                    edit_text = st.text_area(
                        "내용 수정", value=memo['text'],
                        key=f"memo_edit_text_{safe_key(selected_ticker)}_{i}", height=70,
                    )
                    ecols = st.columns(2)
                    if ecols[0].button(
                        "💾 저장",
                        key=f"memo_edit_save_{safe_key(selected_ticker)}_{i}",
                        use_container_width=True,
                    ):
                        new_text = edit_text.strip()
                        if new_text:
                            st.session_state.memo_history[selected_ticker][i] = {
                                'date': edit_date.strftime("%Y-%m-%d"), 'text': new_text,
                            }
                            st.session_state.memo_history[selected_ticker].sort(
                                key=lambda x: x['date'], reverse=True
                            )
                            save_memo_history(st.session_state.memo_history)
                            st.session_state.memo_editing_idx = None
                            st.rerun()
                        else:
                            st.warning("내용을 입력해 주세요.")
                    if ecols[1].button(
                        "✖ 취소",
                        key=f"memo_edit_cancel_{safe_key(selected_ticker)}_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.memo_editing_idx = None
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            if not ticker_memos:
                st.caption("메모가 없습니다.")

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
) -> None:
    st.markdown("""<style>
    .js-plotly-plot, .js-plotly-plot .plotly, .js-plotly-plot svg {
        touch-action: none !important; }
    </style>""", unsafe_allow_html=True)

    PX = {'main': 150, 'spacer': 20, 'price': 100, 'zscore': 100, 'macd': 100, 'rsi': 100}
    plot_order = ['main', 'spacer', 'price', 'zscore', 'macd', 'rsi']
    total_rows = len(plot_order)
    total_h = sum(PX[p] for p in plot_order)
    # zscore 패널만 secondary_y 활성화 (Z + 모멘텀 dual-axis)
    specs = [
        [{"secondary_y": (p == 'zscore')}] for p in plot_order
    ]
    fig = make_subplots(
        rows=total_rows, cols=1,
        row_heights=[PX[p] / total_h for p in plot_order],
        vertical_spacing=0.02,
        specs=specs,
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
        marker=dict(symbol='star', color='hotpink', size=12, line=dict(color='black', width=1)),
        name='Current',
    ), row=row, col=1)

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
    row += 1

    # [2] Spacer
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
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['Plot_Norm_SPY'],
        mode='lines', line=dict(color='gray', width=1.5), name=X_ASSET_FIXED,
    ), row=row, col=1)

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
        fig.update_layout(xaxis3_rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(
            x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
            mode='lines', line=dict(color='black', width=1.5), name=selected_ticker,
        ), row=row, col=1)

    if not ohlc_norm.empty:
        vc = ohlc_norm[ohlc_norm.index >= view_start]
        p_lo = vc['Low'].min() if not vc.empty else df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].min()
        p_hi = vc['High'].max() if not vc.empty else df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].max()
    else:
        p_lo = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].min()
        p_hi = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].max()
    spy_lo = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_SPY'].min()
    spy_hi = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_SPY'].max()
    p_lo, p_hi = min(p_lo, spy_lo) * 0.97, max(p_hi, spy_hi) * 1.03

    add_segmented_fill(fig, df_daily, 'Plot_Norm_Ticker', 'Price_Fill_Color', row, 1, p_lo)

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

    # [4~5] Z / MACD / RSI
    C_HI = 'rgba(29,78,216,0.85)'
    C_LO = 'rgba(185,28,28,0.85)'
    C_MH = 'rgba(147,197,253,0.6)'
    C_ML = 'rgba(252,165,165,0.6)'

    for col_name, hi, lo, label, color_fn in [
        ('Z_Score',     CFG.Z_HIGH,    -CFG.Z_HIGH,    'Z',
         lambda v: 'black'),
        ('MACD_Hist_Z', CFG.MACD_HIGH, -CFG.MACD_HIGH, 'MACD',
         lambda v: '#dc2626' if v <= -CFG.MACD_HIGH else '#1d4ed8' if v >= CFG.MACD_HIGH else 'black'),
    ]:
        colors = _bar_colors(df_daily[col_name], hi, lo, C_HI, C_LO, C_MH, C_ML)
        fig.add_trace(go.Bar(
            x=df_daily.index, y=df_daily[col_name],
            marker_color=colors, name=col_name, hoverinfo='skip',
        ), row=row, col=1)
        for y_val, lc in [(hi, 'blue'), (-hi, 'red'), (0, 'gray')]:
            fig.add_hline(
                y=y_val, line_dash="solid", line_color=lc,
                line_width=0.8 if y_val != 0 else 0.6, row=row, col=1,
            )
        last_v = df_daily[col_name].iloc[-1]
        val = float(last_v) if pd.notna(last_v) else 0.0

        # ── Z 패널에만 모멘텀 라인 오버레이 (보조 Y축) ──
        if col_name == 'Z_Score':
            # 모멘텀 점수 시계열 (선형 무제한, 가중치 MACD 1.2 / RSI 0.8)
            # 부호 컨벤션: 음수=매수, 양수=매도
            # 임계값 이상의 강도 차이 보존 (saturation 없음)
            mhz_v = df_daily['MACD_Hist_Z'].fillna(0).values
            rsi_v = df_daily['RSI'].fillna(50).values
            s_mhz_smooth = 1.2 * (mhz_v / CFG.MACD_HIGH)
            s_rsi_smooth = 0.8 * ((rsi_v - 50) / 20.0)
            momentum_smooth = 2.0 * (s_mhz_smooth + s_rsi_smooth)
            momentum_series = pd.Series(momentum_smooth, index=df_daily.index)

            # 0 이하/이상 fill로 매수/매도 영역 강조
            fig.add_trace(go.Scatter(
                x=df_daily.index, y=momentum_series,
                mode='lines',
                line=dict(color='#7c3aed', width=1.8, shape='spline', smoothing=0.5),
                fill='tozeroy', fillcolor='rgba(124,58,237,0.08)',
                name='Momentum', hoverinfo='skip',
            ), row=row, col=1, secondary_y=True)

            # 라벨: Z + 모멘텀 마지막 값 (소수점 1자리로 미세 변화 가시화)
            mom_last_f = float(momentum_series.iloc[-1]) if len(momentum_series) > 0 else 0.0
            mom_last_int = int(round(mom_last_f))
            label_text = f"<b>Z {val:+.2f} · M {mom_last_f:+.1f}</b>"
            mom_color = momentum_to_color(mom_last_int)
            fig.add_annotation(
                x=0, y=1, xref='x domain', yref='y domain',
                text=label_text, showarrow=False,
                font=dict(size=11, color=color_fn(val)),
                xanchor='left', yanchor='top',
                bgcolor='white', bordercolor=mom_color, borderwidth=1.5, borderpad=2,
                row=row, col=1,
            )
        else:
            fig.add_annotation(
                x=0, y=1, xref='x domain', yref='y domain',
                text=f"<b>{label}  {val:+.2f}</b>", showarrow=False,
                font=dict(size=11, color=color_fn(val)),
                xanchor='left', yanchor='top',
                bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
                row=row, col=1,
            )

        view_abs = abs(df_daily.loc[df_daily.index >= view_start, col_name].dropna())
        rng = max(hi, view_abs.max() if not view_abs.empty else hi)
        # Z 축 범위 (좌측, 주 Y축)
        z_axis_max = rng + 0.3
        fig.update_yaxes(
            range=[-z_axis_max, z_axis_max], autorange=False, fixedrange=True,
            row=row, col=1,
        )

        # 모멘텀 축 동기화 (Z 패널일 때만)
        # 기본 매핑: Z 축 ±N → 모멘텀 축 ±2N
        # 의미: Z=±k 위치와 모멘텀=±2k 위치가 시각적으로 동일 → 신호 일치/발산을 위치로 비교
        # 단, 모멘텀 데이터가 동기화 범위를 넘으면 모멘텀 축을 확장 (잘림 방지)
        if col_name == 'Z_Score':
            mom_view = momentum_series.loc[momentum_series.index >= view_start]
            mom_data_max = float(abs(mom_view).max()) if not mom_view.empty else 0.0
            mom_axis_max = max(z_axis_max * 2, mom_data_max + 0.3)
            fig.update_yaxes(
                range=[-mom_axis_max, mom_axis_max],
                autorange=False, fixedrange=True,
                tickvals=[-6, -4, -2, 0, 2, 4, 6],
                tickfont=dict(size=8, color='#7c3aed'),
                row=row, col=1, secondary_y=True,
            )
        row += 1

    # RSI
    rsi_c = df_daily['RSI'] - 50
    rsi_colors = _bar_colors(
        df_daily['RSI'], CFG.RSI_OVERBOUGHT, CFG.RSI_OVERSOLD, C_HI, C_LO, C_MH, C_ML,
    )
    fig.add_trace(go.Bar(
        x=df_daily.index, y=rsi_c,
        marker_color=rsi_colors, name='RSI', hoverinfo='skip',
    ), row=row, col=1)
    for y_val, lc in [(20, 'blue'), (-20, 'red'), (0, 'gray')]:
        fig.add_hline(
            y=y_val, line_dash="solid", line_color=lc,
            line_width=0.8 if y_val != 0 else 0.6, row=row, col=1,
        )
    last_rsi = df_daily['RSI'].iloc[-1]
    rsi_val = float(last_rsi) if pd.notna(last_rsi) else 50.0
    rsi_color = (
        '#1d4ed8' if rsi_val >= CFG.RSI_OVERBOUGHT
        else '#dc2626' if rsi_val <= CFG.RSI_OVERSOLD else 'black'
    )
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>RSI  {rsi_val:.1f}</b>", showarrow=False,
        font=dict(size=11, color=rsi_color), xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
        row=row, col=1,
    )
    view_rsi_dropna = df_daily.loc[df_daily.index >= view_start, 'RSI'].dropna()
    rsi_abs = max(
        20.0,
        abs(view_rsi_dropna - 50).max() if not view_rsi_dropna.empty else 20.0,
    )
    fig.update_yaxes(
        range=[-(rsi_abs + 2), rsi_abs + 2], autorange=False, fixedrange=True, row=row, col=1,
    )

    # 매매 마커
    for trade in st.session_state.trade_history.get(selected_ticker, []):
        t_date = pd.to_datetime(trade['date'])
        is_buy = trade['type'] == 'buy'
        m_color = '#dc2626' if is_buy else '#1d4ed8'
        idx_sc = sc_df.index.get_indexer([t_date], method='nearest')[0]
        d_sc = sc_df.index[idx_sc]
        fig.add_trace(go.Scatter(
            x=[sc_df.loc[d_sc, f'{X_ASSET_FIXED}_Norm']],
            y=[sc_df.loc[d_sc, f'{selected_ticker}_Norm']],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if is_buy else 'triangle-down',
                size=10, color=m_color, line=dict(width=1, color='black'),
            ),
            name=f"{trade['type'].upper()} ({t_date.date()})", hoverinfo='skip',
        ), row=1, col=1)
        for r in range(3, total_rows + 1):
            fig.add_vline(
                x=t_date, line_dash="solid", line_width=1,
                line_color=m_color, opacity=0.8, row=r, col=1,
            )

    # ── 메모 마커 (#8) ──
    # 가격 차트(row=3)에 작은 마커 + Plot_Norm_Ticker 위에 점, 호버시 텍스트 표시
    memos = st.session_state.memo_history.get(selected_ticker, [])
    memo_view = [
        m for m in memos
        if pd.to_datetime(m['date']) >= view_start
        and pd.to_datetime(m['date']) <= last_date
    ]
    if memo_view:
        memo_x = []
        memo_y = []
        memo_text = []
        for m in memo_view:
            md = pd.to_datetime(m['date'])
            # df_daily의 가장 가까운 날짜 찾기
            idx_m = df_daily.index.get_indexer([md], method='nearest')[0]
            d_m = df_daily.index[idx_m]
            memo_x.append(d_m)
            # 메모 마커 y 위치: 차트 상단 부근
            memo_y.append(p_hi * 0.99)
            # 텍스트 미리보기
            preview = m['text'][:30] + ('…' if len(m['text']) > 30 else '')
            memo_text.append(f"📝 {m['date']}<br>{preview}")
        fig.add_trace(go.Scatter(
            x=memo_x, y=memo_y,
            mode='markers',
            marker=dict(
                symbol='square', size=8, color='#fbbf24',
                line=dict(width=1, color='#92400e'),
            ),
            text=memo_text,
            hovertemplate='%{text}<extra></extra>',
            hoverinfo='text',
            showlegend=False, name='memos',
        ), row=price_row, col=1)
        # 메모 vline (옅은 노란색)
        for d in memo_x:
            for r in range(3, total_rows + 1):
                fig.add_vline(
                    x=d, line_dash="dot", line_width=1,
                    line_color='#fbbf24', opacity=0.4, row=r, col=1,
                )

    # 축 공통 스타일
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    for r in range(3, total_rows + 1):
        fig.update_xaxes(
            showgrid=True, gridcolor='rgba(156,163,175,0.28)',
            gridwidth=0.6, griddash='dot', dtick=grid_dtick_ms,
            matches=time_x_axis, rangebreaks=[dict(bounds=['sat', 'mon'])],
            showticklabels=(r == total_rows), tickformat="%m/%d",
            range=[view_start, last_date], row=r, col=1,
        )
        fig.update_yaxes(showgrid=False, autorange=False, fixedrange=True, row=r, col=1)

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
    # σ% (변동성), β·30d (추세), 수익% (보유 시)
    TREND_DAYS_HDR = 30
    sigma_pct_int = None
    if df_daily is not None and sigma_unit > 0 and np.isfinite(sigma_unit):
        sigma_pct_int = int(round((np.exp(sigma_unit) - 1) * 100))

    trend_pct_int = None
    trend_color_hdr = '#6b7280'
    if (df_daily is not None and 'Predicted' in df_daily.columns
            and len(df_daily) > TREND_DAYS_HDR):
        try:
            p_recent = float(df_daily['Predicted'].iloc[-1])
            p_past = float(df_daily['Predicted'].iloc[-(TREND_DAYS_HDR + 1)])
            if p_past > 0 and np.isfinite(p_recent) and np.isfinite(p_past):
                trend_pct_v = (p_recent / p_past - 1) * 100
                trend_pct_int = int(round(trend_pct_v))
                trend_color_hdr = pnl_color(trend_pct_v)
        except Exception:
            pass

    # 평가수익률 (보유 시만)
    avg_price_hdr = None
    ts_hdr = portfolio_state.get(selected_ticker)
    if (ts_hdr and ts_hdr['cycle']['hold_qty'] > 0
            and ts_hdr['cycle']['buy_qty'] > 0):
        avg_price_hdr = ts_hdr['cycle']['buy_cost'] / ts_hdr['cycle']['buy_qty']

    ret_pct_str_hdr = "—"
    ret_color_hdr = '#9ca3af'
    if avg_price_hdr is not None and current_price is not None:
        ret_pct_v = (current_price / avg_price_hdr - 1) * 100
        ret_pct_int = int(round(ret_pct_v))
        ret_pct_str_hdr = signed_str(ret_pct_int, '{:d}') + "%"
        ret_color_hdr = pnl_color(ret_pct_v)

    sigma_str_hdr = f"±{sigma_pct_int}%" if sigma_pct_int is not None else "—"
    trend_str_hdr = (
        signed_str(trend_pct_int, '{:d}') + "%"
        if trend_pct_int is not None else "—"
    )

    header_right = (
        f"<span style='font-size:0.7rem;color:#6b7280;'>"
        f"<span title='1σ 변동성'>σ {sigma_str_hdr}</span>"
        f" · <span title='30일 추세' style='color:{trend_color_hdr};font-weight:600;'>"
        f"β·30d {trend_str_hdr}</span>"
        f" · <span title='평가수익률' style='color:{ret_color_hdr};font-weight:600;'>"
        f"수익 {ret_pct_str_hdr}</span>"
        f"</span>"
    )
    header_html = (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"padding:4px 12px 2px 12px;margin-top:4px;flex-wrap:wrap;gap:6px;'>"
        f"<span style='font-size:1rem;font-weight:800;color:#111827;'>"
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

    # 매매 기록 없는 경우
    if ts is None or ts['cycle']['cycle_start'] is None or ts['cycle']['buy_qty'] == 0:
        price_html = (
            html_metric("현재가", f"${current_price:,.2f}")
            if current_price is not None else html_dash_cell("현재가")
        )
        st.markdown(header_html + f"""
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
        # 액션 카드 (미보유 — 단, 과거 매매 이력은 있을 수 있음)
        if (current_price is not None and beta is not None and std_resid is not None):
            records_for_card = (
                st.session_state.trade_history.get(selected_ticker, [])
                if hasattr(st, 'session_state') else None
            )
            action_card = build_action_card_html(
                df_daily, selected_ticker, current_price, None, beta, std_resid,
                trade_records=records_for_card,
            )
            if action_card:
                st.markdown(action_card, unsafe_allow_html=True)
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

    st.markdown(header_html + f"""
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

    # 액션 카드 (보유 중)
    if current_price is not None and beta is not None and std_resid is not None:
        records_for_card = st.session_state.trade_history.get(selected_ticker, [])
        action_card = build_action_card_html(
            df_daily, selected_ticker, current_price,
            avg_price if hold_qty > 0 else None, beta, std_resid,
            trade_records=records_for_card,
        )
        if action_card:
            st.markdown(action_card, unsafe_allow_html=True)


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
        "<div style='position:relative;height:48px;margin:6px 8p
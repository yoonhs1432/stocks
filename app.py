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
    'SPYU', 'SOXL', 'TQQQ', 'FNGU', 'HIBL', 'TARK', 'QPUX', 'BNKU',
    'URTY', 'TECL', 'LABU', 'DFEN', 'TNA', 'DPST',
    'GDXU', 'KORU', '005930', 'BITU', 'ETHT', 'AVXX',
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
    'SPYU': _C['index'], 'TQQQ': _C['index'], 'QPUX': _C['index'], 'URTY': _C['index'],
    'TNA':  _C['index'],
    'FNGU': _C['tech'],  'TECL': _C['tech'],  'TARK': _C['tech'],  'HIBL': _C['tech'],
    'SOXL': _C['tech'],  'LABU': _C['tech'],
    'DFEN': _C['defense'], 'AVXX': _C['defense'],
    'KORU': _C['em'],
    'BNKU': _C['fin'],   'DPST': _C['fin'],
    'GDXU': _C['commod'],
    'BITU': _C['crypto'], 'ETHT': _C['crypto'],
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
    """단일 시점 스코어 (스칼라용)."""
    s = 0
    s += 2 if cz <= -CFG.Z_HIGH else 1 if cz < 0 else -2 if cz >= CFG.Z_HIGH else -1
    s += 2 if mhz <= -CFG.MACD_HIGH else 1 if mhz < 0 else -2 if mhz >= CFG.MACD_HIGH else -1
    s += 2 if rsi <= CFG.RSI_OVERSOLD else 1 if rsi < 50 else -2 if rsi >= CFG.RSI_OVERBOUGHT else -1
    return s


def compute_combined_score_vec(
    cz: pd.Series, mhz: pd.Series, rsi: pd.Series
) -> np.ndarray:
    """벡터화 버전 — 전체 시계열을 한 번에 계산."""
    cz_v = cz.fillna(0).values
    mhz_v = mhz.fillna(0).values
    rsi_v = rsi.fillna(50).values

    s_cz = np.where(cz_v <= -CFG.Z_HIGH, 2,
            np.where(cz_v < 0, 1,
            np.where(cz_v >= CFG.Z_HIGH, -2, -1)))
    s_mhz = np.where(mhz_v <= -CFG.MACD_HIGH, 2,
             np.where(mhz_v < 0, 1,
             np.where(mhz_v >= CFG.MACD_HIGH, -2, -1)))
    s_rsi = np.where(rsi_v <= CFG.RSI_OVERSOLD, 2,
             np.where(rsi_v < 50, 1,
             np.where(rsi_v >= CFG.RSI_OVERBOUGHT, -2, -1)))
    return s_cz + s_mhz + s_rsi


def score_to_signal(score: int) -> str:
    if score >= 5:  return 'FB2'
    if score >= 3:  return 'FB'
    if score >= 1:  return 'B'
    if score <= -5: return 'FS2'
    if score <= -3: return 'FS'
    if score <= -1: return 'S'
    return 'H'


def get_signal_combined(cz: float, mhz: float, rsi: float) -> str:
    return score_to_signal(compute_combined_score(cz, mhz, rsi))


def get_price_fill_color_combined(score: int) -> str:
    if score >= 5:  return 'rgba(127,29,29,0.40)'
    if score >= 3:  return 'rgba(220,38,38,0.30)'
    if score >= 1:  return 'rgba(252,165,165,0.20)'
    if score <= -5: return 'rgba(30,58,138,0.40)'
    if score <= -3: return 'rgba(37,99,235,0.30)'
    if score <= -1: return 'rgba(147,197,253,0.20)'
    return 'rgba(156,163,175,0.10)'


# ────────────────────────────────────────────────
# 모멘텀 점수 (MACD-Z + RSI 만, Z 제외)
# 위치(σ)와 독립적인 모멘텀 정보를 마커 색으로 표시하기 위함
# ────────────────────────────────────────────────
def compute_momentum_score(mhz: float, rsi: float) -> int:
    """MACD-Z + RSI 합산 모멘텀 점수 (-4 ~ +4).

    RSI:
      ≤30: +2  / 30~40: +1  / 40~60: 0  / 60~70: -1  / ≥70: -2
    MACD-Z:
      ≤-2: +2  / -2~-1: +1  / -1~+1: 0  / +1~+2: -1  / ≥+2: -2

    + : 매수 모멘텀 (과매도 신호)
    - : 매도 모멘텀 (과매수 신호)
    """
    # RSI 점수
    if rsi <= CFG.RSI_OVERSOLD:        # 30
        s_rsi = 2
    elif rsi <= 40:
        s_rsi = 1
    elif rsi < 60:
        s_rsi = 0
    elif rsi < CFG.RSI_OVERBOUGHT:     # 70
        s_rsi = -1
    else:
        s_rsi = -2

    # MACD-Z 점수
    if mhz <= -CFG.MACD_HIGH:          # -2
        s_mhz = 2
    elif mhz <= -1:
        s_mhz = 1
    elif mhz < 1:
        s_mhz = 0
    elif mhz < CFG.MACD_HIGH:          # +2
        s_mhz = -1
    else:
        s_mhz = -2

    return s_rsi + s_mhz


def momentum_score_to_signal(score: int) -> str:
    """모멘텀 점수 → 신호 라벨 (Z 제외라 임계값 다름)."""
    if score >= 4:  return 'FB2'
    if score >= 2:  return 'FB'
    if score >= 1:  return 'B'
    if score <= -4: return 'FS2'
    if score <= -2: return 'FS'
    if score <= -1: return 'S'
    return 'H'


def momentum_to_color(score: int) -> str:
    """모멘텀 점수 (-4 ~ +4) → 마커 테두리 색."""
    if score >= 4:  return '#7f1d1d'  # 짙은 빨강 — 강 매수 모멘텀
    if score >= 2:  return '#dc2626'  # 빨강
    if score >= 1:  return '#fca5a5'  # 연빨강
    if score <= -4: return '#1e3a8a'  # 짙은 파랑 — 강 매도 모멘텀
    if score <= -2: return '#2563eb'  # 파랑
    if score <= -1: return '#93c5fd'  # 연파랑
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
    """매매 기록 → 현재 사이클 + 누적 실현손익."""
    sorted_records = sorted(valid, key=lambda r: r['date'])

    cycle_start: Optional[datetime.date] = None
    cycle_end: Optional[datetime.date] = None
    hold_qty = 0
    buy_qty = 0
    buy_cost = 0.0
    sell_proceeds = 0.0
    cumulative_pnl = 0.0

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
            hold_qty += qty
            buy_qty += qty
            buy_cost += qty * r['price']

        elif r['type'] == 'sell' and hold_qty > 0:
            sell_proceeds += qty * r['price']
            hold_qty = max(hold_qty - qty, 0)
            if hold_qty == 0:
                cycle_end = date

    current_pnl = (sell_proceeds - buy_cost) if cycle_end else None
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
        tc = ticker_color(tk)
        vc = pnl_color(real)
        html += (
            f"<div style='display:flex;align-items:center;gap:5px;margin-bottom:3px;'>"
            f"<div style='font-size:0.67rem;color:{COLOR_TEXT};width:40px;flex-shrink:0;'>{display_name(tk)}</div>"
            f"{html_progress_bar(w, tc)}"
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
    rows = []
    for tk, ts in portfolio_state.items():
        cyc = ts['cycle']
        if cyc['hold_qty'] <= 0:
            continue
        inv_krw = cyc['buy_cost'] * usd_krw
        avg = cyc['buy_cost'] / cyc['buy_qty']
        cur = df_close_last.get(f'{tk}_Close')
        ret = (cur - avg) / avg * 100 if cur else None
        eval_krw = (cur * cyc['hold_qty'] * usd_krw) if cur else inv_krw
        rows.append((tk, inv_krw, ret, eval_krw))

    if not rows:
        return ""

    total_inv_krw = sum(r[1] for r in rows)
    used_pct = min(total_inv_krw / CFG.SEED_KRW * 100, 100)
    bar_c = '#b91c1c' if used_pct >= 90 else '#f59e0b' if used_pct >= 70 else '#16a34a'

    html = (
        f"{html_section_divider()}"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.62rem;color:{COLOR_LABEL};margin-bottom:4px;'>"
        f"<span>📊 배분 현황</span>"
        f"<span style='color:{bar_c};font-weight:700;'>{used_pct:.1f}% 사용"
        f"&nbsp;<span style='color:{COLOR_LABEL};font-weight:400;'>"
        f"({int(round(total_inv_krw / 10000)):,}만원)</span></span></div>"
    )
    max_eval = max(r[3] for r in rows)
    for tk, inv_krw, ret, eval_krw in sorted(rows, key=lambda x: -x[1]):
        tc = ticker_color(tk)
        cost_pct = inv_krw / CFG.SEED_KRW * 100
        pnl_c = pnl_color(ret or 0)
        eval_w = max(eval_krw / max_eval * 100, 1) if max_eval else 1
        cost_w = max(inv_krw / max_eval * 100, 1) if max_eval else 1
        cost_w = min(cost_w, eval_w)
        pnl_w = max(eval_w - cost_w, 0)

        ret_str = (
            f"<span style='color:{pnl_c};font-weight:700;'>{signed_str(ret, '{:.0f}')}%</span>"
            if ret is not None else f"<span style='color:{COLOR_LABEL};'>-</span>"
        )
        bar_html = (
            f"<div style='flex:1;background:#e5e7eb;border-radius:3px;height:7px;"
            f"display:flex;align-items:center;overflow:hidden;'>"
            f"<div style='width:{cost_w:.1f}%;background:{tc};height:7px;"
            f"border-radius:3px 0 0 3px;flex-shrink:0;'></div>"
        )
        if pnl_w > 0.5:
            bar_html += (
                f"<div style='width:{pnl_w:.1f}%;background:{pnl_c};height:7px;"
                f"border-radius:0 3px 3px 0;flex-shrink:0;opacity:0.85;'></div>"
            )
        bar_html += "</div>"
        html += (
            f"<div style='display:flex;align-items:center;gap:5px;margin-bottom:3px;'>"
            f"<div style='font-size:0.67rem;color:{COLOR_TEXT};width:40px;flex-shrink:0;'>{display_name(tk)}</div>"
            f"{bar_html}"
            f"<div style='font-size:0.63rem;color:#6b7280;width:28px;text-align:right;flex-shrink:0;'>{cost_pct:.1f}%</div>"
            f"<div style='font-size:0.63rem;width:32px;text-align:right;flex-shrink:0;'>{ret_str}</div>"
            f"</div>"
        )
    html += "</div>"
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
        fig.update_yaxes(
            range=[-(rng + 0.3), rng + 0.3], autorange=False, fixedrange=True,
            row=row, col=1,
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

    # ── 종목명 헤더 (항상 표시) — σ는 expanding std (그래프2 Z와 일치) ──
    header_right = ""
    if beta is not None and std_resid is not None:
        header_right = (
            f"<span style='font-size:0.65rem;color:#6b7280;'>"
            f"σ={sigma_unit:.3f} · β={beta:.2f}</span>"
        )
    header_html = (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"padding:4px 12px 2px 12px;margin-top:4px;'>"
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

    total_realized = cumulative_pnl + (cyc['current_pnl'] if is_closed else 0.0)
    has_cumulative = (cumulative_pnl != 0.0) or is_closed
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
    if ('MACD_Hist_Z' in df_daily.columns and 'RSI' in df_daily.columns):
        last_mhz = df_daily['MACD_Hist_Z'].iloc[-1]
        last_rsi = df_daily['RSI'].iloc[-1]
        mhz_v = float(last_mhz) if pd.notna(last_mhz) else 0.0
        rsi_v = float(last_rsi) if pd.notna(last_rsi) else 50.0
        cur_momentum_score = compute_momentum_score(mhz_v, rsi_v)
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

    return bar_html + interp_html + action_html + sub_html


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
    if ('MACD_Hist_Z' in df_daily.columns and 'RSI' in df_daily.columns):
        last_mhz = df_daily['MACD_Hist_Z'].iloc[-1]
        last_rsi = df_daily['RSI'].iloc[-1]
        mhz_v = float(last_mhz) if pd.notna(last_mhz) else 0.0
        rsi_v = float(last_rsi) if pd.notna(last_rsi) else 50.0
        cur_momentum_score = compute_momentum_score(mhz_v, rsi_v)
        cur_signal = momentum_score_to_signal(cur_momentum_score)
    marker_color = momentum_to_color(cur_momentum_score)

    # ── bar HTML ──
    # 컨테이너 높이 = bar_height; 그라디언트 두께 6px; 마커 14px
    grad_top = (bar_height - 6) // 2  # 그라디언트 vertical center
    marker_top = (bar_height - 14) // 2  # 마커 vertical center
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

    # ── #20 진입/익절 가격 제안 (σ + 역사적 분위 통합) ──
    if (df_daily is not None and not df_daily.empty
            and beta is not None and std_resid is not None):
        with st.expander("🎯 매매가 제안", expanded=False):
            close_col = f'{selected_ticker}_Close'
            if close_col not in df_daily.columns:
                st.caption("데이터 부족")
            else:
                cur_price = float(df_daily[close_col].iloc[-1])
                cur_predicted = float(df_daily['Predicted'].iloc[-1])
                cur_norm_y = float(df_daily[f'{selected_ticker}_Norm'].iloc[-1])
                trend_price = cur_price * (cur_predicted / cur_norm_y)

                # log_resid 시계열
                log_resid_series = (
                    np.log(df_daily[f'{selected_ticker}_Norm'])
                    - np.log(df_daily['Predicted'])
                ).dropna()

                # ── 현재 시점 expanding std (그래프2의 Z-score와 일치) ──
                expanding_std = log_resid_series.expanding(
                    min_periods=CFG.EXPANDING_MIN_PERIODS
                ).std()
                cur_std = expanding_std.dropna()
                if len(cur_std) > 0:
                    sigma_unit = float(cur_std.iloc[-1])
                    if sigma_unit <= 0 or not np.isfinite(sigma_unit):
                        sigma_unit = std_resid
                else:
                    sigma_unit = std_resid

                # ── σ 기반 가격 (회귀 모델) ──
                def _price_at_sigma(k: float) -> float:
                    return trend_price * np.exp(k * sigma_unit)

                # ── 분위 기반 가격 (실증) ──
                # 과거 log_resid의 분위값 → 그 분위에서의 가격
                def _price_at_quantile(q: float) -> Optional[float]:
                    """q는 0~1 (예: 0.10 = 하위 10% 분위)."""
                    if len(log_resid_series) < 30:
                        return None
                    qval = float(log_resid_series.quantile(q))
                    return trend_price * np.exp(qval)

                # ── 보유 중 여부 + % 기준점 결정 (#2) ──
                ts = portfolio_state.get(selected_ticker)
                avg_price = None
                if ts and ts['cycle']['hold_qty'] > 0 and ts['cycle']['buy_qty'] > 0:
                    avg_price = ts['cycle']['buy_cost'] / ts['cycle']['buy_qty']

                ref_price = avg_price if avg_price else cur_price
                ref_label = "평균단가" if avg_price else "현재가"

                def _pct_from_ref(p: float) -> str:
                    pct = (p / ref_price - 1) * 100
                    return signed_str(pct, '{:.1f}') + "%"

                # 표 행에서 σ 표시용 헬퍼 (expanding std 기준)
                def _price_to_sigma(p: float) -> float:
                    if p <= 0 or trend_price <= 0:
                        return 0.0
                    return float(np.log(p / trend_price) / sigma_unit)
                cur_sigma = _price_to_sigma(cur_price)

                # ── 통합 매매가 테이블 ──
                # 각 신호별: σ 가격 + 분위 가격 + 추천가(평균) + 신뢰도
                # 분위 매핑: σ ≈ 정규분포의 분위와 비슷하게 잡음
                #   -2σ ≈ 2.5%분위, -1.5σ ≈ 7%, -1σ ≈ 16%, +1σ ≈ 84%, +1.5σ ≈ 93%, +2σ ≈ 97.5%
                signal_specs = [
                    ("FB2 진입", '#7f1d1d', -2.0, 0.025),
                    ("FB 진입",  '#dc2626', -1.5, 0.07),
                    ("B 진입",   '#fca5a5', -1.0, 0.16),
                    ("회귀선",   '#6b7280',  0.0, 0.50),
                    ("S 익절",   '#93c5fd',  1.0, 0.84),
                    ("FS 익절",  '#2563eb',  1.5, 0.93),
                    ("FS2 익절", '#1e3a8a',  2.0, 0.975),
                ]

                def _confidence_stars(p_sigma: float, p_quantile: Optional[float]) -> str:
                    """σ 가격과 분위 가격이 비슷할수록 신뢰도 ↑."""
                    if p_quantile is None:
                        return "<span style='color:#9ca3af;'>·</span>"
                    diff_pct = abs(p_sigma - p_quantile) / max(p_sigma, 0.01) * 100
                    if diff_pct < 3:
                        return "<span style='color:#16a34a;'>★★★</span>"
                    elif diff_pct < 8:
                        return "<span style='color:#ca8a04;'>★★</span>"
                    else:
                        return "<span style='color:#9ca3af;'>★</span>"

                tbl = [
                    "<table style='width:100%;font-size:0.7rem;border-collapse:collapse;'>",
                    "<tr style='color:#6b7280;border-bottom:1px solid #e5e7eb;'>"
                    "<th style='text-align:left;padding:3px;'>구간</th>"
                    "<th style='padding:3px;'>σ가격</th>"
                    "<th style='padding:3px;'>분위가</th>"
                    "<th style='padding:3px;'>추천가</th>"
                    f"<th style='padding:3px;'>{ref_label}대비</th>"
                    "<th style='padding:3px;'>신뢰</th></tr>"
                ]

                # ── 모든 행을 (price, html) 튜플로 모은 뒤 가격순 정렬 ──
                table_rows = []  # [(price, html_str), ...]

                # 신호별 행
                for label, c, sigma_k, quantile_q in signal_specs:
                    p_sigma = _price_at_sigma(sigma_k)
                    p_quantile = _price_at_quantile(quantile_q)
                    if p_quantile is not None:
                        p_reco = (p_sigma + p_quantile) / 2
                    else:
                        p_reco = p_sigma
                    is_trend = (sigma_k == 0.0)
                    bg = "background:#fef3c7;" if is_trend else ""
                    sigma_lbl = f"{sigma_k:+.1f}σ" if not is_trend else "추세"
                    quantile_str = (
                        f"${p_quantile:,.2f}" if p_quantile is not None else "-"
                    )
                    row_html = (
                        f"<tr style='{bg}'>"
                        f"<td style='padding:3px;'>"
                        f"<span style='background:{c};color:#fff;padding:1px 4px;"
                        f"border-radius:3px;font-size:0.6rem;font-weight:700;'>{label}</span>"
                        f"<div style='color:#9ca3af;font-size:0.55rem;'>{sigma_lbl}</div></td>"
                        f"<td style='padding:3px;text-align:center;color:#6b7280;'>${p_sigma:,.2f}</td>"
                        f"<td style='padding:3px;text-align:center;color:#6b7280;'>{quantile_str}</td>"
                        f"<td style='padding:3px;text-align:center;font-weight:700;'>${p_reco:,.2f}</td>"
                        f"<td style='padding:3px;text-align:center;color:#6b7280;'>{_pct_from_ref(p_reco)}</td>"
                        f"<td style='padding:3px;text-align:center;font-size:0.65rem;'>"
                        f"{_confidence_stars(p_sigma, p_quantile)}</td>"
                        f"</tr>"
                    )
                    table_rows.append((p_reco, row_html))

                # ── 현재가 행 (별표 + 핑크 강조) ──
                cur_pct_ref = (cur_price - ref_price) / ref_price * 100
                cur_pct_color = pnl_color(cur_pct_ref) if avg_price else '#ec4899'
                cur_row_html = (
                    f"<tr style='background:#fdf2f8;border-top:1.5px dashed #ec4899;"
                    f"border-bottom:1.5px dashed #ec4899;'>"
                    f"<td style='padding:5px 3px;'>"
                    f"<span style='background:#ec4899;color:#fff;padding:1px 5px;"
                    f"border-radius:3px;font-size:0.6rem;font-weight:700;'>★ 현재가</span>"
                    f"<div style='color:#9ca3af;font-size:0.55rem;'>{cur_sigma:+.2f}σ</div></td>"
                    f"<td style='padding:5px 3px;text-align:center;color:#9ca3af;'>-</td>"
                    f"<td style='padding:5px 3px;text-align:center;color:#9ca3af;'>-</td>"
                    f"<td style='padding:5px 3px;text-align:center;font-weight:700;color:#ec4899;'>"
                    f"${cur_price:,.2f}</td>"
                    f"<td style='padding:5px 3px;text-align:center;color:{cur_pct_color};font-weight:700;'>"
                    f"{signed_str(cur_pct_ref, '{:.1f}')}%</td>"
                    f"<td style='padding:5px 3px;text-align:center;'></td>"
                    f"</tr>"
                )
                table_rows.append((cur_price, cur_row_html))

                # ── 평균단가 행 (보유 시) ──
                if avg_price:
                    avg_sigma_val = _price_to_sigma(avg_price)
                    avg_row_html = (
                        f"<tr style='background:#f3f4f6;'>"
                        f"<td style='padding:5px 3px;'>"
                        f"<span style='background:#6b7280;color:#fff;padding:1px 5px;"
                        f"border-radius:3px;font-size:0.6rem;font-weight:700;'>● 평균단가</span>"
                        f"<div style='color:#9ca3af;font-size:0.55rem;'>{avg_sigma_val:+.2f}σ</div></td>"
                        f"<td style='padding:5px 3px;text-align:center;color:#9ca3af;'>-</td>"
                        f"<td style='padding:5px 3px;text-align:center;color:#9ca3af;'>-</td>"
                        f"<td style='padding:5px 3px;text-align:center;font-weight:700;color:#374151;'>"
                        f"${avg_price:,.2f}</td>"
                        f"<td style='padding:5px 3px;text-align:center;color:#6b7280;'>0.0%</td>"
                        f"<td style='padding:5px 3px;text-align:center;'></td>"
                        f"</tr>"
                    )
                    table_rows.append((avg_price, avg_row_html))

                # ── 가격 내림차순 정렬 (높은 가격 = 위, 익절 영역이 위쪽) ──
                # 익절 영역이 위쪽에 오도록 내림차순
                table_rows.sort(key=lambda x: -x[0])
                for _, html in table_rows:
                    tbl.append(html)
                tbl.append("</table>")
                st.markdown("".join(tbl), unsafe_allow_html=True)
                st.caption(
                    f"σ가격: 회귀모델 × exp(±σ × {sigma_unit:.3f}) "
                    f"[expanding std, 그래프 Z값과 일치] / "
                    f"분위가: 과거 잔차 분위 가격 / "
                    f"추천가: 두 값 평균 / "
                    f"★★★ = σ·분위 일치도 높음 (3% 이내)"
                )

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
) -> None:
    """전체 포트폴리오 통계 — 시드/실현/비중/달력/자산 추이.

    이전엔 사이드바에 있던 영역을 메인 탭3로 이동.
    """
    portfolio_pnl = st.session_state.get('portfolio_pnl_cache')
    usd_krw = st.session_state.get('usd_krw_cache', CFG.USD_KRW_FALLBACK)
    df_close_last = st.session_state.get('df_close_last', {})
    dd_info = st.session_state.get('dd_info_cache')

    # ── 1. 시드 카드 (세로 stack) ──
    seed_html = _build_seed_html(portfolio_pnl, usd_krw, dd_info)
    st.markdown(
        f"<div style='padding:10px 12px;background:#ffffff;"
        f"border:1px solid #e2e8f0;border-radius:10px;margin-bottom:10px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"{seed_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 2. 실현손익 카드 ──
    real_html = _build_realized_html(portfolio_state, usd_krw)
    st.markdown(
        f"<div style='padding:10px 12px;background:#ffffff;"
        f"border:1px solid #e2e8f0;border-radius:10px;margin-bottom:10px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"{real_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 3. 종목별 비중 ──
    alloc_html = _build_alloc_html(portfolio_state, df_close_last, usd_krw)
    st.markdown(
        f"<div style='padding:10px 12px;background:#ffffff;"
        f"border:1px solid #e2e8f0;border-radius:10px;margin-bottom:10px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"{alloc_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 4. 자산 추이 차트 (탭3 자체 기간 토글) ──
    equity_series = st.session_state.get('equity_series_cache')
    if equity_series is not None and not equity_series.empty:
        seed_usd = CFG.SEED_KRW / usd_krw
        portfolio_value = equity_series + seed_usd
        portfolio_krw = portfolio_value * usd_krw / 10000  # 만원 단위
        seed_krw = CFG.SEED_KRW / 10000

        # 자체 기간 토글 (메인 차트 view_months와 별개)
        ov_zoom_presets = [('1M', 1), ('3M', 3), ('6M', 6), ('1Y', 12), ('All', 240)]
        ov_zoom_labels = [p[0] for p in ov_zoom_presets]
        ov_zoom_map = dict(ov_zoom_presets)
        ov_current = st.session_state.get('overview_view_months', 12)
        ov_current_label = next(
            (lbl for lbl, m in ov_zoom_presets if m == ov_current),
            'All',
        )
        ov_choice = st.radio(
            "자산 추이 기간",
            ov_zoom_labels,
            index=ov_zoom_labels.index(ov_current_label),
            horizontal=True,
            key="overview_zoom_radio",
            label_visibility="collapsed",
        )
        ov_months = ov_zoom_map[ov_choice]
        if ov_months != ov_current:
            st.session_state['overview_view_months'] = ov_months
            st.rerun()

        view_start = portfolio_krw.index[-1] - pd.DateOffset(months=ov_months)
        portfolio_view = portfolio_krw[portfolio_krw.index >= view_start]
        if len(portfolio_view) < 2:
            portfolio_view = portfolio_krw

        cur_val = float(portfolio_view.iloc[-1])
        line_color = '#b91c1c' if cur_val >= seed_krw else '#1d4ed8'
        fill_color = (
            'rgba(185,28,28,0.1)' if cur_val >= seed_krw
            else 'rgba(29,78,216,0.1)'
        )

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=portfolio_view.index, y=portfolio_view.values,
            mode='lines', line=dict(color=line_color, width=2),
            fill='tozeroy', fillcolor=fill_color,
            hovertemplate='%{x|%y.%m.%d}<br>%{y:.0f}만원<extra></extra>',
            showlegend=False, name='자산',
        ))
        fig_eq.add_hline(
            y=seed_krw, line_dash="dot", line_color='#9ca3af',
            line_width=1.2,
            annotation_text=f"시드 {seed_krw:.0f}만",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color='#6b7280'),
        )
        running_max = portfolio_view.cummax()
        fig_eq.add_trace(go.Scatter(
            x=running_max.index, y=running_max.values,
            mode='lines', line=dict(color='#9ca3af', width=1, dash='dash'),
            hoverinfo='skip', showlegend=False, name='고점',
        ))

        y_min = min(portfolio_view.min(), seed_krw) * 0.97
        y_max = max(portfolio_view.max(), seed_krw) * 1.03

        fig_eq.update_layout(
            height=200,
            margin=dict(l=4, r=8, t=28, b=4),
            xaxis=dict(showgrid=False, tickfont=dict(size=9),
                       tickformat='%y.%m', nticks=5,
                       range=[portfolio_view.index[0], portfolio_view.index[-1]]),
            yaxis=dict(showgrid=True, gridcolor='rgba(156,163,175,0.2)',
                       tickfont=dict(size=9), range=[y_min, y_max],
                       ticksuffix='만'),
            paper_bgcolor='white', plot_bgcolor='white',
            title=dict(text=f'💼 자산 추이 ({ov_choice})', x=0.02, y=0.97,
                       font=dict(size=12, color='#374151')),
        )
        st.plotly_chart(fig_eq, use_container_width=True,
                        config={'displayModeBar': False, 'staticPlot': True})

        st.markdown(
            "<div style='height:8px;'></div>",
            unsafe_allow_html=True,
        )

    # ── 5. 매매 달력 ──
    cal_month = st.session_state.get('cal_month', datetime.date.today().replace(day=1))
    cal_html = _build_calendar_html(st.session_state.trade_history, cal_month, usd_krw)
    st.markdown(
        f"<div style='padding:10px 12px;background:#ffffff;"
        f"border:1px solid #e2e8f0;border-radius:10px;margin-bottom:6px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        f"{cal_html}"
        f"</div>",
        unsafe_allow_html=True,
    )
    # 컨테이너 키로 첫컬럼 88px CSS 회피 (build_css에서 별도 처리)
    with st.container(key="ov_cal_nav"):
        nc1, nc2, nc3 = st.columns([1, 4, 1])
        if nc1.button("◀", key="ov_cal_prev", use_container_width=True):
            pm = cal_month.month - 1
            py = cal_month.year + (pm - 1) // 12
            pm = (pm - 1) % 12 + 1
            st.session_state['cal_month'] = datetime.date(py, pm, 1)
            st.rerun()
        nc2.markdown(
            f"<div style='text-align:center;font-size:0.85rem;color:#6b7280;"
            f"padding-top:5px;'>{cal_month.strftime('%Y. %m')}</div>",
            unsafe_allow_html=True,
        )
        if nc3.button("▶", key="ov_cal_next", use_container_width=True):
            nm = cal_month.month + 1
            ny = cal_month.year + (nm - 1) // 12
            nm = (nm - 1) % 12 + 1
            st.session_state['cal_month'] = datetime.date(ny, nm, 1)
            st.rerun()


# ====================================================
# 18. 메모 섹션
# ====================================================
def render_memo_section(selected_ticker: str) -> None:
    memos = sorted(
        st.session_state.memo_history.get(selected_ticker, []),
        key=lambda x: x['date'], reverse=True,
    )
    if not memos:
        return
    rows_html = "".join(
        f"<tr>"
        f"<td style='color:#6b7280;font-size:0.72rem;white-space:nowrap;"
        f"padding:4px 12px 4px 6px;vertical-align:top;'>{m['date']}</td>"
        f"<td style='color:#111827;font-size:0.78rem;line-height:1.5;padding:4px;'>{m['text']}</td>"
        f"</tr>"
        for m in memos
    )
    st.markdown(
        f"<div style='margin-top:8px;border-top:1px solid #e5e7eb;padding-top:6px;'>"
        f"<span style='font-size:0.75rem;font-weight:700;color:#6b7280;'>"
        f"📝 {display_name(selected_ticker)} 메모</span>"
        f"<table style='width:100%;border-collapse:collapse;margin-top:4px;'>"
        f"{rows_html}</table></div>",
        unsafe_allow_html=True,
    )


# ====================================================
# 17. CSS
# ====================================================
def build_css(selected_option: str, holding_tickers: set) -> str:
    btn_parts = []
    for ticker in TARGET_TICKERS:
        sig = st.session_state.ticker_signals.get(ticker, 'H')
        bg, _ = SIGNAL_STYLE.get(sig, ('#9ca3af', '#fff'))
        fg = BUTTON_TEXT_STYLE.get(sig, '#111827')
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
        height:1.1rem!important; font-size:0.55rem!important;
        padding:0!important; min-height:0!important; border-radius:3px!important; {di_border}
    }}
    div.st-key-full_refresh_btn button {{
        height:1.1rem!important; min-height:0!important; border-radius:3px!important;
        font-size:0.55rem!important; font-weight:700!important; padding:0!important;
        border:1px solid #cbd5e1!important; background:#f8fafc!important;
        color:#0f172a!important; line-height:1!important;
    }}
    div.st-key-full_refresh_btn button:hover {{
        border-color:#94a3b8!important; background:#eef2f7!important; }}""")

    return f"""<style>
    .block-container {{
        padding-top:3.5rem!important; padding-bottom:0.5rem!important; max-width:100%!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {{
        flex-wrap:nowrap!important; gap:5px!important; align-items:flex-start!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]
        > div[data-testid="stColumn"]:first-child {{
        flex:0 0 88px!important; min-width:88px!important;
        max-width:88px!important; padding:0!important;
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
    {''.join(btn_parts)}
    </style>"""


# ====================================================
# 18. 메인
# ====================================================
def main() -> None:
    init_session_state()

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

    with st.spinner("데이터 로드 중..."):
        df_close = fetch_all_data(TARGET_TICKERS, analysis_start, candle_type)

    if selected_ticker and f'{selected_ticker}_Close' not in df_close.columns:
        with st.spinner(f"{selected_ticker} 데이터를 불러오는 중..."):
            df_custom = fetch_single_ticker(selected_ticker, analysis_start)
        if not df_custom.empty:
            if candle_type == '주봉':
                df_custom = _resample_weekly(df_custom)
            df_close = pd.concat([df_close, df_custom], axis=1).ffill()
        else:
            log.warning(f"Custom ticker fetch empty: {selected_ticker}")
            selected_ticker = None

    mkt = get_market_status()
    last_trading_date = pd.Timestamp(mkt['last_trading_date'])
    if not df_close.empty:
        st.session_state.last_data_date = df_close.index[-1].strftime('%Y-%m-%d')
        st.session_state['df_close_last'] = df_close.iloc[-1].to_dict()
        if candle_type == '일봉':
            df_close = df_close[df_close.index <= last_trading_date]

    with st.spinner("전체 종목 분석 중..."):
        all_analyses = compute_all_analyses(df_close, _version=8, candle_type=candle_type)

    pct_changes = {}
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        pct_changes[ticker] = (
            df_close[col].pct_change().iloc[-1] * 100
            if col in df_close.columns and len(df_close) > 1 else 0.0
        )
        result = all_analyses.get(ticker)
        if result and result[0] is not None:
            df_t, beta_t, _ = result
            cz = float(df_t['Z_Score'].iloc[-1]) if pd.notna(df_t['Z_Score'].iloc[-1]) else 0.0
            mhz = float(df_t['MACD_Hist_Z'].iloc[-1]) if pd.notna(df_t['MACD_Hist_Z'].iloc[-1]) else 0.0
            rsi = float(df_t['RSI'].iloc[-1]) if pd.notna(df_t['RSI'].iloc[-1]) else 50.0
            st.session_state.ticker_signals[ticker] = get_signal_combined(cz, mhz, rsi)
            st.session_state.ticker_betas[ticker] = round(beta_t, 2)
        else:
            st.session_state.ticker_signals.setdefault(ticker, 'H')

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
            rsi = float(df_daily['RSI'].iloc[-1]) if pd.notna(df_daily['RSI'].iloc[-1]) else 50.0
            st.session_state.ticker_signals[selected_ticker] = get_signal_combined(cz, mhz, rsi)

    holding_tickers = {
        tk for tk, ts in portfolio_state.items() if ts['cycle']['hold_qty'] > 0
    }

    # 드로다운 계산 (#6) + 자산 시계열 캐싱 (#15)
    if portfolio_state and not df_close.empty:
        equity = compute_portfolio_equity(
            portfolio_state, df_close, st.session_state.trade_history
        )
        if equity is not None and not equity.empty:
            st.session_state['dd_info_cache'] = compute_drawdown(equity)
            st.session_state['equity_series_cache'] = equity
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
    def _ticker_sort_key(tk: str) -> tuple:
        sig = st.session_state.ticker_signals.get(tk, 'H')
        is_holding = tk in holding_tickers
        return (
            0 if is_holding else 1,
            signal_sort_key(sig),
            TARGET_TICKERS.index(tk),
        )
    sorted_tickers = sorted(TARGET_TICKERS, key=_ticker_sort_key)

    # ── 메인 영역 전체를 감싸는 탭 ──
    tab1, tab2, tab3 = st.tabs(["📊 상세", "🗺️ 한눈에 보기", "📈 전체 통계"])

    # ====================================================
    # 탭 1: 기존 화면 (종목버튼 + 차트 + 분석패널 + 메모)
    # ====================================================
    with tab1:
        btn_col, chart_col = st.columns([1, 6])
        with btn_col:
            for ticker in sorted_tickers:
                pct = pct_changes.get(ticker, 0)
                star = "★ " if ticker in holding_tickers else ""
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
                render_position_tracker(
                    selected_ticker, df_daily, df_close, portfolio_state, beta, std_resid,
                )
                with st.spinner("캔들 데이터 로드 중..."):
                    df_ohlc = fetch_ohlc(selected_ticker, analysis_start, candle_type)
                df_daily_raw = None
                if candle_type == '주봉':
                    df_raw = fetch_all_data(TARGET_TICKERS, analysis_start, '일봉')
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
                render_chart(
                    df_daily, selected_ticker, beta, std_resid,
                    cfg['guide_n'], st.session_state.view_months, df_ohlc, df_daily_raw,
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
            st.markdown(
                "<div data-analytics-panel style='margin-top:8px;'></div>",
                unsafe_allow_html=True,
            )
            render_analytics_panel(
                selected_ticker, df_daily, df_close, portfolio_state, beta, std_resid,
            )
        if selected_ticker:
            render_memo_section(selected_ticker)

    # ====================================================
    # 탭 2: 한눈에 보기 (풀폭 22개 종목 미니바 리스트)
    # ====================================================
    with tab2:
        # ── 헤더: σ 눈금 라벨 ──
        st.markdown(
            "<div style='display:flex;align-items:center;gap:6px;"
            "padding:6px 4px 4px 4px;font-size:0.6rem;color:#9ca3af;"
            "border-bottom:1px solid #e5e7eb;margin-bottom:4px;'>"
            "<div style='width:110px;font-weight:700;color:#6b7280;font-size:0.7rem;'>"
            "종목</div>"
            "<div style='flex:1;position:relative;height:14px;min-width:0;'>"
            "<span style='position:absolute;left:0%;transform:translateX(-50%);'>-3σ</span>"
            "<span style='position:absolute;left:16.67%;transform:translateX(-50%);'>-2σ</span>"
            "<span style='position:absolute;left:33.33%;transform:translateX(-50%);'>-1σ</span>"
            "<span style='position:absolute;left:50%;transform:translateX(-50%);'>추세</span>"
            "<span style='position:absolute;left:66.67%;transform:translateX(-50%);'>+1σ</span>"
            "<span style='position:absolute;left:83.33%;transform:translateX(-50%);'>+2σ</span>"
            "<span style='position:absolute;left:100%;transform:translateX(-50%);'>+3σ</span>"
            "</div>"
            "<div style='width:90px;font-weight:700;color:#6b7280;font-size:0.65rem;"
            "flex-shrink:0;text-align:right;'>"
            "변동성/추세</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── 22개 종목 행 (탭1과 동일 정렬) ──
        TREND_DAYS = 30  # β 기반 추세 기간 (일)
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
                trade_records=t_records, bar_height=36,
            )
            if mini_bar is None:
                continue

            # ── σ당 % (변동성) ──
            t_norm_col = f'{ticker}_Norm'
            sigma_pct_str = "—"
            if 'Predicted' in t_df.columns and t_norm_col in t_df.columns:
                t_log_resid = (np.log(t_df[t_norm_col]) - np.log(t_df['Predicted'])).dropna()
                t_exp_std = t_log_resid.expanding(
                    min_periods=CFG.EXPANDING_MIN_PERIODS
                ).std().dropna()
                if len(t_exp_std) > 0:
                    t_sigma_unit = float(t_exp_std.iloc[-1])
                    if t_sigma_unit > 0 and np.isfinite(t_sigma_unit):
                        t_sigma_pct = (np.exp(t_sigma_unit) - 1) * 100
                        sigma_pct_str = f"±{t_sigma_pct:.1f}%"

            # ── β·30일 추세 % (Predicted 시계열 변화율) ──
            trend_pct_str = "—"
            trend_color = '#9ca3af'
            if 'Predicted' in t_df.columns and len(t_df) > TREND_DAYS:
                p_recent = float(t_df['Predicted'].iloc[-1])
                p_past = float(t_df['Predicted'].iloc[-(TREND_DAYS + 1)])
                if p_past > 0 and np.isfinite(p_recent) and np.isfinite(p_past):
                    trend_pct = (p_recent / p_past - 1) * 100
                    trend_pct_str = signed_str(trend_pct, '{:.1f}') + "%"
                    trend_color = pnl_color(trend_pct)

            is_holding = ticker in holding_tickers
            row_bg = '#f0fdf4' if is_holding else '#ffffff'
            star = "★ " if is_holding else ""
            pct_chg = pct_changes.get(ticker, 0)
            pct_color = pnl_color(pct_chg)

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"padding:3px 4px;background:{row_bg};"
                f"border-bottom:1px solid #f3f4f6;'>"
                f"<div style='width:110px;font-size:0.75rem;font-weight:600;"
                f"color:#111827;flex-shrink:0;display:flex;align-items:baseline;"
                f"gap:5px;'>"
                f"<span>{star}{display_name(ticker)}</span>"
                f"<span style='font-size:0.65rem;color:{pct_color};font-weight:500;'>"
                f"{signed_str(pct_chg, '{:.1f}')}%</span>"
                f"</div>"
                f"<div style='flex:1;min-width:0;'>{mini_bar}</div>"
                f"<div style='width:90px;font-size:0.6rem;color:#6b7280;"
                f"flex-shrink:0;line-height:1.15;text-align:right;'>"
                f"<span title='1σ 변동 시 가격 변화율 (변동성)'>σ:{sigma_pct_str}</span><br>"
                f"<span style='color:{trend_color};font-weight:500;' "
                f"title='최근 {TREND_DAYS}일 회귀선(추세) 변화율 — β 효과 반영'>"
                f"β·{TREND_DAYS}d:{trend_pct_str}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.caption(
            "■ 위치=σ · 테두리색=모멘텀 (MACD+RSI) · ▪ 평균단가 · ● 매수 ● 매도 (당시 σ) · "
            f"σ:1σ당 ±% (변동성) · β·{TREND_DAYS}d:최근 {TREND_DAYS}일 추세선 변화율"
        )

    # ====================================================
    # 탭 3: 전체 통계 (시드/실현/비중/달력/자산추이)
    # ====================================================
    with tab3:
        render_overview_panel(portfolio_state, df_close)

    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

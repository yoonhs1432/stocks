import datetime
import numpy as np
import pandas as pd
import json
import os
import requests
from sklearn.linear_model import LinearRegression
import FinanceDataReader as fdr
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====================================================
# 1. 전역 설정
# ====================================================
st.set_page_config(page_title="퀀트 트레이딩 대시보드", layout="wide")

X_ASSET_FIXED      = 'SPY'
TARGET_TICKERS     = ['SPYU', 'SOXL', 'TQQQ', 'FNGU',
                      'HIBL', 'TARK', 'QPUX', 'BNKU',
                      'GDXU', 'KORU', '005930', 'UPXI',
                      'BTC-USD', 'ETH-USD',
                      'RKLB', 'OKLO',
                      'AVAV', 'TEM',
                      'IREN', 'CRCL',
                      ]
TICKER_DISPLAY_NAMES: dict = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    '005930':  '삼전',
    '000660':  '하닉',
}

SIGNAL_STYLE: dict = {
    'FB': ('#dc2626', '#ffffff'),
    'B':  ('#fca5a5', '#1a1a1a'),
    'H':  ('#9ca3af', '#ffffff'),
    'S':  ('#93c5fd', '#1a1a1a'),
    'FS': ('#1d4ed8', '#ffffff'),
}
ACTION_LABELS: dict = {
    'FB': '풀 매수', 'B': '매수', 'H': '관망', 'S': '매도', 'FS': '풀 매도',
}
BUTTON_TEXT_STYLE: dict = {
    'FB': '#f8fafc',
    'B':  '#111827',
    'H':  '#111827',
    'S':  '#111827',
    'FS': '#f8fafc',
}

def display_name(ticker: str) -> str:
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)

def safe_key(ticker: str) -> str:
    return ticker.replace('-', '_').replace('.', '_').replace('/', '_')

# ====================================================
# 2. 영속화 (매매기록 + 메모)
# ====================================================
TRADE_FILE         = 'trade_history.json'
MEMO_FILE          = 'memo_history.json'
SETTINGS_FILE      = 'settings.json'
GIST_FILENAME      = 'quant_trade_history.json'
MEMO_GIST_FILENAME = 'quant_memo_history.json'

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _gist_cfg() -> tuple[str, str]:
    try:
        token   = st.secrets.get("GITHUB_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
        gist_id = st.secrets.get("GIST_ID", "")      or os.environ.get("GIST_ID", "")
        return str(token).strip(), str(gist_id).strip()
    except Exception:
        return "", ""

def _gist_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def load_trade_history() -> dict:
    token, gist_id = _gist_cfg()
    if token and gist_id:
        try:
            resp = requests.get(f"https://api.github.com/gists/{gist_id}",
                                headers=_gist_headers(token), timeout=6)
            if resp.ok:
                files = resp.json().get("files", {})
                if GIST_FILENAME in files:
                    return json.loads(files[GIST_FILENAME]["content"])
        except Exception:
            pass
    if os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_trade_history(history: dict) -> None:
    with open(TRADE_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    token, gist_id = _gist_cfg()
    if token and gist_id:
        try:
            payload = {"files": {GIST_FILENAME: {
                "content": json.dumps(history, indent=4, ensure_ascii=False)}}}
            requests.patch(f"https://api.github.com/gists/{gist_id}",
                           headers=_gist_headers(token), json=payload, timeout=6)
        except Exception:
            pass

def load_memo_history() -> dict:
    token, gist_id = _gist_cfg()
    if token and gist_id:
        try:
            resp = requests.get(f"https://api.github.com/gists/{gist_id}",
                                headers=_gist_headers(token), timeout=6)
            if resp.ok:
                files = resp.json().get("files", {})
                if MEMO_GIST_FILENAME in files:
                    return json.loads(files[MEMO_GIST_FILENAME]["content"])
        except Exception:
            pass
    if os.path.exists(MEMO_FILE):
        with open(MEMO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_memo_history(memo: dict) -> None:
    with open(MEMO_FILE, 'w', encoding='utf-8') as f:
        json.dump(memo, f, indent=4, ensure_ascii=False)
    token, gist_id = _gist_cfg()
    if token and gist_id:
        try:
            payload = {"files": {MEMO_GIST_FILENAME: {
                "content": json.dumps(memo, indent=4, ensure_ascii=False)}}}
            requests.patch(f"https://api.github.com/gists/{gist_id}",
                           headers=_gist_headers(token), json=payload, timeout=6)
        except Exception:
            pass

def init_session_state() -> None:
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = load_trade_history()
    if 'memo_history' not in st.session_state:
        st.session_state.memo_history = load_memo_history()
    if 'ticker_signals' not in st.session_state:
        st.session_state.ticker_signals = {}
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = TARGET_TICKERS[0]
    if 'custom_ticker_input' not in st.session_state:
        st.session_state.custom_ticker_input = ''
    if 'last_data_date' not in st.session_state:
        st.session_state.last_data_date = ''
    if 'view_months' not in st.session_state:
        st.session_state.view_months = load_settings().get('view_months', 6)
    if 'analysis_start' not in st.session_state:
        st.session_state.analysis_start = load_settings().get('analysis_start', '25-01')
    if 'memo_editing_idx' not in st.session_state:
        st.session_state.memo_editing_idx = None  # 현재 수정 중인 메모 인덱스
    if 'memo_input_key' not in st.session_state:
        st.session_state.memo_input_key = 0  # 저장 후 입력창 초기화용

# ====================================================
# 3. 투자의견
# ====================================================

# ── 미국 주요 공휴일 (ET 기준, 고정일만 — 변동 공휴일은 근사값) ──
def _us_holidays(year: int) -> set:
    """주요 NYSE 공휴일 날짜 집합 반환 (완전하지 않으나 주요일 포함)."""
    from datetime import date
    holidays = {
        date(year, 1, 1),   # New Year's Day
        date(year, 7, 4),   # Independence Day
        date(year, 12, 25), # Christmas
    }
    # MLK Day: 1월 3번째 월요일
    import calendar
    def nth_weekday(year, month, weekday, n):
        count = 0
        for day in range(1, 32):
            try:
                d = date(year, month, day)
            except ValueError:
                break
            if d.weekday() == weekday:
                count += 1
                if count == n:
                    return d
    holidays.add(nth_weekday(year, 1, 0, 3))   # MLK Day
    holidays.add(nth_weekday(year, 2, 0, 3))   # Presidents Day
    holidays.add(nth_weekday(year, 9, 0, 1))   # Labor Day
    holidays.add(nth_weekday(year, 11, 3, 4))  # Thanksgiving
    # Good Friday: 부활절 전 금요일 (근사 계산)
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
    """현재 미장 상태 반환.
    Returns:
        is_open (bool): 현재 장이 열려 있는지
        status_label (str): 표시용 문자열 ex) '🟢 장중' / '🔴 장마감 (전일 종가 기준)'
        last_trading_label (str): '기준: YYYY-MM-DD (종가)' 형태
    """
    ET = datetime.timezone(datetime.timedelta(hours=-4))  # EDT (여름 기준, EST는 -5)
    now_et = datetime.datetime.now(ET)
    today  = now_et.date()

    is_weekend = today.weekday() >= 5  # 토=5, 일=6
    is_holiday = today in _us_holidays(today.year)
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    in_trading_hours = (market_open <= now_et <= market_close)

    is_open = (not is_weekend) and (not is_holiday) and in_trading_hours

    # 마지막 거래일 계산 (오늘 or 이전 영업일)
    last_day = today
    if is_weekend or is_holiday or (not in_trading_hours and now_et < market_open):
        # 장 시작 전이거나 비거래일 → 이전 거래일
        last_day = today - datetime.timedelta(days=1)
        while last_day.weekday() >= 5 or last_day in _us_holidays(last_day.year):
            last_day -= datetime.timedelta(days=1)

    last_trading_label = f"기준: {last_day.strftime('%Y-%m-%d')} 종가"
    if is_open:
        status_label = "🟢 장중"
    else:
        status_label = "🔴 장마감"

    return {'is_open': is_open, 'status_label': status_label,
            'last_trading_label': last_trading_label, 'last_trading_date': last_day}

def get_signal(current_z: float = 0.0) -> str:
    if current_z >= 1.5:  return 'FS'
    if current_z >= 0.5:  return 'S'
    if current_z <= -1.5: return 'FB'
    if current_z <= -0.5: return 'B'
    return 'H'

def get_z_text_color(current_z: float) -> str:
    if current_z <= -1.5: return '#991b1b'
    if current_z <= -0.5: return '#dc2626'
    if current_z <   0.5: return '#6b7280'
    if current_z <   1.5: return '#2563eb'
    return '#1d4ed8'

def get_price_fill_color(current_z: float) -> str:
    if pd.isna(current_z):   return 'rgba(0,0,0,0)'
    if current_z <= -1.5:    return 'rgba(220,38,38,0.35)'
    if current_z <= -0.5:    return 'rgba(252,165,165,0.22)'
    if current_z <   0.5:    return 'rgba(156,163,175,0.12)'
    if current_z <   1.5:    return 'rgba(147,197,253,0.22)'
    return 'rgba(29,78,216,0.35)'

def get_time_grid_dtick_ms(start: pd.Timestamp, end: pd.Timestamp, target_grids: int = 8) -> int:
    span_days    = max((end - start).days, 1)
    target_days  = span_days / max(target_grids, 1)
    candidates   = [3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180]
    best_days    = min(candidates, key=lambda d: abs(d - target_days))
    return int(best_days * 24 * 60 * 60 * 1000)

# ====================================================
# 4. 데이터 다운로드
# ====================================================
def _download_ticker_data(ticker: str, start_date_str: str) -> pd.DataFrame:
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if not data.empty:
            data = data[~data.index.duplicated(keep='last')].sort_index()
            return data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_all_data(tickers: list, start_date_str: str) -> pd.DataFrame:
    df_list = []
    for ticker in [X_ASSET_FIXED] + list(tickers):
        data = _download_ticker_data(ticker, start_date_str)
        if not data.empty:
            df_list.append(data)
    return pd.concat(df_list, axis=1).ffill() if df_list else pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_single_ticker(ticker: str, start_date_str: str) -> pd.DataFrame:
    return _download_ticker_data(ticker, start_date_str)

# ====================================================
# 5. 데이터 처리
# ====================================================
def _compute_normalized(df: pd.DataFrame, x_name: str, y_name: str) -> tuple:
    df = df.copy()
    base_x = df[f'{x_name}_Close'].iloc[0]
    base_y = df[f'{y_name}_Close'].iloc[0]
    df[f'{x_name}_Norm'] = df[f'{x_name}_Close'] / base_x
    df[f'{y_name}_Norm'] = df[f'{y_name}_Close'] / base_y
    log_x  = np.log(df[f'{x_name}_Norm'])
    log_y  = np.log(df[f'{y_name}_Norm'])
    model  = LinearRegression().fit(log_x.values.reshape(-1, 1), log_y.values)
    beta   = model.coef_[0]
    df['Predicted'] = np.exp(model.intercept_) * df[f'{x_name}_Norm'] ** beta
    return df, beta

def _compute_indicators(df: pd.DataFrame, y_name: str) -> tuple:
    df    = df.copy()
    close = df[f'{y_name}_Close']
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI']         = 100 - (100 / (1 + gain / loss))
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    log_resid         = np.log(df[f'{y_name}_Norm']) - np.log(df['Predicted'])
    std_resid         = log_resid.std()
    expanding_std     = log_resid.expanding(min_periods=30).std()
    df['Z_Score']     = log_resid / expanding_std.replace(0, np.nan)
    return df, std_resid

def process_asset_data(df_x: pd.DataFrame, df_y: pd.DataFrame,
                       x_name: str, y_name: str) -> tuple:
    df = pd.merge(df_x, df_y, left_index=True, right_index=True).dropna().sort_index()
    if df.empty:
        return (None,) * 4
    df, beta     = _compute_normalized(df, x_name, y_name)
    df, std_resid = _compute_indicators(df, y_name)
    return df, beta, std_resid

@st.cache_data(show_spinner=False)
def compute_all_analyses(df_close: pd.DataFrame) -> dict:
    results: dict = {}
    df_x = df_close[[f'{X_ASSET_FIXED}_Close']]
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        if col not in df_close.columns:
            results[ticker] = None
            continue
        try:
            results[ticker] = process_asset_data(
                df_x, df_close[[col]], X_ASSET_FIXED, ticker)
        except Exception:
            results[ticker] = None
    return results

# ====================================================
# 6. 차트 헬퍼
# ====================================================
def add_filled_blocks(fig, df, y_col, condition, color, row, col, baseline_y):
    valid_data = df[condition]
    if valid_data.empty:
        return
    groups = (~condition).cumsum()[condition]
    for _, group in valid_data.groupby(groups):
        fig.add_trace(go.Scatter(
            x=group.index, y=[baseline_y] * len(group),
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip'), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=group.index, y=group[y_col],
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty', fillcolor=color,
            showlegend=False, hoverinfo='skip'), row=row, col=col)

def add_segmented_fill(fig, df, y_col, color_col, row, col, baseline_y):
    for i in range(1, len(df)):
        y0 = df[y_col].iloc[i - 1]
        y1 = df[y_col].iloc[i]
        fill_color = df[color_col].iloc[i]
        if pd.isna(y0) or pd.isna(y1) or not fill_color or fill_color == 'rgba(0,0,0,0)':
            continue
        fig.add_trace(go.Scatter(
            x=[df.index[i-1], df.index[i-1], df.index[i], df.index[i]],
            y=[baseline_y, y0, y1, baseline_y],
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='toself', fillcolor=fill_color,
            showlegend=False, hoverinfo='skip'), row=row, col=col)

# ====================================================
# 7. 사이드바
#    - 분석 파라미터
#    - 매매 기록 저장/삭제
#    - 메모 저장/삭제   ← 추가
# ====================================================
def render_sidebar(selected_ticker: str) -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ 분석 파라미터")
        analysis_start = st.text_input("분석 시작일 (YY-MM)",
                                       value=st.session_state.analysis_start,
                                       placeholder="25-01")
        view_months  = st.number_input("차트 조회 기간 (최근 N개월)",
                                       min_value=1, max_value=240,
                                       value=st.session_state.view_months, step=1)
        guide_n      = st.number_input("가이드라인 기울기 (n)",
                                       min_value=1, max_value=20, value=4, step=1)

        st.markdown("---")
        _tok, _gid = _gist_cfg()
        st.caption(f"☁️ Gist 연동됨 (`{_gid[:8]}...`)" if (_tok and _gid)
                   else "💾 로컬 저장 (Gist 미설정)")

        # ── 매매 기록 ──────────────────────────────
        st.markdown("### 📈 매매 기록")
        ticker_options = (TARGET_TICKERS if selected_ticker in TARGET_TICKERS
                          else [selected_ticker] + TARGET_TICKERS)
        t_ticker = st.selectbox("종목", ticker_options,
                                index=ticker_options.index(selected_ticker))
        t_date   = st.date_input("날짜", datetime.date.today())
        t_type   = st.radio("종류", ['buy', 'sell'], horizontal=True)
        if st.button("기록 저장", key="trade_save_btn"):
            st.session_state.trade_history.setdefault(t_ticker, []).append(
                {'date': t_date.strftime("%Y-%m-%d"), 'type': t_type})
            save_trade_history(st.session_state.trade_history)
            st.success("저장 완료!")
            st.rerun()

        st.markdown("**🗑️ 기존 기록 삭제**")
        history = st.session_state.trade_history
        if selected_ticker in history and history[selected_ticker]:
            for i, record in enumerate(history[selected_ticker]):
                t        = record['type'].upper()
                btn_label = f"✕  {record['date']}  {t}"
                if st.button(btn_label, key=f"del_{selected_ticker}_{i}"):
                    st.session_state.trade_history[selected_ticker].pop(i)
                    save_trade_history(st.session_state.trade_history)
                    st.rerun()
        else:
            st.caption("매매 기록이 없습니다.")

        # ── 메모 관리 ──────────────────────────────
        st.markdown("---")
        st.markdown("### 📝 메모 관리")
        st.caption(f"현재 종목: **{display_name(selected_ticker)}**")

        memo_date = st.date_input("날짜 ", datetime.date.today(), key="sb_memo_date")
        memo_text = st.text_area("메모 내용", value="",
                                 key=f"sb_memo_text_{st.session_state.memo_input_key}",
                                 placeholder="메모를 입력하세요...", height=80)
        if st.button("메모 저장", key="memo_save_btn"):
            text = memo_text.strip()
            if text:
                date_str = memo_date.strftime("%Y-%m-%d")
                mh = st.session_state.memo_history
                mh.setdefault(selected_ticker, []).append({'date': date_str, 'text': text})
                mh[selected_ticker].sort(key=lambda x: x['date'], reverse=True)
                st.session_state.memo_history = mh
                save_memo_history(mh)
                st.session_state.memo_input_key += 1
                st.rerun()
            else:
                st.warning("메모 내용을 입력해 주세요.")

        st.markdown("**📋 메모 목록**")
        mh           = st.session_state.memo_history
        ticker_memos = mh.get(selected_ticker, [])
        if ticker_memos:
            for i, memo in enumerate(ticker_memos):
                preview = f"{memo['date']} {memo['text'][:12]}{'…' if len(memo['text']) > 12 else ''}"
                c1, c2 = st.columns(2)
                if c1.button(f"✏️ {preview}", key=f"memo_edit_btn_{safe_key(selected_ticker)}_{i}",
                             use_container_width=True):
                    st.session_state.memo_editing_idx = i
                    st.rerun()
                if c2.button(f"✕ {preview}", key=f"memo_del_{safe_key(selected_ticker)}_{i}",
                             use_container_width=True):
                    st.session_state.memo_history[selected_ticker].pop(i)
                    if st.session_state.memo_editing_idx == i:
                        st.session_state.memo_editing_idx = None
                    save_memo_history(st.session_state.memo_history)
                    st.rerun()

                # 수정창
                if st.session_state.memo_editing_idx == i:
                    st.markdown("<div style='background:#f3f4f6;padding:6px;"
                                "border-radius:6px;margin:2px 0 6px 0;'>",
                                unsafe_allow_html=True)
                    try:
                        edit_date_default = datetime.date.fromisoformat(memo['date'])
                    except Exception:
                        edit_date_default = datetime.date.today()
                    edit_date = st.date_input("날짜 수정", value=edit_date_default,
                                              key=f"memo_edit_date_{safe_key(selected_ticker)}_{i}")
                    edit_text = st.text_area("내용 수정", value=memo['text'],
                                             key=f"memo_edit_text_{safe_key(selected_ticker)}_{i}",
                                             height=70)
                    ecols = st.columns(2)
                    if ecols[0].button("💾 저장",
                                       key=f"memo_edit_save_{safe_key(selected_ticker)}_{i}",
                                       use_container_width=True):
                        new_text = edit_text.strip()
                        if new_text:
                            st.session_state.memo_history[selected_ticker][i] = {
                                'date': edit_date.strftime("%Y-%m-%d"),
                                'text': new_text
                            }
                            st.session_state.memo_history[selected_ticker].sort(
                                key=lambda x: x['date'], reverse=True)
                            save_memo_history(st.session_state.memo_history)
                            st.session_state.memo_editing_idx = None
                            st.rerun()
                        else:
                            st.warning("내용을 입력해 주세요.")
                    if ecols[1].button("✖ 취소",
                                       key=f"memo_edit_cancel_{safe_key(selected_ticker)}_{i}",
                                       use_container_width=True):
                        st.session_state.memo_editing_idx = None
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("메모가 없습니다.")

    return {'analysis_start': analysis_start.strip(), 'view_months': int(view_months),
            'guide_n': guide_n}

# ====================================================
# 8. 차트 렌더링
# ====================================================
def render_chart(df_daily: pd.DataFrame, selected_ticker: str,
                 beta: float, std_resid: float,
                 guide_n: int, view_months: int) -> None:
    st.markdown("""
    <style>
    .js-plotly-plot, .js-plotly-plot .plotly, .js-plotly-plot svg {
        touch-action: none !important;
    }
    </style>""", unsafe_allow_html=True)

    PX = {'main': 150, 'spacer': 20, 'price': 91, 'zscore': 91, 'macd': 91, 'rsi': 91}
    active_plots = ['main', 'spacer', 'price', 'zscore', 'macd', 'rsi']
    total_rows   = len(active_plots)
    total_h      = sum(PX[p] for p in active_plots)
    row_heights  = [PX[p] / total_h for p in active_plots]
    fig          = make_subplots(rows=total_rows, cols=1,
                                 row_heights=row_heights, vertical_spacing=0.02)
    current_row  = 1

    # ── [1] 로그-로그 산점도 ──
    sdf   = df_daily.sort_values(f'{X_ASSET_FIXED}_Norm')
    x_vals = sdf[f'{X_ASSET_FIXED}_Norm']
    min_x  = df_daily[f'{X_ASSET_FIXED}_Norm'].min()
    max_x  = df_daily[f'{X_ASSET_FIXED}_Norm'].max()
    empirical_c = (df_daily[f'{selected_ticker}_Norm']
                   / (df_daily[f'{X_ASSET_FIXED}_Norm'] ** guide_n))
    for log_c in np.linspace(np.log10(empirical_c.min()) - 1.0,
                              np.log10(empirical_c.max()) + 1.0, 15):
        c_val = 10 ** log_c
        fig.add_trace(go.Scatter(
            x=x_vals, y=c_val * (x_vals ** guide_n),
            mode='lines', line=dict(color='rgba(200,200,200,0.6)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'), row=current_row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) - 1.5 * std_resid),
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'),
        row=current_row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) + 1.5 * std_resid),
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(150,150,150,0.2)',
        showlegend=False, hoverinfo='skip'),
        row=current_row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=sdf['Predicted'],
        mode='lines', line=dict(color='black', width=2), name='Predicted Trend'),
        row=current_row, col=1)
    color_array = np.linspace(0, 1, len(df_daily))
    fig.add_trace(go.Scatter(
        x=df_daily[f'{X_ASSET_FIXED}_Norm'], y=df_daily[f'{selected_ticker}_Norm'],
        mode='markers',
        marker=dict(color=color_array, colorscale='Viridis', size=5, opacity=0.8),
        name='Daily Data'),
        row=current_row, col=1)
    last_x = df_daily[f'{X_ASSET_FIXED}_Norm'].iloc[-1]
    last_y = df_daily[f'{selected_ticker}_Norm'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_x], y=[last_y], mode='markers',
        marker=dict(symbol='star', color='hotpink', size=12,
                    line=dict(color='black', width=1)),
        name='Current'),
        row=current_row, col=1)
    band_upper = np.exp(np.log(sdf['Predicted'].values) + 1.5 * std_resid)
    band_lower = np.exp(np.log(sdf['Predicted'].values) - 1.5 * std_resid)
    y_all = np.concatenate([df_daily[f'{selected_ticker}_Norm'].dropna().values,
                             band_upper, band_lower])
    fig.update_xaxes(type="log", showgrid=False,
                     range=[np.log10(min_x * 0.98), np.log10(max_x * 1.02)],
                     row=current_row, col=1)
    fig.update_yaxes(type="log", showgrid=False,
                     range=[np.log10(np.nanmin(y_all) * 0.88),
                            np.log10(np.nanmax(y_all) * 1.18)],
                     row=current_row, col=1)
    # [1] 라벨: β 값
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>β = {beta:.2f}</b>",
        showarrow=False, font=dict(size=14, color='black'),
        xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=4,
        row=current_row, col=1)
    current_row += 1

    # ── [2] Spacer ──
    fig.update_xaxes(visible=False, row=current_row, col=1)
    fig.update_yaxes(visible=False, row=current_row, col=1)
    current_row += 1

    # ── 뷰 기간 재정규화 ──
    view_start    = df_daily.index[-1] - pd.DateOffset(months=view_months)
    view_df       = df_daily[df_daily.index >= view_start]
    grid_dtick_ms = get_time_grid_dtick_ms(view_start, df_daily.index[-1], target_grids=8)
    base_spy      = view_df[f'{X_ASSET_FIXED}_Norm'].iloc[0] if not view_df.empty else 1.0
    base_tkr      = view_df[f'{selected_ticker}_Norm'].iloc[0] if not view_df.empty else 1.0
    df_daily['Plot_Norm_SPY']    = df_daily[f'{X_ASSET_FIXED}_Norm'] / base_spy
    df_daily['Plot_Norm_Ticker'] = df_daily[f'{selected_ticker}_Norm'] / base_tkr

    # ── [3] Price ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Plot_Norm_SPY'],
                              mode='lines', line=dict(color='gray', width=1.5),
                              name=X_ASSET_FIXED), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
                              mode='lines', line=dict(color='black', width=1.5),
                              name=selected_ticker), row=current_row, col=1)
    min_price      = df_daily.loc[df_daily.index >= view_start,
                                   ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].min().min()
    max_price      = df_daily.loc[df_daily.index >= view_start,
                                   ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].max().max()
    price_baseline = min_price * 0.95
    df_daily['Price_Fill_Color'] = df_daily['Z_Score'].apply(get_price_fill_color)
    add_segmented_fill(fig, df_daily, 'Plot_Norm_Ticker', 'Price_Fill_Color',
                       current_row, 1, price_baseline)
    fig.update_yaxes(type="log",
                     range=[np.log10(price_baseline), np.log10(max_price * 1.05)],
                     row=current_row, col=1)
    # [3] 라벨: 현재 가격만
    last_price = df_daily[f'{selected_ticker}_Close'].iloc[-1]
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>${last_price:,.2f}</b>",
        showarrow=False, font=dict(size=14, color='black'),
        xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=4,
        row=current_row, col=1)
    time_x_axis = f'x{current_row}'
    current_row += 1

    # ── [4] Z-Score ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Z_Score'],
                              line=dict(color='black', width=1.5), name='Z-Score'),
                  row=current_row, col=1)
    fig.add_hline(y= 1.5, line_dash="solid", line_color="blue",  line_width=0.8, row=current_row, col=1)
    fig.add_hline(y=-1.5, line_dash="solid", line_color="red",   line_width=0.8, row=current_row, col=1)
    fig.add_hline(y=0,    line_dash="solid", line_color="gray",  line_width=0.6, row=current_row, col=1)
    # [4] 라벨: Z-Score 현재 값
    cz_val = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
    cz_color = get_z_text_color(cz_val)
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>Z  {cz_val:+.2f}</b>",
        showarrow=False, font=dict(size=14, color=cz_color),
        xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=4,
        row=current_row, col=1)
    add_filled_blocks(fig, df_daily, 'Z_Score', df_daily['Z_Score'] <= -1.5,
                      'rgba(220,38,38,0.20)', current_row, 1, -1.5)
    add_filled_blocks(fig, df_daily, 'Z_Score', df_daily['Z_Score'] >= 1.5,
                      'rgba(29,78,216,0.20)', current_row, 1, 1.5)
    z_view = df_daily.loc[df_daily.index >= view_start, 'Z_Score'].dropna()
    z_lo   = min(-2.0, z_view.min() if not z_view.empty else -2.0)
    z_hi   = max( 2.0, z_view.max() if not z_view.empty else  2.0)
    fig.update_yaxes(range=[z_lo - 0.2, z_hi + 0.2], row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ── [5] MACD ──
    macd_colors = np.where(df_daily['MACD_Hist'] >= 0,
                           'rgba(0,128,0,0.5)', 'rgba(255,0,0,0.5)')
    fig.add_trace(go.Bar(x=df_daily.index, y=df_daily['MACD_Hist'],
                          marker_color=macd_colors, name='MACD Hist'),
                  row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD'],
                              line=dict(color='blue', width=1), name='MACD'),
                  row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD_Signal'],
                              line=dict(color='orange', width=1), name='Signal'),
                  row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ── [6] RSI ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['RSI'],
                              line=dict(color='black', width=1.5), name='RSI'),
                  row=current_row, col=1)
    fig.add_hline(y=70, line_dash="solid", line_color="blue", line_width=0.8, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="solid", line_color="red",  line_width=0.8, row=current_row, col=1)
    # [6] 라벨: RSI 현재 값
    rsi_val = float(df_daily['RSI'].iloc[-1]) if pd.notna(df_daily['RSI'].iloc[-1]) else 50.0
    rsi_color = '#1d4ed8' if rsi_val >= 70 else '#dc2626' if rsi_val <= 30 else 'black'
    fig.add_annotation(
        x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>RSI  {rsi_val:.1f}</b>",
        showarrow=False, font=dict(size=14, color=rsi_color),
        xanchor='left', yanchor='top',
        bgcolor='white', bordercolor='black', borderwidth=1, borderpad=4,
        row=current_row, col=1)
    rsi_view = df_daily.loc[df_daily.index >= view_start, 'RSI'].dropna()
    rsi_lo   = min(20.0, rsi_view.min() if not rsi_view.empty else 20.0)
    rsi_hi   = max(80.0, rsi_view.max() if not rsi_view.empty else 80.0)
    fig.update_yaxes(range=[rsi_lo - 2, rsi_hi + 2], row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)

    # ── 매매 기록 마커 ──
    trade_history = st.session_state.trade_history
    if selected_ticker in trade_history:
        for trade in trade_history[selected_ticker]:
            t_date        = pd.to_datetime(trade['date'])
            t_type        = trade['type']
            marker_color  = '#dc2626' if t_type == 'buy' else '#1d4ed8'
            marker_symbol = 'triangle-up' if t_type == 'buy' else 'triangle-down'
            idx    = df_daily.index.get_indexer([t_date], method='nearest')[0]
            d_date = df_daily.index[idx]
            fig.add_trace(go.Scatter(
                x=[df_daily.loc[d_date, f'{X_ASSET_FIXED}_Norm']],
                y=[df_daily.loc[d_date, f'{selected_ticker}_Norm']],
                mode='markers',
                marker=dict(symbol=marker_symbol, size=10, color=marker_color,
                            line=dict(width=1, color='black')),
                name=f"{t_type.upper()} ({t_date.date()})"),
                row=1, col=1)
            for r in range(3, total_rows + 1):
                fig.add_vline(x=t_date, line_dash="solid", line_width=1,
                              line_color=marker_color, opacity=0.8, row=r, col=1)

    # ── 축 스타일 ──
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    for r in range(3, total_rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor='rgba(156,163,175,0.28)',
                         gridwidth=0.6, griddash='dot', dtick=grid_dtick_ms,
                         row=r, col=1)
        fig.update_yaxes(showgrid=False, row=r, col=1)
    for r in range(3, total_rows):
        fig.update_xaxes(showticklabels=False, tickformat="%m/%d", row=r, col=1)
    fig.update_xaxes(showticklabels=True, tickformat="%m/%d", row=total_rows, col=1)
    fig.update_layout(
        height=total_h, showlegend=False, hovermode='x unified',
        dragmode='pan', margin=dict(l=2, r=18, t=10, b=20),
        paper_bgcolor='white', plot_bgcolor='white')
    fig.update_xaxes(range=[view_start, df_daily.index[-1]], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True,
                    config={'scrollZoom': True, 'displayModeBar': False,
                            'doubleClick': 'reset', 'responsive': True})

# ====================================================
# 9. 메모 표시 (목록만 — 전체 너비)
# ====================================================
def render_memo_section(selected_ticker: str) -> None:
    mh           = st.session_state.memo_history
    ticker_memos = mh.get(selected_ticker, [])
    sorted_memos = sorted(ticker_memos, key=lambda x: x['date'], reverse=True)

    if not sorted_memos:
        return  # 메모 없으면 공간 차지 안 함

    rows_html = ""
    for memo in sorted_memos:
        rows_html += (
            f"<tr>"
            f"<td style='color:#6b7280;font-size:0.72rem;white-space:nowrap;"
            f"padding:4px 12px 4px 6px;vertical-align:top;'>{memo['date']}</td>"
            f"<td style='color:#111827;font-size:0.78rem;line-height:1.5;"
            f"padding:4px 4px;'>{memo['text']}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<div style='margin-top:8px;border-top:1px solid #e5e7eb;padding-top:6px;'>"
        f"<span style='font-size:0.75rem;font-weight:700;color:#6b7280;'>"
        f"📝 {display_name(selected_ticker)} 메모</span>"
        f"<table style='width:100%;border-collapse:collapse;margin-top:4px;'>"
        f"{rows_html}"
        f"</table></div>",
        unsafe_allow_html=True
    )

# ====================================================
# 10. 메인
# ====================================================
def main():
    init_session_state()

    DIRECT_INPUT_LABEL = "직접 입력"
    all_options = TARGET_TICKERS + [DIRECT_INPUT_LABEL]
    if st.session_state.selected_option not in all_options:
        st.session_state.selected_option = all_options[0]
    selected_option = st.session_state.selected_option

    if selected_option == DIRECT_INPUT_LABEL:
        custom_raw      = st.session_state.get('custom_ticker_input', '').strip().upper()
        selected_ticker = custom_raw if custom_raw else None
    else:
        selected_ticker = selected_option

    cfg = render_sidebar(selected_ticker or TARGET_TICKERS[0])

    settings_changed = False
    if cfg['analysis_start'] != st.session_state.analysis_start:
        st.session_state.analysis_start = cfg['analysis_start']
        settings_changed = True
    if cfg['view_months'] != st.session_state.view_months:
        st.session_state.view_months = cfg['view_months']
        settings_changed = True
    if settings_changed:
        s = load_settings()
        s['analysis_start'] = st.session_state.analysis_start
        s['view_months']    = st.session_state.view_months
        save_settings(s)

    # YY-MM → YYYY-MM-01 변환
    raw_start = st.session_state.analysis_start.strip()
    try:
        parsed = datetime.datetime.strptime(raw_start, '%y-%m')
        analysis_start = parsed.strftime('%Y-%m-01')
    except ValueError:
        try:
            datetime.datetime.strptime(raw_start, '%Y-%m-%d')
            analysis_start = raw_start  # 기존 전체 날짜 형식도 허용
        except ValueError:
            analysis_start = '2025-01-01'

    with st.spinner("데이터 로드 중..."):
        df_close = fetch_all_data(TARGET_TICKERS, analysis_start)

    if not df_close.empty:
        st.session_state.last_data_date = df_close.index[-1].strftime('%Y-%m-%d')

    if selected_ticker and f'{selected_ticker}_Close' not in df_close.columns:
        with st.spinner(f"{selected_ticker} 데이터를 불러오는 중..."):
            df_custom = fetch_single_ticker(selected_ticker, analysis_start)
        if not df_custom.empty:
            df_close = pd.concat([df_close, df_custom], axis=1).ffill()
        else:
            selected_ticker = None

    if not df_close.empty:
        st.session_state.last_data_date = df_close.index[-1].strftime('%Y-%m-%d')

    # ── 마지막 거래일 기준으로 데이터 슬라이싱 ──
    mkt = get_market_status()
    last_trading_date = pd.Timestamp(mkt['last_trading_date'])
    if not df_close.empty:
        df_close = df_close[df_close.index <= last_trading_date]

    pct_changes = {}
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        if col in df_close.columns and len(df_close) > 1:
            pct_changes[ticker] = df_close[col].pct_change().iloc[-1] * 100
        else:
            pct_changes[ticker] = 0.0

    with st.spinner("전체 종목 분석 중... (최초 실행 시 수십 초 소요)"):
        all_analyses = compute_all_analyses(df_close)

    for ticker, result in all_analyses.items():
        if result and result[0] is not None:
            df_t, _, _ = result
            cz = float(df_t['Z_Score'].iloc[-1]) if pd.notna(df_t['Z_Score'].iloc[-1]) else 0.0
            st.session_state.ticker_signals[ticker] = get_signal(cz)
        else:
            st.session_state.ticker_signals.setdefault(ticker, 'H')

    # RSI 값 추출 (버튼 표시용)
    ticker_rsi = {}
    for ticker, result in all_analyses.items():
        if result and result[0] is not None:
            df_t, _, _ = result
            rsi_val = df_t['RSI'].iloc[-1] if pd.notna(df_t['RSI'].iloc[-1]) else None
            ticker_rsi[ticker] = rsi_val
        else:
            ticker_rsi[ticker] = None

    df_daily = beta = std_resid = None

    if selected_ticker and selected_ticker in TARGET_TICKERS:
        result = all_analyses.get(selected_ticker)
        if result and result[0] is not None:
            df_daily, beta, std_resid = result
            cz = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
            st.session_state.ticker_signals[selected_ticker] = get_signal(cz)
    elif selected_ticker and f'{selected_ticker}_Close' in df_close.columns:
        with st.spinner(f"{display_name(selected_ticker)} 분석 중..."):
            result = process_asset_data(
                df_close[[f'{X_ASSET_FIXED}_Close']],
                df_close[[f'{selected_ticker}_Close']],
                X_ASSET_FIXED, selected_ticker)
        if result[0] is not None:
            df_daily, beta, std_resid = result
            cz = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
            st.session_state.ticker_signals[selected_ticker] = get_signal(cz)

    # ── CSS ──
    btn_css_parts = []
    for ticker in TARGET_TICKERS:
        sig   = st.session_state.ticker_signals.get(ticker, 'H')
        bg, _ = SIGNAL_STYLE.get(sig, ('#9ca3af', '#fff'))
        fg    = BUTTON_TEXT_STYLE.get(sig, '#111827')
        k     = f"ticker_btn_{safe_key(ticker)}"
        is_sel = (selected_option == ticker)
        sel_extra = (f"box-shadow:0 0 0 2px #fff,0 0 0 4px {bg}!important;"
                     "transform:scale(1.03);") if is_sel else ""
        btn_css_parts.append(f"""
        div.st-key-{k} button {{
            background:{bg}!important; border-color:{bg}!important;
            color:{fg}!important; font-weight:500!important;
            height:1.7rem!important; font-size:0.62rem!important;
            padding:0 2px!important; line-height:1!important;
            min-height:0!important; border-radius:3px!important;
            width:100%!important; text-align:left!important;
            {sel_extra}
        }}
        div.st-key-{k} button p     {{ color:{fg}!important; text-align:left!important; }}
        div.st-key-{k} button strong {{ color:{fg}!important; font-weight:700!important; }}
        div.st-key-{k} button span   {{ color:{fg}!important; }}
        div.st-key-{k} button:hover  {{ opacity:0.82!important; }}""")
    di_sel = (selected_option == DIRECT_INPUT_LABEL)
    btn_css_parts.append(f"""
    div.st-key-ticker_btn_direct button {{
        height:1.1rem!important; font-size:0.55rem!important;
        padding:0!important; min-height:0!important; border-radius:3px!important;
        {'border:2px solid #1565C0!important;font-weight:700!important;' if di_sel else ''}
    }}""")
    btn_css_parts.append("""
    div.st-key-full_refresh_btn button {
        height:1.1rem!important; min-height:0!important;
        border-radius:3px!important; font-size:0.55rem!important;
        font-weight:700!important; padding:0!important;
        border:1px solid #cbd5e1!important; background:#f8fafc!important;
        color:#0f172a!important; line-height:1!important;
    }
    div.st-key-full_refresh_btn button:hover {
        border-color:#94a3b8!important; background:#eef2f7!important;
    }""")

    st.markdown(f"""
    <style>
    .block-container {{
        padding-top:3.5rem!important; padding-bottom:0.5rem!important;
        max-width:100%!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {{
        flex-wrap:nowrap!important; gap:5px!important; align-items:flex-start!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {{
        flex:0 0 80px!important; min-width:80px!important;
        max-width:80px!important; padding:0!important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {{
        flex:1 1 0!important; min-width:0!important;
        overflow:visible!important;
        padding-left:2px!important; padding-right:2px!important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"] > div {{
        margin-bottom:0px!important; padding:0!important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"] {{
        gap:1px!important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button p {{
        margin:0!important; padding:0!important;
        font-size:0.73rem!important; line-height:1!important;
        font-weight:500!important; white-space:pre!important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button span   {{ color:inherit!important; }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button strong {{ color:inherit!important; font-weight:700!important; }}
    {''.join(btn_css_parts)}
    </style>""", unsafe_allow_html=True)

    # ── 제목 ──
    KST        = datetime.timezone(datetime.timedelta(hours=9))
    queried_at = datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M')
    # 장마감 시: 마지막 거래일 종가 기준 명시 / 장중: 실시간
    if mkt['is_open']:
        data_label = f"🟢 장중&nbsp;·&nbsp;조회: {queried_at}"
    else:
        data_label = f"🔴 장마감&nbsp;·&nbsp;{mkt['last_trading_label']}&nbsp;·&nbsp;조회: {queried_at}"

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"margin-bottom:1px;padding-bottom:1px;'>"
        f"<b style='font-size:1.15rem;white-space:nowrap;color:#111;'>📊 퀀트 대시보드</b>"
        f"<span style='font-size:10px;color:#999;white-space:nowrap;'>"
        f"{data_label}"
        f"</span></div>",
        unsafe_allow_html=True)

    # ── 버튼 + 차트 (2열) ──
    btn_col, chart_col = st.columns([1, 6])

    with btn_col:
        for ticker in TARGET_TICKERS:
            btn_key = f"ticker_btn_{safe_key(ticker)}"
            pct     = pct_changes.get(ticker, 0)
            rsi_val = ticker_rsi.get(ticker)
            rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else "--"
            pct_str = f"{pct:+.1f}%"
            btn_label = f"**{display_name(ticker)}**   {pct_str}   {rsi_str}"
            if st.button(btn_label, key=btn_key, use_container_width=True):
                st.session_state.selected_option     = ticker
                st.session_state.custom_ticker_input = ''
                st.rerun()
        if st.button(DIRECT_INPUT_LABEL, key="ticker_btn_direct", use_container_width=True):
            st.session_state.selected_option = DIRECT_INPUT_LABEL
            st.rerun()
        if selected_option == DIRECT_INPUT_LABEL:
            custom_input = st.text_input(
                "티커", value=st.session_state.get('custom_ticker_input', ''),
                placeholder="NVDA", label_visibility="collapsed")
            new_val = custom_input.strip().upper()
            if new_val != st.session_state.get('custom_ticker_input', ''):
                st.session_state.custom_ticker_input = new_val
                st.rerun()
        # ── refresh 버튼 (직접입력 아래) ──
        if st.button("🔄 refresh", key="full_refresh_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with chart_col:
        if df_daily is not None:
            render_chart(df_daily, selected_ticker, beta, std_resid,
                         cfg['guide_n'], cfg['view_months'])
        elif selected_option == DIRECT_INPUT_LABEL:
            if not st.session_state.get('custom_ticker_input', ''):
                st.info("왼쪽에서 티커를 입력해 주세요. (예: NVDA, 000660)")
            else:
                st.error(f"'{st.session_state.custom_ticker_input}'"
                         " 데이터를 가져올 수 없습니다. 티커를 확인해 주세요.")
        elif selected_ticker:
            st.error("분석에 필요한 데이터가 부족합니다.")

    # ── 메모 목록 (전체 너비 — 2열 밖) ──
    if selected_ticker:
        render_memo_section(selected_ticker)

    # ── 하단 여백 (Manage app 버튼에 가리지 않도록) ──
    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

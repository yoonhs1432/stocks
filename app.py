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

X_ASSET_FIXED  = 'SPY'
TARGET_TICKERS = [
    'SPYU', 'SOXL', 'TQQQ', 'FNGU', 'HIBL', 'TARK', 'QPUX', 'BNKU',
    'URTY', 'TECL', 'LABU', 'DFEN', 'EDC', 'INDL', 'EURL',
    'GDXU', 'KORU', '005930', 'BTC-USD', 'ETH-USD', 'AVAV',
]
TICKER_DISPLAY_NAMES = {'BTC-USD': 'BTC', 'ETH-USD': 'ETH', '005930': '삼전', '000660': '하닉'}

SIGNAL_STYLE = {
    'FB2': ('#7f1d1d', '#ffffff'), 'FB':  ('#dc2626', '#ffffff'), 'B':   ('#fca5a5', '#1a1a1a'),
    'H':   ('#9ca3af', '#ffffff'), 'S':   ('#93c5fd', '#1a1a1a'), 'FS':  ('#2563eb', '#ffffff'),
    'FS2': ('#1e3a8a', '#ffffff'),
}
BUTTON_TEXT_STYLE = {
    'FB2': '#f8fafc', 'FB': '#f8fafc', 'B': '#111827',
    'H': '#111827', 'S': '#111827', 'FS': '#f8fafc', 'FS2': '#f8fafc',
}
# 신호 이력 마커 스타일
SIG_MARKER = {
    'FB2': ('triangle-up',   '#7f1d1d', 10),
    'FB':  ('triangle-up',   '#dc2626',  8),
    'FS':  ('triangle-down', '#2563eb',  8),
    'FS2': ('triangle-down', '#1e3a8a', 10),
}

def display_name(ticker: str) -> str:
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)

def safe_key(ticker: str) -> str:
    return ticker.replace('-', '_').replace('.', '_').replace('/', '_')

# ====================================================
# 2. 영속화
# ====================================================
TRADE_FILE         = 'trade_history.json'
MEMO_FILE          = 'memo_history.json'
SETTINGS_FILE      = 'settings.json'
GIST_FILENAME      = 'quant_trade_history.json'
MEMO_GIST_FILENAME = 'quant_memo_history.json'

def _gist_cfg() -> tuple[str, str]:
    try:
        token   = st.secrets.get("GITHUB_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
        gist_id = st.secrets.get("GIST_ID", "")      or os.environ.get("GIST_ID", "")
        return str(token).strip(), str(gist_id).strip()
    except Exception:
        return "", ""

def _gist_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def _gist_read(gist_id: str, token: str, filename: str) -> dict | None:
    try:
        resp = requests.get(f"https://api.github.com/gists/{gist_id}",
                            headers=_gist_headers(token), timeout=6)
        if resp.ok:
            files = resp.json().get("files", {})
            if filename in files:
                return json.loads(files[filename]["content"])
    except Exception:
        pass
    return None

def _gist_write(gist_id: str, token: str, filename: str, data: dict) -> None:
    try:
        payload = {"files": {filename: {"content": json.dumps(data, indent=4, ensure_ascii=False)}}}
        requests.patch(f"https://api.github.com/gists/{gist_id}",
                       headers=_gist_headers(token), json=payload, timeout=6)
    except Exception:
        pass

def _load_json(local_file: str, gist_filename: str) -> dict:
    token, gist_id = _gist_cfg()
    if token and gist_id:
        data = _gist_read(gist_id, token, gist_filename)
        if data is not None:
            return data
    if os.path.exists(local_file):
        with open(local_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def _save_json(local_file: str, gist_filename: str, data: dict) -> None:
    with open(local_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    token, gist_id = _gist_cfg()
    if token and gist_id:
        _gist_write(gist_id, token, gist_filename, data)

def load_trade_history() -> dict:  return _load_json(TRADE_FILE, GIST_FILENAME)
def save_trade_history(h: dict):   _save_json(TRADE_FILE, GIST_FILENAME, h)
def load_memo_history()  -> dict:  return _load_json(MEMO_FILE, MEMO_GIST_FILENAME)
def save_memo_history(h: dict):    _save_json(MEMO_FILE, MEMO_GIST_FILENAME, h)

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(s: dict) -> None:
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def init_session_state() -> None:
    defaults = {
        'trade_history':       load_trade_history,
        'memo_history':        load_memo_history,
        'ticker_signals':      dict,
        'selected_option':     lambda: TARGET_TICKERS[0],
        'custom_ticker_input': str,
        'last_data_date':      str,
        'view_months':         lambda: load_settings().get('view_months', 12),
        'analysis_start':      lambda: load_settings().get('analysis_start', '25-01'),
        'memo_editing_idx':    lambda: None,
        'memo_input_key':      int,
        'candle_type':         lambda: '주봉',
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()

# ====================================================
# 3. 시장 상태
# ====================================================
def _us_holidays(year: int) -> set:
    from datetime import date
    def nth_weekday(y, m, wd, n):
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
    holidays = {date(year, 1, 1), date(year, 7, 4), date(year, 12, 25)}
    holidays.add(nth_weekday(year, 1, 0, 3))
    holidays.add(nth_weekday(year, 2, 0, 3))
    holidays.add(nth_weekday(year, 9, 0, 1))
    holidays.add(nth_weekday(year, 11, 3, 4))
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
    ET         = datetime.timezone(datetime.timedelta(hours=-4))
    now_et     = datetime.datetime.now(ET)
    today      = now_et.date()
    is_weekend = today.weekday() >= 5
    is_holiday = today in _us_holidays(today.year)
    mo         = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    mc         = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    in_hours   = mo <= now_et <= mc
    is_open    = not is_weekend and not is_holiday and in_hours
    last_day   = today
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
# 4. 신호 계산
# ====================================================
def compute_combined_score(cz: float, mhz: float, rsi: float) -> int:
    s = 0
    s += 2 if cz  <= -1.5 else 1 if cz  < 0  else -2 if cz  >= 1.5  else -1
    s += 2 if mhz <= -1.0 else 1 if mhz < 0  else -2 if mhz >= 1.0  else -1
    s += 2 if rsi <= 30   else 1 if rsi < 50  else -2 if rsi >= 70   else -1
    return s

def score_to_signal(score: int) -> str:
    if   score >= 5:  return 'FB2'
    elif score >= 3:  return 'FB'
    elif score >= 1:  return 'B'
    elif score <= -5: return 'FS2'
    elif score <= -3: return 'FS'
    elif score <= -1: return 'S'
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

def get_time_grid_dtick_ms(start: pd.Timestamp, end: pd.Timestamp, target_grids: int = 8) -> int:
    span_days   = max((end - start).days, 1)
    target_days = span_days / max(target_grids, 1)
    best_days   = min([3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180],
                      key=lambda d: abs(d - target_days))
    return int(best_days * 24 * 60 * 60 * 1000)

# ====================================================
# 5. 데이터 다운로드
# ====================================================
def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df_w     = df.resample('W-FRI').last().dropna(how='all')
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
    spy      = df[spy_col]
    traded   = (spy != spy.shift(1)) | (spy.index == spy.index[0])
    is_wkday = pd.Series(df.index.weekday < 5, index=df.index)
    return df[traded & is_wkday]

@st.cache_data(show_spinner=False)
def fetch_ohlc(ticker: str, start_date_str: str, candle_type: str = '일봉') -> pd.DataFrame:
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if data.empty:
            return pd.DataFrame()
        data = data[~data.index.duplicated(keep='last')].sort_index()
        cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in data.columns]
        if len(cols) < 4:
            return pd.DataFrame()
        df = data[cols][data.index.weekday < 5].copy()
        return _resample_weekly_ohlc(df) if candle_type == '주봉' else df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_all_data(tickers: list, start_date_str: str, candle_type: str = '일봉') -> pd.DataFrame:
    frames = []
    for ticker in [X_ASSET_FIXED] + list(tickers):
        try:
            data = fdr.DataReader(ticker, start_date_str)
            if not data.empty:
                data = data[~data.index.duplicated(keep='last')].sort_index()
                frames.append(data[['Close']].rename(columns={'Close': f'{ticker}_Close'}))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).ffill()
    df = _filter_trading_days(df)
    return _resample_weekly(df) if candle_type == '주봉' else df

@st.cache_data(show_spinner=False)
def fetch_single_ticker(ticker: str, start_date_str: str) -> pd.DataFrame:
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if not data.empty:
            data = data[~data.index.duplicated(keep='last')].sort_index()
            return data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
    except Exception:
        pass
    return pd.DataFrame()

# ====================================================
# 6. 데이터 처리
# ====================================================
def process_asset_data(df_x: pd.DataFrame, df_y: pd.DataFrame,
                       x_name: str, y_name: str) -> tuple:
    df = pd.merge(df_x, df_y, left_index=True, right_index=True).dropna().sort_index()
    if df.empty:
        return (None,) * 4

    base_x = df[f'{x_name}_Close'].iloc[0]
    base_y = df[f'{y_name}_Close'].iloc[0]
    df[f'{x_name}_Norm'] = df[f'{x_name}_Close'] / base_x
    df[f'{y_name}_Norm'] = df[f'{y_name}_Close'] / base_y
    log_x  = np.log(df[f'{x_name}_Norm'])
    log_y  = np.log(df[f'{y_name}_Norm'])
    model  = LinearRegression().fit(log_x.values.reshape(-1, 1), log_y.values)
    beta   = model.coef_[0]
    df['Predicted'] = np.exp(model.intercept_) * df[f'{x_name}_Norm'] ** beta

    close  = df[f'{y_name}_Close']
    delta  = close.diff()
    gain   = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss   = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI']         = 100 - (100 / (1 + gain / loss))
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    exp_std_macd      = df['MACD_Hist'].expanding(min_periods=30).std()
    exp_mean_macd     = df['MACD_Hist'].expanding(min_periods=30).mean()
    df['MACD_Hist_Z'] = (df['MACD_Hist'] - exp_mean_macd) / exp_std_macd.replace(0, np.nan)
    log_resid         = np.log(df[f'{y_name}_Norm']) - np.log(df['Predicted'])
    std_resid         = log_resid.std()
    df['Z_Score']     = log_resid / log_resid.expanding(min_periods=30).std().replace(0, np.nan)

    def _score_row(r):
        return compute_combined_score(
            float(r['Z_Score'])     if pd.notna(r['Z_Score'])     else 0.0,
            float(r['MACD_Hist_Z']) if pd.notna(r['MACD_Hist_Z']) else 0.0,
            float(r['RSI'])         if pd.notna(r['RSI'])         else 50.0,
        )
    df['Combined_Score']   = df.apply(_score_row, axis=1)
    df['Price_Fill_Color'] = df['Combined_Score'].apply(get_price_fill_color_combined)

    return df, beta, std_resid

@st.cache_data(show_spinner=False)
def compute_all_analyses(df_close: pd.DataFrame, _version: int = 7,
                         candle_type: str = '일봉') -> dict:
    df_x    = df_close[[f'{X_ASSET_FIXED}_Close']]
    results = {}
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        results[ticker] = (
            process_asset_data(df_x, df_close[[col]], X_ASSET_FIXED, ticker)
            if col in df_close.columns else None
        )
    return results

# ====================================================
# 7. 차트 헬퍼
# ====================================================
def _bar_colors(series: pd.Series,
                hi_thr: float, lo_thr: float,
                hi_c: str, lo_c: str, mid_hi_c: str, mid_lo_c: str) -> np.ndarray:
    return np.where(series >= hi_thr, hi_c,
           np.where(series >= 0,      mid_hi_c,
           np.where(series <= lo_thr, lo_c, mid_lo_c)))

def add_segmented_fill(fig, df, y_col, color_col, row, col, baseline_y):
    for i in range(1, len(df)):
        y0, y1 = df[y_col].iloc[i-1], df[y_col].iloc[i]
        fc     = df[color_col].iloc[i]
        if pd.isna(y0) or pd.isna(y1) or not fc or fc == 'rgba(0,0,0,0)':
            continue
        fig.add_trace(go.Scatter(
            x=[df.index[i-1], df.index[i-1], df.index[i], df.index[i]],
            y=[baseline_y, y0, y1, baseline_y],
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='toself', fillcolor=fc,
            showlegend=False, hoverinfo='skip'), row=row, col=col)

# ====================================================
# 8. 사이드바
# ====================================================
def render_sidebar(selected_ticker: str) -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ 분석 파라미터")
        candle_type    = st.radio("봉 기준", ['일봉', '주봉'], horizontal=True,
                                  index=1 if st.session_state.candle_type == '주봉' else 0)
        analysis_start = st.text_input("분석 시작일 (YY-MM)",
                                       value=st.session_state.analysis_start,
                                       placeholder="25-01")
        view_months    = st.number_input("차트 조회 기간 (최근 N개월)",
                                        min_value=1, max_value=240,
                                        value=st.session_state.view_months, step=1)
        guide_n        = st.number_input("가이드라인 기울기 (n)",
                                        min_value=1, max_value=20, value=4, step=1)

        st.markdown("---")
        _tok, _gid = _gist_cfg()
        st.caption(f"☁️ Gist 연동됨 (`{_gid[:8]}...`)" if (_tok and _gid)
                   else "💾 로컬 저장 (Gist 미설정)")

        # ── 매매 기록 ──
        st.markdown("### 📈 매매 기록")
        ticker_options = (TARGET_TICKERS if selected_ticker in TARGET_TICKERS
                          else [selected_ticker] + TARGET_TICKERS)
        t_ticker = st.selectbox("종목", ticker_options,
                                index=ticker_options.index(selected_ticker))
        t_date   = st.date_input("날짜", datetime.date.today())
        t_type   = st.radio("종류", ['buy', 'sell'], horizontal=True)
        t_col1, t_col2 = st.columns(2)
        t_qty   = t_col1.number_input("수량", min_value=0, value=0,
                                      step=1, format="%d")
        t_price = t_col2.number_input("단가($)", min_value=0.0, value=0.0,
                                      step=0.01, format="%.4f")
        if st.button("기록 저장", key="trade_save_btn"):
            record = {'date': t_date.strftime("%Y-%m-%d"), 'type': t_type}
            if t_qty > 0:   record['qty']   = int(t_qty)
            if t_price > 0: record['price'] = t_price
            st.session_state.trade_history.setdefault(t_ticker, []).append(record)
            save_trade_history(st.session_state.trade_history)
            st.success("저장 완료!")
            st.rerun()

        st.markdown("**🗑️ 기존 기록 삭제**")
        history = st.session_state.trade_history
        if selected_ticker in history and history[selected_ticker]:
            for i, record in enumerate(history[selected_ticker]):
                qty_str = f" {record['qty']}주" if record.get('qty') else ""
                prc_str = f" @${record['price']:.2f}" if record.get('price') else ""
                label   = f"✕  {record['date']}  {record['type'].upper()}{qty_str}{prc_str}"
                if st.button(label, key=f"del_{selected_ticker}_{i}"):
                    st.session_state.trade_history[selected_ticker].pop(i)
                    save_trade_history(st.session_state.trade_history)
                    st.rerun()
        else:
            st.caption("매매 기록이 없습니다.")

        # ── 메모 관리 ──
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
                mh = st.session_state.memo_history
                mh.setdefault(selected_ticker, []).append(
                    {'date': memo_date.strftime("%Y-%m-%d"), 'text': text})
                mh[selected_ticker].sort(key=lambda x: x['date'], reverse=True)
                save_memo_history(mh)
                st.session_state.memo_input_key += 1
                st.rerun()
            else:
                st.warning("메모 내용을 입력해 주세요.")

        st.markdown("**📋 메모 목록**")
        mh           = st.session_state.memo_history
        ticker_memos = mh.get(selected_ticker, [])
        for i, memo in enumerate(ticker_memos):
            preview = f"{memo['date']} {memo['text'][:12]}{'…' if len(memo['text']) > 12 else ''}"
            c1, c2  = st.columns(2)
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
                            'date': edit_date.strftime("%Y-%m-%d"), 'text': new_text}
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
        if not ticker_memos:
            st.caption("메모가 없습니다.")

    return {'analysis_start': analysis_start.strip(), 'view_months': int(view_months),
            'guide_n': guide_n, 'candle_type': candle_type}

# ====================================================
# 9. 차트 렌더링
# ====================================================
def render_chart(df_daily: pd.DataFrame, selected_ticker: str,
                 beta: float, std_resid: float,
                 guide_n: int, view_months: int,
                 df_ohlc: pd.DataFrame = None,
                 df_daily_raw: pd.DataFrame = None) -> None:
    st.markdown("""<style>
    .js-plotly-plot, .js-plotly-plot .plotly, .js-plotly-plot svg {
        touch-action: none !important; }
    </style>""", unsafe_allow_html=True)

    PX         = {'main': 150, 'spacer': 20, 'price': 100, 'zscore': 100, 'macd': 100, 'rsi': 100}
    plot_order = ['main', 'spacer', 'price', 'zscore', 'macd', 'rsi']
    total_rows = len(plot_order)
    total_h    = sum(PX[p] for p in plot_order)
    fig        = make_subplots(rows=total_rows, cols=1,
                               row_heights=[PX[p]/total_h for p in plot_order],
                               vertical_spacing=0.02)
    row = 1

    # ── [1] 로그-로그 산점도 ──
    sc_df  = (df_daily_raw if (df_daily_raw is not None and not df_daily_raw.empty) else df_daily)
    sdf    = sc_df.sort_values(f'{X_ASSET_FIXED}_Norm')
    x_vals = sdf[f'{X_ASSET_FIXED}_Norm']
    min_x, max_x = sc_df[f'{X_ASSET_FIXED}_Norm'].min(), sc_df[f'{X_ASSET_FIXED}_Norm'].max()

    emp_c = sc_df[f'{selected_ticker}_Norm'] / (sc_df[f'{X_ASSET_FIXED}_Norm'] ** guide_n)
    for log_c in np.linspace(np.log10(emp_c.min()) - 1.0, np.log10(emp_c.max()) + 1.0, 15):
        fig.add_trace(go.Scatter(
            x=x_vals, y=(10**log_c) * (x_vals**guide_n),
            mode='lines', line=dict(color='rgba(200,200,200,0.6)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) - 1.5 * std_resid),
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) + 1.5 * std_resid),
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(150,150,150,0.2)', showlegend=False, hoverinfo='skip'), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=sdf['Predicted'],
        mode='lines', line=dict(color='black', width=2), name='Predicted Trend'), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=sc_df[f'{X_ASSET_FIXED}_Norm'], y=sc_df[f'{selected_ticker}_Norm'],
        mode='markers',
        marker=dict(color=np.linspace(0, 1, len(sc_df)), colorscale='Viridis', size=5, opacity=0.8),
        name='Daily Data'), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=[sc_df[f'{X_ASSET_FIXED}_Norm'].iloc[-1]],
        y=[sc_df[f'{selected_ticker}_Norm'].iloc[-1]],
        mode='markers',
        marker=dict(symbol='star', color='hotpink', size=12, line=dict(color='black', width=1)),
        name='Current'), row=row, col=1)

    band_upper = np.exp(np.log(sdf['Predicted'].values) + 1.5 * std_resid)
    band_lower = np.exp(np.log(sdf['Predicted'].values) - 1.5 * std_resid)
    y_all = np.concatenate([sc_df[f'{selected_ticker}_Norm'].dropna().values, band_upper, band_lower])
    fig.update_xaxes(type="log", showgrid=False,
                     range=[np.log10(min_x*0.98), np.log10(max_x*1.02)], row=row, col=1)
    fig.update_yaxes(type="log", showgrid=False,
                     range=[np.log10(np.nanmin(y_all)*0.88), np.log10(np.nanmax(y_all)*1.18)],
                     row=row, col=1)
    fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
                       text=f"<b>β = {beta:.2f}</b>", showarrow=False,
                       font=dict(size=11, color='black'), xanchor='left', yanchor='top',
                       bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
                       row=row, col=1)
    row += 1

    # ── [2] Spacer ──
    fig.update_xaxes(visible=False, row=row, col=1)
    fig.update_yaxes(visible=False, row=row, col=1)
    row += 1

    # ── 뷰 기간 설정 ──
    last_date  = df_daily.index[-1]
    first_date = df_daily.index[0]
    view_start = max(last_date - pd.DateOffset(months=view_months), first_date)
    snap_idx   = min(df_daily.index.searchsorted(view_start), len(df_daily) - 1)
    view_start = df_daily.index[snap_idx]

    grid_dtick_ms = get_time_grid_dtick_ms(view_start, last_date)
    base_spy  = df_daily.loc[df_daily.index >= view_start, f'{X_ASSET_FIXED}_Norm'].iloc[0]
    base_tkr  = df_daily.loc[df_daily.index >= view_start, f'{selected_ticker}_Norm'].iloc[0]
    df_daily['Plot_Norm_SPY']    = df_daily[f'{X_ASSET_FIXED}_Norm'] / base_spy
    df_daily['Plot_Norm_Ticker'] = df_daily[f'{selected_ticker}_Norm'] / base_tkr

    # ── [3] Price ──
    price_row = row
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['Plot_Norm_SPY'],
        mode='lines', line=dict(color='gray', width=1.5), name=X_ASSET_FIXED), row=row, col=1)

    ohlc_norm = pd.DataFrame()
    if df_ohlc is not None and not df_ohlc.empty:
        base_close = df_daily[f'{selected_ticker}_Close'].iloc[0]
        base_n     = df_daily[f'{selected_ticker}_Norm'].iloc[0]
        base_vn    = df_daily.loc[df_daily.index >= view_start, f'{selected_ticker}_Norm'].iloc[0]
        scale      = base_n / base_vn / base_close if base_close != 0 else 1.0
        ohlc_norm  = df_ohlc * scale
        fig.add_trace(go.Candlestick(
            x=ohlc_norm.index,
            open=ohlc_norm['Open'], high=ohlc_norm['High'],
            low=ohlc_norm['Low'],   close=ohlc_norm['Close'],
            increasing=dict(line=dict(color='#dc2626', width=1), fillcolor='#dc2626'),
            decreasing=dict(line=dict(color='#1d4ed8', width=1), fillcolor='#1d4ed8'),
            showlegend=False, hoverinfo='skip'), row=row, col=1)
        fig.update_layout(xaxis3_rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(
            x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
            mode='lines', line=dict(color='black', width=1.5), name=selected_ticker), row=row, col=1)

    if not ohlc_norm.empty:
        vc   = ohlc_norm[ohlc_norm.index >= view_start]
        p_lo = vc['Low'].min()  if not vc.empty else df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].min()
        p_hi = vc['High'].max() if not vc.empty else df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].max()
    else:
        p_lo = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].min()
        p_hi = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_Ticker'].max()
    spy_lo = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_SPY'].min()
    spy_hi = df_daily.loc[df_daily.index >= view_start, 'Plot_Norm_SPY'].max()
    p_lo, p_hi = min(p_lo, spy_lo) * 0.97, max(p_hi, spy_hi) * 1.03

    add_segmented_fill(fig, df_daily, 'Plot_Norm_Ticker', 'Price_Fill_Color', row, 1, p_lo)

    # ── ★ [6] 신호 이력 타임라인 마커 ──
    if 'Combined_Score' in df_daily.columns:
        sig_series = df_daily['Combined_Score'].apply(score_to_signal)
        sig_view   = sig_series[df_daily.index >= view_start]
        prev_sig   = None
        for dt, sig in sig_view.items():
            if sig not in SIG_MARKER or sig == prev_sig:
                prev_sig = sig
                continue
            sym, color, sz = SIG_MARKER[sig]
            y_pos = df_daily.loc[dt, 'Plot_Norm_Ticker'] if dt in df_daily.index else None
            if y_pos is None or pd.isna(y_pos):
                prev_sig = sig
                continue
            offset = 1.018 if 'up' in sym else 0.982
            fig.add_trace(go.Scatter(
                x=[dt], y=[y_pos * offset],
                mode='markers',
                marker=dict(symbol=sym, size=sz, color=color, line=dict(width=0)),
                showlegend=False, hoverinfo='skip'),
                row=price_row, col=1)
            prev_sig = sig

    fig.update_yaxes(type="log",
                     range=[np.log10(max(p_lo, 1e-6)), np.log10(max(p_hi, 1e-6))],
                     autorange=False, fixedrange=True, row=row, col=1)
    fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
                       text=f"<b>${df_daily[f'{selected_ticker}_Close'].iloc[-1]:,.2f}</b>",
                       showarrow=False, font=dict(size=11, color='black'),
                       xanchor='left', yanchor='top',
                       bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
                       row=row, col=1)
    time_x_axis = f'x{row}'
    row += 1

    # ── [4~5] Z / MACD / RSI ──
    C_HI = 'rgba(29,78,216,0.85)'
    C_LO = 'rgba(185,28,28,0.85)'
    C_MH = 'rgba(147,197,253,0.6)'
    C_ML = 'rgba(252,165,165,0.6)'

    for col_name, hi, lo, label, color_fn in [
        ('Z_Score',     1.5, -1.5, 'Z',    lambda v: 'black'),
        ('MACD_Hist_Z', 1.0, -1.0, 'MACD', lambda v: '#dc2626' if v <= -1.0 else '#1d4ed8' if v >= 1.0 else 'black'),
    ]:
        colors = _bar_colors(df_daily[col_name], hi, lo, C_HI, C_LO, C_MH, C_ML)
        fig.add_trace(go.Bar(x=df_daily.index, y=df_daily[col_name],
                              marker_color=colors, name=col_name, hoverinfo='skip'), row=row, col=1)
        for y_val, lc in [(hi, 'blue'), (-hi, 'red'), (0, 'gray')]:
            fig.add_hline(y=y_val, line_dash="solid", line_color=lc,
                          line_width=0.8 if y_val != 0 else 0.6, row=row, col=1)
        val = float(df_daily[col_name].iloc[-1]) if pd.notna(df_daily[col_name].iloc[-1]) else 0.0
        fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
                           text=f"<b>{label}  {val:+.2f}</b>", showarrow=False,
                           font=dict(size=11, color=color_fn(val)),
                           xanchor='left', yanchor='top',
                           bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
                           row=row, col=1)
        view_abs = abs(df_daily.loc[df_daily.index >= view_start, col_name].dropna())
        rng      = max(hi, view_abs.max() if not view_abs.empty else hi)
        fig.update_yaxes(range=[-(rng+0.3), rng+0.3], autorange=False, fixedrange=True, row=row, col=1)
        row += 1

    # RSI
    rsi_c      = df_daily['RSI'] - 50
    rsi_colors = _bar_colors(df_daily['RSI'], 70, 30, C_HI, C_LO, C_MH, C_ML)
    fig.add_trace(go.Bar(x=df_daily.index, y=rsi_c,
                          marker_color=rsi_colors, name='RSI', hoverinfo='skip'), row=row, col=1)
    for y_val, lc in [(20, 'blue'), (-20, 'red'), (0, 'gray')]:
        fig.add_hline(y=y_val, line_dash="solid", line_color=lc,
                      line_width=0.8 if y_val != 0 else 0.6, row=row, col=1)
    rsi_val   = float(df_daily['RSI'].iloc[-1]) if pd.notna(df_daily['RSI'].iloc[-1]) else 50.0
    rsi_color = '#1d4ed8' if rsi_val >= 70 else '#dc2626' if rsi_val <= 30 else 'black'
    fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
                       text=f"<b>RSI  {rsi_val:.1f}</b>", showarrow=False,
                       font=dict(size=11, color=rsi_color), xanchor='left', yanchor='top',
                       bgcolor='white', bordercolor='black', borderwidth=1, borderpad=2,
                       row=row, col=1)
    rsi_abs = max(20.0, abs(df_daily.loc[df_daily.index >= view_start, 'RSI'].dropna() - 50).max()
                  if not df_daily.loc[df_daily.index >= view_start, 'RSI'].dropna().empty else 20.0)
    fig.update_yaxes(range=[-(rsi_abs+2), rsi_abs+2], autorange=False, fixedrange=True,
                     row=row, col=1)

    # ── 매매 기록 마커 ──
    for trade in st.session_state.trade_history.get(selected_ticker, []):
        t_date  = pd.to_datetime(trade['date'])
        is_buy  = trade['type'] == 'buy'
        m_color = '#dc2626' if is_buy else '#1d4ed8'
        idx_sc  = sc_df.index.get_indexer([t_date], method='nearest')[0]
        d_sc    = sc_df.index[idx_sc]
        fig.add_trace(go.Scatter(
            x=[sc_df.loc[d_sc, f'{X_ASSET_FIXED}_Norm']],
            y=[sc_df.loc[d_sc, f'{selected_ticker}_Norm']],
            mode='markers',
            marker=dict(symbol='triangle-up' if is_buy else 'triangle-down',
                        size=10, color=m_color, line=dict(width=1, color='black')),
            name=f"{trade['type'].upper()} ({t_date.date()})", hoverinfo='skip'),
            row=1, col=1)
        for r in range(3, total_rows + 1):
            fig.add_vline(x=t_date, line_dash="solid", line_width=1,
                          line_color=m_color, opacity=0.8, row=r, col=1)

    # ── 축 공통 스타일 ──
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    for r in range(3, total_rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor='rgba(156,163,175,0.28)',
                         gridwidth=0.6, griddash='dot', dtick=grid_dtick_ms,
                         matches=time_x_axis, rangebreaks=[dict(bounds=['sat', 'mon'])],
                         showticklabels=(r == total_rows), tickformat="%m/%d",
                         range=[view_start, last_date], row=r, col=1)
        fig.update_yaxes(showgrid=False, autorange=False, fixedrange=True, row=r, col=1)

    fig.update_traces(hoverinfo='skip')
    fig.update_layout(
        height=total_h, showlegend=False, hovermode=False,
        dragmode='pan', margin=dict(l=2, r=18, t=10, b=20),
        paper_bgcolor='white', plot_bgcolor='white', uirevision='constant')

    st.plotly_chart(fig, use_container_width=True,
                    config={'scrollZoom': True, 'displayModeBar': False,
                            'doubleClick': 'reset', 'responsive': True, 'showTips': False})

# ====================================================
# 10. 포지션 트래커
# ====================================================
def _resolve_all_cycles(valid: list) -> tuple[dict, float]:
    """
    거래 기록 전체를 순회해 현재 사이클 정보와 누적 실현손익을 반환한다.

    사이클 규칙:
      - hold_qty == 0 상태에서 매수 → 새 사이클 시작, 카운터 리셋
      - 매도 후 hold_qty == 0 → 사이클 종료, 실현손익 확정
      - 마지막 사이클이 현재 사이클

    반환값:
      cyc  (dict)
        cycle_start  : 현재 사이클 첫 매수일 (datetime.date)
        cycle_end    : 전량 매도 완료일 (datetime.date) or None
        hold_qty     : 현재 보유 수량 (int)
        buy_qty      : 현재 사이클 누적 매수 수량
        buy_cost     : 현재 사이클 누적 매수 금액
        current_pnl  : 현재 사이클 실현손익 (청산 시) or None (보유 중)
      cumulative_pnl (float) : 과거 사이클 실현손익 누적 합 (현재 사이클 제외)
    """
    sorted_records = sorted(valid, key=lambda r: r['date'])

    cycle_start    = None
    cycle_end      = None
    hold_qty       = 0
    buy_qty        = 0
    buy_cost       = 0.0
    sell_proceeds  = 0.0   # 현재 사이클 매도 금액 누적
    cumulative_pnl = 0.0   # 과거 사이클 실현손익 누적

    for r in sorted_records:
        date = datetime.date.fromisoformat(r['date'])
        qty  = int(r['qty'])

        if r['type'] == 'buy':
            if hold_qty == 0:
                # 이전 사이클이 있었다면 누적에 편입
                if cycle_start is not None and cycle_end is not None:
                    cumulative_pnl += sell_proceeds - buy_cost
                # 새 사이클 시작
                cycle_start   = date
                cycle_end     = None
                buy_qty       = 0
                buy_cost      = 0.0
                sell_proceeds = 0.0
            hold_qty += qty
            buy_qty  += qty
            buy_cost += qty * r['price']

        elif r['type'] == 'sell' and hold_qty > 0:
            sell_proceeds += qty * r['price']
            hold_qty      -= qty
            hold_qty       = max(hold_qty, 0)
            if hold_qty == 0:
                cycle_end = date

    # 현재 사이클 실현손익 (청산된 경우에만)
    current_pnl = (sell_proceeds - buy_cost) if cycle_end else None

    cyc = {
        'cycle_start': cycle_start,
        'cycle_end':   cycle_end,
        'hold_qty':    hold_qty,
        'buy_qty':     buy_qty,
        'buy_cost':    buy_cost,
        'current_pnl': current_pnl,
    }
    return cyc, cumulative_pnl


def _calc_portfolio_total_pnl(df_daily: pd.DataFrame) -> float:
    """
    전 종목의 (누적실현손익 + 현재 평가손익) 합계를 반환한다.
      - 누적실현손익 : 모든 청산 사이클의 실현손익 합 (종목별)
      - 현재 평가손익 : 보유 중인 종목만 (현재가 - 평균단가) × 보유수량
      - 청산 완료 종목의 현재 사이클 실현손익도 포함
    """
    total = 0.0
    for ticker in TARGET_TICKERS:
        records = st.session_state.trade_history.get(ticker, [])
        valid   = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]
        if not valid:
            continue
        cyc, cum_pnl = _resolve_all_cycles(valid)
        if cyc['buy_qty'] == 0:
            continue

        # 누적실현손익 (과거 사이클 + 현재 사이클 청산분)
        realized = cum_pnl + (cyc['current_pnl'] if cyc['current_pnl'] is not None else 0.0)

        # 현재 평가손익 (보유 중인 경우만)
        unrealized = 0.0
        if cyc['hold_qty'] > 0:
            col = f'{ticker}_Close'
            if col in df_daily.columns:
                current_price = float(df_daily[col].iloc[-1])
                avg_price     = cyc['buy_cost'] / cyc['buy_qty']
                unrealized    = (current_price - avg_price) * cyc['hold_qty']

        total += realized + unrealized
    return total


def render_position_tracker(selected_ticker: str, df_daily: pd.DataFrame) -> None:
    records = st.session_state.trade_history.get(selected_ticker, [])
    valid   = [r for r in records if r.get('qty', 0) > 0 and r.get('price', 0) > 0]

    # ── 전 종목 누적 손익 (항상 계산) ──
    portfolio_pnl = _calc_portfolio_total_pnl(df_daily)

    # ── 현재가 (항상 필요) ──
    col_close     = f'{selected_ticker}_Close'
    current_price = float(df_daily[col_close].iloc[-1]) if col_close in df_daily.columns else None

    # ── 공통 포맷 헬퍼 ──
    def _fmt_pnl(val: float, pct: float | None = None) -> str:
        sign    = '+' if val >= 0 else ''
        color   = '#b91c1c' if val >= 0 else '#1d4ed8'
        pct_str = f"&nbsp;({sign}{pct:.2f}%)" if pct is not None else ''
        return f"<span style='font-weight:700;color:{color};'>{sign}${val:,.2f}{pct_str}</span>"

    def _dash_cell(label: str) -> str:
        return (f"<div><div style='color:#6b7280;font-size:0.68rem;'>{label}</div>"
                f"<div style='font-weight:700;color:#9ca3af;'>-</div></div>")

    port_sign  = '+' if portfolio_pnl >= 0 else ''
    port_color = '#b91c1c' if portfolio_pnl >= 0 else '#1d4ed8'
    port_html  = (
        f"<div><div style='color:#6b7280;font-size:0.68rem;'>전종목 누적손익</div>"
        f"<div style='font-weight:700;color:{port_color};'>{port_sign}${portfolio_pnl:,.2f}</div></div>"
    )

    # ── 매매 기록 없는 경우: 빈 기본 화면 ──
    if not valid:
        price_html = (
            f"<div><div style='color:#6b7280;font-size:0.68rem;'>현재가</div>"
            f"<div style='font-weight:700;'>${current_price:,.2f}</div></div>"
            if current_price else _dash_cell("현재가")
        )
        st.markdown(f"""
        <div style='display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 8px 0;
                    padding:8px 12px;background:#f8fafc;
                    border:1px solid #e2e8f0;border-radius:8px;font-size:0.78rem;'>
          {price_html}
          {_dash_cell("평균단가")}
          {_dash_cell("보유수량")}
          {_dash_cell("보유기간")}
          {_dash_cell("평가손익")}
          {_dash_cell("누적실현손익")}
          {port_html}
        </div>""", unsafe_allow_html=True)
        return

    cyc, cumulative_pnl = _resolve_all_cycles(valid)
    if cyc['cycle_start'] is None or cyc['buy_qty'] == 0:
        return

    hold_qty  = cyc['hold_qty']
    avg_price = cyc['buy_cost'] / cyc['buy_qty']

    # ── 보유기간 ──
    end_date  = cyc['cycle_end'] if cyc['cycle_end'] else datetime.date.today()
    hold_days = (end_date - cyc['cycle_start']).days

    # ── 보유수량 ──
    qty_display = f"{hold_qty:,}주" if hold_qty > 0 else "-"

    # ── 현재 사이클 손익 ──
    is_closed = cyc['cycle_end'] is not None
    if is_closed:
        pnl_dollar = cyc['current_pnl']
        pnl_pct    = pnl_dollar / cyc['buy_cost'] * 100 if cyc['buy_cost'] else 0.0
        pnl_label  = "실현손익"
    else:
        pnl_dollar = (current_price - avg_price) * hold_qty
        pnl_pct    = (current_price - avg_price) / avg_price * 100 if avg_price else 0.0
        pnl_label  = "평가손익"

    # ── 누적실현손익 ──
    total_realized     = cumulative_pnl + (cyc['current_pnl'] if is_closed else 0.0)
    has_cumulative     = (cumulative_pnl != 0.0) or is_closed
    cumulative_html    = (
        f"<div><div style='color:#6b7280;font-size:0.68rem;'>누적실현손익</div>"
        f"<div>{_fmt_pnl(total_realized)}</div></div>"
        if has_cumulative else _dash_cell("누적실현손익")
    )

    bg_color = '#f0fdf4' if is_closed else '#f8fafc'
    border_c = '#86efac' if is_closed else '#e2e8f0'

    st.markdown(f"""
    <div style='display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 8px 0;
                padding:8px 12px;background:{bg_color};
                border:1px solid {border_c};border-radius:8px;font-size:0.78rem;'>
      <div><div style='color:#6b7280;font-size:0.68rem;'>현재가</div>
           <div style='font-weight:700;'>${current_price:,.2f}</div></div>
      <div><div style='color:#6b7280;font-size:0.68rem;'>평균단가</div>
           <div style='font-weight:700;'>${avg_price:,.2f}</div></div>
      <div><div style='color:#6b7280;font-size:0.68rem;'>보유수량</div>
           <div style='font-weight:700;'>{qty_display}</div></div>
      <div><div style='color:#6b7280;font-size:0.68rem;'>보유기간</div>
           <div style='font-weight:700;'>{hold_days}일</div></div>
      <div><div style='color:#6b7280;font-size:0.68rem;'>{pnl_label}</div>
           <div>{_fmt_pnl(pnl_dollar, pnl_pct)}</div></div>
      {cumulative_html}
      {port_html}
    </div>""", unsafe_allow_html=True)

# ====================================================
# 11. 메모 목록
# ====================================================
def render_memo_section(selected_ticker: str) -> None:
    memos = sorted(st.session_state.memo_history.get(selected_ticker, []),
                   key=lambda x: x['date'], reverse=True)
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
        unsafe_allow_html=True)

# ====================================================
# 12. CSS
# ====================================================
def build_css(selected_option: str) -> str:
    btn_parts = []
    for ticker in TARGET_TICKERS:
        sig      = st.session_state.ticker_signals.get(ticker, 'H')
        bg, _    = SIGNAL_STYLE.get(sig, ('#9ca3af', '#fff'))
        fg       = BUTTON_TEXT_STYLE.get(sig, '#111827')
        k        = f"ticker_btn_{safe_key(ticker)}"
        sel_extra = (f"box-shadow:0 0 0 2px #fff,0 0 0 4px {bg}!important;"
                     "transform:scale(1.03);") if selected_option == ticker else ""
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

    di_border = "border:2px solid #1565C0!important;font-weight:700!important;" \
                if selected_option == "직접 입력" else ""
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
        flex:0 0 80px!important; min-width:80px!important;
        max-width:80px!important; padding:0!important;
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
    {''.join(btn_parts)}
    </style>"""

# ====================================================
# 13. 메인
# ====================================================
def main():
    init_session_state()

    DIRECT_INPUT_LABEL = "직접 입력"
    all_options        = TARGET_TICKERS + [DIRECT_INPUT_LABEL]
    if st.session_state.selected_option not in all_options:
        st.session_state.selected_option = all_options[0]
    selected_option = st.session_state.selected_option

    selected_ticker = (
        st.session_state.get('custom_ticker_input', '').strip().upper() or None
        if selected_option == DIRECT_INPUT_LABEL else selected_option
    )

    cfg = render_sidebar(selected_ticker or TARGET_TICKERS[0])

    if (st.session_state.analysis_start != cfg['analysis_start'] or
            st.session_state.view_months != cfg['view_months']):
        st.session_state.analysis_start = cfg['analysis_start']
        st.session_state.view_months    = cfg['view_months']
        s = load_settings()
        s.update({'analysis_start': cfg['analysis_start'], 'view_months': cfg['view_months']})
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
            selected_ticker = None

    mkt = get_market_status()
    last_trading_date = pd.Timestamp(mkt['last_trading_date'])
    if not df_close.empty:
        st.session_state.last_data_date = df_close.index[-1].strftime('%Y-%m-%d')
        if candle_type == '일봉':
            df_close = df_close[df_close.index <= last_trading_date]

    with st.spinner("전체 종목 분석 중..."):
        all_analyses = compute_all_analyses(df_close, _version=7, candle_type=candle_type)

    pct_changes = {}
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        pct_changes[ticker] = (df_close[col].pct_change().iloc[-1] * 100
                               if col in df_close.columns and len(df_close) > 1 else 0.0)
        result = all_analyses.get(ticker)
        if result and result[0] is not None:
            df_t, _, _ = result
            cz  = float(df_t['Z_Score'].iloc[-1])     if pd.notna(df_t['Z_Score'].iloc[-1])     else 0.0
            mhz = float(df_t['MACD_Hist_Z'].iloc[-1]) if pd.notna(df_t['MACD_Hist_Z'].iloc[-1]) else 0.0
            rsi = float(df_t['RSI'].iloc[-1])          if pd.notna(df_t['RSI'].iloc[-1])          else 50.0
            st.session_state.ticker_signals[ticker] = get_signal_combined(cz, mhz, rsi)
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
                    X_ASSET_FIXED, selected_ticker)
        else:
            result = None

        if result and result[0] is not None:
            df_daily, beta, std_resid = result
            cz  = float(df_daily['Z_Score'].iloc[-1])     if pd.notna(df_daily['Z_Score'].iloc[-1])     else 0.0
            mhz = float(df_daily['MACD_Hist_Z'].iloc[-1]) if pd.notna(df_daily['MACD_Hist_Z'].iloc[-1]) else 0.0
            rsi = float(df_daily['RSI'].iloc[-1])          if pd.notna(df_daily['RSI'].iloc[-1])          else 50.0
            st.session_state.ticker_signals[selected_ticker] = get_signal_combined(cz, mhz, rsi)

    st.markdown(build_css(selected_option), unsafe_allow_html=True)
    KST      = datetime.timezone(datetime.timedelta(hours=9))
    queried  = datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M')
    data_lbl = (f"🟢 장중&nbsp;·&nbsp;조회: {queried}" if mkt['is_open']
                else f"🔴 장마감&nbsp;·&nbsp;{mkt['last_trading_label']}&nbsp;·&nbsp;조회: {queried}")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"margin-bottom:1px;padding-bottom:1px;'>"
        f"<b style='font-size:1.15rem;white-space:nowrap;color:#111;'>📊 퀀트 대시보드</b>"
        f"<span style='font-size:10px;color:#999;white-space:nowrap;'>{data_lbl}</span></div>",
        unsafe_allow_html=True)

    btn_col, chart_col = st.columns([1, 6])
    with btn_col:
        for ticker in TARGET_TICKERS:
            pct = pct_changes.get(ticker, 0)
            if st.button(f"**{display_name(ticker)}**   {pct:+.1f}%",
                         key=f"ticker_btn_{safe_key(ticker)}", use_container_width=True):
                st.session_state.selected_option     = ticker
                st.session_state.custom_ticker_input = ''
                st.rerun()
        if st.button(DIRECT_INPUT_LABEL, key="ticker_btn_direct", use_container_width=True):
            st.session_state.selected_option = DIRECT_INPUT_LABEL
            st.rerun()
        if selected_option == DIRECT_INPUT_LABEL:
            custom_input = st.text_input("티커", value=st.session_state.get('custom_ticker_input', ''),
                                         placeholder="NVDA", label_visibility="collapsed")
            new_val = custom_input.strip().upper()
            if new_val != st.session_state.get('custom_ticker_input', ''):
                st.session_state.custom_ticker_input = new_val
                st.rerun()
        if st.button("🔄 refresh", key="full_refresh_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with chart_col:
        if df_daily is not None:
            # ★ 포지션 트래커 (차트 위)
            render_position_tracker(selected_ticker, df_daily)

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
                            df_raw[[col_raw]], X_ASSET_FIXED, selected_ticker)
                        if result_raw[0] is not None:
                            df_daily_raw = result_raw[0]
            render_chart(df_daily, selected_ticker, beta, std_resid,
                         cfg['guide_n'], cfg['view_months'], df_ohlc, df_daily_raw)
        elif selected_option == DIRECT_INPUT_LABEL:
            if not st.session_state.get('custom_ticker_input', ''):
                st.info("왼쪽에서 티커를 입력해 주세요. (예: NVDA, 000660)")
            else:
                st.error(f"'{st.session_state.custom_ticker_input}' 데이터를 가져올 수 없습니다.")
        elif selected_ticker:
            st.error("분석에 필요한 데이터가 부족합니다.")

    if selected_ticker:
        render_memo_section(selected_ticker)

    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
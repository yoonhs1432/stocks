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
TARGET_TICKERS     = ['SPYU', 'SOXL','TQQQ' , 'FNGU',
                      'HIBL', 'TARK', 'QPUX', 'BNKU',
                      'GDXU', 'KORU', '005930','UPXI',
                      'BTC-USD', 'BITU', 'ETH-USD', 'ETHT',
                      'RKLB','RKLX','OKLO','OKLL',
                      'AVAV','AVXX','TEM','TEMT',
                      'IREN','IRE','CRCL','CRCA',
                     ]
TICKER_DISPLAY_NAMES: dict = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    '005930':  '삼전',
    '000660':  '하닉',
}
DEFAULT_REFRESH_MINS = 10  # 기본 자동 새로고침 간격 (분)

# ── 5단계 투자의견 (한국 증시 관례: 상승=빨강, 하락=파랑) ──
SIGNAL_STYLE: dict = {
    'FB': ('#dc2626', '#ffffff'),   # 풀매수
    'B':  ('#fca5a5', '#1a1a1a'),   # 매수
    'H':  ('#9ca3af', '#ffffff'),   # 관망
    'S':  ('#93c5fd', '#1a1a1a'),   # 매도
    'FS': ('#1d4ed8', '#ffffff'),   # 풀매도
}
ACTION_LABELS: dict = {
    'FB': '풀 매수', 'B': '매수', 'H': '관망', 'S': '매도', 'FS': '풀 매도',
}
LEGEND_ITEMS = [
    ('#dc2626', '풀매수'), ('#fca5a5', '매수'), ('#9ca3af', '관망'),
    ('#93c5fd', '매도'), ('#1d4ed8', '풀매도'),
]

def display_name(ticker: str) -> str:
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)

def safe_key(ticker: str) -> str:
    return ticker.replace('-', '_').replace('.', '_').replace('/', '_')

# ====================================================
# 2. 설정 · 매매 기록 영속화
# ====================================================
TRADE_FILE    = 'trade_history.json'
SETTINGS_FILE = 'settings.json'
GIST_FILENAME = 'quant_trade_history.json'

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

# ── GitHub Gist (클라우드 매매 기록 동기화) ──────────
# Streamlit Cloud: .streamlit/secrets.toml 에 추가
#   GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"
#   GIST_ID      = "abc123def456..."
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

def init_session_state() -> None:
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = load_trade_history()
    if 'ticker_signals' not in st.session_state:
        st.session_state.ticker_signals = {}
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = TARGET_TICKERS[0]
    if 'custom_ticker_input' not in st.session_state:
        st.session_state.custom_ticker_input = ''
    if 'last_data_date' not in st.session_state:
        st.session_state.last_data_date = ''
    # view_months: 설정 파일에서 복원
    if 'view_months' not in st.session_state:
        st.session_state.view_months = load_settings().get('view_months', 12)
    if 'analysis_start' not in st.session_state:
        st.session_state.analysis_start = load_settings().get('analysis_start', '2021-01-01')
    if 'refresh_mins' not in st.session_state:
        st.session_state.refresh_mins = DEFAULT_REFRESH_MINS

# ====================================================
# 3. 투자의견 (5단계)
# ====================================================
def get_signal(current_z: float = 0.0) -> str:
    """Z-Score 기반 5단계 신호 반환."""
    if current_z >= 1.5:   return 'FS'
    if current_z >= 0.5:   return 'S'
    if current_z <= -1.5:  return 'FB'
    if current_z <= -0.5:  return 'B'
    return 'H'

# ====================================================
# 4. 데이터 다운로드 (캐싱)
# ====================================================
@st.cache_data(show_spinner=False)
def fetch_all_data(tickers: list, start_date_str: str) -> pd.DataFrame:
    df_list = []
    for ticker in [X_ASSET_FIXED] + list(tickers):
        try:
            data = fdr.DataReader(ticker, start_date_str)
            if not data.empty:
                data = data[~data.index.duplicated(keep='last')]
                df_list.append(data[['Close']].rename(columns={'Close': f'{ticker}_Close'}))
        except Exception:
            continue
    return pd.concat(df_list, axis=1).ffill() if df_list else pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_single_ticker(ticker: str, start_date_str: str) -> pd.DataFrame:
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if not data.empty:
            data = data[~data.index.duplicated(keep='last')]
            return data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
    except Exception:
        pass
    return pd.DataFrame()

# ====================================================
# 6. 데이터 처리
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
    df, beta = _compute_normalized(df, x_name, y_name)
    df_daily = df.copy()
    df_daily, std_resid = _compute_indicators(df_daily, y_name)
    return df_daily, beta, std_resid

# ====================================================
# 7. 전체 종목 일괄 분석 (캐싱)
#    캐시 히트 시 종목 버튼 클릭 → 즉시 렌더링 (재분석 없음)
# ====================================================
@st.cache_data(show_spinner=False)
def compute_all_analyses(df_close: pd.DataFrame) -> dict:
    """TARGET_TICKERS 전체를 한 번에 분석. 결과 캐싱으로 종목 전환을 즉각 처리."""
    results: dict = {}
    df_x = df_close[[f'{X_ASSET_FIXED}_Close']]
    for ticker in TARGET_TICKERS:
        col = f'{ticker}_Close'
        if col not in df_close.columns:
            results[ticker] = None
            continue
        try:
            price_series = df_close[col].dropna()
            df_y         = df_close[[col]]
            results[ticker] = process_asset_data(
                df_x, df_y, X_ASSET_FIXED, ticker)
        except Exception:
            results[ticker] = None
    return results

# ====================================================
# 8. 차트 헬퍼
# ====================================================
def add_filled_blocks(fig, df: pd.DataFrame, y_col: str, condition: pd.Series,
                      color: str, row: int, col: int, baseline_y: float) -> None:
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

# ====================================================
# 9. 사이드바
# ====================================================
def render_sidebar(selected_ticker: str) -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ 분석 파라미터")
        analysis_start = st.text_input("분석 시작일 (예시:2023-01-01)",
                                       value=st.session_state.analysis_start)
        view_months  = st.number_input("차트 조회 기간 (최근 N개월)",
                                       min_value=1, max_value=240,
                                       value=st.session_state.view_months, step=1)
        guide_n      = st.number_input("가이드라인 기울기 (n)",
                                       min_value=1, max_value=20, value=4, step=1)
        refresh_mins = st.number_input("자동 새로고침 (분)",
                                       min_value=1, max_value=120,
                                       value=st.session_state.refresh_mins, step=1)
        st.markdown("---")
        st.markdown("### 📝 매매 기록 관리")
        _tok, _gid = _gist_cfg()
        st.caption(f"☁️ Gist 연동됨 (`{_gid[:8]}...`)" if (_tok and _gid)
                   else "💾 로컬 저장 (Gist 미설정)")

        st.markdown("**➕ 새로운 기록 추가**")
        ticker_options = (TARGET_TICKERS if selected_ticker in TARGET_TICKERS
                          else [selected_ticker] + TARGET_TICKERS)
        t_ticker = st.selectbox("종목", ticker_options,
                                index=ticker_options.index(selected_ticker))
        t_date   = st.date_input("날짜", datetime.date.today())
        t_type   = st.radio("종류", ['buy', 'sell'], horizontal=True)
        if st.button("기록 저장"):
            st.session_state.trade_history.setdefault(t_ticker, []).append(
                {'date': t_date.strftime("%Y-%m-%d"), 'type': t_type})
            save_trade_history(st.session_state.trade_history)
            st.success("저장 완료!")
            st.rerun()

        st.markdown("---")
        st.markdown("**🗑️ 기존 기록 삭제**")
        history = st.session_state.trade_history
        if selected_ticker in history and history[selected_ticker]:
            for i, record in enumerate(history[selected_ticker]):
                t      = record['type'].upper()
                color  = '#dc2626' if t == 'BUY' else '#1d4ed8'
                cols   = st.columns([6, 1])
                cols[0].markdown(
                    f"<span style='font-size:12px;'>{record['date']}&nbsp;"
                    f"<b style='color:{color};'>{t}</b></span>",
                    unsafe_allow_html=True)
                if cols[1].button("✕", key=f"del_{selected_ticker}_{i}"):
                    st.session_state.trade_history[selected_ticker].pop(i)
                    save_trade_history(st.session_state.trade_history)
                    st.rerun()
        else:
            st.caption("매매 기록이 없습니다.")

    return {'analysis_start': analysis_start.strip(), 'view_months': int(view_months),
            'guide_n': guide_n, 'refresh_mins': int(refresh_mins)}

# ====================================================
# 10. 차트 렌더링
# ====================================================
def render_chart(df_daily: pd.DataFrame, selected_ticker: str,
                 beta: float, std_resid: float,
                 guide_n: int, view_months: int) -> None:
    show_indicators = True
    st.markdown("""
    <style>
    .js-plotly-plot, .js-plotly-plot .plotly, .js-plotly-plot svg {
        touch-action: none !important;
    }
    </style>""", unsafe_allow_html=True)

    PX = {'main': 150, 'spacer': 20, 'price': 91, 'zscore': 91, 'macd': 91, 'rsi': 91}
    active_plots = ['main', 'spacer', 'price', 'zscore']
    if show_indicators:
        active_plots += ['macd', 'rsi']
    total_rows  = len(active_plots)
    total_h     = sum(PX[p] for p in active_plots)
    row_heights = [PX[p] / total_h for p in active_plots]
    fig         = make_subplots(rows=total_rows, cols=1,
                                row_heights=row_heights, vertical_spacing=0.02)
    current_row = 1

    # ── [1] 로그-로그 산점도 ──
    sdf    = df_daily.sort_values(f'{X_ASSET_FIXED}_Norm')
    x_vals = sdf[f'{X_ASSET_FIXED}_Norm']
    min_x  = df_daily[f'{X_ASSET_FIXED}_Norm'].min()
    max_x  = df_daily[f'{X_ASSET_FIXED}_Norm'].max()
    min_y  = df_daily[f'{selected_ticker}_Norm'].min()
    max_y  = df_daily[f'{selected_ticker}_Norm'].max()
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
    # y범위: 실제 데이터 + 회귀밴드 기준 (가이드라인 제외)
    band_upper = np.exp(np.log(sdf['Predicted'].values) + 1.5 * std_resid)
    band_lower = np.exp(np.log(sdf['Predicted'].values) - 1.5 * std_resid)
    y_all  = np.concatenate([df_daily[f'{selected_ticker}_Norm'].dropna().values,
                              band_upper, band_lower])
    y_lo   = np.nanmin(y_all)
    y_hi   = np.nanmax(y_all)
    fig.update_xaxes(type="log", title_text="", showgrid=False,
                     range=[np.log10(min_x * 0.98), np.log10(max_x * 1.02)],
                     row=current_row, col=1)
    fig.update_yaxes(type="log", showgrid=False,
                     range=[np.log10(y_lo * 0.88), np.log10(y_hi * 1.18)],
                     row=current_row, col=1)
    fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
        text=f"<b>{display_name(selected_ticker)}</b>", showarrow=False,
        font=dict(size=11, color='black'), bgcolor='white',
        bordercolor='black', borderwidth=1, borderpad=2,
        xanchor='left', yanchor='top', row=current_row, col=1)
    current_row += 1

    # ── [2] Spacer ──

    fig.update_xaxes(visible=False, row=current_row, col=1)
    fig.update_yaxes(visible=False, row=current_row, col=1)
    current_row += 1

    # ── 뷰 기간 재정규화 ──
    view_start = df_daily.index[-1] - pd.DateOffset(months=view_months)
    view_df    = df_daily[df_daily.index >= view_start]
    base_spy   = view_df[f'{X_ASSET_FIXED}_Norm'].iloc[0] if not view_df.empty else 1.0
    base_tkr   = view_df[f'{selected_ticker}_Norm'].iloc[0] if not view_df.empty else 1.0
    df_daily['Plot_Norm_SPY']    = df_daily[f'{X_ASSET_FIXED}_Norm'] / base_spy
    df_daily['Plot_Norm_Ticker'] = df_daily[f'{selected_ticker}_Norm'] / base_tkr

    # ── [3] Price ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Plot_Norm_SPY'],
                              mode='lines', line=dict(color='gray', width=1.5),
                              name=X_ASSET_FIXED),
                  row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
                              mode='lines', line=dict(color='black', width=1.5),
                              name=selected_ticker),
                  row=current_row, col=1)
    min_price = (df_daily.loc[df_daily.index >= view_start,
                               ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].min().min())
    max_price = (df_daily.loc[df_daily.index >= view_start,
                               ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].max().max())
    price_baseline = min_price * 0.95
    # ── Z-Score 그라데이션: 종목 선 아래만 채우기 ──
    _BANDS = [
        (df_daily['Z_Score'] >= 1.5,                                      'rgba(29,78,216,0.35)'),
        ((df_daily['Z_Score'] >= 0.75) & (df_daily['Z_Score'] <  1.5),   'rgba(93,155,246,0.22)'),
        ((df_daily['Z_Score'] >= 0.0)  & (df_daily['Z_Score'] <  0.75),  'rgba(191,219,254,0.12)'),
        ((df_daily['Z_Score'] >  -0.75) & (df_daily['Z_Score'] < 0.0),   'rgba(254,202,202,0.12)'),
        ((df_daily['Z_Score'] >  -1.5) & (df_daily['Z_Score'] <= -0.75), 'rgba(248,113,113,0.22)'),
        (df_daily['Z_Score'] <= -1.5,                                     'rgba(220,38,38,0.35)'),
    ]
    for cond, color in _BANDS:
        add_filled_blocks(fig, df_daily, 'Plot_Norm_Ticker',
                          cond, color, current_row, 1, price_baseline)
    fig.update_yaxes(type="log",
                     range=[np.log10(price_baseline), np.log10(max_price * 1.05)],
                     row=current_row, col=1)
    fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
        text="<b>Price</b>", showarrow=False,
        font=dict(size=11, color='black'), bgcolor='white',
        bordercolor='black', borderwidth=1, borderpad=3,
        xanchor='left', yanchor='top', row=current_row, col=1)
    time_x_axis = f'x{current_row}'
    current_row += 1

    # ── [4] Z-Score ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Z_Score'],
                              line=dict(color='black', width=1.5), name='Z-Score'),
                  row=current_row, col=1)
    fig.add_hline(y=1.5,  line_dash="dash", line_color="blue",
                  line_width=1.5, row=current_row, col=1)
    fig.add_hline(y=-1.5, line_dash="dash", line_color="red",
                  line_width=1.5, row=current_row, col=1)
    fig.add_hline(y=0,    line_dash="dot",  line_color="gray",
                  line_width=1.0, row=current_row, col=1)
    # 매수 구간 (Z ≤ -1.5): 빨간 채움
    add_filled_blocks(fig, df_daily, 'Z_Score',
                      df_daily['Z_Score'] <= -1.5,
                      'rgba(220,38,38,0.20)', current_row, 1, -1.5)
    # 매도 구간 (Z ≥ +1.5): 파란 채움
    add_filled_blocks(fig, df_daily, 'Z_Score',
                      df_daily['Z_Score'] >= 1.5,
                      'rgba(29,78,216,0.20)', current_row, 1, 1.5)
    z_view   = df_daily.loc[df_daily.index >= view_start, 'Z_Score'].dropna()
    z_lo     = min(-2.0, z_view.min() if not z_view.empty else -2.0)
    z_hi     = max( 2.0, z_view.max() if not z_view.empty else  2.0)
    fig.update_yaxes(range=[z_lo - 0.2, z_hi + 0.2], row=current_row, col=1)
    fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
        text="<b>Z-Score</b>", showarrow=False,
        font=dict(size=11, color='black'), bgcolor='white',
        bordercolor='black', borderwidth=1, borderpad=3,
        xanchor='left', yanchor='top', row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ── [5][6] 보조 지표 ──
    if show_indicators:
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
        fig.update_yaxes(row=current_row, col=1)
        fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
            text="<b>MACD</b>", showarrow=False,
            font=dict(size=11, color='black'), bgcolor='white',
            bordercolor='black', borderwidth=1, borderpad=3,
            xanchor='left', yanchor='top', row=current_row, col=1)
        fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
        current_row += 1
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['RSI'],
                                  line=dict(color='purple', width=1.5), name='RSI'),
                      row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",  line_width=1.2,
                      row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", line_width=1.2,
                      row=current_row, col=1)
        rsi_view = df_daily.loc[df_daily.index >= view_start, 'RSI'].dropna()
        rsi_lo   = min(20.0, rsi_view.min() if not rsi_view.empty else 20.0)
        rsi_hi   = max(80.0, rsi_view.max() if not rsi_view.empty else 80.0)
        fig.update_yaxes(range=[rsi_lo - 2, rsi_hi + 2], row=current_row, col=1)
        fig.add_annotation(x=0, y=1, xref='x domain', yref='y domain',
            text="<b>RSI</b>", showarrow=False,
            font=dict(size=11, color='black'), bgcolor='white',
            bordercolor='black', borderwidth=1, borderpad=3,
            xanchor='left', yanchor='top', row=current_row, col=1)
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
    for r in range(3, total_rows):
        fig.update_xaxes(showticklabels=False, tickformat="%y/%m/%d", row=r, col=1)
    fig.update_xaxes(showticklabels=True, tickformat="%y/%m/%d", row=total_rows, col=1)
    fig.update_layout(
        height=total_h, showlegend=False, hovermode='x unified',
        dragmode='pan', margin=dict(l=2, r=18, t=12, b=20),
        paper_bgcolor='white', plot_bgcolor='white')
    fig.update_xaxes(range=[view_start, df_daily.index[-1]], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True,
                    config={'scrollZoom': True, 'displayModeBar': False,
                            'doubleClick': 'reset', 'responsive': True})

# ====================================================
# 11. 메인
# ====================================================
def main():
    init_session_state()

    # ── 자동 새로고침 트리거 감지 ──────────────────────
    # JS가 ?_ar=1 을 붙여 리로드 → 캐시 전체 클리어 → 새 데이터로 재분석
    # if st.query_params.get('_ar') == '1':
    #     st.query_params.clear()
    #     st.cache_data.clear()

    # ── 종목 선택 상태 ──
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

    # ── 사이드바 (analysis_start 포함) ──
    cfg = render_sidebar(selected_ticker or TARGET_TICKERS[0])

    # 설정 변경 → settings.json 저장
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

    st.session_state.refresh_mins = cfg['refresh_mins']

    analysis_start = st.session_state.analysis_start

    # ── 데이터 로드 ──
    with st.spinner("데이터 로드 중..."):
        df_close = fetch_all_data(TARGET_TICKERS, analysis_start)

    if not df_close.empty:
        st.session_state.last_data_date = df_close.index[-1].strftime('%Y-%m-%d')

    # ── 커스텀 티커 fetch ──
    if selected_ticker and f'{selected_ticker}_Close' not in df_close.columns:
        with st.spinner(f"{selected_ticker} 데이터를 불러오는 중..."):
            df_custom = fetch_single_ticker(selected_ticker, analysis_start)
        if not df_custom.empty:
            df_close = pd.concat([df_close, df_custom], axis=1).ffill()
        else:
            selected_ticker = None

    # ── 전체 종목 일괄 분석 (캐시 히트 시 즉시) ──
    with st.spinner("전체 종목 분석 중... (최초 실행 시 수십 초 소요)"):
        all_analyses = compute_all_analyses(df_close)

    # 버튼 색상용 신호 갱신
    for ticker, result in all_analyses.items():
        if result and result[0] is not None:
            df_t, _, _ = result
            cz = float(df_t['Z_Score'].iloc[-1]) if pd.notna(df_t['Z_Score'].iloc[-1]) else 0.0
            st.session_state.ticker_signals[ticker] = get_signal(cz)
        else:
            st.session_state.ticker_signals.setdefault(ticker, 'H')

    # ── 선택 종목 분석 결과 추출 ──
    df_daily = beta = std_resid = None

    if selected_ticker and selected_ticker in TARGET_TICKERS:
        result = all_analyses.get(selected_ticker)
        if result and result[0] is not None:
            df_daily, beta, std_resid = result
            cz = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
            st.session_state.ticker_signals[selected_ticker] = get_signal(cz)

    elif selected_ticker and f'{selected_ticker}_Close' in df_close.columns:
        # 커스텀 티커: 온디맨드 분석
        price_series = df_close[f'{selected_ticker}_Close'].dropna()
        with st.spinner(f"{display_name(selected_ticker)} 분석 중..."):
            result = process_asset_data(
                df_close[[f'{X_ASSET_FIXED}_Close']],
                df_close[[f'{selected_ticker}_Close']],
                X_ASSET_FIXED, selected_ticker)
        if result[0] is not None:
            df_daily, beta, std_resid = result
            cz = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
            st.session_state.ticker_signals[selected_ticker] = get_signal(cz)

    # ════════════════════════════════════════
    #  CSS: 레이아웃 + 버튼 색상
    # ════════════════════════════════════════
    btn_css_parts = []
    for ticker in TARGET_TICKERS:
        sig    = st.session_state.ticker_signals.get(ticker, 'H')
        bg, fg = SIGNAL_STYLE.get(sig, ('#9ca3af', '#fff'))
        k      = f"ticker_btn_{safe_key(ticker)}"
        is_sel = (selected_option == ticker)
        sel_extra = (f"box-shadow:0 0 0 2px #fff,0 0 0 4px {bg}!important;"
                     "transform:scale(1.03);") if is_sel else ""
        btn_css_parts.append(f"""
        div.st-key-{k} button {{
            background:{bg}!important; border-color:{bg}!important;
            color:#111!important; font-weight:500!important;
            height:1.9rem!important; font-size:0.76rem!important;
            padding:0!important; line-height:1!important;
            min-height:0!important; border-radius:3px!important;
            {sel_extra}
        }}
        div.st-key-{k} button:hover {{ opacity:0.82!important; }}""")
    di_sel = (selected_option == DIRECT_INPUT_LABEL)
    btn_css_parts.append(f"""
    div.st-key-ticker_btn_direct button {{
        height:1.1rem!important; font-size:0.55rem!important;
        padding:0!important; min-height:0!important; border-radius:3px!important;
        {'border:2px solid #1565C0!important;font-weight:700!important;' if di_sel else ''}
    }}""")

    global_css = f"""
    <style>
    .block-container {{
        padding-top: 3.5rem !important; padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {{
        flex-wrap: nowrap !important; gap: 5px !important;
        align-items: flex-start !important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {{
        flex: 0 0 87px !important; min-width: 87px !important;
        max-width: 87px !important; padding: 0 !important;
    }}
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {{
        flex: 1 1 0 !important; min-width: 0 !important;
        overflow: visible !important;
        padding-left: 2px !important; padding-right: 2px !important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"] > div {{
        margin-bottom: 1px !important; padding: 0 !important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child div[data-testid="stVerticalBlock"] {{
        gap: 3px !important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] {{
        gap: 3px !important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
        flex: 1 1 0 !important; min-width: 0 !important;
        max-width: none !important; padding: 0 !important;
    }}
    section[data-testid="stMain"] div[data-testid="stColumn"]:first-child button p {{
        margin: 0 !important; padding: 0 !important;
        font-size: 0.73rem !important; line-height: 1 !important; font-weight: 500 !important; color: #111 !important;
    }}
    {''.join(btn_css_parts)}
    </style>"""
    st.markdown(global_css, unsafe_allow_html=True)

    # ── 제목 ──
    KST        = datetime.timezone(datetime.timedelta(hours=9))
    queried_at = datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M')
    date_part  = (f"기준: {st.session_state.last_data_date}&nbsp;·&nbsp;"
                  if st.session_state.last_data_date else "")
    next_refresh = (datetime.datetime.now(KST)
                    + datetime.timedelta(minutes=cfg['refresh_mins']))
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"margin-bottom:1px;padding-bottom:1px;'>"
        f"<b style='font-size:1.15rem;white-space:nowrap;color:#111;'>📊 퀀트 대시보드</b>"
        f"<span style='font-size:10px;color:#999;white-space:nowrap;'>"
        f"{date_part}조회: {queried_at}"
        f"&nbsp;·&nbsp;다음 갱신: {next_refresh.strftime('%H:%M')}"
        f"</span></div>",
        unsafe_allow_html=True)

    # ── 자동 새로고침 JS ──────────────────────────────
    # N분 후 ?_ar=1 로 리다이렉트 → Python 쪽에서 캐시 클리어
    refresh_ms = cfg['refresh_mins'] * 60 * 1000
    st.markdown(f"""
    <script>
    (function() {{
        if (window._arTimer) clearTimeout(window._arTimer);
        window._arTimer = setTimeout(function() {{
            window.location.href =
                window.location.pathname + '?_ar=1&t=' + Date.now();
        }}, {refresh_ms});
    }})();
    </script>""", unsafe_allow_html=True)

    # ── 범례 ──
    legend_parts = [
        f"<span style='display:inline-flex;align-items:center;gap:2px;white-space:nowrap;'>"
        f"<span style='width:9px;height:9px;border-radius:2px;"
        f"background:{color};display:inline-block;flex-shrink:0;'></span>{label}</span>"
        for color, label in LEGEND_ITEMS]
    legend_html = (
        "<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:center;"
        "font-size:10px;color:#555;margin-bottom:1px;'>"
        + "&nbsp;".join(legend_parts) + "</div>")

    # ── 요약 카드 ──
    if selected_ticker and df_daily is not None:
        cz          = float(df_daily['Z_Score'].iloc[-1]) if pd.notna(df_daily['Z_Score'].iloc[-1]) else 0.0
        sig         = get_signal(cz)
        action_txt  = ACTION_LABELS.get(sig, '관망')
        bg_c, _     = SIGNAL_STYLE.get(sig, ('#9ca3af', '#fff'))
        z_color     = "#dc2626" if cz <= 0 else "#1d4ed8"
        try:
            since_label = pd.to_datetime(st.session_state.analysis_start).strftime("'%y/%m~")
        except Exception:
            since_label = st.session_state.analysis_start
        summary_html = (
            f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;"
            f"padding:2px 10px;border-radius:6px;border-left:4px solid {bg_c};"
            f"background:{bg_c}18;margin-bottom:2px;'>"
            f"<b style='font-size:18px;color:{bg_c};white-space:nowrap;'>{action_txt}</b>"
            f"<span style='font-size:14px;color:{bg_c};font-weight:700;white-space:nowrap;'>{display_name(selected_ticker)}</span>"
            f"<span style='width:1px;height:13px;background:#ddd;display:inline-block;'></span>"
            f"<span style='font-size:13px;color:#666;'>Z-Score&nbsp;"
            f"<b style='color:{z_color};'>{cz:+.2f}</b></span>"
            f"<span style='font-size:13px;color:#666;'>β&nbsp;"
            f"<b style='color:#333;'>{beta:.2f}</b></span>"
            f"<span style='font-size:13px;color:#666;'>기간&nbsp;"
            f"<b style='color:#333;'>{since_label}</b></span>"
            f"</div>")
    else:
        summary_html = "<div style='margin-bottom:3px;'></div>"

    st.markdown(legend_html + summary_html, unsafe_allow_html=True)

    # ── 버튼 + 차트 레이아웃 ──
    btn_col, chart_col = st.columns([1, 5])

    with btn_col:
        for i in range(0, len(TARGET_TICKERS), 2):
            c1, c2 = st.columns(2, gap="small")
            for col_widget, ticker in zip([c1, c2], TARGET_TICKERS[i:i+2]):
                btn_key = f"ticker_btn_{safe_key(ticker)}"
                if col_widget.button(display_name(ticker), key=btn_key, use_container_width=True):
                    st.session_state.selected_option     = ticker
                    st.session_state.custom_ticker_input = ''
                    st.rerun()
        if st.button(DIRECT_INPUT_LABEL, key="ticker_btn_direct", use_container_width=True):
            st.session_state.selected_option = DIRECT_INPUT_LABEL
            st.rerun()
        if selected_option == DIRECT_INPUT_LABEL:
            custom_input = st.text_input(
                "티커",
                value=st.session_state.get('custom_ticker_input', ''),
                placeholder="NVDA", label_visibility="collapsed")
            new_val = custom_input.strip().upper()
            if new_val != st.session_state.get('custom_ticker_input', ''):
                st.session_state.custom_ticker_input = new_val
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

if __name__ == "__main__":
    main()

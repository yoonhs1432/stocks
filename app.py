import time
import datetime
import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import FinanceDataReader as fdr
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====================================================
# 1. 전역 설정
# ====================================================
st.set_page_config(page_title="퀀트 트레이딩 대시보드", layout="wide")

X_ASSET_FIXED = 'SPY'
TARGET_TICKERS = ['SPYU', 'TQQQ', 'SOXL', 'FNGU', 'BTC-USD', 'BITU', 'ETH-USD', 'ETHT',
                  'HIBL', 'TARK', 'QPUX', 'BNKU', 'GDXU', 'KORU', '005930', 'UPXI']

# 티커 -> 화면 표시명 매핑 (없으면 티커 그대로 사용)
TICKER_DISPLAY_NAMES: dict = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    '005930':  '삼성전자',
}

def display_name(ticker: str) -> str:
    return TICKER_DISPLAY_NAMES.get(ticker, ticker)

TRADE_FILE = 'trade_history.json'

# ====================================================
# 2. 매매 기록 관리 (session_state 기반)
# ====================================================
def load_trade_history() -> dict:
    """JSON 파일에서 매매 기록 로드. 파일 없으면 기본값 생성."""
    if os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    default = {
        'BITU': [{'date': '2026-04-01', 'type': 'buy'}, {'date': '2026-04-13', 'type': 'buy'}],
        'TQQQ': [{'date': '2026-04-01', 'type': 'buy'}]
    }
    save_trade_history(default)
    return default


def save_trade_history(history: dict) -> None:
    """매매 기록을 JSON 파일에 저장."""
    with open(TRADE_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def init_session_state() -> None:
    """session_state 초기화. 앱 최초 실행 시 한 번만 호출."""
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = load_trade_history()
    if 'ticker_signals' not in st.session_state:
        st.session_state.ticker_signals = {}  # {ticker: emoji}


def get_action_emoji(win_prob, expected_return) -> str:
    """승률·기대수익으로 투자의견 이모지만 반환."""
    if win_prob is None or expected_return is None:
        return '⚪'
    if win_prob >= 60.0 and expected_return >= 8.0:
        return '🔥'
    elif win_prob >= 55.0 and expected_return >= 4.0:
        return '🟢'
    elif win_prob >= 52.0 and expected_return >= 1.0:
        return '🟡'
    elif win_prob <= 40.0 and expected_return <= -8.0:
        return '🧊'
    elif win_prob <= 45.0 and expected_return <= -4.0:
        return '🔴'
    elif win_prob <= 48.0 and expected_return <= -1.0:
        return '🟠'
    return '⚪'


# ====================================================
# 3. 데이터 다운로드 (캐싱)
# ====================================================
@st.cache_data(show_spinner=False)
def fetch_all_data(tickers: list, start_date_str: str) -> pd.DataFrame:
    """모든 종목의 종가 데이터를 받아 합친 DataFrame 반환."""
    df_list = []
    all_tickers = [X_ASSET_FIXED] + tickers
    for ticker in all_tickers:
        try:
            data = fdr.DataReader(ticker, start_date_str)
            if not data.empty:
                data = data[~data.index.duplicated(keep='last')]
                df_list.append(data[['Close']].rename(columns={'Close': f'{ticker}_Close'}))
        except Exception:
            continue
    if df_list:
        return pd.concat(df_list, axis=1).ffill()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_single_ticker(ticker: str, start_date_str: str) -> pd.DataFrame:
    """단일 티커 데이터 로드 (커스텀 입력용)."""
    try:
        data = fdr.DataReader(ticker, start_date_str)
        if not data.empty:
            data = data[~data.index.duplicated(keep='last')]
            return data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
    except Exception:
        pass
    return pd.DataFrame()


# ====================================================
# 4. 주기 탐색 (캐싱)
# ====================================================
@st.cache_data(show_spinner=False)
def find_available_cycles(price_series: pd.Series, min_w: int = 20, max_w: int = 365, step: int = 5) -> list:
    price_series = price_series.dropna()
    cycles = set()
    for w in range(min_w, max_w + 1, step):
        roll_max = price_series.rolling(window=w, center=True, min_periods=3).max()
        roll_min = price_series.rolling(window=w, center=True, min_periods=3).min()
        is_extrema = (price_series == roll_max) | (price_series == roll_min)
        extrema_indices = np.where(is_extrema)[0]
        if len(extrema_indices) > 1:
            avg_c = int(np.diff(extrema_indices).mean())
            if min_w <= avg_c <= max_w:
                cycles.add(avg_c)
    return sorted(list(cycles)) if cycles else [60]


@st.cache_data(show_spinner=False)
def get_optimal_cycle(price_series: pd.Series, cycles_list: list) -> int:
    if not cycles_list:
        return 60
    detrended = price_series - price_series.rolling(window=100, min_periods=1).mean()
    best_cycle = cycles_list[len(cycles_list) // 2]
    best_score = -np.inf
    for c in cycles_list:
        if len(detrended.dropna()) > c * 2:
            score = detrended.autocorr(lag=c)
            if pd.notna(score) and score > best_score:
                best_score = score
                best_cycle = c
    return best_cycle


# ====================================================
# 5. 데이터 처리 (역할 분리)
# ====================================================
def _compute_normalized(df: pd.DataFrame, x_name: str, y_name: str) -> pd.DataFrame:
    """정규화 및 베타 회귀 계산."""
    df = df.copy()
    base_x = df[f'{x_name}_Close'].iloc[0]
    base_y = df[f'{y_name}_Close'].iloc[0]
    df[f'{x_name}_Norm'] = df[f'{x_name}_Close'] / base_x
    df[f'{y_name}_Norm'] = df[f'{y_name}_Close'] / base_y

    log_x = np.log(df[f'{x_name}_Norm'])
    log_y = np.log(df[f'{y_name}_Norm'])
    model = LinearRegression().fit(log_x.values.reshape(-1, 1), log_y.values)
    beta = model.coef_[0]
    df['Predicted'] = np.exp(model.intercept_) * df[f'{x_name}_Norm'] ** beta
    return df, beta, model


def _compute_indicators(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    """RSI / MACD / Z-Score 계산."""
    df = df.copy()
    close = df[f'{y_name}_Close']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Z-Score (잔차 기준)
    log_resid = np.log(df[f'{y_name}_Norm']) - np.log(df['Predicted'])
    std_resid = log_resid.std()
    df['Z_Score'] = log_resid / std_resid

    return df, std_resid


def _compute_targets(df: pd.DataFrame, y_name: str, selected_cycle: int) -> pd.DataFrame:
    """미래 평균 가격 기반 Target / Target_Return 계산."""
    df = df.copy()
    shifted = df[f'{y_name}_Norm'].shift(-1)
    df['Future_Avg_Price'] = shifted.iloc[::-1].rolling(window=selected_cycle).mean().iloc[::-1]
    df['Target'] = np.where(
        df['Future_Avg_Price'].isna(), np.nan,
        (df['Future_Avg_Price'] > df[f'{y_name}_Norm']).astype(int)
    )
    df['Target_Return'] = np.where(
        df['Future_Avg_Price'].isna(), np.nan,
        (df['Future_Avg_Price'] / df[f'{y_name}_Norm'] - 1) * 100
    )
    return df


def _train_and_predict(df: pd.DataFrame, train_end_date) -> tuple:
    """ML 학습 및 예측. (win_prob, sell_prob, expected_return) 반환."""
    for col in ['Win_Prob', 'Sell_Prob', 'Expected_Return']:
        df[col] = np.nan

    features = ['Z_Score', 'RSI', 'MACD_Hist']
    ml_df = df[features + ['Target', 'Target_Return']].dropna()
    train_df = ml_df.loc[:train_end_date]

    win_prob = sell_prob = expected_return = None

    if len(train_df) > 30:
        X_tr = train_df[features]
        clf = LogisticRegression(class_weight='balanced').fit(X_tr, train_df['Target'])
        reg = Ridge(alpha=1.0).fit(X_tr, train_df['Target_Return'])

        valid_idx = df.dropna(subset=features).index
        feat_valid = df.loc[valid_idx, features]

        if not feat_valid.empty:
            probs = clf.predict_proba(feat_valid)
            df.loc[valid_idx, 'Win_Prob'] = probs[:, 1] * 100
            df.loc[valid_idx, 'Sell_Prob'] = probs[:, 0] * 100
            win_prob, sell_prob = probs[-1, 1] * 100, probs[-1, 0] * 100

            evs = reg.predict(feat_valid)
            df.loc[valid_idx, 'Expected_Return'] = evs
            expected_return = evs[-1]

    return df, win_prob, sell_prob, expected_return


def process_asset_data(df_x: pd.DataFrame, df_y: pd.DataFrame,
                       x_name: str, y_name: str,
                       selected_cycle: int, train_end_date) -> tuple:
    """전체 처리 파이프라인 조합."""
    df = pd.merge(df_x, df_y, left_index=True, right_index=True).dropna().sort_index()
    if df.empty:
        return (None,) * 8

    df, beta, _ = _compute_normalized(df, x_name, y_name)
    df_daily = df.copy()
    df_daily, std_resid = _compute_indicators(df_daily, y_name)
    df_daily = _compute_targets(df_daily, y_name, selected_cycle)
    df_daily, win_prob, sell_prob, expected_return = _train_and_predict(df_daily, train_end_date)

    return df, df_daily, beta, win_prob, sell_prob, expected_return, selected_cycle, std_resid


# ====================================================
# 6. 차트 헬퍼
# ====================================================
def add_filled_blocks(fig, df: pd.DataFrame, y_col: str, condition: pd.Series,
                      color: str, row: int, col: int, baseline_y: float) -> None:
    """조건 구간을 면적으로 채워 표시."""
    valid_data = df[condition]
    if valid_data.empty:
        return
    groups = (~condition).cumsum()[condition]
    for _, group in valid_data.groupby(groups):
        if len(group) < 1:
            continue
        fig.add_trace(go.Scatter(
            x=group.index, y=[baseline_y] * len(group),
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip'
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=group.index, y=group[y_col],
            mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
            fill='tonexty', fillcolor=color,
            showlegend=False, hoverinfo='skip'
        ), row=row, col=col)


# ====================================================
# 7. 사이드바 UI
# ====================================================
def render_sidebar(df_close: pd.DataFrame, selected_ticker: str) -> dict:
    """사이드바를 그리고 사용자 설정값을 dict로 반환."""
    with st.sidebar:
        st.markdown("### ⚙️ 분석 파라미터 조절")
        train_start = st.text_input("학습 시작 (YYYY-MM)", "2021-01")
        train_end = st.text_input("학습 종료 (YYYY-MM)", "2026-03")
        view_months = st.number_input("차트 조회 기간 (최근 N개월)", min_value=1, max_value=240, value=12, step=1)
        guide_n = st.number_input("가이드라인 기울기 (n)", min_value=1, max_value=20, value=4, step=1)

        # 주기 선택 (df_close가 이미 로드된 상태에서 호출되므로 블로킹 없음)
        selected_cycle = 60
        if not df_close.empty and f'{selected_ticker}_Close' in df_close.columns:
            cycles_list = find_available_cycles(df_close[f'{selected_ticker}_Close'], min_w=20, max_w=365)
            optimal_cycle = get_optimal_cycle(df_close[f'{selected_ticker}_Close'], cycles_list)
            default_index = cycles_list.index(optimal_cycle) if optimal_cycle in cycles_list else 0
            selected_cycle = st.selectbox(
                f"🔄 분석 주기 선택 (추천: {optimal_cycle}일)", cycles_list, index=default_index
            )

        st.markdown("---")
        show_indicators = st.checkbox("MACD&RSI 표시", value=False)

        st.markdown("---")
        st.markdown("### 📝 매매 기록 관리")

        # 기록 추가
        with st.expander("➕ 새로운 기록 추가"):
            # 커스텀 티커는 목록에 없을 수 있으므로 동적으로 추가
            ticker_options = TARGET_TICKERS if selected_ticker in TARGET_TICKERS \
                else [selected_ticker] + TARGET_TICKERS
            default_idx = ticker_options.index(selected_ticker)
            t_ticker = st.selectbox("종목", ticker_options, index=default_idx)
            t_date = st.date_input("날짜", datetime.date.today())
            t_type = st.radio("종류", ['buy', 'sell'], horizontal=True)
            if st.button("기록 저장"):
                if t_ticker not in st.session_state.trade_history:
                    st.session_state.trade_history[t_ticker] = []
                st.session_state.trade_history[t_ticker].append(
                    {'date': t_date.strftime("%Y-%m-%d"), 'type': t_type}
                )
                save_trade_history(st.session_state.trade_history)
                st.success("저장 완료!")
                st.rerun()

        # 기록 삭제
        with st.expander("🗑️ 기존 기록 삭제"):
            history = st.session_state.trade_history
            if selected_ticker in history and history[selected_ticker]:
                for i, record in enumerate(history[selected_ticker]):
                    cols = st.columns([3, 2, 1])
                    cols[0].write(record['date'])
                    cols[1].write(record['type'].upper())
                    if cols[2].button("🗑️", key=f"del_{selected_ticker}_{i}"):
                        st.session_state.trade_history[selected_ticker].pop(i)
                        save_trade_history(st.session_state.trade_history)
                        st.rerun()
            else:
                st.info("매매 기록이 없습니다.")

    return {
        'train_start': train_start,
        'train_end': train_end,
        'view_months': view_months,
        'guide_n': guide_n,
        'selected_cycle': selected_cycle,
        'show_indicators': show_indicators,
    }


# ====================================================
# 8. 요약 카드
# ====================================================
def render_summary_card(selected_ticker: str, beta: float, avg_cycle: int,
                        win_prob, sell_prob, expected_return, z_score) -> None:

    if win_prob is not None and expected_return is not None:
        if win_prob >= 60.0 and expected_return >= 8.0:
            action, color = "🔥 적극 매수", "#1e8449"
        elif win_prob >= 55.0 and expected_return >= 4.0:
            action, color = "🟢 매수", "#2ecc71"
        elif win_prob >= 52.0 and expected_return >= 1.0:
            action, color = "🟡 분할 매수", "#f1c40f"
        elif win_prob <= 40.0 and expected_return <= -8.0:
            action, color = "🧊 적극 매도", "#78281f"
        elif win_prob <= 45.0 and expected_return <= -4.0:
            action, color = "🔴 매도", "#e74c3c"
        elif win_prob <= 48.0 and expected_return <= -1.0:
            action, color = "🟠 분할 매도", "#e67e22"
        else:
            action, color = "⚪ 관망", "#7f8c8d"
        wp_color = "#e74c3c" if win_prob >= 50.0 else "#2980b9"
        ev_color = "#e74c3c" if expected_return >= 0.0 else "#2980b9"
        wp_str = f"{win_prob:.1f}%"
        ev_str = f"{expected_return:+.1f}%"
    else:
        action, color = "데이터 부족", "gray"
        wp_color = ev_color = "#7f8c8d"
        wp_str = ev_str = "N/A"

    def item_html(label: str, value: str, val_color: str) -> str:
        return (
            f"<div style='text-align:center;padding:6px 12px;flex:1 1 0;'>"
            f"<div style='font-size:11px;color:#888;margin-bottom:3px;white-space:nowrap;'>{label}</div>"
            f"<div style='font-size:15px;font-weight:700;color:{val_color};white-space:nowrap;'>{value}</div>"
            f"</div>"
        )

    row_style = (
        "display:flex;flex-wrap:nowrap;justify-content:space-around;"
        "align-items:center;overflow-x:auto;-webkit-overflow-scrolling:touch;padding:4px 0;"
    )
    card_html = (
        "<div style='border:1px solid #e0e0e0;border-radius:10px;padding:6px 4px;'>"
        f"<div style='{row_style}'>"
        + item_html("투자 의견", action, color)
        + item_html("승률", wp_str, wp_color)
        + item_html("기대 수익", ev_str, ev_color)
        + "</div>"
        + "<hr style='margin:2px 8px;border:none;border-top:1px solid #f0f0f0;'>"
        + f"<div style='{row_style}'>"
        + item_html("주기", f"{avg_cycle}일", "#2c3e50")
        + item_html("베타", f"{beta:.2f}", "#2c3e50")
        + item_html("Z-Score", f"{z_score:+.2f}" if z_score is not None else "N/A",
                    "#e74c3c" if z_score is not None and z_score >= 1.5
                    else "#2980b9" if z_score is not None and z_score <= -1.5
                    else "#2c3e50")
        + "</div></div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ====================================================
# 9. 차트 렌더링
# ====================================================
def render_chart(df_daily: pd.DataFrame, selected_ticker: str,
                 beta: float, std_resid: float,
                 guide_n: int, view_months: int,
                 show_indicators: bool, avg_cycle: int) -> None:

    active_plots = ['main', 'spacer', 'price', 'win_prob', 'ev']
    row_heights = [0.35, 0.06, 0.2, 0.12, 0.12]
    if show_indicators:
        active_plots += ['macd', 'rsi']
        row_heights += [0.12, 0.12]

    total_rows = len(active_plots)
    fig = make_subplots(rows=total_rows, cols=1, row_heights=row_heights, vertical_spacing=0.02)
    current_row = 1

    # ── [1] Main scatter (로그-로그) ──
    sdf = df_daily.sort_values(f'{X_ASSET_FIXED}_Norm')
    x_vals = sdf[f'{X_ASSET_FIXED}_Norm']

    min_x = df_daily[f'{X_ASSET_FIXED}_Norm'].min()
    max_x = df_daily[f'{X_ASSET_FIXED}_Norm'].max()
    min_y = df_daily[f'{selected_ticker}_Norm'].min()
    max_y = df_daily[f'{selected_ticker}_Norm'].max()

    empirical_c = df_daily[f'{selected_ticker}_Norm'] / (df_daily[f'{X_ASSET_FIXED}_Norm'] ** guide_n)
    log_c_min = np.log10(empirical_c.min())
    log_c_max = np.log10(empirical_c.max())

    for log_c in np.linspace(log_c_min - 1.0, log_c_max + 1.0, 15):
        c_val = 10 ** log_c
        fig.add_trace(go.Scatter(
            x=x_vals, y=c_val * (x_vals ** guide_n),
            mode='lines', line=dict(color='rgba(200,200,200,0.6)', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ), row=current_row, col=1)

    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) - 1.5 * std_resid),
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=current_row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'],
        y=np.exp(np.log(sdf['Predicted']) + 1.5 * std_resid),
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(150,150,150,0.2)',
        showlegend=False, hoverinfo='skip'
    ), row=current_row, col=1)
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=sdf['Predicted'],
        mode='lines', line=dict(color='black', width=2), name='Predicted Trend'
    ), row=current_row, col=1)

    color_array = np.linspace(0, 1, len(df_daily))
    fig.add_trace(go.Scatter(
        x=df_daily[f'{X_ASSET_FIXED}_Norm'], y=df_daily[f'{selected_ticker}_Norm'],
        mode='markers', marker=dict(color=color_array, colorscale='Viridis', size=5, opacity=0.8),
        name='Daily Data'
    ), row=current_row, col=1)

    last_x = df_daily[f'{X_ASSET_FIXED}_Norm'].iloc[-1]
    last_y = df_daily[f'{selected_ticker}_Norm'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_x], y=[last_y], mode='markers',
        marker=dict(symbol='star', color='hotpink', size=16, line=dict(color='black', width=1)),
        name='Current Position'
    ), row=current_row, col=1)

    fig.update_xaxes(type="log", title_text="", showgrid=False,
                     range=[np.log10(min_x * 0.98), np.log10(max_x * 1.02)],
                     row=current_row, col=1)
    fig.update_yaxes(type="log", title_text=display_name(selected_ticker), showgrid=False,
                     range=[np.log10(min_y * 0.90), np.log10(max_y * 1.10)],
                     row=current_row, col=1)
    current_row += 1

    # ── [2] Spacer ──
    fig.update_xaxes(visible=False, row=current_row, col=1)
    fig.update_yaxes(visible=False, row=current_row, col=1)
    current_row += 1

    # ── 뷰 기간 기준 재정규화 ──
    view_start = df_daily.index[-1] - pd.DateOffset(months=view_months)
    view_df = df_daily[df_daily.index >= view_start]
    base_spy = view_df[f'{X_ASSET_FIXED}_Norm'].iloc[0] if not view_df.empty else 1.0
    base_tkr = view_df[f'{selected_ticker}_Norm'].iloc[0] if not view_df.empty else 1.0
    df_daily['Plot_Norm_SPY'] = df_daily[f'{X_ASSET_FIXED}_Norm'] / base_spy
    df_daily['Plot_Norm_Ticker'] = df_daily[f'{selected_ticker}_Norm'] / base_tkr

    # ── [3] Price ──
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['Plot_Norm_SPY'],
        mode='lines', line=dict(color='gray', width=1.5), name=X_ASSET_FIXED
    ), row=current_row, col=1)
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
        mode='lines', line=dict(color='black', width=1.5), name=selected_ticker
    ), row=current_row, col=1)

    min_price = df_daily.loc[df_daily.index >= view_start, ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].min().min()
    max_price = df_daily.loc[df_daily.index >= view_start, ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].max().max()
    price_baseline = min_price * 0.95

    add_filled_blocks(fig, df_daily, 'Plot_Norm_Ticker',
                      df_daily['Win_Prob'] >= 70, 'rgba(255,0,0,0.3)', current_row, 1, price_baseline)
    add_filled_blocks(fig, df_daily, 'Plot_Norm_Ticker',
                      df_daily['Win_Prob'] <= 30, 'rgba(0,0,255,0.3)', current_row, 1, price_baseline)

    fig.update_yaxes(type="log", title_text="Price",
                     range=[np.log10(price_baseline), np.log10(max_price * 1.05)],
                     row=current_row, col=1)
    time_x_axis = f'x{current_row}'
    current_row += 1

    # ── [4] Win Prob ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Win_Prob'],
                             line=dict(color='black', width=1.5), name='Win Prob'),
                  row=current_row, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=current_row, col=1)
    add_filled_blocks(fig, df_daily, 'Win_Prob',
                      df_daily['Win_Prob'] >= 50, 'rgba(255,0,0,0.3)', current_row, 1, 50)
    add_filled_blocks(fig, df_daily, 'Win_Prob',
                      df_daily['Win_Prob'] < 50, 'rgba(0,0,255,0.3)', current_row, 1, 50)
    fig.update_yaxes(range=[0, 100], title_text="Win prob (%)", row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ── [5] Expected Return ──
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Expected_Return'],
                             line=dict(color='black', width=1.5), name='EV'),
                  row=current_row, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)
    add_filled_blocks(fig, df_daily, 'Expected_Return',
                      df_daily['Expected_Return'] >= 0, 'rgba(255,0,0,0.3)', current_row, 1, 0)
    add_filled_blocks(fig, df_daily, 'Expected_Return',
                      df_daily['Expected_Return'] < 0, 'rgba(0,0,255,0.3)', current_row, 1, 0)
    fig.update_yaxes(title_text="EV (%)", row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ── [6][7] 보조 지표 ──
    if show_indicators:
        macd_colors = np.where(df_daily['MACD_Hist'] >= 0, 'rgba(0,128,0,0.5)', 'rgba(255,0,0,0.5)')
        fig.add_trace(go.Bar(x=df_daily.index, y=df_daily['MACD_Hist'],
                             marker_color=macd_colors, name='MACD Hist'),
                      row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD'],
                                 line=dict(color='blue', width=1), name='MACD'),
                      row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD_Signal'],
                                 line=dict(color='orange', width=1), name='Signal'),
                      row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
        current_row += 1

        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['RSI'],
                                 line=dict(color='purple', width=1.5), name='RSI'),
                      row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=current_row, col=1)
        fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)

    # ── 매매 기록 마커 ──
    trade_history = st.session_state.trade_history
    if selected_ticker in trade_history:
        for trade in trade_history[selected_ticker]:
            t_date = pd.to_datetime(trade['date'])
            t_type = trade['type']
            marker_color = 'red' if t_type == 'buy' else 'blue'
            marker_symbol = 'triangle-up' if t_type == 'buy' else 'triangle-down'

            if t_date in df_daily.index:
                d_date = t_date
            else:
                idx = df_daily.index.get_indexer([t_date], method='nearest')[0]
                d_date = df_daily.index[idx]

            fig.add_trace(go.Scatter(
                x=[df_daily.loc[d_date, f'{X_ASSET_FIXED}_Norm']],
                y=[df_daily.loc[d_date, f'{selected_ticker}_Norm']],
                mode='markers',
                marker=dict(symbol=marker_symbol, size=10, color=marker_color,
                            line=dict(width=1, color='black')),
                name=f"{t_type.upper()} ({t_date.date()})"
            ), row=1, col=1)

            for r in range(3, total_rows + 1):
                fig.add_vline(x=t_date, line_dash="dash",
                              line_color=marker_color, opacity=0.6, row=r, col=1)

    # ── 축 스타일 정리 ──
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)

    for r in range(3, total_rows):
        fig.update_xaxes(showticklabels=False, tickformat="%y/%m/%d", row=r, col=1)
    fig.update_xaxes(showticklabels=True, tickformat="%y/%m/%d", row=total_rows, col=1)

    fig.update_layout(
        height=650 if not show_indicators else 850,
        showlegend=False,
        hovermode='x unified',
        dragmode=False,
        margin=dict(l=10, r=10, t=10, b=30)
    )
    fig.update_xaxes(range=[view_start, df_daily.index[-1]], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True,
                     config={'scrollZoom': False, 'displayModeBar': False})


# ====================================================
# 10. 메인 진입점
# ====================================================
def main():
    # session_state 초기화 (최초 1회)
    init_session_state()

    # ── 제목 + 종목 선택 (메인 페이지 상단) ──
    st.markdown("## 📊 퀀트 대시보드")

    RADIO_OPTIONS = TARGET_TICKERS + ["✏️ 직접 입력"]
    radio_choice = st.radio(
        "분석 종목", RADIO_OPTIONS, horizontal=True, label_visibility="collapsed",
        format_func=lambda t: (
            t if t == "✏️ 직접 입력"
            else f"{display_name(t)}{st.session_state.ticker_signals.get(t, '')}"
        )
    )

    if radio_choice == "✏️ 직접 입력":
        custom_input = st.text_input(
            "티커 직접 입력 (예: NVDA, 000660, BTC-USD)",
            placeholder="티커 입력 후 Enter",
            label_visibility="collapsed"
        )
        selected_ticker = custom_input.strip().upper() if custom_input.strip() else None
    else:
        selected_ticker = radio_choice

    if not selected_ticker:
        st.info("분석할 티커를 입력해 주세요.")
        return

    st.markdown("---")

    # ── 데이터 fetch ──
    default_start = "2021-01-01"
    with st.spinner("시장 데이터를 불러오는 중..."):
        df_close = fetch_all_data(TARGET_TICKERS, default_start)

        # 커스텀 티커는 별도 fetch 후 병합
        if f'{selected_ticker}_Close' not in df_close.columns:
            df_custom = fetch_single_ticker(selected_ticker, default_start)
            if df_custom.empty:
                st.error(f"'{selected_ticker}' 데이터를 가져올 수 없습니다. 티커를 확인해 주세요.")
                return
            df_close = pd.concat([df_close, df_custom], axis=1).ffill()

    # 사이드바 렌더링 (selected_ticker 전달)
    cfg = render_sidebar(df_close, selected_ticker)
    train_end_date = pd.to_datetime(cfg['train_end']) + pd.offsets.MonthEnd(0)

    # 분석 실행
    with st.spinner("데이터를 분석 중입니다..."):
        df_x = df_close[[f'{X_ASSET_FIXED}_Close']]
        df_y = df_close[[f'{selected_ticker}_Close']]
        res = process_asset_data(
            df_x, df_y, X_ASSET_FIXED, selected_ticker,
            cfg['selected_cycle'], train_end_date
        )
    df_processed, df_daily, beta, win_prob, sell_prob, expected_return, avg_cycle, std_resid = res

    if df_daily is None:
        st.error("분석에 필요한 데이터가 부족합니다.")
        return

    # 요약 카드
    z_score_now = float(df_daily['Z_Score'].iloc[-1]) if 'Z_Score' in df_daily.columns and not df_daily['Z_Score'].isna().all() else None
    # 분석 결과를 라디오 버튼 이모지용으로 저장
    st.session_state.ticker_signals[selected_ticker] = get_action_emoji(win_prob, expected_return)
    render_summary_card(selected_ticker, beta, avg_cycle, win_prob, sell_prob, expected_return, z_score_now)

    # 차트
    render_chart(
        df_daily, selected_ticker, beta, std_resid,
        cfg['guide_n'], cfg['view_months'],
        cfg['show_indicators'], avg_cycle
    )


if __name__ == "__main__":
    main()

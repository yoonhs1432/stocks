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
# 1. 전역 설정 및 매매 기록 (JSON 영구 저장 로직)
# ====================================================
st.set_page_config(page_title="퀀트 트레이딩 대시보드", layout="wide")

X_ASSET_FIXED = 'SPY'
TARGET_TICKERS = ['SPYU', 'TQQQ', 'SOXL', 'FNGU', 'BTC-USD', 'BITU', 'ETH-USD', 'ETHT', 'HIBL', 'TARK', 'QPUX', 'BNKU', 'GDXU', 'KORU', '005930', 'UPXI']

TRADE_FILE = 'trade_history.json'

def load_trade_history():
    if os.path.exists(TRADE_FILE):
        with open(TRADE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        history = {
            'BITU': [{'date': '2026-04-01', 'type': 'buy'}, {'date': '2026-04-13', 'type': 'buy'}],
            'TQQQ': [{'date': '2026-04-01', 'type': 'buy'}]
        }
        with open(TRADE_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        return history

def save_trade_history(history):
    with open(TRADE_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

TRADE_HISTORY = load_trade_history()

# ====================================================
# 2. 데이터 다운로드 및 주기 스윕 함수
# ====================================================
@st.cache_data(show_spinner=False)
def fetch_all_data(tickers, start_date_str):
    df_list = []
    all_tickers = [X_ASSET_FIXED] + tickers
    for ticker in all_tickers:
        try:
            data = fdr.DataReader(ticker, start_date_str)
            if not data.empty:
                data = data[~data.index.duplicated(keep='last')]
                temp_df = data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
                df_list.append(temp_df)
        except Exception:
            continue
    if df_list:
        df_close = pd.concat(df_list, axis=1).ffill()
        return df_close
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def find_available_cycles(price_series, min_w=20, max_w=365, step=5):
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
def get_optimal_cycle(price_series, cycles_list):
    if not cycles_list: return 60
    detrended = price_series - price_series.rolling(window=100, min_periods=1).mean()
    best_cycle = cycles_list[len(cycles_list)//2]
    best_score = -np.inf
    for c in cycles_list:
        if len(detrended.dropna()) > c * 2:
            score = detrended.autocorr(lag=c)
            if pd.notna(score) and score > best_score:
                best_score = score
                best_cycle = c
    return best_cycle

# ====================================================
# 3. 데이터 처리 함수
# ====================================================
def process_asset_data(df_x, df_y, x_name, y_name, selected_cycle, train_end_date):
    df = pd.merge(df_x, df_y, left_index=True, right_index=True).dropna().sort_index()
    if df.empty: return None, None, None, None, None, None, None, None

    col_x, col_y = f'{x_name}_Norm', f'{y_name}_Norm'
    base_x, base_y = df[f'{x_name}_Close'].iloc[0], df[f'{y_name}_Close'].iloc[0]

    df[col_x] = df[f'{x_name}_Close'] / base_x
    df[col_y] = df[f'{y_name}_Close'] / base_y

    log_x, log_y = np.log(df[col_x]), np.log(df[col_y])

    model = LinearRegression().fit(log_x.values.reshape(-1, 1), log_y.values)
    beta = model.coef_[0]
    df['Predicted'] = np.exp(model.intercept_) * df[col_x] ** beta

    df_daily = df.copy() 

    delta_d = df_daily[f'{y_name}_Close'].diff()
    gain_d = delta_d.where(delta_d > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss_d = (-delta_d.where(delta_d < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df_daily['RSI'] = 100 - (100 / (1 + (gain_d / loss_d)))

    ema12_d = df_daily[f'{y_name}_Close'].ewm(span=12, adjust=False).mean()
    ema26_d = df_daily[f'{y_name}_Close'].ewm(span=26, adjust=False).mean()
    df_daily['MACD'] = ema12_d - ema26_d
    df_daily['MACD_Signal'] = df_daily['MACD'].ewm(span=9, adjust=False).mean()
    df_daily['MACD_Hist'] = df_daily['MACD'] - df_daily['MACD_Signal']

    log_resid_hist = np.log(df_daily[col_y]) - np.log(df_daily['Predicted'])
    std_resid = log_resid_hist.std()
    df_daily['Z_Score'] = log_resid_hist / std_resid

    shifted_price = df_daily[col_y].shift(-1)
    df_daily['Future_Avg_Price'] = shifted_price.iloc[::-1].rolling(window=selected_cycle).mean().iloc[::-1]
    
    df_daily['Target'] = np.where(df_daily['Future_Avg_Price'].isna(), np.nan, (df_daily['Future_Avg_Price'] > df_daily[col_y]).astype(int))
    df_daily['Target_Return'] = np.where(df_daily['Future_Avg_Price'].isna(), np.nan, (df_daily['Future_Avg_Price'] / df_daily[col_y] - 1) * 100)

    for col in ['Win_Prob', 'Sell_Prob', 'Expected_Return']:
        df_daily[col] = np.nan
        
    ml_df = df_daily[['Z_Score', 'RSI', 'MACD_Hist', 'Target', 'Target_Return']].dropna()
    win_prob, sell_prob, expected_return = None, None, None

    ml_train_df = ml_df.loc[:train_end_date]

    if len(ml_train_df) > 30:
        X_train, y_class_train, y_reg_train = ml_train_df[['Z_Score', 'RSI', 'MACD_Hist']], ml_train_df['Target'], ml_train_df['Target_Return'] 
        clf = LogisticRegression(class_weight='balanced').fit(X_train, y_class_train)
        reg = Ridge(alpha=1.0).fit(X_train, y_reg_train)
        
        valid_idx = df_daily.dropna(subset=['Z_Score', 'RSI', 'MACD_Hist']).index
        features_valid = df_daily.loc[valid_idx, ['Z_Score', 'RSI', 'MACD_Hist']]
        
        if not features_valid.empty:
            probs = clf.predict_proba(features_valid)
            df_daily.loc[valid_idx, 'Win_Prob'], df_daily.loc[valid_idx, 'Sell_Prob'] = probs[:, 1] * 100, probs[:, 0] * 100 
            win_prob, sell_prob = probs[-1, 1] * 100, probs[-1, 0] * 100 
            
            evs = reg.predict(features_valid)
            df_daily.loc[valid_idx, 'Expected_Return'] = evs
            expected_return = evs[-1]

    return df, df_daily, beta, win_prob, sell_prob, expected_return, selected_cycle, std_resid

# ====================================================
# 💡 플롯 면적 채우기 전용 도우미 함수 
# ====================================================
def add_filled_blocks(fig, df, y_col, condition, color, row, col, baseline_y):
    valid_data = df[condition]
    if valid_data.empty: return
    groups = (~condition).cumsum()[condition]
    
    for _, group in valid_data.groupby(groups):
        if len(group) < 1: continue
        fig.add_trace(go.Scatter(
            x=group.index, y=[baseline_y]*len(group), 
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
# 4. Streamlit 메인 앱 로직
# ====================================================
def main():
    with st.sidebar:
        # 💡 [Req] 실시간 데이터 새로고침 -> Refresh
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
        st.markdown("### 📊 종목 및 뷰어 옵션")
        selected_ticker = st.selectbox("🎯 분석할 종목", TARGET_TICKERS)
        
        st.markdown("### ⚙️ 분석 파라미터 조절")
        train_start = st.text_input("학습 시작 (YYYY-MM)", "2021-01")
        train_end = st.text_input("학습 종료 (YYYY-MM)", "2026-03")
        view_months = st.number_input("차트 조회 기간 (최근 N개월)", min_value=1, max_value=240, value=12, step=1)
        guide_n = st.number_input("가이드라인 기울기 (n)", min_value=1, max_value=20, value=4, step=1)
        
        # 💡 [Req] 보조지표 표시 체크박스를 파라미터 조절 항목 안으로 이동
        show_indicators = st.checkbox("MACD&RSI 표시", value=False)
        
        fetch_start_date = f"{train_start}-01"
        df_close = fetch_all_data(TARGET_TICKERS, fetch_start_date)
        
        if not df_close.empty and f'{selected_ticker}_Close' in df_close.columns:
            cycles_list = find_available_cycles(df_close[f'{selected_ticker}_Close'], min_w=20, max_w=365)
            optimal_cycle = get_optimal_cycle(df_close[f'{selected_ticker}_Close'], cycles_list)
            default_index = cycles_list.index(optimal_cycle) if optimal_cycle in cycles_list else 0
            selected_cycle = st.selectbox(f"🔄 분석 주기 선택 (추천: {optimal_cycle}일)", cycles_list, index=default_index)
        else:
            selected_cycle = 60

        st.markdown("---")
        st.markdown("### 📝 매매 기록 관리")
        
        # 💡 [Req] 새로운 기록 추가 -> Add
        with st.expander("➕ Add"):
            t_ticker = st.selectbox("종목", TARGET_TICKERS, index=TARGET_TICKERS.index(selected_ticker))
            t_date = st.date_input("날짜", datetime.date.today())
            t_type = st.radio("종류", ['buy', 'sell'], horizontal=True)
            if st.button("기록 저장"):
                if t_ticker not in TRADE_HISTORY: TRADE_HISTORY[t_ticker] = []
                TRADE_HISTORY[t_ticker].append({'date': t_date.strftime("%Y-%m-%d"), 'type': t_type})
                save_trade_history(TRADE_HISTORY)
                st.success("저장 완료!")
                st.rerun()

        # 💡 [Req] 기존 기록 삭제 -> Delete
        with st.expander("🗑️ Delete"):
            if selected_ticker in TRADE_HISTORY and TRADE_HISTORY[selected_ticker]:
                for i, record in enumerate(TRADE_HISTORY[selected_ticker]):
                    cols = st.columns([3, 2, 1])
                    cols[0].write(f"{record['date']}")
                    cols[1].write(f"{record['type'].upper()}")
                    if cols[2].button("🗑️", key=f"del_{selected_ticker}_{i}"):
                        TRADE_HISTORY[selected_ticker].pop(i)
                        save_trade_history(TRADE_HISTORY)
                        st.rerun()
            else:
                st.info("매매 기록이 없습니다.")

    train_end_date = pd.to_datetime(train_end) + pd.offsets.MonthEnd(0)

    with st.spinner('데이터를 분석 중입니다...'):
        if df_close.empty or f'{selected_ticker}_Close' not in df_close.columns:
            st.error("데이터를 가져오는 데 실패했습니다.")
            return

        df_x = df_close[[f'{X_ASSET_FIXED}_Close']]
        df_y = df_close[[f'{selected_ticker}_Close']]
        
        res = process_asset_data(df_x, df_y, X_ASSET_FIXED, selected_ticker, selected_cycle, train_end_date)
        df_processed, df_daily, beta, win_prob, sell_prob, expected_return, avg_cycle, std_resid = res

    # 💡 [Req] 타이틀 변경: {티커} 주가 분석 프로그램
    st.markdown(f"#### 📈 {selected_ticker} 주가 분석 프로그램")
    
    if win_prob is not None and expected_return is not None:
        if win_prob >= 60.0 and expected_return >= 8.0: action, color = "🔥 적극 매수", "#1e8449"
        elif win_prob >= 55.0 and expected_return >= 4.0: action, color = "🟢 매수", "#2ecc71"
        elif win_prob >= 52.0 and expected_return >= 1.0: action, color = "🟡 분할 매수", "#f1c40f"
        elif win_prob <= 40.0 and expected_return <= -8.0: action, color = "🧊 적극 매도", "#78281f"
        elif win_prob <= 45.0 and expected_return <= -4.0: action, color = "🔴 매도", "#e74c3c"
        elif win_prob <= 48.0 and expected_return <= -1.0: action, color = "🟠 분할 매도", "#e67e22"
        else: action, color = "⚪ 관망", "#7f8c8d"
            
        wp_color = "#e74c3c" if win_prob >= 50.0 else "#2980b9"
        ev_color = "#e74c3c" if expected_return >= 0.0 else "#2980b9"
    else:
        action, color, win_prob, expected_return = "데이터 부족", "gray", 0, 0
        wp_color, ev_color = "black", "black"

    cols = st.columns(6, gap="small") 
    def small_metric(label, value, val_color="black"):
        return f"<div style='text-align: center; padding: 2px; white-space: nowrap;'><div style='font-size: 13px; color: gray; margin-bottom: 2px;'>{label}</div><div style='font-size: 16px; font-weight: bold; color: {val_color};'>{value}</div></div>"

    cols[0].markdown(small_metric("티커", selected_ticker, "#2c3e50"), unsafe_allow_html=True)
    cols[1].markdown(small_metric("투자 의견", action, color), unsafe_allow_html=True)
    cols[2].markdown(small_metric("승률", f"{win_prob:.1f}%" if win_prob else "N/A", wp_color), unsafe_allow_html=True)
    cols[3].markdown(small_metric("기대 수익", f"{expected_return:+.1f}%" if expected_return else "N/A", ev_color), unsafe_allow_html=True)
    cols[4].markdown(small_metric("주기", f"{avg_cycle} 일"), unsafe_allow_html=True)
    cols[5].markdown(small_metric("베타", f"{beta:.2f}"), unsafe_allow_html=True)

    # ====================================================
    # 5. 동적 그래프 그리기 
    # ====================================================
    active_plots = ['main', 'spacer', 'price', 'win_prob', 'ev']
    row_heights = [0.35, 0.06, 0.2, 0.12, 0.12]

    if show_indicators:
        active_plots.extend(['macd', 'rsi'])
        row_heights.extend([0.12, 0.12])

    # 💡 [Req] 2,3,4 그래프 침범 방지를 위해 vertical_spacing을 0.02 -> 0.04로 조정
    fig = make_subplots(rows=len(active_plots), cols=1, row_heights=row_heights, vertical_spacing=0.04)
    current_row = 1
    total_rows = len(active_plots)

    # ----------------------------------------
    # [1] Main Plot 
    # ----------------------------------------
    sdf = df_daily.sort_values(f'{X_ASSET_FIXED}_Norm')
    x_vals = sdf[f'{X_ASSET_FIXED}_Norm']
    
    min_x_val_main = df_daily[f'{X_ASSET_FIXED}_Norm'].min()
    max_x_val_main = df_daily[f'{X_ASSET_FIXED}_Norm'].max()
    min_y_val_main = df_daily[f'{selected_ticker}_Norm'].min()
    max_y_val_main = df_daily[f'{selected_ticker}_Norm'].max()

    x_range = [np.log10(min_x_val_main * 0.98), np.log10(max_x_val_main * 1.02)]
    y_range = [np.log10(min_y_val_main * 0.90), np.log10(max_y_val_main * 1.10)]

    y_limit_bottom = 10 ** (y_range[0] - 0.2)
    y_limit_top = 10 ** (y_range[1] + 0.2)

    empirical_c = df_daily[f'{selected_ticker}_Norm'] / (df_daily[f'{X_ASSET_FIXED}_Norm'] ** guide_n)
    log_c_min, log_c_max = np.log10(empirical_c.min()), np.log10(empirical_c.max())
    
    for log_c in np.linspace(log_c_min - 1.0, log_c_max + 1.0, 15): 
        c_val = 10 ** log_c
        y_guide = c_val * (x_vals ** guide_n)
        y_guide_clipped = np.where((y_guide > y_limit_bottom) & (y_guide < y_limit_top), y_guide, np.nan)
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_guide_clipped, mode='lines', 
            line=dict(color='rgba(200, 200, 200, 0.6)', width=1, dash='dot'), 
            showlegend=False, hoverinfo='skip'), row=current_row, col=1)

    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=np.exp(np.log(sdf['Predicted']) - 1.5 * std_resid),
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=current_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=np.exp(np.log(sdf['Predicted']) + 1.5 * std_resid),
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(150, 150, 150, 0.2)', 
        showlegend=False, hoverinfo='skip'), row=current_row, col=1)

    fig.add_trace(go.Scatter(
        x=sdf[f'{X_ASSET_FIXED}_Norm'], y=sdf['Predicted'],
        mode='lines', line=dict(color='black', width=2), name='Predicted Trend'), row=current_row, col=1)

    color_array = np.linspace(0, 1, len(df_daily))
    fig.add_trace(go.Scatter(
        x=df_daily[f'{X_ASSET_FIXED}_Norm'], y=df_daily[f'{selected_ticker}_Norm'],
        mode='markers', marker=dict(color=color_array, colorscale='Viridis', size=5, opacity=0.8), 
        name='Daily Data'), row=current_row, col=1)

    last_x = df_daily[f'{X_ASSET_FIXED}_Norm'].iloc[-1]
    last_y = df_daily[f'{selected_ticker}_Norm'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_x], y=[last_y], mode='markers',
        marker=dict(symbol='star', color='hotpink', size=16, line=dict(color='black', width=1)),
        name='Current Position'), row=current_row, col=1)
    
    fig.update_xaxes(type="log", title_text="", showgrid=False, range=x_range, row=current_row, col=1)
    fig.update_yaxes(type="log", title_text=f"{selected_ticker}", showgrid=False, range=y_range, row=current_row, col=1)
    current_row += 1

    # ----------------------------------------
    # [2] Spacer Row
    # ----------------------------------------
    fig.update_xaxes(visible=False, row=current_row, col=1)
    fig.update_yaxes(visible=False, row=current_row, col=1)
    current_row += 1

    # ----------------------------------------
    # [3] Price (View-based Re-normalization)
    # ----------------------------------------
    view_start_date = df_daily.index[-1] - pd.DateOffset(months=view_months)
    view_df = df_daily[df_daily.index >= view_start_date]
    
    if not view_df.empty:
        base_spy_view = view_df[f'{X_ASSET_FIXED}_Norm'].iloc[0]
        base_ticker_view = view_df[f'{selected_ticker}_Norm'].iloc[0]
    else:
        base_spy_view, base_ticker_view = 1.0, 1.0
        
    df_daily['Plot_Norm_SPY'] = df_daily[f'{X_ASSET_FIXED}_Norm'] / base_spy_view
    df_daily['Plot_Norm_Ticker'] = df_daily[f'{selected_ticker}_Norm'] / base_ticker_view

    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['Plot_Norm_SPY'],
        mode='lines', line=dict(color='gray', width=1.5), name=f'{X_ASSET_FIXED}'), row=current_row, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_daily.index, y=df_daily['Plot_Norm_Ticker'],
        mode='lines', line=dict(color='black', width=1.5), name=f'{selected_ticker}'), row=current_row, col=1)
    
    min_price_val = df_daily.loc[df_daily.index >= view_start_date, ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].min().min()
    max_price_val = df_daily.loc[df_daily.index >= view_start_date, ['Plot_Norm_SPY', 'Plot_Norm_Ticker']].max().max()
    price_baseline = min_price_val * 0.95 

    wp_high = df_daily['Win_Prob'] >= 70
    wp_low = df_daily['Win_Prob'] <= 30
    
    add_filled_blocks(fig, df_daily, 'Plot_Norm_Ticker', wp_high, 'rgba(255,0,0,0.3)', current_row, 1, price_baseline)
    add_filled_blocks(fig, df_daily, 'Plot_Norm_Ticker', wp_low, 'rgba(0,0,255,0.3)', current_row, 1, price_baseline)

    fig.update_yaxes(type="log", title_text="Price", range=[np.log10(price_baseline), np.log10(max_price_val * 1.05)], row=current_row, col=1)
    time_x_axis = f'x{current_row}' 
    current_row += 1

    # ----------------------------------------
    # [4] Win Probability 
    # ----------------------------------------
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Win_Prob'], line=dict(color='black', width=1.5), name='Win Prob'), row=current_row, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=current_row, col=1)
    
    wp_above_50 = df_daily['Win_Prob'] >= 50
    wp_below_50 = df_daily['Win_Prob'] < 50
    add_filled_blocks(fig, df_daily, 'Win_Prob', wp_above_50, 'rgba(255,0,0,0.3)', current_row, 1, 50)
    add_filled_blocks(fig, df_daily, 'Win_Prob', wp_below_50, 'rgba(0,0,255,0.3)', current_row, 1, 50)

    fig.update_yaxes(range=[0, 100], title_text="Win prob (%)", row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ----------------------------------------
    # [5] Expected Return 
    # ----------------------------------------
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['Expected_Return'], line=dict(color='black', width=1.5), name='EV'), row=current_row, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)
    
    ev_above_0 = df_daily['Expected_Return'] >= 0
    ev_below_0 = df_daily['Expected_Return'] < 0
    add_filled_blocks(fig, df_daily, 'Expected_Return', ev_above_0, 'rgba(255,0,0,0.3)', current_row, 1, 0)
    add_filled_blocks(fig, df_daily, 'Expected_Return', ev_below_0, 'rgba(0,0,255,0.3)', current_row, 1, 0)

    fig.update_yaxes(title_text="EV (%)", row=current_row, col=1)
    fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
    current_row += 1

    # ----------------------------------------
    # [6] & [7] 보조지표 
    # ----------------------------------------
    if show_indicators:
        macd_colors = np.where(df_daily['MACD_Hist'] >= 0, 'rgba(0,128,0,0.5)', 'rgba(255,0,0,0.5)')
        fig.add_trace(go.Bar(x=df_daily.index, y=df_daily['MACD_Hist'], marker_color=macd_colors, name='MACD Hist'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD'], line=dict(color='blue', width=1), name='MACD'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)
        current_row += 1

        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=current_row, col=1)
        fig.update_xaxes(matches=time_x_axis, row=current_row, col=1)

    # 매매 기록 렌더링
    if selected_ticker in TRADE_HISTORY:
        for trade in TRADE_HISTORY[selected_ticker]:
            t_date = pd.to_datetime(trade['date'])
            t_type = trade['type']
            marker_color = 'red' if t_type == 'buy' else 'blue'
            marker_symbol = 'triangle-up' if t_type == 'buy' else 'triangle-down'
            if t_date in df_daily.index: d_date = t_date
            else:
                idx = df_daily.index.get_indexer([t_date], method='nearest')[0]
                d_date = df_daily.index[idx]

            x_val = df_daily.loc[d_date, f'{X_ASSET_FIXED}_Norm']
            y_val_main = df_daily.loc[d_date, f'{selected_ticker}_Norm']
            
            fig.add_trace(go.Scatter(
                x=[x_val], y=[y_val_main], mode='markers',
                marker=dict(symbol=marker_symbol, size=10, color=marker_color, line=dict(width=1, color='black')),
                name=f"{t_type.upper()} ({t_date.date()})"
            ), row=1, col=1)
            
            for r in range(3, total_rows + 1):
                fig.add_vline(x=t_date, line_dash="dash", line_color=marker_color, opacity=0.6, row=r, col=1)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)

    for r in range(3, total_rows):
        fig.update_xaxes(showticklabels=False, tickformat="%y/%m/%d", row=r, col=1)
    
    fig.update_xaxes(showticklabels=True, tickformat="%y/%m/%d", row=total_rows, col=1)

    total_chart_height = 650 if not show_indicators else 850
    fig.update_layout(height=total_chart_height, showlegend=False, hovermode='x unified', margin=dict(l=10, r=10, t=10, b=30))

    fig.update_xaxes(range=[view_start_date, df_daily.index[-1]], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
# 美股趨勢掃描
# US Stock Trend Scanner

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import warnings
import talib
from io import BytesIO

warnings.filterwarnings('ignore')

def calculate_sector_trend(tickers, start_date, end_date, sector_name):
    """計算行業趨勢"""
    data = []
    valid_tickers = []
    failed_tickers = []

    st.write(f"分析 {sector_name} 行業股票：{', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")

    progress_placeholder = st.empty()
    for i, ticker in enumerate(tickers):
        try:
            progress_placeholder.progress((i + 1) / len(tickers), f"正在分析 {ticker} ({i+1}/{len(tickers)})")

            # 下載數據
            df_ticker = yf.download(ticker, start=start_date, end=end_date)

            if df_ticker.empty:
                failed_tickers.append(ticker)
                continue

            # 計算20日SMA
            ma20 = talib.SMA(df_ticker['Close'].to_numpy().reshape(-1), timeperiod=20)
            res = np.where(df_ticker['Close'].to_numpy().reshape(-1) > ma20, 1, 0)
            data.append(res)
            valid_tickers.append(ticker)

        except Exception as e:
            failed_tickers.append(ticker)
            continue

    progress_placeholder.empty()

    if not data:
        return pd.Series(dtype='float64'), failed_tickers

    # 數據對齊和計算
    max_len = max(len(arr) for arr in data)
    for i, arr in enumerate(data):
        if len(arr) < max_len:
            data[i] = np.pad(arr, (max_len - len(arr), 0), 'constant', constant_values=0)

    df_temp = pd.DataFrame()
    for i, ticker in enumerate(valid_tickers):
        df_temp[ticker] = data[i]

    row_sums = round(df_temp.sum(axis=1) / len(valid_tickers) * 100)
    return row_sums, failed_tickers

def main():
    """美股趨勢掃描主程式"""
    st.title("🔍 美股趨勢掃描")
    st.markdown("---")

    st.markdown("""
    ### 📋 功能說明
    此工具分析標普500各行業股票相對於20日移動平均線的趨勢強度：
    - 分析11個主要行業板塊的趨勢變化
    - 計算每個行業中股票高於20日均線的百分比
    - 提供歷史趨勢圖表和Excel報告下載
    """)

    # 硬編碼各行業代表股票
    sector_stocks = {
        'XLC': ['GOOGL', 'META', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'CRM'],  # 通訊
        'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW'],  # 非必需消費品
        'XLP': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS'],      # 必需消費品
        'XLE': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO'],    # 能源
        'XLF': ['BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP'],      # 金融
        'XLV': ['UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'MRK'],   # 醫療保健
        'XLI': ['MMM', 'CAT', 'BA', 'GE', 'UPS', 'RTX', 'HON', 'UNP'],      # 工業
        'XLB': ['LIN', 'APD', 'FCX', 'NUE', 'SHW', 'NEM', 'DOW', 'DD'],     # 原材料
        'XLRE': ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'EXR'],   # 地產
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'INTC', 'AMD'],  # 科技
        'XLU': ['NEE', 'SO', 'DUK', 'AEP', 'SRE', 'D', 'PCG', 'EXC']        # 公用事業
    }

    # 對應中文名稱
    sector_names = {
        'XLC': '通訊', 'XLY': '選消', 'XLP': '必消', 'XLE': '能源', 'XLF': '金融',
        'XLV': '健康', 'XLI': '工業', 'XLB': '原材', 'XLRE': '地產', 'XLK': '科技', 'XLU': '公用'
    }

    # 參數設定
    col1, col2 = st.columns(2)
    with col1:
        analysis_days = st.selectbox("📅 分析期間", [200, 300, 400, 500], index=2, key="us_trend_days_select")
    with col2:
        show_chart = st.checkbox("📊 顯示趨勢圖表", value=True, key="us_trend_chart_check")

    if st.button("🚀 開始分析", width='stretch', key="us_trend_analysis_btn"):
        end_date = date.today()
        start_date = end_date - timedelta(days=analysis_days)

        st.markdown("### 📈 行業趨勢分析")

        results = {}
        all_failed_tickers = []

        # 分析各行業
        for sector_code, chinese_name in sector_names.items():
            with st.expander(f"正在分析 {chinese_name} ({sector_code})", expanded=True):
                tickers = sector_stocks[sector_code]

                trend_data, failed = calculate_sector_trend(
                    tickers, start_date, end_date, chinese_name
                )
                results[chinese_name] = trend_data
                all_failed_tickers.extend(failed)

                if not trend_data.empty and len(trend_data) > 0:
                    latest_value = trend_data.iloc[-1]
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(f"{chinese_name} 最新趨勢", f"{latest_value}%")
                    with col_b:
                        if latest_value >= 70:
                            st.success("💚 強勢")
                        elif latest_value >= 50:
                            st.info("💙 中性")
                        else:
                            st.error("❤️ 弱勢")
                else:
                    st.error(f"❌ {chinese_name} 無法取得足夠資料")

        # 建立結果DataFrame
        if any(not data.empty for data in results.values()):
            st.markdown("### 📊 綜合分析結果")

            # 數據整理
            min_length = min([len(data) for data in results.values() if not data.empty])
            if min_length > 0:
                df_results = pd.DataFrame()
                for sector_name, data in results.items():
                    if not data.empty and len(data) >= min_length:
                        df_results[sector_name] = data.tail(min_length).values

                if not df_results.empty:
                    # 添加日期索引
                    try:
                        spy_data = yf.download('SPY', start=start_date, end=end_date)
                        if not spy_data.empty and len(spy_data) >= len(df_results):
                            df_results.index = spy_data.tail(len(df_results)).index
                    except:
                        pass

                    # 最新在上
                    df_results = df_results.iloc[::-1]

                    # 顯示資料表
                    st.dataframe(df_results.head(20), width='stretch')

                    # 行業強度總覽
                    st.markdown("### 🎯 各行業最新強度")
                    cols = st.columns(min(4, len(df_results.columns)))
                    for i, col_name in enumerate(df_results.columns):
                        with cols[i % 4]:
                            latest_val = df_results[col_name].iloc[0]
                            if latest_val >= 70:
                                st.success(f"**{col_name}**\n{latest_val}% 💚")
                            elif latest_val >= 50:
                                st.info(f"**{col_name}**\n{latest_val}% 💙")
                            else:
                                st.error(f"**{col_name}**\n{latest_val}% ❤️")

                    # Excel下載
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df_results.to_excel(writer, sheet_name='行業趨勢分析')

                        # 添加條件格式
                        workbook = writer.book
                        worksheet = writer.sheets['行業趨勢分析']

                        # 條件格式：3色階
                        n_rows, n_cols = len(df_results), len(df_results.columns)
                        cell_range = f'B2:{chr(66 + n_cols - 1)}{n_rows + 1}'

                        worksheet.conditional_format(cell_range, {
                            'type': '3_color_scale',
                            'min_color': '#FF0000',  # 紅色
                            'mid_color': '#FFFFFF',  # 白色
                            'max_color': '#00FF00'   # 綠色
                        })

                    st.download_button(
                        label="📥 下載趨勢分析報告",
                        data=output.getvalue(),
                        file_name=f"美股行業趨勢分析_{date.today().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )

                    # 趨勢圖表
                    if show_chart:
                        st.markdown("### 📈 歷史趨勢圖")
                        chart_data = df_results.iloc[::-1]  # 恢復時間順序
                        st.line_chart(chart_data)

        # 失敗股票報告
        if all_failed_tickers:
            with st.expander("⚠️ 下載失敗的股票", expanded=False):
                failed_unique = list(set(all_failed_tickers))
                st.write(f"共有 {len(failed_unique)} 支股票無法下載資料：")
                for ticker in failed_unique:
                    st.write(f"- {ticker}")

if __name__ == "__main__":
    main()
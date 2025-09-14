# 美股大盤掃描
# US Market Scanner

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import warnings
import talib
from io import BytesIO

warnings.filterwarnings('ignore')

def calculate_sma_trend(tickers, start_date, end_date):
    """計算股票相對於20日均線的趨勢百分比"""
    data = []
    valid_tickers = []
    failed_tickers = []

    with st.progress(0) as progress_bar:
        for i, ticker in enumerate(tickers):
            try:
                progress_bar.progress((i + 1) / len(tickers))

                # 下載數據
                df = yf.download(ticker, start=start_date, end=end_date)

                if df.empty:
                    st.warning(f"⚠️ {ticker} 沒有資料，跳過...")
                    failed_tickers.append(ticker)
                    continue

                # 計算20天的SMA
                ma20 = talib.SMA(df['Close'].to_numpy().reshape(-1), timeperiod=20)
                res = np.where(df['Close'].to_numpy().reshape(-1) > ma20, 1, 0)
                data.append(res)
                valid_tickers.append(ticker)

            except Exception as e:
                st.warning(f"⚠️ {ticker} 下載失敗: {e}")
                failed_tickers.append(ticker)
                continue

    if not data:
        return pd.Series(dtype='float64'), failed_tickers

    # 找到數據中最大的長度並補齊
    max_len = max([len(col) for col in data])
    for i, col in enumerate(data):
        if len(col) < max_len:
            data[i] = np.pad(col, (max_len - len(col), 0), 'constant', constant_values=0)

    # 創建DataFrame並計算百分比
    df_temp = pd.DataFrame()
    for i, col_name in enumerate(valid_tickers):
        df_temp[col_name] = data[i]

    row_sums = round(df_temp.sum(axis=1) / len(valid_tickers) * 100)
    return row_sums, failed_tickers

def main():
    """美股大盤掃描主程式"""
    st.title("📊 美股大盤掃描")
    st.markdown("---")

    st.markdown("""
    ### 📋 功能說明
    此工具分析主要美股指數成分股相對於20日移動平均線的趨勢強度：
    - **SMH**: 費城半導體指數
    - **QQQ**: 納斯達克100指數
    - **DIA**: 道瓊工業指數
    - **SPY**: 標普500指數
    """)

    # 硬編碼主要指數成分股 (示例股票)
    index_stocks = {
        'SMH': ['NVDA', 'TSM', 'AVGO', 'AMD', 'INTC', 'MU', 'QCOM', 'TXN', 'ADI', 'MRVL'],
        'QQQ': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL'],
        'DIA': ['UNH', 'GS', 'HD', 'MCD', 'CAT', 'AMGN', 'V', 'BA', 'TRV', 'AXP'],
        'SPY': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'V', 'UNH', 'JNJ']
    }

    # 參數設定
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("📅 分析天數", [30, 60, 90, 120], index=1)
    with col2:
        show_details = st.checkbox("📊 顯示詳細分析", value=True)

    if st.button("🚀 開始分析", width='stretch'):
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        st.markdown("### 📈 分析進度")

        results = {}
        all_failed_tickers = []

        # 分析各指數
        groups_names = ['費城半導體', '納斯達克', '道瓊工業', '標普500']
        index_keys = ['SMH', 'QQQ', 'DIA', 'SPY']

        for i, (key, name) in enumerate(zip(index_keys, groups_names)):
            with st.expander(f"正在分析 {name} ({key})", expanded=True):
                tickers = index_stocks[key]
                st.write(f"分析標的: {', '.join(tickers)}")

                trend_data, failed = calculate_sma_trend(tickers, start_date, end_date)
                results[name] = trend_data
                all_failed_tickers.extend(failed)

                if not trend_data.empty:
                    latest_value = trend_data.iloc[-1] if len(trend_data) > 0 else 0
                    st.metric(f"{name} 趨勢強度", f"{latest_value}%")
                else:
                    st.error(f"❌ {name} 無法取得資料")

        # 建立結果DataFrame
        if results:
            st.markdown("### 📊 分析結果")

            # 找出最短的資料長度來對齊
            min_length = min([len(data) for data in results.values() if not data.empty])

            df_results = pd.DataFrame()
            for name, data in results.items():
                if not data.empty and len(data) >= min_length:
                    df_results[name] = data.tail(min_length).values

            if not df_results.empty:
                # 使用AAPL的日期作為索引
                try:
                    aapl_data = yf.download('AAPL', start=start_date, end=end_date)
                    if not aapl_data.empty and len(aapl_data) >= len(df_results):
                        df_results.index = aapl_data.tail(len(df_results)).index
                except:
                    pass

                # 最新日期在上
                df_results = df_results.iloc[::-1]

                # 顯示資料表
                st.dataframe(df_results, width='stretch')

                # 趨勢強度總覽
                st.markdown("### 🎯 最新趨勢強度")
                cols = st.columns(len(df_results.columns))
                for i, col_name in enumerate(df_results.columns):
                    with cols[i]:
                        latest_val = df_results[col_name].iloc[0] if len(df_results) > 0 else 0
                        if latest_val >= 70:
                            st.success(f"**{col_name}**\n{latest_val}% 💚")
                        elif latest_val >= 50:
                            st.info(f"**{col_name}**\n{latest_val}% 💙")
                        else:
                            st.error(f"**{col_name}**\n{latest_val}% ❤️")

                # 提供下載
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_results.to_excel(writer, sheet_name='大盤趨勢掃描')

                st.download_button(
                    label="📥 下載Excel報告",
                    data=output.getvalue(),
                    file_name=f"美股大盤趨勢掃描_{date.today().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width='stretch'
                )

                if show_details:
                    st.markdown("### 📈 趨勢圖表")
                    st.line_chart(df_results)

        # 錯誤報告
        if all_failed_tickers:
            with st.expander("⚠️ 下載失敗的股票", expanded=False):
                st.write("以下股票無法下載資料：")
                for ticker in set(all_failed_tickers):
                    st.write(f"- {ticker}")

if __name__ == "__main__":
    main()
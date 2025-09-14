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

    progress_placeholder = st.empty()
    for i, ticker in enumerate(tickers):
        try:
            progress_placeholder.progress((i + 1) / len(tickers), f"正在分析 {ticker} ({i+1}/{len(tickers)})")

            # 下載數據
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                failed_tickers.append(ticker)
                continue

            # 計算20天的SMA
            ma20 = talib.SMA(df['Close'].to_numpy().reshape(-1), timeperiod=20)
            res = np.where(df['Close'].to_numpy().reshape(-1) > ma20, 1, 0)
            data.append(res)
            valid_tickers.append(ticker)

        except Exception as e:
            failed_tickers.append(ticker)
            continue

    progress_placeholder.empty()

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
    此工具分析來自index_memb.xlsx的四大美股指數相對於20日移動平均線的趨勢強度：
    - **SMH**: 費城半導體指數 (30支股票)
    - **QQQ**: 納斯達克100指數 (101支股票)
    - **DIA**: 道瓊工業指數 (30支股票)
    - **SPY**: 標普500指數 (504支股票)
    - 分析期間固定為60天，總計665支股票
    """)

    # INDEX_MEMB 四大指數股票代碼 (來自index_memb.xlsx)
    index_stocks = {
        'SMH': [  # 費城半導體指數 (30支股票)
            'INTC', 'MPWR', 'ENTG', 'CRUS', 'QCOM', 'ADI', 'KLAC', 'MU', 'QRVO', 'LRCX',
            'GFS', 'SWKS', 'AMAT', 'TXN', 'NVDA', 'ARM', 'ASML', 'LSCC', 'MCHP', 'TER',
            'NXPI', 'ON', 'MRVL', 'TSM', 'AMD', 'ONTO', 'MTSI', 'AVGO', 'COHR', 'AMKR'
        ],
        'QQQ': [  # 納斯達克100指數 (101支股票)
            'ORLY', 'TMUS', 'MDLZ', 'GILD', 'ROP', 'VRSK', 'PDD', 'MNST', 'KHC', 'AEP',
            'IDXX', 'LULU', 'PAYX', 'ROST', 'AAPL', 'CCEP', 'CPRT', 'MELI', 'ADP', 'COST',
            'ODFL', 'SBUX', 'INTC', 'FAST', 'GEHC', 'CTAS', 'LIN', 'PEP', 'NFLX', 'DASH',
            'KDP', 'XEL', 'EXC', 'HON', 'VRTX', 'TSLA', 'AZN', 'MSFT', 'AMZN', 'EA',
            'INTU', 'CSX', 'AMGN', 'CMCSA', 'WBD', 'ISRG', 'BKNG', 'QCOM', 'CSGP', 'CDNS',
            'CTSH', 'ANSS', 'ADBE', 'ADSK', 'CSCO', 'REGN', 'CHTR', 'TTWO', 'ADI', 'KLAC',
            'SNPS', 'BKR', 'MAR', 'ZS', 'MU', 'CRWD', 'PCAR', 'META', 'MSTR', 'FTNT',
            'BIIB', 'AXON', 'PYPL', 'GOOGL', 'LRCX', 'FANG', 'GOOG', 'GFS', 'AMAT', 'TXN',
            'NVDA', 'CDW', 'ARM', 'ASML', 'ABNB', 'PLTR', 'WDAY', 'MDB', 'TTD', 'MCHP',
            'NXPI', 'ON', 'MRVL', 'DDOG', 'TEAM', 'AMD', 'CEG', 'DXCM', 'AVGO', 'PANW',
            'APP'
        ],
        'DIA': [  # 道瓊工業指數 (30支股票)
            'VZ', 'V', 'PG', 'AAPL', 'KO', 'JNJ', 'WMT', 'HON', 'SHW', 'BA',
            'HD', 'TRV', 'MSFT', 'AMZN', 'NKE', 'AMGN', 'MCD', 'DIS', 'UNH', 'CAT',
            'MRK', 'CSCO', 'CVX', 'CRM', 'JPM', 'AXP', 'IBM', 'NVDA', 'GS', 'MMM'
        ],
        'SPY': [  # 標普500指數 (504支股票)
            'DLTR', 'DG', 'BG', 'ADM', 'HRL', 'CAG', 'SJM', 'CHD', 'CLX', 'SYY',
            'MDLZ', 'EL', 'MNST', 'TSN', 'KHC', 'PG', 'CL', 'HSY', 'CPB', 'KO',
            'GIS', 'COST', 'MO', 'MKC', 'BF-B', 'PEP', 'KMB', 'TAP', 'KDP', 'WBA',
            'WMT', 'PM', 'KVUE', 'TGT', 'LW', 'KR', 'STZ', 'K', 'ABT', 'MRNA',
            'CAH', 'GILD', 'SOLV', 'HCA', 'HOLX', 'ZBH', 'ZTS', 'COO', 'IDXX', 'UHS',
            'CI', 'BAX', 'TECH', 'COR', 'JNJ', 'MDT', 'GEHC', 'WAT', 'DVA', 'ABBV',
            'CVS', 'STE', 'WST', 'VRTX', 'MCK', 'RMD', 'ELV', 'BDX', 'MTD', 'EW',
            'AMGN', 'MOH', 'RVTY', 'HUM', 'SYK', 'CRL', 'DHR', 'ISRG', 'IQV', 'DGX',
            'TMO', 'UNH', 'HSIC', 'CNC', 'BMY', 'MRK', 'LLY', 'REGN', 'LH', 'A',
            'PFE', 'INCY', 'ALGN', 'VTRS', 'BIIB', 'BSX', 'PODD', 'DXCM', 'AZO', 'ORLY',
            'KMX', 'EBAY', 'GPC', 'CMG', 'LULU', 'ROST', 'LKQ', 'DPZ', 'SBUX', 'TJX',
            'DASH', 'DHI', 'TSCO', 'TSLA', 'WYNN', 'MHK', 'DRI', 'HD', 'AMZN', 'LEN',
            'NKE', 'GRMN', 'BBY', 'LOW', 'LVS', 'NVR', 'PHM', 'HAS', 'BKNG', 'MCD',
            'ULTA', 'WSM', 'YUM', 'CCL', 'POOL', 'MAR', 'DECK', 'RCL', 'HLT', 'TPR',
            'RL', 'MGM', 'NCLH', 'CZR', 'ABNB', 'EXPE', 'F', 'APTV', 'GM', 'NEM',
            'CF', 'BALL', 'MOS', 'AMCR', 'LIN', 'IFF', 'SHW', 'MLM', 'SW', 'VMC',
            'NUE', 'ECL', 'AVY', 'APD', 'LYB', 'STLD', 'CTVA', 'PKG', 'DD', 'EMN',
            'DOW', 'ALB', 'IP', 'PPG', 'FCX', 'FE', 'AWK', 'AEP', 'D', 'PPL',
            'SO', 'ES', 'XEL', 'EXC', 'ATO', 'DUK', 'NEE', 'LNT', 'ED', 'WEC',
            'CNP', 'PNW', 'EVRG', 'ETR', 'CMS', 'AEE', 'DTE', 'AES', 'PCG', 'NI',
            'EIX', 'SRE', 'PEG', 'NRG', 'CEG', 'VST', 'MMC', 'MKTX', 'FDS', 'V',
            'WRB', 'MA', 'AJG', 'CBOE', 'ACGL', 'CB', 'L', 'BRO', 'PGR', 'CINF',
            'AON', 'GL', 'WTW', 'FIS', 'EG', 'ICE', 'AFL', 'TROW', 'AIG', 'HIG',
            'BRK-B', 'SPGI', 'TRV', 'ERIE', 'ALL', 'BLK', 'JKHY', 'BEN', 'MCO', 'CME',
            'AIZ', 'GPN', 'CPAY', 'BAC', 'PFG', 'MSCI', 'SCHW', 'BK', 'NTRS', 'IVZ',
            'HBAN', 'COF', 'STT', 'FITB', 'PRU', 'MET', 'PNC', 'FI', 'JPM', 'USB',
            'AMP', 'RJF', 'TFC', 'MTB', 'RF', 'KEY', 'AXP', 'BX', 'NDAQ', 'PYPL',
            'KKR', 'WFC', 'CFG', 'APO', 'SYF', 'C', 'DFS', 'GS', 'MS', 'AMT',
            'CCI', 'SBAC', 'WY', 'MAA', 'INVH', 'PSA', 'WELL', 'VICI', 'CPT', 'EXR',
            'CSGP', 'VTR', 'UDR', 'PLD', 'EQR', 'ARE', 'DOC', 'AVB', 'ESS', 'REG',
            'EQIX', 'O', 'FRT', 'CBRE', 'BXP', 'KIM', 'SPG', 'IRM', 'HST', 'DLR',
            'VRSK', 'PAYC', 'ROL', 'NOC', 'PAYX', 'CPRT', 'ADP', 'EXPD', 'ODFL', 'UBER',
            'EFX', 'RSG', 'FAST', 'CTAS', 'URI', 'HON', 'LHX', 'BA', 'OTIS', 'BR',
            'VLTO', 'TXT', 'FDX', 'MAS', 'CSX', 'WM', 'NDSN', 'UPS', 'GD', 'DAY',
            'HII', 'GWW', 'IR', 'ALLE', 'CHRW', 'NSC', 'LII', 'JBHT', 'WAB', 'J',
            'IEX', 'ROK', 'LDOS', 'CAT', 'SNA', 'LMT', 'AOS', 'RTX', 'TDG', 'JCI',
            'GE', 'BLDR', 'AME', 'FTV', 'DE', 'UNP', 'XYL', 'PNR', 'SWK', 'HWM',
            'EMR', 'HUBB', 'LUV', 'PCAR', 'CMI', 'AXON', 'ITW', 'TT', 'ETN', 'DOV',
            'PH', 'CARR', 'GNRC', 'PWR', 'MMM', 'DAL', 'GEV', 'UAL', 'VRSN', 'ROP',
            'AAPL', 'ENPH', 'APH', 'DELL', 'INTC', 'MSI', 'FSLR', 'TYL', 'MPWR', 'MSFT',
            'AKAM', 'JNPR', 'INTU', 'STX', 'GDDY', 'HPQ', 'QCOM', 'CDNS', 'CTSH', 'ANSS',
            'TDY', 'ADBE', 'IT', 'ADSK', 'CSCO', 'GEN', 'ADI', 'FICO', 'KLAC', 'KEYS',
            'HPE', 'SNPS', 'ACN', 'ZBRA', 'FFIV', 'MU', 'CRM', 'PTC', 'GLW', 'NOW',
            'CRWD', 'ORCL', 'TRMB', 'FTNT', 'EPAM', 'IBM', 'WDC', 'LRCX', 'TEL', 'NTAP',
            'SWKS', 'AMAT', 'TXN', 'NVDA', 'CDW', 'PLTR', 'WDAY', 'MCHP', 'TER', 'NXPI',
            'ON', 'JBL', 'AMD', 'ANET', 'AVGO', 'PANW', 'SMCI', 'TMUS', 'T', 'VZ',
            'LYV', 'NFLX', 'EA', 'MTCH', 'CMCSA', 'WBD', 'DIS', 'IPG', 'FOXA', 'OMC',
            'CHTR', 'TTWO', 'FOX', 'META', 'PARA', 'NWS', 'GOOGL', 'TKO', 'GOOG', 'NWSA',
            'APA', 'EXE', 'XOM', 'COP', 'OXY', 'SLB', 'CVX', 'WMB', 'KMI', 'BKR',
            'HES', 'VLO', 'EQT', 'CTRA', 'HAL', 'PSX', 'TRGP', 'DVN', 'MPC', 'FANG',
            'EOG', 'TPL', 'OKE', 'MRP-W'
        ]
    }

    # 參數設定 - 固定60天
    days = 60
    show_details = st.checkbox("📊 顯示詳細分析", value=True, key="us_market_details_check")

    if st.button("🚀 開始分析", width='stretch', key="us_market_analysis_btn"):
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        st.markdown("### 📈 分析進度")

        results = {}
        all_failed_tickers = []

        # 分析各指數
        groups_names = ['費城半導體指數', '納斯達克100指數', '道瓊工業指數', '標普500指數']
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
            valid_data = [len(data) for data in results.values() if not data.empty]

            if valid_data:  # 確保有有效數據
                min_length = min(valid_data)

                df_results = pd.DataFrame()
                for name, data in results.items():
                    if not data.empty and len(data) >= min_length:
                        df_results[name] = data.tail(min_length).values
            else:
                df_results = pd.DataFrame()
                st.warning("⚠️ 沒有成功獲取任何指數的資料")

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
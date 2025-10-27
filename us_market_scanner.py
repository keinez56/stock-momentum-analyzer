# 美股大盤掃描
# US Market Scanner

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta, datetime
import warnings
import talib
from io import BytesIO

warnings.filterwarnings('ignore')

def calculate_sma_trend(tickers):
    """計算股票相對於20日均線的趨勢百分比（簡化優化版）"""
    # 先獲取參考日期（使用SPY作為基準）
    try:
        reference_df = yf.download('SPY', period='3mo', progress=False)
        if reference_df.empty:
            return pd.Series(dtype='float64'), []
        reference_dates = reference_df.index
    except:
        return pd.Series(dtype='float64'), []

    data_dict = {}
    failed_tickers = []
    expected_length = len(reference_dates)

    # 一次性批量下載所有股票
    try:
        st.write(f"📥 正在批量下載 {len(tickers)} 支股票數據...")
        tickers_str = ' '.join(tickers)
        df_batch = yf.download(tickers_str, period='3mo', progress=False, group_by='ticker', threads=True)

        st.write(f"✅ 下載完成，開始處理數據...")

        # 處理每支股票
        for ticker in tickers:
            try:
                # 獲取該股票的數據
                if len(tickers) == 1:
                    df_ticker = df_batch
                else:
                    if ticker in df_batch.columns.get_level_values(0):
                        df_ticker = df_batch[ticker]
                    else:
                        failed_tickers.append(ticker)
                        continue

                if df_ticker.empty:
                    failed_tickers.append(ticker)
                    continue

                # 重新索引到參考日期
                df_ticker = df_ticker.reindex(reference_dates, method='ffill')

                if len(df_ticker) != expected_length:
                    failed_tickers.append(ticker)
                    continue

                # 計算20日SMA
                close_array = df_ticker['Close'].to_numpy().reshape(-1)
                ma20 = talib.SMA(close_array, timeperiod=20)

                # 只使用有效的MA20值
                valid_mask = ~np.isnan(ma20)
                if valid_mask.sum() > 0:
                    close_valid = close_array[valid_mask]
                    ma20_valid = ma20[valid_mask]
                    res_valid = np.where(close_valid > ma20_valid, 1, 0)

                    res = np.zeros(len(close_array))
                    res[valid_mask] = res_valid

                    if len(res) == expected_length:
                        data_dict[ticker] = res
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)

            except Exception as e:
                failed_tickers.append(ticker)
                continue

    except Exception as batch_error:
        st.error(f"❌ 批量下載失敗: {str(batch_error)[:100]}")
        return pd.Series(dtype='float64'), tickers

    if not data_dict:
        return pd.Series(dtype='float64'), failed_tickers

    # 使用字典創建DataFrame
    df_temp = pd.DataFrame(data_dict, index=reference_dates)

    # 計算每日高於MA20的股票百分比
    if len(df_temp.columns) > 0:
        row_sums = round(df_temp.sum(axis=1) / len(df_temp.columns) * 100)
    else:
        row_sums = pd.Series(dtype='float64')

    st.write(f"✅ 成功處理 {len(data_dict)} 支股票，失敗 {len(failed_tickers)} 支")

    return row_sums, failed_tickers

def main():
    """美股大盤掃描主程式"""
    st.title("📊 美股大盤掃描")
    st.markdown("---")

    st.markdown("""
    ### 📋 功能說明
    此工具分析四大美股指數相對於20日移動平均線的趨勢強度：
    - **SMH**: 費城半導體指數 (30支股票)
    - **QQQ**: 納斯達克100指數 (101支股票)
    - **DIA**: 道瓊工業指數 (30支股票)
    - **SPY**: 標普500指數 (504支股票)
    - 顯示過去20個交易日的數據，最新日期在頂部
    - 提供表格形式呈現和Excel報告下載
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

    if st.button("🚀 開始分析美股4大指數趨勢", type="primary", width='stretch', key="us_market_analysis_btn"):

        # 創建進度條
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("正在分析美股4大指數趨勢，請稍候..."):
            results = {}
            all_failed_tickers = []

            # 分析各指數
            index_names = ['SMH-費城半導體', 'QQQ-納斯達克100', 'DIA-道瓊工業', 'SPY-標普500']
            index_keys = ['SMH', 'QQQ', 'DIA', 'SPY']
            total_indices = len(index_keys)

            for i, (key, display_name) in enumerate(zip(index_keys, index_names)):
                # 更新進度
                progress = (i + 1) / total_indices
                progress_bar.progress(progress)
                status_text.text(f"正在分析 {display_name} ({i+1}/{total_indices})")

                tickers = index_stocks[key]
                trend_data, failed = calculate_sma_trend(tickers)
                results[display_name] = trend_data
                all_failed_tickers.extend(failed)

        # 清除進度條
        progress_bar.empty()
        status_text.empty()

        # 建立結果DataFrame
        if any(not data.empty for data in results.values()):
            st.markdown("### 📊 美股4大指數趨勢強度表")

            # 數據整理
            valid_data = [len(data) for data in results.values() if not data.empty]

            if valid_data:
                min_length = min(valid_data)
                if min_length > 0:
                    df_results = pd.DataFrame()
                    for index_name, data in results.items():
                        if not data.empty and len(data) >= min_length:
                            df_results[index_name] = data.tail(min_length).values

                    # 添加日期索引
                    latest_date = None
                    try:
                        spy_data = yf.download('SPY', period='3mo', progress=False)
                        if not spy_data.empty and len(spy_data) >= len(df_results):
                            dates = spy_data.tail(len(df_results)).index.strftime('%Y-%m-%d')
                            df_results.index = dates
                            # 保存最新日期用於檔名
                            latest_date = spy_data.tail(len(df_results)).index[-1].strftime('%Y%m%d')
                    except:
                        # 如果無法獲取SPY數據，使用今天往前推算
                        from datetime import date
                        date_range = pd.date_range(end=date.today(), periods=len(df_results), freq='B')
                        df_results.index = date_range.strftime('%Y-%m-%d')
                        latest_date = date_range[-1].strftime('%Y%m%d')

                    # 只取最近20個交易日，最新在上
                    df_display = df_results.tail(20).iloc[::-1]

                    # 顯示統計資訊
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("分析指數數", len(df_display.columns))
                    with col2:
                        strong_indices = sum(1 for col in df_display.columns if df_display[col].iloc[0] >= 70)
                        st.metric("強勢指數", strong_indices)
                    with col3:
                        weak_indices = sum(1 for col in df_display.columns if df_display[col].iloc[0] < 50)
                        st.metric("弱勢指數", weak_indices)
                    with col4:
                        avg_strength = df_display.iloc[0].mean()
                        st.metric("平均強度", f"{avg_strength:.1f}%")

                    # 顯示表格（最新20個交易日，最新在上）
                    st.markdown("**📋 過去20個交易日趨勢強度 (最新在上)**")

                    # 顯示乾淨的表格，不使用顏色編碼
                    st.dataframe(df_display, width='stretch', height=600)

                    # 最新趨勢強度總覽
                    st.markdown("### 🎯 最新趨勢強度總覽")
                    cols = st.columns(4)
                    sorted_indices = df_display.iloc[0].sort_values(ascending=False)

                    for i, (index_name, value) in enumerate(sorted_indices.items()):
                        with cols[i % 4]:
                            if value >= 70:
                                st.success(f"**{index_name}**\n{value}% 💚 強勢")
                            elif value >= 50:
                                st.info(f"**{index_name}**\n{value}% 💙 中性")
                            else:
                                st.error(f"**{index_name}**\n{value}% ❤️ 弱勢")

                    # Excel下載
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # 將數據寫入Excel，包含完整數據（不只20天）
                        full_data = df_results.iloc[::-1]  # 最新在上
                        full_data.to_excel(writer, sheet_name='美股4大指數趨勢')

                        # 添加條件格式
                        workbook = writer.book
                        worksheet = writer.sheets['美股4大指數趨勢']

                        # 設置標題格式
                        title_format = workbook.add_format({
                            'bold': True,
                            'font_size': 12,
                            'align': 'center',
                            'valign': 'vcenter'
                        })

                        # 條件格式：3色階
                        n_rows, n_cols = len(full_data), len(full_data.columns)
                        if n_rows > 0 and n_cols > 0:
                            cell_range = f'B2:{chr(66 + n_cols - 1)}{n_rows + 1}'
                            worksheet.conditional_format(cell_range, {
                                'type': '3_color_scale',
                                'min_value': 0,
                                'mid_value': 50,
                                'max_value': 100,
                                'min_color': '#FF6B6B',  # 紅色
                                'mid_color': '#FFFFFF',  # 白色
                                'max_color': '#51CF66'   # 綠色
                            })

                    output.seek(0)

                    st.download_button(
                        label="📥 下載美股大盤趨勢分析報告 (Excel)",
                        data=output.read(),
                        file_name=f"美股4大指數趨勢分析_{latest_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )

                else:
                    st.error("❌ 無法取得足夠的數據進行分析")
            else:
                st.error("❌ 沒有成功獲取任何指數的資料")

        # 失敗股票報告（簡化顯示）
        if all_failed_tickers:
            failed_unique = list(set(all_failed_tickers))
            st.info(f"ℹ️ 共有 {len(failed_unique)} 支股票無法獲取數據，但分析仍可正常進行")

if __name__ == "__main__":
    main()

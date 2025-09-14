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
            df_ticker = yf.download(ticker, start=start_date, end=end_date, progress=False)

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
    此工具分析來自spx_index.xlsx的美股11大類股趨勢強度：
    - 分析503支SPX成分股，按11大類股分類
    - 計算各類股中股票高於20日均線的百分比
    - 分析期間固定為60天
    - 提供歷史趨勢圖表和Excel報告下載
    """)

    # SPX 11大類股股票代碼 (來自spx_index.xlsx)
    sector_stocks = {
        'XLB': [  # 原材料 (26支股票)
            'NEM', 'CF', 'BALL', 'MOS', 'AMCR', 'LIN', 'IFF', 'SHW', 'MLM', 'SW',
            'VMC', 'NUE', 'ECL', 'AVY', 'APD', 'LYB', 'STLD', 'CTVA', 'PKG', 'DD',
            'EMN', 'DOW', 'ALB', 'IP', 'PPG', 'FCX'
        ],
        'XLC': [  # 通訊服務 (23支股票)
            'TMUS', 'T', 'VZ', 'LYV', 'NFLX', 'EA', 'MTCH', 'CMCSA', 'WBD', 'DIS',
            'IPG', 'FOXA', 'OMC', 'CHTR', 'TTWO', 'FOX', 'META', 'PARA', 'NWS', 'GOOGL',
            'TKO', 'GOOG', 'NWSA'
        ],
        'XLE': [  # 能源 (23支股票)
            'APA', 'EXE', 'XOM', 'COP', 'OXY', 'SLB', 'CVX', 'WMB', 'KMI', 'BKR',
            'HES', 'VLO', 'EQT', 'CTRA', 'HAL', 'PSX', 'TRGP', 'DVN', 'MPC', 'FANG',
            'EOG', 'TPL', 'OKE'
        ],
        'XLF': [  # 金融 (73支股票)
            'MMC', 'MKTX', 'FDS', 'V', 'WRB', 'MA', 'AJG', 'CBOE', 'ACGL', 'CB',
            'L', 'BRO', 'PGR', 'CINF', 'AON', 'GL', 'WTW', 'FIS', 'EG', 'ICE',
            'AFL', 'TROW', 'AIG', 'HIG', 'BRK-B', 'SPGI', 'TRV', 'ERIE', 'ALL', 'BLK',
            'JKHY', 'BEN', 'MCO', 'CME', 'AIZ', 'GPN', 'CPAY', 'BAC', 'PFG', 'MSCI',
            'SCHW', 'BK', 'NTRS', 'IVZ', 'HBAN', 'COF', 'STT', 'FITB', 'PRU', 'MET',
            'PNC', 'FI', 'JPM', 'USB', 'AMP', 'RJF', 'TFC', 'MTB', 'RF', 'KEY',
            'AXP', 'BX', 'NDAQ', 'PYPL', 'KKR', 'WFC', 'CFG', 'APO', 'SYF', 'C',
            'DFS', 'GS', 'MS'
        ],
        'XLI': [  # 工業 (78支股票)
            'VRSK', 'PAYC', 'ROL', 'NOC', 'PAYX', 'CPRT', 'ADP', 'EXPD', 'ODFL', 'UBER',
            'EFX', 'RSG', 'FAST', 'CTAS', 'URI', 'HON', 'LHX', 'BA', 'OTIS', 'BR',
            'VLTO', 'TXT', 'FDX', 'MAS', 'CSX', 'WM', 'NDSN', 'UPS', 'GD', 'DAY',
            'HII', 'GWW', 'IR', 'ALLE', 'CHRW', 'NSC', 'LII', 'JBHT', 'WAB', 'J',
            'IEX', 'ROK', 'LDOS', 'CAT', 'SNA', 'LMT', 'AOS', 'RTX', 'TDG', 'JCI',
            'GE', 'BLDR', 'AME', 'FTV', 'DE', 'UNP', 'XYL', 'PNR', 'SWK', 'HWM',
            'EMR', 'HUBB', 'LUV', 'PCAR', 'CMI', 'AXON', 'ITW', 'TT', 'ETN', 'DOV',
            'PH', 'CARR', 'GNRC', 'PWR', 'MMM', 'DAL', 'GEV', 'UAL'
        ],
        'XLK': [  # 科技 (69支股票)
            'VRSN', 'ROP', 'AAPL', 'ENPH', 'APH', 'DELL', 'INTC', 'MSI', 'FSLR', 'TYL',
            'MPWR', 'MSFT', 'AKAM', 'JNPR', 'INTU', 'STX', 'GDDY', 'HPQ', 'QCOM', 'CDNS',
            'CTSH', 'ANSS', 'TDY', 'ADBE', 'IT', 'ADSK', 'CSCO', 'GEN', 'ADI', 'FICO',
            'KLAC', 'KEYS', 'HPE', 'SNPS', 'ACN', 'ZBRA', 'FFIV', 'MU', 'CRM', 'PTC',
            'GLW', 'NOW', 'CRWD', 'ORCL', 'TRMB', 'FTNT', 'EPAM', 'IBM', 'WDC', 'LRCX',
            'TEL', 'NTAP', 'SWKS', 'AMAT', 'TXN', 'NVDA', 'CDW', 'PLTR', 'WDAY', 'MCHP',
            'TER', 'NXPI', 'ON', 'JBL', 'AMD', 'ANET', 'AVGO', 'PANW', 'SMCI'
        ],
        'XLP': [  # 必需消費品 (38支股票)
            'DLTR', 'DG', 'BG', 'ADM', 'HRL', 'CAG', 'SJM', 'CHD', 'CLX', 'SYY',
            'MDLZ', 'EL', 'MNST', 'TSN', 'KHC', 'PG', 'CL', 'HSY', 'CPB', 'KO',
            'GIS', 'COST', 'MO', 'MKC', 'BF-B', 'PEP', 'KMB', 'TAP', 'KDP', 'WBA',
            'WMT', 'PM', 'KVUE', 'TGT', 'LW', 'KR', 'STZ', 'K'
        ],
        'XLRE': [  # 房地產 (31支股票)
            'AMT', 'CCI', 'SBAC', 'WY', 'MAA', 'INVH', 'PSA', 'WELL', 'VICI', 'CPT',
            'EXR', 'CSGP', 'VTR', 'UDR', 'PLD', 'EQR', 'ARE', 'DOC', 'AVB', 'ESS',
            'REG', 'EQIX', 'O', 'FRT', 'CBRE', 'BXP', 'KIM', 'SPG', 'IRM', 'HST',
            'DLR'
        ],
        'XLU': [  # 公用事業 (31支股票)
            'FE', 'AWK', 'AEP', 'D', 'PPL', 'SO', 'ES', 'XEL', 'EXC', 'ATO',
            'DUK', 'NEE', 'LNT', 'ED', 'WEC', 'CNP', 'PNW', 'EVRG', 'ETR', 'CMS',
            'AEE', 'DTE', 'AES', 'PCG', 'NI', 'EIX', 'SRE', 'PEG', 'NRG', 'CEG',
            'VST'
        ],
        'XLV': [  # 醫療保健 (60支股票)
            'ABT', 'MRNA', 'CAH', 'GILD', 'SOLV', 'HCA', 'HOLX', 'ZBH', 'ZTS', 'COO',
            'IDXX', 'UHS', 'CI', 'BAX', 'TECH', 'COR', 'JNJ', 'MDT', 'GEHC', 'WAT',
            'DVA', 'ABBV', 'CVS', 'STE', 'WST', 'VRTX', 'MCK', 'RMD', 'ELV', 'BDX',
            'MTD', 'EW', 'AMGN', 'MOH', 'RVTY', 'HUM', 'SYK', 'CRL', 'DHR', 'ISRG',
            'IQV', 'DGX', 'TMO', 'UNH', 'HSIC', 'CNC', 'BMY', 'MRK', 'LLY', 'REGN',
            'LH', 'A', 'PFE', 'INCY', 'ALGN', 'VTRS', 'BIIB', 'BSX', 'PODD', 'DXCM'
        ],
        'XLY': [  # 非必需消費品 (51支股票)
            'AZO', 'ORLY', 'KMX', 'EBAY', 'GPC', 'CMG', 'LULU', 'ROST', 'LKQ', 'DPZ',
            'SBUX', 'TJX', 'DASH', 'DHI', 'TSCO', 'TSLA', 'WYNN', 'MHK', 'DRI', 'HD',
            'AMZN', 'LEN', 'NKE', 'GRMN', 'BBY', 'LOW', 'LVS', 'NVR', 'PHM', 'HAS',
            'BKNG', 'MCD', 'ULTA', 'WSM', 'YUM', 'CCL', 'POOL', 'MAR', 'DECK', 'RCL',
            'HLT', 'TPR', 'RL', 'MGM', 'NCLH', 'CZR', 'ABNB', 'EXPE', 'F', 'APTV',
            'GM'
        ]
    }

    # 產業中文名稱對照
    sector_names = {
        'XLB': '原材料',
        'XLC': '通訊服務',
        'XLE': '能源',
        'XLF': '金融',
        'XLI': '工業',
        'XLK': '科技',
        'XLP': '必需消費品',
        'XLRE': '房地產',
        'XLU': '公用事業',
        'XLV': '醫療保健',
        'XLY': '非必需消費品'
    }

    # 參數設定 - 固定60天
    analysis_days = 60
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
            valid_data = [len(data) for data in results.values() if not data.empty]

            if valid_data:  # 確保有有效數據
                min_length = min(valid_data)
                if min_length > 0:
                    df_results = pd.DataFrame()
                    for sector_name, data in results.items():
                        if not data.empty and len(data) >= min_length:
                            df_results[sector_name] = data.tail(min_length).values
                else:
                    df_results = pd.DataFrame()
            else:
                df_results = pd.DataFrame()
                st.warning("⚠️ 沒有成功獲取任何類股的資料")

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
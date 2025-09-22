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

    for ticker in tickers:
        try:
            # 下載數據
            df_ticker = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df_ticker.empty:
                failed_tickers.append(ticker)
                continue

            # 計算20日SMA
            close_array = df_ticker['Close'].to_numpy().reshape(-1)
            ma20 = talib.SMA(close_array, timeperiod=20)

            # 只使用有效的MA20值（排除前20個NaN值）
            valid_mask = ~np.isnan(ma20)
            if valid_mask.sum() > 0:  # 確保有有效數據
                # 只比較有MA20值的部分
                close_valid = close_array[valid_mask]
                ma20_valid = ma20[valid_mask]
                res_valid = np.where(close_valid > ma20_valid, 1, 0)

                # 補齊前面的0值（前20天沒有MA20數據）
                res = np.zeros(len(close_array))
                res[valid_mask] = res_valid

                data.append(res)
                valid_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)

        except Exception as e:
            failed_tickers.append(ticker)
            continue

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

    # 計算每日高於MA20的股票百分比
    if len(valid_tickers) > 0:
        row_sums = round(df_temp.sum(axis=1) / len(valid_tickers) * 100)
    else:
        row_sums = pd.Series(dtype='float64')

    return row_sums, failed_tickers

def main():
    """美股趨勢掃描主程式"""
    st.title("🔍 美股趨勢掃描")
    st.markdown("---")

    st.markdown("""
    ### 📋 功能說明
    此工具分析美股11大類股趨勢強度：
    - 分析SPX成分股，按11大類股分類
    - 計算各類股中股票高於20日均線的百分比
    - 顯示過去20個交易日的數據，最新日期在頂部
    - 強勢(≥70%) 💚、中性(50-70%) 💙、弱勢(<50%) ❤️
    - 提供表格形式呈現和Excel報告下載
    """)

    # SPX 11大類股股票代碼
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

    # 參數設定 - 固定60天（確保有足夠的20個交易日數據 + MA20計算需要的額外天數）
    analysis_days = 60

    if st.button("🚀 開始分析美股11大類股趨勢", type="primary", width='stretch', key="us_trend_analysis_btn"):
        end_date = date.today()
        start_date = end_date - timedelta(days=analysis_days)

        # 創建進度條
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("正在分析美股11大類股趨勢，請稍候..."):
            results = {}
            all_failed_tickers = []
            total_sectors = len(sector_names)

            # 分析各行業
            for i, (sector_code, chinese_name) in enumerate(sector_names.items()):
                # 更新進度
                progress = (i + 1) / total_sectors
                progress_bar.progress(progress)
                status_text.text(f"正在分析 {chinese_name} ({i+1}/{total_sectors})")

                tickers = sector_stocks[sector_code]

                trend_data, failed = calculate_sector_trend(
                    tickers, start_date, end_date, chinese_name
                )
                results[chinese_name] = trend_data
                all_failed_tickers.extend(failed)

        # 清除進度條
        progress_bar.empty()
        status_text.empty()

        # 建立結果DataFrame
        if any(not data.empty for data in results.values()):
            st.markdown("### 📊 美股11大類股趨勢強度表")

            # 數據整理
            valid_data = [len(data) for data in results.values() if not data.empty]

            if valid_data:
                min_length = min(valid_data)
                if min_length > 0:
                    df_results = pd.DataFrame()
                    for sector_name, data in results.items():
                        if not data.empty and len(data) >= min_length:
                            df_results[sector_name] = data.tail(min_length).values

                    # 添加日期索引
                    try:
                        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
                        if not spy_data.empty and len(spy_data) >= len(df_results):
                            dates = spy_data.tail(len(df_results)).index.strftime('%Y-%m-%d')
                            df_results.index = dates
                    except:
                        # 如果無法獲取SPY數據，使用日期範圍
                        date_range = pd.date_range(end=end_date, periods=len(df_results), freq='B')
                        df_results.index = date_range.strftime('%Y-%m-%d')

                    # 只取最近20個交易日，最新在上
                    df_display = df_results.tail(20).iloc[::-1]

                    # 顯示統計資訊
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("分析類股數", len(df_display.columns))
                    with col2:
                        strong_sectors = sum(1 for col in df_display.columns if df_display[col].iloc[0] >= 70)
                        st.metric("強勢類股", strong_sectors)
                    with col3:
                        weak_sectors = sum(1 for col in df_display.columns if df_display[col].iloc[0] < 50)
                        st.metric("弱勢類股", weak_sectors)
                    with col4:
                        avg_strength = df_display.iloc[0].mean()
                        st.metric("平均強度", f"{avg_strength:.1f}%")

                    # 顯示表格（最新20個交易日，最新在上）
                    st.markdown("**📋 過去20個交易日趨勢強度 (最新在上)**")

                    # 使用styler來美化表格
                    def color_cells(val):
                        if val >= 70:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold'
                        elif val >= 50:
                            return 'background-color: #d1ecf1; color: #0c5460'
                        else:
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'

                    styled_df = df_display.style.applymap(color_cells, subset=df_display.columns)
                    st.dataframe(styled_df, width='stretch', height=600)

                    # 最新趨勢強度總覽
                    st.markdown("### 🎯 最新趨勢強度總覽")
                    cols = st.columns(3)
                    sorted_sectors = df_display.iloc[0].sort_values(ascending=False)

                    for i, (sector_name, value) in enumerate(sorted_sectors.items()):
                        with cols[i % 3]:
                            if value >= 70:
                                st.success(f"**{sector_name}**\n{value}% 💚 強勢")
                            elif value >= 50:
                                st.info(f"**{sector_name}**\n{value}% 💙 中性")
                            else:
                                st.error(f"**{sector_name}**\n{value}% ❤️ 弱勢")

                    # Excel下載
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # 將數據寫入Excel，包含完整數據（不只20天）
                        full_data = df_results.iloc[::-1]  # 最新在上
                        full_data.to_excel(writer, sheet_name='美股11大類股趨勢')

                        # 添加條件格式
                        workbook = writer.book
                        worksheet = writer.sheets['美股11大類股趨勢']

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
                        label="📥 下載美股趨勢分析報告 (Excel)",
                        data=output.read(),
                        file_name=f"美股11大類股趨勢分析_{date.today().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )

                else:
                    st.error("❌ 無法取得足夠的數據進行分析")
            else:
                st.error("❌ 沒有成功獲取任何類股的資料")

        # 失敗股票報告（簡化顯示）
        if all_failed_tickers:
            failed_unique = list(set(all_failed_tickers))
            st.info(f"ℹ️ 共有 {len(failed_unique)} 支股票無法獲取數據，但分析仍可正常進行")

if __name__ == "__main__":
    main()
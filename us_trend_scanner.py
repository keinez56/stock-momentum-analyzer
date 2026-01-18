# 美股趨勢掃描
# US Stock Trend Scanner

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta, datetime
import warnings
import talib
from io import BytesIO
from collections import OrderedDict

warnings.filterwarnings('ignore')

def calculate_sector_trend(tickers, sector_name):
    """計算行業趨勢（逐一下載版本）"""
    # 先獲取參考日期（使用SPY作為基準）
    # 使用明確的日期範圍，確保包含最新數據
    from datetime import date, timedelta

    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # 3個月

        reference_df = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if reference_df.empty:
            return pd.Series(dtype='float64'), []
        reference_dates = reference_df.index

        st.write(f"📅 {sector_name} 日期範圍: {reference_dates[0].strftime('%Y-%m-%d')} 至 {reference_dates[-1].strftime('%Y-%m-%d')}")
    except:
        return pd.Series(dtype='float64'), []

    data_dict = {}
    failed_tickers = []
    expected_length = len(reference_dates)
    total = len(tickers)

    st.write(f"📥 {sector_name}: 開始下載 {total} 支股票...")

    # 逐一下載每支股票
    for i, ticker in enumerate(tickers, 1):
        # 每5支顯示一次進度（行業股票數量較少）
        if i % 5 == 0 or i == 1 or i == total:
            st.write(f"  {sector_name} 進度: {i}/{total}")

        try:
            # 單獨下載一支股票（使用與SPY相同的日期範圍）
            df_ticker = yf.download(ticker, start=start_date, end=end_date, progress=False)

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

        except Exception:
            failed_tickers.append(ticker)
            continue

    if not data_dict:
        st.warning(f"⚠️ {sector_name}: 沒有成功下載任何股票")
        return pd.Series(dtype='float64'), failed_tickers

    # 使用字典創建DataFrame
    df_temp = pd.DataFrame(data_dict, index=reference_dates)

    # 計算每日高於MA20的股票百分比
    if len(df_temp.columns) > 0:
        row_sums = round(df_temp.sum(axis=1) / len(df_temp.columns) * 100)
    else:
        row_sums = pd.Series(dtype='float64')

    if failed_tickers:
        st.write(f"✅ {sector_name}: 成功 {len(data_dict)} 支，失敗 {len(failed_tickers)} 支 ({', '.join(failed_tickers)})")
    else:
        st.write(f"✅ {sector_name}: 成功 {len(data_dict)} 支，失敗 0 支")

    return row_sums, failed_tickers

def main():
    """美股趨勢掃描主程式"""
    st.title("🔍 美股趨勢掃描")
    st.markdown("---")

    st.markdown("""
    ### 📋 功能說明
    此工具分析美股11大類股趨勢強度：
    - 分析SPX成分股，按11大類股分類（通訊、選消、必消、能源、金融、醫療、工業、材料、地產、資訊、公用）
    - 計算各類股中股票高於20日均線的百分比
    - 顯示過去20個交易日的數據，最新日期在頂部
    - 提供表格形式呈現和Excel報告下載
    """)

    # SPX 11大類股股票代碼 (2025年1月更新)
    sector_stocks = {
        'XLB': [  # 原材料 (26支股票)
            'LIN', 'NEM', 'SHW', 'ECL', 'VMC', 'APD', 'MLM', 'DD', 'FCX', 'NUE',
            'CTVA', 'IP', 'PPG', 'STLD', 'PKG', 'AMCR', 'DOW', 'IFF', 'CF', 'BALL',
            'AVY', 'LYB', 'MOS', 'ALB', 'EMN'
        ],
        'XLC': [  # 通訊服務 (23支股票)
            'META', 'GOOGL', 'GOOG', 'NFLX', 'WBD', 'VZ', 'EA', 'DIS', 'CMCSA', 'TTWO',
            'TMUS', 'T', 'CHTR', 'LYV', 'TTD', 'OMC', 'TKO', 'FOXA', 'NWSA', 'IPG',
            'MTCH', 'FOX', 'NWS'
        ],
        'XLE': [  # 能源 (22支股票)
            'XOM', 'CVX', 'COP', 'WMB', 'EOG', 'MPC', 'KMI', 'PSX', 'SLB', 'VLO',
            'BKR', 'OKE', 'TRGP', 'EQT', 'OXY', 'FANG', 'EXE', 'DVN', 'HAL', 'TPL',
            'CTRA', 'APA'
        ],
        'XLF': [  # 金融 (76支股票)
            'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
            'BLK', 'SCHW', 'SPGI', 'PGR', 'COF', 'BX', 'HOOD', 'CB', 'MMC', 'CME',
            'ICE', 'KKR', 'AJG', 'PNC', 'COIN', 'AON', 'BK', 'MCO', 'USB', 'FISV',
            'PYPL', 'TRV', 'APO', 'TFC', 'ALL', 'AFL', 'MET', 'AMP', 'AIG', 'MSCI',
            'NDAQ', 'HIG', 'PRU', 'FIS', 'ACGL', 'WTW', 'STT', 'MTB', 'IBKR', 'RJF',
            'FITB', 'BRO', 'SYF', 'CINF', 'NTRS', 'HBAN', 'CBOE', 'RF', 'WRB', 'CFG',
            'TROW', 'GPN', 'CPAY', 'KEY', 'L', 'PFG', 'EG', 'GL', 'AIZ', 'JKHY',
            'FDS', 'IVZ', 'ERIE', 'BEN'
        ],
        'XLI': [  # 工業 (80支股票)
            'GE', 'CAT', 'RTX', 'UBER', 'GEV', 'BA', 'ETN', 'UNP', 'HON', 'ADP',
            'DE', 'LMT', 'PH', 'TT', 'GD', 'MMM', 'WM', 'NOC', 'EMR', 'TDG',
            'JCI', 'CTAS', 'ITW', 'NSC', 'CSX', 'PWR', 'UPS', 'URI', 'CMI', 'LHX',
            'AXON', 'FAST', 'FDX', 'PCAR', 'CARR', 'RSG', 'AME', 'GWW', 'CPRT', 'PAYX',
            'ROK', 'DAL', 'OTIS', 'XYL', 'VRSK', 'WAB', 'EFX', 'IR', 'UAL', 'EME',
            'BR', 'VLTO', 'ODFL', 'LDOS', 'HUBB', 'DOV', 'J', 'SNA', 'PNR', 'FTV',
            'LII', 'EXPD', 'CHRW', 'TXT', 'ALLE', 'MAS', 'BLDR', 'NDSN', 'IEX', 'HII',
            'SWK', 'DAY', 'JBHT', 'GNRC', 'AOS'
        ],
        'XLK': [  # 科技 (70支股票)
            'NVDA', 'MSFT', 'AAPL', 'AVGO', 'PLTR', 'ORCL', 'CSCO', 'IBM', 'AMD', 'CRM',
            'MU', 'APP', 'NOW', 'INTU', 'LRCX', 'QCOM', 'AMAT', 'TXN', 'INTC', 'ANET',
            'APH', 'ACN', 'KLAC', 'ADBE', 'PANW', 'CRWD', 'ADI', 'CDNS', 'SNPS', 'MSI',
            'ADSK', 'TEL', 'GLW', 'NXPI', 'FTNT', 'STX', 'ROP', 'DDOG', 'WDAY', 'DELL',
            'WDC', 'MPWR', 'FICO', 'MCHP', 'HPE', 'CTSH', 'KEYS', 'SMCI', 'TDY', 'HPQ',
            'PTC', 'FSLR', 'VRSN', 'JBL', 'NTAP', 'TYL', 'TER', 'CDW', 'ON', 'GDDY',
            'FFIV', 'TRMB', 'IT', 'GEN', 'ZBRA', 'SWKS', 'AKAM', 'EPAM'
        ],
        'XLP': [  # 必需消費品 (37支股票)
            'WMT', 'COST', 'PG', 'KO', 'PM', 'MDLZ', 'PEP', 'MO', 'CL', 'MNST',
            'KMB', 'TGT', 'KR', 'SYY', 'KDP', 'KVUE', 'ADM', 'HSY', 'GIS', 'KHC',
            'K', 'DG', 'CHD', 'EL', 'STZ', 'DLTR', 'MKC', 'TSN', 'CLX', 'BG',
            'SJM', 'CAG', 'LW', 'TAP', 'HRL', 'CPB', 'BF-B'
        ],
        'XLRE': [  # 房地產 (31支股票)
            'WELL', 'PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'DLR', 'CBRE', 'CCI', 'CSGP',
            'VICI', 'VTR', 'IRM', 'EXR', 'AVB', 'EQR', 'SBAC', 'WY', 'ESS', 'INVH',
            'MAA', 'KIM', 'DOC', 'ARE', 'REG', 'CPT', 'BXP', 'UDR', 'HST', 'FRT',
            'O'
        ],
        'XLU': [  # 公用事業 (30支股票)
            'NEE', 'CEG', 'SO', 'DUK', 'VST', 'AEP', 'SRE', 'D', 'XEL', 'EXC',
            'ETR', 'PEG', 'WEC', 'ED', 'PCG', 'NRG', 'DTE', 'AEE', 'ATO', 'PPL',
            'AWK', 'ES', 'CNP', 'FE', 'EIX', 'NI', 'EVRG', 'LNT', 'AES', 'PNW'
        ],
        'XLV': [  # 醫療保健 (60支股票)
            'LLY', 'JNJ', 'ABBV', 'UNH', 'ABT', 'MRK', 'TMO', 'AMGN', 'ISRG', 'PFE',
            'BSX', 'GILD', 'DHR', 'SYK', 'MDT', 'VRTX', 'CVS', 'BMY', 'MCK', 'CI',
            'ELV', 'HCA', 'ZTS', 'REGN', 'COR', 'BDX', 'IDXX', 'EW', 'RMD', 'A',
            'CAH', 'GEHC', 'IQV', 'HUM', 'MTD', 'DXCM', 'STE', 'LH', 'BIIB', 'PODD',
            'DGX', 'ZBH', 'WST', 'WAT', 'CNC', 'HOLX', 'INCY', 'COO', 'VTRS', 'BAX',
            'UHS', 'SOLV', 'MOH', 'RVTY', 'TECH', 'MRNA', 'CRL', 'ALGN', 'HSIC', 'DVA'
        ],
        'XLY': [  # 非必需消費品 (51支股票)
            'AMZN', 'TSLA', 'HD', 'MCD', 'BKNG', 'TJX', 'LOW', 'DASH', 'SBUX', 'ORLY',
            'NKE', 'RCL', 'AZO', 'HLT', 'MAR', 'GM', 'ABNB', 'CMG', 'ROST', 'F',
            'DHI', 'YUM', 'GRMN', 'EBAY', 'CCL', 'TSCO', 'LEN', 'PHM', 'EXPE', 'ULTA',
            'WSM', 'TPR', 'NVR', 'DRI', 'GPC', 'LULU', 'APTV', 'LVS', 'BBY', 'DECK',
            'DPZ', 'RL', 'WYNN', 'NCLH', 'POOL', 'HAS', 'LKQ', 'MHK', 'MGM', 'KMX',
            'CVNA'
        ]
    }

    # 產業中文名稱對照（按指定順序排列）
    sector_names = OrderedDict([
        ('XLC', '通訊'),
        ('XLY', '選消'),
        ('XLP', '必消'),
        ('XLE', '能源'),
        ('XLF', '金融'),
        ('XLV', '醫療'),
        ('XLI', '工業'),
        ('XLB', '材料'),
        ('XLRE', '地產'),
        ('XLK', '資訊'),
        ('XLU', '公用')
    ])

    if st.button("🚀 開始分析美股11大類股趨勢", type="primary", width='stretch', key="us_trend_analysis_btn"):

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
                    tickers, chinese_name
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

                    # 顯示乾淨的表格，不使用顏色編碼
                    st.dataframe(df_display, width='stretch', height=600)

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
                        file_name=f"美股11大類股趨勢分析_{latest_date}.xlsx",
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
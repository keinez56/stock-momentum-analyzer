import streamlit as st
import pandas as pd
import numpy as np
import talib
import yfinance as yf
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os
import time
from io import BytesIO
from US_momentum import process_us_stock_data, calculate_us_technical_indicators

warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(
    page_title="股市動能分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f4e79;
        color: #1f4e79;
        font-weight: 500;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        color: #856404;
        font-weight: 500;
    }
    .sidebar-info {
        background-color: #1f4e79;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: 500;
        border: 2px solid #4a90e2;
    }
</style>
""", unsafe_allow_html=True)

# 複製優化後的函數
def safe_get_value(series: pd.Series, index: int = -1) -> float:
    """安全獲取數值，避免 .values[0] 錯誤"""
    try:
        if len(series) == 0:
            return np.nan
        value = series.iloc[index]
        # 確保返回純數值而非 pandas Series
        if hasattr(value, 'item'):
            return float(value.item())
        elif hasattr(value, 'values'):
            return float(value.values[0]) if len(value.values) > 0 else np.nan
        else:
            return float(value)
    except (IndexError, AttributeError, TypeError, ValueError):
        return np.nan

def classify_stock_code(stock_code: str) -> str:
    """將台股數字代碼轉為 yfinance 可用格式"""
    stock_code_tw = f"{stock_code}.TW"
    try:
        data_tw = yf.download(stock_code_tw, start='2024-01-01', end='2025-01-01', progress=False)
        if not data_tw.empty:
            return stock_code_tw
    except Exception:
        pass
    return f"{stock_code}.TWO"

def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """計算所有技術指標"""
    if df.empty or len(df) < 60:
        return {}

    close_array = np.ravel(df['Close'].to_numpy())
    high_array = np.ravel(df['High'].to_numpy())
    low_array = np.ravel(df['Low'].to_numpy())

    indicators = {}

    # 基本價格資料
    indicators['close'] = safe_get_value(df['Close'])
    # 修正 higher_high 計算：近5日最高價是否創一年新高
    try:
        recent_5_max = float(df['Close'].iloc[-5:].max())
        year_max_before_5 = float(df['Close'].iloc[:-5].max()) if len(df) > 5 else 0.0
        indicators['higher_high'] = bool(recent_5_max > year_max_before_5)
    except:
        indicators['higher_high'] = False

    # 成交量變化 - 重寫計算邏輯
    try:
        # 確保有足夠的數據
        if len(df) >= 20:
            # 獲取最新成交量
            volume_series = df['Volume'].dropna()
            if len(volume_series) >= 20:
                last_volume = float(volume_series.iloc[-1])
                # 計算前20日成交量平均（不包含最新一日）
                vol_20_mean = float(volume_series.iloc[-21:-1].mean() if len(volume_series) >= 21 else volume_series.iloc[-20:].mean())

                if vol_20_mean > 0 and last_volume > 0:
                    vol_change = (last_volume / vol_20_mean - 1) * 100
                    indicators['volume_change'] = round(vol_change, 2)
                    indicators['vc_30'] = bool(vol_change > 30)
                    print(f"Debug - Volume calc: last={last_volume:.0f}, mean={vol_20_mean:.0f}, change={vol_change:.2f}%")
                else:
                    indicators['volume_change'] = 0.0
                    indicators['vc_30'] = False
                    print(f"Debug - Invalid volume data: last={last_volume}, mean={vol_20_mean}")
            else:
                indicators['volume_change'] = 0.0
                indicators['vc_30'] = False
                print("Debug - Not enough volume data")
        else:
            indicators['volume_change'] = 0.0
            indicators['vc_30'] = False
            print(f"Debug - DataFrame too small: {len(df)} days")
    except Exception as e:
        print(f"Volume calculation error: {e}")
        indicators['volume_change'] = 0.0
        indicators['vc_30'] = False

    # 報酬率
    try:
        day_ret = safe_get_value(df['Close'].pct_change()) * 100
        indicators['day_return'] = float(day_ret) if not np.isnan(day_ret) else 0.0
    except:
        indicators['day_return'] = 0.0

    try:
        if len(df) >= 5:
            week_ret = safe_get_value(df['Close'].pct_change(periods=5).dropna()) * 100
            indicators['week_return'] = float(week_ret) if not np.isnan(week_ret) else 0.0
        else:
            indicators['week_return'] = 0.0
    except:
        indicators['week_return'] = 0.0

    try:
        if len(df) >= 22:
            month_ret = safe_get_value(df['Close'].pct_change(periods=22).dropna()) * 100
            indicators['month_return'] = float(month_ret) if not np.isnan(month_ret) else 0.0
        else:
            indicators['month_return'] = 0.0
    except:
        indicators['month_return'] = 0.0

    # RSI 指標
    rsi5 = talib.RSI(close_array, timeperiod=5)
    rsi14 = talib.RSI(close_array, timeperiod=14)
    indicators['rsi5'] = rsi5[-1] if len(rsi5) > 0 else np.nan
    indicators['rsi14'] = rsi14[-1] if len(rsi14) > 0 else np.nan

    # MACD 指標
    macd, macdsignal, macdhist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['macd'] = macd[-1] if len(macd) > 0 else np.nan
    indicators['macdsignal'] = macdsignal[-1] if len(macdsignal) > 0 else np.nan
    indicators['macdhist'] = macdhist[-1] if len(macdhist) > 0 else np.nan
    indicators['macdhist_signal'] = (macdhist[-1] > 0 and macdhist[-2] < 0) if len(macdhist) >= 2 else False

    # 移動平均線
    ma5 = talib.SMA(close_array, timeperiod=5)
    ma20 = talib.SMA(close_array, timeperiod=20)
    ma60 = talib.SMA(close_array, timeperiod=60)
    indicators['ma5'] = ma5[-1] if len(ma5) > 0 else np.nan
    indicators['ma20'] = ma20[-1] if len(ma20) > 0 else np.nan
    indicators['ma60'] = ma60[-1] if len(ma60) > 0 else np.nan
    indicators['crossover'] = ((ma20[-2] - ma5[-2]) > 0 and (ma5[-1] - ma20[-1]) > 0) if len(ma5) >= 2 and len(ma20) >= 2 else False

    # 布林通道
    upperband, middleband, lowerband = talib.BBANDS(close_array, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    if len(upperband) >= 3:
        indicators['bband'] = ((upperband[-1] - lowerband[-1]) - (upperband[-2] - lowerband[-2])) > 0 and ((upperband[-2] - lowerband[-2]) - (upperband[-3] - lowerband[-3])) > 0
        indicators['bband_middleband'] = middleband[-1] - middleband[-2] > 0 if len(middleband) >= 2 else False
        last_close = safe_get_value(df['Close'], -1)
        prev_close = safe_get_value(df['Close'], -2)
        indicators['bband_crossover'] = lowerband[-1] < last_close and lowerband[-2] > prev_close if len(lowerband) >= 2 else False
    else:
        indicators['bband'] = False
        indicators['bband_middleband'] = False
        indicators['bband_crossover'] = False

    # 威廉指標
    willr = talib.WILLR(high_array, low_array, close_array, timeperiod=14)
    indicators['willr_d'] = willr[-1] if len(willr) >= 1 else np.nan
    indicators['willr_d1'] = willr[-2] if len(willr) >= 2 else np.nan

    return indicators

def prepare_stock_codes():
    """準備股票代碼"""
    try:
        if not os.path.exists("2024-換股.xlsx"):
            st.error("❌ 找不到 2024-換股.xlsx 檔案，請確認檔案是否存在")
            return None

        data = pd.read_excel("2024-換股.xlsx")
        tickers = data["股票代碼"]
        name = data["股票名稱"]

        # 應用分類函式
        classified_codes = tickers.apply(classify_stock_code)

        # 建立 DataFrame 並加上指數
        result_df = pd.DataFrame({
            "股票名稱": name,
            "原始代碼": tickers,
            "YFinance代碼": classified_codes
        })

        # 加上指數列
        index_df = pd.DataFrame({
            "股票名稱": ["加權指數", "櫃買指數"],
            "原始代碼": ["^TWII", "^TWOII"],
            "YFinance代碼": ["^TWII", "^TWOII"]
        })

        # 合併
        final_df = pd.concat([result_df, index_df], ignore_index=True)

        # 寫入 Excel
        final_df.to_excel("代碼.xlsx", index=False)
        return True
    except Exception as e:
        st.error(f"❌ 準備股票代碼時發生錯誤: {e}")
        return None

def process_stock_data(progress_bar, status_text):
    """處理股票數據並計算技術指標"""
    try:
        data = pd.read_excel("代碼.xlsx")
        tickers = data["YFinance代碼"]
        names = data["股票名稱"]
        today = date.today()
        start_day = today - timedelta(365)

        results = []
        total_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            # 更新進度條
            progress = (i + 1) / total_tickers
            progress_bar.progress(progress)
            status_text.text(f"正在處理 {ticker} ({i+1}/{total_tickers})")

            try:
                df = yf.download(ticker, start=start_day, end=today, auto_adjust=False, progress=False)

                if df.empty:
                    continue

                if len(df) < 60:
                    continue

                indicators = calculate_technical_indicators(df)

                if indicators:
                    result = {
                        'Ticker': ticker,
                        'Name': names.iloc[i] if i < len(names) else '',
                        'Close': indicators.get('close', np.nan),
                        'Daily_return': indicators.get('day_return', np.nan),
                        'Week_return': indicators.get('week_return', np.nan),
                        'Month_return': indicators.get('month_return', np.nan),
                        'HigherHigh': indicators.get('higher_high', False),
                        'VolumnChange': indicators.get('volume_change', np.nan),
                        'VC_30': indicators.get('vc_30', False),
                        'RSI_5': indicators.get('rsi5', np.nan),
                        'RSI_14': indicators.get('rsi14', np.nan),
                        'Macd': indicators.get('macd', np.nan),
                        'Macdsignal': indicators.get('macdsignal', np.nan),
                        'Macdhist': indicators.get('macdhist', np.nan),
                        'macdhist_signal': indicators.get('macdhist_signal', False),
                        'Ma5': indicators.get('ma5', np.nan),
                        'Ma20': indicators.get('ma20', np.nan),
                        'Ma60': indicators.get('ma60', np.nan),
                        'Crossover': indicators.get('crossover', False),
                        'BBand': indicators.get('bband', False),
                        'BBand_middleband': indicators.get('bband_middleband', False),
                        'BBand_crossover': indicators.get('bband_crossover', False),
                        'willr_D': indicators.get('willr_d', np.nan),
                        'willr_D1': indicators.get('willr_d1', np.nan)
                    }
                    results.append(result)

            except Exception as e:
                continue

        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"❌ 處理股票數據時發生錯誤: {e}")
        return None

def generate_excel_file():
    """生成最新的 Excel 檔案"""
    # 準備股票代碼
    if prepare_stock_codes() is None:
        return None

    # 創建進度條
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 處理股票數據
    dframe = process_stock_data(progress_bar, status_text)

    if dframe is not None and not dframe.empty:
        # 計算複合動能指標
        dframe['Composite_Momentum_s'] = (
            (dframe['RSI_5'] - 50) +
            (dframe['Macdhist'] - dframe['macdhist_signal'].astype(float)) +
            (dframe['Ma5'] - dframe['Ma20']) / dframe['Ma20'] * 100
        )
        dframe['Composite_Momentum_l'] = (
            (dframe['RSI_14'] - 50) +
            (dframe['Macdhist'] - dframe['macdhist_signal'].astype(float)) +
            (dframe['Ma20'] - dframe['Ma60']) / dframe['Ma60'] * 100
        )

        # 輸出到檔案
        filename = 'TW動能觀察.xlsx'
        try:
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                dframe.to_excel(writer, sheet_name='stock_1', index=False)

            # 清除進度條
            progress_bar.empty()
            status_text.empty()

            return filename, dframe
        except Exception as e:
            st.error(f"❌ 輸出檔案時發生錯誤: {e}")
            return None, None
    else:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ 沒有成功處理任何股票數據")
        return None, None

def generate_us_excel_file():
    """生成美股 Excel 檔案"""
    # 檢查美股代碼檔案
    if not os.path.exists("US_stocks.xlsx"):
        st.error("❌ 找不到 US_stocks.xlsx 檔案，請確認檔案是否存在")
        return None, None

    try:
        # 處理美股數據
        with st.spinner("正在處理美股數據..."):
            dframe = process_us_stock_data("US_stocks.xlsx")

        if dframe is not None and not dframe.empty:
            # 計算複合動能指標
            dframe['Composite_Momentum_s'] = (
                (dframe['RSI_5'] - 50) +
                (dframe['Macdhist'] - dframe['macdhist_signal'].astype(float)) +
                (dframe['Ma5'] - dframe['Ma20']) / dframe['Ma20'] * 100
            )
            dframe['Composite_Momentum_l'] = (
                (dframe['RSI_14'] - 50) +
                (dframe['Macdhist'] - dframe['macdhist_signal'].astype(float)) +
                (dframe['Ma20'] - dframe['Ma60']) / dframe['Ma60'] * 100
            )

            # 輸出到檔案
            filename = 'US動能觀察.xlsx'
            try:
                with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                    dframe.to_excel(writer, sheet_name='stock_1', index=False)

                return filename, dframe
            except Exception as e:
                st.error(f"❌ 輸出美股檔案時發生錯誤: {e}")
                return None, None
        else:
            st.error("❌ 沒有成功處理任何美股數據")
            return None, None
    except Exception as e:
        st.error(f"❌ 處理美股數據時發生錯誤: {e}")
        return None, None

# Streamlit 主介面
def main():
    st.markdown('<div class="main-header">📊 股市動能分析系統</div>', unsafe_allow_html=True)

    # 創建分頁
    tab1, tab2 = st.tabs(["🇹🇼 台股分析", "🇺🇸 美股分析"])

    # 側邊欄資訊
    with st.sidebar:
        st.markdown("### 📊 系統說明")
        st.markdown("""
        <div class="sidebar-info">
        <strong>功能特色：</strong><br>
        • 即時股票技術指標分析<br>
        • 動能複合指標計算<br>
        • Excel 格式報告下載<br>
        • 支援台股與美股市場
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 包含指標")
        st.markdown("""
        - RSI (5日/14日)
        - MACD 指標
        - 移動平均線 (5/20/60日)
        - 布林通道
        - 威廉指標
        - 成交量分析
        - 複合動能指標
        """)

    # 台股分析頁面
    with tab1:
        st.markdown("### 🇹🇼 台股動能分析")

        # 主要內容區域
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🚀 獲取最新台股動能分析報告")
            st.markdown("""
            <div class="info-box">
            點擊下方按鈕開始生成最新的台股動能分析報告。系統將自動下載最新股價數據，
            計算各項技術指標，並生成 Excel 格式的分析報告供您下載。
            </div>
            """, unsafe_allow_html=True)

            # 檢查檔案是否存在
            if os.path.exists('TW動能觀察.xlsx'):
                file_time = os.path.getmtime('TW動能觀察.xlsx')
                file_date = pd.Timestamp.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"""
                <div class="warning-box">
                <strong>📁 現有檔案：</strong> TW動能觀察.xlsx<br>
                <strong>📅 更新時間：</strong> {file_date}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📈 今日市場概況")
            today = date.today()
            st.markdown(f"**分析日期：** {today.strftime('%Y年%m月%d日')}")
            st.markdown("**市場狀態：** 📊 開放交易")
            st.markdown("**數據來源：** Yahoo Finance")

        st.markdown("---")

        # 生成台股報告按鈕
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 生成最新台股動能分析報告", type="primary", use_container_width=True):
                with st.spinner("正在生成最新台股報告，請稍候..."):
                    filename, dframe = generate_excel_file()

                    if filename and dframe is not None:
                        st.markdown("""
                        <div class="success-box">
                        ✅ <strong>台股報告生成成功！</strong><br>
                        已成功處理所有股票數據並計算技術指標
                        </div>
                        """, unsafe_allow_html=True)

                        # 顯示統計資訊
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("處理股票數", len(dframe))
                        with col2:
                            try:
                                strong_momentum = len(dframe[dframe['Composite_Momentum_s'] > 10])
                            except:
                                strong_momentum = 0
                            st.metric("強勢股票", strong_momentum)
                        with col3:
                            try:
                                high_rsi = len(dframe[dframe['RSI_14'] > 70])
                            except:
                                high_rsi = 0
                            st.metric("超買股票", high_rsi)
                        with col4:
                            try:
                                # 計算 VC_30 為 True 的數量
                                if 'VC_30' in dframe.columns and not dframe.empty:
                                    volume_surge = sum(dframe['VC_30'] == True)
                                else:
                                    volume_surge = 0
                            except:
                                volume_surge = 0
                            st.metric("量增股票", volume_surge)

                        # 提供下載按鈕
                        with open(filename, "rb") as file:
                            st.download_button(
                                label="📥 下載 TW動能觀察.xlsx",
                                data=file.read(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                        # 顯示數據預覽
                        st.markdown("### 📊 數據預覽")
                        st.dataframe(dframe.head(10), use_container_width=True)

    # 美股分析頁面
    with tab2:
        st.markdown("### 🇺🇸 美股動能分析")

        # 主要內容區域
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🚀 獲取最新美股動能分析報告")
            st.markdown("""
            <div class="info-box">
            點擊下方按鈕開始生成最新的美股動能分析報告。系統將自動下載最新股價數據，
            計算各項技術指標，並生成 Excel 格式的分析報告供您下載。
            </div>
            """, unsafe_allow_html=True)

            # 檢查檔案是否存在
            if os.path.exists('US動能觀察.xlsx'):
                file_time = os.path.getmtime('US動能觀察.xlsx')
                file_date = pd.Timestamp.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"""
                <div class="warning-box">
                <strong>📁 現有檔案：</strong> US動能觀察.xlsx<br>
                <strong>📅 更新時間：</strong> {file_date}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📈 美股市場概況")
            today = date.today()
            st.markdown(f"**分析日期：** {today.strftime('%Y年%m月%d日')}")
            st.markdown("**市場狀態：** 📊 開放交易")
            st.markdown("**數據來源：** Yahoo Finance")

        st.markdown("---")

        # 生成美股報告按鈕
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 生成最新美股動能分析報告", type="primary", use_container_width=True):
                filename, dframe = generate_us_excel_file()

                if filename and dframe is not None:
                    st.markdown("""
                    <div class="success-box">
                    ✅ <strong>美股報告生成成功！</strong><br>
                    已成功處理所有股票數據並計算技術指標
                    </div>
                    """, unsafe_allow_html=True)

                    # 顯示統計資訊
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("處理股票數", len(dframe))
                    with col2:
                        try:
                            strong_momentum = len(dframe[dframe['Composite_Momentum_s'] > 10])
                        except:
                            strong_momentum = 0
                        st.metric("強勢股票", strong_momentum)
                    with col3:
                        try:
                            high_rsi = len(dframe[dframe['RSI_14'] > 70])
                        except:
                            high_rsi = 0
                        st.metric("超買股票", high_rsi)
                    with col4:
                        try:
                            volume_surge = sum(dframe['VC_30'] == True)
                        except:
                            volume_surge = 0
                        st.metric("量增股票", volume_surge)

                    # 提供下載按鈕
                    with open(filename, "rb") as file:
                        st.download_button(
                            label="📥 下載 US動能觀察.xlsx",
                            data=file.read(),
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                    # 顯示數據預覽
                    st.markdown("### 📊 數據預覽")
                    st.dataframe(dframe.head(10), use_container_width=True)

    # 頁腳
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    🔧 股市動能分析系統 | 📈 技術指標即時計算 | 💼 台股美股雙重支援
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
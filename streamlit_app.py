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

# 帳號密碼設定
USERS = {
    "admin": "admin123",
    "vivian": "vivian123"
}

def check_login():
    """檢查登入狀態"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

def login_page():
    """登入頁面"""
    st.markdown('<div class="main-header">🔐 股市動能分析系統 - 用戶登入</div>', unsafe_allow_html=True)

    # 登入表單
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            st.markdown("### 📋 請輸入登入資訊")

            username = st.text_input("👤 使用者名稱", placeholder="請輸入使用者名稱")
            password = st.text_input("🔑 密碼", type="password", placeholder="請輸入密碼")

            login_button = st.form_submit_button("🚀 登入", use_container_width=True)

            if login_button:
                if username in USERS and USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ 登入成功！")
                    st.rerun()
                else:
                    st.error("❌ 帳號或密碼錯誤，請重新輸入！")

        # 顯示可用帳號提示（僅供測試使用）
        with st.expander("📝 測試帳號"):
            st.markdown("""
            **測試帳號 1:**
            - 使用者名稱: `admin`
            - 密碼: `admin123`

            **測試帳號 2:**
            - 使用者名稱: `vivian`
            - 密碼: `vivian123`
            """)

def logout():
    """登出功能"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

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

    # KD指標 (隨機指標)
    slowk, slowd = talib.STOCH(high_array, low_array, close_array, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    indicators['k5'] = slowk[-1] if len(slowk) >= 1 else np.nan
    indicators['d5'] = slowd[-1] if len(slowd) >= 1 else np.nan

    # 成交量5日平均
    try:
        if len(df) >= 5:
            volume_5_mean = float(df['Volume'].iloc[-5:].mean())
            current_volume = float(df['Volume'].iloc[-1])
            indicators['volume_5_mean'] = volume_5_mean
            indicators['volume_above_5ma'] = current_volume > volume_5_mean
        else:
            indicators['volume_5_mean'] = 0.0
            indicators['volume_above_5ma'] = False
    except Exception as e:
        print(f"計算5日成交量平均時發生錯誤: {e}")
        indicators['volume_5_mean'] = 0.0
        indicators['volume_above_5ma'] = False

    # 成交量20日平均
    try:
        if len(df) >= 20:
            volume_20_mean = float(df['Volume'].iloc[-20:].mean())
            current_volume = float(df['Volume'].iloc[-1])
            indicators['volume_20_mean'] = volume_20_mean
            indicators['volume_below_20ma'] = current_volume < volume_20_mean
        else:
            indicators['volume_20_mean'] = 0.0
            indicators['volume_below_20ma'] = False
    except Exception as e:
        print(f"計算20日成交量平均時發生錯誤: {e}")
        indicators['volume_20_mean'] = 0.0
        indicators['volume_below_20ma'] = False

    # 短線上漲動能指標 (5個條件全部滿足)
    try:
        condition1 = indicators.get('close', 0) > indicators.get('ma5', 0) if not np.isnan(indicators.get('close', np.nan)) and not np.isnan(indicators.get('ma5', np.nan)) else False
        condition2 = indicators.get('volume_above_5ma', False)
        condition3 = indicators.get('k5', 0) > indicators.get('d5', 0) if not np.isnan(indicators.get('k5', np.nan)) and not np.isnan(indicators.get('d5', np.nan)) else False
        condition4 = indicators.get('rsi14', 0) > 50 if not np.isnan(indicators.get('rsi14', np.nan)) else False
        condition5 = indicators.get('macdhist', 0) > 0 if not np.isnan(indicators.get('macdhist', np.nan)) else False

        indicators['short_uptrend_momentum'] = bool(condition1 and condition2 and condition3 and condition4 and condition5)

        # 調試資訊
        print(f"Debug - 短線上漲動能: close>{indicators.get('ma5', 0):.2f}={condition1}, vol_above_5ma={condition2}, K>{indicators.get('d5', 0):.2f}={condition3}, RSI>{indicators.get('rsi14', 0):.2f}>50={condition4}, MACD>{indicators.get('macdhist', 0):.4f}>0={condition5}, 結果={indicators['short_uptrend_momentum']}")

    except Exception as e:
        print(f"計算短線上漲動能時發生錯誤: {e}")
        indicators['short_uptrend_momentum'] = False

    # 短線下跌訊號指標 (4個條件全部滿足)
    try:
        condition1_down = indicators.get('close', 0) < indicators.get('ma5', 0) if not np.isnan(indicators.get('close', np.nan)) and not np.isnan(indicators.get('ma5', np.nan)) else False
        condition2_down = indicators.get('volume_below_20ma', False)
        condition3_down = indicators.get('k5', 0) < indicators.get('d5', 0) if not np.isnan(indicators.get('k5', np.nan)) and not np.isnan(indicators.get('d5', np.nan)) else False
        condition4_down = indicators.get('macdhist', 0) < 0 if not np.isnan(indicators.get('macdhist', np.nan)) else False

        indicators['short_downtrend_signal'] = bool(condition1_down and condition2_down and condition3_down and condition4_down)

        # 調試資訊
        print(f"Debug - 短線下跌訊號: close<{indicators.get('ma5', 0):.2f}={condition1_down}, vol_below_20ma={condition2_down}, K<{indicators.get('d5', 0):.2f}={condition3_down}, MACD<{indicators.get('macdhist', 0):.4f}<0={condition4_down}, 結果={indicators['short_downtrend_signal']}")

    except Exception as e:
        print(f"計算短線下跌訊號時發生錯誤: {e}")
        indicators['short_downtrend_signal'] = False

    # 機構出貨指標 (3個條件全部滿足)
    try:
        condition1_inst = indicators.get('close', 0) < indicators.get('ma20', 0) if not np.isnan(indicators.get('close', np.nan)) and not np.isnan(indicators.get('ma20', np.nan)) else False
        condition2_inst = indicators.get('volume_above_5ma', False)

        # 計算三日累積下跌幅度
        if len(df) >= 4:
            close_3days_ago = safe_get_value(df['Close'], -4)  # 4天前的收盤價 (包含今天共3天)
            current_close = safe_get_value(df['Close'], -1)   # 今天的收盤價
            if not np.isnan(close_3days_ago) and not np.isnan(current_close) and close_3days_ago > 0:
                decline_3days = ((close_3days_ago - current_close) / close_3days_ago) * 100
                condition3_inst = decline_3days > 5  # 下跌超過5%
                indicators['decline_3days'] = decline_3days
            else:
                condition3_inst = False
                indicators['decline_3days'] = 0
        else:
            condition3_inst = False
            indicators['decline_3days'] = 0

        indicators['institutional_selling'] = bool(condition1_inst and condition2_inst and condition3_inst)

        # 調試資訊
        print(f"Debug - 機構出貨指標: close<{indicators.get('ma20', 0):.2f}={condition1_inst}, vol_above_5ma={condition2_inst}, 3日跌幅{indicators.get('decline_3days', 0):.2f}%>5%={condition3_inst}, 結果={indicators['institutional_selling']}")

    except Exception as e:
        print(f"計算機構出貨指標時發生錯誤: {e}")
        indicators['institutional_selling'] = False
        indicators['decline_3days'] = 0

    return indicators

def prepare_stock_codes():
    """準備股票代碼"""
    try:
        # 台股代碼列表 (硬編碼)
        taiwan_stocks = {
            2330: "台灣積體電路製造",
            2308: "台達電子工業",
            3595: "山太士",
            3708: "上緯國際投資",
            2408: "南亞科技",
            1504: "東元電機",
            2317: "鴻海精密工業",
            2383: "台光電子材料",
            3665: "貿聯",
            2382: "廣達電腦",
            3231: "緯創資通",
            3163: "波若威科技",
            3363: "上詮光纖通信",
            1802: "台灣玻璃工業",
            1303: "南亞塑膠工業",
            2359: "所羅門",
            2328: "廣宇科技",
            6188: "廣明光電",
            2634: "漢翔航空工業",
            8033: "雷虎科技",
            2498: "宏達電",
            8358: "金居開發"
        }

        # 建立DataFrame
        tickers = list(taiwan_stocks.keys())
        names = list(taiwan_stocks.values())

        # 應用分類函式
        classified_codes = [classify_stock_code(ticker) for ticker in tickers]

        # 建立 DataFrame 並加上指數
        result_df = pd.DataFrame({
            "股票名稱": names,
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
                        'willr_D1': indicators.get('willr_d1', np.nan),
                        'K5': indicators.get('k5', np.nan),
                        'D5': indicators.get('d5', np.nan),
                        'Volume_5MA': indicators.get('volume_5_mean', np.nan),
                        'Volume_Above_5MA': indicators.get('volume_above_5ma', False),
                        'Volume_20MA': indicators.get('volume_20_mean', np.nan),
                        'Volume_Below_20MA': indicators.get('volume_below_20ma', False),
                        'Decline_3Days': indicators.get('decline_3days', 0),
                        'Short_Uptrend_Momentum': indicators.get('short_uptrend_momentum', False),
                        'Short_Downtrend_Signal': indicators.get('short_downtrend_signal', False),
                        'Institutional_Selling': indicators.get('institutional_selling', False)
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
    if not os.path.exists("2025-美股換股.xlsx"):
        st.warning("⚠️ 本地檔案模式：2025-美股換股.xlsx 不存在，請使用自訂檔案上傳功能")
        return None, None

    try:
        # 處理美股數據
        with st.spinner("正在處理美股數據..."):
            dframe = process_us_stock_data()

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

def process_custom_file(uploaded_file, progress_bar, status_text):
    """處理使用者上傳的檔案並計算技術指標"""
    try:
        # 讀取上傳的檔案
        data = pd.read_excel(uploaded_file)

        # 自動識別股票代碼欄位
        ticker_column = None
        name_column = None

        # 檢查各種可能的欄位名稱
        ticker_keywords = ['代碼', 'code', 'ticker', 'symbol', '股票代碼', 'stock_code', '證券代號', 'Ticker', 'Code', 'Symbol']
        name_keywords = ['名稱', 'name', '股票名稱', 'stock_name', '證券名稱', 'Name', '公司名稱', 'company']

        # 尋找股票代碼欄位
        for col in data.columns:
            for keyword in ticker_keywords:
                if keyword in str(col):
                    ticker_column = col
                    break
            if ticker_column:
                break

        # 尋找股票名稱欄位
        for col in data.columns:
            for keyword in name_keywords:
                if keyword in str(col):
                    name_column = col
                    break

        # 如果找不到特定欄位名，使用第一欄作為代碼，第二欄作為名稱
        if ticker_column is None:
            ticker_column = data.columns[0]
        if name_column is None and len(data.columns) > 1:
            name_column = data.columns[1]

        tickers = data[ticker_column].dropna()
        names = data[name_column].dropna() if name_column else pd.Series(['Unknown'] * len(tickers))

        # 開始處理股票數據
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
                # 清理股票代碼
                ticker = str(ticker).strip()
                if not ticker or ticker.lower() == 'nan':
                    continue

                # 智能判斷股票代碼格式並嘗試不同組合
                possible_tickers = []

                # 如果是純數字（4位數），很可能是台股代碼
                if ticker.isdigit() and len(ticker) == 4:
                    # 台股優先順序：先試 .TW（上市），再試 .TWO（上櫃）
                    possible_tickers = [f"{ticker}.TW", f"{ticker}.TWO"]
                    print(f"台股代碼檢測: {ticker} -> 嘗試 {possible_tickers}")

                # 如果是純數字但不是4位數，可能是其他市場
                elif ticker.isdigit():
                    possible_tickers = [ticker, f"{ticker}.TW", f"{ticker}.TWO"]

                # 如果包含字母且不包含點號，可能是美股代碼
                elif ticker.isalpha() and '.' not in ticker:
                    # 美股代碼直接使用，無需後綴
                    possible_tickers = [ticker.upper()]  # 美股代碼通常大寫
                    print(f"美股代碼檢測: {ticker} -> {possible_tickers}")

                # 如果已經包含交易所後綴，直接使用
                elif '.' in ticker:
                    possible_tickers = [ticker]
                    print(f"完整代碼檢測: {ticker}")

                # 其他情況，嘗試各種可能
                else:
                    possible_tickers = [ticker, ticker.upper(), f"{ticker}.TW", f"{ticker}.TWO"]

                df = None
                for test_ticker in possible_tickers:
                    try:
                        df = yf.download(test_ticker, start=start_day, end=today, auto_adjust=False, progress=False)
                        if not df.empty and len(df) >= 60:
                            ticker = test_ticker  # 使用成功的代碼
                            break
                    except:
                        continue

                if df is None or df.empty or len(df) < 60:
                    continue

                # 計算技術指標
                indicators = calculate_technical_indicators(df)

                if indicators:
                    result = {
                        'Ticker': ticker,
                        'Name': names.iloc[i] if i < len(names) else 'Unknown',
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
                        'willr_D1': indicators.get('willr_d1', np.nan),
                        'K5': indicators.get('k5', np.nan),
                        'D5': indicators.get('d5', np.nan),
                        'Volume_5MA': indicators.get('volume_5_mean', np.nan),
                        'Volume_Above_5MA': indicators.get('volume_above_5ma', False),
                        'Volume_20MA': indicators.get('volume_20_mean', np.nan),
                        'Volume_Below_20MA': indicators.get('volume_below_20ma', False),
                        'Decline_3Days': indicators.get('decline_3days', 0),
                        'Short_Uptrend_Momentum': indicators.get('short_uptrend_momentum', False),
                        'Short_Downtrend_Signal': indicators.get('short_downtrend_signal', False),
                        'Institutional_Selling': indicators.get('institutional_selling', False)
                    }
                    results.append(result)

            except Exception as e:
                continue

        return pd.DataFrame(results), ticker_column, name_column

    except Exception as e:
        st.error(f"❌ 處理上傳檔案時發生錯誤: {e}")
        return None, None, None

# Streamlit 主介面
def main():
    # 檢查登入狀態
    check_login()

    # 如果未登入，顯示登入頁面
    if not st.session_state.logged_in:
        login_page()
        return

    # 已登入，顯示主要內容
    st.markdown('<div class="main-header">📊 股市動能分析系統</div>', unsafe_allow_html=True)

    # 創建分頁
    tab1, tab2, tab3 = st.tabs(["🇹🇼 台股分析", "🇺🇸 美股分析", "📁 自訂檔案分析"])

    # 側邊欄資訊
    with st.sidebar:
        # 用戶資訊和登出按鈕
        st.markdown("### 👤 用戶資訊")
        st.markdown(f"""
        <div class="sidebar-info">
        <strong>歡迎回來：</strong>{st.session_state.username}<br>
        <strong>登入時間：</strong>{pd.Timestamp.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 登出", use_container_width=True):
            logout()

        st.markdown("---")
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
                        col1, col2, col3, col4, col5 = st.columns(5)
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
                        with col5:
                            try:
                                # 計算短線上漲動能為True的數量
                                if 'Short_Uptrend_Momentum' in dframe.columns and not dframe.empty:
                                    short_uptrend = sum(dframe['Short_Uptrend_Momentum'] == True)
                                else:
                                    short_uptrend = 0
                            except:
                                short_uptrend = 0
                            st.metric("短線上漲", short_uptrend)

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
            點擊下方按鈕開始生成最新的美股動能分析報告。系統將從 <strong>2025-美股換股.xlsx</strong>
            的C欄讀取美股代碼，自動下載最新股價數據，計算各項技術指標，並生成 Excel 格式的分析報告供您下載。
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
                    col1, col2, col3, col4, col5 = st.columns(5)
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
                    with col5:
                        try:
                            # 計算短線上漲動能為True的數量
                            if 'Short_Uptrend_Momentum' in dframe.columns and not dframe.empty:
                                short_uptrend = sum(dframe['Short_Uptrend_Momentum'] == True)
                            else:
                                short_uptrend = 0
                        except:
                            short_uptrend = 0
                        st.metric("短線上漲", short_uptrend)

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

    # 自訂檔案分析頁面
    with tab3:
        st.markdown("### 📁 自訂檔案動能分析")

        # 主要內容區域
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📤 上傳您的股票列表")
            st.markdown("""
            <div class="info-box">
            上傳包含股票代碼的Excel檔案，系統將<strong>智能識別股票代碼欄位</strong>並進行技術分析。<br><br>

            <strong>🎯 智能代碼識別：</strong><br>
            • 台股4位數代碼（如：2330）→ 自動嘗試 .TW/.TWO<br>
            • 美股字母代碼（如：AAPL）→ 直接使用<br>
            • 完整代碼（如：2330.TW）→ 直接使用<br>

            <strong>📋 支援欄位名稱：</strong><br>
            股票代碼、代碼、code、ticker、symbol、證券代號等
            </div>
            """, unsafe_allow_html=True)

            # 檔案上傳介面
            uploaded_file = st.file_uploader(
                "選擇Excel檔案",
                type=['xlsx', 'xls'],
                help="請上傳包含股票代碼的Excel檔案"
            )

            if uploaded_file is not None:
                try:
                    # 預覽上傳檔案的內容
                    preview_data = pd.read_excel(uploaded_file)
                    st.markdown("#### 📋 檔案預覽")
                    st.dataframe(preview_data.head(10), use_container_width=True)

                    # 顯示檔案資訊
                    st.markdown(f"**檔案名稱：** {uploaded_file.name}")
                    st.markdown(f"**總行數：** {len(preview_data)}")
                    st.markdown(f"**欄位數：** {len(preview_data.columns)}")
                    st.markdown(f"**檔案欄位：** {', '.join(preview_data.columns)}")

                except Exception as e:
                    st.error(f"❌ 檔案讀取錯誤: {e}")

        with col2:
            st.markdown("### 📈 分析設定")
            today = date.today()
            st.markdown(f"**分析日期：** {today.strftime('%Y年%m月%d日')}")
            st.markdown("**數據來源：** Yahoo Finance")
            st.markdown("**分析期間：** 近一年數據")

            # 檔案格式說明
            st.markdown("### 📝 智能識別規則")
            st.markdown("""
            **🏷️ 自動欄位識別：**
            - 代碼欄：股票代碼、代碼、code、ticker、symbol
            - 名稱欄：股票名稱、名稱、name、company

            **🎯 智能代碼轉換：**
            - **2330** → 嘗試 2330.TW → 2330.TWO
            - **AAPL** → 直接使用 AAPL
            - **2330.TW** → 直接使用 2330.TW

            **📊 分析結果：**
            - 同時包含三大技術指標
            - 自動統計各類股票數量
            - 提供分類顯示和完整報告
            """)

        st.markdown("---")

        # 分析按鈕和結果
        if uploaded_file is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 開始分析自訂股票列表", type="primary", use_container_width=True):

                    # 創建進度條
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner("正在分析您的股票列表，請稍候..."):
                        # 重置檔案指針到開頭
                        uploaded_file.seek(0)

                        # 處理自訂檔案
                        dframe, ticker_col, name_col = process_custom_file(uploaded_file, progress_bar, status_text)

                    # 清除進度條
                    progress_bar.empty()
                    status_text.empty()

                    if dframe is not None and not dframe.empty:
                        # 計算複合動能指標
                        try:
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
                        except:
                            pass

                        st.markdown(f"""
                        <div class="success-box">
                        ✅ <strong>自訂股票分析完成！</strong><br>
                        成功識別代碼欄位：<strong>{ticker_col}</strong><br>
                        成功識別名稱欄位：<strong>{name_col if name_col else '未找到'}</strong>
                        </div>
                        """, unsafe_allow_html=True)

                        # 顯示統計資訊
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("成功分析", len(dframe))
                        with col2:
                            try:
                                strong_momentum = len(dframe[dframe['Composite_Momentum_s'] > 10]) if 'Composite_Momentum_s' in dframe.columns else 0
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
                                volume_surge = sum(dframe['VC_30'] == True) if 'VC_30' in dframe.columns else 0
                            except:
                                volume_surge = 0
                            st.metric("量增股票", volume_surge)
                        with col5:
                            try:
                                short_uptrend = sum(dframe['Short_Uptrend_Momentum'] == True) if 'Short_Uptrend_Momentum' in dframe.columns else 0
                            except:
                                short_uptrend = 0
                            st.metric("短線上漲", short_uptrend)

                        # 生成下載檔案
                        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        filename = f'自訂股票動能分析_{timestamp}.xlsx'

                        try:
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                dframe.to_excel(writer, sheet_name='stock_analysis', index=False)

                            output.seek(0)

                            # 提供下載按鈕
                            st.download_button(
                                label="📥 下載分析結果",
                                data=output.read(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"❌ 生成下載檔案時發生錯誤: {e}")

                        # 顯示詳細分析結果
                        st.markdown("### 📊 詳細分析結果")

                        # 分類顯示
                        if 'Short_Uptrend_Momentum' in dframe.columns:
                            uptrend_stocks = dframe[dframe['Short_Uptrend_Momentum'] == True]
                            if not uptrend_stocks.empty:
                                st.markdown("#### 🚀 短線上漲動能強勁")
                                st.dataframe(uptrend_stocks[['Ticker', 'Name', 'Close', 'RSI_14', 'Macdhist', 'Ma5', 'Ma20']], use_container_width=True)

                        if 'Short_Downtrend_Signal' in dframe.columns:
                            downtrend_stocks = dframe[dframe['Short_Downtrend_Signal'] == True]
                            if not downtrend_stocks.empty:
                                st.markdown("#### 📉 短線下跌訊號")
                                st.dataframe(downtrend_stocks[['Ticker', 'Name', 'Close', 'RSI_14', 'K5', 'D5']], use_container_width=True)

                        if 'Institutional_Selling' in dframe.columns:
                            inst_selling_stocks = dframe[dframe['Institutional_Selling'] == True]
                            if not inst_selling_stocks.empty:
                                st.markdown("#### 🏛️ 機構出貨跡象")
                                st.dataframe(inst_selling_stocks[['Ticker', 'Name', 'Close', 'Ma20', 'Decline_3Days']], use_container_width=True)

                        # 完整數據預覽
                        st.markdown("### 📋 完整數據預覽")
                        st.dataframe(dframe, use_container_width=True)

                    else:
                        st.error("❌ 無法分析任何股票，請檢查檔案格式是否正確或股票代碼是否有效")

    # 頁腳
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    🔧 股市動能分析系統 | 📈 技術指標即時計算 | 💼 台股美股雙重支援 | 📁 自訂檔案分析
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
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
import sys

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import US_momentum
    import us_trend_scanner
    import us_market_scanner
    import institutional_data
    import revenue_scraper

    process_us_stock_data = US_momentum.process_us_stock_data
    calculate_us_technical_indicators = US_momentum.calculate_us_technical_indicators
    us_trend_scanner_main = us_trend_scanner.main
    us_market_scanner_main = us_market_scanner.main
    get_institutional_trading = institutional_data.get_institutional_trading
    get_institutional_trading_batch = institutional_data.get_institutional_trading_batch
    get_revenue_batch = revenue_scraper.get_revenue_batch
    get_revenue_finmind = revenue_scraper.get_revenue_finmind

except ImportError as e:
    st.error(f"模組導入錯誤: {e}")
    st.error(f"當前工作目錄: {os.getcwd()}")
    st.error(f"檔案所在目錄: {current_dir}")
    st.error(f"目錄內容: {os.listdir(current_dir)}")
    st.stop()

warnings.filterwarnings('ignore')

# 移除帳號密碼設定 - 開放所有使用者使用

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

def get_institutional_data(stock_code: str) -> Dict[str, float]:
    """獲取股票的三大法人買賣超資料"""
    try:
        # 獲取最近5個交易日的資料
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        # 轉換股票代碼格式（移除 .TW 或 .TWO 後綴）
        clean_code = stock_code.replace('.TW', '').replace('.TWO', '')

        df = get_institutional_trading(clean_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df.empty:
            return {
                'foreign_net': 0,
                'trust_net': 0,
                'dealer_net': 0,
                'total_net': 0
            }

        # 取最新一天的資料
        latest_data = df.iloc[-1]

        return {
            'foreign_net': float(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) if pd.notna(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) else 0,
            'trust_net': float(latest_data.get('投信買賣超股數', 0)) if pd.notna(latest_data.get('投信買賣超股數', 0)) else 0,
            'dealer_net': float(latest_data.get('自營商買賣超股數(自行買賣)', 0)) if pd.notna(latest_data.get('自營商買賣超股數(自行買賣)', 0)) else 0,
            'total_net': float(latest_data.get('三大法人買賣超股數', 0)) if pd.notna(latest_data.get('三大法人買賣超股數', 0)) else 0
        }

    except Exception as e:
        print(f"獲取 {stock_code} 三大法人資料時發生錯誤: {e}")
        return {
            'foreign_net': 0,
            'trust_net': 0,
            'dealer_net': 0,
            'total_net': 0
        }

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

    # 注意：all_time_high 在 process_stock_data 中單獨計算（需要10年資料）

    # 52週最高價、最低價及相對位置
    try:
        current_close = float(df['Close'].iloc[-1])
        week_52_high = float(df['High'].max())  # 52週最高價
        week_52_low = float(df['Low'].min())    # 52週最低價
        indicators['week_52_high'] = week_52_high
        indicators['week_52_low'] = week_52_low
        # 距離52週最高價差幾% (負數表示低於最高價)
        if week_52_high > 0:
            indicators['pct_from_52_high'] = round(((current_close - week_52_high) / week_52_high) * 100, 2)
        else:
            indicators['pct_from_52_high'] = 0.0
        # 距離52週最低價高幾% (正數表示高於最低價)
        if week_52_low > 0:
            indicators['pct_from_52_low'] = round(((current_close - week_52_low) / week_52_low) * 100, 2)
        else:
            indicators['pct_from_52_low'] = 0.0
    except:
        indicators['week_52_high'] = np.nan
        indicators['week_52_low'] = np.nan
        indicators['pct_from_52_high'] = np.nan
        indicators['pct_from_52_low'] = np.nan

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

    # YTD 報酬率 (年初至今報酬率)
    try:
        current_year = date.today().year
        # 找出今年第一個交易日的收盤價
        df_ytd = df[df.index >= f'{current_year}-01-01']
        if len(df_ytd) >= 2:
            first_close = float(df_ytd['Close'].iloc[0])
            current_close = float(df_ytd['Close'].iloc[-1])
            if first_close > 0:
                ytd_ret = ((current_close - first_close) / first_close) * 100
                indicators['ytd_return'] = round(ytd_ret, 2)
            else:
                indicators['ytd_return'] = 0.0
        else:
            indicators['ytd_return'] = 0.0
    except:
        indicators['ytd_return'] = 0.0

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
            8299: "群聯電子",
            2408: "南亞科技",
            2344: "華邦電子",
            2454: "聯發科技",
            6770: "力積電",
            3260: "威剛科技",
            2330: "台灣積體電路製造",
            6239: "力成科技",
            7769: "宏矽科技",
            8996: "高力熱處理",
            2308: "台達電子工業",
            1519: "華城電機",
            1504: "東元電機",
            2313: "華通電腦",
            3491: "昇達科技",
            8046: "南亞電路板",
            1303: "南亞塑膠工業",
            1802: "台灣玻璃工業",
            1717: "長興材料",
            8422: "可寧衛",
            6806: "森崴能源",
            1319: "東陽實業",
            6275: "元山科技",
            5452: "佶優科技",
            2241: "艾姆勒車電",
            2317: "鴻海精密工業",
            8431: "匯鑽科技",
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

        # 批量下載三大法人資料（使用智能日期選擇）
        status_text.text("正在批量下載三大法人資料...")
        progress_bar.progress(0.05)

        # 準備台股代碼列表（移除 .TW/.TWO 後綴）
        taiwan_stock_codes = []
        for ticker in tickers:
            clean_code = ticker.replace('.TW', '').replace('.TWO', '')
            if clean_code.isdigit() and len(clean_code) == 4:
                taiwan_stock_codes.append(clean_code)

        # 批量下載三大法人資料（使用智能日期選擇）
        institutional_batch_data = {}
        if taiwan_stock_codes:
            try:
                from institutional_data import get_institutional_trading_batch, get_trading_date_for_stock_data

                # 嘗試多個日期獲取資料
                retry_count = 0
                max_retries = 5
                while retry_count < max_retries and not institutional_batch_data:
                    try:
                        # 使用智能日期選擇，不指定特定日期
                        institutional_batch_data = get_institutional_trading_batch(taiwan_stock_codes)
                        if institutional_batch_data:
                            status_text.text(f"成功下載 {len(institutional_batch_data)} 檔股票的三大法人資料")
                            break
                        else:
                            retry_count += 1
                            status_text.text(f"嘗試取得三大法人資料 ({retry_count}/{max_retries})...")
                    except Exception as retry_e:
                        retry_count += 1
                        status_text.text(f"重試 {retry_count}/{max_retries}: {str(retry_e)[:50]}...")

                if institutional_batch_data:
                    # 同步調整股價資料的日期範圍
                    stock_end_date = get_trading_date_for_stock_data()
                    start_day = stock_end_date - timedelta(365)
                    status_text.text(f"股價資料期間: {start_day.strftime('%Y-%m-%d')} 至 {stock_end_date.strftime('%Y-%m-%d')}")
                else:
                    st.warning("⚠️ 無法取得三大法人資料，可能是非交易日或資料尚未公布")
                    stock_end_date = today
                    start_day = today - timedelta(365)

            except Exception as e:
                st.warning(f"批量下載三大法人資料失敗: {e}")
                institutional_batch_data = {}
                # 保持原來的日期範圍
                stock_end_date = today
                start_day = today - timedelta(365)

        # 批量下載營收資料
        revenue_batch_data = {}
        if taiwan_stock_codes:
            try:
                status_text.text("正在下載營收資料...")
                progress_bar.progress(0.1)
                revenue_batch_data = get_revenue_batch(taiwan_stock_codes)
                if revenue_batch_data:
                    status_text.text(f"成功下載 {len(revenue_batch_data)} 檔股票的營收資料")
            except Exception as e:
                st.warning(f"下載營收資料失敗: {e}")
                revenue_batch_data = {}

        for i, ticker in enumerate(tickers):
            # 更新進度條
            progress = (i + 1) / total_tickers
            progress_bar.progress(progress)
            status_text.text(f"正在處理 {ticker} ({i+1}/{total_tickers})")

            try:
                df = yf.download(ticker, start=start_day, end=stock_end_date, auto_adjust=False, progress=False)

                if df.empty:
                    continue

                if len(df) < 60:
                    continue

                indicators = calculate_technical_indicators(df)

                # 計算十年歷史新高 (All_Time_High)
                try:
                    ten_year_start = stock_end_date - timedelta(days=365*10)
                    df_10yr = yf.download(ticker, start=ten_year_start, end=stock_end_date, auto_adjust=False, progress=False)
                    if not df_10yr.empty:
                        current_close = float(df['Close'].iloc[-1])
                        ten_year_max = float(df_10yr['Close'].max())
                        # 允許小誤差（0.01%）來判斷是否相等
                        indicators['all_time_high'] = bool(current_close >= ten_year_max * 0.9999)
                    else:
                        indicators['all_time_high'] = False
                except:
                    indicators['all_time_high'] = False

                # 獲取基本面資料 (EPS, P/E, ROE)
                fundamental_data = {'eps': np.nan, 'pe': np.nan, 'roe': np.nan}
                try:
                    stock_info = yf.Ticker(ticker).info
                    if stock_info:
                        fundamental_data['eps'] = stock_info.get('trailingEps', np.nan)
                        fundamental_data['pe'] = stock_info.get('trailingPE', np.nan)
                        roe_value = stock_info.get('returnOnEquity', np.nan)
                        if roe_value is not None and not np.isnan(roe_value):
                            fundamental_data['roe'] = round(roe_value * 100, 2)  # 轉為百分比
                except Exception as e:
                    print(f"獲取 {ticker} 基本面資料失敗: {e}")

                # 獲取三大法人買賣超資料（從批量下載的資料中取得）
                clean_code = ticker.replace('.TW', '').replace('.TWO', '')
                institutional_data = {'foreign_net': 0, 'trust_net': 0, 'dealer_net': 0, 'total_net': 0}

                if clean_code in institutional_batch_data:
                    batch_data = institutional_batch_data[clean_code]
                    if not batch_data.empty:
                        latest_data = batch_data.iloc[-1]
                        institutional_data = {
                            'foreign_net': float(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) if pd.notna(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) else 0,
                            'trust_net': float(latest_data.get('投信買賣超股數', 0)) if pd.notna(latest_data.get('投信買賣超股數', 0)) else 0,
                            'dealer_net': float(latest_data.get('自營商買賣超股數(自行買賣)', 0)) if pd.notna(latest_data.get('自營商買賣超股數(自行買賣)', 0)) else 0,
                            'total_net': float(latest_data.get('三大法人買賣超股數', 0)) if pd.notna(latest_data.get('三大法人買賣超股數', 0)) else 0
                        }

                if indicators:
                    # 獲取營收資料
                    revenue_data = {'latest_month': '', 'latest_revenue_billion': np.nan, 'is_new_high': False}
                    if clean_code in revenue_batch_data:
                        rev = revenue_batch_data[clean_code]
                        revenue_data = {
                            'latest_month': rev.get('latest_month', ''),
                            'latest_revenue_billion': rev.get('latest_revenue_billion', np.nan),
                            'is_new_high': rev.get('is_new_high', False)
                        }

                    result = {
                        'Ticker': ticker,
                        'Name': names.iloc[i] if i < len(names) else '',
                        'Close': indicators.get('close', np.nan),
                        'Daily_return': indicators.get('day_return', np.nan),
                        'Week_return': indicators.get('week_return', np.nan),
                        'Month_return': indicators.get('month_return', np.nan),
                        'YTD_Return': indicators.get('ytd_return', np.nan),
                        'HigherHigh': indicators.get('higher_high', False),
                        'All_Time_High': indicators.get('all_time_high', False),
                        'Week_52_High': indicators.get('week_52_high', np.nan),
                        'Week_52_Low': indicators.get('week_52_low', np.nan),
                        'Pct_From_52_High': indicators.get('pct_from_52_high', np.nan),
                        'Pct_From_52_Low': indicators.get('pct_from_52_low', np.nan),
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
                        'Institutional_Selling': indicators.get('institutional_selling', False),
                        # 新增三大法人買賣超欄位
                        'Foreign_Net': institutional_data.get('foreign_net', 0),
                        'Trust_Net': institutional_data.get('trust_net', 0),
                        'Dealer_Net': institutional_data.get('dealer_net', 0),
                        'Total_Net': institutional_data.get('total_net', 0),
                        # 新增營收欄位
                        'Revenue_Month': revenue_data.get('latest_month', ''),
                        'Revenue_Billion': revenue_data.get('latest_revenue_billion', np.nan),
                        'Revenue_New_High': revenue_data.get('is_new_high', False),
                        # 新增基本面欄位
                        'EPS': fundamental_data.get('eps', np.nan),
                        'PE': fundamental_data.get('pe', np.nan),
                        'ROE': fundamental_data.get('roe', np.nan)
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

def process_us_stock_data_with_progress(progress_bar, status_text):
    """處理美股數據並顯示進度條"""
    try:
        status_text.text("正在處理美股數據...")
        progress_bar.progress(0.1)

        # 直接使用 US_momentum.py 中的函數
        dframe = process_us_stock_data()

        if dframe is not None and not dframe.empty:
            progress_bar.progress(1.0)
            status_text.text(f"成功處理 {len(dframe)} 檔美股數據")
            # 清除進度條
            progress_bar.empty()
            status_text.empty()
            return dframe
        else:
            st.error("❌ US_momentum.process_us_stock_data 返回空數據")
            progress_bar.empty()
            status_text.empty()
            return None

    except Exception as e:
        st.error(f"❌ 處理美股數據時發生錯誤: {e}")
        progress_bar.empty()
        status_text.empty()
        return None

def generate_us_excel_file():
    """生成美股 Excel 檔案"""
    try:
        # 創建進度條
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 處理美股數據
        dframe = process_us_stock_data_with_progress(progress_bar, status_text)

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
        st.write(f"📄 正在讀取檔案: {uploaded_file.name}")
        data = pd.read_excel(uploaded_file)
        st.write(f"✅ 成功讀取檔案，共 {len(data)} 行，欄位: {data.columns.tolist()}")

        # 自動識別股票代碼欄位
        ticker_column = None

        # 檢查各種可能的欄位名稱
        ticker_keywords = ['代碼', 'code', 'ticker', 'symbol', '股票代碼', 'stock_code', '證券代號', 'Ticker', 'Code', 'Symbol', '股票代码']

        # 尋找股票代碼欄位
        for col in data.columns:
            for keyword in ticker_keywords:
                if keyword in str(col):
                    ticker_column = col
                    break
            if ticker_column:
                break

        # 如果找不到特定欄位名，使用第一欄作為代碼
        if ticker_column is None:
            ticker_column = data.columns[0]
            st.write(f"⚠️ 未找到明確的代碼欄位，使用第一欄: {ticker_column}")
        else:
            st.write(f"✅ 識別到代碼欄位: {ticker_column}")

        tickers = data[ticker_column].dropna()

        st.write(f"📊 找到 {len(tickers)} 個股票代碼")

        # 開始處理股票數據
        today = date.today()
        start_day = today - timedelta(365)
        results = []
        total_tickers = len(tickers)

        # 批量下載三大法人資料（使用智能日期選擇）
        status_text.text("正在批量下載三大法人資料...")
        progress_bar.progress(0.05)

        # 準備台股代碼列表
        taiwan_stock_codes = []
        for ticker in tickers:
            ticker_str = str(ticker).strip()
            if ticker_str.isdigit() and len(ticker_str) == 4:
                taiwan_stock_codes.append(ticker_str)

        # 批量下載三大法人資料（使用智能日期選擇）
        institutional_batch_data = {}
        stock_end_date = today  # 預設使用今天，避免變數未定義

        if taiwan_stock_codes:
            try:
                from institutional_data import get_institutional_trading_batch, get_trading_date_for_stock_data
                # 使用智能日期選擇，不指定特定日期
                institutional_batch_data = get_institutional_trading_batch(taiwan_stock_codes)
                status_text.text(f"成功下載 {len(institutional_batch_data)} 檔股票的三大法人資料")

                # 同步調整股價資料的日期範圍
                stock_end_date = get_trading_date_for_stock_data()
                start_day = stock_end_date - timedelta(365)
                status_text.text(f"股價資料期間: {start_day.strftime('%Y-%m-%d')} 至 {stock_end_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                st.warning(f"批量下載三大法人資料失敗，將使用預設值: {e}")
                institutional_batch_data = {}
                # 保持原來的日期範圍
                stock_end_date = today
                start_day = today - timedelta(365)
        else:
            # 純美股或其他非台股列表，不需要三大法人資料
            st.write("📊 檢測到非台股代碼列表，跳過三大法人資料下載")
            stock_end_date = today

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
                download_success = False
                for test_ticker in possible_tickers:
                    try:
                        print(f"嘗試下載 {test_ticker}...")
                        df = yf.download(test_ticker, start=start_day, end=stock_end_date, auto_adjust=False, progress=False)
                        if not df.empty and len(df) >= 60:
                            ticker = test_ticker  # 使用成功的代碼
                            download_success = True
                            print(f"✅ 成功下載 {test_ticker}，共 {len(df)} 筆數據")
                            break
                        else:
                            print(f"⚠️ {test_ticker} 數據不足: {len(df)} 筆")
                    except Exception as e:
                        print(f"❌ 下載 {test_ticker} 失敗: {e}")
                        continue

                if df is None or df.empty or len(df) < 60:
                    print(f"⚠️ 跳過 {ticker}: 無法獲取足夠數據")
                    continue

                # 計算技術指標
                indicators = calculate_technical_indicators(df)

                # 獲取基本面資料 (EPS, P/E, ROE) 和營收資料
                fundamental_data = {'eps': np.nan, 'pe': np.nan, 'roe': np.nan}
                revenue_data = {'latest_period': '', 'latest_revenue_billion': np.nan, 'is_new_high': False}
                try:
                    ticker_obj = yf.Ticker(ticker)
                    stock_info = ticker_obj.info
                    if stock_info:
                        fundamental_data['eps'] = stock_info.get('trailingEps', np.nan)
                        fundamental_data['pe'] = stock_info.get('trailingPE', np.nan)
                        roe_value = stock_info.get('returnOnEquity', np.nan)
                        if roe_value is not None and not np.isnan(roe_value):
                            fundamental_data['roe'] = round(roe_value * 100, 2)  # 轉為百分比

                    # 判斷是台股還是美股來獲取營收資料
                    is_taiwan_stock = '.TW' in ticker or '.TWO' in ticker
                    if is_taiwan_stock:
                        # 台股使用 FinMind API 獲取月營收
                        clean_code = ticker.replace('.TW', '').replace('.TWO', '')
                        try:
                            rev_result = get_revenue_finmind(clean_code)
                            if rev_result:
                                revenue_data = {
                                    'latest_period': rev_result.get('latest_month', ''),
                                    'latest_revenue_billion': rev_result.get('latest_revenue_billion', np.nan),
                                    'is_new_high': rev_result.get('is_new_high', False)
                                }
                        except Exception as e:
                            print(f"獲取 {ticker} 台股營收資料失敗: {e}")
                    else:
                        # 美股使用 yfinance 獲取季度營收
                        quarterly_financials = ticker_obj.quarterly_financials
                        if quarterly_financials is not None and not quarterly_financials.empty:
                            revenue_row = None
                            for idx in quarterly_financials.index:
                                if 'Total Revenue' in str(idx) or 'Revenue' == str(idx):
                                    revenue_row = idx
                                    break
                            if revenue_row is not None:
                                revenues = quarterly_financials.loc[revenue_row].dropna()
                                if len(revenues) > 0:
                                    latest_revenue = float(revenues.iloc[0])
                                    quarter_month = revenues.index[0].month
                                    quarter_num = (quarter_month - 1) // 3 + 1
                                    latest_quarter = f"{revenues.index[0].year}/Q{quarter_num}"
                                    revenue_data['latest_period'] = latest_quarter
                                    revenue_data['latest_revenue_billion'] = round(latest_revenue / 1000000000, 2)
                                    if len(revenues) > 1:
                                        historical_max = float(revenues.iloc[1:].max())
                                        revenue_data['is_new_high'] = latest_revenue > historical_max
                except Exception as e:
                    print(f"獲取 {ticker} 基本面資料失敗: {e}")

                # 獲取三大法人買賣超資料（從批量下載的資料中取得）
                institutional_data = {'foreign_net': 0, 'trust_net': 0, 'dealer_net': 0, 'total_net': 0}
                if '.TW' in ticker or '.TWO' in ticker:
                    clean_code = ticker.replace('.TW', '').replace('.TWO', '')
                    if clean_code in institutional_batch_data:
                        batch_data = institutional_batch_data[clean_code]
                        if not batch_data.empty:
                            latest_data = batch_data.iloc[-1]
                            institutional_data = {
                                'foreign_net': float(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) if pd.notna(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) else 0,
                                'trust_net': float(latest_data.get('投信買賣超股數', 0)) if pd.notna(latest_data.get('投信買賣超股數', 0)) else 0,
                                'dealer_net': float(latest_data.get('自營商買賣超股數(自行買賣)', 0)) if pd.notna(latest_data.get('自營商買賣超股數(自行買賣)', 0)) else 0,
                                'total_net': float(latest_data.get('三大法人買賣超股數', 0)) if pd.notna(latest_data.get('三大法人買賣超股數', 0)) else 0
                            }
                elif ticker.isdigit() and len(ticker) == 4:
                    if ticker in institutional_batch_data:
                        batch_data = institutional_batch_data[ticker]
                        if not batch_data.empty:
                            latest_data = batch_data.iloc[-1]
                            institutional_data = {
                                'foreign_net': float(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) if pd.notna(latest_data.get('外陸資買賣超股數(不含外資自營商)', 0)) else 0,
                                'trust_net': float(latest_data.get('投信買賣超股數', 0)) if pd.notna(latest_data.get('投信買賣超股數', 0)) else 0,
                                'dealer_net': float(latest_data.get('自營商買賣超股數(自行買賣)', 0)) if pd.notna(latest_data.get('自營商買賣超股數(自行買賣)', 0)) else 0,
                                'total_net': float(latest_data.get('三大法人買賣超股數', 0)) if pd.notna(latest_data.get('三大法人買賣超股數', 0)) else 0
                            }

                if indicators:
                    result = {
                        'Ticker': ticker,
                        'Close': indicators.get('close', np.nan),
                        'Daily_return': indicators.get('day_return', np.nan),
                        'Week_return': indicators.get('week_return', np.nan),
                        'Month_return': indicators.get('month_return', np.nan),
                        'YTD_Return': indicators.get('ytd_return', np.nan),
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
                        'Institutional_Selling': indicators.get('institutional_selling', False),
                        # 新增三大法人買賣超欄位
                        'Foreign_Net': institutional_data.get('foreign_net', 0),
                        'Trust_Net': institutional_data.get('trust_net', 0),
                        'Dealer_Net': institutional_data.get('dealer_net', 0),
                        'Total_Net': institutional_data.get('total_net', 0),
                        # 新增營收欄位
                        'Revenue_Period': revenue_data.get('latest_period', ''),
                        'Revenue_Billion': revenue_data.get('latest_revenue_billion', np.nan),
                        'Revenue_New_High': revenue_data.get('is_new_high', False),
                        # 新增基本面欄位
                        'EPS': fundamental_data.get('eps', np.nan),
                        'PE': fundamental_data.get('pe', np.nan),
                        'ROE': fundamental_data.get('roe', np.nan)
                    }
                    results.append(result)

            except Exception as e:
                print(f"❌ 處理股票 {ticker} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                continue

        st.write(f"✅ 成功處理 {len(results)} 檔股票")
        return pd.DataFrame(results), ticker_column

    except Exception as e:
        st.error(f"❌ 處理上傳檔案時發生錯誤: {e}")
        import traceback
        st.error(f"詳細錯誤: {traceback.format_exc()}")
        return None, None

# Streamlit 主介面
def main():
    # 直接顯示主要內容，不需要登入驗證
    st.markdown('<div class="main-header">📊 股市動能分析系統</div>', unsafe_allow_html=True)

    # 創建分頁
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🇹🇼 台股分析", "🇺🇸 美股分析", "📁 自訂檔案分析", "🔍 美股趨勢掃描", "📊 美股大盤掃描", "📖 指標說明"])

    # 側邊欄資訊
    with st.sidebar:
        st.markdown("### 📊 BBM-RTI 動能分析系統")
        st.markdown("""
        <div class="sidebar-info">
        <strong>系統特色：</strong><br>
        • 全球總經趨勢追蹤<br>
        • 價值＋動能雙重篩選<br>
        • 即時技術指標分析<br>
        • Excel 報告下載<br>
        • 台股美股雙重支援
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 七大技術指標")
        st.markdown("""
        - RSI 相對強弱 (5日/14日)
        - MACD 動能趨勢
        - 移動平均線 (5/20/60日)
        - 布林通道
        - 威廉指標 %R
        - 成交量分析
        - 複合動能指標
        """)

        st.markdown("### ⚠️ 免責聲明")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #888;">
        指標為趨勢與動能的量化描述，非預測工具。
        投資需考量基本面與消息面，本系統結果不構成投資建議。
        股市波動大、風險高，投資人應自行承擔風險與盈虧。
        </div>
        """, unsafe_allow_html=True)

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
            if st.button("🔄 生成最新台股動能分析報告", type="primary", width='stretch'):
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
                        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
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
                        with col6:
                            try:
                                # 計算收盤創歷史新高的數量
                                if 'All_Time_High' in dframe.columns and not dframe.empty:
                                    all_time_high_count = sum(dframe['All_Time_High'] == True)
                                else:
                                    all_time_high_count = 0
                            except:
                                all_time_high_count = 0
                            st.metric("收盤創新高", all_time_high_count)
                        with col7:
                            try:
                                # 計算營收創新高的數量
                                if 'Revenue_New_High' in dframe.columns and not dframe.empty:
                                    revenue_new_high = sum(dframe['Revenue_New_High'] == True)
                                else:
                                    revenue_new_high = 0
                            except:
                                revenue_new_high = 0
                            st.metric("營收創新高", revenue_new_high)

                        # 提供下載按鈕
                        with open(filename, "rb") as file:
                            st.download_button(
                                label="📥 下載 TW動能觀察.xlsx",
                                data=file.read(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                width='stretch'
                            )

                        # 顯示數據預覽
                        st.markdown("### 📊 數據預覽")
                        st.dataframe(dframe.head(10), width='stretch')

    # 美股分析頁面
    with tab2:
        st.markdown("### 🇺🇸 美股動能分析")

        # 主要內容區域
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🚀 獲取最新美股動能分析報告")
            st.markdown("""
            <div class="info-box">
            點擊下方按鈕開始生成最新的美股動能分析報告。系統將分析內建的美股代碼列表，
            自動下載最新股價數據，計算各項技術指標，並生成 Excel 格式的分析報告供您下載。
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
            if st.button("🔄 生成最新美股動能分析報告", type="primary", width='stretch'):
                filename, dframe = generate_us_excel_file()

                if filename and dframe is not None:
                    st.markdown("""
                    <div class="success-box">
                    ✅ <strong>美股報告生成成功！</strong><br>
                    已成功處理所有股票數據並計算技術指標
                    </div>
                    """, unsafe_allow_html=True)

                    # 顯示統計資訊
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
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
                    with col6:
                        try:
                            # 計算收盤創歷史新高的數量
                            if 'All_Time_High' in dframe.columns and not dframe.empty:
                                all_time_high_count = sum(dframe['All_Time_High'] == True)
                            else:
                                all_time_high_count = 0
                        except:
                            all_time_high_count = 0
                        st.metric("收盤創新高", all_time_high_count)

                    # 提供下載按鈕
                    with open(filename, "rb") as file:
                        st.download_button(
                            label="📥 下載 US動能觀察.xlsx",
                            data=file.read(),
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width='stretch'
                        )

                    # 顯示數據預覽
                    st.markdown("### 📊 數據預覽")
                    st.dataframe(dframe.head(10), width='stretch')

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
                    st.dataframe(preview_data.head(10), width='stretch')

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
                if st.button("🚀 開始分析自訂股票列表", type="primary", width='stretch'):

                    # 創建進度條
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with st.spinner("正在分析您的股票列表，請稍候..."):
                        # 重置檔案指針到開頭
                        uploaded_file.seek(0)

                        # 處理自訂檔案
                        dframe, ticker_col = process_custom_file(uploaded_file, progress_bar, status_text)

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
                        成功識別代碼欄位：<strong>{ticker_col}</strong>
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
                                width='stretch'
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
                                st.dataframe(uptrend_stocks[['Ticker', 'Close', 'RSI_14', 'Macdhist', 'Ma5', 'Ma20']], width='stretch')

                        if 'Short_Downtrend_Signal' in dframe.columns:
                            downtrend_stocks = dframe[dframe['Short_Downtrend_Signal'] == True]
                            if not downtrend_stocks.empty:
                                st.markdown("#### 📉 短線下跌訊號")
                                st.dataframe(downtrend_stocks[['Ticker', 'Close', 'RSI_14', 'K5', 'D5']], width='stretch')

                        if 'Institutional_Selling' in dframe.columns:
                            inst_selling_stocks = dframe[dframe['Institutional_Selling'] == True]
                            if not inst_selling_stocks.empty:
                                st.markdown("#### 🏛️ 機構出貨跡象")
                                st.dataframe(inst_selling_stocks[['Ticker', 'Close', 'Ma20', 'Decline_3Days']], width='stretch')

                        # 完整數據預覽
                        st.markdown("### 📋 完整數據預覽")
                        st.dataframe(dframe, width='stretch')

                    else:
                        st.error("❌ 無法分析任何股票，請檢查檔案格式是否正確或股票代碼是否有效")

    # 新增的美股趨勢掃描分頁
    with tab4:
        us_trend_scanner_main()

    # 新增的美股大盤掃描分頁
    with tab5:
        us_market_scanner_main()

    # 指標說明分頁
    with tab6:
        st.markdown("### 📖 BBM-RTI 股票動能分析系統 - 指標說明")

        # 系統介紹
        st.markdown("""
        <div class="info-box">
        <strong>系統簡介</strong><br>
        本動能模型根據全球總體經濟，找尋趨勢產業與優質國家，最終精選出長期成長動能的個股，並依模型分數進行最適資產配置。
        <br><br>
        本模型依據基本面、技術面，結合消息面，如政府政策方向、中長期產業趨勢、營收獲利高成長、法人資金佈局，
        以及美國13F持股與國會議員申報資訊，系統化聚焦「價值＋動能」兼具的關鍵標的，掌握股價穩健成長，策略攻守兼備的潛力投資機會。
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # 欄位說明表格
        st.markdown("### 📋 欄位說明對照表")

        field_data = {
            "英文欄位": ["Ticker", "Close", "Daily_return", "Week_return", "Month_return", "YTD_Return", "HigherHigh", "All_Time_High",
                       "Week_52_High", "Week_52_Low", "Pct_From_52_High", "Pct_From_52_Low",
                       "VolumeChange", "VC_30", "RSI_5", "RSI_14", "MACD", "MACDsignal", "MACDhist",
                       "macdhist_signal", "MA5", "MA20", "MA60", "Crossover", "BBand", "BBand_middleband",
                       "BBand_crossover", "willr_D", "willr_D1", "K5", "D5", "Volume_5MA", "Volume_above_5MA",
                       "Volume_20MA", "Volume_below_20MA", "Decline_3Days", "Short_Uptrend_Momentum",
                       "Short_Downtrend_Signal", "Institutional_Selling", "Foreign_Net", "Trust_Net",
                       "Dealer_Net", "Total_Net", "Revenue_Month", "Revenue_Billion", "Revenue_New_High",
                       "EPS", "PE", "ROE",
                       "Composite_Momentum_S", "Composite_Momentum_L"],
            "中文名稱": ["股票代碼", "收盤價", "日報酬率", "週報酬率", "月報酬率", "YTD報酬率", "創新高(5日)", "收盤創歷史新高",
                       "52週最高價", "52週最低價", "距52週高點%", "距52週低點%",
                       "成交量變化", "量能超標30%", "RSI(5)", "RSI(14)", "MACD指標", "MACD訊號線", "MACD柱狀圖",
                       "MACD柱狀轉折", "5日均線", "20日均線", "60日均線", "均線黃金交叉", "布林通道擴張", "布林中軌上升",
                       "布林下軌突破", "威廉指標%D", "威廉指標%D前值", "KD K值(5)", "KD D值(5)", "5日成交量均線", "量大於5日均量",
                       "20日成交量均線", "量低於20MA", "3日累積跌幅", "短期上升動能",
                       "短期下跌訊號", "機構出貨指標", "外資淨買賣", "投信淨買賣",
                       "自營商淨買賣", "三大法人合計", "營收月份", "當月營收(億)", "營收創新高",
                       "每股盈餘", "本益比", "股東權益報酬率",
                       "短期綜合動能", "長期綜合動能"],
            "簡要說明": ["個股代號", "當日收盤價格", "當日漲跌幅", "近一週(5日)漲跌幅", "近一個月(22日)漲跌幅", "年初至今報酬率", "近5日是否創一年新高", "收盤價是否創十年內歷史新高",
                       "52週內最高價格", "52週內最低價格", "收盤價距離52週最高點差距%", "收盤價高於52週最低點幾%",
                       "當日量相對20日均量變化%", "成交量超過20日均量30%", "5日相對強弱指標", "14日相對強弱指標", "動能趨勢指標(12,26,9)", "MACD的9日平滑線", "MACD與訊號線差值",
                       "柱狀圖由負轉正訊號", "短期移動平均", "中短期移動平均", "中期移動平均", "MA5向上穿越MA20", "通道連續2日擴張", "中軌(20MA)上升中",
                       "價格向上突破下軌", "超買超賣指標(14日)", "前一期威廉%D值", "隨機指標K值(5,3,3)", "隨機指標D值(5,3,3)", "5日成交量移動平均", "目前量高於5日均量",
                       "近20日平均成交量", "目前量低於20日均量", "近3日累積下跌幅度%", "短線上漲力道(5條件)",
                       "短線轉弱訊號(4條件)", "大戶減碼訊號(3條件)", "外資買賣超(僅台股)", "投信買賣超(僅台股)",
                       "自營商買賣超(僅台股)", "法人總淨買賣(僅台股)", "最新公布營收的月份(僅台股)", "當月營收金額(億元)(僅台股)", "當月營收是否創歷史新高(僅台股)",
                       "每股盈餘(過去12個月)", "本益比(價格/EPS)", "股東權益報酬率(%)",
                       "短期多指標綜合動能", "中長期多指標綜合動能"]
        }

        field_df = pd.DataFrame(field_data)
        st.dataframe(field_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 指標詳細說明
        st.markdown("### 📊 七大技術指標詳細說明")

        # RSI
        with st.expander("1️⃣ RSI 相對強弱指標 (Relative Strength Index)", expanded=False):
            st.markdown("""
            **目的：** 衡量股價一定期間內的上漲力與下跌力。

            **設定：**
            - RSI_5：短線敏感型指標，反應快速
            - RSI_14：標準強弱判讀，較為穩定

            **判讀方式：**
            - RSI > 70 → 超買區，股價可能過熱
            - RSI < 30 → 超賣區，股價可能超跌
            - RSI_5 < 20 → 短線過度修正，可能反彈
            """)

        # MACD
        with st.expander("2️⃣ MACD 指標 (Moving Average Convergence Divergence)", expanded=False):
            st.markdown("""
            **目的：** 判斷趨勢方向及動能變化。

            **主要構成：**
            - **MACD 線**：短期(12日)與長期(26日)EMA差值
            - **Signal 線**：MACD的9日平滑線
            - **柱狀圖(Hist)**：MACD與Signal線差值，顯示多空動能增減

            **判讀方式：**
            - MACD > Signal → 多方動能增強
            - MACD < Signal → 空方動能增加
            - 柱狀圖由負轉正 → 多頭反轉訊號 (macdhist_signal = True)
            """)

        # 移動平均線
        with st.expander("3️⃣ 移動平均線 (Moving Averages)", expanded=False):
            st.markdown("""
            **目的：** 呈現股價的短、中、長期趨勢。

            **設定：**
            - MA5：短期情緒與短線動能 (週線)
            - MA20：中期趨勢，常視為月線成本
            - MA60：長期趨勢 (季線)

            **判讀方式：**
            - **多頭排列** (MA5 > MA20 > MA60)：上升趨勢確認
            - **空頭排列** (MA5 < MA20 < MA60)：下降趨勢成立
            - **Crossover = True**：MA5向上穿越MA20，黃金交叉訊號
            - 價格突破 MA20 → 試圖改變中期方向
            """)

        # 布林通道
        with st.expander("4️⃣ 布林通道 (Bollinger Bands)", expanded=False):
            st.markdown("""
            **目的：** 分析波動度與支撐壓力區。

            **構成：**
            - 上軌：20MA + 2倍標準差
            - 中軌：20日移動平均線
            - 下軌：20MA - 2倍標準差

            **系統指標說明：**
            - **BBand = True**：通道連續2日擴張，進入大波動行情
            - **BBand_middleband = True**：中軌上升中，中期趨勢向上
            - **BBand_crossover = True**：價格向上突破下軌，可能反彈

            **一般判讀：**
            - 處於上軌 → 多頭強勢但短線偏熱
            - 處於下軌 → 空方強勢但可能短線超賣
            - 通道收窄 → 低波動期，可能醞釀突破
            """)

        # 威廉指標
        with st.expander("5️⃣ 威廉指標 (Williams %R)", expanded=False):
            st.markdown("""
            **目的：** 短線超買／超賣的快速判斷工具。

            **設定：** 14日週期

            **判讀區間：**
            - %D > -20 → 超買區
            - %D < -80 → 超賣區

            **特點：** 比RSI更敏感，反轉線索明顯，但需搭配量價確認。
            """)

        # 成交量分析
        with st.expander("6️⃣ 成交量分析 (Volume Analysis)", expanded=False):
            st.markdown("""
            **目的：** 驗證趨勢是否具有支撐力。

            **系統指標說明：**
            - **VolumeChange**：當日成交量相對20日均量的變化百分比
            - **VC_30 = True**：成交量超過20日均量30%，資金明顯流入
            - **Volume_above_5MA = True**：目前量高於5日均量，短期量能增加
            - **Volume_below_20MA = True**：目前量低於20日均量，量能萎縮

            **判讀方式：**
            - 價漲量增 → 趨勢健康
            - 價漲量縮 → 上漲動能不足
            - 價跌量增 → 可能加速下跌
            """)

        # 複合動能指標
        with st.expander("7️⃣ 複合動能指標 (Composite Momentum Score)", expanded=False):
            st.markdown("""
            **目的：** 整合所有指標後給出單一「動能分數」，方便比較不同股票的強弱。

            **計算公式：**
            ```
            短期動能 (Composite_Momentum_S) =
                (RSI_5 - 50) +
                (MACDhist - macdhist_signal) +
                (MA5 - MA20) / MA20 × 100

            長期動能 (Composite_Momentum_L) =
                (RSI_14 - 50) +
                (MACDhist - macdhist_signal) +
                (MA20 - MA60) / MA60 × 100
            ```

            **用途：**
            - 用單一分數比較不同股票的強弱
            - 協助建構選股清單
            - 偵測動能加速或鈍化的時點
            - 作為停利／停損管理參考

            **判讀：**
            - 分數 > 10：強勢股票
            - 分數 < -10：弱勢股票
            """)

        st.markdown("---")

        # 三大核心信號
        st.markdown("### 🎯 三大核心交易信號")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #28a745;">
            <strong>📈 短期上升動能</strong><br>
            <small>Short_Uptrend_Momentum</small><br><br>
            <strong>5個條件全滿足：</strong><br>
            1. 收盤價 > MA5<br>
            2. 成交量 > 5日均量<br>
            3. K值 > D值<br>
            4. RSI14 > 50<br>
            5. MACD柱狀 > 0
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #dc3545;">
            <strong>📉 短期下跌訊號</strong><br>
            <small>Short_Downtrend_Signal</small><br><br>
            <strong>4個條件全滿足：</strong><br>
            1. 收盤價 < MA5<br>
            2. 成交量 < 20日均量<br>
            3. K值 < D值<br>
            4. MACD柱狀 < 0<br>
            &nbsp;
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #ffc107;">
            <strong>🏦 機構出貨指標</strong><br>
            <small>Institutional_Selling</small><br><br>
            <strong>3個條件全滿足：</strong><br>
            1. 收盤價 < MA20<br>
            2. 成交量 > 5日均量<br>
            3. 3日累積跌幅 > 5%<br>
            &nbsp;<br>
            &nbsp;
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 注意事項
        st.markdown("### ⚠️ 重要注意事項")
        st.markdown("""
        <div class="warning-box">
        <strong>投資風險提示：</strong><br><br>
        • 技術指標為趨勢與動能的<strong>量化描述</strong>，並非預測工具<br>
        • 股票投資需綜合觀察<strong>基本面、技術面與消息面</strong><br>
        • 本系統分析結果<strong>不應構成投資建議</strong><br>
        • 股票市場波動大、風險高，<strong>投資人應自行承擔風險與盈虧</strong><br>
        • 建議搭配其他分析工具與個人判斷，審慎評估後再行投資
        </div>
        """, unsafe_allow_html=True)

    # 頁腳
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    🔧 股市動能分析系統 | 📈 技術指標即時計算 | 💼 台股美股雙重支援 | 📁 自訂檔案分析
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
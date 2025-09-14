import numpy as np
import pandas as pd
import talib
import yfinance as yf
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 去除科學記號
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

# 讀取資料
data = pd.read_excel("2024-換股.xlsx")
tickers = data["股票代碼"]
name = data["股票名稱"]

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

def safe_get_value(series: pd.Series, index: int = -1) -> float:
    """安全獲取數值，避免 .values[0] 錯誤"""
    try:
        value = series.iloc[index]
        return float(value.item() if hasattr(value, 'item') else value)
    except (IndexError, AttributeError, TypeError):
        return np.nan

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
    indicators['higher_high'] = max(df['Close'].iloc[-5:]) > max(df['Close'].iloc[:-5])

    # 成交量變化
    vol_change = (safe_get_value(df['Volume']) / df['Volume'].iloc[-20:].mean() - 1) * 100
    indicators['volume_change'] = vol_change
    indicators['vc_30'] = vol_change > 30

    # 報酬率
    indicators['day_return'] = df['Close'].pct_change().iloc[-1] * 100
    indicators['week_return'] = df['Close'].pct_change(periods=5).dropna().iloc[-1] * 100 if len(df) >= 5 else np.nan
    indicators['month_return'] = df['Close'].pct_change(periods=22).dropna().iloc[-1] * 100 if len(df) >= 22 else np.nan

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

def process_stock_data() -> pd.DataFrame:
    """處理股票數據並計算技術指標"""
    data = pd.read_excel("代碼.xlsx")
    tickers = data["YFinance代碼"]
    names = data["股票名稱"]
    today = date.today()
    start_day = today - timedelta(365)

    results = []

    for i, ticker in enumerate(tickers):
        print(f"正在處理 {ticker} ({i+1}/{len(tickers)})...")

        try:
            df = yf.download(ticker, start=start_day, end=today, auto_adjust=False, progress=False)

            if df.empty:
                print(f"⚠️ {ticker} 無法獲取數據，跳過...")
                continue

            if len(df) < 60:
                print(f"⚠️ {ticker} 的資料少於 60 天，可能影響計算，跳過...")
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
            print(f"❌ 處理 {ticker} 時發生錯誤: {e}")
            continue

    return pd.DataFrame(results)

# 處理股票數據
dframe = process_stock_data()

if not dframe.empty:
    print(f"成功處理 {len(dframe)} 支股票")
    print(dframe.head())

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

    # 輸出結果
    try:
        with pd.ExcelWriter('TW動能觀察.xlsx', engine='xlsxwriter') as writer:
            dframe.to_excel(writer, sheet_name='stock_1', index=False)
        print("✅ 資料已成功輸出至 TW動能觀察.xlsx")
    except Exception as e:
        print(f"❌ 輸出檔案時發生錯誤: {e}")
else:
    print("❌ 沒有成功處理任何股票數據")
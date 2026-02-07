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

def validate_us_stock_code(stock_code: str) -> str:
    """驗證美股代碼格式"""
    # 美股代碼通常不需要後綴，直接返回
    stock_code = stock_code.strip().upper()

    # 檢查是否已經有後綴
    if '.' in stock_code:
        return stock_code

    # 對於一些特殊情況，可能需要添加交易所後綴
    # 但大部分美股不需要
    return stock_code

def calculate_us_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """計算美股技術指標"""
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

    # 注意：all_time_high 在 process_us_stock_data 中單獨計算（需要10年資料）

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

    # 成交量變化 - 美股成交量計算
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
                    print(f"Debug - US Volume calc: last={last_volume:.0f}, mean={vol_20_mean:.0f}, change={vol_change:.2f}%")
                else:
                    indicators['volume_change'] = 0.0
                    indicators['vc_30'] = False
                    print(f"Debug - US Invalid volume data: last={last_volume}, mean={vol_20_mean}")
            else:
                indicators['volume_change'] = 0.0
                indicators['vc_30'] = False
                print("Debug - US Not enough volume data")
        else:
            indicators['volume_change'] = 0.0
            indicators['vc_30'] = False
            print(f"Debug - US DataFrame too small: {len(df)} days")
    except Exception as e:
        print(f"US Volume calculation error: {e}")
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
        print(f"計算美股5日成交量平均時發生錯誤: {e}")
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
        print(f"計算美股20日成交量平均時發生錯誤: {e}")
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
        print(f"Debug - 美股短線上漲動能: close>{indicators.get('ma5', 0):.2f}={condition1}, vol_above_5ma={condition2}, K>{indicators.get('d5', 0):.2f}={condition3}, RSI>{indicators.get('rsi14', 0):.2f}>50={condition4}, MACD>{indicators.get('macdhist', 0):.4f}>0={condition5}, 結果={indicators['short_uptrend_momentum']}")

    except Exception as e:
        print(f"計算美股短線上漲動能時發生錯誤: {e}")
        indicators['short_uptrend_momentum'] = False

    # 短線下跌訊號指標 (4個條件全部滿足)
    try:
        condition1_down = indicators.get('close', 0) < indicators.get('ma5', 0) if not np.isnan(indicators.get('close', np.nan)) and not np.isnan(indicators.get('ma5', np.nan)) else False
        condition2_down = indicators.get('volume_below_20ma', False)
        condition3_down = indicators.get('k5', 0) < indicators.get('d5', 0) if not np.isnan(indicators.get('k5', np.nan)) and not np.isnan(indicators.get('d5', np.nan)) else False
        condition4_down = indicators.get('macdhist', 0) < 0 if not np.isnan(indicators.get('macdhist', np.nan)) else False

        indicators['short_downtrend_signal'] = bool(condition1_down and condition2_down and condition3_down and condition4_down)

        # 調試資訊
        print(f"Debug - 美股短線下跌訊號: close<{indicators.get('ma5', 0):.2f}={condition1_down}, vol_below_20ma={condition2_down}, K<{indicators.get('d5', 0):.2f}={condition3_down}, MACD<{indicators.get('macdhist', 0):.4f}<0={condition4_down}, 結果={indicators['short_downtrend_signal']}")

    except Exception as e:
        print(f"計算美股短線下跌訊號時發生錯誤: {e}")
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
        print(f"Debug - 美股機構出貨指標: close<{indicators.get('ma20', 0):.2f}={condition1_inst}, vol_above_5ma={condition2_inst}, 3日跌幅{indicators.get('decline_3days', 0):.2f}%>5%={condition3_inst}, 結果={indicators['institutional_selling']}")

    except Exception as e:
        print(f"計算美股機構出貨指標時發生錯誤: {e}")
        indicators['institutional_selling'] = False
        indicators['decline_3days'] = 0

    return indicators

def get_us_market_date() -> date:
    """獲取美股市場的最新交易日期"""
    import pytz

    # 使用美東時間
    us_eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(us_eastern)

    print(f"目前美東時間: {now_eastern.strftime('%Y-%m-%d %H:%M:%S')}")

    # 使用當前美東時間的日期
    target_date = now_eastern.date()

    # 如果是週末，往前調整到週五
    if target_date.weekday() == 5:  # Saturday
        target_date -= timedelta(days=1)
    elif target_date.weekday() == 6:  # Sunday
        target_date -= timedelta(days=2)

    print(f"美股數據使用日期: {target_date}")
    return target_date

def process_us_stock_data(input_file: str = None) -> pd.DataFrame:
    """處理美股數據並計算技術指標"""
    try:
        # 美股代碼列表 (硬編碼)
        us_stocks = [
            'SMH', 'MU', 'WDC', 'STX', 'SNDK', 'LITE', 'NVDA', 'AVGO', 'MRVL', 'AMD',
            'INTC', 'CRWV', 'NBIS', 'APLD', 'NVTS', 'ORCL', 'MSFT', 'GOOGL', 'TSLA', 'NFLX',
            'AAPL', 'META', 'AMZN', 'IBM', 'PLTR', 'ZETA', 'VSAT', 'RBLX', 'QUBT', 'ONDS',
            'RKLB', 'URA', 'KTOS', 'IREN', 'UUUU', 'QS', 'SMR', 'LEU', 'VST', 'XME',
            'XLP', 'WMT', 'COST', 'BYND', 'LIY', 'NVO', 'ISRG', 'SDGR', 'RXRX', 'RGC',
            'MP', 'CRML', 'LAC', 'UAMY'
        ]

        # 過濾掉空值和重複，並清理代碼
        valid_data = []
        for ticker in us_stocks:
            if ticker and str(ticker).strip():
                clean_ticker = str(ticker).strip().upper()
                valid_data.append(clean_ticker)

        if not valid_data:
            raise ValueError("沒有找到有效的美股代碼")

        print(f"載入了 {len(valid_data)} 個美股代碼")

        results = []

        for i, ticker in enumerate(valid_data):
            print(f"正在處理美股 {ticker} ({i+1}/{len(valid_data)})...")

            try:
                # 驗證美股代碼
                validated_ticker = validate_us_stock_code(ticker)
                # 使用period參數獲取最近一年數據，讓yfinance自動確定最新日期
                df = yf.download(validated_ticker, period='1y', auto_adjust=False, progress=False)

                if df.empty:
                    print(f"⚠️ {ticker} 無法獲取數據，跳過...")
                    continue

                if len(df) < 60:
                    print(f"⚠️ {ticker} 的資料少於 60 天，可能影響計算，跳過...")
                    continue

                indicators = calculate_us_technical_indicators(df)

                # 計算十年歷史新高 (All_Time_High)
                try:
                    df_10yr = yf.download(validated_ticker, period='10y', auto_adjust=False, progress=False)
                    if not df_10yr.empty:
                        current_close = float(df['Close'].iloc[-1])
                        ten_year_max = float(df_10yr['Close'].max())
                        # 允許小誤差（0.01%）來判斷是否相等
                        indicators['all_time_high'] = bool(current_close >= ten_year_max * 0.9999)
                    else:
                        indicators['all_time_high'] = False
                except:
                    indicators['all_time_high'] = False

                # 獲取基本面資料 (EPS, P/E, ROE) 和營收資料
                fundamental_data = {'eps': np.nan, 'pe': np.nan, 'roe': np.nan}
                revenue_data = {'latest_quarter': '', 'latest_revenue_billion': np.nan, 'is_new_high': False}
                try:
                    ticker_obj = yf.Ticker(validated_ticker)
                    stock_info = ticker_obj.info
                    if stock_info:
                        fundamental_data['eps'] = stock_info.get('trailingEps', np.nan)
                        fundamental_data['pe'] = stock_info.get('trailingPE', np.nan)
                        roe_value = stock_info.get('returnOnEquity', np.nan)
                        if roe_value is not None and not np.isnan(roe_value):
                            fundamental_data['roe'] = round(roe_value * 100, 2)  # 轉為百分比

                    # 獲取季度營收資料
                    quarterly_financials = ticker_obj.quarterly_financials
                    if quarterly_financials is not None and not quarterly_financials.empty:
                        # 找營收行 (Total Revenue)
                        revenue_row = None
                        for idx in quarterly_financials.index:
                            if 'Total Revenue' in str(idx) or 'Revenue' == str(idx):
                                revenue_row = idx
                                break

                        if revenue_row is not None:
                            revenues = quarterly_financials.loc[revenue_row].dropna()
                            if len(revenues) > 0:
                                # 最新季度營收
                                latest_revenue = float(revenues.iloc[0])
                                latest_quarter = revenues.index[0].strftime('%Y/Q%q').replace('Q1', 'Q1').replace('Q2', 'Q2').replace('Q3', 'Q3').replace('Q4', 'Q4')
                                # 簡化季度顯示
                                quarter_month = revenues.index[0].month
                                quarter_num = (quarter_month - 1) // 3 + 1
                                latest_quarter = f"{revenues.index[0].year}/Q{quarter_num}"

                                revenue_data['latest_quarter'] = latest_quarter
                                revenue_data['latest_revenue_billion'] = round(latest_revenue / 1000000000, 2)  # 轉為十億美元

                                # 判斷是否創新高（與歷史季度比較）
                                if len(revenues) > 1:
                                    historical_max = float(revenues.iloc[1:].max())
                                    revenue_data['is_new_high'] = latest_revenue > historical_max
                except Exception as e:
                    print(f"獲取 {validated_ticker} 基本面資料失敗: {e}")

                if indicators:
                    result = {
                        'Ticker': validated_ticker,
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
                        # 新增營收欄位
                        'Revenue_Quarter': revenue_data.get('latest_quarter', ''),
                        'Revenue_Billion': revenue_data.get('latest_revenue_billion', np.nan),
                        'Revenue_New_High': revenue_data.get('is_new_high', False),
                        # 新增基本面欄位
                        'EPS': fundamental_data.get('eps', np.nan),
                        'PE': fundamental_data.get('pe', np.nan),
                        'ROE': fundamental_data.get('roe', np.nan)
                    }
                    results.append(result)

            except Exception as e:
                print(f"❌ 處理美股 {ticker} 時發生錯誤: {e}")
                continue

        return pd.DataFrame(results)
    except Exception as e:
        print(f"❌ 處理美股數據時發生錯誤: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # 處理美股數據
    print("開始處理美股動能分析...")

    try:
        dframe = process_us_stock_data()

        if not dframe.empty:
            print(f"成功處理 {len(dframe)} 支美股")

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
                with pd.ExcelWriter('US動能觀察.xlsx', engine='xlsxwriter') as writer:
                    dframe.to_excel(writer, sheet_name='stock_1', index=False)
                print("✅ 美股資料已成功輸出至 US動能觀察.xlsx")
            except Exception as e:
                print(f"❌ 輸出美股檔案時發生錯誤: {e}")
        else:
            print("❌ 沒有成功處理任何美股數據")
    except FileNotFoundError:
        print("❌ 找不到相關檔案")
        print("請確認所需檔案存在且名稱正確")
    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {e}")
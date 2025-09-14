import numpy as np
import pandas as pd
import talib
import yfinance as yf
from datetime import date, timedelta

def safe_get_value(series: pd.Series, index: int = -1) -> float:
    """安全獲取數值，避免 .values[0] 錯誤"""
    try:
        if len(series) == 0:
            return np.nan
        value = series.iloc[index]
        if hasattr(value, 'item'):
            return float(value.item())
        elif hasattr(value, 'values'):
            return float(value.values[0]) if len(value.values) > 0 else np.nan
        else:
            return float(value)
    except (IndexError, AttributeError, TypeError, ValueError):
        return np.nan

def test_stock_indicators(ticker):
    """測試單一股票的指標計算"""
    print(f"\n🧪 測試股票: {ticker}")
    print("=" * 50)

    try:
        # 下載股票數據
        today = date.today()
        start_day = today - timedelta(365)
        df = yf.download(ticker, start=start_day, end=today, auto_adjust=False, progress=False)

        if df.empty:
            print(f"❌ {ticker} 無法獲取數據")
            return

        if len(df) < 60:
            print(f"❌ {ticker} 資料不足（{len(df)}天）")
            return

        print(f"✅ 成功獲取 {len(df)} 天數據")

        # 準備技術指標計算所需的陣列
        close_array = np.ravel(df['Close'].to_numpy())
        high_array = np.ravel(df['High'].to_numpy())
        low_array = np.ravel(df['Low'].to_numpy())

        # 1. 收盤價和5日MA
        ma5 = talib.SMA(close_array, timeperiod=5)
        current_close = safe_get_value(df['Close'])
        current_ma5 = ma5[-1] if len(ma5) > 0 else np.nan
        condition1 = current_close > current_ma5

        print(f"📊 條件1 - 收盤價 > 5日MA:")
        print(f"   收盤價: {current_close:.2f}")
        print(f"   5日MA: {current_ma5:.2f}")
        print(f"   結果: {condition1} ({'✅' if condition1 else '❌'})")

        # 2. 成交量 vs 5日平均
        current_volume = safe_get_value(df['Volume'])
        volume_5_mean = float(df['Volume'].iloc[-5:].mean())
        condition2 = current_volume > volume_5_mean

        print(f"\n📊 條件2 - 成交量 > 5日平均:")
        print(f"   當日成交量: {current_volume:,.0f}")
        print(f"   5日平均量: {volume_5_mean:,.0f}")
        print(f"   結果: {condition2} ({'✅' if condition2 else '❌'})")

        # 3. KD指標
        slowk, slowd = talib.STOCH(high_array, low_array, close_array,
                                  fastk_period=5, slowk_period=3, slowk_matype=0,
                                  slowd_period=3, slowd_matype=0)
        k5 = slowk[-1] if len(slowk) >= 1 else np.nan
        d5 = slowd[-1] if len(slowd) >= 1 else np.nan
        condition3 = k5 > d5

        print(f"\n📊 條件3 - K5 > D5:")
        print(f"   K5: {k5:.2f}")
        print(f"   D5: {d5:.2f}")
        print(f"   結果: {condition3} ({'✅' if condition3 else '❌'})")

        # 4. RSI > 50
        rsi14 = talib.RSI(close_array, timeperiod=14)
        current_rsi = rsi14[-1] if len(rsi14) > 0 else np.nan
        condition4 = current_rsi > 50

        print(f"\n📊 條件4 - RSI > 50:")
        print(f"   RSI(14): {current_rsi:.2f}")
        print(f"   結果: {condition4} ({'✅' if condition4 else '❌'})")

        # 5. MACD柱狀圖 > 0
        macd, macdsignal, macdhist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
        current_macdhist = macdhist[-1] if len(macdhist) > 0 else np.nan
        condition5 = current_macdhist > 0

        print(f"\n📊 條件5 - MACD柱狀圖 > 0:")
        print(f"   MACD Histogram: {current_macdhist:.4f}")
        print(f"   結果: {condition5} ({'✅' if condition5 else '❌'})")

        # 綜合結果
        short_uptrend_momentum = condition1 and condition2 and condition3 and condition4 and condition5

        print(f"\n🎯 短線上漲動能綜合判斷:")
        print(f"   條件1 (收盤>5MA): {condition1}")
        print(f"   條件2 (量>5日均): {condition2}")
        print(f"   條件3 (K5>D5): {condition3}")
        print(f"   條件4 (RSI>50): {condition4}")
        print(f"   條件5 (MACD>0): {condition5}")
        print(f"   最終結果: {short_uptrend_momentum} ({'🚀 短線上漲動能強勁!' if short_uptrend_momentum else '⚠️ 條件未全部滿足'})")

        # 6. 短線下跌訊號測試 (4個條件)
        print(f"\n" + "="*50)
        print("🔻 測試第二個指標：短線下跌訊號")
        print("="*50)

        # 計算成交量20日平均
        if len(df) >= 20:
            volume_20_mean = float(df['Volume'].iloc[-20:].mean())
            volume_condition_down = current_volume < volume_20_mean
            print(f"\n📊 條件2 - 成交量 < 20日平均:")
            print(f"   當日成交量: {current_volume:,.0f}")
            print(f"   20日平均量: {volume_20_mean:,.0f}")
            print(f"   結果: {volume_condition_down} ({'✅' if volume_condition_down else '❌'})")
        else:
            volume_condition_down = False
            print(f"\n📊 條件2 - 成交量 < 20日平均: 資料不足")

        # 條件檢查
        condition1_down = current_close < current_ma5
        condition2_down = volume_condition_down
        condition3_down = k5 < d5
        condition4_down = current_macdhist < 0

        print(f"\n📊 條件1 - 收盤價 < 5日MA:")
        print(f"   收盤價: {current_close:.2f}")
        print(f"   5日MA: {current_ma5:.2f}")
        print(f"   結果: {condition1_down} ({'✅' if condition1_down else '❌'})")

        print(f"\n📊 條件3 - K5 < D5:")
        print(f"   K5: {k5:.2f}")
        print(f"   D5: {d5:.2f}")
        print(f"   結果: {condition3_down} ({'✅' if condition3_down else '❌'})")

        print(f"\n📊 條件4 - MACD柱狀圖 < 0:")
        print(f"   MACD Histogram: {current_macdhist:.4f}")
        print(f"   結果: {condition4_down} ({'✅' if condition4_down else '❌'})")

        # 綜合結果
        short_downtrend_signal = condition1_down and condition2_down and condition3_down and condition4_down

        print(f"\n🎯 短線下跌訊號綜合判斷:")
        print(f"   條件1 (收盤<5MA): {condition1_down}")
        print(f"   條件2 (量<20日均): {condition2_down}")
        print(f"   條件3 (K5<D5): {condition3_down}")
        print(f"   條件4 (MACD<0): {condition4_down}")
        print(f"   最終結果: {short_downtrend_signal} ({'📉 短線下跌訊號強烈!' if short_downtrend_signal else '⚠️ 條件未全部滿足'})")

        # 7. 機構出貨指標測試 (3個條件)
        print(f"\n" + "="*50)
        print("🏛️ 測試第三個指標：機構出貨指標")
        print("="*50)

        # 計算三日累積下跌幅度
        if len(df) >= 4:
            close_3days_ago = safe_get_value(df['Close'], -4)  # 4天前的收盤價
            decline_3days = ((close_3days_ago - current_close) / close_3days_ago) * 100 if close_3days_ago > 0 else 0
            print(f"\n📊 條件3 - 三日累積下跌超過5%:")
            print(f"   3日前收盤價: {close_3days_ago:.2f}")
            print(f"   今日收盤價: {current_close:.2f}")
            print(f"   3日累積跌幅: {decline_3days:.2f}%")
            decline_condition = decline_3days > 5
            print(f"   結果: {decline_condition} ({'✅' if decline_condition else '❌'})")
        else:
            decline_3days = 0
            decline_condition = False
            print(f"\n📊 條件3 - 三日累積下跌超過5%: 資料不足")

        # 計算20日MA
        ma20 = talib.SMA(close_array, timeperiod=20)
        current_ma20 = ma20[-1] if len(ma20) > 0 else np.nan

        # 條件檢查
        condition1_inst = current_close < current_ma20 if not np.isnan(current_ma20) else False
        condition2_inst = current_volume > volume_5_mean
        condition3_inst = decline_condition

        print(f"\n📊 條件1 - 收盤價 < 20日MA:")
        print(f"   收盤價: {current_close:.2f}")
        print(f"   20日MA: {current_ma20:.2f}" if not np.isnan(current_ma20) else "   20日MA: 資料不足")
        print(f"   結果: {condition1_inst} ({'✅' if condition1_inst else '❌'})")

        print(f"\n📊 條件2 - 成交量 > 5日平均:")
        print(f"   當日成交量: {current_volume:,.0f}")
        print(f"   5日平均量: {volume_5_mean:,.0f}")
        print(f"   結果: {condition2_inst} ({'✅' if condition2_inst else '❌'})")

        # 綜合結果
        institutional_selling = condition1_inst and condition2_inst and condition3_inst

        print(f"\n🎯 機構出貨指標綜合判斷:")
        print(f"   條件1 (收盤<20MA): {condition1_inst}")
        print(f"   條件2 (量>5日均): {condition2_inst}")
        print(f"   條件3 (3日跌>5%): {condition3_inst}")
        print(f"   最終結果: {institutional_selling} ({'🏛️ 機構出貨跡象明顯!' if institutional_selling else '⚠️ 條件未全部滿足'})")

        # 顯示最近5天價格走勢
        print(f"\n📈 最近5天價格走勢:")
        recent_data = df[['Close', 'Volume']].tail(5)
        for i, (date_idx, row) in enumerate(recent_data.iterrows()):
            print(f"   {date_idx.strftime('%Y-%m-%d')}: 收盤 {row['Close']:.2f}, 成交量 {row['Volume']:,.0f}")

    except Exception as e:
        print(f"❌ 測試 {ticker} 時發生錯誤: {e}")

if __name__ == "__main__":
    # 測試幾檔不同的股票
    test_stocks = ["2330.TW", "AAPL", "TSLA"]

    for stock in test_stocks:
        test_stock_indicators(stock)
        print("\n" + "="*80 + "\n")
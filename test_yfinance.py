"""测试 yfinance 下载功能"""
import yfinance as yf
import pandas as pd

def test_download():
    print("=" * 60)
    print("测试 yfinance 批量下载")
    print("=" * 60)

    # 测试1: 单个股票
    print("\n【测试1】单个股票 (AAPL)")
    print("-" * 60)
    try:
        df = yf.download('AAPL', period='1mo', progress=False)
        print(f"✅ 成功! 数据形状: {df.shape}")
        print(f"最新日期: {df.index[-1]}")
        print(f"最新收盘价: {df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__} - {str(e)[:100]}")

    # 测试2: 10支股票批量下载
    print("\n【测试2】批量下载10支股票")
    print("-" * 60)
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'INTC', 'QCOM']
    try:
        tickers_str = ' '.join(tickers)
        df_batch = yf.download(tickers_str, period='1mo', progress=False, group_by='ticker', threads=True)

        if df_batch.empty:
            print("❌ 返回空数据!")
        else:
            print(f"✅ 成功! 数据形状: {df_batch.shape}")

            # 检查列结构
            if hasattr(df_batch.columns, 'get_level_values'):
                stocks = list(df_batch.columns.get_level_values(0).unique())
                print(f"获取到的股票 ({len(stocks)}支): {stocks}")

                # 测试访问单个股票数据
                test_ticker = stocks[0]
                print(f"\n测试访问 {test_ticker} 的数据:")
                test_data = df_batch[test_ticker]
                print(f"  - 数据行数: {len(test_data)}")
                print(f"  - 最新收盘价: {test_data['Close'].iloc[-1]:.2f}")
            else:
                print("⚠️ 列结构不是MultiIndex")

    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}")
        print(f"错误信息: {str(e)[:200]}")
        import traceback
        print("\n完整堆栈:")
        traceback.print_exc()

    # 测试3: 504支股票 (SPY成分股数量)
    print("\n【测试3】大批量下载 (30支股票)模拟")
    print("-" * 60)
    large_tickers = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'INTC', 'QCOM',
        'NFLX', 'ADBE', 'CSCO', 'AVGO', 'TXN', 'ORCL', 'IBM', 'CRM', 'NOW', 'INTU',
        'SHOP', 'PYPL', 'SQ', 'DOCU', 'ZM', 'TEAM', 'SNOW', 'DDOG', 'NET', 'CRWD'
    ]
    try:
        tickers_str = ' '.join(large_tickers)
        df_large = yf.download(tickers_str, period='1mo', progress=False, group_by='ticker', threads=True)

        if df_large.empty:
            print("❌ 返回空数据!")
        else:
            stocks = list(df_large.columns.get_level_values(0).unique()) if hasattr(df_large.columns, 'get_level_values') else []
            print(f"✅ 成功! 获取到 {len(stocks)}/{len(large_tickers)} 支股票")
            print(f"数据形状: {df_large.shape}")

    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}")
        print(f"错误信息: {str(e)[:200]}")

if __name__ == "__main__":
    test_download()

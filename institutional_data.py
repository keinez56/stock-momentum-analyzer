import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import time
import urllib3

# 忽略SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_institutional_trading(stock_code, start_date, end_date):
    """
    下載指定股票代碼一段時間的三大法人買賣超資訊

    Parameters:
    stock_code (str): 股票代碼，例如 '2330'
    start_date (str): 開始日期，格式 'YYYY-MM-DD'
    end_date (str): 結束日期，格式 'YYYY-MM-DD'

    Returns:
    pandas.DataFrame: 包含三大法人買賣超資訊的DataFrame
    """

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/111.25 (KHTML, like Gecko) Chrome/99.0.2345.81 Safari/123.36'}

    # 轉換日期格式
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    all_data = []

    current_date = start
    while current_date <= end:
        # 跳過週末
        if current_date.weekday() < 5:  # 0-4 代表週一到週五
            date_str = current_date.strftime('%Y%m%d')
            url = f'https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALL&response=csv'

            try:
                print(f"正在下載 {current_date.strftime('%Y-%m-%d')} 的資料...")
                res = requests.get(url, headers=headers, verify=False)

                if res.status_code == 200 and res.text:
                    # 去除指數價格
                    lines = [l for l in res.text.split('\n') if len(l.split(',"'))>=10]

                    if lines:
                        # 將list轉為txt方便用csv讀取
                        df = pd.read_csv(io.StringIO(','.join(lines)))
                        # 將不必要的符號去除
                        df = df.map(lambda s:(str(s).replace('=','').replace(',','').replace('"','')))

                        # 篩選指定股票代碼
                        if stock_code in df['證券代號'].values:
                            stock_data = df[df['證券代號'] == stock_code].copy()
                            stock_data['日期'] = current_date.strftime('%Y-%m-%d')
                            # 將數字轉為數值型態
                            for col in stock_data.columns:
                                if col not in ['證券代號', '證券名稱', '日期']:
                                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                            all_data.append(stock_data)
                        else:
                            print(f"  找不到股票代碼 {stock_code} 的資料")
                else:
                    print(f"  無法取得資料，狀態碼: {res.status_code}")

            except Exception as e:
                print(f"  下載 {current_date.strftime('%Y-%m-%d')} 資料時發生錯誤: {e}")

            # 避免請求過於頻繁
            time.sleep(1)

        current_date += timedelta(days=1)

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        # 重新排列欄位，將日期放在前面
        cols = ['日期'] + [col for col in result_df.columns if col != '日期']
        result_df = result_df[cols]
        return result_df
    else:
        print("未找到任何資料")
        return pd.DataFrame()

# 使用範例
if __name__ == "__main__":
    # 下載台積電(2330)最近一週的三大法人買賣超資訊
    stock_code = "2330"
    start_date = "2024-09-12"  # 修改為您想要的開始日期
    end_date = "2024-09-13"    # 修改為您想要的結束日期

    df = get_institutional_trading(stock_code, start_date, end_date)
    print(f"\n{stock_code} 三大法人買賣超資訊:")
    print(df)
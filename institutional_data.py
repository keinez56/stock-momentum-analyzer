import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import time
import urllib3
from typing import List, Dict

# 忽略SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_all_institutional_data(date_str: str) -> pd.DataFrame:
    """
    一次下載指定日期的全部台股三大法人買賣超資料

    Parameters:
    date_str (str): 日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'

    Returns:
    pandas.DataFrame: 包含所有股票三大法人買賣超資訊的DataFrame
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/111.25 (KHTML, like Gecko) Chrome/99.0.2345.81 Safari/123.36'}

    # 處理日期格式
    if '-' in date_str:
        date_str = date_str.replace('-', '')

    url = f'https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALL&response=csv'

    try:
        print(f"正在下載 {date_str} 的全部三大法人資料...")
        res = requests.get(url, headers=headers, verify=False)

        if res.status_code == 200 and res.text:
            # 去除指數價格，只保留股票資料
            lines = [l for l in res.text.split('\n') if len(l.split(',"')) >= 10]

            if lines:
                # 將list轉為txt方便用csv讀取
                df = pd.read_csv(io.StringIO(','.join(lines)))
                # 將不必要的符號去除
                df = df.map(lambda s: (str(s).replace('=', '').replace(',', '').replace('"', '')))

                # 添加日期欄位
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                df['日期'] = formatted_date

                # 將數字欄位轉為數值型態
                for col in df.columns:
                    if col not in ['證券代號', '證券名稱', '日期']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                return df
            else:
                print("  沒有找到有效的資料行")
                return pd.DataFrame()
        else:
            print(f"  無法取得資料，狀態碼: {res.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"  下載 {date_str} 資料時發生錯誤: {e}")
        return pd.DataFrame()

def get_institutional_trading_batch(stock_codes: List[str], date_str: str) -> Dict[str, pd.DataFrame]:
    """
    批量取得多檔股票在指定日期的三大法人買賣超資料

    Parameters:
    stock_codes (List[str]): 股票代碼列表
    date_str (str): 日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'

    Returns:
    Dict[str, pd.DataFrame]: 以股票代碼為key的字典，值為該股票的三大法人資料
    """
    # 一次下載全部資料
    all_data = get_all_institutional_data(date_str)

    if all_data.empty:
        return {}

    result = {}
    for stock_code in stock_codes:
        stock_data = all_data[all_data['證券代號'] == stock_code]
        if not stock_data.empty:
            result[stock_code] = stock_data.copy()
        else:
            print(f"  找不到股票代碼 {stock_code} 的資料")

    return result

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
    # 測試新的批量下載功能
    stock_codes = ["2330", "2454", "6805"]  # 台積電、聯發科、富邦媒
    date_str = "20250916"  # 修改為您想要的日期

    print("=== 測試批量下載功能 ===")
    batch_result = get_institutional_trading_batch(stock_codes, date_str)

    for stock_code, data in batch_result.items():
        print(f"\n{stock_code} 三大法人買賣超資訊:")
        print(data)

    print("\n=== 測試單檔下載功能（舊版） ===")
    # 下載台積電(2330)最近一週的三大法人買賣超資訊
    stock_code = "2330"
    start_date = "2024-09-12"  # 修改為您想要的開始日期
    end_date = "2024-09-13"    # 修改為您想要的結束日期

    df = get_institutional_trading(stock_code, start_date, end_date)
    print(f"\n{stock_code} 三大法人買賣超資訊:")
    print(df)
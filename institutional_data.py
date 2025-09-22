import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import time
import urllib3
from typing import List, Dict
import pytz

# 忽略SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_latest_trading_date_for_institutional_data() -> str:
    """
    獲取三大法人資料的最新可用交易日期

    邏輯：
    1. 如果現在是台灣時間下午6點後，使用今日（如果是交易日）
    2. 如果是台灣時間下午6點前，使用前一個交易日
    3. 自動跳過週末和假日

    Returns:
    str: 格式為 YYYYMMDD 的日期字串
    """
    # 設定台灣時區
    taiwan_tz = pytz.timezone('Asia/Taipei')
    now_taiwan = datetime.now(taiwan_tz)

    print(f"目前台灣時間: {now_taiwan.strftime('%Y-%m-%d %H:%M:%S')}")

    # 判斷是否已過下午6點（18:00）
    cutoff_time = now_taiwan.replace(hour=18, minute=0, second=0, microsecond=0)

    if now_taiwan >= cutoff_time and now_taiwan.weekday() < 5:  # 週一到週五且已過6點
        # 使用今日
        target_date = now_taiwan.date()
        print(f"已過下午6點且為交易日，使用今日: {target_date}")
    else:
        # 使用前一個交易日
        target_date = now_taiwan.date() - timedelta(days=1)
        print(f"未過下午6點或非交易日，往前找交易日...")

    # 往前找最近的交易日（跳過週末）
    while target_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        target_date -= timedelta(days=1)
        print(f"跳過週末，調整為: {target_date}")

    date_str = target_date.strftime('%Y%m%d')
    print(f"最終使用日期: {date_str}")

    return date_str

def get_trading_date_for_stock_data() -> datetime:
    """
    獲取股價資料應該使用的日期（與三大法人資料對應）

    Returns:
    datetime: 股價資料的結束日期
    """
    institutional_date_str = get_latest_trading_date_for_institutional_data()
    institutional_date = datetime.strptime(institutional_date_str, '%Y%m%d')

    # 股價資料應該使用這個日期的下一個交易日
    # 因為我們通常會取前N天的資料，結束日期需要包含目標日期
    stock_end_date = institutional_date + timedelta(days=1)

    print(f"股價資料結束日期: {stock_end_date.strftime('%Y-%m-%d')}")
    return stock_end_date

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
                df = df.applymap(lambda s: (str(s).replace('=', '').replace(',', '').replace('"', '')))

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

def get_institutional_trading_batch(stock_codes: List[str], date_str: str = None) -> Dict[str, pd.DataFrame]:
    """
    批量取得多檔股票在指定日期的三大法人買賣超資料

    Parameters:
    stock_codes (List[str]): 股票代碼列表
    date_str (str, optional): 日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'。如果不提供，自動使用最新可用日期

    Returns:
    Dict[str, pd.DataFrame]: 以股票代碼為key的字典，值為該股票的三大法人資料
    """
    # 如果沒有提供日期，使用智能日期選擇
    if date_str is None:
        date_str = get_latest_trading_date_for_institutional_data()
        print(f"使用智能選擇的日期: {date_str}")

    # 一次下載全部資料
    all_data = get_all_institutional_data(date_str)

    # 如果當日沒有資料，嘗試前多個交易日
    if all_data.empty:
        print(f"日期 {date_str} 沒有資料，嘗試前多個交易日...")
        try:
            current_date = datetime.strptime(date_str, '%Y%m%d') if '-' not in date_str else datetime.strptime(date_str.replace('-', ''), '%Y%m%d')
            # 往前找最近的交易日，擴展到15天
            for i in range(1, 16):  # 最多往前找15天
                prev_date = current_date - timedelta(days=i)
                if prev_date.weekday() < 5:  # 跳過週末
                    prev_date_str = prev_date.strftime('%Y%m%d')
                    print(f"嘗試日期: {prev_date_str}")
                    all_data = get_all_institutional_data(prev_date_str)
                    if not all_data.empty:
                        print(f"成功取得 {prev_date_str} 的資料")
                        break
                    # 增加延遲避免太頻繁請求
                    time.sleep(0.5)
        except Exception as e:
            print(f"日期轉換錯誤: {e}")

    if all_data.empty:
        print("警告: 無法取得任何三大法人資料，可能是非交易日或系統維護中")
        return {}

    result = {}
    found_count = 0
    for stock_code in stock_codes:
        stock_data = all_data[all_data['證券代號'] == stock_code]
        if not stock_data.empty:
            result[stock_code] = stock_data.copy()
            found_count += 1
        else:
            print(f"  找不到股票代碼 {stock_code} 的資料")

    print(f"成功找到 {found_count}/{len(stock_codes)} 檔股票的三大法人資料")
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
                        df = df.applymap(lambda s:(str(s).replace('=','').replace(',','').replace('"','')))

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
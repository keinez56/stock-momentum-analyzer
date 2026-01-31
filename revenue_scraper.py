# -*- coding: utf-8 -*-
"""
台股營收資料爬蟲模組
使用 FinMind API 爬取台股每月營收資料並判斷是否創新高
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


def get_revenue_finmind(stock_id: str, token: str = None) -> Optional[Dict]:
    """
    從 FinMind API 爬取股票的月營收資料

    Args:
        stock_id: 股票代碼 (例如: "2330")
        token: FinMind API token (可選，免費版每日有限制)

    Returns:
        包含營收資料的字典，或 None（如果失敗）
    """
    try:
        # 清理股票代碼
        clean_id = str(stock_id).replace('.TW', '').replace('.TWO', '').strip()

        # 設定日期範圍（近3年資料用於比較）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

        # FinMind API
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockMonthRevenue",
            "data_id": clean_id,
            "start_date": start_date,
            "end_date": end_date
        }

        if token:
            params["token"] = token

        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if data.get('status') != 200 or not data.get('data'):
            print(f"FinMind API 無法取得 {clean_id} 的營收資料")
            return None

        df = pd.DataFrame(data['data'])

        if df.empty:
            return None

        # 按日期排序（最新在前）
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False).reset_index(drop=True)

        # 取得最新月份營收
        latest = df.iloc[0]
        latest_revenue = float(latest['revenue'])  # 單位：元
        latest_month = latest['date'].strftime('%Y/%m')

        # 計算歷史最高營收（排除最新月份）
        if len(df) > 1:
            historical_max = float(df.iloc[1:]['revenue'].max())
        else:
            historical_max = 0

        # 判斷是否創新高
        is_new_high = latest_revenue > historical_max if historical_max > 0 else False

        return {
            'stock_id': clean_id,
            'latest_month': latest_month,
            'latest_revenue': latest_revenue,  # 單位：元
            'latest_revenue_billion': latest_revenue / 100000000,  # 轉換為億元 (元/1億)
            'historical_max': historical_max,
            'historical_max_billion': historical_max / 100000000,
            'is_new_high': is_new_high,
            'yoy_growth': float(latest.get('revenue_year_growth', 0)) if 'revenue_year_growth' in latest else None,
            'mom_growth': float(latest.get('revenue_month_growth', 0)) if 'revenue_month_growth' in latest else None
        }

    except Exception as e:
        print(f"FinMind 爬取 {stock_id} 營收時發生錯誤: {e}")
        return None


def get_revenue_twse(stock_id: str) -> Optional[Dict]:
    """
    從證交所公開資料爬取營收（備用方案）

    Args:
        stock_id: 股票代碼

    Returns:
        包含營收資料的字典
    """
    try:
        clean_id = str(stock_id).replace('.TW', '').replace('.TWO', '').strip()

        # 計算上個月的年月
        now = datetime.now()
        if now.month == 1:
            report_year = now.year - 1
            report_month = 12
        else:
            report_year = now.year
            report_month = now.month - 1

        # 證交所 API（上市公司）
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?date={report_year}{report_month:02d}01&stockNo={clean_id}&response=json"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }

        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()

        if data.get('stat') == 'OK' and data.get('data'):
            # 解析資料
            # 這是備用方案，結構可能需要調整
            pass

        return None

    except Exception as e:
        print(f"TWSE API 爬取 {stock_id} 營收時發生錯誤: {e}")
        return None


def get_revenue_batch(stock_ids: list, progress_callback=None) -> Dict[str, Dict]:
    """
    批量爬取多檔股票的營收資料

    Args:
        stock_ids: 股票代碼列表
        progress_callback: 進度回呼函數 (current, total, stock_id)

    Returns:
        字典，key 為股票代碼，value 為營收資料
    """
    results = {}
    total = len(stock_ids)

    # 過濾出台股代碼
    tw_stock_ids = []
    for stock_id in stock_ids:
        clean_id = str(stock_id).replace('.TW', '').replace('.TWO', '').strip()
        if clean_id.isdigit() and len(clean_id) == 4:
            tw_stock_ids.append(clean_id)

    if not tw_stock_ids:
        return results

    # 嘗試一次性從 FinMind 取得所有股票資料
    try:
        # 設定日期範圍
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')

        # FinMind API - 一次請求多檔股票
        url = "https://api.finmindtrade.com/api/v4/data"

        all_revenue_data = []

        # FinMind 免費版有限制，逐檔查詢較穩定
        for i, stock_id in enumerate(tw_stock_ids):
            if progress_callback:
                progress_callback(i + 1, len(tw_stock_ids), stock_id)

            params = {
                "dataset": "TaiwanStockMonthRevenue",
                "data_id": stock_id,
                "start_date": start_date,
                "end_date": end_date
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                data = response.json()

                if data.get('status') == 200 and data.get('data'):
                    df = pd.DataFrame(data['data'])

                    if not df.empty:
                        # 按日期排序
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date', ascending=False).reset_index(drop=True)

                        # 取得最新月份
                        latest = df.iloc[0]
                        latest_revenue = float(latest['revenue'])
                        latest_month = latest['date'].strftime('%Y/%m')

                        # 歷史最高
                        if len(df) > 1:
                            historical_max = float(df.iloc[1:]['revenue'].max())
                        else:
                            historical_max = 0

                        is_new_high = latest_revenue > historical_max if historical_max > 0 else False

                        results[stock_id] = {
                            'stock_id': stock_id,
                            'latest_month': latest_month,
                            'latest_revenue': latest_revenue,
                            'latest_revenue_billion': latest_revenue / 100000000,  # 元轉億元
                            'historical_max': historical_max,
                            'historical_max_billion': historical_max / 100000000,
                            'is_new_high': is_new_high
                        }

            except Exception as e:
                print(f"爬取 {stock_id} 時發生錯誤: {e}")
                continue

            # 避免請求過於頻繁
            time.sleep(0.3)

    except Exception as e:
        print(f"批量爬取營收時發生錯誤: {e}")

    return results


def format_revenue(revenue: float, unit: str = 'ntd') -> str:
    """
    格式化營收數字為易讀格式

    Args:
        revenue: 營收數值
        unit: 原始單位 ('ntd' 元, 'billion' 億)

    Returns:
        格式化後的字串
    """
    if unit == 'ntd':
        # FinMind 回傳的單位是「元」
        # 元轉億元: /100000000
        billion = revenue / 100000000
        if billion >= 1:
            return f"{billion:.2f}億"
        elif revenue >= 10000:
            return f"{revenue/10000:.2f}萬"
        else:
            return f"{revenue:.0f}元"
    elif unit == 'billion':
        return f"{revenue:.2f}億"
    else:
        return f"{revenue:.2f}"


# 測試函數
if __name__ == "__main__":
    print("=" * 50)
    print("測試 FinMind API 爬取營收資料")
    print("=" * 50)

    # 測試單一股票
    test_stocks = ["2330", "2454", "2317"]

    for stock in test_stocks:
        print(f"\n測試爬取 {stock} 營收資料...")
        result = get_revenue_finmind(stock)

        if result:
            print(f"  股票代碼: {result['stock_id']}")
            print(f"  最新月份: {result['latest_month']}")
            print(f"  當月營收: {format_revenue(result['latest_revenue'])} ({result['latest_revenue_billion']:.2f}億)")
            print(f"  歷史最高: {format_revenue(result['historical_max'])} ({result['historical_max_billion']:.2f}億)")
            print(f"  是否創新高: {'是' if result['is_new_high'] else '否'}")
        else:
            print(f"  爬取失敗")

        time.sleep(1)

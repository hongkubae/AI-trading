# fetch_okx_data.py

import requests
import pandas as pd
import os

OKX_ENDPOINT = "https://www.okx.com"

def fetch_okx_5min_latest(inst_id="ETH-USDT", limit=200):
    """
    OKX에서 5분봉 '최신 limit개'를 받아온다.
    - 가장 최근 봉부터 ~ 과거 순서로 data가 온다고 가정.
    - 반환: DF(['ts','open','high','low','close','volume'])
    """
    url = f"{OKX_ENDPOINT}/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": "5m",
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data_json = r.json()

    if "data" not in data_json:
        print("[fetch_okx_5min_latest] 'data' not in response.")
        return pd.DataFrame()

    records = data_json["data"]
    if not records:
        return pd.DataFrame()

    all_rows = []
    # OKX는 [최근 -> 과거] 순으로 줄 때가 많으므로
    for row in records:
        # row: [ts(ms), open, high, low, close, volume, ...]
        all_rows.append({
            "ts": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })
    df = pd.DataFrame(all_rows)
    # 과거 -> 최근순 정렬
    df.sort_values(by="ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def update_historical_csv(inst_id="ETH-USDT", csv_path="data/historical_data.csv"):
    """
    1) 기존 CSV 로드
    2) OKX에서 '최신 200봉'을 받아옴(fetch_okx_5min_latest)
    3) 기존 + 새 데이터 병합 -> 중복 제거 -> 정렬 -> 저장
    => 5분마다 호출하면, 새 봉이 생길 때마다 CSV rows 증가
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.isfile(csv_path):
        df_existing = pd.read_csv(csv_path)
    else:
        df_existing = pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    # 최신 200봉 가져오기
    new_df = fetch_okx_5min_latest(inst_id=inst_id, limit=200)
    if new_df.empty:
        print("[update_historical_csv] No new data fetched (empty).")
        return

    combined = pd.concat([df_existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    combined.sort_values(by="ts", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(csv_path, index=False)
    print(f"[update_historical_csv] Updated {csv_path}, rows={len(combined)}")

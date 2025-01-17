# fetch_okx_data.py

import requests
import pandas as pd
import os

OKX_ENDPOINT = "https://www.okx.com"

def fetch_okx_5min_after(inst_id="ETH-USDT", after_ts=None, limit=200):
    """
    OKX 5분봉을 after_ts(밀리초) 이후로 최대 limit개 요청.
    반환: DataFrame(['ts','open','high','low','close','volume'])
    """
    url = f"{OKX_ENDPOINT}/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": "5m",
        "limit": limit
    }
    if after_ts is not None:
        # 마지막 ts 이후 봉만 받기 위해 after=last_ts+1
        params["after"] = str(after_ts + 1)

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data_json = r.json()

    if "data" not in data_json:
        print("[fetch_okx_5min_after] No 'data' field in response.")
        return pd.DataFrame()

    records = data_json["data"]
    if not records:
        return pd.DataFrame()

    all_rows = []
    for row in records:
        # row: [timestamp(ms), open, high, low, close, volume, ...]
        all_rows.append({
            "ts": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })
    df = pd.DataFrame(all_rows)
    df.sort_values(by="ts", inplace=True)  # 과거->최근
    df.reset_index(drop=True, inplace=True)
    return df

def update_historical_csv(inst_id="ETH-USDT", csv_path="data/historical_data.csv"):
    """
    - 기존 CSV 불러옴
    - 마지막 ts 찾기
    - after=last_ts로 OKX 5분봉 받아 옴
    - 기존 + 새 데이터 병합 -> 중복 제거 -> 정렬 -> 저장
    => 5분마다 실행 시, CSV가 점차 커짐
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.isfile(csv_path):
        df_existing = pd.read_csv(csv_path)
        if df_existing.empty:
            last_ts = None
        else:
            last_ts = df_existing["ts"].max()
    else:
        df_existing = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        last_ts = None

    new_df = fetch_okx_5min_after(inst_id=inst_id, after_ts=last_ts, limit=200)
    if new_df.empty:
        print("[update_historical_csv] No new data or empty fetch.")
        return

    combined = pd.concat([df_existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["ts"], inplace=True)
    combined.sort_values(by="ts", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(csv_path, index=False)
    print(f"[update_historical_csv] Updated {csv_path}, rows={len(combined)}")

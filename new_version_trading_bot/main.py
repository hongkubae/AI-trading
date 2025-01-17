# main.py

import time
import os
import traceback
import pandas as pd
import numpy as np

from fetch_okx_data import update_historical_csv
from model_manager import train_model, load_model

# 예: 간단 매매 로직에서 필요
from indicators import add_indicators

########################################
# 전역 설정
########################################
FETCH_INTERVAL = 300   # 5분마다 데이터 누적
TRAIN_INTERVAL = 3600  # 1시간마다 모델 재학습
HISTORICAL_CSV = "data/historical_data.csv"

MODEL_TYPE = "RF"      # "RF", "MLP", "LSTM" 등
MODEL_PATH = "best_model.pkl"

LAST_TRAIN_TIME = 0.0
current_model = None

# 페이퍼 트레이딩 간단 상태
current_position = 0   # 0=none, 1=long
entry_price = 0.0
STOP_LOSS_PERCENT = -5.0
TAKE_PROFIT_PERCENT = 10.0
LEVERAGE = 10

# 슬리피지·수수료 예시
SLIPPAGE_PCT = 0.02
FEE_PCT = 0.05

def log(msg):
    print(f"[MAIN] {msg}")

def apply_slippage_and_fee(price, side="BUY"):
    """
    예시: BUY 시 가격을 약간 높이고, SELL 시 가격을 약간 낮춤
    수수료도 포함
    """
    s = SLIPPAGE_PCT/100.0
    f = FEE_PCT/100.0
    if side=="BUY":
        return price*(1 + s + f)
    else:
        return price*(1 - s - f)

def calculate_pnl(current_price):
    global entry_price
    if entry_price==0:
        return 0.0
    return (current_price - entry_price)/entry_price * 100.0 * LEVERAGE

def do_trading_step(model):
    """1회 트레이딩 로직 (간단)"""
    global current_position, entry_price

    if not os.path.isfile(HISTORICAL_CSV):
        log("No CSV found.")
        return

    df_all = pd.read_csv(HISTORICAL_CSV)
    if len(df_all)<60:
        log("Not enough bars for indicators.")
        return

    # 최근 200봉으로 인디케이터 계산
    df_recent = df_all.tail(200).copy()
    df_ind = add_indicators(df_recent)
    df_ind["ma_short"] = df_ind["close"].rolling(10).mean()
    df_ind["ma_long"]  = df_ind["close"].rolling(60).mean()
    df_ind.dropna(inplace=True)

    if df_ind.empty:
        log("Still not enough after dropna.")
        return

    # 피처
    features = [
        "close","volume","RSI14","MOM10","ROC10",
        "MACD_line","MACD_signal","MACD_hist",
        "ATR14","BB_mid","BB_up","BB_low",
        "ma_short","ma_long"
    ]
    last_row = df_ind.iloc[-1]
    raw_price = last_row["close"]
    X_latest = pd.DataFrame([last_row[features].values], columns=features)

    # 모델 예측
    signal_pred = model.predict(X_latest)[0]

    # 매수/매도 판단
    if signal_pred==1 and current_position==0:
        buy_price = apply_slippage_and_fee(raw_price, side="BUY")
        current_position=1
        entry_price=buy_price
        log(f"LONG ENTER at {buy_price:.2f}, raw={raw_price:.2f}")
    elif signal_pred==0 and current_position==1:
        sell_price = apply_slippage_and_fee(raw_price, side="SELL")
        pnl = (sell_price - entry_price)/entry_price*100.0 * LEVERAGE
        log(f"LONG EXIT at {sell_price:.2f}, PnL={pnl:.2f}% (raw={raw_price:.2f})")
        current_position=0
        entry_price=0.0

    # 스탑로스/익절
    if current_position==1:
        sim_sell = apply_slippage_and_fee(raw_price, side="SELL")
        pnl_now = (sim_sell - entry_price)/entry_price*100.0 * LEVERAGE
        if pnl_now<=STOP_LOSS_PERCENT:
            log(f"STOP-LOSS at {sim_sell:.2f}, PnL={pnl_now:.2f}%")
            current_position=0
            entry_price=0.0
        elif pnl_now>=TAKE_PROFIT_PERCENT:
            log(f"TAKE-PROFIT at {sim_sell:.2f}, PnL={pnl_now:.2f}%")
            current_position=0
            entry_price=0.0

def main():
    global LAST_TRAIN_TIME, current_model

    log("Starting main loop, 5min intervals for data + 1h for training.")

    while True:
        try:
            now = time.time()

            # (A) 5분마다 CSV 업데이트
            log("Updating historical_data.csv from OKX...")
            update_historical_csv("ETH-USDT", HISTORICAL_CSV)

            # (B) 1시간마다 모델 학습
            if (now - LAST_TRAIN_TIME) > TRAIN_INTERVAL or (current_model is None):
                log("Training model now...")
                final_acc = train_model(
                    csv_path=HISTORICAL_CSV,
                    model_path=MODEL_PATH,
                    model_type=MODEL_TYPE,
                    use_timeseries_cv=True
                )
                log(f"train_model => final_acc={final_acc}")
                LAST_TRAIN_TIME = now

                log("Loading newly trained model...")
                current_model = load_model(MODEL_TYPE, MODEL_PATH)

            # (C) 간단 매매 실행
            if current_model:
                do_trading_step(current_model)
            else:
                log("No model loaded, skipping trade step.")

            # 5분 후 반복
            log("Sleeping 5min..")
            time.sleep(FETCH_INTERVAL)

        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
            # 오류 발생 시 1분 쉬고 재시도
            time.sleep(60)

if __name__=="__main__":
    main()

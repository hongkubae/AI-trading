# main.py

import time
import os
import traceback
import pandas as pd
import numpy as np

from fetch_okx_data import update_historical_csv
from model_manager import train_model, load_model
from indicators import add_indicators

############################################
# 주기 설정
############################################
FETCH_INTERVAL = 300   # 5분마다 CSV 업데이트
TRADE_INTERVAL = 600   # 10분마다 트레이딩
TRAIN_INTERVAL = 3600  # 1시간마다 모델 학습

HISTORICAL_CSV = "data/historical_data.csv"
MODEL_TYPE = "RF"
MODEL_PATH = "best_model.pkl"

# 타임스탬프 기록
last_fetch_time = 0.0
last_trade_time = 0.0
last_train_time = 0.0

current_model = None

# 페이퍼 트레이딩 상태
current_position = 0  # 0=none, 1=long
entry_price = 0.0
STOP_LOSS_PERCENT = -5.0
TAKE_PROFIT_PERCENT = 10.0
LEVERAGE = 10

# 슬리피지, 수수료
SLIPPAGE_PCT = 0.02
FEE_PCT = 0.05

############################################
# 거래 기록 로깅 함수들
############################################
import os
import datetime

TRADES_CSV = "data/paper_trades.csv"

def init_trades_csv():
    """paper_trades.csv가 없으면 생성. header만 있는 빈 파일"""
    os.makedirs("data", exist_ok=True)
    if not os.path.isfile(TRADES_CSV):
        df = pd.DataFrame(columns=["timestamp","side","price","PnL"])
        df.to_csv(TRADES_CSV, index=False)
        print("[MAIN] paper_trades.csv initialized.")

def log_trade(side, price, pnl=0.0):
    """
    매매 이벤트가 발생했을 때, paper_trades.csv에 기록.
    side: "BUY","SELL","STOP-LOSS","TAKE-PROFIT" 등
    price: 체결 가격
    pnl: 수익률(%) (매도 시점 기준)
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {
        "timestamp": now_str,
        "side": side,
        "price": price,
        "PnL": pnl
    }
    df = pd.DataFrame([new_row])
    df.to_csv(TRADES_CSV, mode="a", header=False, index=False)

############################################
def log(msg):
    print(f"[MAIN] {msg}")

def apply_slippage_and_fee(price, side="BUY"):
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
    return (current_price - entry_price)/entry_price*100.0 * LEVERAGE

def do_trading_step(model):
    global current_position, entry_price

    if not os.path.isfile(HISTORICAL_CSV):
        log("No CSV found, skip trading.")
        return

    df_all = pd.read_csv(HISTORICAL_CSV)
    if len(df_all)<60:
        log("Not enough bars(60) for indicators => skip trade.")
        return

    df_recent = df_all.tail(200).copy()
    df_ind = add_indicators(df_recent)
    df_ind["ma_short"] = df_ind["close"].rolling(10).mean()
    df_ind["ma_long"]  = df_ind["close"].rolling(60).mean()
    df_ind.dropna(inplace=True)

    if df_ind.empty:
        log("Still not enough after dropna => skip trade.")
        return

    features = [
        "close","volume","RSI14","MOM10","ROC10",
        "MACD_line","MACD_signal","MACD_hist",
        "ATR14","BB_mid","BB_up","BB_low",
        "ma_short","ma_long"
    ]
    last_row = df_ind.iloc[-1]
    raw_price = last_row["close"]
    X_latest = pd.DataFrame([last_row[features].values], columns=features)

    signal_pred = model.predict(X_latest)[0]

    # 매수
    if signal_pred==1 and current_position==0:
        buy_price = apply_slippage_and_fee(raw_price, side="BUY")
        current_position=1
        entry_price=buy_price
        log(f"LONG ENTER at {buy_price:.2f}, raw={raw_price:.2f}")
        # 로깅
        log_trade(side="BUY", price=buy_price, pnl=0.0)

    # 매도 청산
    elif signal_pred==0 and current_position==1:
        sell_price = apply_slippage_and_fee(raw_price, side="SELL")
        pnl = (sell_price - entry_price)/entry_price*100.0 * LEVERAGE
        log(f"LONG EXIT at {sell_price:.2f}, PnL={pnl:.2f}% (raw={raw_price:.2f})")
        # 로깅
        log_trade(side="SELL", price=sell_price, pnl=pnl)
        current_position=0
        entry_price=0.0

    # Stop-Loss / Take-Profit
    if current_position==1:
        sim_sell_price = apply_slippage_and_fee(raw_price, side="SELL")
        pnl_now = (sim_sell_price - entry_price)/entry_price*100.0 * LEVERAGE
        if pnl_now<=STOP_LOSS_PERCENT:
            log(f"STOP-LOSS triggered at {sim_sell_price:.2f}, PnL={pnl_now:.2f}%")
            log_trade(side="STOP-LOSS", price=sim_sell_price, pnl=pnl_now)
            current_position=0
            entry_price=0.0
        elif pnl_now>=TAKE_PROFIT_PERCENT:
            log(f"TAKE-PROFIT triggered at {sim_sell_price:.2f}, PnL={pnl_now:.2f}%")
            log_trade(side="TAKE-PROFIT", price=sim_sell_price, pnl=pnl_now)
            current_position=0
            entry_price=0.0

def main():
    global last_fetch_time, last_trade_time, last_train_time
    global current_model

    # paper_trades.csv 초기화
    init_trades_csv()

    log("Starting main loop with separate intervals for fetch/trade/train.")

    # 루프 주기
    FETCH_INTERVAL = 300
    TRADE_INTERVAL = 600
    TRAIN_INTERVAL = 3600

    last_fetch_time = 0
    last_trade_time = 0
    last_train_time = 0

    while True:
        try:
            now = time.time()

            # (A) 5분마다 CSV 업데이트
            if (now - last_fetch_time)>=FETCH_INTERVAL:
                log("5min -> update_historical_csv")
                update_historical_csv("ETH-USDT", HISTORICAL_CSV)
                last_fetch_time = now

            # (B) 10분마다 트레이딩
            if current_model and (now - last_trade_time)>=TRADE_INTERVAL:
                log("10min -> do trading step")
                do_trading_step(current_model)
                last_trade_time = now

            # (C) 1시간마다 모델 학습
            if (now - last_train_time)>=TRAIN_INTERVAL or (current_model is None):
                log("1h -> train_model & load_model")
                final_acc = train_model(
                    csv_path=HISTORICAL_CSV,
                    model_path=MODEL_PATH,
                    model_type="RF",
                    use_timeseries_cv=True
                )
                log(f"[MAIN] train_model => final_acc={final_acc}")
                current_model = load_model("RF", MODEL_PATH)
                last_train_time = now

            time.sleep(1)

        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
            time.sleep(5)

if __name__=="__main__":
    main()

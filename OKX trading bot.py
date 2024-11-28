import numpy as np
import time
import hmac
import hashlib
import base64
import requests
import json
import datetime
from datetime import timezone
import logging
import os
from threading import Thread
import pandas as pd

# 환경 변수 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='leveraged_trading_log.log', filemode='a')

# OKX API 설정
API_KEY = 'edec8843-54c3-4cf9-ba7a-ecd92433bfb5'
API_SECRET = '80535AE66048DA42B7C65302B1A1AEC8'
PASSPHRASE = 'Rjsgml2153!'
BASE_URL = "https://www.okx.com"

# 수수료율 설정 (예시로 0.1%를 가정)
TAKER_FEE_RATE = 0.001
MAKER_FEE_RATE = 0.0005

# API와 연결하기 위한 시그니처 생성
def generate_signature(timestamp, method, request_path, body=''):
    message = f"{timestamp}{method}{request_path}{body}"
    hmac_key = base64.b64decode(API_SECRET)
    signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(signature.digest())

# API 호출을 위한 헤더 생성
def get_headers(method, request_path, body=''):
    timestamp = datetime.datetime.now(timezone.utc).isoformat()
    signature = generate_signature(timestamp, method, request_path, body)

    headers = {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': signature.decode(),
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'Content-Type': 'application/json'
    }
    return headers

# API 호출 함수
def call_api(method, endpoint, params=None, body=None):
    url = f"{BASE_URL}{endpoint}"
    headers = get_headers(method, endpoint, body)
    response = requests.request(method, url, headers=headers, params=params, data=body)
    if response.status_code == 200:
        return response.json().get('data', None)
    else:
        logging.error(f"API Error: {response.status_code}, {response.text}")
        return None

# 계좌 잔고 조회 함수
def get_account_balance(currency='USDT'):
    request_path = '/api/v5/account/balance'
    response = call_api('GET', request_path)
    if response:
        for balance in response:
            if balance['ccy'] == currency:
                return float(balance['availEq'])
    return 0.0

# 수수료를 고려한 주문 크기 계산 함수
def calculate_order_size(available_balance, fee_rate):
    # 수수료를 고려하여 주문 크기 계산 (잔고에서 수수료 차감)
    return available_balance / (1 + fee_rate)

# 주문 실행 함수
def place_order(inst_id, td_mode, side, ord_type, sz, pos_side, leverage):
    request_path = '/api/v5/trade/order'
    body = json.dumps({
        "instId": inst_id,
        "tdMode": td_mode,
        "side": side,
        "ordType": ord_type,
        "sz": sz,
        "posSide": pos_side,
        "lever": leverage
    })
    response = call_api('POST', request_path, body=body)
    if response:
        logging.info(f"Order response: {response}")
        print(f"Order placed: {side} {sz} {ord_type} at leverage {leverage} for {inst_id}")
    else:
        logging.error(f"Failed to place order for {inst_id}")
        print(f"Failed to place order: {side} {sz} {ord_type} at leverage {leverage} for {inst_id}")

# CSV 파일에서 최신 예측된 가격과 현재 가격 가져오기
def get_latest_prediction_from_csv(filename='predictions.csv'):
    try:
        df = pd.read_csv(filename)
        if not df.empty:
            latest_row = df.iloc[-1]
            predicted_price = latest_row['predicted_price']
            latest_price = latest_row['latest_price']
            return predicted_price, latest_price
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
    return None, None

# 트레이딩 로직
def trading_logic(inst_id='BTC-USDT-SWAP', leverage='10'):
    while True:
        # CSV 파일에서 최신 가격과 예측 가격 가져오기
        predicted_price, latest_price = get_latest_prediction_from_csv()

        if predicted_price is None or latest_price is None:
            logging.error("Failed to fetch latest or predicted price from CSV.")
            time.sleep(600)
            continue

        # 계좌의 사용 가능한 USDT 잔고 가져오기
        available_balance = get_account_balance()
        if available_balance <= 0:
            logging.error("No available balance to place an order.")
            time.sleep(60)
            continue

        # 시장가 주문으로 거래할 때 수수료를 고려한 주문 크기 계산
        sz = str(calculate_order_size(available_balance, TAKER_FEE_RATE))

        if predicted_price > latest_price:
            # 가격 상승 예상 -> 지정가 주문으로 롱 포지션 매수 시도
            place_order(inst_id, 'cross', 'buy', 'limit', sz, 'long', leverage)
            logging.info(f"Predicted price is higher than latest price. Placing LONG limit order with leverage {leverage}")

            # 지정가 주문이 체결되지 않으면 시장가 주문으로 전환 (예시로 60초 대기 후)
            time.sleep(60)
            # 지정가 주문이 체결되지 않은 경우 -> 시장가 주문으로 롱 포지션 매수
            place_order(inst_id, 'cross', 'buy', 'market', sz, 'long', leverage)
            logging.info(f"Limit order not filled. Placing LONG market order with leverage {leverage}")

        elif predicted_price < latest_price:
            # 가격 하락 예상 -> 지정가 주문으로 숏 포지션 매수 시도
            place_order(inst_id, 'cross', 'sell', 'limit', sz, 'short', leverage)
            logging.info(f"Predicted price is lower than latest price. Placing SHORT limit order with leverage {leverage}")

            # 지정가 주문이 체결되지 않으면 시장가 주문으로 전환 (예시로 60초 대기 후)
            time.sleep(60)
            # 지정가 주문이 체결되지 않은 경우 -> 시장가 주문으로 숏 포지션 매수
            place_order(inst_id, 'cross', 'sell', 'market', sz, 'short', leverage)
            logging.info(f"Limit order not filled. Placing SHORT market order with leverage {leverage}")

        # 10분 대기 후 반복
        time.sleep(600)

if __name__ == '__main__':
    # 레버리지 트레이딩 로직 실행
    trading_thread = Thread(target=trading_logic, args=('BTC-USDT-SWAP',))
    trading_thread.start()

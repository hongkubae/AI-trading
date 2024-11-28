import numpy as np
import pandas as pd
import time
import hmac
import hashlib
import base64
import requests
import json
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import logging
import os
from flask import Flask, request, jsonify
import threading

# 환경 변수 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading_log.log', filemode='a')

# OKX API 설정
API_KEY = 'edec8843-54c3-4cf9-ba7a-ecd92433bfb5'
API_SECRET = '80535AE66048DA42B7C65302B1A1AEC8'
PASSPHRASE = 'Rjsgml2153!'
BASE_URL = "https://www.okx.com"

# Flask 애플리케이션 설정
app = Flask(__name__)

# 웹훅 시그널 수신 상태를 저장하는 딕셔너리 (쓰레드 간 공유)
webhook_signal = {'ENTER_LONG': False, 'ENTER_SHORT': False}

# API와 연결하기 위한 시그니처 생성
def generate_signature(timestamp, method, request_path, body=''):
    message = f"{timestamp}{method}{request_path}{body}"
    hmac_key = base64.b64decode(API_SECRET)
    signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(signature.digest())

# API 호출을 위한 헤더 생성
def get_headers(method, request_path, body=''):
    timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
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

# 과거 데이터를 가져와 전처리하고 학습 데이터로 변환
def get_historical_data(inst_id, bar='1H', limit=100):
    endpoint = f"/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    return call_api('GET', endpoint)

# 데이터를 전처리하여 학습에 사용할 수 있는 형태로 변환
def preprocess_data(data):
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'confirmations', 'trade_count'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# 데이터에 기술적 지표를 추가하여 특성 강화
def calculate_technical_indicators(df):
    # RSI 계산
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD 계산
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 스토캐스틱 오실레이터 계산
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stochastic'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    # 볼린저 밴드 계산
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

    return df

# 데이터를 스케일링하여 학습에 적합한 형태로 변환
def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'volume', 'RSI', 'MACD', 'Signal', 'Stochastic', 'BB_Middle', 'BB_Upper', 'BB_Lower']].values)
    return scaled_data, scaler

# 시계열 데이터를 LSTM 모델의 입력 형태로 변환
def create_dataset(scaled_data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])  # Close price를 예측 목표로 설정
    X, y = np.array(X), np.array(y)
    return X, y

# 주문 실행 함수
def place_order(inst_id, td_mode, side, ord_type, sz, pos_side, stop_loss=None, take_profit=None):
    request_path = '/api/v5/trade/order'
    body = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": side,
        "ordType": ord_type,
        "sz": sz,
        "posSide": pos_side
    }
    # Stop Loss와 Take Profit 추가
    if stop_loss:
        body["slTriggerPx"] = stop_loss
    if take_profit:
        body["tpTriggerPx"] = take_profit

    response = call_api('POST', request_path, body=json.dumps(body))
    if response:
        logging.info(f"Order response: {response}")
    else:
        logging.error(f"Failed to place order for {inst_id}")

# 실시간 예측 및 시그널에 기반한 매매 수행
def predict_and_trade_with_signal(inst_id, model_path='trained_lstm_model_with_rl.h5', look_back=60):
    # 모델 불러오기 및 컴파일
    model = load_model(model_path)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    while True:
        # 실시간 데이터를 계속 업데이트
        data = get_historical_data(inst_id, bar='1H', limit=200)
        if data:
            df = preprocess_data(data)
            df = calculate_technical_indicators(df)
            df.dropna(inplace=True)

            _, scaler = scale_data(df)
            scaled_data = scaler.transform(df[['close', 'volume', 'RSI', 'MACD', 'Signal', 'Stochastic', 'BB_Middle', 'BB_Upper', 'BB_Lower']].values)
            X, _ = create_dataset(scaled_data, look_back)

            predicted_price = model.predict(X[-1].reshape(1, look_back, X.shape[2]))
            predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((predicted_price.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0][0]
            real_price = df['close'].values[-1]

            # 매매 판단 (예측 결과와 웹훅 신호 모두 고려)
            if predicted_price > real_price:
                logging.info(f"AI 예측에 따르면 가격 상승 예상: {predicted_price}")
                # TradingView 시그널이 매수일 경우 매수 진행
                if webhook_signal.get('ENTER_LONG', False):
                    place_order(inst_id, 'cross', 'buy', 'market', '1', 'long')
                    logging.info(f"Buy order placed for {inst_id} at predicted price: {predicted_price}")
            elif predicted_price < real_price:
                logging.info(f"AI 예측에 따르면 가격 하락 예상: {predicted_price}")
                # TradingView 시그널이 매도일 경우 매도 진행
                if webhook_signal.get('ENTER_SHORT', False):
                    place_order(inst_id, 'cross', 'sell', 'market', '1', 'short')
                    logging.info(f"Sell order placed for {inst_id} at predicted price: {predicted_price}")

        # 실시간으로 데이터를 1시간마다 가져와 예측
        time.sleep(3600)

# 웹훅 처리 함수
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.json
        if 'action' in data and 'instrument' in data:
            inst_id = data['instrument']
            action = data['action']

            # TradingView에서 받은 시그널에 따른 상태 업데이트
            if action == 'ENTER_LONG':
                webhook_signal['ENTER_LONG'] = True
            elif action == 'ENTER_SHORT':
                webhook_signal['ENTER_SHORT'] = True
            elif action == 'EXIT_LONG':
                webhook_signal['ENTER_LONG'] = False
            elif action == 'EXIT_SHORT':
                webhook_signal['ENTER_SHORT'] = False

            logging.info(f"Webhook received: {action} for {inst_id}")
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "error": "Invalid data"}), 400
    except Exception as e:
        logging.error(f"Error processing webhook: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Flask 서버와 예측 매매 함수의 동시 실행
if __name__ == '__main__':
    # 웹훅 서버 실행
    webhook_thread = threading.Thread(target=lambda: app.run(port=5000))
    webhook_thread.start()

    # 실시간 예측 매매 실행
    predict_and_trade_thread = threading.Thread(target=predict_and_trade_with_signal, args=('BTC-USDT-SWAP',))
    predict_and_trade_thread.start()
import time
import requests
import json
import hmac
import base64
import datetime
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import random
import os
import csv
from sklearn.feature_selection import SelectKBest, f_classif
import gym
from stable_baselines3 import PPO
from ta.trend import SMAIndicator, MACD  # 기존에 있던 SMAIndicator 임포트에 MACD를 추가

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading_bot.log', filemode='a')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # 터미널 출력 최소화
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# OKX API 설정
API_KEY = 'edec8843-54c3-4cf9-ba7a-ecd92433bfb5'
API_SECRET = '80535AE66048DA42B7C65302B1A1AEC8'
PASSPHRASE = 'Rjsgml2153!'
BASE_URL = "https://www.okx.com"

# 머신러닝 모델 경로
MODEL_PATH = 'trading_model.pkl'

# 거래 기록 CSV 파일 초기화
def initialize_trade_history_csv():
    if not os.path.exists('trade_history.csv'):
        with open('trade_history.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Symbol', 'Action', 'Price', 'Quantity', 'Balance'])
        logging.info("거래 기록 CSV 파일이 생성되었습니다.")

# 머신러닝 모델 생성 및 저장 (자동 학습 개선 기능 추가)
def create_and_save_model():
    data = []
    try:
        with open('trade_history.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 건너뛰기 
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        logging.warning("거래 기록 CSV 파일을 찾을 수 없습니다. 가상 데이터를 사용합니다.")

    if len(data) < 10:
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=42)
    else:
        df = pd.DataFrame(data, columns=['Timestamp', 'Symbol', 'Action', 'Price', 'Quantity', 'Balance'])
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0.0)
        X = df[['Price', 'Quantity', 'Balance']].values
        y = (df['Action'] == 'buy').astype(int).values

        # 데이터 피처링 추가
        selector = SelectKBest(score_func=f_classif, k=2)
        X = selector.fit_transform(X, y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump({'model': model, 'scaler': scaler}, MODEL_PATH)
    logging.info(f"모델이 '{MODEL_PATH}'에 저장되었습니다.")

    accuracy = model.score(X_test, y_test)
    logging.info(f"모델 테스트 정확도: {accuracy:.2f}")

# 모델 자동 학습 및 업데이트 기능
def auto_train_model():
    logging.info("모델 자동 학습 시작")
    create_and_save_model()
    logging.info("모델 자동 학습 완료")

# 강화 학습 환경 설정 및 모델 생성
def reinforcement_learning():
    logging.info("강화 학습 모델 학습 시작")

    # 사용자 정의 환경 구현 필요
    # 거래 데이터를 사용해 주문 체결, 슬리피지, 거래 수수료 등을 포함한 현실적인 트레이딩 환경 개발
    class TradingEnv(gym.Env):
        def __init__(self):
            super(TradingEnv, self).__init__()
            # Define action and observation space
            self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
            self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)
            self.current_step = 0
            # Load historical data or initialize simulation data
            self.data = self.load_data()

        def load_data(self):
            data = []
            try:
                with open('trade_history.csv', mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)  # 헤더 건너뛰기
                    for row in reader:
                        data.append(row)
            except FileNotFoundError:
                logging.warning("거래 기록 CSV 파일을 찾을 수 없습니다. 가상 데이터를 사용합니다.")
            return data

        def reset(self):
            self.current_step = 0
            return self._next_observation()

        def _next_observation(self):
            # 현재 스텝의 데이터 반환
            return np.array([random.uniform(0, 100) for _ in range(10)])

        def step(self, action):
            reward = 0  # 보상 계산 필요
            self.current_step += 1
            done = self.current_step >= len(self.data)
            obs = self._next_observation()
            return obs, reward, done, {}

        def render(self, mode='human'):
            pass

    env = TradingEnv()  # 실제 트레이딩 환경 구현 필요
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("reinforcement_trading_model")
    logging.info("강화 학습 모델 학습 완료 및 저장")

# 모델 성능 모니터링 및 피드백 루프 추가
def monitor_model_performance():
    logging.info("모델 성능 모니터링 시작")
    # 모델의 성능을 주기적으로 평가하고 성능 지표를 기록하여 모델 개선에 활용
    data = []
    try:
        with open('trade_history.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 건너뛰기
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        logging.warning("거래 기록 CSV 파일을 찾을 수 없습니다. 모니터링을 건너뜁니다.")
        return

    if len(data) >= 10:
        df = pd.DataFrame(data, columns=['Timestamp', 'Symbol', 'Action', 'Price', 'Quantity', 'Balance'])
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0.0)
        X = df[['Price', 'Quantity', 'Balance']].values
        y = (df['Action'] == 'buy').astype(int).values

        try:
            loaded_data = joblib.load(MODEL_PATH)
            if isinstance(loaded_data, dict) and 'scaler' in loaded_data and 'model' in loaded_data:
                scaler = loaded_data['scaler']
                model = loaded_data['model']
                X = scaler.transform(X)
                accuracy = model.score(X, y)
                logging.info(f"현재 모델 정확도: {accuracy:.2f}")
            else:
                logging.error("로드된 데이터가 올바른 형식이 아닙니다. 모델 파일을 다시 확인하세요.")
        except Exception as e:
            logging.error(f"모델 로드 중 오류 발생: {e}")

# API 요청 헤더 생성
def create_headers(method, request_path, body=""):
    timestamp = datetime.datetime.utcnow().isoformat("T", "milliseconds") + "Z"
    body = json.dumps(body) if body else ""
    sign = base64.b64encode(hmac.new(
        API_SECRET.encode(),
        (timestamp + method + request_path + body).encode(),
        digestmod='sha256'
    ).digest()).decode()
    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }
    return headers

# 과거 데이터 가져오기
def get_historical_data(symbol, bar="1H", limit=100):
    print(f"히스토리 데이터 요청 - 심볼: {symbol}, 바: {bar}, 제한: {limit}")
    logging.debug(f"히스토리 데이터 요청 - 심볼: {symbol}, 바: {bar}, 제한: {limit}")
    endpoint = f"/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    url = BASE_URL + endpoint
    headers = create_headers("GET", endpoint)
    response = requests.get(url, headers=headers)

    print(f"API 응답 상태 코드: {response.status_code}")
    print(f"API 응답 내용: {response.text}")

    if response.status_code == 200:
        data = response.json()
        logging.debug(f"API 응답 데이터: {data}")
        if not isinstance(data, dict) or 'data' not in data:
            logging.error("API 응답이 올바르지 않거나 'data' 키가 없습니다.")
            return []

        candles = []
        for item in data['data']:
            try:
                candles.append({
                    "close": float(item[4]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "volume": float(item[5])
                })
            except (ValueError, TypeError) as e:
                logging.error(f"데이터 변환 오류: {e}, 데이터: {item}")
                continue

        print(f"히스토리 데이터 수신 성공 - 데이터 개수: {len(candles)}")
        return candles
    elif response.status_code == 429:
        logging.warning("API 요청이 너무 많습니다. 잠시 대기 후 다시 시도합니다.")
        time.sleep(60)
        return []
    elif response.status_code == 401:
        logging.error("인증 오류: API 키 또는 비밀번호가 잘못되었습니다.")
        return []
    else:
        logging.error(f"API 요청 실패: 상태 코드 {response.status_code}, 응답 내용: {response.text}")
        return []

# 포지션 상태 확인
def check_position_status(symbol, pos_side):
    logging.debug(f"포지션 상태 확인 - 심볼: {symbol}, 방향: {pos_side}")
    endpoint = f"/api/v5/account/positions?instId={symbol}"
    url = BASE_URL + endpoint
    headers = create_headers("GET", endpoint)
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        logging.debug(f"포지션 응답 데이터: {data}")
        for position in data.get('data', []):
            if position['posSide'] == pos_side:
                return float(position['pos'])
    else:
        logging.error(f"포지션 상태 확인 실패: 상태 코드 {response.status_code}, 응답 내용: {response.text}")
    return 0

# 간단한 매수/매도 스트래티지
def simple_strategy(data):
    df = pd.DataFrame(data)
    logging.debug(f"전략 데이터프레임 생성 - 데이터 길이: {len(df)}")
    short_ma = SMAIndicator(df['close'], window=10).sma_indicator().iloc[-1]
    long_ma = SMAIndicator(df['close'], window=50).sma_indicator().iloc[-1]
    rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]
    macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    macd_value = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]

    logging.debug(f"단기 MA: {short_ma}, 장기 MA: {long_ma}, RSI: {rsi}, MACD: {macd_value}, MACD 시그널: {macd_signal}")

    # RSI 값 동적 최적화
    buy_threshold = 45 + random.uniform(-5, 5)  # 동적으로 변경되는 매수 임계값
    sell_threshold = 55 + random.uniform(-5, 5)  # 동적으로 변경되는 매도 임계값

    if short_ma > long_ma and rsi < buy_threshold and macd_value > macd_signal:
        logging.info("매수 신호 발생!")
        return "buy"
    elif short_ma < long_ma and rsi > sell_threshold and macd_value < macd_signal:
        logging.info("매도 신호 발생! (숏 포지션)")
        return "sell"
    else:
        logging.info("신호 없음, 대기 중...")
        return "hold"

# 거래 기록을 CSV에 저장하는 함수 추가
def log_trade_to_csv(trade_data):
    logging.debug(f"거래 기록 저장 - 데이터: {trade_data}")
    with open('trade_history.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'Symbol', 'Action', 'Price', 'Quantity', 'Balance'])
        writer.writerow([
            trade_data['timestamp'], trade_data['symbol'], trade_data['action'],
            trade_data['price'], trade_data['quantity'], trade_data['balance']
        ])

# 백테스트 기능 추가
def backtest_strategy(data, strategy_func):
    df = pd.DataFrame(data)
    balance = 10000
    position = 0
    transaction_fee_rate = 0.001
    slippage_rate = 0.001
    trade_history = []

    for i in range(len(df)):
        logging.debug(f"백테스트 루프 - 인덱스: {i}, 잔액: {balance}, 포지션: {position}")
        signal = strategy_func(df.iloc[:i+1])
        close_price = df['close'].iloc[i] * (1 + slippage_rate if signal == 'buy' else 1 - slippage_rate)
        if signal == 'buy' and balance > 0:
            position = balance / close_price * (1 - transaction_fee_rate)
            balance = 0
            trade_history.append({'action': 'buy', 'price': close_price, 'position': position, 'balance': balance})
            logging.info(f"백테스트 매수: 가격 {close_price}, 포지션 {position}")
        elif signal == 'sell' and position > 0:
            balance = position * close_price * (1 - transaction_fee_rate)
            position = 0
            trade_history.append({'action': 'sell', 'price': close_price, 'position': position, 'balance': balance})
            logging.info(f"백테스트 매도: 가격 {close_price}, 잔액 {balance}")
    
    final_balance = balance + (position * df['close'].iloc[-1] * (1 - transaction_fee_rate))
    logging.info(f"백테스트 최종 잔액: {final_balance:.2f}")
    
    print(f"백테스트 완료 - 최종 잔액: {final_balance:.2f}")
    for trade in trade_history:
        print(f"동작: {trade['action']}, 가격: {trade['price']}, 포지션: {trade['position']}, 잔액: {trade['balance']}")

    return final_balance

# 백테스트 후 매수/매도 스트래티지 진행
def run_trading_bot(symbol):
    # 거래 기록 CSV 파일 초기화
    initialize_trade_history_csv()

    print("트레이딩 봇 시작")
    logging.info(f"트레이딩 봇 시작 - 심볼: {symbol}")
    data = get_historical_data(symbol)
    if data:
        print("히스토리 데이터 수신 성공")
        backtest_strategy(data, simple_strategy)
    else:
        print("히스토리 데이터 수신 실패")
        logging.error("히스토리 데이터를 가져오지 못했습니다. 백테스트를 건너뜁니다.")

    # 자동 학습 주기 설정 (매일 1회 학습)
    next_train_time = datetime.datetime.now() + datetime.timedelta(days=1)

    while True:
        print("실시간 데이터 가져오기 시도")
        logging.info("실시간 데이터 가져오기")
        data = get_historical_data(symbol)
        if data:
            print("실시간 데이터 수신 성공")
            signal = simple_strategy(data)
            logging.info(f"생성된 신호: {signal}")
            print(f"생성된 신호: {signal}")
            if signal == "buy":
                open_position(symbol, side="buy", size=1)
            elif signal == "sell":
                close_position(symbol, side="sell", size=1)
        else:
            print("실시간 데이터 수신 실패")
            logging.error("실시간 데이터를 가져오지 못했습니다.")
        
        # 자동 학습 시간 도래 시 모델 학습
        if datetime.datetime.now() >= next_train_time:
            auto_train_model()
            next_train_time = datetime.datetime.now() + datetime.timedelta(days=1)
        
        # 모델 성능 모니터링
        monitor_model_performance()
        
        time.sleep(10)  # 기존 30초에서 10초로 변경

# 포지션 열기 (재시도 로직 추가)
def open_position(symbol, side, pos_side=None, price=None, retries=3, retry_delay=5):
    # 지갑 잔액 확인
    balance = get_wallet_balance(symbol.split('-')[0])
    if balance is None or balance <= 0:
        logging.error("지갑 잔액이 부족하여 포지션을 열 수 없습니다.")
        return

    size = balance  # 지갑 잔액 최대치로 주문 사이즈 설정

    if side == "buy":
        pos_side = "long"
    elif side == "sell":
        pos_side = "short"

    logging.info(f"포지션 열기 시도 - 심볼: {symbol}, 방향: {side}, 수량: {size}, 가격: {price}")
    # 나머지 기존 로직은 그대로 유지

    logging.info(f"포지션 열기 시도 - 심볼: {symbol}, 방향: {side}, 수량: {size}, 가격: {price}")
    endpoint = "/api/v5/trade/order"
    url = BASE_URL + endpoint
    order_type = "limit" if price else "market"
    body = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": side,
        "ordType": order_type,
        "sz": str(size),
        "posSide": pos_side
    }
    if price:
        body["px"] = str(price)

    headers = create_headers("POST", endpoint, body)

    for attempt in range(retries):
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 200:
            response_data = response.json()
            if response_data['code'] == '0':
                logging.info(f"{side} 포지션 오픈 성공 ({pos_side}): {response_data}")
                trade_data = {
                    'timestamp': datetime.datetime.now(),
                    'symbol': symbol,
                    'action': side,
                    'price': price if price else 0.0,
                    'quantity': size,
                    'balance': check_position_status(symbol, pos_side)
                }
                log_trade_to_csv(trade_data)
                print(f"{side.capitalize()} 포지션 오픈 성공 - 심볼: {symbol}, 가격: {trade_data['price']}, 수량: {trade_data['quantity']}")
                return
            else:
                logging.error(f"포지션 오픈 실패 - 오류 코드: {response_data['code']}, 메시지: {response_data['msg']}")
                if response_data['msg'].lower() == 'insufficient balance':
                    logging.error("잔액 부족으로 포지션을 열 수 없습니다.")
                    break
        else:
            logging.error(f"포지션 오픈 실패 (시도 {attempt + 1}/{retries}): {response.text}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

# 포지션 청산 (포지션 상태 확인 추가)
def close_position(symbol, side, size, pos_side=None, price=None):
    if side == "buy":
        pos_side = "short"
    elif side == "sell":
        pos_side = "long"
    logging.info(f"포지션 청산 시도 - 심볼: {symbol}, 방향: {side}, 수량: {size}, 가격: {price}")
    position_status = check_position_status(symbol, pos_side)
    if position_status is None or position_status <= 0:
        logging.warning(f"포지션을 청산할 수 없습니다. 현재 포지션 상태가 유효하지 않습니다 (posSide: {pos_side}).")
        print(f"포지션 청산 실패 - 현재 유효한 포지션이 없습니다 (심볼: {symbol}, 방향: {pos_side})")
        return

    endpoint = "/api/v5/trade/order"
    url = BASE_URL + endpoint
    order_type = "limit" if price else "market"
    body = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": side,
        "ordType": order_type,
        "sz": str(size),
        "posSide": pos_side
    }
    if price:
        body["px"] = str(price)

    headers = create_headers("POST", endpoint, body)
    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        response_data = response.json()
        if response_data['code'] == '0':
            logging.info(f"{side} 포지션 청산 성공 ({pos_side}): {response_data}")
            trade_data = {
                'timestamp': datetime.datetime.now(),
                'symbol': symbol,
                'action': side,
                'price': price if price else 0.0,
                'quantity': size,
                'balance': check_position_status(symbol, pos_side)
            }
            log_trade_to_csv(trade_data)
            print(f"{side.capitalize()} 포지션 청산 성공 - 심볼: {symbol}, 가격: {trade_data['price']}, 수량: {trade_data['quantity']}")
        else:
            logging.error(f"포지션 청산 실패 - 오류 코드: {response_data['code']}, 메시지: {response_data['msg']}")
    else:
        logging.error(f"포지션 청산 실패: {response.text}")

if __name__ == "__main__":
    symbol = "BTC-USDT"
    run_trading_bot(symbol)

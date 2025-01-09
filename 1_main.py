import ccxt
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv
import requests
import logging
from ta.momentum import StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==========================
# 1. 환경 변수에서 API 키 로드
# ==========================
load_dotenv()  # .env 파일 로드

OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_API_PASSWORD = os.getenv("OKX_API_PASSWORD")  # Passphrase

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================
# 2. 로깅 설정
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

# ==========================
# 3. 텔레그램 알림 함수
# ==========================
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram 봇 토큰 또는 챗 ID가 설정되지 않았습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info("✅ Telegram 메시지 전송 성공!")
        else:
            logging.warning(f"⚠️ Telegram 메시지 전송 실패: {response.text}")
    except Exception as e:
        logging.warning(f"⚠️ Telegram 메시지 전송 중 에러 발생: {e}")

# ==========================
# 4. 지표 계산 클래스
# ==========================
class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame, higher_df: pd.DataFrame, exchange):
        self.df = df
        self.higher_df = higher_df
        self.exchange = exchange

    def calculate_poc(self, df: pd.DataFrame, bin_size: float = 10.0) -> float:
        """
        포인트 오브 컨트롤(POC)을 계산하는 함수.
        """
        try:
            bins = np.arange(df['Low'].min(), df['High'].max() + bin_size, bin_size)
            df['Price_Bin'] = pd.cut(df['Close'], bins=bins, include_lowest=True, right=False)
            
            volume_profile = df.groupby('Price_Bin')['Volume'].sum().reset_index()
            volume_profile['Bin_Mid'] = volume_profile['Price_Bin'].apply(lambda x: x.mid)
            
            poc = volume_profile.loc[volume_profile['Volume'].idxmax()]['Bin_Mid']
            return poc
        except Exception as e:
            logging.error(f"POC 계산 중 오류 발생: {e}")
            return 0.0

    def identify_pivots(self, df: pd.DataFrame, window: int = 5) -> (list, list):
        """
        피봇 고점과 저점을 식별하는 함수.
        """
        try:
            pivots_high = []
            pivots_low = []
            for i in range(window, len(df) - window):
                high = df['High'][i]
                if all(high > df['High'][i - window:i]) and all(high > df['High'][i + 1:i + window + 1]):
                    pivots_high.append((df.index[i], high))
                low = df['Low'][i]
                if all(low < df['Low'][i - window:i]) and all(low < df['Low'][i + 1:i + window + 1]):
                    pivots_low.append((df.index[i], low))
            return pivots_high, pivots_low
        except Exception as e:
            logging.error(f"피봇 포인트 식별 중 오류 발생: {e}")
            return [], []

    def draw_trendlines(self, pivots_high: list, pivots_low: list):
        """
        피봇 포인트를 기반으로 추세선을 그리는 함수.
        """
        try:
            plt.figure(figsize=(14,7))
            plt.plot(self.df['Close'], label='Close Price')

            for pivot in pivots_high:
                plt.scatter(pivot[0], pivot[1], color='red', marker='^')
            
            for pivot in pivots_low:
                plt.scatter(pivot[0], pivot[1], color='green', marker='v')
            
            if len(pivots_high) >= 2:
                plt.plot([pivots_high[0][0], pivots_high[-1][0]], [pivots_high[0][1], pivots_high[-1][1]], color='red', linestyle='--', label='Down TrendLine')
            
            if len(pivots_low) >= 2:
                plt.plot([pivots_low[0][0], pivots_low[-1][0]], [pivots_low[0][1], pivots_low[-1][1]], color='green', linestyle='--', label='Up TrendLine')
            
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"추세선 그리기 중 오류 발생: {e}")

    def analyze_market_structure(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        시장 구조를 분석하여 고점 고점, 저점 저점을 식별하는 함수.
        """
        try:
            df['Market_Structure'] = 'Neutral'
            pivots_high, pivots_low = self.identify_pivots(df, window=window)
            
            for i in range(1, len(pivots_high)):
                if pivots_high[i][1] > pivots_high[i-1][1]:
                    df.loc[pivots_high[i][0], 'Market_Structure'] = 'Higher High'
                else:
                    df.loc[pivots_high[i][0], 'Market_Structure'] = 'Lower High'
            
            for i in range(1, len(pivots_low)):
                if pivots_low[i][1] < pivots_low[i-1][1]:
                    df.loc[pivots_low[i][0], 'Market_Structure'] = 'Lower Low'
                else:
                    df.loc[pivots_low[i][0], 'Market_Structure'] = 'Higher Low'
            
            return df
        except Exception as e:
            logging.error(f"시장 구조 분석 중 오류 발생: {e}")
            return df

    def identify_fvg(self, df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
        """
        페어 밸류 갭(FVG)을 식별하는 함수.
        """
        try:
            df['FVG'] = False
            for i in range(window, len(df) - window):
                if df['Close'][i-1] > df['Open'][i-1]:
                    # 강세 갭: 이전 캔들이 상승 캔들이고, 현재 캔들이 이전 캔들의 저가보다 낮을 때
                    if df['Low'][i] > df['High'][i - window]:
                        df.loc[df.index[i], 'FVG'] = True
                elif df['Close'][i-1] < df['Open'][i-1]:
                    # 약세 갭: 이전 캔들이 하락 캔들이고, 현재 캔들이 이전 캔들의 고가보다 낮을 때
                    if df['High'][i] < df['Low'][i - window]:
                        df.loc[df.index[i], 'FVG'] = True
            return df
        except Exception as e:
            logging.error(f"FVG 식별 중 오류 발생: {e}")
            return df

    def plot_fvg(self, df: pd.DataFrame):
        """
        페어 밸류 갭(FVG)을 시각화하는 함수.
        """
        try:
            fvg = df[df['FVG']]
            plt.figure(figsize=(14,7))
            plt.plot(df['Close'], label='Close Price')
            plt.scatter(fvg.index, fvg['Close'], color='purple', marker='o', label='FVG')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"FVG 시각화 중 오류 발생: {e}")

    def identify_liquidity_levels(self, df: pd.DataFrame, higher_timeframe_df: pd.DataFrame, window: int = 5) -> list:
        """
        유동성 레벨을 식별하는 함수.
        """
        try:
            pivots_high, pivots_low = self.identify_pivots(higher_timeframe_df, window=window)
            liquidity_levels = [pivot[1] for pivot in pivots_high + pivots_low]
            return liquidity_levels
        except Exception as e:
            logging.error(f"유동성 레벨 식별 중 오류 발생: {e}")
            return []

    def get_key_levels(self, symbol: str, timeframes: list) -> list:
        """
        다양한 시간대에서 주요 레벨을 가져오는 함수.
        """
        key_levels = []
        try:
            for tf in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=tf, limit=1)
                if ohlcv:
                    latest = ohlcv[-1]
                    high = latest[2]
                    low = latest[3]
                    open_ = latest[1]
                    close = latest[4]
                    key_levels.extend([high, low, open_, close])
        except Exception as e:
            logging.error(f"{symbol} 주요 레벨 가져오기 중 오류 발생: {e}")
        return key_levels

    def plot_key_levels(self, df: pd.DataFrame, key_levels: list):
        """
        주요 레벨을 시각화하는 함수.
        """
        try:
            plt.figure(figsize=(14,7))
            plt.plot(df['Close'], label='Close Price')
            for level in key_levels:
                plt.axhline(y=level, color='magenta', linestyle='--', linewidth=1)
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"주요 레벨 시각화 중 오류 발생: {e}")

    def identify_order_blocks(self, df: pd.DataFrame, volume_threshold: float = 1.5) -> pd.DataFrame:
        """
        오더 블록을 식별하는 함수.
        """
        try:
            mean_volume = df['Volume'].mean()
            df['Order_Block'] = 0
            for i in range(1, len(df) - 1):
                if df['Volume'][i] > mean_volume * volume_threshold:
                    # 매수 오더 블록
                    if df['Close'][i] > df['Open'][i]:
                        df.loc[df.index[i], 'Order_Block'] = 1
                    # 매도 오더 블록
                    elif df['Close'][i] < df['Open'][i]:
                        df.loc[df.index[i], 'Order_Block'] = -1
            return df
        except Exception as e:
            logging.error(f"오더 블록 식별 중 오류 발생: {e}")
            return df

    def plot_order_blocks(self, df: pd.DataFrame):
        """
        오더 블록을 시각화하는 함수.
        """
        try:
            plt.figure(figsize=(14,7))
            plt.plot(df['Close'], label='Close Price')

            buy_blocks = df[df['Order_Block'] == 1]
            sell_blocks = df[df['Order_Block'] == -1]

            plt.scatter(buy_blocks.index, buy_blocks['Close'], color='green', marker='^', label='Buy Order Block')
            plt.scatter(sell_blocks.index, sell_blocks['Close'], color='red', marker='v', label='Sell Order Block')

            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"오더 블록 시각화 중 오류 발생: {e}")

    def identify_displacement(self, df: pd.DataFrame, window: int = 100, multiplier: float = 1.0) -> pd.DataFrame:
        """
        변위를 식별하는 함수.
        """
        try:
            # 캔들 범위 계산
            df['Candle_Range'] = df.apply(lambda row: abs(row['Close'] - row['Open']), axis=1)
            # 표준 편차 계산
            df['Range_STD'] = df['Candle_Range'].rolling(window=window).std()
            # 변위 식별
            df['Displacement'] = df['Candle_Range'] > (df['Range_STD'] * multiplier)
            return df
        except Exception as e:
            logging.error(f"변위 식별 중 오류 발생: {e}")
            return df

    def plot_displacement(self, df: pd.DataFrame):
        """
        변위를 시각화하는 함수.
        """
        try:
            displacement = df[df['Displacement']]
            plt.figure(figsize=(14,7))
            plt.plot(df['Close'], label='Close Price')
            plt.scatter(displacement.index, displacement['Close'], color='yellow', marker='*', label='Displacement')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"변위 시각화 중 오류 발생: {e}")

    def calculate_indicators(self) -> (pd.DataFrame, list, list, list, list):
        """
        모든 지표를 계산하고 DataFrame을 반환하는 함수.
        """
        try:
            # EMA200
            ema200 = EMAIndicator(close=self.df['Close'], window=200)
            self.df['EMA200'] = ema200.ema_indicator()

            # ATR
            atr = AverageTrueRange(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=14)
            self.df['ATR'] = atr.average_true_range()

            # Bollinger Bands
            bb = BollingerBands(close=self.df['Close'], window=20, window_dev=2)
            self.df['BB_upper'] = bb.bollinger_hband()
            self.df['BB_lower'] = bb.bollinger_lband()

            # StochRSI
            stochrsi = StochRSIIndicator(close=self.df['Close'], window=14, smooth1=3, smooth2=3)
            self.df['StochRSI_k'] = stochrsi.stochrsi_k()
            self.df['StochRSI_d'] = stochrsi.stochrsi_d()

            # MACD
            macd = MACD(close=self.df['Close'], window_slow=26, window_fast=12, window_sign=9)
            self.df['MACD'] = macd.macd()
            self.df['MACD_signal'] = macd.macd_signal()
            self.df['MACD_hist'] = macd.macd_diff()

            # ADX
            adx = ADXIndicator(high=self.df['High'], low=self.df['Low'], close=self.df['Close'], window=14)
            self.df['ADX'] = adx.adx()

            # POC 계산
            self.df['POC'] = self.calculate_poc(self.df, bin_size=self.df['ATR'].iloc[-1] * 1.0)

            # 피봇 포인트 식별
            pivots_high, pivots_low = self.identify_pivots(self.df, window=5)
            self.df['Pivots_High'] = 0
            self.df['Pivots_Low'] = 0
            for pivot in pivots_high:
                self.df.loc[pivot[0], 'Pivots_High'] = pivot[1]
            for pivot in pivots_low:
                self.df.loc[pivot[0], 'Pivots_Low'] = pivot[1]

            # 추세선 그리기 (시각화는 별도로 수행)
            self.draw_trendlines(pivots_high, pivots_low)

            # 시장 구조 분석
            self.df = self.analyze_market_structure(self.df, window=5)

            # FVG 식별
            self.df = self.identify_fvg(self.df, window=2)

            # 유동성 레벨 식별
            liquidity_levels = self.identify_liquidity_levels(self.df, self.higher_df, window=5)

            # 주요 레벨 식별
            timeframes = ["1d", "1w", "1M"]  # 일간, 주간, 월간
            key_levels = self.get_key_levels(self.df.name, timeframes)

            # 오더 블록 식별
            self.df = self.identify_order_blocks(self.df, volume_threshold=1.5)

            # 변위 식별
            self.df = self.identify_displacement(self.df, window=100, multiplier=1.0)

            self.df['Liquidity_Levels'] = pd.Series([liquidity_levels for _ in range(len(self.df))])

            self.df.dropna(inplace=True)
            return self.df, liquidity_levels, key_levels, pivots_high + pivots_low, self.df['Displacement'].tolist()
        except Exception as e:
            logging.error(f"지표 계산 중 에러 발생: {e}")
            return pd.DataFrame(), [], [], [], []

# ==========================
# 5. 개선된 Volume Profile 계산 함수
# ==========================
def calculate_volume_profile(df: pd.DataFrame, atr: float, bin_size_multiplier: float = 1.0, value_area_percent: float = 70.0):
    """
    개선된 Volume Profile 계산 함수.
    """
    try:
        min_price = df['Low'].min()
        max_price = df['High'].max()
        
        logging.info(f"Volume Profile 계산: min_price={min_price}, max_price={max_price}, ATR={atr}, bin_size_multiplier={bin_size_multiplier}, value_area_percent={value_area_percent}")
        
        # bin_size 계산
        bin_size = atr * bin_size_multiplier
        if bin_size <= 0:
            logging.error("bin_size must be positive.")
            raise ValueError("bin_size must be positive.")
        
        # 가격 구간 설정
        bins = np.arange(min_price, max_price + bin_size, bin_size)
        if bins.size == 0:
            logging.warning("Generated bins array is empty. Adjusting bin_size.")
            bin_size = 1.0  # 최소 bin_size로 조정
            bins = np.arange(min_price, max_price + bin_size, bin_size)
        
        # 거래량 집계
        df['Up_Volume'] = np.where(df['Close'] >= df['Open'], df['Volume'], 0)
        df['Down_Volume'] = np.where(df['Close'] < df['Open'], df['Volume'], 0)
    
        # 가격 구간에 따른 거래량 집계
        df['Price_Bin'] = pd.cut(df['Close'], bins=bins, include_lowest=True, right=False)
        volume_profile = df.groupby('Price_Bin').agg({'Up_Volume': 'sum', 'Down_Volume': 'sum'}).reset_index()
    
        # Total Volume 계산
        volume_profile['Total_Volume'] = volume_profile['Up_Volume'] + volume_profile['Down_Volume']
        total_volume = volume_profile['Total_Volume'].sum()
    
        if volume_profile.empty or total_volume == 0:
            logging.warning("Volume profile is empty or total volume is zero.")
            return {'POC': 0, 'VAH': 0, 'VAL': 0}
    
        # POC 계산
        poc_row = volume_profile.loc[volume_profile['Total_Volume'].idxmax()]
        poc = poc_row['Price_Bin'].mid
    
        # VAH 및 VAL 계산
        sorted_volume = volume_profile.sort_values(by='Total_Volume', ascending=False)
        sorted_volume['Cumulative_Volume'] = sorted_volume['Total_Volume'].cumsum()
        target_volume = total_volume * (value_area_percent / 100.0)
        value_area = sorted_volume[sorted_volume['Cumulative_Volume'] <= target_volume]
    
        if not value_area.empty:
            vah = value_area['Price_Bin'].max().right
            val = value_area['Price_Bin'].min().left
        else:
            vah = max_price
            val = min_price
    
        logging.info(f"Volume Profile 결과: POC={poc}, VAH={vah}, VAL={val}")
        return {'POC': poc, 'VAH': vah, 'VAL': val}
    except Exception as e:
        logging.error(f"Volume Profile 계산 중 오류 발생: {e}")
        return {'POC': 0, 'VAH': 0, 'VAL': 0}

# ==========================
# 6. 매매 신호 결정 클래스
# ==========================
class SignalGenerator:
    def __init__(self, df: pd.DataFrame, volume_profile: dict, higher_trend: str, adx_threshold=20.0):
        self.df = df
        self.volume_profile = volume_profile
        self.higher_trend = higher_trend
        self.adx_threshold = adx_threshold

    def generate_signal(self) -> str:
        if len(self.df) < 2:
            return "HOLD"
        latest = self.df.iloc[-1]
        previous = self.df.iloc[-2]

        # StochRSI 교차
        k_cross_up = (previous["StochRSI_k"] < previous["StochRSI_d"]) and (latest["StochRSI_k"] > latest["StochRSI_d"])
        k_cross_down = (previous["StochRSI_k"] > previous["StochRSI_d"]) and (latest["StochRSI_k"] < latest["StochRSI_d"])

        price = latest["Close"]
        ema200 = latest["EMA200"]
        adx = latest["ADX"]
        macd = latest["MACD"]
        macd_signal = latest["MACD_signal"]

        poc = self.volume_profile.get('POC', 0)
        vah = self.volume_profile.get('VAH', 0)
        val = self.volume_profile.get('VAL', 0)

        bb_upper = latest["BB_upper"]
        bb_lower = latest["BB_lower"]

        signal = "HOLD"

        # 매수 신호
        if (price < bb_lower) and (latest["StochRSI_k"] < 0.2) and k_cross_up and (adx > self.adx_threshold) and (macd < macd_signal) and (price > val):
            if self.higher_trend == "bullish":
                signal = "BUY"
        # 매도 신호
        elif (price > bb_upper) and (latest["StochRSI_k"] > 0.8) and k_cross_down and (adx > self.adx_threshold) and (macd > macd_signal) and (price < vah):
            if self.higher_trend == "bearish":
                signal = "SELL"
        else:
            # EMA200 기반 추가 조건
            if price > ema200 and self.higher_trend == "bullish":
                if (price <= bb_lower) and (latest["StochRSI_k"] < 0.2) and k_cross_up and (adx > self.adx_threshold) and (macd < macd_signal) and (price > val):
                    signal = "BUY"
                elif (val < price < vah) and (latest["StochRSI_k"] < 0.2) and k_cross_up and (adx > self.adx_threshold) and (macd < macd_signal):
                    signal = "BUY"
            elif price < ema200 and self.higher_trend == "bearish":
                if (price >= bb_upper) and (latest["StochRSI_k"] > 0.8) and k_cross_down and (adx > self.adx_threshold) and (macd > macd_signal) and (price < vah):
                    signal = "SELL"
                elif (val < price < vah) and (latest["StochRSI_k"] > 0.8) and k_cross_down and (adx > self.adx_threshold) and (macd > macd_signal):
                    signal = "SELL"

        return signal

# ==========================
# 7. 백테스트 클래스
# ==========================
class Backtester:
    def __init__(self, symbols, exchange, timeframe="5m"):
        self.symbols = symbols
        self.exchange = exchange
        self.timeframe = timeframe
        self.results = {symbol: {'trades': [], 'final_balance': 10000, 'total_pnl': 0, 'win_rate': 0} for symbol in symbols}
        self.ORDER_AMOUNT = 0.1  # 시뮬레이션에 필요한 기본 주문 수량을 추가

    def fetch_historical_data(self, symbol, start_date, end_date, higher_timeframe="4h"):
        since = self.exchange.parse8601(start_date)
        end = self.exchange.parse8601(end_date)
        all_ohlcv = []
        while since < end:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=higher_timeframe, since=since, limit=200)
                if not ohlcv:
                    break
                all_ohlcv += ohlcv
                since = ohlcv[-1][0] + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                logging.warning(f"{symbol} 4시간봉 데이터 fetch 중 에러 발생: {e}")
                break
        df_higher = pd.DataFrame(all_ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        if df_higher.empty:
            logging.warning(f"{symbol} 4시간봉 데이터가 비어 있습니다.")
        else:
            df_higher["Date"] = pd.to_datetime(df_higher["timestamp"], unit="ms")
            df_higher.set_index("Date", inplace=True)
            df_higher.drop("timestamp", axis=1, inplace=True)
            df_higher.name = symbol  # 이름 설정
        return df_higher

    def run_backtest(self, symbol, backtest_start, backtest_end):
        logging.info(f"백테스트 시작: {symbol} ({backtest_start} ~ {backtest_end})")
        df_higher = self.fetch_historical_data(symbol, backtest_start, backtest_end)
        if df_higher.empty:
            logging.warning(f"{symbol} 백테스트를 위한 4시간봉 데이터가 없습니다.")
            return

        indicator_calc_higher = IndicatorCalculator(df_higher, higher_df=df_higher, exchange=self.exchange)
        df_higher, _, _, _, _ = indicator_calc_higher.calculate_indicators()

        if df_higher.empty:
            logging.warning(f"{symbol} 4시간봉 데이터에서 지표 계산 후 데이터가 비어 있습니다.")
            return

        # 5분봉 데이터 수집
        try:
            required_periods = 200  # EMA200의 window 크기
            buffer_periods = 50  # 추가적인 buffer
            limit = required_periods + buffer_periods  # 총 250개 데이터 포인트
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=self.exchange.parse8601(backtest_start), limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("Date", inplace=True)
            df.drop("timestamp", axis=1, inplace=True)
            df.name = symbol
        except Exception as e:
            logging.warning(f"{symbol} 5분봉 데이터 fetch 중 에러 발생: {e}")
            return

        indicator_calc = IndicatorCalculator(df, higher_df=df_higher, exchange=self.exchange)
        df, liquidity_levels, key_levels, pivots, displacement = indicator_calc.calculate_indicators()

        # 데이터 유효성 검사: ATR 값이 존재하는지 확인
        if df.empty or 'ATR' not in df.columns or df['ATR'].isna().all():
            logging.error(f"{symbol}의 5분봉 데이터에 ATR 값이 없습니다. 백테스트를 건너뜁니다.")
            return

        # Volume Profile 계산
        try:
            atr = df['ATR'].iloc[-1]
            volume_profile = calculate_volume_profile(df, atr=atr, bin_size_multiplier=1.0, value_area_percent=70.0)
        except ValueError as ve:
            logging.error(f"{symbol} Volume Profile 계산 중 오류 발생: {ve}")
            return
        except Exception as e:
            logging.error(f"{symbol} Volume Profile 계산 중 예외 발생: {e}")
            return

        # 상위 타임프레임 추세 결정
        latest_higher = df_higher.iloc[-1]
        higher_trend = "bullish" if latest_higher['Close'] > latest_higher['EMA200'] else "bearish"

        # 매매 신호 결정
        signal_gen = SignalGenerator(df, volume_profile, higher_trend)
        signal = signal_gen.generate_signal()

        # 시뮬레이션: 단순화된 주문 실행
        initial_balance = self.results[symbol]['final_balance']
        balance = initial_balance
        position = None
        entry_price = 0

        for idx in range(len(df)):
            current = df.iloc[idx]
            date = df.index[idx]

            # 매매 신호 생성
            try:
                # ATR 값을 해당 시점의 ATR로 설정
                atr = current['ATR']
                # 현재까지의 데이터로 Volume Profile 계산
                if idx + 1 < 200:
                    # EMA200 계산을 위한 충분한 데이터가 없는 경우 Volume Profile을 건너뜁니다.
                    continue
                volume_profile = calculate_volume_profile(df.iloc[:idx+1], atr=atr, bin_size_multiplier=1.0, value_area_percent=70.0)
                signal_gen = SignalGenerator(df.iloc[:idx+1], volume_profile, higher_trend)
                signal = signal_gen.generate_signal()
            except ValueError as ve:
                logging.error(f"{symbol} Volume Profile 계산 중 오류 발생 (매매 신호 생성 단계): {ve}")
                signal = "HOLD"
            except Exception as e:
                logging.error(f"{symbol} Volume Profile 계산 중 예외 발생 (매매 신호 생성 단계): {e}")
                signal = "HOLD"

            # 매수
            if signal == "BUY" and position is None:
                position = 'long'
                entry_price = current['Close']
                balance -= self.ORDER_AMOUNT * entry_price  # 단순화된 매수 비용
                self.results[symbol]['trades'].append({'Type': 'BUY', 'Price': entry_price, 'Time': date})
            # 매도
            elif signal == "SELL" and position == 'long':
                pnl = (current['Close'] - entry_price) * self.ORDER_AMOUNT
                balance += self.ORDER_AMOUNT * current['Close']
                position = None
                self.results[symbol]['trades'].append({'Type': 'SELL', 'Price': current['Close'], 'Time': date, 'P&L': pnl})
                self.results[symbol]['total_pnl'] += pnl
                if pnl > 0:
                    self.results[symbol]['win_rate'] += 1

        # 포지션 정리
        if position == 'long':
            pnl = (df.iloc[-1]['Close'] - entry_price) * self.ORDER_AMOUNT
            balance += self.ORDER_AMOUNT * df.iloc[-1]['Close']
            self.results[symbol]['trades'].append({'Type': 'SELL', 'Price': df.iloc[-1]['Close'], 'Time': df.index[-1], 'P&L': pnl})
            self.results[symbol]['total_pnl'] += pnl
            if pnl > 0:
                self.results[symbol]['win_rate'] += 1

        # 성과 계산
        total_trades = len([t for t in self.results[symbol]['trades'] if 'P&L' in t])
        win_trades = self.results[symbol]['win_rate']
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        self.results[symbol]['final_balance'] = balance
        self.results[symbol]['win_rate'] = win_rate

        logging.info(f"{symbol} 백테스트 종료: Final Balance = {balance}, Total P&L = {self.results[symbol]['total_pnl']}, Win Rate = {win_rate}%")

    def print_backtest_results(self):
        for symbol, data in self.results.items():
            logging.info(f"--- {symbol} 백테스트 결과 ---")
            logging.info(f"Final Balance: {data['final_balance']}")
            logging.info(f"Total P&L: {data['total_pnl']}")
            logging.info(f"Win Rate: {data['win_rate']}%")
            logging.info(f"Total Trades: {len([t for t in data['trades'] if 'P&L' in t])}")
            logging.info("Trades:")
            for trade in data['trades']:
                logging.info(trade)

# ==========================
# 8. 리스크 관리 클래스
# ==========================
class RiskManager:
    def __init__(self, max_drawdown=20.0, max_risk_per_trade=2.0):
        """
        :param max_drawdown: 전체 계좌에서 허용되는 최대 손실 비율 (%)
        :param max_risk_per_trade: 각 거래에서 허용되는 최대 손실 비율 (%)
        """
        self.max_drawdown = max_drawdown
        self.max_risk_per_trade = max_risk_per_trade
        self.initial_balance = None
        self.current_balance = None
        self.max_balance = None

    def update_balance(self, balance):
        if self.initial_balance is None:
            self.initial_balance = balance
            self.max_balance = balance
        self.current_balance = balance
        if balance > self.max_balance:
            self.max_balance = balance

    def check_drawdown(self):
        if self.initial_balance is None or self.current_balance is None:
            return False
        drawdown = ((self.max_balance - self.current_balance) / self.max_balance) * 100
        logging.info(f"현재 계좌 잔고: {self.current_balance}, 최대 잔고: {self.max_balance}, Drawdown: {drawdown}%")
        return drawdown >= self.max_drawdown

    def calculate_position_size(self, account_balance, entry_price, atr, leverage):
        risk_amount = account_balance * (self.max_risk_per_trade / 100)
        position_size = risk_amount / (atr * leverage)
        return position_size

# ==========================
# 9. 전략 모니터링 클래스
# ==========================
class StrategyMonitor:
    def __init__(self):
        self.metrics = {}

    def update_metrics(self, symbol, pnl, win):
        if symbol not in self.metrics:
            self.metrics[symbol] = {'total_pnl': 0, 'wins': 0, 'losses': 0}
        self.metrics[symbol]['total_pnl'] += pnl
        if win:
            self.metrics[symbol]['wins'] += 1
        else:
            self.metrics[symbol]['losses'] += 1

    def log_metrics(self):
        for symbol, data in self.metrics.items():
            total_trades = data['wins'] + data['losses']
            win_rate = (data['wins'] / total_trades) * 100 if total_trades > 0 else 0
            logging.info(f"--- {symbol} 전략 성과 ---")
            logging.info(f"Total P&L: {data['total_pnl']}")
            logging.info(f"Win Rate: {win_rate}% ({data['wins']} wins, {data['losses']} losses)")

# ==========================
# 10. 트레이딩 봇 클래스
# ==========================
class TradingBot:
    def __init__(self, symbols, timeframe="5m"):
        self.symbols = symbols
        self.timeframe = timeframe
        self.exchange = self.initialize_exchange()
        self.LEVERAGE = 10
        self.ORDER_AMOUNT = 0.1  # 기본 주문 수량
        self.PROFIT_THRESHOLD = 3.5  # 익절 기준 (%)
        self.LOSS_THRESHOLD = -1.5   # 손절 기준 (%)
        self.ATR_MULTIPLIER = 1.5
        self.SLIPPAGE_RATE = 0.001  # 0.1%
        self.TRADING_FEE_RATE = 0.001  # 0.1%
        self.symbol_settings = self.initialize_symbol_settings()
        self.risk_manager = RiskManager()
        self.strategy_monitor = StrategyMonitor()

    def initialize_exchange(self):
        try:
            exchange = ccxt.okx({
                "apiKey": OKX_API_KEY,
                "secret": OKX_API_SECRET,
                "password": OKX_API_PASSWORD,
                "enableRateLimit": True,
            })
            logging.info("OKX 인증 성공")
            return exchange
        except Exception as e:
            logging.error(f"OKX 인증 실패: {e}")
            exit()

    def initialize_symbol_settings(self):
        settings = {}
        for symbol in self.symbols:
            settings[symbol] = {
                'entries': [],
                'total_trades': 0,
                'winning_trades': 0,
                'MIN_AMOUNT': self.get_min_order_amount(symbol)
            }
            # 레버리지 설정 (롱/숏)
            try:
                self.exchange.set_leverage(self.LEVERAGE, symbol, params={"mgnMode": "isolated", "posSide": "long"})
                logging.info(f"{symbol} 롱 포지션 레버리지 {self.LEVERAGE}배, isolated 설정 완료")
            except Exception as e:
                logging.warning(f"{symbol} 롱 포지션 레버리지 설정 실패: {e}")

            try:
                self.exchange.set_leverage(self.LEVERAGE, symbol, params={"mgnMode": "isolated", "posSide": "short"})
                logging.info(f"{symbol} 숏 포지션 레버리지 {self.LEVERAGE}배, isolated 설정 완료")
            except Exception as e:
                logging.warning(f"{symbol} 숏 포지션 레버리지 설정 실패: {e}")
        return settings

    def get_min_order_amount(self, symbol):
        try:
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            logging.info(f"{symbol}의 최소 주문량: {min_amount} ETH/SOL")
            return min_amount
        except Exception as e:
            logging.warning(f"{symbol}의 최소 주문 금액을 가져오는 데 실패했습니다: {e}")
            return 0.1  # 기본값

    def apply_slippage(self, order_price: float, side: str) -> float:
        if side == "buy":
            return order_price * (1 + self.SLIPPAGE_RATE)  # 매수 시 가격 상승
        elif side == "sell":
            return order_price * (1 - self.SLIPPAGE_RATE)  # 매도 시 가격 하락
        else:
            return order_price

    def calculate_fee(self, amount: float, price: float) -> float:
        return amount * price * self.TRADING_FEE_RATE

    def fetch_data(self, symbol, higher_timeframe="4h"):
        try:
            # 4시간봉 데이터 수집
            higher_ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=higher_timeframe, limit=200)
            df_higher = pd.DataFrame(higher_ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
            df_higher["Date"] = pd.to_datetime(df_higher["timestamp"], unit="ms")
            df_higher.set_index("Date", inplace=True)
            df_higher.drop("timestamp", axis=1, inplace=True)
            df_higher.name = symbol  # 이름 설정

            # 지표 계산
            indicator_calc_higher = IndicatorCalculator(df_higher, higher_df=df_higher, exchange=self.exchange)
            df_higher, _, _, _, _ = indicator_calc_higher.calculate_indicators()

            if df_higher.empty:
                logging.warning(f"{symbol} 4시간봉 데이터에서 지표 계산 후 데이터가 비어 있습니다.")
                higher_trend = "neutral"
            else:
                # 상위 타임프레임 추세 결정
                latest_higher = df_higher.iloc[-1]
                higher_trend = "bullish" if latest_higher['Close'] > latest_higher['EMA200'] else "bearish"
        except Exception as e:
            logging.warning(f"{symbol} 4시간봉 데이터 fetch/계산 실패: {e}")
            higher_trend = "neutral"

        try:
            # 5분봉 데이터 수집
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=300)  # limit 증가
            df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("Date", inplace=True)
            df.drop("timestamp", axis=1, inplace=True)
            df.name = symbol

            # 지표 계산
            indicator_calc = IndicatorCalculator(df, higher_df=df_higher, exchange=self.exchange)
            df, liquidity_levels, key_levels, pivots, displacement = indicator_calc.calculate_indicators()

            # 데이터 유효성 검사: ATR 값이 존재하는지 확인
            if df.empty or 'ATR' not in df.columns or df['ATR'].isna().all():
                logging.error(f"{symbol}의 5분봉 데이터에 ATR 값이 없습니다. 데이터를 건너뜁니다.")
                return None, None, "neutral"

            return df, {'liquidity_levels': liquidity_levels, 'key_levels': key_levels, 'pivots': pivots, 'displacement': displacement}, higher_trend
        except Exception as e:
            logging.warning(f"{symbol} 5분봉 데이터 fetch/계산 실패: {e}")
            return None, None, "neutral"

    def generate_signal(self, df, extra_info, higher_trend) -> str:
        if df is None or extra_info is None:
            return "HOLD"

        # Volume Profile 기반 신호 생성 예시
        volume_profile = {'POC': df['POC'].iloc[-1]}
        signal_gen = SignalGenerator(df, volume_profile, higher_trend)
        return signal_gen.generate_signal()

    def execute_trade(self, symbol, signal, last_price, atr):
        settings = self.symbol_settings[symbol]
        try:
            balance = self.exchange.fetch_balance({'type': 'future'})
        except Exception as e:
            logging.warning(f"{symbol} 잔고 정보 가져오기 실패: {e}")
            return

        usdt_free = balance['free'].get('USDT', 0)
        required_margin_per_order = (self.ORDER_AMOUNT * last_price) / self.LEVERAGE

        # 리스크 관리: 포트폴리오 전체 최대 손실 한도 확인
        self.risk_manager.update_balance(balance['total']['USDT'])
        if self.risk_manager.check_drawdown():
            logging.warning(f"전체 계좌의 손실이 최대 손실 한도({self.risk_manager.max_drawdown}%)를 초과했습니다. 거래를 중단합니다.")
            send_telegram_message("⚠️ 전체 계좌의 손실이 최대 손실 한도를 초과했습니다. 트레이딩을 중단합니다.")
            exit()

        # 승률 계산
        if settings['total_trades'] > 0:
            win_rate = (settings['winning_trades'] / settings['total_trades']) * 100
        else:
            win_rate = 0.0

        # 리스크 관리: 각 거래별 최대 손실 한도에 따라 포지션 크기 조정
        position_size = self.risk_manager.calculate_position_size(self.risk_manager.current_balance, last_price, atr, self.LEVERAGE)
        position_size = max(position_size, settings['MIN_AMOUNT'])  # 최소 주문 단위 준수

        # 최종 신호 매매 실행
        if signal == "BUY":
            while usdt_free >= required_margin_per_order and position_size >= settings['MIN_AMOUNT']:
                try:
                    executed_price = self.apply_slippage(last_price, 'buy')
                    fee = self.calculate_fee(position_size, executed_price)
                    order = self.exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side="buy",
                        amount=position_size,
                        params={"tdMode": "isolated", "posSide": "long"}
                    )
                    logging.info(f"{symbol} 롱 포지션 진입: {position_size} ETH at 가격 {last_price:.2f}")
                    send_telegram_message(
                        f"📈 *{symbol} LONG 진입*\n수량: {position_size} ETH\n가격: {executed_price:.2f}\n✅ *승률*: {win_rate:.2f}%\n💸 *수수료*: {fee:.2f} USDT"
                    )

                    executed_price = float(last_price)
                    settings['entries'].append({'side': 'long', 'amount': position_size, 'entry_price': executed_price})
                    usdt_free -= required_margin_per_order
                except Exception as e:
                    logging.warning(f"{symbol} 롱 진입 실패: {e}")
                    break
        elif signal == "SELL":
            while usdt_free >= required_margin_per_order and position_size >= settings['MIN_AMOUNT']:
                try:
                    executed_price = self.apply_slippage(last_price, 'sell')
                    fee = self.calculate_fee(position_size, executed_price)
                    order = self.exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side="sell",
                        amount=position_size,
                        params={"tdMode": "isolated", "posSide": "short"}
                    )
                    logging.info(f"{symbol} 숏 포지션 진입: {position_size} ETH at 가격 {last_price:.2f}")
                    send_telegram_message(
                        f"📉 *{symbol} SHORT 진입*\n수량: {position_size} ETH\n가격: {executed_price:.2f}\n✅ *승률*: {win_rate:.2f}%\n💸 *수수료*: {fee:.2f} USDT"
                    )

                    executed_price = float(last_price)
                    settings['entries'].append({'side': 'short', 'amount': position_size, 'entry_price': executed_price})
                    usdt_free -= required_margin_per_order
                except Exception as e:
                    logging.warning(f"{symbol} 숏 진입 실패: {e}")
                    break
        else:
            logging.info(f"{symbol} HOLD: 추가 주문 없음")

    def manage_positions(self, symbol, current_price, atr):
        settings = self.symbol_settings[symbol]
        for entry in settings['entries'].copy():
            entry_price = entry['entry_price']
            side = entry['side']
            if side == 'long':
                pnl = ((current_price - entry_price) / entry_price) * 100 * self.LEVERAGE
                stop_loss_price = entry_price - (atr * self.ATR_MULTIPLIER)
                take_profit_price = entry_price + (atr * (self.PROFIT_THRESHOLD / self.ATR_MULTIPLIER))
                if pnl >= self.PROFIT_THRESHOLD or pnl <= self.LOSS_THRESHOLD or current_price <= stop_loss_price:
                    try:
                        executed_price = self.apply_slippage(current_price, 'sell')
                        fee = self.calculate_fee(entry['amount'], executed_price)
                        order = self.exchange.create_order(
                            symbol=symbol,
                            type="market",
                            side="sell",
                            amount=entry['amount'],
                            params={"tdMode": "isolated", "posSide": "long"}
                        )
                        logging.info(f"{symbol} 롱 청산: {entry['amount']} ETH, P&L={pnl:.2f}%")
                        settings['total_trades'] += 1
                        if pnl > 0:
                            settings['winning_trades'] += 1
                            self.strategy_monitor.update_metrics(symbol, pnl, True)
                        else:
                            self.strategy_monitor.update_metrics(symbol, pnl, False)
                        win_rate = (settings['winning_trades'] / settings['total_trades']) * 100
                        send_telegram_message(
                            f"🔴 *{symbol} LONG 청산*\n가격: {executed_price:.2f}\nP&L: {pnl:.2f}%\n✅ *승률*: {win_rate:.2f}%\n💸 *수수료*: {fee:.2f} USDT"
                        )
                        settings['entries'].remove(entry)
                    except Exception as e:
                        logging.warning(f"{symbol} 롱 청산 실패: {e}")
            else:  # short
                pnl = ((entry_price - current_price) / entry_price) * 100 * self.LEVERAGE
                stop_loss_price = entry_price + (atr * self.ATR_MULTIPLIER)
                take_profit_price = entry_price - (atr * (self.PROFIT_THRESHOLD / self.ATR_MULTIPLIER))
                if pnl >= self.PROFIT_THRESHOLD or pnl <= self.LOSS_THRESHOLD or current_price >= stop_loss_price:
                    try:
                        executed_price = self.apply_slippage(current_price, 'buy')
                        fee = self.calculate_fee(entry['amount'], executed_price)
                        order = self.exchange.create_order(
                            symbol=symbol,
                            type="market",
                            side="buy",
                            amount=entry['amount'],
                            params={"tdMode": "isolated", "posSide": "short"}
                        )
                        logging.info(f"{symbol} 숏 청산: {entry['amount']} ETH, P&L={pnl:.2f}%")
                        settings['total_trades'] += 1
                        if pnl > 0:
                            settings['winning_trades'] += 1
                            self.strategy_monitor.update_metrics(symbol, pnl, True)
                        else:
                            self.strategy_monitor.update_metrics(symbol, pnl, False)
                        win_rate = (settings['winning_trades'] / settings['total_trades']) * 100
                        send_telegram_message(
                            f"🔴 *{symbol} SHORT 청산*\n가격: {executed_price:.2f}\nP&L: {pnl:.2f}%\n✅ *승률*: {win_rate:.2f}%\n💸 *수수료*: {fee:.2f} USDT"
                        )
                        settings['entries'].remove(entry)
                    except Exception as e:
                        logging.warning(f"{symbol} 숏 청산 실패: {e}")

    def run(self):
        logging.info("=== 실거래 시작 ===")
        while True:
            for symbol in self.symbols:
                try:
                    # 1. 데이터 수집 및 지표 계산
                    df, extra_info, higher_trend = self.fetch_data(symbol)

                    if df is None or extra_info is None:
                        continue

                    # 2. 매매 신호 생성
                    signal = self.generate_signal(df, extra_info, higher_trend)
                    last = df.iloc[-1]

                    logging.info(
                        f"[TRADE] {symbol} 시그널: {signal}, price={last['Close']:.2f}, "
                        f"ADX={last['ADX']:.2f}, MACD={last['MACD']:.4f}, "
                        f"EMA200={last['EMA200']:.2f}, BB_U={last['BB_upper']:.2f}, BB_L={last['BB_lower']:.2f}, "
                        f"VAH={extra_info['volume_profile']['VAH']:.2f}, VAL={extra_info['volume_profile']['VAL']:.2f}, "
                        f"Higher Trend={higher_trend}"
                    )

                    # 3. 매매 신호에 따른 주문 실행
                    atr = last['ATR']
                    self.execute_trade(symbol, signal, last['Close'], atr)

                    # 4. 엔트리별 부분청산/스탑로스 관리
                    self.manage_positions(symbol, last['Close'], atr)

                except Exception as e:
                    logging.warning(f"{symbol} 실거래 루프 중 에러 발생: {e}")

            # 5. 주기적 대기 (5분봉 기준)
            time.sleep(300)  # 5분 대기

    # ==========================
    # 11. 백테스트 및 포워드 테스트 함수
    # ==========================
    def perform_backtest_and_forward_test(self, backtest_start, backtest_end, forward_start, forward_end):
        backtester = Backtester(self.symbols, self.exchange, timeframe=self.timeframe)
        for symbol in self.symbols:
            backtester.run_backtest(symbol, backtest_start, backtest_end)
        backtester.print_backtest_results()

        # 포워드 테스트
        logging.info("포워드 테스트 시작")
        for symbol in self.symbols:
            logging.info(f"{symbol} 포워드 테스트는 실시간 데이터에서 자동으로 수행됩니다.")
        # 실제로 포워드 테스트는 별도의 클래스로 구현하거나, 실시간 데이터에서 수행됩니다.
        # 여기서는 단순히 백테스트 후 실시간 거래를 진행하는 형태로 통합.

# ==========================
# 12. 메인 실행
# ==========================
if __name__ == "__main__":
    symbols = ["ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"]  # 거래할 자산 리스트
    timeframe = "5m"

    # 트레이딩 봇 인스턴스 생성
    bot = TradingBot(symbols, timeframe)

    # 백테스트 수행 (예시 기간)
    backtest_start = '2021-01-01T00:00:00Z'
    backtest_end = '2023-12-31T23:59:59Z'
    forward_start = '2024-01-01T00:00:00Z'
    forward_end = '2025-01-08T00:00:00Z'

    # 백테스트 실행
    bot.perform_backtest_and_forward_test(backtest_start, backtest_end, forward_start, forward_end)

    # 실거래 실행
    bot.run()

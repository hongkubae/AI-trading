# indicators.py

import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_mom(series, period=10):
    return series - series.shift(period)

def calculate_roc(series, period=10):
    shifted = series.shift(period)
    return (series / shifted - 1)*100

def calculate_ema(series, period=12):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(df_close, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(df_close, fast)
    ema_slow = calculate_ema(df_close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_atr(df, period=14):
    df["tr"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["close"].shift(1)).abs()
    df["tr3"] = (df["low"] - df["close"].shift(1)).abs()
    df["true_range"] = df[["tr","tr2","tr3"]].max(axis=1)
    atr = df["true_range"].rolling(period).mean()
    df.drop(["tr","tr2","tr3","true_range"], axis=1, inplace=True)
    return atr

def calculate_bollinger(df_close, period=20, n_std=2):
    ma = df_close.rolling(period).mean()
    std = df_close.rolling(period).std()
    upper = ma + n_std*std
    lower = ma - n_std*std
    return ma, upper, lower

def add_indicators(df):
    """
    다양한 지표(RSI, MOM, ROC, MACD, ATR, Bollinger) 추가
    """
    df["RSI14"] = calculate_rsi(df["close"], 14)
    df["MOM10"] = calculate_mom(df["close"], 10)
    df["ROC10"] = calculate_roc(df["close"], 10)

    macd_line, macd_signal, macd_hist = calculate_macd(df["close"], 12, 26, 9)
    df["MACD_line"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    df["ATR14"] = calculate_atr(df, 14)
    ma20, bb_up, bb_low = calculate_bollinger(df["close"], 20, 2)
    df["BB_mid"] = ma20
    df["BB_up"] = bb_up
    df["BB_low"] = bb_low

    df.dropna(inplace=True)
    return df

# model_manager.py

import os
import joblib
import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ParameterGrid

import torch
import torch.nn as nn
import torch.optim as optim

from indicators import add_indicators

######################################
# (A) PyTorch LSTM Model
######################################
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def create_lstm_dataset(df, feature_cols, target_col="signal", seq_len=10):
    X_list, y_list = [], []
    for i in range(len(df)-seq_len):
        X_seq = df[feature_cols].iloc[i:i+seq_len].values
        y_val = df[target_col].iloc[i+seq_len]
        X_list.append(X_seq)
        y_list.append(y_val)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def evaluate_lstm_accuracy(model, X_val_t, y_val_t, threshold=0.5):
    model.eval()
    with torch.no_grad():
        out = model(X_val_t)
        preds = (out.squeeze() >= threshold).long()
        targets = y_val_t.long()
        correct = (preds == targets).sum().item()
        total = len(targets)
        return correct/total

######################################
# (B) load_model 함수
######################################
def load_model(model_type="RF", model_path="best_model.pkl"):
    """
    RF/MLP -> joblib.load(.pkl)
    LSTM   -> torch.load_state_dict(.pt)
    """
    import os

    if model_type in ["RF","MLP"]:
        if os.path.isfile(model_path):
            return joblib.load(model_path)
        else:
            print(f"[load_model] .pkl file not found: {model_path}")
            return None
    elif model_type=="LSTM":
        if os.path.isfile(model_path):
            # 이 예시에서는 input_dim=14(피처 개수와 일치해야)
            model = LSTMModel(input_dim=14, hidden_dim=32, num_layers=1)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        else:
            print(f"[load_model] .pt file not found: {model_path}")
            return None
    else:
        print("[load_model] Unknown model_type:", model_type)
        return None

######################################
# (C) Time Series Validation
######################################
from sklearn.model_selection import TimeSeriesSplit

def time_series_validation(df, features, label_col="signal",
                           model_type="RF", splits=10, method="walk_forward"):
    df = df.copy()
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    X_full = df[features]
    y_full = df[label_col]

    results = []
    if model_type not in ["RF","MLP"]:
        print("[time_series_validation] (demo) only RF/MLP implemented.")
        return 0.0

    if method=="walk_forward":
        n = len(df)
        base_ratio=0.5
        train_end = int(n*base_ratio)
        split_size = (n - train_end)//splits
        for i in range(splits):
            train_df = df.iloc[:train_end]
            val_df   = df.iloc[train_end:train_end+split_size]
            if val_df.empty:
                break
            X_train = train_df[features]
            y_train = train_df[label_col]
            X_val   = val_df[features]
            y_val   = val_df[label_col]

            if model_type=="RF":
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            else:
                model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)

            model.fit(X_train, y_train)
            acc_val = model.score(X_val, y_val)
            results.append(acc_val)
            train_end+=split_size

        avg_acc = np.mean(results) if results else 0
        print(f"[time_series_validation] walk_forward splits={splits}, avg={avg_acc:.4f}")
        return avg_acc
    else:
        tscv = TimeSeriesSplit(n_splits=splits)
        idx=1
        for train_idx, test_idx in tscv.split(X_full):
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

            if model_type=="RF":
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            else:
                model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)

            model.fit(X_train,y_train)
            acc_val = model.score(X_test,y_test)
            results.append(acc_val)
            idx+=1

        avg_acc = np.mean(results) if results else 0
        print(f"[time_series_validation] TimeSeriesSplit splits={splits}, avg={avg_acc:.4f}")
        return avg_acc

######################################
# (D) Hyperparam random search (demo)
######################################
from sklearn.model_selection import ParameterGrid
def random_search_hyperparam(X_train, y_train, X_val, y_val,
                             model_type="RF", n_iter=5):
    best_acc=-1
    best_params=None
    param_grid={}
    if model_type=="RF":
        param_grid={
            "n_estimators":[50,100,150],
            "max_depth":[5,10,15,20]
        }
    else: # MLP
        param_grid={
            "hidden_layer_sizes":[(64,32),(128,64),(64,64,32)],
            "max_iter":[200,300]
        }
    all_params = list(ParameterGrid(param_grid))
    random.shuffle(all_params)
    all_params = all_params[:n_iter]

    for params in all_params:
        if model_type=="RF":
            model = RandomForestClassifier(**params, random_state=42)
        else:
            model = MLPClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        acc_val = model.score(X_val, y_val)
        if acc_val>best_acc:
            best_acc=acc_val
            best_params=params
    return best_params, best_acc

######################################
# (E) train_model(...)
######################################
def train_model(csv_path="data/historical_data.csv",
                model_path="best_model.pkl",
                model_type="RF",
                use_timeseries_cv=True):
    """
    개선사항:
    1) 데이터 로드 + 지표 + 복잡 라벨링
    2) 클래스 분포 체크
    3) 워크포워드 splits=10
    4) Hyperparam random search
    """
    if not os.path.isfile(csv_path):
        print("[train_model] CSV not found:", csv_path)
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[train_model] CSV empty.")
        return None

    # 인디케이터
    df = add_indicators(df)
    # MA
    df["ma_short"] = df["close"].rolling(10).mean()
    df["ma_long"]  = df["close"].rolling(60).mean()
    # 라벨링: 미래5봉 수익률 & ma_short>ma_long
    OFFSET=5
    df["future_return"] = df["close"].shift(-OFFSET)/df["close"] -1
    df["cond_return"]   = (df["future_return"]>0).astype(int)
    df.dropna(inplace=True)
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ex) signal = (cond_return & (ma_short>ma_long))
    cond1 = (df["cond_return"]==1)
    cond2 = (df["ma_short"]>df["ma_long"])
    df["signal"] = (cond1 & cond2).astype(int)

    if len(df)<200:
        print("[train_model] Not enough data (<200).")
        return None

    print("[train_model] signal dist:", df["signal"].value_counts().to_dict())

    # 시계열 검증
    features = [
        "close","volume","RSI14","MOM10","ROC10","MACD_line","MACD_signal","MACD_hist",
        "ATR14","BB_mid","BB_up","BB_low",
        "ma_short","ma_long"
    ]

    if use_timeseries_cv and model_type in ["RF","MLP"]:
        cv_acc = time_series_validation(
            df, features, label_col="signal",
            model_type=model_type,
            splits=10,  # 10
            method="walk_forward"
        )
        print(f"[train_model] => timeseries_cv_acc={cv_acc:.4f}")

    # 최종 학습(80:20)
    X = df[features]
    y = df["signal"]
    split_idx = int(len(df)*0.8)
    train_df = df.iloc[:split_idx]
    val_df   = df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df["signal"]
    X_val   = val_df[features]
    y_val   = val_df["signal"]

    if model_type in ["RF","MLP"]:
        # RandomSearch
        best_params, best_acc = random_search_hyperparam(X_train,y_train,X_val,y_val,
                                                         model_type=model_type,
                                                         n_iter=5)
        print(f"[train_model:{model_type}] best_params={best_params}, best_acc={best_acc:.4f}")

        if model_type=="RF":
            model = RandomForestClassifier(**best_params, random_state=42)
        else:
            model = MLPClassifier(**best_params, random_state=42)
        model.fit(X_train,y_train)
        final_acc = model.score(X_val,y_val)
        print(f"[train_model:{model_type}] final ValAcc={final_acc:.4f}")
        joblib.dump(model, model_path)
        print(f"[train_model:{model_type}] Model saved => {model_path}")
        return final_acc

    elif model_type=="LSTM":
        # 생략 or 기존 LSTM 로직
        pass
    else:
        print("[train_model] unknown model_type:", model_type)
        return None

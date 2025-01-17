# backtest_utils.py

import pandas as pd
from indicators import add_indicators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def walk_forward_validation(df, train_ratio=0.8):
    df = add_indicators(df.copy())
    df["ma_short"] = df["close"].rolling(10).mean()
    df["ma_long"] = df["close"].rolling(60).mean()
    df.dropna(inplace=True)
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)

    features = ["close","volume","RSI14","MOM10","ROC10","ma_short","ma_long"]
    df.dropna(inplace=True)

    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df["signal"]
    X_val = val_df[features]
    y_val = val_df["signal"]

    model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    print(f"[walk_forward_validation] Accuracy={acc:.4f}")
    return model

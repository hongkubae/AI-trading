# telegram_utils.py

import requests
import matplotlib.pyplot as plt
import io

TELEGRAM_BOT_TOKEN = ""
CHAT_ID = ""

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        print("[send_telegram_message] Token or chat_id missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    requests.post(url, data=data)

def send_chart_telegram(df, title="Chart"):
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        print("[send_chart_telegram] Token or chat_id missing.")
        return
    plt.figure(figsize=(10,5))
    if "ts" in df.columns and "close" in df.columns:
        plt.plot(df["ts"], df["close"], label="Close Price")
    plt.title(title)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": buf}
    data = {"chat_id": CHAT_ID, "caption": title}
    requests.post(url, files=files, data=data)

    buf.close()
    plt.close()

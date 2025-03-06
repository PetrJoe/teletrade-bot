import asyncio
import json
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from aiogram import Bot
import websockets
import joblib
from dotenv import load_dotenv


load_dotenv()

DERIV_API_TOKEN = os.getenv('DERIV_API_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# 🔹 Initialize Telegram Bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# 🔹 Paths for storing model and data
MODEL_PATH = "linear_model.pkl"
DATA_PATH = "candles_data.csv"

# 🔹 Symbol Mapping
symbol_names = {
    "R_75": "Volatility 75 Index",
    "R_50": "Volatility 50 Index",
    "R_10": "Volatility 10 Index",
    "R_25": "Volatility 25 Index",
}

# 🔹 Function to Fetch Market Data via WebSockets
async def get_candles(symbol, timeframe, count=100):
    granularity = {"M5": 300, "M15": 900, "H1": 3600}
    async with websockets.connect("wss://ws.deriv.com/websockets/v3") as ws:
        await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
        await ws.recv()

        request = json.dumps({
            "ticks_history": symbol,
            "count": count,
            "granularity": granularity[timeframe],
            "end": "latest"
        })
        await ws.send(request)
        response = await ws.recv()
        data = json.loads(response)

        return data.get("candles", [])

# 🔹 Load or Build Linear Regression Model
def load_or_build_linear_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    model = LinearRegression()
    return model

model = load_or_build_linear_model()
scaler = MinMaxScaler()

# 🔹 Train & Save Linear Regression Model
def train_linear_model(candles):
    df = pd.DataFrame(candles)
    df["close"] = df["close"].astype(float)
    df.to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

    data = df["close"].values.reshape(-1, 1)
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(20, len(data)):
        X.append(data[i-20:i].flatten())
        y.append(data[i])

    X, y = np.array(X), np.array(y)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

# 🔹 AI-Based Forecasting Signal
def ai_forecast(candles):
    df = pd.DataFrame(candles)
    df["close"] = df["close"].astype(float)

    last_20 = df["close"].values[-20:].reshape(-1, 1)
    last_20 = scaler.transform(last_20).flatten().reshape(1, -1)

    predicted_price = model.predict(last_20)[0]
    predicted_price = predicted_price * (np.max(df["close"]) - np.min(df["close"])) + np.min(df["close"])

    entry_price = df["close"].iloc[-1]
    if predicted_price > entry_price:
        return "BUY", entry_price, entry_price * 1.02, entry_price * 0.99
    else:
        return "SELL", entry_price, entry_price * 0.98, entry_price * 1.01

# 🔹 Weekly Retraining Scheduler
async def weekly_retrain():
    while True:
        await asyncio.sleep(604800)  # 1 week
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            train_linear_model(df.to_dict(orient="records"))
            print(f"Model retrained on {datetime.datetime.now()}")

# 🔹 Function to Send Trading Signals to Telegram
async def send_signal(symbol, signal, entry, tp, sl):
    message = (
        f"📢 **{symbol_names[symbol]} Trading Signal**\n"
        f"🔹 **Direction:** {signal}\n"
        f"🔹 **Entry Price:** {entry:.2f}\n"
        f"🔹 **Take Profit:** {tp:.2f}\n"
        f"🔹 **Stop Loss:** {sl:.2f}\n"
    )
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")

# 🔹 Main Trading Loop
async def main():
    asyncio.create_task(weekly_retrain())  # Start weekly retraining
    while True:
        for symbol in ["R_75", "R_50", "R_10", "R_25"]:
            candles_m5 = await get_candles(symbol, "M5")
            candles_m15 = await get_candles(symbol, "M15")
            candles_h1 = await get_candles(symbol, "H1")

            if not candles_m5 or not candles_m15 or not candles_h1:
                continue

            signal_m5, entry_m5, tp_m5, sl_m5 = ai_forecast(candles_m5)
            signal_m15, entry_m15, tp_m15, sl_m15 = ai_forecast(candles_m15)
            signal_h1, entry_h1, tp_h1, sl_h1 = ai_forecast(candles_h1)

            if signal_m5 == signal_m15 == signal_h1:
                await send_signal(symbol, signal_m5, entry_m5, tp_m5, sl_m5)

        await asyncio.sleep(60)

# 🔹 Start the Bot
asyncio.run(main())

import asyncio  # type: ignore
import json
import logging
import os
from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd
import websockets
from aiogram import Bot, Dispatcher, types
from httpx import AsyncClient
from prophet import Prophet  # type: ignore
from pydantic import BaseModel

from teletrade_bot.config import settings


class FixedLengthShiftList(BaseModel):
    data: list[int] = []
    max_size: int

    def append(self, item):
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append(item)


volatility_data = {
    index: FixedLengthShiftList(max_size=settings.MAX_DATA_POINTS)
    for index in settings.VIX_INDICES
}
trade_signals = {
    index: FixedLengthShiftList(max_size=settings.MAX_DATA_POINTS)
    for index in settings.VIX_INDICES
}
entry_prices = {index: None for index in settings.VIX_INDICES}

bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot=bot)  # Updated for aiogram v3


# Fetch live VIX data
async def get_live_vix(index: str) -> tuple[float | None, datetime | None]:
    async with websockets.connect(settings.DERIV_API_URL) as ws:
        await ws.send(json.dumps({"ticks": index}))
        while True:
            try:
                response = await ws.recv()
                data = json.loads(response)
                if "tick" in data:
                    return float(data["tick"]["quote"]), datetime.now()
            except Exception as e:
                logging.error(f"Error fetching VIX data for {index}: {e}")
                await asyncio.sleep(5)
                return None, None


# Technical indicators
def calculate_rsi(values: FixedLengthShiftList) -> float:
    values = values.data
    if len(values) < settings.RSI_PERIOD:
        return 50
    deltas = np.diff(values)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)
    avg_gain = np.mean(gains[-settings.RSI_PERIOD :])
    avg_loss = np.mean(losses[-settings.RSI_PERIOD :])
    return (
        100
        if avg_loss == 0
        else (0 if avg_gain == 0 else round(100 - (100 / (1 + avg_gain / avg_loss)), 2))
    )


def calculate_sma(values: FixedLengthShiftList) -> float:
    values = values.data
    return (
        np.mean(values[-settings.SMA_PERIOD :])
        if len(values) >= settings.SMA_PERIOD
        else np.mean(values)
    )


def calculate_ema(values: FixedLengthShiftList) -> float:
    values = values.data
    if len(values) < settings.EMA_PERIOD:
        return np.mean(values)
    ema = values[0]
    multiplier = 2 / (settings.EMA_PERIOD + 1)
    for price in values[1:]:
        ema = (price - ema) * multiplier + ema
    return round(ema, 2)


# AI forecasting with data persistence
def ai_forecast(index: str):
    data_file = f"{index}_data.csv"
    try:
        df = pd.read_csv(data_file)
        df["ds"] = pd.to_datetime(df["ds"])
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=3, freq="5min")
        return model.predict(future).iloc[-3:][["ds", "yhat"]]
    except Exception as e:
        logging.error(f"AI Forecasting error for {index}: {e}")
        return None


# Sentiment analysis
async def fetch_sentiment() -> float:
    try:
        async with AsyncClient() as client:
            response = await client.get(settings.SENTIMENT_API_URL)
            return cast(float, response.json().get("sentiment_score", 0))
    except Exception as e:
        logging.error(f"Error fetching sentiment data: {e}")
        return 0


# Save data point
def save_data_point(index: str, timestamp: datetime, value) -> None:
    data_file = f"{index}_data.csv"
    df = pd.DataFrame({"ds": [timestamp], "y": [value]})
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        df = pd.concat([existing_df, df])
        # Keep only last 7 days
        cutoff = datetime.now() - timedelta(days=settings.DATA_RETENTION_DAYS)
        df = df[pd.to_datetime(df["ds"]) > cutoff]
    df.to_csv(data_file, index=False)


# Trading logic
async def trading_loop():
    while True:
        sentiment_score = await fetch_sentiment()

        for index in settings.VIX_INDICES:
            try:
                current_vix, timestamp = await get_live_vix(index)
                if current_vix is None:
                    continue

                volatility_data[index].append(current_vix)
                save_data_point(index, timestamp, current_vix)

                rsi = calculate_rsi(volatility_data[index])
                sma = calculate_sma(volatility_data[index])
                ema = calculate_ema(volatility_data[index])

                ai_predictions = ai_forecast(index)
                ai_trend = (
                    "Unknown"
                    if ai_predictions is None
                    else "Up"
                    if ai_predictions["yhat"].iloc[-1] > current_vix
                    else "Down"
                )

                signal = "HOLD 🤝"
                if sentiment_score < -50:
                    signal = "AVOID TRADING ⚠️ (Negative sentiment detected)"
                elif ema > sma and rsi < 30 and ai_trend == "Up":
                    signal = "BUY 📈 (AI Confirms Uptrend)"
                    settings.entry_prices[index] = current_vix
                elif ema < sma and rsi > 70 and ai_trend == "Down":
                    signal = "SELL 📉 (AI Confirms Downtrend)"
                    settings.entry_prices[index] = current_vix

                if settings.entry_prices[index]:
                    change = (
                        (current_vix - settings.entry_prices[index])
                        / settings.entry_prices[index]
                    ) * 100
                    if change <= -settings.STOP_LOSS_PERCENT:
                        signal = "STOP LOSS 🚨 (Trade closed)"
                        settings.entry_prices[index] = None
                    elif change >= settings.TAKE_PROFIT_PERCENT:
                        signal = "TAKE PROFIT ✅ (Trade closed)"
                        settings.entry_prices[index] = None

                trade_signals[index].append(signal)

                message = (
                    f"📊 *Live Trading Signal for {index}*\n"
                    f"💹 *VIX:* {current_vix:.2f}\n"
                    f"📈 *RSI:* {rsi}\n"
                    f"📊 *SMA:* {sma:.2f}\n"
                    f"📉 *EMA:* {ema:.2f}\n"
                    f"🔹 *AI Trend:* {ai_trend}\n"
                    f"🔹 *Sentiment Score:* {sentiment_score}\n"
                    f"🔹 *Signal:* {signal}"
                )

                await bot.send_message(
                    chat_id=settings.TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode=types.ParseMode.MARKDOWN,
                )

            except Exception as e:
                logging.error(f"Trading loop error for {index}: {e}")

        await asyncio.sleep(settings.TRADING_INTERVAL)


# Startup function
async def on_startup():
    logging.info("Bot is starting...")
    asyncio.create_task(trading_loop())


# Main execution
async def main():
    await on_startup()
    await dp.start_polling()  # Start polling with the Dispatcher


def run():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())  # Run the async main function


if __name__ == "__main__":
    run()

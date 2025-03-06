import asyncio
import json
import os
import datetime
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from aiogram import Bot #type: ignore
import websockets #type: ignore
import joblib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

TELEGRAM_BOT_TOKEN="7722875454:AAHaunZl0Vog4TKI7R2k1Bw21jQem6DwCII"
TELEGRAM_CHAT_ID="6172426644"
DERIV_API_TOKEN="CdA9PgARAvqDTVY"
DERIV_APP_ID = "1089"


# Initialize Telegram Bot
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Paths for storing model and data
MODEL_PATH = "linear_model.pkl"
DATA_PATH = "candles_data.csv"

# Symbol Mapping
symbol_names = {
    "R_75": "Volatility 75 Index",
    "R_50": "Volatility 50 Index",
    "R_10": "Volatility 10 Index",
    "R_25": "Volatility 25 Index",
}

async def test_bot():
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 Trading Bot Online!\n\nMonitoring markets for signals...")
        logging.info("Telegram bot connection successful")
    except Exception as e:
        logging.error(f"Telegram connection failed: {str(e)}")
        raise


async def test_deriv_connection():
    logging.info("🔄 Initializing Deriv server connection test...")
    
    uri = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
    
    try:
        async with websockets.connect(
            uri,
            ping_interval=None,
            ping_timeout=None
        ) as ws:
            # Test basic connection
            ping_request = json.dumps({"ping": 1})
            await ws.send(ping_request)
            ping_response = await ws.recv()
            ping_data = json.loads(ping_response)
            
            if "pong" not in ping_data:
                logging.error("❌ Initial server ping failed")
                return False
                
            # Test API token authorization
            auth_request = json.dumps({
                "authorize": DERIV_API_TOKEN,
                "req_id": 1
            })
            await ws.send(auth_request)
            auth_response = await ws.recv()
            auth_data = json.loads(auth_response)
            
            if "error" in auth_data:
                logging.error(f"❌ API Authorization failed: {auth_data['error']['message']}")
                return False
            
            if "authorize" in auth_data:
                balance = auth_data['authorize'].get('balance', 'N/A')
                currency = auth_data['authorize'].get('currency', 'N/A')
                logging.info(f"✅ Connection Successful!")
                logging.info(f"💰 Account Balance: {balance} {currency}")
                logging.info(f"🔑 API Token Valid")
                return True
            
    except Exception as e:
        logging.error(f"❌ Connection error: {str(e)}")
        return False


async def get_candles(symbol, timeframe, count=100):
    logging.info(f"Fetching {timeframe} candles for {symbol}")
    granularity = {"M5": 300, "M15": 900, "H1": 3600}
    try:
        async with websockets.connect("wss://ws.derivws.com/websockets/v3?app_id=1089") as ws:
            await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
            auth_response = await ws.recv()
            
            request = json.dumps({
                "ticks_history": symbol,
                "count": count,
                "granularity": granularity[timeframe],
                "end": "latest"
            })
            await ws.send(request)
            response = await ws.recv()
            data = json.loads(response)
            
            if "error" in data:
                logging.error(f"API Error: {data['error']['message']}")
                return []
                
            candles = data.get("candles", [])
            logging.info(f"Successfully retrieved {len(candles)} candles")
            return candles
    except Exception as e:
        logging.error(f"Error fetching candles: {str(e)}")
        return []

def load_or_build_linear_model():
    logging.info("Loading/Building ML model")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logging.info("Existing model loaded")
    else:
        model = LinearRegression()
        logging.info("New model initialized")
    return model

model = load_or_build_linear_model()
scaler = MinMaxScaler()

def train_linear_model(candles):
    logging.info("Training linear model")
    try:
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
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error(f"Model training failed: {str(e)}")

def ai_forecast(candles):
    logging.info("Generating AI forecast")
    try:
        df = pd.DataFrame(candles)
        df["close"] = df["close"].astype(float)

        last_20 = df["close"].values[-20:].reshape(-1, 1)
        last_20 = scaler.transform(last_20).flatten().reshape(1, -1)

        predicted_price = model.predict(last_20)[0]
        predicted_price = predicted_price * (np.max(df["close"]) - np.min(df["close"])) + np.min(df["close"])

        entry_price = df["close"].iloc[-1]
        if predicted_price > entry_price:
            signal = "BUY"
            tp = entry_price * 1.02
            sl = entry_price * 0.99
        else:
            signal = "SELL"
            tp = entry_price * 0.98
            sl = entry_price * 1.01
            
        logging.info(f"Generated {signal} signal at {entry_price:.2f}")
        return signal, entry_price, tp, sl
    except Exception as e:
        logging.error(f"Forecast generation failed: {str(e)}")
        return None, None, None, None

async def weekly_retrain():
    while True:
        logging.info("Starting scheduled model retraining")
        await asyncio.sleep(604800)  # 1 week
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            train_linear_model(df.to_dict(orient="records"))
            logging.info("Weekly retraining completed")

async def send_signal(symbol, signal, entry, tp, sl):
    message = (
        f"📊 Trading Signal Alert\n\n"
        f"🎯 Asset: {symbol_names[symbol]}\n"
        f"📈 Signal: {signal}\n"
        f"💰 Entry: {entry:.2f}\n"
        f"🎯 Take Profit: {tp:.2f}\n"
        f"🛑 Stop Loss: {sl:.2f}\n\n"
        f"⏰ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown")
        logging.info(f"Signal sent for {symbol}: {signal}")
    except Exception as e:
        logging.error(f"Failed to send signal: {str(e)}")

async def main():
    logging.info("🚀 Initializing Trading Bot")
    
    # Verify all connections
    # deriv_connected = await test_deriv_connection()
    # if not deriv_connected:
    #     logging.critical("❌ Deriv connection failed - Check API token and internet connection")
    #     return
    # logging.info("✅ Deriv connection established")
    
    # await test_bot()
    # logging.info("✅ Telegram connection established")
    
    # Start background tasks
    asyncio.create_task(weekly_retrain())
    logging.info("📊 Weekly model retraining scheduled")
    
    while True:
        try:
            for symbol in symbol_names.keys():
                logging.info(f"📈 Analyzing {symbol}")
                
                # Fetch market data
                candles_m5 = await get_candles(symbol, "M5")
                candles_m15 = await get_candles(symbol, "M15")
                candles_h1 = await get_candles(symbol, "H1")

                if not all([candles_m5, candles_m15, candles_h1]):
                    logging.warning(f"⚠️ Insufficient data for {symbol}, skipping...")
                    continue

                # Generate signals
                signal_m5, entry_m5, tp_m5, sl_m5 = ai_forecast(candles_m5)
                signal_m15, entry_m15, tp_m15, sl_m15 = ai_forecast(candles_m15)
                signal_h1, entry_h1, tp_h1, sl_h1 = ai_forecast(candles_h1)

                # Check signal agreement across timeframes
                if all(x is not None for x in [signal_m5, signal_m15, signal_h1]) and signal_m5 == signal_m15 == signal_h1:
                    logging.info(f"🎯 Signal confirmation found for {symbol}: {signal_m5}")
                    await send_signal(symbol, signal_m5, entry_m5, tp_m5, sl_m5)
                
            logging.info("⏳ Waiting for next analysis cycle")
            await asyncio.sleep(60)
            
        except Exception as e:
            logging.error(f"❌ Main loop error: {str(e)}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot shutdown initiated by user")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")

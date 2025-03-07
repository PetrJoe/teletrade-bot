from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    DERIV_API_TOKEN: str
    DERIV_APP_ID: str

    VIX_INDICES: list[str] = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    STOP_LOSS_PERCENT: int = 2
    TAKE_PROFIT_PERCENT: int = 4  # TODO: Verify if this should be a float
    RSI_PERIOD: int = 5
    SMA_PERIOD: int = 10
    EMA_PERIOD: int = 5
    TRADING_INTERVAL: int = 300
    DATA_RETENTION_DAYS: int = 7
    SENTIMENT_API_URL: str = "https://api.sentiment-analysis.com/news"

    # Paths for storing model and data
    MODEL_PATH: str = "linear_model.pkl"
    DATA_PATH: str = "candles_data.csv"

    # Symbol Mapping
    symbol_names: dict[str, str] = {
        "R_75": "Volatility 75 Index",
        "R_50": "Volatility 50 Index",
        "R_10": "Volatility 10 Index",
        "R_25": "Volatility 25 Index",
    }

    @computed_field
    @property
    def MAX_DATA_POINTS(self) -> int:
        return int(24 * 60 * self.DATA_RETENTION_DAYS / (self.TRADING_INTERVAL / 60))

    @computed_field
    @property
    def DERIV_API_URL(self) -> str:
        return f"wss://ws.derivws.com/websockets/v3?app_id={self.DERIV_APP_ID}"


settings = Settings()  # type: ignore

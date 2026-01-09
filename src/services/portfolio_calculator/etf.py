import numpy as np
import pandas as pd
import yfinance as yf

from typing import List, Dict

from src.services.portfolio_calculator.return_analytics_mixin import ReturnAnalyticsMixin

class ETF(ReturnAnalyticsMixin):
    
    def __init__(self, tickers: List[str], holdings: List[float],
                 shares_outstanding: int):
        self.tickers: List[str] = tickers
        self.holdings: np.ndarray = np.array(holdings, dtype=np.float64)
        self.shares_outstanding: int = shares_outstanding
        self.num_holdings: int = len(tickers)
        # period -> historical_data DataFrame
        self._history_cache: Dict[str, pd.DataFrame] = {}

    def clear_history_cache(self) -> None:
        self._history_cache.clear()

    def get_historical_data(self, period: str) -> pd.DataFrame:
        # return cached if present
        if period in self._history_cache:
            # return a copy so callers can’t mutate the cache
            return self._history_cache[period].copy()

        # Fetch all ticker data
        ticker_data: List[pd.DataFrame] = [
            yf.Ticker(ticker).history(period=period) for ticker in self.tickers
        ]
        index = ticker_data[0].index

        # Stack prices into (num_tickers, num_days) arrays
        open_prices = np.vstack([df['Open'].to_numpy() for df in ticker_data])
        close_prices = np.vstack([df['Close'].to_numpy() for df in ticker_data])
        dividends = np.vstack([df['Dividends'].to_numpy() for df in ticker_data])

        # Vectorized weighted sum: holdings @ prices -> (num_days,)
        etf_open = self.holdings @ open_prices / self.shares_outstanding
        etf_close = self.holdings @ close_prices / self.shares_outstanding
        etf_dividends = self.holdings @ dividends / self.shares_outstanding

        historical_data = pd.DataFrame({
            'Open': etf_open,
            'Close': etf_close,
            'Dividends': etf_dividends,
        }, index=index)

        # 3) Store in cache and return a copy
        self._history_cache[period] = historical_data
        return historical_data.copy()

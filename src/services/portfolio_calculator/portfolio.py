import numpy as np
import pandas as pd

from typing import List, Dict, Sequence

from src.services.portfolio_calculator.return_analytics_mixin import ReturnAnalyticsMixin
from src.services.portfolio_calculator.etf import ETF

class Portfolio(ReturnAnalyticsMixin):
    """
    Portfolio of ETFs with the same analytics interface as ETF.
    """

    YEARLY_MARKET_DAYS: int = 252
    HISTORY_COLUMNS: List[str] = ['Open', 'Close', 'Dividends']
    RETURNS_COLUMNS: List[str] = ['Share Price', 'Shares', 'Total Value', 'Accumulated Dividends']

    def __init__(
        self,
        etfs: Sequence[ETF],
        holdings: Sequence[float],
        shares_outstanding: float = 1.0
    ):
        if len(etfs) == 0:
            raise ValueError("Portfolio must contain at least one ETF.")
        if len(etfs) != len(holdings):
            raise ValueError("Length of etfs and holdings must match.")

        self.etfs: List[ETF] = list(etfs)
        self.holdings: np.ndarray = np.array(holdings, dtype=np.float64)
        self.shares_outstanding: float = float(shares_outstanding)
        self.num_positions: int = len(self.etfs)
        self._history_cache: Dict[str, pd.DataFrame] = {}

    def clear_history_cache(self) -> None:
        self._history_cache.clear()

    def get_historical_data(self, period: str) -> pd.DataFrame:
        if period in self._history_cache:
            return self._history_cache[period].copy()

        # Fetch all ETF histories
        first_hist = self.etfs[0].get_historical_data(period)
        base_index = first_hist.index

        etf_hists = [
            etf.get_historical_data(period).reindex(base_index).ffill().bfill()
            for etf in self.etfs
        ]

        # Stack into (num_etfs, num_days) arrays
        open_prices = np.vstack([h["Open"].to_numpy() for h in etf_hists])
        close_prices = np.vstack([h["Close"].to_numpy() for h in etf_hists])
        dividends = np.vstack([h["Dividends"].to_numpy() for h in etf_hists])

        # Vectorized weighted sum: holdings @ prices -> (num_days,)
        scale = 1.0 / self.shares_outstanding
        portfolio_open = (self.holdings @ open_prices) * scale
        portfolio_close = (self.holdings @ close_prices) * scale
        portfolio_div = (self.holdings @ dividends) * scale

        historical_data = pd.DataFrame({
            "Open": portfolio_open,
            "Close": portfolio_close,
            "Dividends": portfolio_div,
        }, index=base_index)

        self._history_cache[period] = historical_data
        return historical_data.copy()
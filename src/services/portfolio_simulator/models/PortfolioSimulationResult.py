import pandas as pd

from typing import List
from datetime import date
from pydantic import BaseModel


class PortfolioSimulationResult(BaseModel):
    dates: List[date]
    share_prices: List[float]
    shares: List[float]
    total_values: List[float]
    accumulated_dividends: List[float]
    annualized_return: float
    sharpe_ratio: float
    final_value: float
    projected_annual_dividend_income: float

    @classmethod
    def from_dataframe_with_metrics(
        cls, 
        df: pd.DataFrame, 
        annualized_return: float, 
        sharpe_ratio: float,
        final_value: float,
        projected_annual_dividend_income: float
    ) -> "PortfolioSimulationResult":
        return cls(
            dates=[d.date() for d in df.index],
            share_prices=df['Share Price'].tolist(),
            shares=df['Shares'].tolist(),
            total_values=df['Total Value'].tolist(),
            accumulated_dividends=df['Accumulated Dividends'].tolist(),
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            final_value=final_value,
            projected_annual_dividend_income=projected_annual_dividend_income
        )
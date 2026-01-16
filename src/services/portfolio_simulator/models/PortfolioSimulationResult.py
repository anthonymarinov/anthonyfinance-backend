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
        projected_annual_dividend_income: float,
        max_data_points: int = 0
    ) -> "PortfolioSimulationResult":
        """
        Create result from DataFrame, optionally sampling to reduce data points.
        
        Args:
            df: DataFrame with portfolio data
            annualized_return: Calculated annualized return
            sharpe_ratio: Calculated Sharpe ratio
            final_value: Final portfolio value
            projected_annual_dividend_income: Projected yearly dividend income
            max_data_points: Maximum data points to include (0 = no limit)
        """
        # Sample the DataFrame if needed to reduce payload size
        if max_data_points > 0 and len(df) > max_data_points:
            # Always include first and last points
            step = len(df) / (max_data_points - 1)
            indices = [0] + [int(i * step) for i in range(1, max_data_points - 1)] + [len(df) - 1]
            # Remove duplicates and sort
            indices = sorted(set(indices))
            df = df.iloc[indices]
        
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
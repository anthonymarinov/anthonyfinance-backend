from typing import List

from pydantic import BaseModel, Field


class PortfolioSimulationParams(BaseModel):
    """Request body for Portfolio simulation endpoint."""
    
    tickers: List[str] = Field(..., description="List of ETF tickers")
    allocations: List[float] = Field(..., description="List of allocations corresponding to each ticker")
    starting_value: float = Field(1000, description="Portfolio starting value in USD")
    period: str = Field("1y", description="Historical data period (e.g., '1mo', '3mo', '1y', '5y')")
    personal_contributions: float = Field(0.0, description="Amount of personal contributions made periodically")
    contribution_period: int = Field(0, description="Number of market days between contributions (0 for no contributions)")
    include_dividends: bool = Field(True, description="Whether to include dividends in returns")
    is_drip_active: bool = Field(False, description="Whether dividend reinvestment plan (DRIP) is active")
    annual_risk_free_return: float = Field(0.03, description="Annual risk-free return rate for Sharpe ratio calculation")

    class Config:
        extra = "forbid"

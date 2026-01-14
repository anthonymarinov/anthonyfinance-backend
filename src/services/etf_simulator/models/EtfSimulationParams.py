from typing import List

from pydantic import BaseModel, Field


class EtfSimulationParams(BaseModel):
    """Request body for ETF Simulation endpoint."""
    
    tickers: List[str] = Field(..., description="List of ETF tickers")
    holdings: List[float] = Field(..., description="List of holdings corresponding to each ticker")
    shares_outstanding: int = Field(1, description="Total shares outstanding for the ETF")
    starting_shares: float = Field(1, description="Starting number of shares owned")
    period: str = Field("1y", description="Historical data period (e.g., '1mo', '3mo', '1y', '5y')")
    personal_contributions: float = Field(0.0, description="Amount of personal contributions made periodically")
    contribution_period: int = Field(0, description="Number of market days between contributions (0 for no contributions)")
    include_dividends: bool = Field(True, description="Whether to include dividends in returns")
    is_drip_active: bool = Field(False, description="Whether dividend reinvestment plan (DRIP) is active")
    annual_risk_free_return: float = Field(0.03, description="Annual risk-free return rate for Sharpe ratio calculation")

    class Config:
        extra = "forbid"

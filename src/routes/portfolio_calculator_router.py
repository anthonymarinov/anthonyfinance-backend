from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict

from src.services.portfolio_calculator.etf import ETF

router = APIRouter()


@router.get("/portfolio-calculator/etf/get-historical-data", response_model=Dict)
def get_etf_historical_data(
    tickers: List[str] = Query(..., description="List of ETF tickers"),
    holdings: List[float] = Query(..., description="List of holdings corresponding to each ticker"),
    shares_outstanding: int = Query(1, description="Total shares outstanding for the ETF"),
    period: str = Query("1y", description="Historical data period (e.g., '1mo', '3mo', '1y', '5y')")
):
    """
    Get historical data for an ETF composed of multiple tickers.
    """
    try:
        etf = ETF(tickers=tickers, holdings=holdings, shares_outstanding=shares_outstanding)
        return etf.get_historical_data(period=period).to_dict(orient='index')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
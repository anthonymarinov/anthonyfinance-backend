from fastapi import APIRouter, Depends, HTTPException

from src.services.portfolio_calculator.etf import ETF
from src.services.portfolio_calculator.models.etf_returns_params import EtfReturnsParams
from src.services.portfolio_calculator.models.returns_result import ReturnsResult

router = APIRouter()


@router.post("/portfolio-calculator/etf/get-returns", response_model=ReturnsResult)
def get_etf_returns(params: EtfReturnsParams) -> ReturnsResult:
    """
    Get returns for an ETF composed of multiple tickers.
    """
    try:
        etf: ETF = ETF(
            tickers=params.tickers,
            holdings=params.holdings,
            shares_outstanding=params.shares_outstanding
        )
        returns_result: ReturnsResult = etf.get_returns_result(
            starting_shares=params.starting_shares,
            period=params.period,
            personal_contributions=params.personal_contributions,
            contribution_period=params.contribution_period,
            include_dividends=params.include_dividends,
            is_drip_active=params.is_drip_active,
            annual_risk_free_return=params.annual_risk_free_return
        )
        return returns_result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
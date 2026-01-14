from fastapi import APIRouter, HTTPException

from src.services.etf_simulator.Etf import ETF
from src.services.etf_simulator.models.EtfSimulationParams import EtfSimulationParams
from src.services.etf_simulator.models.EtfSimulationResult import EtfSimulationResult

router = APIRouter()


@router.post("/etf-simulator", response_model=EtfSimulationResult)
def simulate_etf(params: EtfSimulationParams) -> EtfSimulationResult:
    """
    Simulate an ETF based on provide parameters.
    
    :param params: ETF simulation input parameters
    :type params: EtfSimulationParams
    :return: ETF simulation results
    :rtype: EtfSimulationResult
    """
    try:
        etf: ETF = ETF(
            tickers=params.tickers,
            holdings=params.holdings,
            shares_outstanding=params.shares_outstanding
        )
        returns_result: EtfSimulationResult = etf.get_returns_result(
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
    
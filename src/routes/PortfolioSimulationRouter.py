from fastapi import APIRouter
from src.services.portfolio_simulator.models.PortfolioSimulationParams import PortfolioSimulationParams
from src.services.portfolio_simulator.models.PortfolioSimulationResult import PortfolioSimulationResult
from src.services.portfolio_simulator.Portfolio import Portfolio

router = APIRouter()

@router.post("/portfolio-simulator", response_model=PortfolioSimulationResult)
def simulate_portfolio(params: PortfolioSimulationParams) -> PortfolioSimulationResult:
    """
    Simulate a portfolio based on provided parameters.
    
    :param params: Portfolio simulation input parameters
    :type params: PortfolioSimulationParams
    :return: Portfolio simulation results
    :rtype: PortfolioSimulationResult
    """
    # Create portfolio with tickers, allocations, and starting value
    portfolio = Portfolio(
        tickers=params.tickers,
        allocations=params.allocations,
        starting_portfolio_value=params.starting_value
    )
    
    # Get simulation results
    return portfolio.get_returns_result(
        period=params.period,
        personal_contributions=params.personal_contributions,
        contribution_period=params.contribution_period,
        include_dividends=params.include_dividends,
        is_drip_active=params.is_drip_active,
        annual_risk_free_return=params.annual_risk_free_return,
        max_data_points=params.max_data_points
    )
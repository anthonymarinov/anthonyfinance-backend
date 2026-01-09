import numpy as np
import pandas as pd

from typing import List, Optional
from src.services.portfolio_calculator.models.returns_result import ReturnsResult

class ReturnAnalyticsMixin():
    """
    Mixin expecting:
      - a concrete `get_historical_data(self, period: str) -> pd.DataFrame`
    """

    YEARLY_MARKET_DAYS: int = 252
    HISTORY_COLUMNS: List[str] = ['Open', 'Close', 'Dividends']
    RETURNS_COLUMNS: List[str] = ['Share Price', 'Shares', 'Total Value', 'Accumulated Dividends']

    def get_historical_data(self, period: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_returns(
        self,
        starting_shares: float,
        period: str,
        personal_contributions: float,
        contribution_period: int,
        include_dividends: bool,
        is_drip_active: bool
    ) -> pd.DataFrame:
        # contribution_period is in units of market days (days out of 252 market days per year)
        historical_data: pd.DataFrame = self.get_historical_data(period)
        close_prices = historical_data['Close'].to_numpy()
        dividends = historical_data['Dividends'].to_numpy()
        n: int = len(close_prices)

        # Pre-allocate arrays for performance
        share_prices: np.ndarray = close_prices.copy()
        total_shares_arr: np.ndarray = np.empty(n, dtype=np.float64)
        total_value_arr: np.ndarray = np.empty(n, dtype=np.float64)
        dividends_received_arr: np.ndarray = np.empty(n, dtype=np.float64)

        total_shares: float = starting_shares
        dividends_received: float = 0.0
        cash_dividends: float = 0.0
        count_dividends_in_value: bool = include_dividends and not is_drip_active

        for i in range(n):
            close_price: float = close_prices[i]
            dividend_per_share: float = dividends[i]

            if contribution_period and (i > 0) and (i % contribution_period == 0):
                total_shares += personal_contributions / close_price

            if dividend_per_share > 0.0:
                div_total: float = dividend_per_share * total_shares
                dividends_received += div_total

                if include_dividends:
                    if is_drip_active:
                        total_shares += div_total / close_price
                    else:
                        cash_dividends += div_total

            total_shares_arr[i] = total_shares
            dividends_received_arr[i] = dividends_received
            total_value_arr[i] = total_shares * close_price + (cash_dividends if count_dividends_in_value else 0.0)

        return pd.DataFrame(
            {
                self.RETURNS_COLUMNS[0]: share_prices,
                self.RETURNS_COLUMNS[1]: total_shares_arr,
                self.RETURNS_COLUMNS[2]: total_value_arr,
                self.RETURNS_COLUMNS[3]: dividends_received_arr,
            },
            index=historical_data.index
        )

    def get_annualized_return(
        self,
        period: str,
        include_dividends: bool,
        is_drip_active: bool,
        returns: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Annualized total return (CAGR) over the given period.
        Assumes we buy 1 share at the start and make no further contributions.
        - include_dividends=False: price-only return
        - include_dividends=True, is_drip_active=False: dividends accrue as cash and are counted in value
        - include_dividends=True, is_drip_active=True: dividends are DRIP'ed at the close

        If `returns` dataframe is included, we use it instead and override everything else
        (more optimal for repetitive calcs).
        """
        num_shares: float = 1.0

        if returns is not None:
            start_value: float = returns["Share Price"].iloc[0] * num_shares
            end_value = returns["Total Value"].iloc[-1].item()
            num_years: float = len(returns) / self.YEARLY_MARKET_DAYS
            return (end_value / start_value) ** (1.0 / num_years) - 1.0

        hist: pd.DataFrame = self.get_historical_data(period)
        n_days: int = len(hist)
        if n_days < 2:
            return float("nan")

        num_years: float = n_days / self.YEARLY_MARKET_DAYS

        returns_df: pd.DataFrame = self.get_returns(
            starting_shares=num_shares,
            period=period,
            personal_contributions=0.0,
            contribution_period=0,
            include_dividends=include_dividends,
            is_drip_active=is_drip_active
        )

        start_value: float = returns_df["Share Price"].iloc[0] * num_shares
        end_value = returns_df["Total Value"].iloc[-1].item()

        return (end_value / start_value) ** (1.0 / num_years) - 1.0

    def get_sharpe_ratio(
        self,
        period: str,
        annual_risk_free_return: float,
        include_dividends: bool = False,
        is_drip_active: bool = False,
        returns: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Sharpe ratio, using daily total-return series and annual risk-free rate (fraction).
        """
        # if returns is passed as arg, we bypass everything else and use it instead
        if returns is not None:
            returns_df = returns
        else:
            returns_df = self.get_returns(
                starting_shares=1.0,
                period=period,
                personal_contributions=0.0,
                contribution_period=0,
                include_dividends=include_dividends,
                is_drip_active=is_drip_active
            )

        daily_returns = returns_df["Total Value"].pct_change().dropna()

        daily_risk_free = annual_risk_free_return / self.YEARLY_MARKET_DAYS
        excess_daily_returns = daily_returns - daily_risk_free

        mean_excess = excess_daily_returns.mean()
        std_excess = daily_returns.std(ddof=1)

        daily_sharpe = mean_excess / std_excess
        annualized_sharpe = daily_sharpe * np.sqrt(self.YEARLY_MARKET_DAYS)

        return annualized_sharpe
    
    def get_returns_result(
        self,
        starting_shares: float,
        period: str,
        personal_contributions: float,
        contribution_period: int,
        include_dividends: bool,
        is_drip_active: bool,
        annual_risk_free_return: float
    ) -> ReturnsResult:
        returns_df = self.get_returns(
            starting_shares=starting_shares,
            period=period,
            personal_contributions=personal_contributions,
            contribution_period=contribution_period,
            include_dividends=include_dividends,
            is_drip_active=is_drip_active
        )

        annualized_return = self.get_annualized_return(
            period=period,
            include_dividends=include_dividends,
            is_drip_active=is_drip_active,
            returns=returns_df
        )

        sharpe_ratio = self.get_sharpe_ratio(
            period=period,
            annual_risk_free_return=annual_risk_free_return,
            include_dividends=include_dividends,
            is_drip_active=is_drip_active,
            returns=returns_df
        )

        return ReturnsResult.from_dataframe_with_metrics(
            returns_df,
            annualized_return,
            sharpe_ratio
        )
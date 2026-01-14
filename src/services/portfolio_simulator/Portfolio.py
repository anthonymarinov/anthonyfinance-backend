import numpy as np
import pandas as pd
import yfinance as yf

from typing import List, Dict, Sequence, Optional

from src.services.portfolio_simulator.models.PortfolioSimulationResult import PortfolioSimulationResult

class Portfolio():
    YEARLY_MARKET_DAYS: int = 252
    HISTORY_COLUMNS: List[str] = ['Open', 'Close', 'Dividends']
    RETURNS_COLUMNS: List[str] = ['Share Price', 'Shares', 'Total Value', 'Accumulated Dividends']

    def __init__(
        self,
        tickers: Sequence[str],
        allocations: Sequence[float],
        starting_portfolio_value: float
    ):
        if len(tickers) == 0:
            raise ValueError("Portfolio must contain at least one ticker.")
        if len(tickers) != len(allocations):
            raise ValueError("Length of tickers and allocations must match.")
        
        # Validate allocations sum to ~100%
        total_allocation = sum(allocations)
        if not (99.9 <= total_allocation <= 100.1):
            raise ValueError(f"Allocations must sum to 100%, got {total_allocation}%")

        self.tickers: List[str] = list(tickers)
        self.allocations: np.ndarray = np.array(allocations, dtype=np.float64) / 100.0  # Convert to decimal
        self.starting_portfolio_value: float = float(starting_portfolio_value)
        self.num_positions: int = len(self.tickers)
        self._history_cache: Dict[str, pd.DataFrame] = {}
        self._initial_shares_cache: Dict[str, np.ndarray] = {}

    def clear_history_cache(self) -> None:
        self._history_cache.clear()
        self._initial_shares_cache.clear()

    def _get_initial_shares(self, period: str) -> np.ndarray:
        """
        Calculate initial shares for each ticker based on starting portfolio value 
        and allocations at the first day of the period.
        
        Returns array of shape (num_positions,) with initial shares for each ticker.
        """
        if period in self._initial_shares_cache:
            return self._initial_shares_cache[period].copy()
        
        # Fetch first-day prices for all tickers
        ticker_data: List[pd.DataFrame] = [
            yf.Ticker(ticker).history(period=period) for ticker in self.tickers
        ]
        
        # Get initial prices (first day close price)
        initial_prices = np.array([df['Close'].iloc[0] for df in ticker_data], dtype=np.float64)
        
        # Calculate dollar allocation for each position
        dollar_allocations = self.starting_portfolio_value * self.allocations
        
        # Calculate initial shares: allocation / price
        initial_shares = dollar_allocations / initial_prices
        
        self._initial_shares_cache[period] = initial_shares
        return initial_shares.copy()

    def get_historical_data(self, period: str) -> pd.DataFrame:
        """
        Get historical portfolio data (Open, Close, Dividends) weighted by allocations.
        Returns a portfolio-level view where prices represent the total portfolio value.
        """
        if period in self._history_cache:
            return self._history_cache[period].copy()

        # Fetch all ticker histories
        ticker_data: List[pd.DataFrame] = [
            yf.Ticker(ticker).history(period=period) for ticker in self.tickers
        ]
        base_index = ticker_data[0].index
        
        # Reindex all to same dates
        ticker_data = [df.reindex(base_index).ffill().bfill() for df in ticker_data]

        # Stack into (num_positions, num_days) arrays
        open_prices = np.vstack([df["Open"].to_numpy() for df in ticker_data])
        close_prices = np.vstack([df["Close"].to_numpy() for df in ticker_data])
        dividends = np.vstack([df["Dividends"].to_numpy() for df in ticker_data])
        
        # Get initial shares for each position
        initial_shares = self._get_initial_shares(period)
        
        # Calculate portfolio values: each ticker contributes (shares * price)
        # Result shape: (num_days,)
        portfolio_open = initial_shares @ open_prices
        portfolio_close = initial_shares @ close_prices
        portfolio_div = initial_shares @ dividends

        historical_data = pd.DataFrame({
            "Open": portfolio_open,
            "Close": portfolio_close,
            "Dividends": portfolio_div,
        }, index=base_index)

        self._history_cache[period] = historical_data
        return historical_data.copy()
    
    def get_returns(
        self,
        period: str,
        personal_contributions: float,
        contribution_period: int,
        include_dividends: bool,
        is_drip_active: bool
    ) -> pd.DataFrame:
        """
        Calculate portfolio returns over time with optional contributions and dividend reinvestment.
        
        Returns DataFrame with columns: Share Price, Shares, Total Value, Accumulated Dividends
        - Share Price: Portfolio value per "unit" (normalized to starting value)
        - Shares: Cumulative "units" of portfolio held
        - Total Value: Total portfolio value in dollars
        - Accumulated Dividends: Cumulative dividends received
        """
        # Get historical data (already represents total portfolio value based on initial allocation)
        historical_data: pd.DataFrame = self.get_historical_data(period)
        close_prices = historical_data['Close'].to_numpy()
        dividends = historical_data['Dividends'].to_numpy()
        n: int = len(close_prices)
        
        # Normalize to price per "share" of portfolio (starting value = 1 share)
        initial_value = close_prices[0]
        share_prices = close_prices / initial_value
        dividend_per_share = dividends / initial_value

        # Pre-allocate arrays for performance
        total_shares_arr: np.ndarray = np.empty(n, dtype=np.float64)
        total_value_arr: np.ndarray = np.empty(n, dtype=np.float64)
        dividends_received_arr: np.ndarray = np.empty(n, dtype=np.float64)

        total_shares: float = 1.0  # Start with 1 "share" of the portfolio
        dividends_received: float = 0.0
        cash_dividends: float = 0.0
        count_dividends_in_value: bool = include_dividends and not is_drip_active

        for i in range(n):
            share_price: float = share_prices[i]
            dividend_ps: float = dividend_per_share[i]

            # Add personal contributions (buy more shares at current price)
            if contribution_period and (i > 0) and (i % contribution_period == 0):
                # Contribution buys more portfolio shares at current value
                contribution_shares = personal_contributions / (share_price * initial_value)
                total_shares += contribution_shares

            # Handle dividends
            if dividend_ps > 0.0:
                div_total: float = dividend_ps * total_shares * initial_value
                dividends_received += div_total

                if include_dividends:
                    if is_drip_active:
                        # DRIP: reinvest dividends into more shares
                        drip_shares = div_total / (share_price * initial_value)
                        total_shares += drip_shares
                    else:
                        # Cash dividends accumulate
                        cash_dividends += div_total

            total_shares_arr[i] = total_shares
            dividends_received_arr[i] = dividends_received
            
            # Total value = (shares * current share price * initial value) + cash dividends
            total_value_arr[i] = (
                total_shares * share_price * initial_value + 
                (cash_dividends if count_dividends_in_value else 0.0)
            )

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
        Calculates return based on starting portfolio value to ending value.
        - include_dividends=False: price-only return
        - include_dividends=True, is_drip_active=False: dividends accrue as cash and are counted in value
        - include_dividends=True, is_drip_active=True: dividends are DRIP'ed

        If `returns` dataframe is included, we use it instead (more optimal for repetitive calcs).
        """
        if returns is not None:
            start_value: float = self.starting_portfolio_value
            end_value = returns["Total Value"].iloc[-1].item()
            num_years: float = len(returns) / self.YEARLY_MARKET_DAYS
            return (end_value / start_value) ** (1.0 / num_years) - 1.0

        hist: pd.DataFrame = self.get_historical_data(period)
        n_days: int = len(hist)
        if n_days < 2:
            return float("nan")

        num_years: float = n_days / self.YEARLY_MARKET_DAYS

        returns_df: pd.DataFrame = self.get_returns(
            period=period,
            personal_contributions=0.0,
            contribution_period=0,
            include_dividends=include_dividends,
            is_drip_active=is_drip_active
        )

        start_value: float = self.starting_portfolio_value
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
        period: str,
        personal_contributions: float,
        contribution_period: int,
        include_dividends: bool,
        is_drip_active: bool,
        annual_risk_free_return: float
    ) -> PortfolioSimulationResult:
        """
        Get complete simulation results including returns data and performance metrics.
        """
        returns_df = self.get_returns(
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

        return PortfolioSimulationResult.from_dataframe_with_metrics(
            returns_df,
            annualized_return,
            sharpe_ratio
        )

import numpy as np
import pandas as pd
import yfinance as yf

from typing import List, Dict, Optional

from src.services.etf_simulator.models.EtfSimulationResult import EtfSimulationResult

class ETF():
    YEARLY_MARKET_DAYS: int = 252
    HISTORY_COLUMNS: List[str] = ['Open', 'Close', 'Dividends']
    RETURNS_COLUMNS: List[str] = ['Share Price', 'Shares', 'Total Value', 'Accumulated Dividends']
    
    def __init__(self, tickers: List[str], holdings: List[float],
                 shares_outstanding: int):
        self.tickers: List[str] = tickers
        self.holdings: np.ndarray = np.array(holdings, dtype=np.float64)
        self.shares_outstanding: int = shares_outstanding
        self.num_holdings: int = len(tickers)
        # period -> historical_data DataFrame
        self._history_cache: Dict[str, pd.DataFrame] = {}

    def clear_history_cache(self) -> None:
        self._history_cache.clear()

    def get_historical_data(self, period: str) -> pd.DataFrame:
        # return cached if present
        if period in self._history_cache:
            # return a copy so callers can’t mutate the cache
            return self._history_cache[period].copy()

        # Fetch all ticker data
        ticker_data: List[pd.DataFrame] = [
            yf.Ticker(ticker).history(period=period) for ticker in self.tickers
        ]
        index = ticker_data[0].index

        # Stack prices into (num_tickers, num_days) arrays
        open_prices = np.vstack([df['Open'].to_numpy() for df in ticker_data])
        close_prices = np.vstack([df['Close'].to_numpy() for df in ticker_data])
        dividends = np.vstack([df['Dividends'].to_numpy() for df in ticker_data])

        # Vectorized weighted sum: holdings @ prices -> (num_days,)
        etf_open = self.holdings @ open_prices / self.shares_outstanding
        etf_close = self.holdings @ close_prices / self.shares_outstanding
        etf_dividends = self.holdings @ dividends / self.shares_outstanding

        historical_data = pd.DataFrame({
            'Open': etf_open,
            'Close': etf_close,
            'Dividends': etf_dividends,
        }, index=index)

        # 3) Store in cache and return a copy
        self._history_cache[period] = historical_data
        return historical_data.copy()
    
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
        # Track contributions for TWR calculation
        contributions_arr: np.ndarray = np.zeros(n, dtype=np.float64)

        total_shares: float = starting_shares
        dividends_received: float = 0.0
        cash_dividends: float = 0.0
        count_dividends_in_value: bool = include_dividends and not is_drip_active

        for i in range(n):
            close_price: float = close_prices[i]
            dividend_per_share: float = dividends[i]

            if contribution_period and (i > 0) and (i % contribution_period == 0):
                total_shares += personal_contributions / close_price
                contributions_arr[i] = personal_contributions

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

        result_df = pd.DataFrame(
            {
                self.RETURNS_COLUMNS[0]: share_prices,
                self.RETURNS_COLUMNS[1]: total_shares_arr,
                self.RETURNS_COLUMNS[2]: total_value_arr,
                self.RETURNS_COLUMNS[3]: dividends_received_arr,
            },
            index=historical_data.index
        )
        # Store contributions as an attribute for TWR calculation
        result_df.attrs['contributions'] = contributions_arr
        return result_df

    def get_annualized_return(
        self,
        period: str,
        include_dividends: bool,
        is_drip_active: bool,
        returns: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Time-Weighted Return (TWR) annualized over the given period.
        
        TWR isolates investment performance from the impact of cash flows (contributions).
        It calculates sub-period returns between each contribution and chain-links them.
        
        - include_dividends=False: price-only return
        - include_dividends=True, is_drip_active=False: dividends accrue as cash and are counted in value
        - include_dividends=True, is_drip_active=True: dividends are DRIP'ed at the close

        If `returns` dataframe is included, we use it instead (more optimal for repetitive calcs).
        """
        if returns is not None:
            returns_df = returns
            num_years: float = len(returns_df) / self.YEARLY_MARKET_DAYS
        else:
            hist: pd.DataFrame = self.get_historical_data(period)
            n_days: int = len(hist)
            if n_days < 2:
                return float("nan")
            num_years = n_days / self.YEARLY_MARKET_DAYS

            returns_df = self.get_returns(
                starting_shares=1.0,
                period=period,
                personal_contributions=0.0,
                contribution_period=0,
                include_dividends=include_dividends,
                is_drip_active=is_drip_active
            )

        total_values = returns_df["Total Value"].to_numpy()
        contributions = returns_df.attrs.get('contributions', np.zeros(len(total_values)))
        
        # Calculate Time-Weighted Return using sub-period linking
        # TWR = product of (1 + sub-period return) - 1
        cumulative_twr = 1.0
        
        for i in range(1, len(total_values)):
            # Value before contribution on day i
            value_before = total_values[i]
            # Value on previous day (after any contribution that day)
            value_prev = total_values[i - 1]
            
            # If there was a contribution on day i, we need to adjust
            # The contribution happens at the start of day i, so we calculate
            # return from (prev_value + contribution) to current value
            contribution = contributions[i] if i < len(contributions) else 0.0
            
            if contribution > 0:
                # Sub-period return: from previous day's end value to current day before contribution
                # Then a new sub-period starts after contribution
                adjusted_prev = value_prev + contribution
                if adjusted_prev > 0:
                    sub_period_return = value_before / adjusted_prev
                    cumulative_twr *= sub_period_return
            else:
                # No contribution: simple return calculation
                if value_prev > 0:
                    sub_period_return = value_before / value_prev
                    cumulative_twr *= sub_period_return
        
        # Annualize the TWR
        if cumulative_twr <= 0:
            return float("-inf")
        
        annualized_return = cumulative_twr ** (1.0 / num_years) - 1.0
        return annualized_return

    def get_sharpe_ratio(
        self,
        period: str,
        annual_risk_free_return: float,
        include_dividends: bool = False,
        is_drip_active: bool = False,
        returns: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Sharpe ratio using contribution-adjusted daily returns.
        
        When contributions are present, we calculate daily returns that exclude
        the impact of cash inflows to get a true measure of risk-adjusted performance.
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

        total_values = returns_df["Total Value"].to_numpy()
        contributions = returns_df.attrs.get('contributions', np.zeros(len(total_values)))
        
        # Calculate contribution-adjusted daily returns
        # For each day, we calculate return as: current_value / (prev_value + contribution)
        n = len(total_values)
        daily_returns = np.empty(n - 1, dtype=np.float64)
        
        for i in range(1, n):
            prev_value = total_values[i - 1]
            curr_value = total_values[i]
            contribution = contributions[i] if i < len(contributions) else 0.0
            
            # Adjusted previous value includes any contribution made on current day
            adjusted_prev = prev_value + contribution
            if adjusted_prev > 0:
                daily_returns[i - 1] = (curr_value / adjusted_prev) - 1.0
            else:
                daily_returns[i - 1] = 0.0

        daily_risk_free = annual_risk_free_return / self.YEARLY_MARKET_DAYS
        excess_daily_returns = daily_returns - daily_risk_free

        mean_excess = np.mean(excess_daily_returns)
        std_returns = np.std(daily_returns, ddof=1)
        
        if std_returns == 0:
            return 0.0

        daily_sharpe = mean_excess / std_returns
        annualized_sharpe = daily_sharpe * np.sqrt(self.YEARLY_MARKET_DAYS)

        return annualized_sharpe

    def get_projected_annual_dividend_income(
        self,
        returns: pd.DataFrame
    ) -> float:
        """
        Calculate projected annual dividend income at end of study period.
        
        This is the annual income you'd expect going forward based on recent dividend rate
        and your final share count.
        """
        accumulated_dividends = returns["Accumulated Dividends"].to_numpy()
        shares = returns["Shares"].to_numpy()
        num_days = len(returns)
        
        final_shares = shares[-1] if len(shares) > 0 else 0.0
        
        if num_days <= self.YEARLY_MARKET_DAYS:
            # Less than 1 year: annualize the total dividends, scaled by final share count
            total_dividends = accumulated_dividends[-1] if len(accumulated_dividends) > 0 else 0.0
            avg_shares = np.mean(shares) if len(shares) > 0 else 0.0
            if avg_shares == 0:
                return 0.0
            annualization_factor = self.YEARLY_MARKET_DAYS / num_days
            # Scale by final shares / average shares to project forward
            return total_dividends * annualization_factor * (final_shares / avg_shares)
        else:
            # More than 1 year: use last year's dividends, scaled by final share count
            lookback_days = min(self.YEARLY_MARKET_DAYS, num_days)
            start_index = num_days - lookback_days
            start_dividends = accumulated_dividends[start_index] if start_index < len(accumulated_dividends) else 0.0
            end_dividends = accumulated_dividends[-1] if len(accumulated_dividends) > 0 else 0.0
            last_year_dividends = end_dividends - start_dividends
            
            # Get average shares over the lookback period
            lookback_shares = shares[start_index:]
            avg_lookback_shares = np.mean(lookback_shares) if len(lookback_shares) > 0 else 0.0
            if avg_lookback_shares == 0:
                return 0.0
            
            # Annualize if lookback is less than a year, and scale to final share count
            annualization_factor = self.YEARLY_MARKET_DAYS / lookback_days
            return last_year_dividends * annualization_factor * (final_shares / avg_lookback_shares)
    
    def get_returns_result(
        self,
        starting_shares: float,
        period: str,
        personal_contributions: float,
        contribution_period: int,
        include_dividends: bool,
        is_drip_active: bool,
        annual_risk_free_return: float
    ) -> EtfSimulationResult:
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

        final_value = returns_df["Total Value"].iloc[-1]
        
        projected_annual_dividend_income = self.get_projected_annual_dividend_income(
            returns=returns_df
        )

        return EtfSimulationResult.from_dataframe_with_metrics(
            returns_df,
            annualized_return,
            sharpe_ratio,
            final_value,
            projected_annual_dividend_income
        )

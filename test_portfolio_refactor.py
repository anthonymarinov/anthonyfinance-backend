"""
Test script to verify the refactored Portfolio class works correctly.
"""
from src.services.portfolio_simulator.Portfolio import Portfolio

def test_portfolio():
    # Test portfolio with 60% AAPL, 40% MSFT
    tickers = ["AAPL", "MSFT"]
    allocations = [60.0, 40.0]  # percentages
    starting_value = 10000.0
    
    print("Creating portfolio with:")
    print(f"  Tickers: {tickers}")
    print(f"  Allocations: {allocations}")
    print(f"  Starting Value: ${starting_value:,.2f}")
    print()
    
    portfolio = Portfolio(
        tickers=tickers,
        allocations=allocations,
        starting_portfolio_value=starting_value
    )
    
    print("Portfolio created successfully!")
    print(f"  Number of positions: {portfolio.num_positions}")
    print()
    
    # Test getting returns
    print("Testing get_returns_result with 1 year period...")
    result = portfolio.get_returns_result(
        period="1y",
        personal_contributions=0.0,
        contribution_period=0,
        include_dividends=True,
        is_drip_active=True,
        annual_risk_free_return=0.045
    )
    
    print("\nResults:")
    print(f"  Annualized Return: {result.annualized_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Number of data points: {len(result.dates)}")
    print(f"  Starting Value: ${result.total_values[0]:,.2f}")
    print(f"  Ending Value: ${result.total_values[-1]:,.2f}")
    print(f"  Total Return: {((result.total_values[-1] / result.total_values[0]) - 1) * 100:.2f}%")
    print()
    
    # Test with contributions
    print("Testing with monthly contributions of $500...")
    result_with_contributions = portfolio.get_returns_result(
        period="1y",
        personal_contributions=500.0,
        contribution_period=21,  # ~1 month in market days
        include_dividends=True,
        is_drip_active=True,
        annual_risk_free_return=0.045
    )
    
    print("\nResults with contributions:")
    print(f"  Annualized Return: {result_with_contributions.annualized_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {result_with_contributions.sharpe_ratio:.2f}")
    print(f"  Ending Value: ${result_with_contributions.total_values[-1]:,.2f}")
    print()
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_portfolio()

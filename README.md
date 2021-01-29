# Algo Trading

| Software  | Package Requirements |
| ------------- | ------------- |
| Python  | robin_stocks, MatPlotLib, NumPy, Pandas, SciPy|

## Summary
The purpose of this project is to provide a generic framework to design and execute algorithmic trading strategies using Robinhood. This project provides traders with the ability to quickly access and visualize stock and option data, view current user holdings, and estimate the percent return of options contracts using the Black-Scholes Model. In the future, I plan to expand this project to provide more features for algorithmic trading such as backtesting and performance analysis.

## Design

### Profile
The Profile class is used to represent a user's Robinhood profile. The class provides access to the user's total equity, cash, and entire portfolio including both stock and option positions.

### Stock
The stock class is used to represent a stock and provide access to stock data. The constructor of the stock class requires the ticker of the particular stock of interest. The stock class includes methods to access many different data points about a particular stock such as:
* Share Price
* Market Cap
* Sector
* Company CEO
* Quarterly Earnings Reports
* Daily, weekly, monthly, 3-month, yearly, and 5-year charts

### StockHolding
The StockHolding class is used to represent a user stock holding. The constructor requires the ticker of the particular stock of interest, the number of shares purchased, and the average cost per share. The StockHolding class extends the stock class and thus has access to all methods of the stock class. In addition, the stockholding class includes methods to calculate and access total equity, percent return, and profit/loss. Finally, the StockHolding class includes a method to close the current position, which will trigger a market sell order on Robinhood upon execution.

### StockPortfolio
The StockPortfolio class is used to represent a user stock portfolio. The constructor requires a list of StockHoldings, which can be accessed directly from the user's portfolio using the from_my_holdings class method. The StockPortfolio class includes the method get_chart, which displays a pie chart of the user's stock holdings, as well as the method liquidate, which triggers a market sell order of the entire stock portfolio on Robinhood upon execution.

### Option
The Option class is used to represent an option contract and provide access to option data. The constructor requires only the option ID, a unique alphanumerical value used to represent a particular option contract. The from_search_by_strike_and_date class method allows the user to search options contracts by ticker, strike price, and expiration date. Similar to the Stock class, the option class provides methods to access option data such as bid-ask spread, the greeks, and number of days until expiration among others. In addition, the option class provides several methods to visualize the hypothetical percent return of a particular option contract. The get_hypothetical method displays a graph connected to three sliders to represent time, stock price, and implied volatility. The user can manipulate these sliders to estimate the total percent return of their option contract. The option class also includes methods to plot the percent return of the contract due to change in time, volatility, and stock price while holding other variables constant.

### OptionHolding
The OptionHolding class is used to represent a user option holding. The constructor requires an optionID, the quantity of contracts purchased, average price, and strategy (long or short). The class methods of from_my_holdings, from_my_holdings_by_ticker, and from_search_by_strike_and_date provide alternative ways to access user option holdings without using the option ID. Similar to the StockHolding class, the OptionHolding class extends the Option class and thus has access to all the methods of the Option class. In addition, the OptionHolding class provides access to total equity, percent return, probability of profit, maximum gain and loss, and total profit/loss.

### OptionPortfolio
The OptionPortfolio class is used to represent a user's option portfolio. The constructor requires a list of OptionHoldings, which can be accessed directly from the user's portfolio using the from_my_holdings class method. The OptionPortfolio class also contains a method to display a pie chart of the user's option holdings.

### Portfolio
The Portfolio class is used to represent a user's entire portfolio of stock and option positions. The class includes methods to access the user's individual stock and option portfolios as well as display a pie chart of the user's entire portfolio.
    
## Portfolio Optimization
The portfolio_optimization file is an example of a use case for the Robinhood-Trading Interface. This simple buy-and-hold trading algorithm uses an inital amount of equity, a list of tickers, a start date, and an end date to calculate the Sharpe Ratio optimizing allocations for each stock in the portfolio. The program then uses the Robinhood Trading Interface to automatically purchase fractional shares of each stock according to these allocations.



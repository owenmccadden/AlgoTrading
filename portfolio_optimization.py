import pandas_datareader as web
from scipy.optimize import minimize
from robinhoodTrading import *

r.authentication.login(username='', password='')
p = Profile()


class Optimization:
    def __init__(self, equity, tickers, start, end):
        self.risk_free_rate = 0.1
        if equity > float(p.cash):
            print('Error. Not enough cash to build portfolio')
            return
        else:
            self.equity = equity
        self.equity = equity
        self.tickers = tickers
        self.allocations = np.full((1, len(self.tickers)), (1 / len(self.tickers)))[0]
        self.start = start
        self.end = end
        self.dataframes = []
        self.closes = []
        self.columns = []
        self.portfolio = None
        self.closes = None
        self.cumulative_return = 0
        self.mean_return = 0
        self.std_return = 0
        self.sharp_ratio = 0
        self.annualized_sharp_ratio = 0

        for ticker, allo in zip(tickers, self.allocations):
            df = web.DataReader(str(ticker), 'yahoo', start, end)
            df = df[['Adj Close']]
            df['Normed Return'] = df['Adj Close'] / df.iloc[0]['Adj Close']
            df['Allocation'] = df['Normed Return'] * allo
            df[str(ticker) + ' Position'] = df['Allocation'] * equity
            self.dataframes.append(df[str(ticker) + ' Position'])
            self.columns.append(str(ticker) + ' Position')

        self.portfolio = pd.concat(self.dataframes, axis=1)
        self.portfolio.columns = self.columns
        self.portfolio['Total'] = self.portfolio.sum(axis=1)
        self.portfolio['Daily Returns'] = self.portfolio['Total'].pct_change(1)
        self.cumulative_return = 100 * (self.portfolio['Total'][-1] / self.portfolio['Total'][0] - 1)
        self.mean_return = self.portfolio['Daily Returns'].mean()
        self.std_return = self.portfolio['Daily Returns'].std()
        self.sharp_ratio = (self.portfolio['Daily Returns'].mean()) / self.portfolio['Daily Returns'].std()
        self.annualized_sharp_ratio = self.sharp_ratio * (252 ** 2)

        closes = []
        cols = []
        for i in self.tickers:
            df = web.DataReader(i, 'yahoo', start, end)
            df = df['Adj Close']
            closes.append(df)
            cols.append(i + ' Adj. Close')

        self.closes = pd.concat(closes, axis=1)
        self.closes.columns = cols

        log_returns = np.log(self.closes / self.closes.shift(1))

        def get_ret_vol_sr(weights):
            weights = np.array(weights)
            ret = np.sum(log_returns.mean() * weights * 252)
            vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
            sr = ret / vol
            return np.array([ret, vol, sr])

        def neg_sharpe(weights):
            return get_ret_vol_sr(weights)[2] * -1

        def check_sum(weights):
            return np.sum(weights) - 1

        cons = ({'type': 'eq', 'fun': check_sum})
        bounds = ((0, 1),) * len(self.tickers)
        init_guess = np.ones(len(self.tickers)) / len(self.tickers)
        opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        rvsr = get_ret_vol_sr(opt_results.x)

        self.optimal_allocations = opt_results.x
        self.optimal_equity_allocations = np.around(opt_results.x * self.equity, 2)
        self.optimal_expected_return = 100 * round(rvsr[0], 2)
        self.optimal_expected_volatility = 100 * round(rvsr[1], 2)
        self.optimal_sharpe_ratio = round(rvsr[2], 2)

    def plot_total_value(self):
        self.portfolio['Total'].plot(figsize=(12, 10))
        plt.show()

    def plot_all_positions(self):
        self.portfolio.drop(['Total', 'Daily Returns'], axis=1).plot(figsize=(12, 10))
        plt.show()

    def get_sharpe_ratio(self):
        return self.sharp_ratio

    def get_annualized_sharp_ratio(self):
        return self.annualized_sharp_ratio

    def monte_carlo_optimization(self, n):
        all_weights = np.zeros((n, len(self.closes.columns)))
        log_returns = np.log(self.closes / self.closes.shift(1))
        returns = np.zeros(n)
        vol = np.zeros(n)
        sharpe = np.zeros(n)

        for i in range(n):
            weights = np.array(np.random.random(len(self.closes.columns)))
            weights = weights / np.sum(weights)
            all_weights[i, :] = weights

            returns[i] = np.sum(log_returns.mean() * weights * 252)

            vol[i] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))

            sharpe[i] = returns[i] / vol[i]

        plt.figure(figsize=(12, 8))
        plt.scatter(vol, returns, c=sharpe, cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(vol[sharpe.argmax()], returns[sharpe.argmax()], c='red', s=75, edgecolors='black')
        plt.show()

        print('Optimal Allocation:')
        print('')
        optimal_allocation = all_weights[sharpe.argmax()]
        for i in range(len(self.tickers)):
            print(str(self.tickers[i]) + ' ' + str(round(optimal_allocation[i] * 100)) + '%')
            print('$' + str(round(self.equity * optimal_allocation[i])))
            print('')

        print('Sharpe Ratio of Optimal Allocation:')
        print(round(sharpe.max(), 2))

        print('')
        print('Expected Return:')
        print(str((100 * round(returns[sharpe.argmax()], 2))) + '%')

        print('')
        print('Expected Volatility:')
        print(str((100 * round(vol[sharpe.argmax()], 2))) + '%')

    def build_optimized_portfolio(self):
        for i in range(0, len(self.tickers)):
            r.orders.order_buy_fractional_by_price(self.tickers[i], self.optimal_equity_allocations[i],
                                                   timeInForce='gfd')

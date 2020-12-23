import robin_stocks as r
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import datetime
from datetime import date
import pandas as pd
import math
from scipy.stats import norm
import time

username = ""
password = ""

r.authentication.login(username=username, password=password)

def reformat(d, option=False):
    year = int(str(d[0]) + str(d[1]) + str(d[2]) + str(d[3]))
    month = int(str(d[5]) + str(d[6]))
    day = int(str(d[8]) + str(d[9]))
    if not option:
        try:
            hour = int(str(d[11]) + str(d[12]))
        except IndexError:
            hour = 0
    else:
        hour = 16
    try:
        minute = int(str(d[14]) + str(d[15]))
    except IndexError:
        minute = 0
    try:
        second = int(str(d[17]) + str(d[18]))
    except IndexError:
        second = 0
    return datetime.datetime(year, month, day, hour=hour, minute=minute, second=second)


class EarningsReport:
    def __init__(self, ticker, year, quarter, estimate, actual, date):
        earnings = r.stocks.get_earnings(ticker)
        self.ticker = ticker
        try:
            self.year = year
        except TypeError:
            self.year = 'none'
        try:
            self.quarter = quarter
        except TypeError:
            self.quarter = 'none'
        try:
            self.estimate = float(estimate)
        except TypeError:
            self.estimate = 'none'
        try:
            self.actual = float(actual)
        except TypeError:
            self.actual = 'none'
        try:
            self.date = date
        except TypeError:
            self.date = 'none'
        try:
            self.beat = self.actual > self.estimate
        except TypeError:
            self.beat = 'none'
        self.summary = 'Q' + str(self.quarter) + ' ' + str(self.year)

    def get_summary(self):
        return self.summary

    def get_year(self):
        return self.year

    def get_quarter(self):
        return self.quarter

    def get_estimate(self):
        return self.estimate

    def get_actual(self):
        return self.actual

    def get_date(self):
        return self.date

    def get_beat(self):
        return self.beat


class Stock:
    def __init__(self, ticker):
        stock = r.stocks.get_stock_quote_by_symbol(ticker)
        fundamentals = r.stocks.get_fundamentals(ticker)[0]
        earnings = r.stocks.get_earnings(ticker)
        self.ticker = ticker
        try:
            self.name = r.stocks.find_instrument_data(ticker)[0]['simple_name']
        except TypeError:
            self.name = 'none'
        try:
            self.description = fundamentals['description']
        except TypeError:
            self.description = 'none'
        try:
            self.sector = fundamentals['sector']
        except TypeError:
            self.sector = 'none'
        try:
            self.industry = fundamentals['industry']
        except TypeError:
            self.industry = 'none'
        try:
            self.ceo = fundamentals['ceo']
        except TypeError:
            self.ceo = 'none'
        try:
            self.location = fundamentals['headquarters_city'] + ', ' + fundamentals['headquarters_state']
        except TypeError:
            self.location = 'none'
        try:
            self.num_employees = float(fundamentals['num_employees'])
        except TypeError:
            self.num_employees = 'none'
        try:
            self.year_founded = float(fundamentals['year_founded'])
        except TypeError:
            self.year_founded = 'none'
        try:
            self.share_price = float(stock['last_trade_price'])
        except TypeError:
            self.share_price = 'none'
        try:
            self.after_hours_price = float(stock['last_extended_hours_trade_price'])
        except TypeError:
            self.after_hours_price = 'none'
        try:
            self.stock_volume = float(fundamentals['volume'])
        except TypeError:
            self.stock_volume = 'none'
        try:
            self.average_volume = float(fundamentals['average_volume'])
        except TypeError:
            self.average_volume = 'none'
        try:
            self.high_52_weeks = float(fundamentals['high_52_weeks'])
        except TypeError:
            self.high_52_weeks = 'none'
        try:
            self.low_52_weeks = float(fundamentals['low_52_weeks'])
        except TypeError:
            self.low_52_weeks = 'none'
        try:
            self.float = float(fundamentals['float'])
        except TypeError:
            self.float = 'none'
        try:
            self.market_cap = float(fundamentals['market_cap'])
        except TypeError:
            self.market_cap = 'none'
        try:
            self.shares_outstanding = float(fundamentals['shares_outstanding'])
        except TypeError:
            self.shares_outstanding = 'none'
        try:
            self.pb_ratio = float(fundamentals['pb_ratio'])
        except TypeError:
            self.pb_ratio = 'none'
        try:
            self.pe_ratio = fundamentals['pe_ratio']
        except TypeError:
            self.pe_ratio = 'none'
        self.earnings = []
        try:
            for i in earnings:
                try:
                    self.earnings.append(EarningsReport(self.ticker, i['year'], i['quarter'], i['eps']['estimate'],
                                                        i['eps']['actual'], reformat(i['report']['date'])))
                except TypeError:
                    self.earnings.append('none')
        except TypeError:
            self.earnings = 'none'

    def get_ticker(self):
        return self.ticker

    def get_share_price(self):
        return self.share_price

    def get_after_hours_price(self):
        return self.after_hours_price

    def get_stock_volume(self):
        return self.stock_volume

    def get_average_volume(self):
        return self.average_volume

    def get_high_52_weeks(self):
        return self.high_52_weeks

    def get_low_52_weeks(self):
        return self.low_52_weeks

    def get_float(self):
        return self.float

    def get_market_cap(self):
        return self.market_cap

    def get_shares_outstanding(self):
        return self.shares_outstanding

    def get_pb_ratio(self):
        return self.pb_ratio

    def get_pe_ratio(self):
        return self.pe_ratio

    def get_data(self, interval, span):
        values = []
        dates = []
        for i in r.stocks.get_stock_historicals(self.ticker, interval=interval, span=span):
            values.append(float(i['open_price']))
            dates.append(reformat(i['begins_at']))
        df = pd.DataFrame(data=values, index=dates, columns=[self.get_ticker() + ' price'])
        return df

    def get_graph(self, interval, span):
        df = self.get_data(interval, span)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=5, colspan=1)
        ax1.plot(df.index, df[self.get_ticker() + ' price'])
        plt.title(self.ticker)
        plt.xticks([])
        plt.show()

    def get_daily_chart(self):
        self.get_graph('5minute', 'day')

    def get_monthly_chart(self):
        self.get_graph('day', 'month')

    def get_3_month_chart(self):
        self.get_graph('day', '3month')

    def get_yearly_chart(self):
        self.get_graph('day', 'year')

    def get_5_year_chart(self):
        self.get_graph('week', '5year')

    def get_earnings(self):
        return self.earnings

    def get_earnings_by_quarter(self, quarter):
        for i in self.earnings:
            if i.summary == quarter:
                return i


class StockHolding(Stock):
    def __init__(self, ticker, share_quantity, avg_cost):
        Stock.__init__(self, ticker)
        self.share_quantity = share_quantity
        self.avg_cost = avg_cost
        self.equity = int(self.get_share_price()) * self.share_quantity
        try:
            self.percent_change = (self.get_share_price() - self.avg_cost) / self.avg_cost * 100
        except ZeroDivisionError:
            self.percent_change = 0;

    @classmethod
    def from_my_holdings(cls, ticker):
        my_stocks = r.build_holdings()
        return cls(ticker, float(my_stocks[ticker]['quantity']), float(my_stocks[ticker]['average_buy_price']))

    def get_share_quantity(self):
        return self.share_quantity

    def get_avg_cost(self):
        return self.avg_cost

    def get_equity(self):
        return self.equity

    def get_percent_change(self):
        return self.percent_change

    def get_profit(self):
        return self.get_share_price() - self.avg_cost * self.share_quantity

    def close_position(self):
        r.order_sell_market(self.share_quantity)


class StockPortfolio:
    def __init__(self, holdings):
        self.stock_holdings = []
        for i in holdings:
                self.stock_holdings.append(i)

    @classmethod
    def from_my_holdings(cls):
        holdings = []
        for key, value in r.build_holdings().items():
            holdings.append(StockHolding.from_my_holdings(str(key)))

        return cls(holdings)

    def get_stock_holdings(self):
        return self.stock_holdings

    def liquidate(self):
        for i in self.stock_holdings:
            i.close_position()

    def get_chart(self):
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect='equal'))
        values = []
        labels = []

        for i in self.get_stock_holdings():
            values.append(i.get_equity())
            labels.append(i.get_ticker())

        values = np.array(values).astype(np.float)

        def func(pct, vals):
            absolute = int(pct / 100. * np.sum(vals))
            return '{:.1f}%\n(${:d})'.format(pct, absolute)

        wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values), textprops=dict(color='w'))

        ax.legend(wedges, labels, title='Tickers', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=10, weight='bold')

        ax.set_title('Stock Portfolio')

        plt.show()


class Option(Stock):
    def __init__(self, option_id):
        self.id = option_id
        self.ticker = r.options.get_option_instrument_data_by_id(self.id, info='chain_symbol')
        Stock.__init__(self, self.ticker)
        self.strike = float(r.options.get_option_instrument_data_by_id(self.id, info='strike_price'))
        self.expiration = reformat(r.options.get_option_instrument_data_by_id(self.id, info='expiration_date'),
                                   option=True)
        self.type = r.options.get_option_instrument_data_by_id(self.id)['type']
        self.option_price = float(r.options.get_option_market_data_by_id(self.id, info='adjusted_mark_price')[0])
        self.bid = float(r.options.get_option_market_data_by_id(self.id, info='bid_price')[0])
        self.ask = float(r.options.get_option_market_data_by_id(self.id, 'ask_price')[0])
        self.break_even = float(r.options.get_option_market_data_by_id(self.id, 'break_even_price')[0])
        self.open_interest = float(r.options.get_option_market_data_by_id(self.id, info='open_interest')[0])
        self.option_volume = float(r.options.get_option_market_data_by_id(self.id, info='volume')[0])
        self.long_prob = float(r.options.get_option_market_data_by_id(self.id, info='chance_of_profit_long')[0])
        self.short_prob = float(r.options.get_option_market_data_by_id(self.id, info='chance_of_profit_short')[0])
        self.delta = float(r.options.get_option_market_data_by_id(self.id, info='delta')[0])
        self.gamma = float(r.options.get_option_market_data_by_id(self.id, 'gamma')[0])
        self.iv = float(r.options.get_option_market_data_by_id(self.id, 'implied_volatility')[0])
        self.rho = float(r.options.get_option_market_data_by_id(self.id, info='rho')[0])
        self.theta = float(r.options.get_option_market_data_by_id(self.id, info='theta')[0])
        self.vega = float(r.options.get_option_market_data_by_id(self.id, info='vega')[0])

    @classmethod
    def from_search_by_strike_and_date(cls, ticker, exp_date, strike, type):
        return cls(r.options.find_options_by_expiration_and_strike(ticker, exp_date, strike,
                                                                   optionType=type, info='id')[0])

    def get_id(self):
        return self.id

    def get_summary(self):
        strike = round(float(self.strike), ndigits=2)
        return self.ticker + ' ' + str(self.expiration.month) + '/' + str(self.expiration.day) + '/' + \
               str(self.expiration.year) + ' $' + str(strike) + ' ' + self.get_type()


    def get_hypothetical(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.4)
        plt.title(self.get_summary())
        plt.ylabel('Percent Return')
        plt.ylim(-100, 2000)
        plt.xlim(0, 1)

        labels = [item.get_text() for item in ax.get_xticklabels()]

        empty_string_labels = [''] * len(labels)
        ax.set_xticklabels(empty_string_labels)

        t = np.arange(0, 1, 0.99999)
        data = []
        for i in t:
            data.append(0)
        l, = plt.plot(t, data, lw=2)
        ax.margins(x=0)

        axcolor = 'lightgoldenrodyellow'
        ax_price = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
        ax_time = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
        ax_vol = plt.axes([0.2, 0.3, 0.65, 0.03], facecolor=axcolor)

        price = Slider(ax_price, 'Price', self.get_share_price() / 5, self.get_share_price()*5,
                       valinit=self.get_share_price())
        time = Slider(ax_time, 'Days Remaining', 0, self.get_days_until_expiration(), valinit=self.get_days_until_expiration())
        vol = Slider(ax_vol, 'Implied Volatility', self.get_iv() / 5, self.get_iv() * 5, valinit=self.get_iv())

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        def update(val):
            new_data = []
            for i in t:
                new_data.append(self.get_expected_percent_return(price.val, time.val, vol.val))
            l.set_ydata(new_data)
            print(self.get_expected_price(price.val, time.val, vol.val))
            fig.canvas.draw_idle()

        price.on_changed(update)
        time.on_changed(update)
        vol.on_changed(update)

        def reset(event):
            price.reset()
            time.reset()
            vol.reset()

        button.on_clicked(reset)
        plt.show()

    def get_strike(self):
        return self.strike

    def get_expiration(self):
        return self.expiration

    def get_days_until_expiration(self):
        d = datetime.date(self.get_expiration().year, self.get_expiration().month, self.get_expiration().day) \
            - date.today()
        return d.days

    def get_type(self):
        return self.type

    @staticmethod
    def get_call_price(p, x, r, t, v):
        t = t / 365

        def get_d1():
            return ((np.log(p / x)) + (r + 0.5 * (math.pow(v, 2))) * t) / (v * np.sqrt(t))

        return norm.cdf(get_d1()) * p - norm.cdf(get_d1() - v * np.sqrt(t)) * x / math.pow(np.e, r * t)

    @staticmethod
    def get_put_price(p, x, r, t, v):
        t = t / 365

        def get_d1():
            return ((np.log(p / x)) + (r + 0.5 * (math.pow(v, 2))) * t) / (v * np.sqrt(t))

        return norm.cdf(-1 * (get_d1() - v * np.sqrt(t))) * x / math.pow(np.e, r * t) - norm.cdf(-1 * get_d1()) * p

    def get_black_scholes(self):
        if self.type == 'call':
            return self.get_call_price(float(self.get_share_price()), self.strike, 0.01,
                                       self.get_days_until_expiration(), self.iv)
        else:
            return self.get_put_price(float(self.get_share_price()), self.strike, 0.01,
                                      self.get_days_until_expiration(), self.iv)

    def get_option_price(self):
        return self.option_price

    def get_expected_price(self, price, time_remaining, vol):
        if self.type == 'call':
            return self.get_call_price(price, self.strike, 0.01, time_remaining, vol)
        else:
            return self.get_put_price(price, self.strike, 0.01, time_remaining, vol)

    def get_expected_percent_return(self, price, time_remaining, vol):
        expected_price = self.get_expected_price(price, time_remaining, vol)
        return (expected_price - self.get_option_price()) / self.get_option_price() * 100

    def get_price_discrepancy(self):
        return self.option_price - self.get_black_scholes()

    def get_discrepancy_percentage(self):
        return self.get_price_discrepancy() / self.get_black_scholes() * 100

    def get_bid(self):
        return self.bid

    def get_ask(self):
        return self.ask

    def get_bid_ask_spread(self):
        return self.ask - self.bid

    def get_spread_percentage(self):
        return self.get_bid_ask_spread() / self.bid * 100

    def get_break_even(self):
        return self.break_even

    def get_open_interest(self):
        return self.open_interest

    def get_option_volume(self):
        return self.option_volume

    def get_long_probability(self):
        return self.long_prob

    def get_short_probability(self):
        return self.short_prob

    def get_delta(self):
        return self.delta

    def get_delta_return(self, x):
        initial_price = self.get_option_price()
        expected_price = self.get_expected_price(x, self.get_days_until_expiration(), self.get_iv())
        return (expected_price - initial_price) / initial_price * 100

    def get_delta_chart(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.4)
        plt.title(self.get_summary())
        plt.ylabel('Percent Return')
        plt.ylim(-100, 500)
        plt.xlabel('Stock Price')
        # plt.xlim(self.get_share_price() / 2, self.get_share_price() * 2)
        plt.xlim(0, 60)
        # t = np.arange(self.get_share_price() / 2, self.get_share_price() * 2, 1)
        t = np.arange(0, 60, 1)

        data = []
        for i in t:
            data.append(self.get_delta_return(i))
        l, = plt.plot(t, data, lw=2)
        ax.margins(x=0)

        axcolor = 'lightgoldenrodyellow'
        plt.show()

    def get_gamma(self):
        return self.gamma

    def get_iv(self):
        return self.iv

    def get_rho(self):
        return self.rho

    def get_theta(self):
        return self.theta

    def get_theta_return(self, x):
        initial_price = self.get_option_price()
        expected_price = self.get_expected_price(self.get_share_price(), self.get_days_until_expiration() - x,
                                                 self.get_iv())
        return (expected_price - initial_price) / initial_price * 100

    def get_theta_chart(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.4)
        plt.title(self.get_summary())
        plt.ylabel('Percent Return')
        plt.ylim(-100, 50)
        plt.xlabel('Days From Now')
        plt.xlim(0, self.get_days_until_expiration())
        t = np.arange(0, self.get_days_until_expiration(), 1)
        data = []
        for i in t:
            data.append(self.get_theta_return(i))
        l, = plt.plot(t, data, lw=2)
        ax.margins(x=0)

        axcolor = 'lightgoldenrodyellow'
        plt.show()

    def get_vega(self):
        return self.vega


class OptionHolding(Option):
    def __init__(self, option_id, quantity, avg_price, strategy):
        Option.__init__(self, option_id)
        self.id = option_id
        self.quantity = float(quantity)
        self.avg_price = float(avg_price)
        self.strategy = strategy
        self.equity = self.get_option_price() * self.quantity * 100
        self.percent_return = (self.get_option_price() - self.avg_price) / self.avg_price * 100
        self.profit = (self.get_option_price() - self.avg_price) * self.quantity * 100

    @classmethod
    def from_my_holdings(cls, option_id):
        my_options = r.options.get_open_option_positions()
        quantity = None
        avg_price = None
        strategy = None

        for i in my_options:
            if i['option_id'] == option_id:
                quantity = float(i['quantity'])
                avg_price = float(i['average_price'])
                strategy = i['type']

        return cls(option_id, quantity, avg_price, strategy)

    @classmethod
    def from_my_holdings_by_ticker(cls, ticker):
        my_options = r.options.get_open_option_positions()
        option_id = None
        quantity = None
        avg_price = None
        strategy = None

        for i in my_options:
            if i['chain_symbol'] == ticker:
                option_id = i['option_id']
                quantity = float(i['quantity'])
                avg_price = float(i['average_price'])
                strategy = i['type']

        return cls(option_id, quantity, avg_price, strategy)

    @classmethod
    def from_search_by_strike_and_date(cls, ticker, exp_date, strike, type, quantity, avg_price, strategy):
        id = r.options.find_options_by_expiration_and_strike(ticker, exp_date, strike, optionType=type,
                                                             info='id')
        return cls(id, quantity, avg_price, strategy)

    def get_quantity(self):
        return self.quantity

    def get_average_price(self):
        return self.avg_price

    def get_strategy(self):
        return self.strategy

    def get_probability(self):
        if self.strategy == 'long':
            return self.get_long_probability()
        else:
            return self.get_short_probability()

    def get_equity(self):
        return self.equity

    def get_percent_return(self):
        return self.percent_return

    def get_profit(self):
        return self.profit

    def get_max_loss(self):
        if self.strategy == 'long':
            return self.avg_price * self.quantity * 100
        else:
            return 'infinite'

    def get_max_gain(self):
        if self.strategy == 'long':
            return 'infinite'
        else:
            return self.avg_price * self.quantity * 100


class OptionPortfolio:
    def __init__(self, holdings):
        self.options = []
        for i in holdings:
            self.options.append(i)

    @classmethod
    def from_my_holdings(cls):
        my_options = r.options.get_open_option_positions()
        holdings = []
        for i in my_options:
            holdings.append(OptionHolding(i['option_id'], i['quantity'], i['average_price'], i['type']))

        return cls(holdings)

    def get_option_positions(self):
        return self.options

    def get_options_by_ticker(self, ticker):
        positions = []
        for i in self.options:
            if i.get_ticker() == ticker:
                positions.append(i)

        return positions

    def get_chart(self):
        fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect='equal'))
        values = []
        labels = []

        for i in self.options:
            labels.append(i.get_summary())
            values.append(i.get_equity())

        values = np.array(values).astype(np.float)

        def func(pct, vals):
            absolute = int(pct / 100. * np.sum(vals))
            return '{:.1f}%\n(${:d})'.format(pct, absolute)

        wedges, texts, autotexts = ax.pie(values, autopct=lambda pct: func(pct, values), textprops=dict(color='w'))

        ax.legend(wedges, labels, title='Positions', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=10, weight='bold')

        ax.set_title('Option Portfolio')

        plt.show()

class Profile:
    def __init__(self):
        profile = r.build_user_profile()
        self.equity = profile['equity']
        self.cash = profile['cash']
        self.dividend_total = profile['dividend_total']
        self.portfolio = Portfolio(StockPortfolio.from_my_holdings(), OptionPortfolio.from_my_holdings())

    def get_equity(self):
        return self.equity

    def get_cash(self):
        return self.cash

    def get_portfolio(self):
        return self.portfolio


class Portfolio:
    def __init__(self, stock_portfolio, option_portfolio):
        self.stock_portfolio = stock_portfolio
        self.option_portfolio = option_portfolio
        self.stocks = stock_portfolio.get_stock_holdings()
        self.options = option_portfolio.get_option_positions()
        self.equity = r.build_user_profile()['equity']
        self.cash = r.build_user_profile()['cash']

    def get_stock_portfolio(self):
        return self.stock_portfolio

    def get_stocks(self):
        return self.stocks

    def get_option_portfolio(self):
        return self.option_portfolio

    def get_options(self):
        return self.options

    def get_chart(self):
        ig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))

        holdings = [self.cash]
        labels = ['Cash']
        for i in self.stocks:
            holdings.append(i.get_equity())
            labels.append(i.get_ticker())

        for i in self.options:
            holdings.append(i.get_equity())
            labels.append(i.get_summary())

        def get_wedge_labels(i):
            return '${}'.format(i)

        wedge_labels = []

        for i in holdings:
            i = float(i)
            i = round(i, 2)

        for i in holdings:
            wedge_labels.append(get_wedge_labels(i))

        wedges, texts = ax.pie(holdings, wedgeprops=dict(width=0.5), startangle=90)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        ax.legend(wedges, labels, title='Positions', loc='upper left', bbox_to_anchor=(-0.5, 0, 0.5, 1))

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(wedge_labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment, **kw)

        ax.set_title("Portfolio")
        plt.show()
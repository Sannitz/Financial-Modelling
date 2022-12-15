"""
Paper Traders Helper Functions

Very custom to team members local machines. Please adjust file directories if necessary and contact the team for API_KEYS to EOD. 
"""


import pandas as pd
import numpy as np
from statsmodels.stats.stattools import jarque_bera
from pypfopt import risk_models
from eod import EodHistoricalData
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


API_KEY = '5e6b41389b9ab8.33631133' # API Keys for EOD data. Please contact the team if necessary
client = EodHistoricalData(API_KEY)


def get_bloomberg_data(full: bool = False) -> pd.DataFrame:
    """
    Get updated Bloomberg data. Very custom to the local machine of a team member and needs to be adjusted if otherwise used. 

    :param full: (bool) Boolean if all assets should be loaded from the local directory or only FI assets. 
    :return: (pd.DataFrame) returns a dataframe with the new data
    """
    df = pd.DataFrame()
    if full: names = ['dbc', 'ch', 'it', 'rf', 'shel', 'smi', 'wmt']
    else: names = ['ch', 'it', 'rf']

    for n in names:
        tmp = pd.read_excel(f'C:/Users/gobel/code/research_env/USI/financial_modelling/data/{n}.xlsx')
        tmp.set_index(pd.to_datetime(tmp.Date), inplace=True)
        if len(tmp.columns) > 2: 
            tmp.columns = ['a', 'b', 'c']
            df[n] = tmp.c
        else:            
            tmp.columns = ['a', 'b']
            df[n] = tmp.b

    if full: df.columns = ['DBC.US','ch', 'it', 'rf', 'SHEL', 'CSSMI.SW', 'WMT']
    else: df.columns = names
    return df


def get_updated_data() -> pd.DataFrame:
    """
    Uses EOD data (key needed -> please contact team if necessary) to get updated information on the traded assets. 

    :return: (pd.DataFrame)
    """
    df = pd.DataFrame()

    for i in ['DBC.US', 'SHEL', 'CSSMI.SW', 'WMT']:
        tmp = pd.DataFrame(client.get_prices_eod(i, period='d', order='a', from_='2010-01-01'))
        tmp.set_index(pd.to_datetime(tmp.date), inplace=True)
        df[i] = tmp.adjusted_close
    return df


def add_fx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that takes a dataframe of prices and adds the FX prices for EUR & CHF / USD.     
    """
    eur = pd.DataFrame(client.get_prices_eod('EURUSD.FOREX', period='d', order='a', from_='2010-01-01'))
    eur.set_index(pd.to_datetime(eur.date), inplace=True)
    df['EUR'] = eur.adjusted_close

    chf = pd.DataFrame(client.get_prices_eod('CHFUSD.FOREX', period='d', order='a', from_='2010-01-01'))
    chf.set_index(pd.to_datetime(chf.date), inplace=True)
    df['CHF'] = chf.adjusted_close
    return df


def calculate_beta(df: pd.DataFrame, becnhmark: str = 'AOK'):
    out = df.copy()
    tmp = pd.DataFrame(client.get_prices_eod(becnhmark, period='d', order='a', from_='2010-01-01'))
    tmp.set_index(pd.to_datetime(tmp.date), inplace=True)
    out[becnhmark] = tmp.close.pct_change()
    cov = out.cov()
    return cov.loc['Portfolio', becnhmark] / out[becnhmark].var()


def plot_asset_returns(df: pd.DataFrame):
    c_ret = df.copy()
    _, ax = plt.subplots(figsize=(15,7), facecolor='white')
    c_ret -= 1 
    
    c_ret.columns = ['Portfolio', 'CH Bond', 'DBC', 'IT Bond', 'SHEL', 'SMI', 'WMT', 'Risk Free']
        
    for column in c_ret.drop('Portfolio', axis=1):
        ax.plot(c_ret[column], marker='', linewidth=1, alpha=0.6, label=column)

    ax.plot(c_ret['Portfolio'], marker='', color='royalblue', linewidth=4, alpha=0.9, label='Portfolio')
    # ax.plot(c_ret, label = df.columns)
    # ax.set_xlim(0,12)

    # num=0
    # for i in c_ret.values[-1]:
    #     name=list(c_ret)[num]
    #     num+=1
    #     if name != 'Portfolio':
    #         ax.text(c_ret.index[-1] + timedelta(hours=2), i, name, horizontalalignment='left', color='grey')
    # ax.set_xlim(c_ret.index[0], c_ret.index[-1] + timedelta(days=2))
    # ax.text(c_ret.index[-1] + timedelta(hours=2), c_ret.Portfolio.tail(1), 'Portfolio', horizontalalignment='left', color='royalblue')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.set_title('Returns per asset')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    # plt.show()

def get_ir(df: pd.DataFrame, benchmark: str = 'AOK', weekly=False):
    out = df.copy()
    tmp = pd.DataFrame(client.get_prices_eod(benchmark, period='d', order='a', from_='2015-01-01'))
    tmp.set_index(pd.to_datetime(tmp.date), inplace=True)
    if weekly: tmp = tmp.resample('1w').last()
    out[benchmark] = tmp.adjusted_close.pct_change().dropna()
    ret = out.ewm(span=252*5).mean().iloc[-1]

    ann_port = (1+ret.Portfolio)**252 - 1
    ann_bench = (1+ret[benchmark])**252 - 1

    ir = (out.Portfolio - out[benchmark]).std() * np.sqrt(252)
    print((ann_port - ann_bench), ann_port, ann_bench)
    print((ret.Portfolio - ret[benchmark])*252, ret.Portfolio, ret[benchmark])
    return (ann_port - ann_bench) / ir, ir

def calculate_sharpe(returns: float, std: float, risk_fee: float = 0.04557, periods: int = 1):
    num = returns - risk_fee / 252 * periods
    denom = std * np.sqrt(periods)
    return num / denom

def calculate_sortino(returns: pd.DataFrame, risk_fee: float = 0.04557, periods: int = 1):
    num = returns.mean() - risk_fee / 252 * periods
    denom = returns[returns < 0].std() * np.sqrt(periods)
    return num / denom

def get_treynor(returns: pd.DataFrame, risk_free: float = 0.04557, period = False, benchmark: str = 'AOK'):
    beta = calculate_beta(returns, benchmark)
    if period: return ((returns.Portfolio[-1] - 1 - risk_free/252)*len(returns.index)) / beta
    else: return (returns.mean()*252 - risk_free) / beta


def portfolio_return(start: str, end: str, returns: pd.DataFrame, risk_free: float = 0.04557, benchmark: str = 'AOK',  plot: bool =True) -> dict:
    # names = ['Portfolio', 'Risky Portfolio', 'DBC.US', 'SHEL', 'CSSMI.SW', 'WMT', 'EUR', 'CHF']
    df = returns[start:end] #[names]
    n = len(df.index)
    c_ret = (1+df).cumprod()
    c_ret.iloc[0] = 1.0
    
    if plot:
        plot_asset_returns(c_ret)

    drawdowns = get_max_drawdown(c_ret.Portfolio)
    max_drawdown = max(drawdowns)


    _, ax = plt.subplots(2, 1, figsize=(15,7), facecolor='white', sharex=True,  height_ratios=[2, 1])

    ax[0].plot(c_ret.Portfolio-1, label='% Return')
    ax[1].plot(-drawdowns, label='% Drawdown', color='red')

    ax[0].set_title('Paper Traders Performance')
    for i in range(2):
        ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax[i].legend()
    ax[1].set_xlabel('Date')
    ax[0].set_ylabel('Return')
    plt.subplots_adjust(wspace=0, hspace=0)

    exp_return = df.Portfolio.ewm(span=252*5).mean().iloc[-1]
    daily_return = df.Portfolio.mean()
    daily_std = df.Portfolio.std()
    exp_std = risk_models.exp_cov(df.Portfolio, span=252*5, returns_data=True).loc['Portfolio', 'Portfolio']


    out = {}
    out['start'] = c_ret.index[0]
    out['end'] = c_ret.index[-1]
    out['Total Return Pct'] = round((c_ret.Portfolio[-1] - 1)*100, 4) 
    out['Total Return $'] = round((c_ret.Portfolio[-1] - 1) * 100_000_000, 4)
    out['Daily avg. Return'] = round(daily_return*100, 4)
    out['Daily Expected Return'] = round(exp_return*100, 4)
    out['Daily standard deviation'] = round(daily_std*100, 4) 
    out['Daily Expected std'] = round(exp_std, 4)
    out['Daily Sharpe (Ex-Post)'] = round(calculate_sharpe(exp_return, daily_std), 4)
    out['Daily Sharpe (Ex-Ante)'] = round(calculate_sharpe(daily_return, daily_std), 4)
    out['Period Sharpe (Ex-Post)'] = round(calculate_sharpe(exp_return, daily_std, periods=n), 4)
    out['Period Sharpe (Ex-Ante)'] = round(calculate_sharpe(daily_return, daily_std, periods=n), 4)
    out['Period Sharpe'] = round(((c_ret.Portfolio[-1] - 1) - risk_free/252*n) / (df.Portfolio.std() * np.sqrt(n)), 4)
    out['Period Sharpe Ann'] = round(((c_ret.Portfolio[-1]**(252/n) - 1) - risk_free) / (df.Portfolio.std() * np.sqrt(252)), 4)
    out['Daily Sortino'] = round((df.Portfolio.mean() - risk_free/252) / df[df.Portfolio < 0].Portfolio.std(), 4)
    out['Period Sortino'] = round(((c_ret.Portfolio[-1] - 1) - risk_free/252*n) / (df[df.Portfolio < 0].Portfolio.std() * np.sqrt(n)), 4)
    out['Daily Treynor (AOK)'] = round(get_treynor(df[['Portfolio']], benchmark='SHY').Portfolio, 4)
    ir, te = get_ir(df[['Portfolio']])
    out['Information Ratio (AOK)'] = round(ir, 4)
    out['Tracking Error (AOK)'] = round(te, 4)
    out['Max. Drawdown'] = round(max_drawdown*100, 4)

    return out


def get_usd_prices(df):
    ret = df.pct_change()
    it_coupon = 0.04
    ch_coupon = 0.015
    rf_coupon = 0.045

    df['it'] = (1 + (it_coupon / 252 + ret.it + ret.EUR)).cumprod()*100
    df['ch'] = (1 + (ch_coupon / 252 + ret.ch + ret.CHF)).cumprod()*100
    df['rf'] = (1 + (rf_coupon / 252 + ret.rf)).cumprod()*100
    df['CSSMI.SW'] = df['CSSMI.SW'] + df.CHF
    return df


def get_max_drawdown(cumulative_returns: pd.DataFrame) -> pd.DataFrame:
    highwatermarks = cumulative_returns.cummax()
    drawdowns = (1 + highwatermarks)/(1 + cumulative_returns) - 1
    return drawdowns


def create_portfolio_return_usd(df: pd.DataFrame, allocation):
    t_r = pd.DataFrame([df[i] for i in allocation.keys()]).T
    tmp_ret = t_r.dropna(how='all').to_numpy()
    port_ret = pd.DataFrame(tmp_ret @ list(allocation.values()), index=df.index, columns=['Portfolio'])
    for i in allocation.keys():
        port_ret[i] = df[i] * allocation[i]
    return port_ret


def create_portfolio_returns(df: pd.DataFrame, weights: dict, risk_aversion: float = 0.5) -> pd.DataFrame:
    """
    Function that takes prices of our portflio, weights the returns and calculates the portfolio returns. 

    :param df: (pd.DataFrame) dataframe containing the assets pre-defined by the paper traders team
    :param weights: (dict) This is a dictionary that contains the weights for each asset
    :param risk_aversion: (float) this value defines how much is invested into the risky asset. 

    :return: (pd.DataFrame) A dataframe containig the pct. returns of the risky portfolio as well as the normal one.    
    """
    capital = 100_000_000
    ret = df.pct_change()

    rf = ret.rf
    # We hold our bonds until maturity which is why we get the daily return for them and adjust for the FX 
    it_coupon = 0.04
    ch_coupon = 0.015
    rf_coupon = 0.045

    # ret['it'] = df.loc['2022-11-07'].it / 100 / 252 + ret.EUR 
    # ret['ch'] = df.loc['2022-11-07'].ch / 100 / 252 + ret.CHF
    ret['it'] = it_coupon / 252 + ret.it + ret.EUR
    ret['ch'] = ch_coupon / 252 + ret.ch + ret.CHF
    ret['CSSMI.SW'] = ret['CSSMI.SW'] + ret.CHF
    ret.drop(['rf', 'CHF', 'EUR'], axis=1, inplace=True)

    t_r = pd.DataFrame([ret[i] for i in weights.keys()]).T

    tmp_ret = t_r.dropna(how='all').to_numpy()
    port_ret = pd.DataFrame(tmp_ret @ list(weights.values()), index=ret.index[1:])

    # Create our final portfolio
    c_ret = pd.DataFrame(port_ret[0] * risk_aversion + (1-risk_aversion) * capital * (rf_coupon / 252 + rf)) # Todays risk free yield per day
    # print(c_ret)
    c_ret['ji'] = port_ret
    c_ret.columns = ['Portfolio', 'Risky Portfolio']
    return c_ret


def convert_treasury_price(price: str) -> float:
    """
    Function that takes US Treasury bond prices and converts them from these freedom units to human prices.

    :param price: (str) Price in Freedom units (32th)

    :return: (float) Price of a bond
    """
    t1 = price.split('-')
    p1 = float(t1[0])
    if len(t1) > 1: 
        t2 = t1[1].split(' ')
        if t2[0][-1] == '+':
            return p1 + (float(t1[1][:-1])+0.5)/32 
        elif len(t2) > 1:
            p2 = float(t2[0])
            t3 = t2[1].split('/')
            p3 = float(t3[0]) / float(t3[1])
            return p1 + (p2 + p3)/32
        else: 
            return p1 + float(t1[1])/32



def check_normality(returns):
    # Use Jarque Bera test to see if our returns are normal
    _, pvalue, _, _ = jarque_bera(returns)

    if pvalue > 0.05:
        print('The portfolio returns are likely normal.')
    else:
        print('The portfolio returns are likely not normal.')

    
def value_at_risk(returns: pd.DataFrame, alpha:float = 0.95, value_invested: int = 100000000, lookback_days: int = 252*3):
    """
    Calculates the value at risk over a certain lookback period. 

    :param returns: (pd.DataFrame) dataframe of returns
    :param alpha: (float) percentile for the cutoff
    :param value_invested: (int) amount of money at risk 
    :param lookback_days: (int) How many days of past data should be taken into account

    :return: (float) Value at Risk in Quote currency
    """
    returns = returns.iloc[-lookback_days:]
    return np.percentile(returns, 100 * (1-alpha)) * value_invested

def es(returns, alpha=0.95, value_invested=100000000, lookback_days=520*3):
    """
    Calculates the conditional value at risk / expected shortfall over a certain lookback period. 

    :param returns: (pd.DataFrame) dataframe of returns
    :param alpha: (float) percentile for the cutoff
    :param value_invested: (int) amount of money at risk 
    :param lookback_days: (int) How many days of past data should be taken into account

    :return: (float) Value at Risk in Quote currency
    """
    var = value_at_risk(returns, alpha, value_invested, lookback_days)
    returns = returns.iloc[-lookback_days:]
    var_pct_loss = var / value_invested
    
    return value_invested * np.nanmean(returns[returns < var_pct_loss])
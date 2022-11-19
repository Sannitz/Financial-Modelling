import pandas as pd
import numpy as np
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt
from eod import EodHistoricalData
from datetime import timedelta


API_KEY = '' # API Keys for EOD data. Please contact the team if necessary
client = EodHistoricalData(API_KEY)


def get_bloomberg_data(full=False):
    df = pd.DataFrame()
    if full: names = ['dbc', 'ch', 'it', 'rf', 'shel', 'smi', 'wmt']
    else: names = ['ch', 'it', 'rf']

    for n in names:
        tmp = pd.read_excel(f'C:/Users/gobel/code/research_env/USI/financial_modelling/data/{n}.xlsx')
        tmp.set_index(pd.to_datetime(tmp.Date), inplace=True)
        tmp.columns = ['a', 'b']
        df[n] = tmp.b

    if full: df.columns = ['DBC.US','ch', 'it', 'rf', 'SHEL', 'CSSMI.SW', 'WMT']
    else: df.columns = names
    return df


def get_updated_data():
    df = pd.DataFrame()

    for i in ['DBC.US', 'SHEL', 'CSSMI.SW', 'WMT']:
        tmp = pd.DataFrame(client.get_prices_eod(i, period='d', order='a', from_='2010-01-01'))
        tmp.set_index(pd.to_datetime(tmp.date), inplace=True)
        df[i] = tmp.adjusted_close
    return df


def add_fx(df: pd.DataFrame) -> pd.DataFrame:
    eur = pd.DataFrame(client.get_prices_eod('EURUSD.FOREX', period='d', order='a', from_='2010-01-01'))
    eur.set_index(pd.to_datetime(eur.date), inplace=True)
    df['EUR'] = eur.adjusted_close

    chf = pd.DataFrame(client.get_prices_eod('CHFUSD.FOREX', period='d', order='a', from_='2010-01-01'))
    chf.set_index(pd.to_datetime(chf.date), inplace=True)
    df['CHF'] = chf.adjusted_close
    return df


def portfolio_return(returns, start, end, plot=True):
    names = ['Portfolio', 'Risky Portfolio', 'DBC.US', 'SHEL', 'CSSMI.SW', 'WMT', 'EUR', 'CHF']
    df = returns[start:end][names]
    
    if plot:
        fig, ax = plt.subplots(figsize=(15,7), facecolor='white')
        c_ret = (1+df).cumprod()
        c_ret.iloc[0] = 1.0
        
        for column in c_ret.drop('Portfolio', axis=1):
            ax.plot(c_ret[column], marker='', linewidth=1, alpha=0.6)

        ax.plot(c_ret['Portfolio'], marker='', color='royalblue', linewidth=4, alpha=0.9)
        # ax.plot(c_ret, label = df.columns)
        # ax.set_xlim(0,12)
        num=0
        for i in c_ret.values[-1]:
            name=list(c_ret)[num]
            num+=1
            if name != 'Portfolio':
                ax.text(c_ret.index[-1] + timedelta(hours=2), i, name, horizontalalignment='left', size='small', color='grey')
        ax.set_xlim(c_ret.index[0], c_ret.index[-1] + timedelta(days=2))
        ax.text(c_ret.index[-1] + timedelta(hours=2), c_ret.Portfolio.tail(1), 'Portfolio', horizontalalignment='left', size='small', color='royalblue')
        # ax.legend()
        ax.set_title('Returns per asset')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        # plt.show()

    print(f'Returns: {round((c_ret.Portfolio[-1] - 1)*100, 4)}%\nDaily avg. portfolio return: {round(df.Portfolio.mean()*100, 4)}% \nDaily standard deviation: {round(df.Portfolio.std()*100, 4)}%')


def create_portfolio_returns(df: pd.DataFrame, weights: dict, risk_aversion: float = 0.5):
    ret = df.pct_change()
    # We hold our bonds until maturity which is why we get the daily return for them and adjust for the FX 
    ret['it'] = df.loc['2022-11-07'].it / 100 / 252 + ret.EUR 
    ret['ch'] = df.loc['2022-11-07'].ch / 100 / 252 + ret.CHF
    ret['CSSMI.SW'] = ret['CSSMI.SW'] + ret.CHF
    ret.drop(['rf', 'CHF', 'EUR'], axis=1, inplace=True)

    t_r = pd.DataFrame([ret[i] for i in weights.keys()]).T

    tmp_ret = t_r.dropna(how='all').to_numpy()
    port_ret = pd.DataFrame(tmp_ret @ list(weights.values()), index=ret.index[1:])

    # Create our final portfolio
    c_ret = port_ret * risk_aversion + (1-risk_aversion) * df.loc['2022-11-07'].rf / 100 / 252 # Todays risk free yield per day
    c_ret['ji'] = port_ret
    c_ret.columns = ['Portfolio', 'Risky Portfolio']
    return c_ret



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
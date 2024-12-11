# %%
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# %%
def get_asset_df(asset_name):
    asset_map = {
        'jpy': 'JPY%3DX',
        'eur': 'EURUSD%3DX',
        'gbp': 'GBPUSD%3DX',
        'franc': 'CHFUSD%3DX',
        'us100': 'NQ%3DF',
        'us30': 'YM%3DF',
        'us500': 'ES%3DF',
        'crude oil': 'CL%3DF',
        'gold': 'GC%3DF',
        'bitcoin': 'BTC-USD',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'NVIDIA': 'NVDA',
        'Berkshire': 'BRK-B',
        'Meta': 'META',
        'Tesla': 'TSLA',
        'UGI': 'UNH',
        'Exxon': 'XOM',
        'Johnson&Johnson': 'JNJ',
        'Visa': 'V',
        'Procter&Gamble': 'PG',
        'JPMorgan': 'JPM',
        'Mastercard': 'MA',
        'EliLilly': 'LLY',
        'Chevron': 'CVX',
        'HomeDepot': 'HD',
        'Pfizer': 'PFE',
        'AbbVie': 'ABBV',
        'Merck': 'MRK',
        'Pepsi.': 'PEP',
        'Coca-Cola': 'KO',
        'Broadcom': 'AVGO',
        'Costco': 'COST',
        'WaltDisney': 'DIS',
        'Comcast': 'CMCSA',
        'Cisco': 'CSCO',
        'Intel': 'INTC',
        'Salesforce': 'CRM',
        'TexasInstruments ': 'TXN',
        'ThermoFisher ': 'TMO',
        'Bristol-Myers Squibb': 'BMY',
        'Verizon': 'VZ',
        'Nike': 'NKE',
        'Oracle': 'ORCL',
        'NextEra Energy': 'NEE',
        'McDonald ': 'MCD',
        'Adobe': 'ADBE',
        'PhilipMorris ': 'PM',
        'AT&T': 'T',
        'AbbottLaboratories': 'ABT',
        'AdvancedMicro': 'AMD',
        'UnionPacific': 'UNP',
        'Medtronic': 'MDT',
        'CVSHealth': 'CVS',
        'Qualcomm': 'QCOM',
        'Honeywell': 'HON',
        'GoldmanSachs ': 'GS'
    }
    
    if asset_name not in asset_map:
        raise ValueError(f"Asset '{asset_name}' not recognized. Available options are: {', '.join(asset_map.keys())}")
    
    ticker = asset_map[asset_name]
    
    df = yf.download(ticker, start='2000-01-01', end=datetime.now()+timedelta(days=1), interval='1d')
    df.columns = df.columns.str.lower()
    columns_to_drop = ['stock splits', 'dividends']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
    df = df.reset_index(level=0)
    df.columns = df.columns.str.lower()
    df['date'] = df['date'].dt.tz_localize('UTC')  
    df['date'] = df['date'].dt.tz_convert(None)
    
    return df

def adx(asset_name, length):
    df = get_asset_df(asset_name)
    df.ta.adx(length=length,append=True)
    df=df.iloc[-100:]

    ADX=f'ADX_{length}'
    DMP=f'DMP_{length}'
    DMN=f'DMN_{length}'

    plt.figure(figsize=(14, 14))
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df[ADX], label=ADX, color='purple')
    plt.plot(df.index, df[DMP], label=DMP, color='green')
    plt.plot(df.index, df[DMN], label=DMN, color='red')
    plt.axhline(y=20, color='black', linestyle='--', label='ADX Threshold')
    plt.title('Average Directional Index (ADX), +DI, and -DI')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    print(df[['close',ADX,DMP,DMN]].tail(1))
    if df[ADX].tail(1).iloc[0]>20 and df[DMP].tail(1).iloc[0]>df[DMN].tail(1).iloc[0]:
        print('trend is strong and upper trend')
    elif df[ADX].tail(1).iloc[0]>20 and df[DMP].tail(1).iloc[0]<df[DMN].tail(1).iloc[0]:
        print('trend is strong and lower trend')
    else:
        print('trend is weak')

def rsi(asset_name, length):
    df = get_asset_df(asset_name)
    df.ta.rsi(length=length, append=True)
    df = df.iloc[-100:]  # Keep only the last 100 records
    RSI = f'RSI_{length}'

    # Plotting RSI
    plt.figure(figsize=(14,14))
    
    plt.subplot(2,1,2)
    plt.plot(df.index,df[RSI],label='RSI',color='black')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(40, color='red', linestyle='--', label='Oversold (40)')
    plt.title('RSI indicator')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    print(df[['close',RSI]].tail(1))
    if df[RSI].tail(1).iloc[0]>70:
        print('overbought')
    elif df[RSI].tail(1).iloc[0]<40:
        print('oversold')
    else:
        print('normal')

def bbands(asset_name, length):
    df = get_asset_df(asset_name)
    df.ta.bbands(length=length,append=True)
    df=df.iloc[-150:]
    std_dev=2.0

    lower_band = f'BBL_{length}_{std_dev}'
    middle_band = f'BBM_{length}_{std_dev}'
    upper_band = f'BBU_{length}_{std_dev}'

    plt.figure(figsize=(14, 14))
    plt.plot(df['close'], label='Price', color='black')
    plt.plot(df[lower_band], label="BBANDS Lower", color='red',linestyle='--')
    plt.plot(df[middle_band], label="BBANDS Middle", linestyle='--')
    plt.plot(df[upper_band], label="BBANDS Upper", color='red',linestyle='--')

    plt.title(f'Bollinger Bands (Length: {length}, Std Dev: {std_dev})')
    plt.legend()
    plt.show()



def stoc(asset_name, length):
    df = get_asset_df(asset_name)
    df.ta.stoch(k=length, d=3, smooth_k=3, append=True)
    df=df.iloc[-150:]
    STOCHk=f'STOCHk_{length}_3_3'
    STOCHd=f'STOCHd_{length}_3_3'

    plt.figure(figsize=(14, 14))  
    
    plt.subplot(2, 1, 2)  # Second subplot
    plt.plot(df.index, df[STOCHk], label='%K', color='blue')
    plt.plot(df.index, df[STOCHd], label='%D', color='green')
    plt.axhline(80, color='red', linestyle='--')
    plt.axhline(20, color='red', linestyle='--')
    plt.title('Stochastic Oscillator')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    
    print((df[['close',STOCHk,STOCHd]].tail(1)))
    
    if df[STOCHk].tail(1).iloc[0]>80 and df[STOCHd].tail(1).iloc[0]>80:
        print('overbought')
    elif df[STOCHk].tail(1).iloc[0]<20 and df[STOCHd].tail(1).iloc[0]<20:
        print('oversold')
    else:
        print('normal')


def ema(asset_name,length1, length2):
    df = get_asset_df(asset_name)
    df=df.iloc[-250:]

    df[f'ema_{length1}'] = df['close'].ewm(span=length1, adjust=False).mean()
    df[f'ema_{length2}'] = df['close'].ewm(span=length2, adjust=False).mean()

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['close'], label='Price', color='black', linewidth=1)
    plt.plot(df[f'ema_{length1}'], label=f'EMA {length1}', color='blue', linewidth=1.5)
    plt.plot(df[f'ema_{length2}'], label=f'EMA {length2}', color='green', linewidth=1.5)

    plt.title(f'Exponential Moving Averages (EMA {length1} and EMA {length2})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def sma(asset_name,length1, length2):
    df = get_asset_df(asset_name)
    df=df.iloc[-250:]

    df[f'sma_{length1}'] = df['close'].rolling(window=length1).mean()
    df[f'sma_{length2}'] = df['close'].rolling(window=length2).mean()

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['close'], label='Price', color='black', linewidth=1)
    plt.plot(df[f'sma_{length1}'], label=f'SMA {length1}', color='blue', linewidth=1.5)
    plt.plot(df[f'sma_{length2}'], label=f'SMA {length2}', color='green', linewidth=1.5)

    plt.title(f'Simple Moving Averages (SMA {length1} and SMA {length2})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def lwma(asset_name,length1, length2):
    df = get_asset_df(asset_name)
    df=df.iloc[-250:]
    def lwma_(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(window=length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    
    df[f'lwma_{length1}'] = lwma_(df['close'], length1)
    df[f'lwma_{length2}'] = lwma_(df['close'], length2)

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['close'], label='Price', color='black', linewidth=1)
    plt.plot(df[f'lwma_{length1}'], label=f'LWMA {length1}', color='blue', linewidth=1.5)
    plt.plot(df[f'lwma_{length2}'], label=f'LWMA {length2}', color='green', linewidth=1.5)

    plt.title(f'Linear Weighted Moving Averages (LWMA {length1} and LWMA {length2})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def parabolic_sar(asset_name):
    df = get_asset_df(asset_name)

    df.ta.psar(append=True)
    df=df.iloc[-100:]

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(df['close'], label='Price', color='black', linewidth=1)
    plt.scatter(df.index, df['PSARl_0.02_0.2'], label='Parabolic SAR', color='blue', marker='.')

    plt.title('Parabolic SAR Indicator')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def macd(asset_name):
    df = get_asset_df(asset_name)
    df.ta.macd(append=True)
    df=df.iloc[-200:]

    plt.figure(figsize=(14,14))
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['MACD_12_26_9'], label='MACD', color='red')
    plt.plot(df.index, df['MACDs_12_26_9'], label='Signal', color='green')
    plt.bar(df.index, df['MACDh_12_26_9'], label='Histogram', color='gray')
    plt.legend()
    
    plt.show()
    
    print(df[['close','MACD_12_26_9','MACDs_12_26_9']].tail(1))
    if df['MACD_12_26_9'].tail(1).iloc[0]>0 and df['MACD_12_26_9'].tail(1).iloc[0]>df['MACDs_12_26_9'].tail(1).iloc[0]:
        print('bullish trend')
    elif df['MACD_12_26_9'].tail(1).iloc[0]<0 and df['MACD_12_26_9'].tail(1).iloc[0]<df['MACDs_12_26_9'].tail(1).iloc[0]:
        print('bearish trend')
    else:
        print('consolidation')


def atr(asset_name,length):
    df = get_asset_df(asset_name)
    df.ta.atr(length=length,append=True)
    df=df.iloc[-100:]
    plt.figure(figsize=(14, 8))
    

    
    plt.subplot(2, 1, 2)
    plt.plot(df['ATRr_' + str(length)], label='ATR', color='orange')
    plt.title(f'Average True Range (ATR) - Length: {length}')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()



# %%
# adx('Apple',21)

# %%




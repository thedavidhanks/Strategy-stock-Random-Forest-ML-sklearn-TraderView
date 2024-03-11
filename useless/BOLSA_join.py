import glob
import os
from datetime import date, timedelta
from datetime import datetime
import pandas as pd


import yfinance as yf
from yahoo_fin.stock_info import get_data, tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table
""" get list of all stocks currently traded
    on NASDAQ exchange """
nasdaq_ticker_list = tickers_nasdaq()

""" get list of all stocks currently in the S&P 500 """
sp500_ticker_list = tickers_sp500()

""" get other tickers not in NASDAQ (based off nasdaq.com)"""
other_tickers = tickers_other()

""" get information on stock from quote page """
info = get_quote_table("amzn")

tickers = yf.Tickers('msft aapl goog TWLO')
yf.tickers_nasdaq()
# access each ticker using (example)
tickers.tickers['TWLO'].info


list_files = glob.glob("d_price/TRAVIEW_stra/_data_stra_*.csv")
# for f in list_files:
#     print(f)
#     df_dol = pd.read_csv(f, index_col=None, sep='\t')

df = pd.DataFrame()
for f in list_files:
    print(f)
    df_w = pd.read_csv(f,  sep='\t')
    df = pd.concat([df, df_w], ignore_index=True)

print(df)

df_S_all = df.sort_values(['ticker'], ascending=True)
df_S_all = df_S_all.drop_duplicates(subset=['ticker'], keep="last")
df_S_all = df_S_all.dropna(how='any')
df_S_all.reset_index(drop=True, inplace=True)
df_S_all = df_S_all.drop(columns=[df_S_all.columns[0]])
df_S_all.to_csv("d_price/TRAVIEW_stra/_full_data_stra_.csv",sep="\t", index=None)
import requests
import pandas as pd
import datetime as dt
import time
import json
import yfinance as yf
from datetime import datetime

import _KEYS_DICT
START_DATE = '2000-01-01T00:00:00Z'
END_DATE =  datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ')
TARGET_TIME = '1Day'#'2024-02-23T23:59:59Z'
INTERVAL_WEBULL = "m3"  # ""d1" #y1 => diario y5=> semanal  m3=> diario     d5 => 5 minutos     d1 => 1 minuto


def get_yahoo_api_data(TICKER, period="max", interval="1d"):
    # df_webull, __ = get_df_webull_realTime(INTERVAL_WEBULL, TICKER, None)
    df_yh = yf.download(tickers=TICKER, period="max", interval="1d", prepos=False)
    # df_yh.index = df_yh.index.tz_convert(None)  # location zone adapt to current zone
    df_yh.reset_index(inplace=True)
    df_yh = df_yh.rename(columns={'Datetime': 'Date'})
    df_yh = df_yh.drop(columns=['Adj Close'])
    df_yh['Date'] = df_yh['Date'] + pd.Timedelta(hours=5)
    df_yh['Date'] = df_yh['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_yh = df_yh.sort_values('Date', ascending=False).round(2)
    df_yh = df_yh[df_yh['Date'] > "2014:01:01"]
    df_yh['Date'] = pd.to_datetime(df_yh['Date'])
    return df_yh


# Set your parameters
# symbol = 'AAPL'  # Replace with the symbol you're interested in
START_DATE = '2000-01-01T00:00:00Z'
END_DATE = '2024-02-23T23:59:59Z'
CSV_NAME = "@FAV"
stocks_list = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
stocks_list = stocks_list + ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
stocks_list = stocks_list +[  "LYFT", "ADBE", "UBER", "ZI", "QCOM",  "SPOT", "NVDA", "PTON","CRWD", "NVST", "HUBS", "EPAM",  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD", "CRSR","AMZN","AMD" , "ADSK",  ]
stocks_list = stocks_list +  [ "U", "DDOG", "MELI", "TWLO", "UBER", "GTLB", "RIVN",    "PYPL", "GTLB", "MDB", "TSLA", "UPST"]
stocks_list = stocks_list + _KEYS_DICT.DICT_COMPANYS["@FOLO1"] +_KEYS_DICT.DICT_COMPANYS["@FOLO2"]+_KEYS_DICT.DICT_COMPANYS["@FOLO3"]

stocks_list = _KEYS_DICT.DICT_COMPANYS["@FULL_ALL"] # set(stocks_list)

stocks_list = ["AAPL", "MELI"]
for symbol in stocks_list:
    # Fetch the data
    print("Starting data fetching process Stock: ", symbol)
    try:
        df = get_yahoo_api_data(symbol)
        df = df.sort_values(['Date'], ascending=True)
        print("Data fetching process completed df.shape: ", df.shape)

        # Save the data as a CSV file
        if df is not None:
            df.index = df['Date']  #.to_pydatetime()
            # df.to_csv(f"data/alpa_{symbol}_1min.csv",sep="\t")
            # print(f"Data saved as ", f"data/alpa_{symbol}_1min.csv")
            max_recent_date = df.index.max().strftime("%Y%m%d")   # pd.to_datetime().strftime("%Y%m%d")
            min_recent_date = df.index.min().strftime("%Y%m%d")
            print("d_price/yahoo/yahoo_" + symbol + '_' + TARGET_TIME + "_" + max_recent_date + "__" + min_recent_date + ".csv   df.shape: ", df.shape)
            # df.to_csv("d_price/alpaca/alpaca_" + symbol + '_' + '5Min' + "_" + max_recent_date + "__" + min_recent_date + ".csv",sep="\t", index=None)
            df.to_csv("d_price/yahoo/yahoo_" + symbol + '_' + TARGET_TIME + "_.csv",sep="\t", index=None)
            print("\tSTART: ", str(df.index.min()),  "  END: ", str(df.index.max()) , " shape: ", df.shape, "\n")
        else:
            print ("error none in stock: ", symbol)
    except Exception as ex:
        print("Error  get_bars", ex)
        continue


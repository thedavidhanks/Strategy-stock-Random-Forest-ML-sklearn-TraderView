from datetime import datetime

from UtilsL import get_yahoo_api_data
import _KEYS_DICT # when needed.

START_DATE = '2000-01-01T00:00:00Z'
END_DATE =  datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ')
TARGET_TIME = '1Day'#'2024-02-23T23:59:59Z'
INTERVAL_WEBULL = "m3"  # ""d1" #y1 => diario y5=> semanal  m3=> diario     d5 => 5 minutos     d1 => 1 minuto

# Set your parameters
# symbol = 'AAPL'  # Replace with the symbol you're interested in
START_DATE = '2000-01-01T00:00:00Z'
END_DATE = '2024-02-23T23:59:59Z'

# Set the list of stocks
# presets in _KEYS_DICT can be used to get the list of stocks from the CSV files
# For example 
# CSV_NAME = "@FAV"
# stocks_list = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
# stocks_list = stocks_list + ["SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
# stocks_list = stocks_list +[  "LYFT", "ADBE", "UBER", "ZI", "QCOM",  "SPOT", "NVDA", "PTON","CRWD", "NVST", "HUBS", "EPAM",  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD", "CRSR","AMZN","AMD" , "ADSK",  ]
# stocks_list = stocks_list +  [ "U", "DDOG", "MELI", "TWLO", "UBER", "GTLB", "RIVN",    "PYPL", "GTLB", "MDB", "TSLA", "UPST"]
# stocks_list = stocks_list + _KEYS_DICT.DICT_COMPANYS["@FOLO1"] +_KEYS_DICT.DICT_COMPANYS["@FOLO2"]+_KEYS_DICT.DICT_COMPANYS["@FOLO3"]
# # -- OR --
# stocks_list = _KEYS_DICT.DICT_COMPANYS["@FULL_ALL"] # set(stocks_list)

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
            df.index = df['Date']
            max_recent_date = df.index.max().strftime("%Y%m%d")
            min_recent_date = df.index.min().strftime("%Y%m%d")
            print("d_price/yahoo/yahoo_" + symbol + '_' + TARGET_TIME + "_" + max_recent_date + "__" + min_recent_date + ".csv   df.shape: ", df.shape)
            df.to_csv("d_price/yahoo/yahoo_" + symbol + '_' + TARGET_TIME + "_.csv",sep="\t", index=None)
            print("\tSTART: ", str(df.index.min()),  "  END: ", str(df.index.max()) , " shape: ", df.shape, "\n")
        else:
            print ("error none in stock: ", symbol)
    except Exception as ex:
        print("Error  get_bars", ex)
        continue


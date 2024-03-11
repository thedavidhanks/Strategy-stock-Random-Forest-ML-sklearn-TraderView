import glob
import os
from datetime import date, timedelta
from datetime import datetime
import pandas as pd
import yfinance as yf
from  Strategy_TW_class import Strategy,OPEN,CLOSE,LONG,SHORT
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import _KEYS_DICT

from UtilsL import bcolors
# from predict_POOL_handle import get_tech_data_nasq, get_df_webull_realTime, df_yhoo_, merge_dataframes_bull_yhoo
#
# import API_alpaca_historical
# from BOLSA_zone_blaid_tools import *
from technical_parameters_konk_tools_Prepro import *

list_files = glob.glob("d_price\pine_tree\model_info_*_.csv")

# from Utils import Utils_send_message
# from Utils.Plot_scrent_shot import get_traderview_screem_shot

# url_trader_view = Utils_send_message.get_traderview_url("WH")
# path_imgs_tech, path_imgs_finan = get_traderview_screem_shot(url_trader_view, _KEYS_DICT.PATH_PNG_TRADER_VIEW + "" + "WH", will_stadistic_png=False)


df_all = pd.read_csv("d_price/RF/aa_RF_full_eval.csv",index_col=None, sep='\t')
df_TW = pd.read_csv("d_price/TRAVIEW_stra/_full_data_stra_.csv",index_col=None, sep='\t')
# df_dol[df_dol['ticker'] == TICKER]["dolars_avg "]
# Index(['ticker', 'path', 'path_candle', 'path_stra', 'Net Profit',
#        'Net Profit_per', 'Total Closed Trades', 'Percent Profitable',
#        'Profit Factor', 'Max Drawdown', 'Max Drawdown_per', 'Avg Trade',
#        'Avg Trade_per', 'Avg # Bars in Trades'],
#       dtype='object')
df_TW['Avg Trade_per'] = pd.to_numeric(df_TW['Avg Trade_per'].map(lambda x: x.replace("%", "").replace('−', '-'   )))
df_TW['Percent Profitable'] = pd.to_numeric(df_TW['Percent Profitable'].map(lambda x: x.replace("%", "").replace('−', '-'   )))

list_files = glob.glob("d_price\pine_tree\model_info_*_.csv")
df = pd.DataFrame()
for f in list_files:
    # print(f)
    df_w = pd.read_csv(f,index_col=None, sep='\t')
    df_w['ticker'] = re.search(r'model_info_(\w*)_.csv', f).group(1)
    df = pd.concat([df, df_w], ignore_index=True)

dict_models = {}
TARGET_TIME = '1Day'
stocks_list = df['ticker'].unique()
START_DATE = '2000-01-01T00:00:00Z'
END_DATE =  datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ')
TARGET_TIME = '1Day'#'2024-02-23T23:59:59Z'
INTERVAL_WEBULL = "m3"  # ""d1" #y1 => diario y5=> semanal  m3=> diario     d5 => 5 minutos     d1 => 1 minuto


def get_yahoo_api_data(period="max", interval="1d"):
    # df_webull, __ = get_df_webull_realTime(INTERVAL_WEBULL, TICKER, None)
    df_yh = yf.download(tickers=TICKER, period="max", interval="1d", prepos=False)
    # df_yh.index = df_yh.index.tz_convert(None)  # location zone adapt to current zone
    df_yh.reset_index(inplace=True)
    df_yh = df_yh.rename(columns={'Datetime': 'Date'})
    df_yh = df_yh.drop(columns=['Adj Close'])
    df_yh['Date'] = df_yh['Date'] + pd.Timedelta(hours=5)
    df_yh['Date'] = df_yh['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_yh = df_yh.sort_values('Date', ascending=False).round(2)
    df_yh = df_yh[df_yh['Date'] > "2022:01:01"]
    df_yh['Date'] = pd.to_datetime(df_yh['Date'])
    return df_yh
def register_MULTI_in_zTelegram_Registers(df_r, path): # = "d_result/Konk_buy_"+datetime.now().strftime("%Y_%m_%d")+".csv"
    if os.path.isfile(path):
        df_r.to_csv(path, sep="\t", mode='a', header=False)
    else:
        df_r.to_csv(path, sep="\t")
        print("Created MULTI : " + path)
def log_get_pineScript(df_kon, row_best_model, rf_mod):
    print(bcolors.OKBLUE + "\n COMPRA " + TICKER + bcolors.ENDC)
    if df_kon is not None:
        print("Score: ", df_kon.iloc[-1]['predict'], "path_model: ", row_best_model["path_model"], "Perc_eval: ",round(row_best_model["Perc_eval"], 3))
    df_feature_importances = pd.DataFrame({'Columns': rf_mod.feature_names_in_, 'Importance': [x.round(4) for x in rf_mod.feature_importances_]})
    print("Importance:\n", df_feature_importances, "\n")
    exported_text, sas_text, py_text, code_TVW = export_code(rf_mod.estimators_[0], 0, list(rf_mod.feature_names_in_))
    name_pine_tree = "d_result/pine_TW/" + TICKER + "_d" + str(row_best_model['max_depth']) + "_q" + str(round(row_best_model['Perc_eval'], 3)) + "_id" + str(row_best_model["id_count"]) + "_pine.pine"
    print("\t", name_pine_tree)
    with open(name_pine_tree, "w") as text_file:
        text_file.write(STRING_START_STRATEGY.replace("{TICKER}", TICKER).replace("xxxxx", "q" + str(
            row_best_model['final_score'])) + code_TVW + STRING_END_STRATEGY)
    print(" path: ", name_pine_tree)
    return name_pine_tree

# stocks_list = ['DNLI']
for TICKER in stocks_list:
    print("stock: ", TICKER)
    df_s = df_all[df_all['ticker'] == TICKER]
    df_s = df_s.sort_values(['final_score'], ascending=False)
    df_TW_sto = df_TW[df_TW['ticker'] == TICKER]

    if len(df_s) == 0 :
        print(bcolors.BOLD + "WARN There are no Models. accion: "+TICKER+ bcolors.ENDC)
        continue
    # if df_dol[df_dol['ticker'] == TICKER]["dolars_avg"].iloc[0] <= 1:
    #     print(bcolors.BOLD + "WARN Menos de un dolar por accion de rentabilidad. accion: "+TICKER+ " dol: "+str(df_dol[df_dol['ticker'] == TICKER]["dolars_avg"].iloc[0])+ bcolors.ENDC)
    #     continue
    if df_s.iloc[0]['final_score'] <= 0.190 :
        print(bcolors.BOLD + "WARN There are no Models. stock: "+TICKER+ " Final_score: "+str(df_s.iloc[0]['final_score']) + bcolors.ENDC)
        continue
    if len(df_TW_sto) > 0 and (df_TW['Avg Trade_per'].iloc[0] <= 1 or df_TW['Percent Profitable'].iloc[0] <= 50):
        print(bcolors.BOLD + "WARN There are no models with Level TW. stock: "+TICKER+ " Avg Trade_per: "+str(df_TW['Avg Trade_per'].iloc[0] ) + " Percent Profitable: "+str(df_TW['Percent Profitable'].iloc[0]) + bcolors.ENDC)
        continue
    row_best_model = df_s.iloc[0]
    rf_mod, dict_model = pickle.load(open(row_best_model["path_model"], 'rb'))
    name_pine_tree = log_get_pineScript(None, row_best_model, rf_mod)
    if dict_model['Perc'] >= 0.40 :
        print(bcolors.BOLD + "WARN very high Precision  (riks of overfit). stock: "+TICKER+ " Final_score: "+str(dict_model['Perc']) + bcolors.ENDC)
        continue
    path_read_csv = "d_price/yahoo/yahoo_" + TICKER + '_' + TARGET_TIME + "_.csv"
    df_alpa = pd.read_csv(path_read_csv, sep="\t")
    df_yh = get_yahoo_api_data()

    df_yh['Date'] = pd.to_datetime(df_yh['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_alpa['Date'] = pd.to_datetime(df_alpa['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_alpa = df_alpa.drop( df_alpa[ df_alpa['Date'].isin( df_yh['Date']  ) ].index ) # quito las ultimas para poner las de Yahoo
    df_bl = pd.concat([df_alpa, df_yh]).sort_values('Date', ascending=True)
    df_bl = df_bl.drop_duplicates(subset=['Date'],keep="first")
    if df_bl.iloc[0]['Volume'] == 0:
        print("El volumen para Yahoo API(alpa) es 0. Stock: " + TICKER)

    print("Read csv Path: ", path_read_csv, " Shape: ", df_alpa.shape)
    _, df_kon_predict = preprocess_df_to_predict_with_konkorde_blaid(df_bl,path_read_csv,  TICKER, TARGET_TIME)
    df_kon = df_kon_predict[:][rf_mod.feature_names_in_]
    if len(df_kon) <= 0:
        print(bcolors.BOLD + "WARN failure to obtain the average in preprocess_df_to_predict_with_konkorde_blaid: "+TICKER+ " Final_score: "+str(dict_model['Perc']) + bcolors.ENDC)
        continue
    df_kon['predict'] = np.reshape(rf_mod.predict(df_kon), (-1, 1))
    df_kon_predict['predict'] = df_kon['predict']


    DAY_EVAL = -1 # -1 si no esta el mercado abierto , antes de 15:30 , despues -2
    DICT_STRATEGY = {}
    for index, row in df_kon_predict[-9:].iterrows():
        # print("index")
        op_operation = row['predict']
        close = row['Close']
        Position_avg_price = row['Open']  # (row['Open'] +row['High']+row['Low']+row['Close']) /4
        if (op_operation <= 1.0):
            print("Update ", row['Date'])
            for k in DICT_STRATEGY.keys():
                if DICT_STRATEGY[k].type_open_close == OPEN:
                    DICT_STRATEGY[k].update(row_update_data=row)
                    DICT_STRATEGY[k].check_stoploss_take_profit(row_update_data=row)

        if (op_operation >= 1.68):  # // buy
            if row['Date'].strftime('%Y-%m-%d') == (date.today() - timedelta(days=1)).strftime('%Y-%m-%d'):
                print(row['Date'].strftime('%Y-%m-%d'))
            stop = row['Close'] * 0.965  #:=
            limit1 = row['Close'] * 1.03  #:=
            limit2 = row['Close'] * 1.02  #:=
            # row_update_data, name, type_long_short, units, stop = None, limit = None, comment = ""
            key_id = TICKER + "_" + row['Date'].strftime("%Y%m%d")


            DICT_STRATEGY[key_id] = Strategy(TICKER,op_operation, row_update_data=row, name="x", type_long_short=LONG, units=1,stop=stop, limit=None, comment="start")
            print("Created key: "+key_id + "\t"+TICKER, "Enter ", row['Date'], row['Open'], " Per:", op_operation)
        if (op_operation <= 0.1):  # // sell
            for k in DICT_STRATEGY.keys():
                if DICT_STRATEGY[k].type_open_close == OPEN:
                    DICT_STRATEGY[k].close(row_update_data=row)
        # for k in DICT_STRATEGY.keys():
        #     if DICT_STRATEGY[k].type_open_close == OPEN:
        #         DICT_STRATEGY[k].check_stoploss_take_profit(row_update_data=row)
    per_sum = 0;
    sum_dolars = 0
    for k in DICT_STRATEGY.keys():
        text_strate = " Date_start: " + str(DICT_STRATEGY[k].Date_start) + " Earn: " + str(round(DICT_STRATEGY[k].PER_EARN, 3)) + "\tEnter: " + str(round(DICT_STRATEGY[k].PRICE_ENTER, 3)) + "\tExit: " + str(round(DICT_STRATEGY[k].PRICE_CLOSE, 3)) + "\tDate_end: " + str(DICT_STRATEGY[k].Date_end)
        per = DICT_STRATEGY[k].PER_EARN * 100 / DICT_STRATEGY[k].PRICE_ENTER
        per_sum = per_sum + per
        sum_dolars = sum_dolars + DICT_STRATEGY[k].Dolars_earn
    # (stocks_sell - stocks_buy)
    if len(DICT_STRATEGY) > 0:
        df_stra = pd.DataFrame( {"ticker": TICKER, "entres": len(DICT_STRATEGY.keys()), "sum_dolars": round(sum_dolars, 2), "per_sum": round(per_sum, 4),"per_sum_avg ": round(per_sum / len(DICT_STRATEGY.keys()), 2),"dolars_avg": round(per_sum / len(DICT_STRATEGY.keys()), 2),'final_score': row_best_model['final_score']}, index=[0])
        # register_MULTI_in_zTelegram_Registers(df_stra.round(3), "d_result/stre/" + TICKER + "_d" + str(round(per_sum / len(DICT_STRATEGY.keys()), 2)) + "_p" + str(round(per_sum / len(DICT_STRATEGY.keys()), 1)) + ".csv")
        register_MULTI_in_zTelegram_Registers(df_stra.round(3),"d_result/win_loss_today_" + datetime.now().strftime("%Y_%m_%d") + ".csv")
        print(TICKER, "end Total per : ", per_sum, " sum_dolars: ", sum_dolars)
    # if df_kon.iloc[DAY_EVAL]['predict'] > 1.68:
    #     df_f1 = pd.DataFrame({"Date": df_kon_predict.iloc[DAY_EVAL]['Date'], "Tikcer": TICKER, "Predict_score": round(df_kon.iloc[DAY_EVAL]['predict'],3),
    #                           "final_score": row_best_model['final_score'],"dolars_avg" : df_dol[df_dol['ticker'] == TICKER]["dolars_avg"].iloc[0],
    #                           "entres" : df_dol[df_dol['ticker'] == TICKER]["entres"].iloc[0], "model": row_best_model["path_model"],"path_pine": name_pine_tree}, index=[0])
    #     register_MULTI_in_zTelegram_Registers(df_f1,  path = "d_result/Konk_buy_"+datetime.now().strftime("%Y_%m_%d")+".csv")
    # print(" Score: ", df_kon.iloc[-1]['predict'], "path_model: " , row_best_model["path_model"],"Perc_eval: ", round(row_best_model["Perc_eval"], 3) )

print("end")
import glob
import os
from datetime import datetime
import pandas as pd
import yfinance as yf
import re
from  Strategy_TW_class import Strategy,OPEN,CLOSE,LONG,SHORT
from UtilsL import bcolors

from technical_parameters_konk_tools import *
from technical_parameters_konk_tools_Prepro import *
DOLARS_TO_OPERA = 100

list_files = glob.glob("d_price\pine_tree\model_info_*_.csv")


df_all = pd.read_csv("d_price/RF/aa_RF_full_eval.csv",index_col=None, sep='\t')
stocks_list = df_all['ticker'].unique()
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


df_all = pd.read_csv("d_price/RF/aa_RF_full_eval.csv",index_col=None, sep='\t')
stocks_list = df_all['ticker'].unique()
print(str(stocks_list))
for TICKER in stocks_list:
    print(bcolors.HEADER, "stock: ", TICKER, bcolors.ENDC )
    if os.path.isfile("d_result/stre/"+TICKER +".csv"):
        print("El fichero ya existe file: ", "d_result/stre/"+TICKER +".csv")
        continue
    df_s = df_all[df_all['ticker'] == TICKER]
    df_s = df_s.sort_values(['final_score'], ascending=False)
    if len(df_s) == 0:
        print(bcolors.HEADER,"No hay modelos con nivel para esta accion: ",TICKER, bcolors.ENDC )
        continue
    row_best_model = df_s.iloc[0]
    rf_mod, dict_model = pickle.load(open(row_best_model["path_model"], 'rb'))

    path_read_csv = "d_price/yahoo/yahoo_" + TICKER + '_' + TARGET_TIME + "_.csv"
    df_alpa = pd.read_csv(path_read_csv, sep="\t")
    df_yh = get_yahoo_api_data()

    df_yh['Date'] = pd.to_datetime(df_yh['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_alpa['Date'] = pd.to_datetime(df_alpa['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_alpa = df_alpa.drop( df_alpa[ df_alpa['Date'].isin( df_yh['Date'][:6] ) ].index ) # quito las ultimas para poner las de Yahoo
    # pd.conca t([df_alpa, df_yh]).sort_values('Date', ascending=False)
    df_bl = pd.concat([df_alpa, df_yh[:6]]).sort_values('Date', ascending=True)
    if len(df_bl) == 0 :
        print(bcolors.BOLD + "WARN No hay Modelos. accion: "+TICKER+ bcolors.ENDC)
        continue
    if df_bl.iloc[0]['Volume'] == 0:
        print("WARN EL volumen para Yahoo API(alpa)  es 0. Stock: " + TICKER)


    print("Read csv Path: ", path_read_csv, " Shape: ", df_alpa.shape)
    _, df_kon_predict = preprocess_df_to_predict_with_konkorde_blaid(df_bl,path_read_csv,  TICKER, TARGET_TIME)
    df_kon = df_kon_predict[rf_mod.feature_names_in_] # df_kon_predict[-5:][rf_mod.feature_names_in_]

    df_kon['predict'] = np.reshape(rf_mod.predict(df_kon), (-1, 1))
    df_kon_predict['predict'] = df_kon['predict']

    df_feature_importances = pd.DataFrame({'Columns': rf_mod.feature_names_in_, 'Importance': [x.round(4) for x in rf_mod.feature_importances_]})
    # print("Importance:\n", df_feature_importances, "\n")
    exported_text, sas_text, py_text, code_TVW = export_code(rf_mod.estimators_[0], 0, list(rf_mod.feature_names_in_))
    name_pine_tree = "d_price/pine_tree/" + TICKER + "_d" + str(row_best_model['max_depth']) + "_q" + str(round(row_best_model['Perc_eval'], 3)) + "_id" + str(row_best_model["id_count"]) + "_pine.java"
    print("\t", name_pine_tree)
    with open(name_pine_tree, "w") as text_file:
        text_file.write(STRING_START_STRATEGY.replace("{TICKER}", TICKER).replace("xxxxx", "q"+str(row_best_model['final_score']))       + code_TVW + STRING_END_STRATEGY)
    print( " path: ", name_pine_tree)

    DICT_STRATEGY = {}
    for index, row in df_kon_predict[-5:].iterrows():
        # print("index")
        op_operation = row['predict']
        close = row['Close']
        Position_avg_price = row['Open']# (row['Open'] +row['High']+row['Low']+row['Close']) /4
        if (op_operation <= 1.0):
            print("Update ", row['Date'])
            for k in  DICT_STRATEGY.keys():
                if DICT_STRATEGY[k].type_open_close == OPEN:
                    DICT_STRATEGY[k].update(row_update_data= row)
                    DICT_STRATEGY[k].check_stoploss_take_profit(row_update_data= row)

        if (op_operation >= 1.68):# // buy
            stop = row['Close'] * 0.965 #:=
            limit1 = row['Close'] * 1.03 #:=
            limit2 = row['Close'] * 1.02 #:=
            # row_update_data, name, type_long_short, units, stop = None, limit = None, comment = ""
            DICT_STRATEGY[ row['Date']] = Strategy(TICKER, per_score=op_operation, row_update_data= row, name="x", type_long_short=LONG, units=1, stop = stop, limit = None, comment ="start")
            print(TICKER ,"Enter ", row['Date'], row['Open'] , " Per:", op_operation)
            # DICT_STRATEGY[row['Date']].entry(row_update_data= row, name="x", type_long_short=Strategy.LONG, units=1, stop = stop, limit = None, comment = "")
        if (op_operation <= 0.1) :#// sell
            for k in DICT_STRATEGY.keys():
                if DICT_STRATEGY[k].type_open_close == OPEN:
                    DICT_STRATEGY[k].close(row_update_data=row)
            # Strategy.close(close_price=row['Close'], name="x", date=row['Date'], comment="")
            # DICT_STRATEGY[row['Date']].close(close_price= row['Close'], name="x", date=row['Date'], comment="")
        for k in DICT_STRATEGY.keys():
            if DICT_STRATEGY[k].type_open_close == OPEN:
                DICT_STRATEGY[k].check_stoploss_take_profit(row_update_data=row)
        # pd.DataFrame(DICT_STRATEGY)
    #FIN DE OPERAR
    per_sum = 0;sum_dolars  =0
    for k in DICT_STRATEGY.keys():
        text_strate = " Date_start: "+str(DICT_STRATEGY[k].Date_start)+ " Earn: "+str(round(DICT_STRATEGY[k].PER_EARN, 3)) + "\tEnter: "+str(round(DICT_STRATEGY[k].PRICE_ENTER,3)) + "\tExit: "+ str(round(DICT_STRATEGY[k].PRICE_CLOSE,3))+  "\tDate_end: "+str(DICT_STRATEGY[k].Date_end)
        per = DICT_STRATEGY[k].PER_EARN * 100 / DICT_STRATEGY[k].PRICE_ENTER
        per_sum = per_sum+per
        sum_dolars = sum_dolars + DICT_STRATEGY[k].Dolars_earn
 # (stocks_sell - stocks_buy)
    if len(DICT_STRATEGY) > 0:
        df_stra = pd.DataFrame({"ticker": TICKER, "entres": len(DICT_STRATEGY.keys()),"sum_dolars": round(sum_dolars,2),  "per_sum": round(per_sum,4),
                                "per_sum_avg ": round(per_sum / len(DICT_STRATEGY.keys()) , 2),"dolars_avg": round(per_sum / len(DICT_STRATEGY.keys()) , 2),
                                'final_score' : row_best_model['final_score'] }, index=[0])
        register_MULTI_in_zTelegram_Registers(df_stra.round(3), "d_result/stra_simulator/" + TICKER +"_d"+str( round(per_sum / len(DICT_STRATEGY.keys()) , 2) ) +"_p"+str(round(per_sum / len(DICT_STRATEGY.keys()),1)) +".csv")
        register_MULTI_in_zTelegram_Registers(df_stra.round(3), "d_result/win_loss_" + datetime.now().strftime("%Y_%m_%d") + ".csv")
        print(TICKER, "end Total per : ", per_sum , " sum_dolars: ", sum_dolars)
    # except Exception as ex:
    #     print("Exception: ", ex)

print("end")


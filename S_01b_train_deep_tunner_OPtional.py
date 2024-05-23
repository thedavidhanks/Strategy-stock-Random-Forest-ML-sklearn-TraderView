# https://es.tradingview.com/script/lLFiT3av/
# Blai5 Koncorde by manboto copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import dtreeviz
import itertools
import pandas as pd

# Importing libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from technical_parameters_konk_tools import *

print("dtreevis version: ",dtreeviz.__version__)
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import pandas_ta as ta
# from useless.pine_script_convert_funtions import nz
# https://es.tradingview.com/script/lLFiT3av/
# Blai5 Koncorde by manboto copy
import _KEYS_DICT

from sklearn.tree import _tree

#https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree?rq=4
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)) )

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold) )
            recurse(tree_.children_right[node], depth + 1)
        else:
            # print("{}return {}".format(indent, tree_.value[node]) )
            print("{}return {}".format(indent, np.argmax(tree_.value[node][0]))  +" # "+ str(tree_.value[node]) )

    recurse(0, 1)

DOLLARS_TO_BUY = 100
def rolling_buy_sell_val_BUY(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()

        list_value_sell = []
        # list_value_sell_debug = []
        for i in range(1, len(df_sub)):
            units_stocks_buy_value = 100 / df_sub['Close'][0]
            dol_selled = units_stocks_buy_value * df_sub['Close'][i]
            importance_c = len(df_sub) - i
            # if (i+1) % 2 == 0:
            list_value_sell += [dol_selled] * importance_c
            # list_value_sell_debug.append( (dol_selled.round(1), df_sub['Close'][i].round(1)) )

            per_change = get_per_change(dol_selled , 100)
            per_close_change = get_per_change( df_sub['Close'][i], df_sub['Close'][0])
            # if round(per_change, 8) != round(per_close_change, 8):
            #     print("aa")

            # assert round(per_change, 8) == round(per_close_change, 8)
        avg_end_dol = np.mean(list_value_sell)
        avg_end_per = get_per_change(avg_end_dol, df_sub['Close'][0])
        return avg_end_dol #, avg_end_per
        # print("EEE")

CSV_NAME = "@FAV"
TARGET_TIME = '1Day'
# TARGET_TIME = '5Min'
CSV_NAME = "@FAV"
stocks_list = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
stocks_list = stocks_list + ["SHOP", "RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
stocks_list = stocks_list +[  "LYFT", "ADBE", "UBER", "ZI", "QCOM",  "SPOT", "NVDA", "PTON","CRWD", "NVST", "HUBS", "EPAM",  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD", "CRSR","AMZN","AMD" , "ADSK",  ]
stocks_list = stocks_list +  [ "U", "DDOG", "MELI", "TWLO", "UBER", "GTLB", "RIVN",    "PYPL", "GTLB", "MDB", "TSLA", "UPST"]
# stocks_list = stocks_list + _KEYS_DICT.DICT_COMPANYS["@FOLO1"] +_KEYS_DICT.DICT_COMPANYS["@FOLO2"]+_KEYS_DICT.DICT_COMPANYS["@FOLO3"]

stocks_list = set(stocks_list)


stocks_list = ["AAPL", "MELI"]


for TICKER in stocks_list:
    path_read_csv = "d_price/alpaca/alpaca_" + TICKER + '_' + TARGET_TIME + "_.csv"
    df_bl = pd.read_csv(path_read_csv, sep="\t")
    print("Read csv Path: ", path_read_csv, " Shape: ", df_bl.shape)

    # PREPROCESS
    df_bl['Date'] = pd.to_datetime(df_bl['Date'])
    df_bl.index = df_bl['Date'].dt.strftime("%Y-%m-%d")
    max_recent_date = df_bl['Date'].max().strftime("%Y%m%d")  # pd.to_datetime().strftime("%Y%m%d")
    min_recent_date = df_bl['Date'].min().strftime("%Y%m%d")
    print("d_price/alpaca/alpaca_" + TICKER + '_' + TARGET_TIME + "_" + max_recent_date + "__" + min_recent_date + ".csv   df.shape: ", df_bl.shape)

    # GET YOUR TECH INDICATOR
    path_save_img = path_read_csv.replace(".csv", ".png").replace("/alpaca/", "/img_kond/")
    df_kon = get_konkorde_params_GOOD(df_bl, num_back_candles_img=80, path_save_img= path_save_img)

    Y_TARGET = 'avg_end_per'
    df_kon['azul_mean'] = df_kon['azul'].rolling(min_periods=1, window=5).mean()
    df_kon['verde_mean'] = df_kon['verde'].rolling(min_periods=1, window=5).mean()
    df_kon['marron_mean'] = df_kon['marron'].rolling(min_periods=1, window=5).mean()
    df_kon["verde_azul" ] = df_kon['verde'] - df_kon['azul']
    df_kon["verde_media"] = df_kon['verde'] - df_kon['media']
    df_kon["media_azul"] = df_kon['media'] - df_kon['azul']
    # df_kon["media_marron"] = df_kon['media'] - df_kon['marron']

    df_kon['avg_end_dol'] = df_kon.Close.rolling(min_periods=8, window=8).apply(rolling_buy_sell_val_BUY).shift(-7)
    df_kon['avg_end_perAux'] = percentage_change(100, df_kon["avg_end_dol"])
    cols_remove = ['tprice', 'pvi', 'nvi', 'source', 'pvi_ema', 'pvim', 'pvimax', 'pvimin', 'oscp',
                   'nvi_ema', 'nvim', 'nvimax', 'nvimin',   'xmf', 'BollOsc', 'xrsi', 'stoc','Open', 'High', 'Low', 'Close', 'Volume' ]
    df_kon = df_kon.drop(columns=cols_remove)
    # GET YOUR TECH INDICATOR END

    df_kon = df_kon.sort_values(['Date'], ascending=True)
    df_kon = df_kon.drop_duplicates(subset=['Date'],keep="first").dropna(how='any')
    df_kon.reset_index(drop=True, inplace=True)

    # GET THE GROUND TRUE
    df_kon[Y_TARGET] = -1; PER_VALEU_CHANGE = 1 # porcentage de cambio de compra o venta
    df_kon.loc[df_kon['avg_end_perAux'] > PER_VALEU_CHANGE, Y_TARGET] = 2#BUY
    df_kon.loc[df_kon['avg_end_perAux'] < -PER_VALEU_CHANGE, Y_TARGET] = 0 #SEEL
    df_kon.loc[(df_kon['avg_end_perAux'] < PER_VALEU_CHANGE) & (df_kon['avg_end_perAux'] > -PER_VALEU_CHANGE), Y_TARGET] = 1 #NONE

    print("Y_TARGET count() ",  df_kon.groupby(Y_TARGET)['Date'].count() )
    df_kon[Y_TARGET].describe()
    X = df_kon.drop(columns= [ Y_TARGET, 'avg_end_dol','avg_end_perAux', 'Date'])  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    y = df_kon[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
    # GET THE GROUND TRUE END

    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state = 42)
    print('\ndf', df_kon.shape, '\t', 'X_train', X_train.shape, '\t','y_train', y_train.shape, '\n',
          'X_test', X_test.shape, '\t','y_test', y_test.shape, '\t')
    X_xxx = np.asarray(X_train); y_xxx= np.asarray(y_train)

    # Create TUNNER of the model, to know which option is the correct configuration.
    dict_paramGrid_rf = {"n_estimators": [1,2,4,6,8,10],
                    "max_features": [None, 0.4,  0.5, 0.6, 0.7, "sqrt", "log2"],
                    "max_depth": [int(x) for x in np.linspace(10, 45, 5)] + [None],
                    "min_samples_split": [1, 2, 5, 10],
                    "min_samples_leaf": [1, 2, 4,6,9,16,21],
                    "bootstrap": [True, False],
                    'criterion': ['poisson', 'friedman_mse', 'absolute_error', 'squared_error']}
    # dict_paramGrid_rf = {"n_estimators": [1],
    #                 "max_features": [ 0.6, 0.625, 0.65, 0.675, 0.7] ,#[  0.5, 0.6, 0.7],0.4,  0.5,  , "sqrt", "log
    #                 "max_depth": [4,5],
    #                 "min_samples_split": [2, 6,10,16,22],
    #                 "min_samples_leaf": [1,2,5, 9,13 ],
    #                 "max_leaf_nodes": [122,28,32],
    #                 "bootstrap": [  False],
    #                 'criterion': [ 'poisson', 'friedman_mse',  'squared_error']} # aboit 'absolute_error', xq da numeros absolutos
    keys, values = zip(*dict_paramGrid_rf.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dr_res = pd.DataFrame()
    id_count = 0
    dict_models = {}
    # FIND TUNNER
    for d_key in permutations_dicts:
        try:
            print("ID: ", id_count, d_key)
            rf_model = RandomForestRegressor(n_estimators=d_key['n_estimators'],max_features=d_key['max_features'], max_depth=d_key['max_depth'], max_leaf_nodes=d_key['max_leaf_nodes'],
                                  min_samples_split=d_key['min_samples_split'],min_samples_leaf=d_key['min_samples_leaf'], bootstrap=d_key['bootstrap'], criterion=d_key['criterion'])
            rf_model.fit(X_train, y_train)
            dict_models[id_count] = rf_model

            prediction = np.reshape( rf_model.predict(X_test) , (-1, 1))
            df_r, dict_r, dict_r_pos = get_percentage_exist_Regresion(y_test, prediction, per_level=0.5)
            dict_r["id_count"] = id_count
            dict_r = {**dict_r, **dict_r_pos}
            # df_feature_importances = pd.DataFrame({'Columns': rf_model.feature_names_in_, 'Importance':  [x.round(4) for x in rf_model.feature_importances_]})
            dr_res = pd.concat([dr_res, pd.DataFrame( {**dict_r , **d_key}, index=[0])], ignore_index=True)
        except Exception as ex:
            print("ERROR   ", ex)
        id_count = id_count+1

    dr_r = dr_res.sort_values(['Perc_pos'], ascending=True)
    dr_r['Perc_eval'] = (dr_r['Perc'] * 2 + dr_r['Perc_pos'] * 4 )/6
    dr_r = dr_r.sort_values(['Perc_eval'], ascending=False).head(20)[2:] # los dos mas fueres overfit con la semilla¿?

    # SAVED INFO
    count_5 = 0
    count_4 = 0
    for index, row in dr_r.iterrows():
        rf_mod =  dict_models[row["id_count"]]
        if row['max_depth'] == 5:
            count_5 = count_5 +1
            if count_5 >= 2:
                continue
        elif row['max_depth'] == 4:
            count_4 = count_4 +1
            if count_4 >= 2:
                continue
        if row['Perc_eval'] < 0.2:
            print("WARN stock lower than 0.2 ticker: ",TICKER)
            break
        df_feature_importances = pd.DataFrame({'Columns': rf_mod.feature_names_in_, 'Importance':  [x.round(4) for x in rf_mod.feature_importances_]})
        print("Importance:\n", df_feature_importances, "\n")

        exported_text, sas_text, py_text, code_TVW= export_code(rf_mod.estimators_[0], 0, list(rf_mod.feature_names_in_))
        # print(code_TVW)
        name_pine_tree = "d_price/pine_tree/"+TICKER+"_d"+str(row['max_depth'] )+"_q"+str(round(row['Perc_eval'],3))+"_id"+str(row["id_count"])+"_.pine"
        print("\t", name_pine_tree)
        with open(name_pine_tree, "w") as text_file:
            text_file.write( STRING_START_STRATEGY.replace("{TICKER}",TICKER ) + code_TVW + STRING_END_STRATEGY )
        print(index, " path: ", name_pine_tree)
    dr_r.to_csv("d_price/pine_tree/model_info_"+TICKER+"_.csv",sep="\t", index=None)

print("end")



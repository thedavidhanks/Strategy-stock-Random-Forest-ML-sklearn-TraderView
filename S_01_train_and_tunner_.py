# https://es.tradingview.com/script/lLFiT3av/
# Blai5 Koncorde by manboto copy
import numpy as np
import talib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import dtreeviz
import itertools
import pandas as pd
import glob

from technical_parameters_konk_tools import *
from technical_parameters_konk_tools_Prepro import *

print(dtreeviz.__version__)
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import pandas_ta as ta
# from useless.pine_script_convert_funtions import nz
# https://es.tradingview.com/script/lLFiT3av/
# Blai5 Koncorde by manboto copy
import _KEYS_DICT

# Importing libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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



CSV_NAME = "@FAV"
TARGET_TIME = '1Day'
# TARGET_TIME = '5Min'
CSV_NAME = "@FAV"
stocks_list = _KEYS_DICT.DICT_COMPANYS[CSV_NAME]
stocks_list = stocks_list + ["SHOP", "RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"]
stocks_list = stocks_list +[  "LYFT", "ADBE", "UBER", "ZI", "QCOM",  "SPOT", "NVDA", "PTON","CRWD", "NVST", "HUBS", "EPAM",  "SNAP",  "ETSY", "SOFI", "STNE","PDD", "INMD", "CRSR","AMZN","AMD" , "ADSK",  ]
stocks_list = stocks_list +  [ "U", "DDOG", "MELI", "TWLO", "UBER", "GTLB", "RIVN",    "PYPL", "GTLB", "MDB", "TSLA", "UPST"]
# stocks_list = stocks_list + _KEYS_DICT.DICT_COMPANYS["@FOLO1"] +_KEYS_DICT.DICT_COMPANYS["@FOLO2"]+_KEYS_DICT.DICT_COMPANYS["@FOLO3"]

# stocks_list = set(stocks_list)
stocks_list = _KEYS_DICT.DICT_COMPANYS["@FULL_ALL"] # set(stocks_list)
id = stocks_list.index("CX")
stocks_list = ["AAPL", "MELI"]
#BIIB =>  465  CX TEAM 3214   1000:1465  2000:2465
for TICKER in stocks_list[::-1]:
    try:
        list_files = glob.glob(f"d_price\pine_tree\model_info_{TICKER}_.csv")
        # if len(list_files) >=1:
        #     print("Already evaluated ", TICKER)
        #     continue
        path_read_csv = "d_price/yahoo/yahoo_" + TICKER + '_' + TARGET_TIME + "_.csv"
        df_bl = pd.read_csv(path_read_csv, sep="\t")
        print("Read csv Path: ", path_read_csv, " Shape: ", df_bl.shape)
        if len(df_bl) < 1000:
            print("WARN very few rows. Stock: ",TICKER , " Shape: " ,df_bl.shape)
            continue

        df_kon , _= preprocess_df_to_predict_with_konkorde_blaid(df_bl,path_read_csv,  TICKER, TARGET_TIME)

        X = df_kon.drop(columns=[Y_TARGET, 'avg_end_dol', 'avg_end_perAux', 'Date'])  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
        y = df_kon[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state = 42)
        print('\ndf', df_kon.shape, '\t', 'X_train', X_train.shape, '\t','y_train', y_train.shape, '\n',
              'X_test', X_test.shape, '\t','y_test', y_test.shape, '\t')
        # X_xxx = np.asarray(X_train); y_xxx= np.asarray(y_train)


        dict_paramGrid_rf = {"n_estimators": [1],
                        "max_features": [ 0.6, 0.625, 0.65, 0.675, 0.7] ,#[  0.5, 0.6, 0.7],0.4,  0.5,  , "sqrt", "log
                        "max_depth": [4,5],
                        "min_samples_split": [2, 6,10,16,22],
                        "min_samples_leaf": [1,2,5, 9,13 ],
                        "max_leaf_nodes": [122,28,32],
                        "bootstrap": [  False],
                        'criterion': [ 'poisson', 'friedman_mse',  'squared_error']} # aboit 'absolute_error', xq da numeros absolutos
        keys, values = zip(*dict_paramGrid_rf.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        dr_res = pd.DataFrame()
        id_count = 0
        dict_models = {}
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
        dr_r = dr_r.sort_values(['Perc_eval'], ascending=False).head(20) # los dos mas fueres overfit con la semilla¿?
        # for k,v in dict_paramGrid_rf.items():
        #     print("\n", k)
        #     print(dr_r[k].value_counts())
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


    except Exception as ex:
        print("Error ",TICKER, " Exception: ",ex)
#     continue
#
#
#
#     forest_regre1 = RandomForestRegressor(n_estimators=3, max_depth=5, max_leaf_nodes=36)
#     # orest_regre = RandomForestClassifier(n_estimators=5, max_depth=16, max_leaf_nodes=36)
#     #'poisson', 'absolute_error', 'friedman_mse', 'squared_error'
#     # forest_regre2 = RandomForestRegressor(n_estimators=1, criterion="poisson",  max_depth=5, random_state=18, max_leaf_nodes=36)
#     # deci_regre = DecisionTreeRegressor(max_depth=4, criterion="squared_error", random_state=0)
#     # deci_regre2 = DecisionTreeRegressor(max_depth=3, random_state=42)
#     # ddd = DecisionTreeRegressor(max_depth=2, criterion="absolute_error")
#     # eee = DecisionTreeRegressor(criterion="squared_error", random_state=0)
#     #
#     # forest_regre = RandomForestClassifier(n_estimators=3,criterion="gini", max_features='sqrt', max_depth=8, random_state=42 )
#     # deci_regre = DecisionTreeClassifier(max_depth=5, criterion="log_loss", random_state=42, max_leaf_nodes=36)# {'log_loss', 'entropy', 'gini'}
#     # deci_regre = DecisionTreeClassifier(max_depth=3, random_state=42)
#     list_models = [ dict_models[1333] ]#, forest_regre2, deci_regre, deci_regre2]
#     for rf_model in list_models:
#         print("\n", rf_model)
#         # print("Columns: ", list(X_train.columns))
#         rf_model.fit(X_train, y_train)
#         # Predicting the Test set results
#         prediction = rf_model.predict(X_test)  # Predict over X_test
#         prediction = np.reshape(prediction, (-1, 1))
#         df_r, dict_r, dict_r_pos =  get_percentage_exist_Regresion(y_test, prediction, per_level=0.5)
#
#         df_feature_importances = pd.DataFrame({'Columns': rf_model.feature_names_in_, 'Importance':  [x.round(4) for x in rf_model.feature_importances_]})
#         print("Importance:\n", df_feature_importances, "\n")
#         if "RandomForest" in str(rf_model):
#             eval_model = rf_model.estimators_[0]
#         elif "DecisionTree" in str(rf_model):
#             eval_model = rf_model
#
#
#         # tree_to_code(rf_model.estimators_[0], rf_model.feature_names_in_)
#         # r = export_text(rf_model.estimators_[0], feature_names=list(rf_model.feature_names_in_))
#         exported_text, sas_text, py_text, code_TVW= export_code(eval_model, 0, list(rf_model.feature_names_in_))
#         print(code_TVW)
#
#         if  "RandomForest" in str(rf_model) :
#             key_name = str(rf_model.estimator) + "_" + str(rf_model.criterion) + "_" + "".join([x[0] for x in rf_model.feature_names_in_])
#             print("print_RandomForest", key_name, rf_model.feature_names_in_)
#             for i in range(0, len( rf_model.estimators_)):
#                 fig, axes = plt.subplots(figsize = (20,12) )
#                 tree.plot_tree(rf_model.estimators_[i] , feature_names=df_kon.columns, class_names=["buy", "none", "sell"] , filled=True , fontsize=7);
#                 fig.savefig('vf_'+key_name+"_"+str(i)+'_.png', dpi=100)
#         elif  "DecisionTree" in str(rf_model) :
#             key_name = str(rf_model.max_features_) + "_" + str(rf_model.criterion) + "_" + "".join([x[0] for x in rf_model.feature_names_in_])
#             print("print_DecisionTree", key_name, rf_model.feature_names_in_)
#             viz_model = dtreeviz.model(rf_model,
#                             X_train,  # pandas.DataFrame
#                            y_train,#tree_index=1,  # pandas.Series
#                            target_name=Y_TARGET,feature_names=list(X_train.columns) , class_names=["buy", "none", "sell"] ) #, fancy=False,
#             viz_model.view(scale=0.8).save("view_"+key_name+".svg")  # viz_model.show()
#             # viz_model.view(orientation="LR").save("viewLR_"+key_name+".svg")
#             viz_model.view(fancy=False).save("viewFancy_"+key_name+".svg")
#             # viz_model.view(depth_range_to_display=(1, 2)).save("viewDepth_"+key_name+".svg")  # root is level 0
#             x = X_test.iloc[70]
#             viz_model.view(x=x).save("view70_"+key_name+".svg")
#             viz_model.view(x=x, show_just_path=True).save("view70B_"+key_name+".svg")
#             print(viz_model.explain_prediction_path(x))
#             plt.close()
#             viz_model.instance_feature_importance(x )#.save("r_viz_model_importance.svg")
#             plt.savefig("viewFeature_"+key_name+".png");plt.close()
#             viz_model.leaf_sizes(figsize=(3.5, 2))#.save("r_viz_model_leaf_sizes.svg")
#             plt.savefig("viewLeafs_"+key_name+".png");plt.close()
#             viz_model.rtree_leaf_distributions()#.save("r_viz_model_leaf_distributions.svg")
#             plt.savefig("viewLeafD_"+key_name+".png");plt.close()
#             viz_model.node_stats(node_id=4)#.save("r_viz_model_node_stats.svg")
#             plt.savefig("viewNode_"+key_name+".png");plt.close()
#             viz_model.leaf_purity(figsize=(3.5, 2))#.save("r_viz_model_leaf_purity.svg")
#             plt.savefig("viewLeafP_"+key_name+".png")
#
#         # viz_model.ctree_feature_space(show={'splits', 'title'}, features=['azul', 'verde'],figsize=(5, 1))
#         # viz_model.rtree_feature_space(show={'marron', 'media'}, features=['azul', 'verde'])
#         # viz_model.rtree_feature_space3D(features=['azul', 'verde'],fontsize=10,elev=30, azim=20,show={'splits', 'title'},colors={'tessellation_alpha': .5})
#
#
#     df_kon = df_kon
#     df_kon.index = df_kon['Date']  # pd.to_datetime(df_plot['Date'])
#
#     for i_step in range(0, len(df_kon), 220):
#         df_plot = df_kon[i_step:i_step+220]
#         df_plot.reset_index(drop=True, inplace=True)
#         fig, ax =  plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [7, 2]}) # plt.subplots(2)
#         # figure = plt.gcf()  # get current figure
#         fig.set_size_inches(36, 8)
#         plt.axhline(y=0, color='black', linestyle='-')
#         plt.title(path_save_img.replace("d_price/img_kond/alpaca_", ""))
#         ax[0].autoscale(enable=True)
#         ax[0].plot(df_plot['verde'], color="#006600")
#         ax[0].plot(df_plot['marron'], color="#330000")
#         ax[0].plot(df_plot['media'], color="#FF0000")
#         ax[0].plot(df_plot['azul'], color="#000066")
#         # ax[0].plot(  (df_plot['Close'] - df_plot['Close'].min()) *0.8, color="#9544a9", linestyle="dashdot")
#         # ax[1].plot(df_kon[['p_espejo_fake', 'p_espejo_good', 'p_cero', 'p_cero_2', 'p_primavera']] )
#         # ax[1].legend(['p_espejo_fake', 'p_espejo_good', 'p_cero', 'p_cero_2', 'p_primavera'])
#
#         # df_kon[['p_espejo_fake', 'p_espejo_good', 'p_cero', 'p_cero_2', 'p_primavera']].plot(ax=ax[1] )
#
#         ax[0].fill_between(df_plot['Date'].astype(str), df_plot['marron'], df_plot['verde'], facecolor='#66FF66')
#         ax[0].fill_between(df_plot['Date'].astype(str), df_plot['marron'], [0.0] * len(df_plot), facecolor='#FFCC99')
#         ax[0].fill_between(df_plot['Date'].astype(str), df_plot['azul'], [0.0] * len(df_plot), facecolor='#00FFFF')
#         ax[0].set_xticks(df_plot['Date'].astype(str))
#         ax[0].set_xticklabels(df_plot['Date'].dt.strftime('%Y-%m-%d'))
#         # from matplotlib import dates
#         # plt.gca().xaxis.set_major_locator(dates.MonthLocator())
#         # plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%b\n%Y"))
#         _ = plt.xticks(rotation=90)
#         ax[0].set_xticklabels([t if not i % 3 else "" for i, t in enumerate(ax[0].get_xticklabels())])
#         print("\tSaved plot path: ", path_save_img)
#         fig.tight_layout()
#         path_save_img_i_step = path_save_img.replace(".png",str(i_step) +"_.png")
#         print("\t", path_save_img_i_step)
#         plt.savefig(path_save_img_i_step, bbox_inches='tight', dpi=100)
#
#     print("end")
#     #//PATRON PRIMAVERA
#
#
# print("end")
# dict_paramGrid_rf = {"n_estimators": [1,2,4,6,8,10],
#                 "max_features": [None, 0.4,  0.5, 0.6, 0.7, "sqrt", "log2"],
#                 "max_depth": [int(x) for x in np.linspace(10, 45, 5)] + [None],
#                 "min_samples_split": [1, 2, 5, 10],
#                 "min_samples_leaf": [1, 2, 4,6,9,16,21],
#                 "bootstrap": [True, False],
#                 'criterion': ['poisson', 'friedman_mse', 'absolute_error', 'squared_error']}
# df_result = pd.read_csv("Models/TF_multi/_RESULTS_profit_multi_all.csv", index_col=0,sep='\t')

# df['Volume_1'] = df['Volume'].shift(-1)
# // Pececillos
# close[1] will contain the price at the close of the preceding bar
# pvi := volume > volume[1] ? nz(pvi[1]) + (close - close[1]) / close[1]: nz(pvi[1])
# df['pvi'] = np.where(df['Volume'] > df['Volume'].shift(-1),
#                      df['pvi'].shift(-1) + ((df['Volume'] - df['Volume'].shift(-1)) / df['Volume'].shift(-1)),
#                      df['pvi'].shift(-1))
# df['pvi_aux'] = 0
# # for index, row in df.iterrows():
# for i in range(0, len(df) ):
#     row = df.iloc[ i]
#     if row['pvi'] != 0:
#         df.iloc[i, df.columns.get_loc('pvi_aux')] = row['pvi']
#     else:
#         count_reverse = i
#         for i_r in range(count_reverse, 0 ,-1) :
#             # print(i_r, df.iloc[i_r-1, df.columns.get_loc('pvi_aux')])
#             if df.iloc[i_r-1, df.columns.get_loc('pvi')] != 0:
#                 value_last_not_0 = df.iloc[i_r-1, df.columns.get_loc('pvi')]
#                 df.iloc[i_r, df.columns.get_loc('pvi')]  = df.iloc[i_r-1, df.columns.get_loc('pvi')]
#                 break
# df['pvi_A'] = 0
# df['Volume_1'] = df['Volume'].shift(-1)
# for i in range(1, len(df) ):
#     row = df.iloc[ i]
#     if row['Volume'] > row['Volume_1']:
#         pvi_value_i = df.iloc[i - 1, df.columns.get_loc('pvi_A')] + (row['Volume'] - row['Volume_1']) / row['Volume_1']
#         df.iloc[i, df.columns.get_loc('pvi_A')] = pvi_value_i
#     else:
#         df.iloc[i, df.columns.get_loc('pvi_A')] = df.iloc[i -1, df.columns.get_loc('pvi_A')]


# sum(df['Volume'] * ((df['source'] - df['source'].shift(-1) <= 0 ? 0: source), 14)
# for index, row in df.iterrows():
#     if index > 0:
#         prev_source = df.at[index - 1, 'source']
#         if (row['source'] - prev_source) <= 0 :
#             upper_s = 0
#         else:
#             upper_s = row['source']
#         if (row['source'] - prev_source) >= 0 :
#             lower_s = 0
#         else:
#             lower_s = row['source']
#     else:
#         upper_s = 0.
#         lower_s = 0.
#     df.at[index, 'upper_s'] = upper_s
#     df.at[index, 'lower_s'] = lower_s




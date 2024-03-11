import glob
import pickle

import pandas as pd
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from technical_parameters_konk_tools import *
from technical_parameters_konk_tools_Prepro import *

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
print("Number of stocks: ", len(stocks_list))
df_full = pd.DataFrame()


for TICKER in stocks_list:
    path_read_csv = "d_price/yahoo/yahoo_" + TICKER + '_' + TARGET_TIME + "_.csv"
    df_bl = pd.read_csv(path_read_csv, sep="\t")
    print("Read csv Path: ", path_read_csv, " Shape: ", df_bl.shape)
    if len(df_bl) < 1300:
        print("WARN very few rows. Stock: ", TICKER, " Shape: ", df_bl.shape)
        continue
    df_kon, _df_kon_predict = preprocess_df_to_predict_with_konkorde_blaid(df_bl,path_read_csv,  TICKER, TARGET_TIME)
    X = df_kon.drop(columns=[Y_TARGET, 'avg_end_dol', 'avg_end_perAux', 'Date'])  # iloc[:,:-1] df.iloc[:,:-1] #iloc[:,:-1]
    y = df_kon[Y_TARGET]  # df.iloc[:,-1] #.iloc[:,-1] #.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state = 77)
    print('\ndf', df_kon.shape, '\t', 'X_train', X_train.shape, '\t','y_train', y_train.shape, '\n',
          'X_test', X_test.shape, '\t','y_test', y_test.shape, '\t')

    df_stock_eval = df[df['ticker'] == TICKER]
    dr_res = pd.DataFrame()
    id_count = 0
    dict_models = {}
    for index, row in df_stock_eval.iterrows():
        print("ID: ", index)
        try:
            rf_moo = RandomForestRegressor(n_estimators=row['n_estimators'], max_features=get_float_or_str(row['max_features']), max_depth=row['max_depth'], max_leaf_nodes=row['max_leaf_nodes'],
                                           min_samples_split=row['min_samples_split'], min_samples_leaf=row['min_samples_leaf'], bootstrap=row['bootstrap'], criterion=row['criterion'])
            rf_moo.fit(X_train, y_train)
            dict_models[TICKER +"_"+str(row['id_count'])] = rf_moo

            prediction = np.reshape(rf_moo.predict(X_test), (-1, 1))
            df_r, dict_r, dict_r_pos = get_percentage_exist_Regresion(y_test, prediction, per_level=0.5)
            dict_r = {**dict_r, **dict_r_pos}
            dict_r['ticker'] = TICKER
            dict_r['id_count'] = row['id_count']
            dict_r['path_model'] = save_model_rf(rf_moo, TICKER, row, dict_r)
            # df_feature_importances = pd.DataFrame({'Columns': rf_model.feature_names_in_, 'Importance':  [x.round(4) for x in rf_model.feature_importances_]})
            dr_res = pd.concat([dr_res, pd.DataFrame({**dict_r}, index=[0])], ignore_index=True)
        except Exception as ex:
            print("ERROR   ", ex)
    # df.rename(columns=lambda x: x +"_b", inplace=True)
    print("end")

    df3 = df_stock_eval.merge(dr_res, on=['ticker', 'id_count'], how='inner', suffixes=('_1', '_2'))
    df3['final_score'] = (((df3['Perc_1'] + df3['Perc_2']) *2 + (df3['Perc_pos_1'] + df3['Perc_pos_2']) *4) / 12).round(3)
    df3 = df3.drop(columns=['Perc_1', 'Total_1', 'Bad_1', 'Good_1',  'Perc_pos_1','Total_pos_1', 'Bad_pos_1', 'Good_pos_1','Perc_2', 'Total_2', 'Bad_2', 'Good_2', 'Perc_pos_2', 'Total_pos_2','Bad_pos_2', 'Good_pos_2'])
    df_full = pd.concat([df_full,df3], ignore_index=True)
    print("sss")

df_full.to_csv("d_price/RF/aa_RF_full_eval.csv", sep="\t", index=None)
print("end")

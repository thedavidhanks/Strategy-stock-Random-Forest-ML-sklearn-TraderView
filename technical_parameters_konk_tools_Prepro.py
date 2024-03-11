import re

import numpy as np
import dtreeviz
print(dtreeviz.__version__)
import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import pickle

from technical_parameters_konk_tools import *


### PROCESAR PARA ENTRENAR
def manage_dates_from_csv(df_bl):
    df_bl['Date'] = pd.to_datetime(df_bl['Date'])
    df_bl.index = df_bl['Date'].dt.strftime("%Y-%m-%d")
    max_recent_date = df_bl['Date'].max().strftime("%Y%m%d")  # pd.to_datetime().strftime("%Y%m%d")
    min_recent_date = df_bl['Date'].min().strftime("%Y%m%d")
    return df_bl, max_recent_date, min_recent_date


Y_TARGET = 'avg_end_per'
DOLLARS_TO_BUY = 100


def rolling_buy_sell_val_BUY(df_ind):
    global df_all
    df_sub = df_all.loc[df_ind.index].reset_index()

    list_value_sell = []
    # list_value_sell_debug = []
    for i in range(1, len(df_sub)):
        units_stocks_buy_value = 100 / df_sub['Close'][0]
        dol_selled = units_stocks_buy_value * df_sub['Close'][i]
        importance_c = len(df_sub) - i
        # if (i+1) % 2 == 0:
        list_value_sell += [dol_selled] * importance_c
        # list_value_sell_debug.append( (dol_selled.round(1), df_sub['Close'][i].round(1)) )

        per_change = get_per_change(dol_selled, 100)
        per_close_change = get_per_change(df_sub['Close'][i], df_sub['Close'][0])
        # if round(per_change, 8) != round(per_close_change, 8):
        #     print("aa")

        # assert round(per_change, 8) == round(per_close_change, 8)
    avg_end_dol = np.mean(list_value_sell)
    avg_end_per = get_per_change(avg_end_dol, df_sub['Close'][0])
    return avg_end_dol  # , avg_end_per
    # print("EEE")
df_all = None
def get_means_konkorde_blaid(df_kon):
    global df_all
    df_all = df_kon
    df_kon['azul_mean'] = df_kon['azul'].rolling(min_periods=1, window=5).mean()
    df_kon['verde_mean'] = df_kon['verde'].rolling(min_periods=1, window=5).mean()
    df_kon['marron_mean'] = df_kon['marron'].rolling(min_periods=1, window=5).mean()
    df_kon["verde_azul"] = df_kon['verde'] - df_kon['azul']
    df_kon["verde_media"] = df_kon['verde'] - df_kon['media']
    df_kon["media_azul"] = df_kon['media'] - df_kon['azul']
    # df_kon["media_marron"] = df_kon['media'] - df_kon['marron']

    df_kon['avg_end_dol'] = df_kon.Close.rolling(min_periods=8, window=8).apply(rolling_buy_sell_val_BUY).shift(-7)
    df_kon['avg_end_perAux'] = percentage_change(100, df_kon["avg_end_dol"])
    return df_kon


def get_GT_day_candle(df_kon):
    # df_kon[Y_TARGET] = np.where( (df_kon[Y_TARGET] > 0.6) | ( df_kon[Y_TARGET] < -0.6), df_kon[Y_TARGET], 0 )
    df_kon[Y_TARGET] = -1;
    PER_VALEU_CHANGE = 1  # porcentage de cambio de compra o venta
    df_kon.loc[df_kon['avg_end_perAux'] > PER_VALEU_CHANGE, Y_TARGET] = 2  # BUY
    df_kon.loc[df_kon['avg_end_perAux'] < -PER_VALEU_CHANGE, Y_TARGET] = 0  # SEEL
    df_kon.loc[(df_kon['avg_end_perAux'] < PER_VALEU_CHANGE) & (
                df_kon['avg_end_perAux'] > -PER_VALEU_CHANGE), Y_TARGET] = 1  # NONE
    return df_kon


def decribe_GT_format(df_kon):
    # df_kon[Y_TARGET] = np.where((df_kon['avg_end_perAux'] < 1) & (df_kon['avg_end_perAux'] > -1), 1, df_kon['avg_end_perAux'])
    print("Y_TARGET count() ", df_kon.groupby(Y_TARGET)['Date'].count())
    df_kon[Y_TARGET].describe()


def preprocess_df_to_predict_with_konkorde_blaid(df_bl,path_read_csv,  TICKER, TARGET_TIME):
    df_bl, max_recent_date, min_recent_date = manage_dates_from_csv(df_bl)
    path_save_img = path_read_csv.replace(".csv", ".png").replace("/alpaca/", "/img_kond/")
    df_kon = get_konkorde_params_GOOD(df_bl, num_back_candles_img=80, path_save_img=path_save_img)
    # df_kon = get_paterns_konk_moments(df_kon, path_save_img)
    # df_kon["p_corte"] = np.where((df_kon['p_corte'] == -1)  , 0, df_kon['p_corte'])
    # df_kon[['p_espejo_fake', 'p_espejo_good', 'p_cero', 'p_cero_2', 'p_corte', 'p_primavera']]  = df_kon[['p_espejo_fake', 'p_espejo_good', 'p_cero', 'p_cero_2', 'p_corte', 'p_primavera']] * 100
    # df_kon = df_kon.drop(columns=["p_corte"])
    df_kon = get_means_konkorde_blaid(df_kon)
    print(
        "d_price/alpaca/alpaca_" + TICKER + '_' + TARGET_TIME + "_" + max_recent_date + "__" + min_recent_date + ".csv   df.shape: ",
        df_bl.shape)
    cols_remove = ['tprice', 'pvi', 'nvi', 'source', 'pvi_ema', 'pvim', 'pvimax', 'pvimin', 'oscp',
                   'nvi_ema', 'nvim', 'nvimax', 'nvimin', 'xmf', 'BollOsc', 'xrsi', 'stoc', 'Open', 'High', 'Low',
                   'Close', 'Volume']
    cols_remove_predict = ['tprice', 'pvi', 'nvi', 'source', 'pvi_ema', 'pvim', 'pvimax', 'pvimin', 'oscp',
                   'nvi_ema', 'nvim', 'nvimax', 'nvimin', 'xmf', 'BollOsc', 'xrsi', 'stoc', ]

    df_kon = df_kon.sort_values(['Date'], ascending=True)
    df_kon_predict = df_kon.copy().drop(columns=cols_remove_predict + [ 'avg_end_dol','avg_end_perAux'])
    df_kon = df_kon.drop(columns=cols_remove)
    df_kon_predict = df_kon_predict.drop_duplicates(subset=['Date'], keep="first").dropna(how='any')
    df_kon = df_kon.drop_duplicates(subset=['Date'], keep="first").dropna(how='any')
    df_kon.reset_index(drop=True, inplace=True)
    df_kon = get_GT_day_candle(df_kon)

    # import seaborn
    # sns_plot = seaborn.pairplot(df_kon, hue=Y_TARGET, height=2.5,palette = "Paired", corner=True)
    # plt.savefig('plot_pairplot_relation_GT_'+TICKER+'.png');plt.clf();
    # sns_plot = seaborn.heatmap(df_kon.corr())
    # plt.savefig('plot_heatmap_relation_GT_' + TICKER + '.png');plt.clf();

    # if TARGET_TIME == '5Min':
    #     df_kon = get_column__date_val__date_range_open_close(df_kon)
    #     df_kon.loc[df_kon['date_val'] == 0, Y_TARGET] = 1 #NONE fuera de rango
    #     df_kon = df_kon.drop(columns=['date_val'])
    # decribe_GT_format(df_kon)  # LOGGING
    return df_kon, df_kon_predict

def save_model_rf(rf_moo, TICKER, row, dict_r):
    path_RF = 'd_price/RF/' + TICKER + "_" + str(row['id_count']) + ".rfmodel"
    path_RFtxt = path_RF.replace(".rfmodel", "__" + str(dict_r['Perc_pos']) + ".txt")
    f = open(path_RFtxt, "a")
    f.write(str(dict_r))
    f.close()
    print("\t",path_RF)
    with open(path_RF, 'wb') as file:
        pickle.dump((rf_moo, dict_r), file)
    return path_RF
### PROCESAR PARA ENTRENAR
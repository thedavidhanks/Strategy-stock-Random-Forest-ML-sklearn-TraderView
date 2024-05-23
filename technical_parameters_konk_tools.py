import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import pandas_ta as ta
from sklearn.tree import _tree
import string
from sklearn.tree import export_text
# from useless.pine_script_convert_funtions import nz
# https://es.tradingview.com/script/lLFiT3av/
# Blai5 Koncorde by manboto copy
import _KEYS_DICT


def plot_koncorde_simple(df_plot, path_save_img, centrer_red = None):
    df_plot.index = df_plot['Date']  # pd.to_datetime(df_plot['Date'])
    df_plot.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots()
    # figure = plt.gcf()  # get current figure
    fig.set_size_inches(26, 8)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title(path_save_img.replace("d_price/img_kond/alpaca_", ""))
    ax.autoscale(enable=True)
    ax.plot(df_plot['verde'], color="#006600")
    ax.plot(df_plot['marron'], color="#330000")
    ax.plot(df_plot['media'], color="#FF0000")
    ax.plot(df_plot['azul'], color="#000066")

    if centrer_red is not None:
        # c = df_plot['verde'].max() / df_plot['Close'].max()
        ax.plot( (df_plot['Close']) - (df_plot['Close'].min()  * 1.1), color="#9544a9", linestyle="dashdot")
        ax.vlines(x=centrer_red, ymin=0, ymax=100,  colors='r', linestyle = "dashdot")

    ax.fill_between(df_plot['Date'].astype(str), df_plot['marron'], df_plot['verde'], facecolor='#66FF66')
    ax.fill_between(df_plot['Date'].astype(str), df_plot['marron'], [0.0] * len(df_plot), facecolor='#FFCC99')
    ax.fill_between(df_plot['Date'].astype(str), df_plot['azul'], [0.0] * len(df_plot), facecolor='#00FFFF')
    ax.set_xticks(df_plot['Date'].astype(str))
    ax.set_xticklabels(df_plot['Date'].dt.strftime('%Y-%m-%d'))
    # from matplotlib import dates
    # plt.gca().xaxis.set_major_locator(dates.MonthLocator())
    # plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%b\n%Y"))
    _ = plt.xticks(rotation=90)
    ax.set_xticklabels([t if not i % 2 else "" for i, t in enumerate(ax.get_xticklabels())])
    print("\tSaved plot path: ", path_save_img)
    fig.tight_layout()
    plt.savefig(path_save_img, bbox_inches='tight', dpi=60)
def get_crash_points(df, col_name_A, col_name_B, col_result, highlight_result_in_next_cell = 1 ):
    df["diff"] = df[col_name_A] - df[col_name_B]
    df[col_result] = 0

    df.loc[((df["diff"] >= 0) & (df["diff"].shift() < 0)), col_result] = 1
    df.loc[((df["diff"] <= 0) & (df["diff"].shift() > 0)), col_result] = -1
    #TODO test with oder numer than 1
    if highlight_result_in_next_cell > 0:
        df.loc[((df[col_result].shift(highlight_result_in_next_cell) == 1)), col_result] = 1
        df.loc[((df[col_result].shift(highlight_result_in_next_cell) == -1)), col_result] = -1

    df = df.drop(columns=['diff'])

    return df
# https://github.com/voice32/stock_market_indicators/blob/master/indicators.py
def positive_volume_index(df, periods=15, close_col='Close', vol_col='Volume'):
    df['pvi'] = 0.

    df.reset_index(drop=True, inplace=True)
    for index, row in df.iterrows():
        if index > 0:
            prev_pvi = df.at[index - 1, 'pvi']
            prev_close = df.at[index - 1, close_col]
            if row[vol_col] > df.at[index - 1, vol_col]:
                # pvi = prev_pvi + (row[close_col] - prev_close / prev_close * prev_pvi)
                pvi = prev_pvi + ((row[close_col] - prev_close) / prev_close )# * prev_pvi)
            else:
                pvi = prev_pvi
        else:
            pvi = 0
        # data.set_value(index, 'pvi', pvi)
        df.at[index, 'pvi'] = pvi
    df['pvi_ema'] = df['pvi'].ewm(ignore_na=False, min_periods=periods, com=periods, adjust=True).mean()

    return df
def negative_volume_index(df, periods=15, close_col='Close', vol_col='Volume'):
    df['nvi'] = 0.

    df.reset_index(drop=True, inplace=True)
    for index, row in df.iterrows():
        if index > 0:
            prev_nvi = df.at[index - 1, 'nvi']
            prev_close = df.at[index - 1, close_col]
            if row[vol_col] < df.at[index - 1, vol_col]:
                # nvi = prev_nvi + (row[close_col] - prev_close / prev_close * prev_nvi)
                nvi = prev_nvi + ((row[close_col] - prev_close) / prev_close )# * prev_pvi)
            else:
                nvi = prev_nvi
        else:
            nvi = 0
        df.at[index, 'nvi'] = nvi
    df['nvi_ema'] = df['nvi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()

    return df

def get_konkorde_params_GOOD(df, num_back_candles_img, path_save_img):
    # df['calc_nvi'] =  df.ta.nvi( cumulative=True, append=False) #calc_nvi(df)
    # tprice=ohlc4
    df['tprice'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4 # equivale a ohlc4
    df['pvi'] = 0
    df['nvi'] = 0
    lengthEMA = 255 #input.int(255, minval=1)
    m = 15 #input(15)
    df['source'] = ( df['High'] + df['Low'] + df['Close']) / 3 #hlc3

    #PECECILLOS
    df = positive_volume_index(df)
    # pvim = ta.ema(pvi, m)
    df['pvim'] = ta.ema(df['pvi'], 15)  # df['pvi_ema'] #ta.ema(df['pvi'], 255) # df['pvi_ema'] #ta.ema(df['pvi'], m) # df['pvi_ema'] #
    # pvimax = ta.high (df['pvim'], 90)
    df['pvimax'] = df['pvim'].rolling(90).max() # df.shift(90).rolling(90)['pvim'].max()
    # pvimin = ta.lowest(df['pvim'], 90)
    df['pvimin'] = df['pvim'].rolling(90).min() #df.rolling(-90)['pvim'].min()
    df['oscp'] = (df['pvi'] - df['pvim']) * 100 / (df['pvimax'] - df['pvimin']).replace(0.0,0.000000000000001) # to avoid /0.0  infinitive
    #
    # df_plot = df
    # df_plot.index = df_plot['Date'] #pd.to_datetime(df_plot['Date'])
    # df_plot = df_plot[-46:]
    # fig, ax = plt.subplots()
    # ax.hlines(y=0, xmin=0, xmax=len(df_plot['oscp']), linewidth=2, color='black')
    # ax.plot(df_plot['oscp'], color="green")
    # ax.set_xticks(df_plot.index)
    # _ = plt.xticks(rotation=90)
    # plt.savefig("TEST_kond_green.png", bbox_inches='tight')
    # TIBURONES
    # nvi := volume < volume[1] ? nz(nvi[1]) + (close - close[1]) / close[1]: nz(nvi[1])
    df = negative_volume_index(df)
    df['nvim']  = ta.ema(df['nvi'], m)
    df['nvimax'] = df['nvim'].rolling(90).max()
    df['nvimin'] = df['nvim'].rolling(90).min()
    df['azul'] = (df['nvi'] - df['nvim']) * 100 / (df['nvimax'] - df['nvimin']).replace(0.0,0.000000000000001)
    # // MoneyFlowIndex
    # upper_s = math.sum(volume * (ta.change(source) <= 0 ? 0: source), 14)
    # lower_s = math.sum(volume * (ta.change(source) >= 0 ? 0: source), 14)
    # upper_s = sum(df['Volume'] * (  (source - source.shift(-1)  <= 0 ? 0: source), 14)
    # lower_s = sum(df['Volume'] * (change(source) >= 0 ? 0: source), 14)
    df.reset_index(drop=True, inplace=True)
    # sum(volume * (change(source) <= 0 ? 0: source), 14)
    upper_value = df['Volume'] * (np.where(df['source'] - df['source'].shift(1) <= 0 , 0,  df['source']))
    df['upper_s'] = upper_value.rolling(14).sum()
    lower_value = df['Volume'] * (np.where(df['source'] - df['source'].shift(1) >= 0, 0, df['source']))
    df['lower_s'] = lower_value.rolling(14).sum()
    # rsi = 100 - 100 / (1 + upper_s / lower_s)
    df['xmf'] = 100.0 - 100.0 / (1.0 + (df['upper_s'] / df['lower_s']).replace(0.0,0.000000000000001) )
    # // Bollinger
    mult = 2.0
    basis = ta.sma(df['tprice'], 25)
    dev = mult * ta.stdev(df['tprice'], 25)
    upper = basis + dev
    lower = basis - dev
    OB1 = (upper + lower) / 2.0
    OB2 = (upper - lower).replace(0.0,0.000000000000001) # to avoid /0.0  infinitive
    df['BollOsc']  = (df['tprice'] - OB1) / OB2 * 100 #OK
    df['xrsi']  = ta.rsi(df['tprice'], 14) #OK
    length = 21; smoothFastD = 3;src = df['tprice'];
    # calc_stoch(src, length, smoothFastD) = >
    # ll = ta.lowest(df['low'], length)
    # hh = ta.highest(df['high'], length)
    ll = df['Low'].rolling(length).min() #OK
    hh = df['High'].rolling(length).max() #OK
    k = 100 * (src - ll) / (hh - ll).replace(0.0,0.000000000000001) #OK
    df['stoc']  = ta.sma(k, smoothFastD)

    df['marron'] = (df['xrsi'] + df['xmf'] + df['BollOsc'] + df['stoc'] / 3) / 2
    df['verde']  = df['marron'] + df['oscp']
    df['media']  = ta.ema(df['marron'], m)

    df_plot = df
    df_plot = df_plot[-150:]
    # plot_koncorde_simple(df_plot, path_save_img)

    # cols_remove = ['tprice', 'pvi','nvi', 'source', 'pvi_ema', 'pvim', 'pvimax', 'pvimin', 'oscp',
    #    'nvi_ema', 'nvim', 'nvimax', 'nvimin',  'upper_s', 'lower_s','xmf', 'BollOsc', 'xrsi', 'stoc', ]
    cols_remove = ['upper_s', 'lower_s']
    df = df.drop(columns=cols_remove)
    return df


def plot_print_patern_result(df_sub, df_ind, name_patern, df_kon, path_save_img):
    try:
        print("FIND "+name_patern+" \n", df_sub[['verde', 'azul', 'marron', 'media', ]].round(1))
        df_p = df_kon.iloc[np.r_[df_ind.index.start - 26: df_ind.index.start + 26]]
        len_max = len(df_sub['Date']) -1
        print("\tpath: ",path_save_img.replace("_.png", "_" + (df_sub['Date'][len_max].strftime('%Y_%m_%d')) + name_patern + ".png"))
        plot_koncorde_simple(df_p, path_save_img.replace("_.png", "_" + (df_sub['Date'][len_max].strftime('%Y_%m_%d')) + name_patern + ".png"),centrer_red=30)  # df_kon.iloc[df_ind.index.start - 30 : df_ind.index.start + 30]
    except Exception as ex:
        print("WARN Exception plot_print_patern_result() ", ex)

def get_paterns_konk_moments(df_kon, path_save_img):
    #Empecemos por el fácil. Este patrón es el típico con el que todos empezamos porque se trata de lineas cortando algo. El patrón nos da señal de entrada alcista cuando la Linea Señal (linea roja) cruza por debajo del precio (montaña) y se posiciona bajista cuando cruza por encima. Os dejo la imagen a modo de ejemplo
    def p_corte(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()

        if  all(x == 0 for x in df_sub["crh_marron_media"]):
            return 0
        if df_sub["crh_marron_media"][0:3].sum() == 0 and  df_sub["crh_marron_media"][3] == 1 : # or ( df_sub["crh_marron_media"][2] == 1 and df_sub["crh_marron_media"][3] != -1) ):
            GREEN_THERSHOLD = 40;BLUE_THERSHOLD = 15;MARRON_THERSHOLD = 80; #POSITIVE
            is_blue_up2 = df_sub['azul'][0:3].mean() < BLUE_THERSHOLD or df_sub['azul'][1:4].mean() < BLUE_THERSHOLD
            is_verde_up2 = df_sub['verde'][0:3].mean() < GREEN_THERSHOLD or df_sub['verde'][1:4].mean() < GREEN_THERSHOLD
            is_marron_up2 = df_sub['marron'][0:3].mean() < MARRON_THERSHOLD or df_sub['marron'][1:4].mean() < MARRON_THERSHOLD
            if all([is_blue_up2, is_verde_up2,is_marron_up2]):
                # plot_print_patern_result(df_sub,df_ind,"p_corte_up",df_kon, path_save_img)
                return 1
            return 0
        if df_sub["crh_marron_media"][0:3].sum() == 0 and  df_sub["crh_marron_media"][3] == -1 :
            GREEN_THERSHOLD = 85;BLUE_THERSHOLD = 20;MARRON_THERSHOLD = 70;  # POSITIVE
            is_blue_up2 = df_sub['azul'][0:3].mean() > BLUE_THERSHOLD or df_sub['azul'][1:4].mean() > BLUE_THERSHOLD
            is_verde_up2 = df_sub['verde'][0:3].mean() > GREEN_THERSHOLD or df_sub['verde'][1:4].mean() > GREEN_THERSHOLD
            is_marron_up2 = df_sub['marron'][0:3].mean() > MARRON_THERSHOLD or df_sub['marron'][1:4].mean() > MARRON_THERSHOLD
            if all([ is_blue_up2, is_verde_up2, is_marron_up2]):
                # plot_print_patern_result(df_sub, df_ind, "p_corte_down",df_kon, path_save_img)
                return -1
        return 0

    def p_cero_1(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()
        if df_sub['azul'].mean() < -25:
            print("FIND p_cero_1() media azul muy baja Media:", df_sub['azul'].mean() )
            return 0
        VERDE_THEREHOLD = 20;MARRON_THERSHOLD = 50
        is_pos_first_v = df_sub['verde'][0] > VERDE_THEREHOLD
        is_neg_midle_v = df_sub['verde'][1] + VERDE_THEREHOLD  < df_sub['verde'][2]
        is_pos_last_v = df_sub['verde'][2] > VERDE_THEREHOLD
        is_pos_first_m = df_sub['marron'][0] > MARRON_THERSHOLD
        is_neg_midle_m = df_sub['marron'][1] + MARRON_THERSHOLD < df_sub['marron'][2] #diferencia de mas de 45
        is_pos_last_m = df_sub['marron'][2] > MARRON_THERSHOLD
        if all([is_pos_first_v,is_neg_midle_v,  is_pos_last_v, is_pos_first_m,is_neg_midle_m,  is_pos_last_m]):
            print("FIND p_cero_1() \n", df_sub[['verde', 'azul', 'marron', 'media', ]].round(1))
            df_p = df_kon.iloc[np.r_[df_ind.index.start- 26   : df_ind.index.start + 26] ]
            print("\tpath: ", path_save_img.replace("_.png", "_"+(df_sub['Date'][2].strftime('%Y_%m_%d'))+"p_cero_1.png"  ))
            plot_koncorde_simple(df_p, path_save_img.replace("_.png", "_" + (df_sub['Date'][2].strftime('%Y_%m_%d')) + "p_cero_1.png"), centrer_red=26) # df_kon.iloc[df_ind.index.start - 30 : df_ind.index.start + 30]
            return 1
        return 0
    def p_cero_2(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()
        # if df_sub['azul'].mean() < -20:
        #     print("FIND p_cero_2() media azul muy baja Media:", df_sub['azul'].mean() )
        #     return 0
        VERDE_THEREHOLD = 20;MARRON_THERSHOLD = 50
        is_pos_first_v = df_sub['verde'][0] > VERDE_THEREHOLD
        is_neg_midle_v = df_sub['verde'][1] + VERDE_THEREHOLD < df_sub['verde'][3] or df_sub['verde'][2] + VERDE_THEREHOLD < df_sub['verde'][3]
        is_pos_last_v = df_sub['verde'][3] > VERDE_THEREHOLD
        is_pos_first_m = df_sub['marron'][0] > MARRON_THERSHOLD
        is_neg_midle_m = df_sub['marron'][1] + MARRON_THERSHOLD < df_sub['marron'][3] or df_sub['marron'][2] +MARRON_THERSHOLD < df_sub['marron'][3]
        is_pos_last_m = df_sub['marron'][3] > MARRON_THERSHOLD
        if all([is_pos_first_v, is_neg_midle_v, is_pos_last_v, is_pos_first_m, is_neg_midle_m, is_pos_last_m]):
            print("FIND p_cero_2() \n", df_sub[['verde', 'azul', 'marron', 'media', ]].round(1))
            df_p = df_kon.iloc[np.r_[df_ind.index.start- 26   : df_ind.index.start + 26] ]
            print("\tpath: ", path_save_img.replace("_.png", "_"+(df_sub['Date'][3].strftime('%Y_%m_%d'))+"p_cero_2.png"  ))
            plot_koncorde_simple(df_p, path_save_img.replace("_.png", "_" + (df_sub['Date'][3].strftime('%Y_%m_%d')) + "p_cero_2.png"), centrer_red=26) # df_kon.iloc[df_ind.index.start - 30 : df_ind.index.start + 30]
            return 1
        return 0
    def p_cero_3(df_ind):
        #Cuando la tendencia del precio (la montaña) tiende a cero, significa que habrá un cambio en su tendencia. La forma sencilla de operar esto es posicionarse alcista cuando se aproxima a cero. Es una operación de bajo riesgo y muy efectiva.
        # El Stop Loss siempre se debe colocar por debajo de la última vela bajista. Veamos estos ejemplos en un valor alcista como es Cie Automotive.
        df_sub = df_kon.loc[df_ind.index].reset_index()
        if  all(x > 15 for x in df_sub["marron"]) or all(x > 15 for x in df_sub["verde"]):
            return 0
        VERDE_THEREHOLD = 20; MARRON_THERSHOLD = 20
        is_pos_first_v = (df_sub['verde'][0] + 10 > df_sub['verde'][1]) and ( df_sub['verde'][1] < 10 or df_sub['verde'][2] < 10)
        is_last_first_v = df_sub['verde'][3] > VERDE_THEREHOLD
        is_last_mean_v = df_sub['verde'][1:3].mean() < 0 or df_sub['verde'][1] < 0 or df_sub['verde'][2] < 0
        is_pos_first_m = df_sub['marron'][0] + 10 > df_sub['marron'][1]
        is_last_first_m = df_sub['marron'][3] > MARRON_THERSHOLD
        is_last_mean_m = df_sub['marron'][1:3].mean() < 10 or df_sub['marron'][1] < 10 or df_sub['marron'][2] < 10
        if all([is_pos_first_v, is_last_first_v, is_last_mean_v, is_pos_first_m, is_last_first_m, is_last_mean_m]):
            plot_print_patern_result(df_sub, df_ind, "p_cero_3",df_kon, path_save_img)
            return 1
        return 0
    def p_primavera(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()
        is_pos_first_v = df_sub['crh_marron_media'][0] == 1
        is_neg_midle_v1 = (df_sub['verde'][0] +25 < df_sub['verde'][1] < df_sub['verde'][2]) \
                          or (df_sub['verde'][0]  < df_sub['verde'][1] +25 < df_sub['verde'][2])
        is_neg_midle_v2 = (df_sub['marron'][0] +15 < df_sub['marron'][1] < df_sub['marron'][2]) \
                          or (df_sub['marron'][0]  < df_sub['marron'][1] +15 < df_sub['marron'][2])
        is_neg_midle_v = is_neg_midle_v1 or is_neg_midle_v2
        is_avg_good = df_sub['media'][0]  <= df_sub['media'][1] <= df_sub['media'][2]
        is_azul_algo = df_sub['azul'].mean() > 5 # any(i > 30 for i in df_sub['azul'])
        is_pos_last_m = df_sub['marron'][2] > 60 and df_sub['marron'].mean() < 100
        if all([is_pos_first_v, is_neg_midle_v, is_azul_algo,is_avg_good, is_pos_last_m]):
            plot_print_patern_result(df_sub, df_ind, "p_primavera",df_kon, path_save_img)
            return 1
        return 0
    def p_espejo_good(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()
        GREEN_THERSHOLD = -18; BLUE_THERSHOLD = 20;
        is_green_low1 = df_sub['verde'].mean() < GREEN_THERSHOLD
        is_green_low2 = df_sub['verde'][0:3].mean() < GREEN_THERSHOLD or df_sub['verde'][1:4].mean() < GREEN_THERSHOLD
        is_blue_up1 = df_sub['azul'].mean() > BLUE_THERSHOLD
        is_blue_up2 = df_sub['azul'][0:3].mean() > BLUE_THERSHOLD or df_sub['azul'][1:4].mean() > BLUE_THERSHOLD
        is_stronger_blue = df_sub['azul'].mean() > abs(df_sub['verde'].mean()) -6

        is_fake_espejo = (is_green_low1 or is_green_low2) and (is_blue_up1 or is_blue_up2) and is_stronger_blue

        if is_fake_espejo:
            plot_print_patern_result(df_sub, df_ind, "p_espejo_fake")
        is_arponeada = any(x==1 for x in df_sub["crh_azul_media"][1:4] )  and  df_sub["crh_marron_media"][3] != -1 #and not any(x==-1 for x in df_sub["crh_azul_media"][2:4] ) #Arponea la media a la ballena
        if  is_fake_espejo and is_arponeada:
            plot_print_patern_result(df_sub, df_ind, "p_espejo_good",df_kon, path_save_img)
            return 1
        return 0

    def p_espejo_oso(df_ind):
        df_sub = df_kon.loc[df_ind.index].reset_index()
        GREEN_THERSHOLD = 20; BLUE_THERSHOLD = -20;
        is_green_low1 = df_sub['verde'].mean() > GREEN_THERSHOLD
        is_green_low2 = df_sub['verde'][0:3].mean() > GREEN_THERSHOLD or df_sub['verde'][1:4].mean() > GREEN_THERSHOLD
        is_blue_up1 = df_sub['azul'].mean() < BLUE_THERSHOLD
        is_blue_up2 = df_sub['azul'][0:3].mean() < BLUE_THERSHOLD or df_sub['azul'][1:4].mean() < BLUE_THERSHOLD
        is_stronger_blue = df_sub['verde'].mean() > abs(df_sub['azul'].mean()) -6

        is_fake_espejo = (is_green_low1 or is_green_low2) and (is_blue_up1 or is_blue_up2) and is_stronger_blue

        # if is_fake_espejo:
        #     plot_print_patern_result(df_sub, df_ind, "p_espejo_fake")
        is_arponeada = any(x==-1 for x in df_sub["crh_marron_media"][2:4] )  and  df_sub["crh_marron_media"][3] != 1 #and not any(x==-1 for x in df_sub["crh_azul_media"][2:4] ) #Arponea la media a la ballena
        if  is_fake_espejo and is_arponeada:
            plot_print_patern_result(df_sub, df_ind, "p_espejo_oso",df_kon, path_save_img)
            return 1
        return 0

    # df_kon["p_espejo_fake"] = df_kon['verde'].rolling(4).apply(p_espejo_fake)
    # df_kon.index = df_kon['Date'].loc['2023-05-01':'2023-12-30']
    # df_kon = df_kon.loc['2023-05-01':'2023-12-30']
    # df_kon = df_kon[(df_kon['Date'] > '2023-05-01') & (df_kon['Date'] < '2023-12-30')]
    # df_kon = df_kon[(df_kon['Date'] > '2023-10-17') & (df_kon['Date'] < '2024-03-01')]
    plot_koncorde_simple(df_kon, "TEST_kond_PatronsXX.png")
    df_kon = get_crash_points(df_kon, "marron", "media", col_result="crh_marron_media", highlight_result_in_next_cell=0)
    df_kon = get_crash_points(df_kon, "azul", "media", col_result="crh_azul_media", highlight_result_in_next_cell=0)
    df_kon["p_corte"] = df_kon['verde'].rolling(4).apply(p_corte)
    df_kon["p_cero_3"] = df_kon['verde'].rolling(4).apply(p_cero_3)
    df_kon["p_corte"] = np.where((df_kon['crh_marron_media'] == 1) & (df_kon['marron'] > 80), 1, 0)
    df_kon["p_primavera"] = df_kon['verde'].rolling(3).apply(p_primavera)
    df_kon["p_espejo_good"] = df_kon['verde'].rolling(4).apply(p_espejo_good)
    df_kon["p_espejo_oso"] = df_kon['verde'].rolling(4).apply(p_espejo_oso)

    # df_kon = df_kon.drop(columns=[ "crh_primavera_aux", "crh_corte_aux"])
    return df_kon

def get_per_change(current, previous):
    if current == previous:
        return 100.0
    try:
        return ( (current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0


def percentage_change(col1, col2):
    return ((col2 - col1) / col1) * 100


def get_percentage_exist_Regresion(y_test, prediction, per_level):
    df_eval = pd.DataFrame()
    df_eval["y_test"] = y_test
    df_eval["Prediction"] = prediction
    df_eval["is_same_sig"] = False;

    df_eval['Prediction_aux'] = 1 #None
    df_eval.loc[df_eval['Prediction'] > 1 +per_level , 'Prediction_aux'] = 2 #BUY
    df_eval.loc[df_eval['Prediction'] < 1 -per_level, 'Prediction_aux'] = 0 #SEEL

    df_eval.loc[(df_eval['Prediction_aux'] == df_eval["y_test"]), "is_same_sig"] = True
    # df_eval.loc[(df_eval['Prediction'] < 1) & (df_eval['Prediction'] > -1), Y_TARGET] = 1 #NONE
    df_pos = df_eval.copy()
    df_pos = df_pos.loc[df_pos["y_test"] != 1]
    # df_pos = df_pos.loc[df_eval["Prediction"] > per_level ] #solo las que la predicion esta por encima de 0.5 seran evaluadas
    # df_pos.loc[( df_eval['Prediction_aux'] ==  df_eval["y_test"] ), "is_same_sig"] = True
    # df_pos.loc[(df_pos["y_test"] > per_level) & (df_pos["Prediction"] > per_level), "is_same_sig"] = True
    df_r_pos = df_pos.groupby("is_same_sig").count()
    df_r_pos['per'] = (df_r_pos['y_test'] / (df_r_pos['y_test'].iloc[0] + df_r_pos['y_test'].iloc[1])).round(3)
    print( "POSITIVOS: ", df_r_pos)
    dict_r_pos = {"Perc_pos": df_r_pos['per'][True], "Total_pos": df_r_pos['y_test'].sum(), "Bad_pos": df_r_pos['y_test'][False],
                  "Good_pos": df_r_pos['y_test'][True]}

    # df_eval = df_eval.loc[(df_eval["Prediction"] > per_level) | (df_eval["Prediction"] < per_level *-1)] #solo las que la predicion esta por encima de 0.5 o menor de -0.5 seran evaluadas
    # df_eval.loc[(df_eval["y_test"] < (per_level*-1)) & (df_eval["Prediction"] < (per_level*-1)), "is_same_sig"] = True
    # df_eval.loc[(df_eval["y_test"] > per_level) & (df_eval["Prediction"] > per_level), "is_same_sig"] = True
    df_r = df_eval.groupby("is_same_sig").count()
    df_r['per'] = (df_r['y_test']  / (df_r['y_test'].iloc[0] + df_r['y_test'].iloc[1])).round(3)
    print("MIXTOS: ", df_r)
    dict_r = {"Perc":df_r['per'][True],"Total":df_r['y_test'] .sum() , "Bad":df_r['y_test'][False], "Good":df_r['y_test'][True]  }
    return df_r, dict_r, dict_r_pos

def get_percentage_exist_Regresion_absolut(y_test, prediction, per_level = 0.5):
    df_eval = pd.DataFrame()
    df_eval["y_test"] = y_test
    df_eval["Prediction"] = prediction
    df_eval["is_same_sig"] = False;

    df_pos = df_eval.copy()
    # df_pos = df_pos.loc[df_eval["y_test"] == True]
    df_pos = df_pos.loc[df_eval["Prediction"] > per_level ] #solo las que la predicion esta por encima de 0.5 seran evaluadas
    df_pos.loc[(df_pos["y_test"] < (per_level * -1)) & (df_pos["Prediction"] < (per_level * -1)), "is_same_sig"] = True
    df_pos.loc[(df_pos["y_test"] > per_level) & (df_pos["Prediction"] > per_level), "is_same_sig"] = True
    df_r_pos = df_pos.groupby("is_same_sig").count()
    df_r_pos['per'] = (df_r_pos['y_test'] / (df_r_pos['y_test'][0] + df_r_pos['y_test'][1])).round(3)
    print( "POSITIVOS: ", df_r_pos)
    # dict_r_pos = {"Perc_pos": df_r_pos['per'][True], "Total_pos": df_r_pos['y_test'].sum(), "Bad_pos": df_r_pos['y_test'][False],
    #               "Good_pos": df_r_pos['y_test'][True]}

    df_eval = df_eval.loc[(df_eval["Prediction"] > per_level) | (df_eval["Prediction"] < per_level *-1)] #solo las que la predicion esta por encima de 0.5 o menor de -0.5 seran evaluadas
    df_eval.loc[(df_eval["y_test"] < (per_level*-1)) & (df_eval["Prediction"] < (per_level*-1)), "is_same_sig"] = True
    df_eval.loc[(df_eval["y_test"] > per_level) & (df_eval["Prediction"] > per_level), "is_same_sig"] = True
    df_r = df_eval.groupby("is_same_sig").count()
    df_r['per'] = (df_r['y_test']  / (df_r['y_test'][0] + df_r['y_test'][1])).round(3)
    print("MIXTOS: ", df_r)
    dict_r = {"Perc":df_r['per'][True],"Total":df_r['y_test'] .sum() , "Bad":df_r['y_test'][False], "Good":df_r['y_test'][True]  }
    return df_r, dict_r#, dict_r_pos







#https://stackoverflow.com/questions/65682375/translate-python-export-text-decision-rules-to-sas-if-then-do-end-code
def get_sas_from_text(tree, tree_id, features, text, spacing=2):
    # tree is a decision tree from a RandomForestClassifier for instance
    # tree id is a number I use for naming the table I create
    # features is a list of features names
    # text is the output of the export_text function from sklearn.tree
    # spacing is used to handle the file size
    skip, dash = ' '*spacing, '-'*(spacing-1)
    code = 'data decision_tree_' + str(tree_id) + ';'
    code += ' set input_data; '
    n_end_last = 0 # Number of 'END;' to add at the end of the data step
    splitted_text = text.split('\n')
    text_list = []
    for i, line in enumerate(splitted_text):
        line = line.rstrip().replace('|',' ')

        # Handling rows for IF conditions
        if '<' in line or '>' in line:
            line, val = line.rsplit(maxsplit=1)
            line = line.replace(' ' + dash, 'if')
            line = '{} {:g} THEN DO;'.format(line, float(val))
            n_end_last += 1 # need to add an END;
            if i > 0 and 'PREDICTED_VALUE' in text_list[i-1]: # If there's a PREDICTED_VALUE in line above, then condition is ELSE DO
                line = "ELSE DO; " + line
                n_end_last += 1 # add another END
        # Handling rows for PREDICTED_VALUE
        else:
            line = line.replace(' {} class:'.format(dash), 'PREDICTED_VALUE =')
            line += ';'
            line += '\n end;' # Immediately add END after PREDICTED_VALUE = .. ;
            n_end_last -= 1
        text_list.append(line)
        code += skip + line + '\n'
    code = code[:-1]
    code += 'end; '+ '\n'* n_end_last # add END;
    code += 'run;'
    return code
#https://stackoverflow.com/questions/65682375/translate-python-export-text-decision-rules-to-sas-if-then-do-end-code
# Function to export the tree rules' code : in Python (works) and in SAS (issue with DO; END;
def export_code(tree, tree_id, feature_names, max_depth=100, spacing=2):
    if spacing < 2:
        raise ValueError('spacing must be > 1')

    # Clean up feature names (for correctness)
    nums = string.digits
    alnums = string.ascii_letters + nums
    clean = lambda s: ''.join(c if c in alnums else '_' for c in s)
    features = [clean(x) for x in feature_names]
    features = ['_'+x if x[0] in nums else x for x in features if x]
    if len(set(features)) != len(feature_names):
        raise ValueError('invalid feature names')

    # First: export tree to text
    res = export_text(tree, feature_names=features,
                        max_depth=max_depth,
                        decimals=6,
                        spacing=spacing-1,show_weights=True)

    code_sas = get_sas_from_text(tree, tree_id, features, res, spacing)
    code_py = get_py_from_text(tree, tree_id, features, res, spacing)
    code_TVW = get_TVW_from_text(tree, tree_id, features, res, spacing)
    return res, code_sas, code_py, code_TVW # to take a look at the different code outputs

def get_TVW_from_text(tree, tree_id, features, text, spacing):
    skip, dash = ' '*spacing, '-'*(spacing-1)
    skip, dash = '\t', '-' * (spacing - 1)
    code = 'decision_tree_'+ str(tree_id) + '({}) =>\n'.format(', '.join(features))
    for line in repr(tree).split('\n'):
        if "var float ret = -1 " not in code:
            code += skip + "var float ret = -1 // # " + line + '\n'
    for line in text.split('\n'):
        line = line.rstrip().replace('|','\t')
        if '<' in line or '>' in line:
            line, val = line.rsplit(maxsplit=1)
            line = line.replace('\t' + dash, 'if(')
            line = line.replace('\t if(' , '\tif(') #LUIS
            line = '{} {:g} )'.format(line, float(val))
        else:
            line = re.sub(r"( weights: \[\d{1,3}.\d*\, \d{1,3}.\d*\, \d{1,3}.\d*\])( \w*: \w*.\w*)", r"\2 //#\1", line)
            line = line.replace('{} class:'.format(dash), 'ret :=')#return #for x in list_stocks_models
            line = line.replace('{} value:'.format(dash), 'ret :=')  # for max_features='sqrt' option
            line = re.sub(r"\[(-?\d{1,3}.\d*)\]", r"\1", line) #            re.search(r"\[(-?\d{1,3}.\d*)\]", line, re.IGNORECASE).group(1)
            line = line.replace('\t ret :=', 'ret :=')#.replace('.0', '')
            line = line.replace('\tret :=', 'ret :=')

        line = line.replace('\t ', '\t')#LUIS
        code += skip + line + '\n'

    return code

#https://stackoverflow.com/questions/65682375/translate-python-export-text-decision-rules-to-sas-if-then-do-end-code
# Python function
def get_py_from_text(tree, tree_id, features, text, spacing):
    skip, dash = ' '*spacing, '-'*(spacing-1)
    code = 'def decision_tree_'+ str(tree_id) + '({}):\n'.format(', '.join(features))
    for line in repr(tree).split('\n'):
        code += skip + "# " + line + '\n'
    for line in text.split('\n'):
        line = line.rstrip().replace('|',' ')
        if '<' in line or '>' in line:
            line, val = line.rsplit(maxsplit=1)
            line = line.replace(' ' + dash, 'if')
            line = '{} {:g}:'.format(line, float(val))
        else:
            line = line.replace(' {} class:'.format(dash), 'return')
        code += skip + line + '\n'

    return code

from math import ceil, floor
def float_round(num, places = 0, direction = floor):
    return direction(num * (10**places)) / float(10**places)


def get_column__date_val__date_range_open_close(df_kon):
    TIME_ALPHA_OPEN = "13:35:00";
    TIME_ALPHA_CLOSE = "19:25:00";
    df_data_valid = df_kon.set_index(df_kon['Date']).between_time(TIME_ALPHA_OPEN, TIME_ALPHA_CLOSE).drop(columns=df_kon.columns)
    df_data_valid['date_val'] = 1
    df_kon = pd.concat([df_kon.set_index(df_kon['Date']), df_data_valid], axis=1).reset_index(drop=True)
    df_kon['date_val'] = df_kon['date_val'].replace(np.NaN, 0)
    return df_kon



import pandas as pd
# CROSSOVER tradingview pine line python
# https://stackoverflow.com/questions/70755993/cross-crossunder-and-crossover-function-of-pinescript-in-python
def crossover_series(x: pd.Series, y: pd.Series, cross_distance: int = None) -> pd.Series:
    shift_value = 1 if not cross_distance else cross_distance
    return (x > y) & (x.shift(shift_value) < y.shift(shift_value))

def crossunder_series(x: pd.Series, y: pd.Series, cross_distance: int = None) -> pd.Series:
    shift_value = 1 if not cross_distance else cross_distance
    return (x < y) & (x.shift(shift_value) > y.shift(shift_value))


def get_float_or_str(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
        return None
    try:
        aaa = float(element)
        return aaa
    except ValueError:
        return element

STRING_START_STRATEGY ="""
//@version=5
strategy("konk_{TICKER}_xxxxx", overlay=true, margin_long=100, margin_short=100, pyramiding=5)
"""
STRING_END_STRATEGY ="""    ret //return
pvi = 0.0
nvi = 0.0

tprice = ohlc4
lengthEMA = input.int(255, minval=1)
m = input(15)
source = hlc3

// Pececillos
pvi := volume > volume[1] ? nz(pvi[1]) + (close - close[1]) / close[1]: nz(pvi[1])
pvim = ta.ema(pvi, m)
pvimax = ta.highest(pvim, 90)
pvimin = ta.lowest(pvim, 90)
oscp = (pvi - pvim) * 100 / (pvimax - pvimin)
// Tiburones
nvi := volume < volume[1] ? nz(nvi[1]) + (close - close[1]) / close[1]: nz(nvi[1])
nvim = ta.ema(nvi, m)
nvimax = ta.highest(nvim, 90)
nvimin = ta.lowest(nvim, 90)
azul = (nvi - nvim) * 100 / (nvimax - nvimin)
// Money Flow Index
upper_s = math.sum(volume * (ta.change(source) <= 0 ? 0: source), 14)
lower_s = math.sum(volume * (ta.change(source) >= 0 ? 0: source), 14)
xmf = 100.0 - 100.0 / (1.0 + upper_s / lower_s)
// Bollinger
mult = input(2.0)
basis = ta.sma(tprice, 25)
dev = mult * ta.stdev(tprice, 25)
upper = basis + dev
lower = basis - dev
OB1 = (upper + lower) / 2.0
OB2 = upper - lower
BollOsc = (tprice - OB1) / OB2 * 100
xrsi = ta.rsi(tprice, 14)
calc_stoch(src, length, smoothFastD) =>
    ll = ta.lowest(low, length)
    hh = ta.highest(high, length)
    k = 100 * (src - ll) / (hh - ll)
    ta.sma(k, smoothFastD)

stoc = calc_stoch(tprice, 21, 3)
marron = (xrsi + xmf + BollOsc + stoc / 3) / 2
verde = marron + oscp
media = ta.ema(marron, m)
azul_mean = ta.sma(azul, 5)
verde_mean = ta.sma(verde, 5)
marron_mean = ta.sma(marron, 5)

verde_azul = verde - azul
verde_media = verde - media
media_azul = media - azul
media_marron = media - marron
bandacero = 0

var float stop = na
var float limit1 = na
var float limit2 = na
// https://stackoverflow.com/questions/64524742/pine-script-tradingview-how-to-move-a-stop-loss-to-the-take-profit-level
percent2points(percent) =>
    strategy.position_avg_price * percent / 100 / syminfo.mintick
// sl & tp in % %
sl = percent2points(input(2.92, title="stop loss %%"))
tp1 = percent2points(input(1.12, title="take profit 1 %%"))
tp2 = percent2points(input(2.31, title="take profit 2 %%"))
tp3 = percent2points(input(3.91, title="take profit 3 %%"))
activateTrailingOnThirdStep = input(false,title="activate trailing on third stage (tp3 is amount, tp2 is offset level)")
curProfitInPts() =>
    if strategy.position_size > 0
        (high - strategy.position_avg_price) / syminfo.mintick
    else if strategy.position_size < 0
        (strategy.position_avg_price - low) / syminfo.mintick
    else
        0
calcStopLossPrice(OffsetPts) =>
    if strategy.position_size > 0
        strategy.position_avg_price - OffsetPts * syminfo.mintick
    else if strategy.position_size < 0
        strategy.position_avg_price + OffsetPts * syminfo.mintick
    else
        0
calcProfitTrgtPrice(OffsetPts) =>
    calcStopLossPrice(-OffsetPts)
getCurrentStage() =>
    var stage = 0
    if strategy.position_size == 0 
        stage := 0
    if stage == 0 and strategy.position_size != 0
        stage := 1
    else if stage == 1 and curProfitInPts() >= tp1
        stage := 2
    else if stage == 2 and curProfitInPts() >= tp2
        stage := 3
    stage
stopLevel = -1.
profitLevel = calcProfitTrgtPrice(tp3)

// based on current stage set up exit
// note: we use same exit ids ("x") consciously, for MODIFY the exit's parameters
curStage = getCurrentStage()
float op_operation = decision_tree_0(azul, marron, verde, media, azul_mean, verde_mean, marron_mean, verde_azul, verde_media, media_azul)
if (op_operation <= 1.0)
    if curStage == 1
        stopLevel := calcStopLossPrice(sl)
        strategy.exit("x", loss = sl, profit = tp3, comment = "sl or tp3")
    else if curStage == 2
        stopLevel := calcStopLossPrice(0)
        strategy.exit("x", stop = stopLevel, profit = tp3, comment = "breakeven or tp3")
    else if curStage == 3
        stopLevel := calcStopLossPrice(-tp1)
        strategy.exit("x", stop = stopLevel, profit = tp3, comment = "tp1 or tp3")
    else
        strategy.cancel("x")
// https://stackoverflow.com/questions/64524742/pine-script-tradingview-how-to-move-a-stop-loss-to-the-take-profit-level

// LUIS
if (op_operation >= 1.68) // buy
    stop := close * 0.965
    limit1 := close * 1.03
    limit2 := close * 1.02
    strategy.entry("x", strategy.long, 1, stop=stop, comment="in")

if (op_operation <= 0.1) // sell
    strategy.close("x", comment = "under Le1")
"""
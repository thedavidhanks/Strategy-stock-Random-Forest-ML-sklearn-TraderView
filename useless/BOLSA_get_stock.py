import glob
import re
import _KEYS_DICT
import pandas as pd
# txt_files = glob.glob(r"C:\Users\Luis\Desktop\Stock\*.txt")
# dict_stock = {}
#
#
# for path in txt_files:
#     f = open(path, "r")
#     text = f.read()
#     print(text)
#     list_stocks = re.findall(r'\n([A-Z]{1,5})\t[A-Z]{1}\w*', text)
#     dict_stock[path] = list(set(list_stocks))
# str(dict_stock)
['Growth Technology Stocks', 'Most Actives', 'Most Shorted Stocks', 'Small cap gainers']
stock_full =  _KEYS_DICT.DICT_COMPANYS['Growth Technology Stocks']
la = _KEYS_DICT.DICT_COMPANYS['Growth Technology Stocks']
lb = _KEYS_DICT.DICT_COMPANYS['Most Actives']
lc = _KEYS_DICT.DICT_COMPANYS['Most Shorted Stocks']
ld = _KEYS_DICT.DICT_COMPANYS['Small cap gainers']


df_dol = pd.read_csv("d_result/Konk_earn_2024_02_29.csv",index_col=None, sep='\t')
m = df_dol['entres'].mean()
e = df_dol['dolars_avg'].mean()
f = df_dol[df_dol['dolars_avg'] >1 ].mean()
df_all = pd.read_csv("d_price/RF/aa_full.csv",index_col=None, sep='\t')
stocks_list_df = df_all['ticker'].unique()
stock_list =  list(set(la + lb + lc + ld  + list(df_all['ticker'].unique()) ))
stock_list.sort()
print(str(stock_list))
print("sss")
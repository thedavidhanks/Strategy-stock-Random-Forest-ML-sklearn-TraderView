import pandas as pd
import os
from datetime import date, timedelta
from datetime import datetime
import uuid

from UtilsL import bcolors
# from Utils import Utils_send_message
# from Utils.Plot_scrent_shot import get_traderview_screem_shot
# from Utils.UtilsL import bcolors
#
# # https://mothereff.in/twitalics type letters
# from api_twitter import twi_  # "𝘾𝙤𝙣𝙛𝙞𝙙𝙚𝙣𝙘𝙚 𝙤𝙛 𝙢𝙤𝙙𝙚𝙡𝙨:""""📈 𝗕𝗨𝗬 📈    𝗦𝗘𝗟𝗟 𝘾𝙤𝙣𝙛𝙞𝙙𝙚𝙣𝙘𝙚 𝙤𝙛 𝙢𝙤𝙙𝙚𝙡𝙨:📊𝙉𝙖𝙢𝙚𝙨:"""
# tweet_text = alert_message_without_tags.replace(" BUY ", "𝗕𝗨𝗬").replace(" SELL ", "𝗦𝗘𝗟𝗟").replace(
#     "Confidence of models:", "𝙈𝙤𝙙𝙚𝙡 𝙏𝙧𝙪𝙨𝙩:").replace("📊⚙Model names:","📊⚙𝙉𝙖𝙢𝙚𝙨: #𝙩𝙧𝙖𝙙𝙚𝙧")  # UNICODE:   𝙈𝙤𝙙𝙚𝙡 𝙣𝙖𝙢𝙚𝙨:
#tweet_text = tweet_text.replace("\t\t", ' ').replace("_mult_", '_mu_')[:270]  # MAX  [:280+36] tweet limit 20 por cada url

BAR_UP = "📈"
BAR_DOWN = "📉"
BAR_SIMPLE = "📊"
FLECHA_UP = "⬆"
FLECHA_DOWN = "⬇"
TWITER_ON = False
DOLARS_TO_OPERA = 100
# DICT_STRATEGY = {}
OPEN = "open";CLOSE = "close"
LONG = "long";SHORT = "short"
Stop_sl = 0
Limit_tp = 0
class Strategy:
    """
    A class to represent a trade and holds all data related to opening and closing of an un-levered position
    """
    id = ""
    Position_avg_price = 0
    Mintick = 0.01  # 0.01 #Tick sizes were once quoted in fractions (e.g., 1/16th of $1), but today are predominantly based on decimals and expressed in cents.
    Position_size = 1 #1 buy 2 sell 0 none Dirección y tamaño de la posición actual en el mercado. Si el valor es > 0, la posición en el mercado es larga. Si el valor es < 0, la posición es corta. El valor absoluto es la cantidad de contratos/acciones/lotes/unidades en negociación (tamaño de la posición).
    # print(row['Date'], f"sl={sl} tp1={tp1} tp2={tp2} tp3={tp3} position_avg_price={Position_avg_price}") #, sl, tp1, tp2, tp3)
    name = ""
    comment =""
    type_long_short = None
    type_open_close = None
    Date_start = None;Date_end = None;
    PRICE_ENTER = 0
    PRICE_CLOSE = 0
    PER_EARN = 0.0
    Dolars_earn = 0
    Stop_sl_class = 0
    Limit_tp_class = 0
    ticker = ""
    pre_score = 0
    sl = 0
    tp1 = 0
    tp2 = 0
    tp3 = 0
    row_update_data = None
    path_imgs_tech = []
    def percent2points(self, percent):
        # strategy.position_avg_price * percent / 100 / syminfo.mintick
        return round(self.Position_avg_price * percent / 100 / self.Mintick, 3)
    def gets_update_sl_tp(self):        # // sl & tp in % %
        self.sl = self.percent2points(2.92 )#, title="stop loss %%"))
        self.tp1 = self.percent2points(1.12)#, title="take profit 1 %%"))
        self.tp2 = self.percent2points(2.31)#, title="take profit 2 %%"))
        self.tp3 = self.percent2points(3.91)#, title="take profit 3 %%"))
        # print(row['Date'], f" self.sl={self.sl} self.tp1={self.tp1} self.tp2={self.tp2} self.tp3={self.tp3} Position_avg_price={self.Position_avg_price}")
    def curProfitInPts(self):
        if self.Position_size > 0:#BUY
            # print(f"Position_size > 0 High.sl={self.row_update_data['High']} self.Position_avg_price={self.Position_avg_price} self.Mintick={self.Mintick}")
            return (self.row_update_data['High'] - self.Position_avg_price) / self.Mintick
        elif self.Position_size < 0:#SELL
            return (self.Position_avg_price - self.row_update_data['Low']) / self.Mintick
        else:
            return 0
    def calcStopLossPrice(self,OffsetPts):
        if self.Position_size > 0: #BUY
            return self.Position_avg_price - OffsetPts * self.Mintick
        elif self.Position_size < 0:#SELL
            return self.Position_avg_price + OffsetPts * self.Mintick
        else:
            return 0
    def calcProfitTrgtPrice(self, OffsetPts):
        return self.calcStopLossPrice(-OffsetPts)

    stage = 0
    def getCurrentStage(self):
        # self.stage = 0
        if self.Position_size == 0 :
            self.stage = 0
        if self.stage == 0 and self.Position_size != 0:
            # print(f"stage == 0  r1   close={close} curProfitInPts={self.curProfitInPts()} self.Position_size={self.Position_size} ")
            self.stage = 1
        elif self.stage == 1 and self.curProfitInPts() >= self.tp1:
            # print("stage == 1 r2   close={close} curProfitInPts={self.curProfitInPts()} tp1={self.tp1} ")
            self.stage = 2
        elif self.stage == 2 and self.curProfitInPts() >= self.tp2:
            # print("stage == 2 r3   close={close} curProfitInPts={self.curProfitInPts()} tp2={self.tp2} ")
            self.stage = 3
        return self.stage

    def __init__(self, ticker, per_score, row_update_data, name, type_long_short, units, stop= None, limit = None, comment=""):
        global Stop_sl
        global Limit_tp
        self.id = str(uuid.uuid4())[:5]
        self.ticker = ticker
        self.Position_avg_price = (row_update_data['Open'] +row_update_data['High']+row_update_data['Low']+row_update_data['Close'] ) /4
        self.row_update_data = row_update_data
        self.PRICE_ENTER = row_update_data['Open']
        self.pre_score = per_score
        self.type_open_close = OPEN
        self.name = name
        self.Date_start = row_update_data['Date']
        self.type_long_short = type_long_short
        self.units = units
        Stop_sl = 0 ; Limit_tp = 0;
        if stop is not None:
            Stop_sl = stop
        if limit is not None:
            Limit_tp = limit
        else:
            Limit_tp = row_update_data['Open'] *10
        self.comment = comment+" " +datetime.today().strftime('%Y_%m_%d')
        print(bcolors.OKBLUE, "Open strategy ", bcolors.ENDC, "date: ", self.Date_start, "Position_avg_price: ", self.Position_avg_price, "Stop_sl: ",Stop_sl, "name: ", self.name)

        stopLevel = -1.
        self.gets_update_sl_tp()
        profitLevel = self.calcProfitTrgtPrice(self.tp3)  # ok
        curStage = self.getCurrentStage()
        path_TW_candle = "d_price/TRAVIEW_stra/"+ticker+"_stra.png"
        self.path_imgs_tech = []
        if os.path.isfile(path_TW_candle):
            self.path_imgs_tech = [path_TW_candle]
            # if not (path_TW_candle  in self.path_imgs_tech):
            #     self.path_imgs_tech.append( path_TW_candle )
        else:
            print("WARN la ruta del fichero candle TW no existe Path: "+path_TW_candle)
        self.update_registre_history_tp_sl()

    def update(self,row_update_data ):
        global Stop_sl
        global Limit_tp
        if self.type_open_close == CLOSE:
            return
        self.row_update_data = row_update_data
        stopLevel = -1.
        self.gets_update_sl_tp()
        profitLevel = self.calcProfitTrgtPrice(self.tp3)  # ok
        curStage = self.getCurrentStage()

        if curStage == 1:
            stopLevel = self.calcStopLossPrice(self.sl)  #:=
            # print(row['Date'], bcolors.OKCYAN,  "curStage == 111 ", bcolors.ENDC +  f"  close={close} stopLevel={stopLevel} tp3={self.tp3} curStage={curStage}")
            # DICT_STRATEGY[k].exit("x", loss=sl, profit=tp3, comment="sl or tp3")
            Limit_tp = self.tp3
            self.comment = "1 curStage sl or tp3 update tp: "+str(round(Limit_tp,3))+" " +datetime.today().strftime('%Y_%m_%d')
        elif curStage == 2:
            stopLevel = self.calcStopLossPrice(0)  #:=
            # print(row['Date'], bcolors.OKCYAN,  "curStage == 222 ", bcolors.ENDC + f" curStage == 222 close={close} stopLevel={stopLevel} tp3={self.tp3} curStage={curStage}")
            # DICT_STRATEGY[k].exit("x", stop=stopLevel, profit=tp3, comment="breakeven or tp3")
            Stop_sl = stopLevel
            Limit_tp = self.tp3
            self.comment = "2 breakeven or tp3 update sp: "+str(round(Stop_sl,3)) +" tp: "+str(round(Limit_tp,3))+" " +datetime.today().strftime('%Y_%m_%d')
        elif curStage == 3:
            stopLevel = self.calcStopLossPrice(-self.tp1)  #:=
            # print(row['Date'],  bcolors.OKCYAN,  "curStage == 333 ", bcolors.ENDC +f" curStage == 333 close={close} stopLevel={stopLevel} -tp1={-self.tp1} curStage={curStage}")
            # DICT_STRATEGY[k].exit("x", stop=stopLevel, profit=tp3, comment="tp1 or tp3")
            Stop_sl = stopLevel
            Limit_tp = self.tp3
            self.comment = "3 breakeven or tp3 update sp: "+str(round(Stop_sl,3)) +" tp: "+str(round(Limit_tp,3))+" " +datetime.today().strftime('%Y_%m_%d')
    def close(self, row_update_data):
        global Stop_sl
        global Limit_tp
        if self.type_open_close == CLOSE:
            return
        self.PRICE_CLOSE = row_update_data['Close']
        self.row_update_data = row_update_data
        print(bcolors.OKGREEN,  "Close Tomo el Close ubicado en: ", bcolors.ENDC , Stop_sl, "percentage: ", self.PER_EARN)
        self.comment = "closed by low score: "+str(round( row_update_data['predict'] ,3))+" " +datetime.today().strftime('%Y_%m_%d')
        self.write_operation_Close_change(row_update_data)
        self.update_registre_history_tp_sl()
        # if self.row_update_data['Date'].strftime('%Y-%m-%d') == (date.today() - timedelta(days=1)).strftime('%Y-%m-%d'):
        #     try:
        #         BOLSA_051_alpaca.close_remove_operation(self)
        #         if TWITER_ON:
        #             text = "📊 Close 📊  SL by Score " + self.ticker + " Id: " + self.id + \
        #                    "\nOpen price: " + str(self.PRICE_ENTER) + " Close price: " + str(self.PRICE_CLOSE) + \
        #                    "\n Earn: "  +str(round((self.PER_EARN - 1) * 100, 1)) +"%"
        #             twi_.put_tweet_with_images(text, list_images_path=[])
        #     except Exception as ex:
        #         print("Error  📊 Close 📊  SL by Score ", ex)

    def check_stoploss_take_profit(self, row_update_data ):
        global Stop_sl
        global Limit_tp
        if self.type_open_close == CLOSE:
            return
        self.Stop_sl_class = Stop_sl
        self.Limit_tp_class = Limit_tp
        self.row_update_data = row_update_data
        if row_update_data['High'] >= Limit_tp:
            self.PRICE_CLOSE = Limit_tp
            print(bcolors.OKGREEN, "Close Tomo el Take profit ubicado en: ", bcolors.ENDC  , "Date: ",row_update_data['Date'], Limit_tp , "percentage: ",self.PER_EARN )
            self.comment = "end touch tp: "+str(Limit_tp)+" " +datetime.today().strftime('%Y_%m_%d')
            self.write_operation_Close_change(row_update_data)
            # if self.row_update_data['Date'].strftime('%Y-%m-%d') == (date.today() - timedelta(days=1)).strftime('%Y-%m-%d'):
            #     try:
            #         BOLSA_051_alpaca.close_remove_operation(self)
            #         if TWITER_ON:
            #             text = "📊 Close 📊 touch TP position at " + self.ticker + " Id: " + self.id + \
            #                    "\nOpen price: " + str(self.PRICE_ENTER) + " Close price: " + str(self.PRICE_CLOSE) + \
            #                    "\n📊TP: " + str(self.Limit_tp_class) + " Earn: " +str(round((self.PER_EARN - 1) * 100, 1))+"%"
            #             twi_.put_tweet_with_images(text, list_images_path=[])
            #     except Exception as ex:
            #         print("Error  📊 Close 📊 touch TP ", ex)

        if row_update_data['Low'] <= Stop_sl:
            self.PRICE_CLOSE = Stop_sl
            print(bcolors.OKGREEN, "Close Tomo el Stop loss ubicado en: ", bcolors.ENDC ,"sploss:" ,Stop_sl , "percentage: ",self.PER_EARN, "Date_end: ",row_update_data['Date'], "Date_start: ",self.Date_start  )
            self.comment = "end touch sl: " + str(Stop_sl)+" " +datetime.today().strftime('%Y_%m_%d')
            self.write_operation_Close_change(row_update_data)
            # if self.row_update_data['Date'].strftime('%Y-%m-%d') == (date.today() - timedelta(days=1)).strftime('%Y-%m-%d'):
            #     try:
            #         BOLSA_051_alpaca.close_remove_operation(self)
            #         if TWITER_ON:
            #             text = "📊 Close 📊 touch  touch SL position at " + self.ticker +" Id: " + self.id +\
            #                    "\nOpen price: " + str(self.PRICE_ENTER)  + " Close price: " + str(self.PRICE_CLOSE)+\
            #                     "\n"+BAR_DOWN+"SL: " + str(self.Stop_sl_class) + " Earn: " +str(round((self.PER_EARN-1) *100, 1))+"%"
            #             twi_.put_tweet_with_images(text, list_images_path=[])
            #     except Exception as ex:
            #         print("Error  📊 Close 📊 touch SL ", ex)
        self.update_registre_history_tp_sl()

    def write_operation_Close_change(self, row_update_data):
        print(row_update_data)
        self.type_open_close = CLOSE
        self.Date_end = row_update_data['Date']
        self.row_update_data = row_update_data
        self.PER_EARN = (self.PRICE_CLOSE - self.PRICE_ENTER)  # *100 /self.PRICE_ENTER
        stocks_broght = DOLARS_TO_OPERA /  self.PRICE_ENTER
        dollars_sell = stocks_broght * self.PRICE_CLOSE
        self.Dolars_earn = dollars_sell - DOLARS_TO_OPERA
        self.id = self.ticker + "_" + self.Date_start.strftime("%Y%m%d")
        df_stra = pd.DataFrame({"id": self.id,"ticker": self.ticker, "Date_start": self.Date_start.strftime("%Y-%m-%d"), "Date_end": self.Date_end.strftime("%Y-%m-%d"), "time_take":self.Date_end - self.Date_start,
                      "dolars_earn":self.Dolars_earn , "earn": round(self.PER_EARN,3), "enter": self.PRICE_ENTER, "close": self.PRICE_CLOSE,
                      "sl":  self.Stop_sl_class, "tp": self.Limit_tp_class ,"per_score":self.pre_score,"comment": self.comment, "name": self.name, "type_open_close ": self.type_open_close,
                      "sl1":  self.sl, "tp1": self.tp1 , "tp2": self.tp2 , "tp3": self.tp3, "current_date": self.row_update_data['Date'].strftime('%Y-%m-%d')  }, index=[0])
        register_MULTI_in_zTelegram_Registers(df_stra.round(3), "d_result/stra_simulator/" + self.ticker + ".csv")
        print("\tprecio compra: ", self.PRICE_ENTER, " precio venta: ", self.PRICE_CLOSE)

    def update_registre_history_tp_sl(self):
        if  self.Date_start !=None  and self.Date_end != None:
            time_taked =  self.Date_end -  self.Date_start
            time_out =self.Date_end.strftime("%Y-%m-%d"),
        else:
            time_taked = "-"
            time_out = ""
        if self.Stop_sl_class == 0 and self.Limit_tp_class == 0:
            self.Stop_sl_class = Stop_sl
            self.Limit_tp_class = Limit_tp
        self.id = self.ticker + "_" + self.Date_start.strftime("%Y%m%d")
        df_stra = pd.DataFrame({"id": self.id,"ticker": self.ticker, "Date_start": self.Date_start.strftime("%Y-%m-%d"), "Date_end": time_out, "time_take":time_taked,
                      "dolars_earn":self.Dolars_earn , "earn": round(self.PER_EARN,3), "enter": self.PRICE_ENTER, "close": self.PRICE_CLOSE,
                      "sl":  self.Stop_sl_class, "tp": self.Limit_tp_class ,"per_score":self.pre_score,"comment": self.comment, "name": self.name, "type_open_close ": self.type_open_close,
                      "sl1":  self.sl, "tp1": self.tp1 , "tp2": self.tp2 , "tp3": self.tp3, "current_date": self.row_update_data['Date'].strftime('%Y-%m-%d')  }, index=[0])
        df_stra = df_stra[["id", "ticker", "enter", "sl", "tp", "close", "per_score", "Date_start", "Date_end", "type_open_close ","time_take", "dolars_earn", "earn", "comment", "name", "sl1", "tp1", "tp2", "tp3", "current_date"]]
        register_MULTI_in_zTelegram_Registers(df_stra.round(3), "d_result/stra_simulator/" + self.ticker + ".csv")



        if self.row_update_data['Date'].strftime('%Y-%m-%d') == (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')  :
            df_stra_aux = df_stra[["id", "ticker","enter",  "sl", "tp","close","per_score", "Date_start", "Date_end", "type_open_close ", "time_take", "dolars_earn", "earn",  "comment", "name", "sl1", "tp1", "tp2", "tp3", "current_date"]]
            register_MULTI_in_zTelegram_Registers(df_stra_aux.round(3), "d_result/today_operation/today_" + datetime.today().strftime('%Y_%m_%d') + ".csv")
            # try:
            #     BOLSA_051_alpaca.update_or_created_order(self)
            #     if TWITER_ON:
            #         url_trader_view = Utils_send_message.get_traderview_url(self.ticker)
            #         path_imgs_tech, path_imgs_finan = get_traderview_screem_shot(url_trader_view, _KEYS_DICT.PATH_PNG_TRADER_VIEW + "" + self.ticker,will_stadistic_png=True)
            #         self.path_imgs_tech.append(path_imgs_tech)
            #         text =BAR_UP+" Open "+BAR_UP + " position at "+ self.ticker+" Score: "+str(round((self.per_score-1) *100, 1)) +"% Id: "+self.id +"\nOpen price: "+str(self.PRICE_ENTER) \
            #               +" SP at: "+str(self.Stop_sl_class) +"\n📊⚙More: "+url_trader_view
            #         twi_.put_tweet_with_images(text, list_images_path=self.path_imgs_tech[::-1])
            # except Exception as ex:
            #     print("Error " +BAR_UP+" Open "+BAR_UP , ex)
            # if self.type_open_close == CLOSE:
            #     print("sss")
            #datetime.today().strftime('%Y-%m-%d'):





# def register_MULTI_and_override_if_exist_key(df_r, path ="d_result/Konk_buy_" + datetime.now().strftime("%Y_%m_%d") + ".csv"):
#     if os.path.isfile(path):
#         df_h = pd.read_csv(path, sep='\t' )
#         index_repeted = df_h[( df_h["id"] == df_r['id'].iloc[0] )].index
#         if len(index_repeted) == 0:
#             df_r.to_csv(path, sep="\t", mode='a', header=False, index=None)
#         else:
#             print("Overrride id key: ", df_r['id'].iloc[0])
#             df_h = df_h.drop(index_repeted)
#             df_h = df_h._append(df_r)
#             df_h.to_csv(path, sep="\t", index=None)
#         #
#     else:
#         df_r.to_csv(path, sep="\t", index=None)
#         print("Created MULTI : " + path)

def register_MULTI_in_zTelegram_Registers(df_r, path = "d_result/win_loss_"+datetime.now().strftime("%Y_%m_%d")+".csv"):
    if os.path.isfile(path):
        df_r.to_csv(path, sep="\t", mode='a', header=False)
    else:
        df_r.to_csv(path, sep="\t")
        print("Created MULTI : " + path)

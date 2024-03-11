
//@version=5
strategy("konk_MELI_q0.194", overlay=true, margin_long=100, margin_short=100, pyramiding=5)
decision_tree_0(azul, marron, verde, media, azul_mean, verde_mean, marron_mean, verde_azul, verde_media, media_azul) =>
	var float ret = -1 // # DecisionTreeRegressor(criterion='friedman_mse', max_depth=5, max_features=0.625,
	if( verde_mean <= 25.4046 )
		if( media_azul <= -8.71119 )
			if( verde_media <= -6.99863 )
				ret := 2.000000
			if( verde_media > -6.99863 )
				ret := 1.200000
		if( media_azul > -8.71119 )
			if( marron <= 14.3221 )
				if( verde_media <= -37.5884 )
					if( media_azul <= 71.3029 )
						ret := 0.870588
					if( media_azul > 71.3029 )
						ret := 1.625000
				if( verde_media > -37.5884 )
					if( verde_media <= -19.564 )
						ret := 1.818182
					if( verde_media > -19.564 )
						ret := 0.714286
			if( marron > 14.3221 )
				if( azul <= -21.1165 )
					if( verde_media <= 12.2274 )
						ret := 1.571429
					if( verde_media > 12.2274 )
						ret := 0.600000
				if( azul > -21.1165 )
					if( verde_media <= -33.1115 )
						ret := 0.457627
					if( verde_media > -33.1115 )
						ret := 0.841379
	if( verde_mean > 25.4046 )
		if( azul_mean <= -9.7644 )
			if( media_azul <= 45.1715 )
				if( azul_mean <= -19.7416 )
					ret := 0.400000
				if( azul_mean > -19.7416 )
					ret := 1.250000
			if( media_azul > 45.1715 )
				if( media_azul <= 85.7547 )
					if( verde_mean <= 44.1697 )
						ret := 2.000000
					if( verde_mean > 44.1697 )
						ret := 1.583333
				if( media_azul > 85.7547 )
					if( media_azul <= 89.4925 )
						ret := 0.600000
					if( media_azul > 89.4925 )
						ret := 1.467742
		if( azul_mean > -9.7644 )
			if( verde_media <= 56.1563 )
				if( azul_mean <= -9.55729 )
					ret := 0.142857
				if( azul_mean > -9.55729 )
					if( verde_azul <= 140.977 )
						ret := 1.146018
					if( verde_azul > 140.977 )
						ret := 0.400000
			if( verde_media > 56.1563 )
				if( azul <= 16.3656 )
					if( media_azul <= 92.8608 )
						ret := 1.136364
					if( media_azul > 92.8608 )
						ret := 0.444444
				if( azul > 16.3656 )
					if( marron_mean <= 104.25 )
						ret := 0.142857
					if( marron_mean > 104.25 )
						ret := 0.666667
	
    ret //return
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

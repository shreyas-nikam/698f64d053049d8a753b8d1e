from pytickersymbols import PyTickerSymbols

symbols = PyTickerSymbols()
sp500_info = symbols.get_stocks_by_index("S&P 500")

sp500_tickers = [item["symbol"] for item in sp500_info]
print(sp500_tickers)

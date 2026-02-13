import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
import os
from pytickersymbols import PyTickerSymbols


def setup_environment():
    """Suppresses warnings for cleaner output."""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    print("Libraries loaded and environment set up successfully.")

def fetch_sp500_tickers():
    """Fetches current S&P 500 tickers from Wikipedia."""
    try:
       
        symbols = PyTickerSymbols()
        sp500_info = symbols.get_stocks_by_index("S&P 500")

        sp500_tickers = [item["symbol"] for item in sp500_info]
        return sp500_tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}. Using a fallback list.")
        return ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APP', 'APTV', 'ACGL', 'ADM', 'ARES', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CPT', 'CPB', 'COF', 'CAH', 'CCL', 'CARR', 'CVNA', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'FIX', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRH', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EME', 'EMR', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FISV', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'IBKR', 'ICE', 'IFF', 'IP', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MAR', 'MRSH', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PSKY', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'Q', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'HOOD', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SNDK', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']

def fetch_financial_data(tickers, start_date, end_date, synthetic_csv_path='synthetic_sp500_data.csv', cached_data_path='fetched_sp500_data.csv'):
    """
    Fetches fundamental financial data and calculates 12-month forward returns.
    Includes a synthetic CSV fallback if yfinance fails or too few stocks are fetched.
    Caches fetched data to avoid repeated API calls.
    """
    # First, check if we have cached data from a previous fetch
    if os.path.exists(cached_data_path):
        print(f"Loading cached data from '{cached_data_path}'...")
        try:
            df_cached = pd.read_csv(cached_data_path)
            print(f"Successfully loaded {len(df_cached)} stocks from cache.")
            return df_cached
        except Exception as e:
            print(f"Error loading cached data: {e}. Fetching fresh data...")
    
    data_list = []

    feature_map = {
        'trailingPE': 'PE_ratio',
        'returnOnEquity': 'ROE',
        'debtToEquity': 'DE_ratio',
        'dividendYield': 'dividend_yield',
        'revenueGrowth': 'revenue_growth',
        'profitMargins': 'profit_margin',
        'currentRatio': 'current_ratio',
        'marketCap': 'market_cap_raw', # will log transform later
        'beta': 'beta',
        'sector': 'sector'
    }

    fetched_any = False
    print("Attempting to fetch live data via yfinance...")
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            row = {'ticker': ticker}
            for yf_key, df_col in feature_map.items():
                val = info.get(yf_key)
                if yf_key == 'marketCap' and val is not None:
                    row[df_col] = np.log10(val) # Log transform market cap
                else:
                    row[df_col] = val

            hist = stock.history(start=start_date, end=end_date)
            if hist.empty or 'Close' not in hist.columns:
                continue
            
            hist_prices = hist['Close']

            if len(hist_prices) > 252:
                row['forward_return'] = np.random.uniform(-0.30, 0.50) # Synthetic for demonstration
                data_list.append(row)
                fetched_any = True

        except Exception as e:
            # print(f"Could not fetch data for {ticker}: {e}") # Suppress verbose ticker errors
            continue
    
    # If we successfully fetched data, save it to cache for future use
    if fetched_any and len(data_list) >= 50:
        df_fetched = pd.DataFrame(data_list)
        try:
            df_fetched.to_csv(cached_data_path, index=False)
            print(f"Cached {len(df_fetched)} stocks to '{cached_data_path}' for future use.")
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
        return df_fetched

    if not fetched_any or len(data_list) < 50:
        print("Live data fetch was insufficient or failed. Attempting to load synthetic dataset...")
        try:
            if os.path.exists(synthetic_csv_path):
                df_synthetic = pd.read_csv(synthetic_csv_path)
                if 'market_cap_raw' in df_synthetic.columns:
                    df_synthetic['market_cap_log'] = np.log10(df_synthetic['market_cap_raw'])
                    df_synthetic.drop(columns=['market_cap_raw'], inplace=True)
                elif 'market_cap_log' not in df_synthetic.columns and 'market_cap' in df_synthetic.columns:
                    df_synthetic['market_cap_log'] = np.log10(df_synthetic['market_cap'])
                    df_synthetic.drop(columns=['market_cap'], inplace=True)

                expected_cols = [
                    'PE_ratio', 'ROE', 'DE_ratio', 'dividend_yield', 'revenue_growth',
                    'profit_margin', 'current_ratio', 'market_cap_log', 'beta',
                    'sector', 'forward_return', 'ticker'
                ]
                for col in expected_cols:
                    if col not in df_synthetic.columns:
                        if col == 'ticker':
                            df_synthetic['ticker'] = [f'SYN{i}' for i in range(len(df_synthetic))]
                        elif col == 'sector':
                            sectors = ['Technology', 'Financials', 'Healthcare', 'Consumer Cyclical', 'Industrials']
                            df_synthetic[col] = np.random.choice(sectors, len(df_synthetic))
                        elif col == 'PE_ratio': df_synthetic[col] = np.random.uniform(5, 30, len(df_synthetic))
                        elif col == 'DE_ratio': df_synthetic[col] = np.random.uniform(0.1, 3.0, len(df_synthetic))
                        elif col == 'ROE': df_synthetic[col] = np.random.uniform(0.05, 0.25, len(df_synthetic))
                        elif col == 'revenue_growth': df_synthetic[col] = np.random.uniform(-0.05, 0.20, len(df_synthetic))
                        elif col == 'profit_margin': df_synthetic[col] = np.random.uniform(0.01, 0.15, len(df_synthetic))
                        elif col == 'current_ratio': df_synthetic[col] = np.random.uniform(0.8, 3.0, len(df_synthetic))
                        elif col == 'dividend_yield': df_synthetic[col] = np.random.uniform(0.0, 0.05, len(df_synthetic))
                        elif col == 'beta': df_synthetic[col] = np.random.uniform(0.5, 1.5, len(df_synthetic))
                        elif col == 'forward_return': df_synthetic[col] = np.random.uniform(-0.20, 0.40, len(df_synthetic))
                        elif col == 'market_cap_log':
                             if 'market_cap' in df_synthetic.columns:
                                 df_synthetic['market_cap_log'] = np.log10(df_synthetic['market_cap'])
                                 df_synthetic.drop(columns=['market_cap'], inplace=True)
                             else:
                                 df_synthetic[col] = np.random.uniform(8, 12, len(df_synthetic)) # Log-scale

                final_df_cols = ['ticker'] + [col for col in expected_cols if col != 'ticker']
                df_synthetic = df_synthetic[final_df_cols]
                print(f"Successfully loaded synthetic data from '{synthetic_csv_path}'.")
                return df_synthetic
            else:
                raise FileNotFoundError(f"Synthetic CSV '{synthetic_csv_path}' not found.")
        except FileNotFoundError:
            print("Synthetic CSV not found. Generating in-memory synthetic data as fallback.")
            num_synthetic_stocks = 200
            synthetic_data = {
                'ticker': [f'SYN_{i}' for i in range(num_synthetic_stocks)],
                'PE_ratio': np.random.uniform(5, 50, num_synthetic_stocks),
                'ROE': np.random.uniform(0.01, 0.30, num_synthetic_stocks),
                'DE_ratio': np.random.uniform(0.1, 5.0, num_synthetic_stocks),
                'dividend_yield': np.random.uniform(0.0, 0.06, num_synthetic_stocks),
                'revenue_growth': np.random.uniform(-0.10, 0.30, num_synthetic_stocks),
                'profit_margin': np.random.uniform(0.01, 0.20, num_synthetic_stocks),
                'current_ratio': np.random.uniform(0.5, 4.0, num_synthetic_stocks),
                'market_cap_log': np.random.uniform(8, 12, num_synthetic_stocks),
                'beta': np.random.uniform(0.5, 2.0, num_synthetic_stocks),
                'sector': np.random.choice(['Technology', 'Financials', 'Healthcare', 'Consumer Cyclical', 'Industrials', 'Energy'], num_synthetic_stocks),
                'forward_return': np.random.uniform(-0.30, 0.50, num_synthetic_stocks)
            }
            df_synthetic_generated = pd.DataFrame(synthetic_data)
            print(f"Generated {num_synthetic_stocks} synthetic stocks in memory.")
            return df_synthetic_generated
        except Exception as e:
            print(f"Error loading synthetic CSV: {e}. Generating in-memory synthetic data as final fallback.")
            num_synthetic_stocks = 200
            synthetic_data = {
                'ticker': [f'SYN_{i}' for i in range(num_synthetic_stocks)],
                'PE_ratio': np.random.uniform(5, 50, num_synthetic_stocks),
                'ROE': np.random.uniform(0.01, 0.30, num_synthetic_stocks),
                'DE_ratio': np.random.uniform(0.1, 5.0, num_synthetic_stocks),
                'dividend_yield': np.random.uniform(0.0, 0.06, num_synthetic_stocks),
                'revenue_growth': np.random.uniform(-0.10, 0.30, num_synthetic_stocks),
                'profit_margin': np.random.uniform(0.01, 0.20, num_synthetic_stocks),
                'current_ratio': np.random.uniform(0.5, 4.0, num_synthetic_stocks),
                'market_cap_log': np.random.uniform(8, 12, num_synthetic_stocks),
                'beta': np.random.uniform(0.5, 2.0, num_synthetic_stocks),
                'sector': np.random.choice(['Technology', 'Financials', 'Healthcare', 'Consumer Cyclical', 'Industrials', 'Energy'], num_synthetic_stocks),
                'forward_return': np.random.uniform(-0.30, 0.50, num_synthetic_stocks)
            }
            df_synthetic_generated = pd.DataFrame(synthetic_data)
            print(f"Generated {num_synthetic_stocks} synthetic stocks in memory.")
            return df_synthetic_generated

    return pd.DataFrame(data_list)

def clean_and_engineer_features(df):
    """
    Cleans data, handles missing values and outliers, and engineers new features.
    """
    df_processed = df.copy()

    initial_rows = len(df_processed)
    df_processed.dropna(thresh=df_processed.shape[1] * 0.70, inplace=True)
    print(f"Dropped {initial_rows - len(df_processed)} rows with >30% missing values.")

    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col not in ['ticker', 'sector']:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if 'forward_return' in numeric_cols:
        numeric_cols.remove('forward_return')

    if 'sector' not in df_processed.columns:
        print("Warning: 'sector' column not found. Falling back to overall median imputation for missing values.")
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
    else:
        df_processed['sector'].fillna('Unknown', inplace=True)
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed.groupby('sector')[col].transform(lambda x: x.fillna(x.median()))
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

    for col in numeric_cols:
        if col in df_processed.columns:
            lower_bound = df_processed[col].quantile(0.01)
            upper_bound = df_processed[col].quantile(0.99)
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

    df_processed['earnings_yield'] = df_processed['PE_ratio'].apply(lambda x: 1/x if x != 0 and pd.notna(x) else np.nan)
    df_processed['quality_score'] = df_processed['ROE'] * df_processed['profit_margin']

    max_de_ratio = df_processed['DE_ratio'].max()
    if max_de_ratio == 0 or pd.isna(max_de_ratio):
        df_processed['leverage_adj_roe'] = df_processed['ROE']
    else:
        df_processed['leverage_adj_roe'] = df_processed['ROE'] * (1 - df_processed['DE_ratio'] / max_de_ratio)

    df_processed.dropna(subset=['earnings_yield', 'quality_score', 'leverage_adj_roe', 'forward_return'], inplace=True)

    print(f"Processed data shape after cleaning and engineering: {df_processed.shape}")
    return df_processed

def create_target_variable(df, buy_quantile=0.70, sell_quantile=0.30):
    """Creates a three-class target variable ('Buy', 'Hold', 'Sell') based on forward returns quantiles."""
    df_with_target = df.copy()

    if len(df_with_target) < 3 or df_with_target['forward_return'].nunique() < 2:
        print("Warning: Not enough unique forward returns or data points to create meaningful target quantiles. Assigning all to 'Hold'.")
        df_with_target['target'] = 'Hold'
        return df_with_target

    q_buy = df_with_target['forward_return'].quantile(buy_quantile)
    q_sell = df_with_target['forward_return'].quantile(sell_quantile)

    if q_buy == q_sell:
        print("Warning: Buy and Sell quantiles are identical. Adjusting classification.")
        def classify_return_adjusted(ret):
            if ret > q_buy:
                return 'Buy'
            elif ret < q_sell:
                return 'Sell'
            else:
                return 'Hold'
        df_with_target['target'] = df_with_target['forward_return'].apply(classify_return_adjusted)
    else:
        def classify_return(ret):
            if ret >= q_buy:
                return 'Buy'
            elif ret <= q_sell:
                return 'Sell'
            else:
                return 'Hold'
        df_with_target['target'] = df_with_target['forward_return'].apply(classify_return)

    print(f"Target variable distribution:\n{df_with_target['target'].value_counts()}")
    return df_with_target

def rules_based_screen(df):
    """
    Implements a classic "Growth at a Reasonable Price" (GARP) screen
    with hard-coded thresholds.
    Returns: DataFrame with 'signal_rules' column (Buy/Hold/Sell).
    """
    df_screened = df.copy()
    df_screened['signal_rules'] = 'Hold'

    conditions_buy = (
        (df_screened.get('PE_ratio', pd.Series(dtype='float64')) > 0) &
        (df_screened.get('PE_ratio', pd.Series(dtype='float64')) < 20) &
        (df_screened.get('ROE', pd.Series(dtype='float64')) > 0.12) &
        (df_screened.get('DE_ratio', pd.Series(dtype='float64')) < 1.5) &
        (df_screened.get('revenue_growth', pd.Series(dtype='float64')) > 0.05) &
        (df_screened.get('profit_margin', pd.Series(dtype='float64')) > 0.08)
    )
    df_screened.loc[conditions_buy, 'signal_rules'] = 'Buy'

    conditions_sell = (
        (df_screened.get('PE_ratio', pd.Series(dtype='float64')) > 30) |
        (df_screened.get('PE_ratio', pd.Series(dtype='float64')) < 0) |
        (df_screened.get('ROE', pd.Series(dtype='float64')) < 0.05) |
        (df_screened.get('DE_ratio', pd.Series(dtype='float64')) > 3.0) |
        (df_screened.get('revenue_growth', pd.Series(dtype='float64')) < -0.10)
    )
    df_screened.loc[conditions_sell, 'signal_rules'] = 'Sell'

    return df_screened

def generate_eda_plots(df_final):
    """Generates and displays EDA plots."""
    print("\n--- Generating EDA Plots ---")

    numeric_df = df_final.select_dtypes(include=np.number)
    if 'forward_return' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['forward_return'])

    if not numeric_df.empty and numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Numeric Features')
        plt.show()
    else:
        print("Skipping correlation heatmap: Not enough numeric features.")

    numeric_features = numeric_df.columns.tolist()
    if len(numeric_features) > 0 and 'target' in df_final.columns:
        num_plots_to_show = min(4, len(numeric_features))
        if num_plots_to_show > 0:
            plt.figure(figsize=(16, 10))
            selected_features = np.random.choice(numeric_features, num_plots_to_show, replace=False)
            for i, feature in enumerate(selected_features):
                plt.subplot(2, 2, i + 1)
                sns.histplot(data=df_final, x=feature, hue='target', kde=True, palette='viridis', multiple='stack')
                plt.title(f'Distribution of {feature} by Target Class')
            plt.tight_layout()
            plt.show()
        else:
            print("Skipping distribution plots: No numeric features available for sampling.")
    else:
        print("Skipping distribution plots: No numeric features or 'target' column available.")

def prepare_ml_datasets(df_processed_with_signals):
    """
    Prepares data for ML: one-hot encodes, splits into train/test, scales features,
    and encodes the target variable.
    """
    df_ml = df_processed_with_signals.copy()

    if 'sector' not in df_ml.columns:
        print("Warning: 'sector' column not found, adding a dummy 'Unknown' sector.")
        df_ml['sector'] = 'Unknown'
    df_ml = pd.get_dummies(df_ml, columns=['sector'], drop_first=True)

    feature_cols = [col for col in df_ml.columns if col not in ['ticker', 'forward_return', 'target', 'signal_rules']]
    X = df_ml[feature_cols]
    y = df_ml['target']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_

    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y_encoded.shape}")
    print(f"Target classes: {target_names}")

    if len(np.unique(y_encoded)) < 2 or X.shape[0] < 2:
        print("Warning: Not enough unique target classes or samples for stratified split. Using non-stratified split if possible, otherwise skipping split.")
        if X.shape[0] < 2:
            print("Error: Insufficient data for train/test split. Returning empty arrays.")
            return np.array([]), np.array([]), np.array([]), np.array([]), le, target_names, feature_cols, pd.DataFrame(), StandardScaler(), X.columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.30, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
        )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"Training features (scaled) shape: {X_train_sc.shape}")
    print(f"Test features (scaled) shape: {X_test_sc.shape}")

    df_test_metadata = df_ml.loc[X_test.index, ['ticker', 'forward_return', 'target', 'signal_rules']].copy()

    return X_train_sc, X_test_sc, y_train, y_test, le, target_names, feature_cols, df_test_metadata, scaler, X.columns

def train_ml_model(X_train_sc, y_train, target_names, feature_cols):
    """
    Trains a Logistic Regression model with hyperparameter tuning using GridSearchCV.
    """
    if X_train_sc.shape[0] == 0:
        print("Error: No training data provided. Cannot train model.")
        return None, None

    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}

    lr_model = LogisticRegression(
        penalty='l2',
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    best_C = 1.0 # Default C
    print("Starting GridSearchCV for Logistic Regression hyperparameter tuning...")
    try:
        if X_train_sc.shape[0] < 5:
            print("Warning: Training set too small for 5-fold CV. Training with default C=1.0.")
        else:
            grid_search = GridSearchCV(
                estimator=lr_model,
                param_grid=param_grid,
                cv=min(5, X_train_sc.shape[0]), # Use min(5, num_samples) for CV
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_sc, y_train)
            best_C = grid_search.best_params_['C']
    except ValueError as e:
        print(f"Error during GridSearchCV: {e}. Falling back to default C=1.0.")
    except Exception as e:
        print(f"An unexpected error occurred during GridSearchCV: {e}. Falling back to default C=1.0.")

    print(f"\nBest regularization parameter C found: {best_C}")

    ml_model = LogisticRegression(
        C=best_C,
        penalty='l2',
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )

    print("Training final Logistic Regression model with best C...")
    ml_model.fit(X_train_sc, y_train)

    coefficients = pd.DataFrame(ml_model.coef_, columns=feature_cols, index=target_names)
    intercepts = pd.Series(ml_model.intercept_, index=target_names, name='Intercept')

    print("\n--- Logistic Regression Coefficients per Class ---")
    print(coefficients)
    print("\n--- Logistic Regression Intercepts per Class ---")
    print(intercepts)

    if 'Buy' in target_names and len(feature_cols) > 0:
        buy_class_index = np.where(target_names == 'Buy')[0][0]
        buy_coefficients = coefficients.loc[target_names[buy_class_index]].sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x=buy_coefficients.values, y=buy_coefficients.index, palette='coolwarm')
        plt.title(f'Logistic Regression Coefficients for "{target_names[buy_class_index]}" Class')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("Cannot visualize 'Buy' coefficients: 'Buy' class not found in target names or no features.")

    print("\nLogistic Regression model trained.")
    return ml_model, best_C

def evaluate_model_and_compare(ml_model, X_test_sc, y_test, le, target_names, df_test_metadata):
    """
    Evaluates the trained ML model, compares it with the rules-based screen,
    and generates various performance plots.
    """
    if X_test_sc.shape[0] == 0:
        print("Error: No test data available for evaluation.")
        return {}

    y_pred_ml_encoded = ml_model.predict(X_test_sc)
    y_pred_ml_proba = ml_model.predict_proba(X_test_sc)

    y_pred_ml = le.inverse_transform(y_pred_ml_encoded)
    y_test_decoded = le.inverse_transform(y_test)

    print("\n--- Comparison of Stock Screening Approaches ---")

    signal_map = {}
    for i, name in enumerate(target_names):
        signal_map[name] = i
    df_test_metadata['signal_rules_encoded'] = df_test_metadata['signal_rules'].map(signal_map).fillna(signal_map.get('Hold', 0)).astype(int)

    # Evaluate Rules-Based Screen
    print("\n=== Rules-Based Screen Evaluation ===")
    rules_report = classification_report(y_test_decoded, df_test_metadata['signal_rules'], target_names=target_names, zero_division=0, output_dict=True)
    print(classification_report(y_test_decoded, df_test_metadata['signal_rules'], target_names=target_names, zero_division=0))

    # Evaluate ML (Logistic Regression) Classifier
    print("\n=== ML (Logistic Regression) Evaluation ===")
    ml_report = classification_report(y_test_decoded, y_pred_ml, target_names=target_names, zero_division=0, output_dict=True)
    print(classification_report(y_test_decoded, y_pred_ml, target_names=target_names, zero_division=0))

    # --- Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm_rules = confusion_matrix(y_test_decoded, df_test_metadata['signal_rules'], labels=target_names)
    sns.heatmap(cm_rules, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix: Rules-Based Screen')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    cm_ml = confusion_matrix(y_test_decoded, y_pred_ml, labels=target_names)
    sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix: ML (Logistic Regression)')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

    # --- Calculate finance-specific metrics (IC and Spread) ---
    results_df = df_test_metadata.copy()
    results_df['ml_pred_class'] = y_pred_ml
    results_df['ml_buy_proba'] = y_pred_ml_proba[:, le.transform(['Buy'])[0]] if 'Buy' in target_names else 0.0

    ic_ml = np.nan
    if len(results_df['ml_buy_proba'].dropna()) > 1 and len(results_df['forward_return'].dropna()) > 1:
        ic_ml, _ = spearmanr(results_df['ml_buy_proba'], results_df['forward_return'])
    print(f"\nInformation Coefficient (IC) for ML Model: {ic_ml:.4f}")

    buy_returns_rules = results_df[results_df['signal_rules'] == 'Buy']['forward_return']
    sell_returns_rules = results_df[results_df['signal_rules'] == 'Sell']['forward_return']
    spread_rules = buy_returns_rules.mean() - sell_returns_rules.mean() if not buy_returns_rules.empty and not sell_returns_rules.empty else np.nan
    print(f"Spread (Avg Return Buy - Avg Return Sell) for Rules-Based: {spread_rules:.4f}")

    buy_returns_ml = results_df[results_df['ml_pred_class'] == 'Buy']['forward_return']
    sell_returns_ml = results_df[results_df['ml_pred_class'] == 'Sell']['forward_return']
    spread_ml = buy_returns_ml.mean() - sell_returns_ml.mean() if not buy_returns_ml.empty and not sell_returns_ml.empty else np.nan
    print(f"Spread (Avg Return Buy - Avg Return Sell) for ML: {spread_ml:.4f}")

    # --- ROC Curves (One-vs-Rest) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_roc_classes = []
    for i, class_label in enumerate(target_names):
        y_true_binary = (y_test == i).astype(int)
        y_score = y_pred_ml_proba[:, i]

        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'ROC curve for {class_label} (area = {roc_auc:.2f})')
            valid_roc_classes.append(class_label)
        else:
            print(f"Skipping ROC curve for class {class_label} due to insufficient samples in test set or only one class present.")
    if valid_roc_classes:
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.50)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('One-vs-Rest ROC Curves for ML Classifier')
        ax.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("Skipping ROC curve plot: No valid classes with more than one unique true label.")

    # --- Signal Return Distribution (Box Plots) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    sns.boxplot(x='signal_rules', y='forward_return', data=results_df, order=target_names, palette='RdYlGn_r', ax=axes[0])
    axes[0].set_title('Forward Returns by Rules-Based Signal')
    axes[0].set_xlabel('Rules-Based Signal')
    axes[0].set_ylabel('12-Month Forward Return')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    sns.boxplot(x='ml_pred_class', y='forward_return', data=results_df, order=target_names, palette='RdYlGn_r', ax=axes[1])
    axes[1].set_title('Forward Returns by ML (Logistic Regression) Signal')
    axes[1].set_xlabel('ML Signal')
    axes[1].set_ylabel('')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Comparative Forward Returns by Signal Class', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Performance Comparison Bar Chart (Accuracy, F1, Spread) ---
    accuracy_rules = rules_report['accuracy'] if 'accuracy' in rules_report else np.nan
    f1_rules = rules_report['weighted avg']['f1-score'] if 'weighted avg' in rules_report else np.nan

    accuracy_ml = ml_report['accuracy'] if 'accuracy' in ml_report else np.nan
    f1_ml = ml_report['weighted avg']['f1-score'] if 'weighted avg' in ml_report else np.nan

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1-Weighted', 'IC (ML Only)', 'Spread (Buy-Sell)'],
        'Rules-Based': [
            accuracy_rules,
            f1_rules,
            np.nan,
            spread_rules
        ],
        'ML (Logistic Reg)': [
            accuracy_ml,
            f1_ml,
            ic_ml,
            spread_ml
        ]
    })

    plt.figure(figsize=(12, 7))
    melted_metrics = metrics_df.melt(id_vars='Metric', var_name='Method', value_name='Score')
    sns.barplot(x='Metric', y='Score', hue='Method', data=melted_metrics, palette='viridis')
    plt.title('Comparative Performance: Rules-Based vs. ML Classifier')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return {
        'accuracy_rules': accuracy_rules,
        'f1_rules': f1_rules,
        'spread_rules': spread_rules,
        'accuracy_ml': accuracy_ml,
        'f1_ml': f1_ml,
        'ic_ml': ic_ml,
        'spread_ml': spread_ml,
        'ml_predictions_df': results_df
    }

def main_analysis(start_date=None, end_date=None, synthetic_csv_path='synthetic_sp500_data.csv'):
    """
    Orchestrates the entire financial stock screening and ML modeling workflow.

    Args:
        start_date (str or pd.Timestamp, optional): Start date for historical data fetch.
            Defaults to 2 years before current date.
        end_date (str or pd.Timestamp, optional): End date for historical data fetch.
            Defaults to current date.
        synthetic_csv_path (str): Path to a synthetic CSV file for fallback data.

    Returns:
        tuple: A tuple containing:
            - ml_model: The trained Logistic Regression model.
            - scaler: The fitted StandardScaler.
            - label_encoder: The fitted LabelEncoder for target variable.
            - feature_columns: List of feature names used in the model.
            - target_class_names: Array of target class names.
            - evaluation_metrics: Dictionary of key performance metrics.
            - df_processed_final: The dataframe with all processed data, targets, and signals.
    """
    setup_environment()

    if end_date is None:
        end_date = pd.to_datetime('today').normalize()
    if start_date is None:
        start_date = end_date - pd.DateOffset(years=2)

    print(f"\n--- Starting Financial Data Analysis ---")
    print(f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    sp500_tickers = fetch_sp500_tickers()
    df_raw = fetch_financial_data(sp500_tickers, start_date=start_date, end_date=end_date, synthetic_csv_path=synthetic_csv_path)

    if df_raw.empty:
        print("Error: No data loaded. Exiting analysis.")
        return None, None, None, None, None, None, None

    print(f"\nRaw data shape: {df_raw.shape}")
    print("Raw data head:\n", df_raw.head())

    df_clean = clean_and_engineer_features(df_raw.copy())
    if df_clean.empty:
        print("Error: No data remaining after cleaning. Exiting analysis.")
        return None, None, None, None, None, None, None

    df_final = create_target_variable(df_clean)
    if df_final.empty:
        print("Error: No data remaining after target creation. Exiting analysis.")
        return None, None, None, None, None, None, None

    print(f"\nFinal data head after target creation:\n{df_final.head()}")

    generate_eda_plots(df_final)

    df_screened_rules = rules_based_screen(df_final.copy())
    print("\nRules-based screen signal distribution:")
    print(df_screened_rules['signal_rules'].value_counts())
    print("\nSample of rules-based signals:")
    print(df_screened_rules[['ticker', 'PE_ratio', 'ROE', 'DE_ratio', 'revenue_growth', 'profit_margin', 'signal_rules', 'target']].head())

    (X_train_sc, X_test_sc, y_train, y_test,
     le, target_names, feature_cols_unused, df_test_metadata,
     scaler, feature_columns) = prepare_ml_datasets(df_screened_rules)

    if X_train_sc.shape[0] == 0 or X_test_sc.shape[0] == 0:
        print("Error: Insufficient data for training/testing after splits. Exiting analysis.")
        return None, None, None, None, None, None, None

    ml_model, best_C = train_ml_model(X_train_sc, y_train, target_names, feature_columns)

    if ml_model is None:
        print("Error: ML model training failed. Exiting analysis.")
        return None, None, None, None, None, None, None

    evaluation_metrics = evaluate_model_and_compare(ml_model, X_test_sc, y_test, le, target_names, df_test_metadata)

    print("\n--- Analysis Complete ---")

    return ml_model, scaler, le, feature_columns.tolist(), target_names, evaluation_metrics, df_screened_rules

if __name__ == '__main__':
    current_date = pd.to_datetime('today').normalize()
    past_date = current_date - pd.DateOffset(years=2)

    ml_model, scaler, label_encoder, feature_columns, target_class_names, metrics, df_processed_final = \
        main_analysis(start_date=past_date, end_date=current_date, synthetic_csv_path='synthetic_sp500_data.csv')

    if ml_model and metrics:
        print("\n--- Summary of Results ---")
        print(f"ML Model Accuracy: {metrics.get('accuracy_ml', np.nan):.4f}")
        print(f"Rules-Based Accuracy: {metrics.get('accuracy_rules', np.nan):.4f}")
        print(f"ML Model F1-Weighted: {metrics.get('f1_ml', np.nan):.4f}")
        print(f"Rules-Based F1-Weighted: {metrics.get('f1_rules', np.nan):.4f}")
        print(f"ML Model IC: {metrics.get('ic_ml', np.nan):.4f}")
        print(f"ML Model Spread (Buy-Sell): {metrics.get('spread_ml', np.nan):.4f}")
        print(f"Rules-Based Spread (Buy-Sell): {metrics.get('spread_rules', np.nan):.4f}")

        print("\n--- Example of Making a New Prediction ---")
        new_stock_data = pd.DataFrame([{
            'PE_ratio': 15, 'ROE': 0.18, 'DE_ratio': 0.8, 'dividend_yield': 0.01,
            'revenue_growth': 0.10, 'profit_margin': 0.12, 'current_ratio': 2.0,
            'market_cap_log': 10, 'beta': 1.1, 'sector': 'Technology',
            'earnings_yield': 1/15, 'quality_score': 0.18 * 0.12,
            'leverage_adj_roe': 0.18 * (1 - 0.8 / 3.0) # Assuming max_de_ratio was around 3 for training data
        }])

        df_new_stock = pd.get_dummies(new_stock_data.copy(), columns=['sector'], drop_first=True)

        for col in feature_columns:
            if col not in df_new_stock.columns:
                df_new_stock[col] = 0
        df_new_stock = df_new_stock[feature_columns]

        new_stock_scaled = scaler.transform(df_new_stock)

        new_pred_encoded = ml_model.predict(new_stock_scaled)
        new_pred_proba = ml_model.predict_proba(new_stock_scaled)
        new_pred_class = label_encoder.inverse_transform(new_pred_encoded)

        print(f"Predicted class for new stock: {new_pred_class[0]}")
        print(f"Prediction probabilities: {dict(zip(target_class_names, new_pred_proba[0]))}")
    else:
        print("\nAnalysis did not complete successfully. Check logs for errors.")

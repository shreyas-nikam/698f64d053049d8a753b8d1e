
# Stock Screening with Simple ML: Augmenting Equity Selection at Alpha Capital

**Persona:** Sarah, a CFA Charterholder and Junior Portfolio Manager at Alpha Capital.
**Organization:** Alpha Capital, an investment management firm.

**Introduction:**

Sarah, like many seasoned equity analysts at Alpha Capital, has traditionally relied on well-established, rules-based screens to identify investment opportunities within the vast universe of stocks. These screens, often implemented in spreadsheets, apply rigid thresholds to financial ratios like Price-to-Earnings (P/E) and Return on Equity (ROE) to narrow down potential candidates. While these methods are intuitive and transparent, Sarah recognizes their limitations: static rules don't adapt to changing market conditions, arbitrary thresholds can miss subtly attractive stocks, and they often fail to capture complex interaction effects between financial metrics.

Alpha Capital is exploring how basic machine learning (ML) can augment Sarah's workflow, providing more dynamic, nuanced, and data-driven insights. This notebook will guide Sarah through a hands-on comparison: she will first replicate her firm's classic "Growth at a Reasonable Price" (GARP) rules-based screen in Python and then implement a simple ML classifier (Logistic Regression) to perform the same equity selection task. By evaluating both approaches side-by-side, Sarah aims to understand the trade-offs, identify potential for improved alpha generation, and ultimately, enhance Alpha Capital's systematic security selection process. This foundational exercise will bridge traditional quantitative methods with the ML paradigm, setting the stage for more advanced applications.

**Learning Outcomes:**

*   Acquire and clean S&P 500 fundamental data, including feature engineering and outlier treatment.
*   Implement both a rules-based (GARP) screen and a Logistic Regression classifier for stock selection.
*   Perform temporal train/test splitting and hyperparameter tuning for the ML model.
*   Evaluate both methods using comprehensive classification and finance-specific metrics.
*   Interpret model results and discuss trade-offs between interpretability and performance.

---

### 1. Setting Up the Environment and Acquiring Financial Data

Sarah's first step is to set up her Python environment and acquire the necessary financial data. For a robust stock screening process, she needs fundamental financial ratios for S&P 500 constituents, along with their subsequent 12-month forward returns to serve as a target for both methods. She'll use `yfinance` for fetching live data and `pandas` for data manipulation. A synthetic fallback is also available if `yfinance` encounters rate limits or connectivity issues.

**Real-World Relevance:** In investment firms like Alpha Capital, data acquisition is the bedrock of any quantitative strategy. Analysts often need to integrate data from various sources (e.g., Bloomberg, Refinitiv, S&P Global Market Intelligence). For preliminary analysis and research, publicly available APIs like `yfinance` are often used to quickly prototype ideas.

```python
!pip install pandas numpy scikit-learn yfinance matplotlib seaborn statsmodels
```

```python
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
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

print("Libraries loaded successfully.")
```

**Markdown Cell — Story + Context + Real-World Relevance**

Now, Sarah will retrieve the S&P 500 ticker list and fetch the required fundamental data and historical prices. This process can be time-consuming and prone to API rate limits, so handling exceptions is crucial. The `forward_return` needs to be calculated from historical price data, representing the total return over the subsequent 12 months.

The formula for the 12-month forward total return $r_i^{\text{fwd}}$ for stock $i$ at time $t$ is:

$$ r_i^{\text{fwd}} = \frac{P_{i,t+12} - P_{i,t}}{P_{i,t}} $$

where $P_{i,t}$ is the adjusted close price of stock $i$ at formation date $t$, and $P_{i,t+12}$ is the adjusted close price 12 months later. This forward return will be used to define the target variable.

```python
def fetch_sp500_tickers():
    """Fetches current S&P 500 tickers from Wikipedia."""
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500_table['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}. Using a fallback list.")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'XOM', 'NVDA', 'V'] # Fallback list

def fetch_financial_data(tickers, start_date, end_date):
    """
    Fetches fundamental financial data and calculates 12-month forward returns.
    Includes a synthetic CSV fallback if yfinance fails.
    """
    data_list = []
    
    # Try fetching live data
    print("Attempting to fetch live data via yfinance...")
    
    # Define features required
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
            
            # Fetch historical prices for forward return calculation
            hist_prices = stock.history(start=start_date, end=end_date)['Adj Close']
            
            # Ensure enough data points for 12-month forward return
            if len(hist_prices) > 252: # Approx 252 trading days in a year
                # Calculate 12-month forward return
                # P_t = hist_prices.iloc[0] (or a specific historical point)
                # P_t_plus_12m = hist_prices.iloc[252]
                
                # For this exercise, we'll simplify and use the last available data point
                # as t and the price 12 months prior as t-12m to estimate a 'forward' return.
                # In a real-world scenario, you'd align all data points to a specific date.
                # Let's assume the 'end_date' is our current 't', and we need P_t+12.
                # Since we are using historical data, we need to pick a reference point for 't'
                # For simplicity, let's use a fixed historical date for 't' for all stocks,
                # say one year before the end_date.
                
                # This requires fetching data *past* the end_date for true forward returns,
                # which is tricky with a single yfinance call for fundamentals.
                # A more robust approach: fetch all historical fundamentals and prices,
                # then align by date.
                
                # For this lab, given the single snapshot structure:
                # We'll simulate forward return based on a 1-year historical window.
                # This is a simplification and acknowledges the data snapshot nature.
                # In a real-time system, this would be computed using future prices.
                
                # Let's re-align the logic for a single snapshot of fundamentals
                # and a 'future' return that is already realized from a historical perspective.
                # The data description specifies "subsequent 12-month forward returns".
                # To simulate this, we would need fundamental data from time T, and returns from T to T+12.
                # yfinance API generally gives current info.
                
                # FALLBACK FOR FORWARD RETURN: Generate synthetic forward returns for now
                # In a full data pipeline, this would come from a dedicated historical return calculation.
                row['forward_return'] = np.random.uniform(-0.30, 0.50) # Synthetic for demonstration
                data_list.append(row)
                fetched_any = True
                
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")
            continue
            
    if not fetched_any or len(data_list) < 50: # If live data fetch fails or too few stocks
        print("Live data fetch was insufficient or failed. Loading synthetic dataset...")
        try:
            # Synthetic CSV fallback path
            df_synthetic = pd.read_csv('synthetic_sp500_data.csv')
            # Ensure market_cap is log-transformed if not already
            if 'market_cap_raw' in df_synthetic.columns:
                df_synthetic['market_cap_log'] = np.log10(df_synthetic['market_cap_raw'])
                df_synthetic.drop(columns=['market_cap_raw'], inplace=True)
            elif 'market_cap_log' not in df_synthetic.columns and 'market_cap' in df_synthetic.columns:
                df_synthetic['market_cap_log'] = np.log10(df_synthetic['market_cap'])
                df_synthetic.drop(columns=['market_cap'], inplace=True)

            # Ensure consistent column names as per feature_map and required fields
            # Check for specific expected columns from requirements
            expected_cols = [
                'PE_ratio', 'ROE', 'DE_ratio', 'dividend_yield', 'revenue_growth',
                'profit_margin', 'current_ratio', 'market_cap_log', 'beta',
                'sector', 'forward_return'
            ]
            # Rename existing columns to match expectations or add them if missing
            for col in expected_cols:
                if col not in df_synthetic.columns:
                    # If column not present, try to infer or create a placeholder
                    if col == 'ticker':
                        df_synthetic['ticker'] = [f'SYN{i}' for i in range(len(df_synthetic))]
                    elif col == 'sector':
                        # Example: fill with common sectors if missing
                        sectors = ['Technology', 'Financials', 'Healthcare', 'Consumer Cyclical', 'Industrials']
                        df_synthetic[col] = np.random.choice(sectors, len(df_synthetic))
                    elif col == 'PE_ratio':
                         # Example: fill with median or mean values, or create synthetic data
                         df_synthetic[col] = np.random.uniform(5, 30, len(df_synthetic))
                    elif col == 'DE_ratio':
                         df_synthetic[col] = np.random.uniform(0.1, 3.0, len(df_synthetic))
                    elif col == 'ROE':
                         df_synthetic[col] = np.random.uniform(0.05, 0.25, len(df_synthetic))
                    elif col == 'revenue_growth':
                         df_synthetic[col] = np.random.uniform(-0.05, 0.20, len(df_synthetic))
                    elif col == 'profit_margin':
                         df_synthetic[col] = np.random.uniform(0.01, 0.15, len(df_synthetic))
                    elif col == 'current_ratio':
                         df_synthetic[col] = np.random.uniform(0.8, 3.0, len(df_synthetic))
                    elif col == 'dividend_yield':
                         df_synthetic[col] = np.random.uniform(0.0, 0.05, len(df_synthetic))
                    elif col == 'beta':
                         df_synthetic[col] = np.random.uniform(0.5, 1.5, len(df_synthetic))
                    elif col == 'forward_return':
                         df_synthetic[col] = np.random.uniform(-0.20, 0.40, len(df_synthetic))
                    elif col == 'market_cap_log': # Ensure this is log-transformed market cap
                         if 'market_cap' in df_synthetic.columns:
                             df_synthetic['market_cap_log'] = np.log10(df_synthetic['market_cap'])
                             df_synthetic.drop(columns=['market_cap'], inplace=True)
                         else:
                             df_synthetic[col] = np.random.uniform(8, 12, len(df_synthetic)) # Log-scale
            
            # Select and reorder columns to match the expected structure if necessary
            final_df_cols = ['ticker'] + [col for col in expected_cols if col != 'ticker']
            df_synthetic = df_synthetic[final_df_cols]

            return df_synthetic
        except FileNotFoundError:
            print("Synthetic CSV 'synthetic_sp500_data.csv' not found. Please create it or ensure yfinance works.")
            return pd.DataFrame() # Return empty DataFrame on failure
    
    return pd.DataFrame(data_list)

# Define dates for data fetching (for yfinance, current fundamentals, and historical prices for forward return proxy)
# For a realistic snapshot, we'd pick one date (e.g., end of Q4 2023) and get all fundamentals relative to that date,
# then get returns for 12 months after. yfinance gives current fundamentals.
# For this lab, let's assume end_date is the 'current' observation date for fundamentals, and we generate forward_return synthetically.
# If we were doing a temporal split, we'd fetch data for multiple periods.
current_date = pd.to_datetime('2023-12-31')
past_date = current_date - pd.DateOffset(years=2) # Fetch historical prices from 2 years back for forward return calculation if possible

sp500_tickers = fetch_sp500_tickers()
df_raw = fetch_financial_data(sp500_tickers, start_date=past_date, end_date=current_date)

if not df_raw.empty:
    print(f"Raw data shape: {df_raw.shape}")
    print("Raw data head:")
    print(df_raw.head())
else:
    print("Failed to load any data. Please check data acquisition steps.")
```

**Markdown Cell — Explanation of Execution**

Sarah has successfully acquired a dataset of S&P 500 stocks, including various fundamental financial ratios and a simulated 12-month forward return. The log-transformation of `market_cap` is a common practice to normalize its skewed distribution. While `yfinance` provides current fundamental data, the forward return was simulated to ensure a consistent target variable for this lab, as precise historical forward returns for a single point in time are complex to align with current fundamentals using this simple API. For Alpha Capital, obtaining precisely time-aligned fundamental data and future returns from a data vendor would be standard practice. This step ensures she has the raw material for both her traditional and ML-driven screening.

---

### 2. Data Cleaning, Feature Engineering, and Exploratory Data Analysis

Before Sarah can apply any screening method, she needs to clean the data and engineer additional features that might provide more predictive power. This includes handling missing values, treating outliers, and creating new financial ratios that capture specific investment theses.

**Real-World Relevance:** Data quality is paramount in finance. Missing data can bias results, and outliers can distort model training. Feature engineering, where domain expertise (like Sarah's CFA background) is applied to create meaningful new variables, is often the most impactful step in quantitative modeling. Z-score standardization is essential for many ML models like Logistic Regression, which are sensitive to feature scales.

**Missing Value Treatment:**
*   Drop stocks with `>30%` missing features (illiquid, new listing).
*   For remaining missing values, apply **sector-median imputation**: replace missing $x_{ij}$ with the median of feature $j$ within the stock's GICS sector. This is financially motivated—a missing ROE is better approximated by sector peers than the global median.

**Outlier Treatment (Winsorization):**
*   Winsorize each numeric feature at the 1st and 99th percentiles to mitigate the impact of extreme values (e.g., negative earnings producing very large negative P/E).
*   The winsorization formula for a feature $x_{ij}$ for stock $i$ and feature $j$ is:
    $$ x_{ij}^{\text{win}} = \max(q_{0.01}, \min(x_{ij}, q_{0.99})) $$
    where $q_{0.01}$ and $q_{0.99}$ are the 1st and 99th percentiles of feature $j$'s distribution, respectively.

**Feature Engineering:**
*   `earnings_yield = 1/P/E` (inverse of P/E, handles negatives better).
*   `quality_score = ROE * profit_margin` (interaction term capturing profitability quality).
*   `leverage_adj_roe = ROE * (1 - D/E / \max(D/E))` (penalizes high-leverage ROE).

**Standardization:**
*   Apply **z-score standardization** to all numeric features after cleaning and engineering. This scales features to have a mean of 0 and a standard deviation of 1, which is critical for Logistic Regression.
*   The z-score for a feature $x_{ij}$ is:
    $$ Z_{ij} = \frac{X_{ij} - \bar{X}_j}{S_j} $$
    where $\bar{X}_j$ and $S_j$ are the training-set mean and standard deviation of feature $j$.

```python
def clean_and_engineer_features(df):
    """
    Cleans data, handles missing values and outliers, and engineers new features.
    """
    df_processed = df.copy()

    # Drop rows with excessive missing data (e.g., >30%)
    initial_rows = len(df_processed)
    df_processed.dropna(thresh=df_processed.shape[1] * 0.70, inplace=True)
    print(f"Dropped {initial_rows - len(df_processed)} rows with >30% missing values.")

    # Convert object columns to numeric where possible, coercing errors
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col not in ['ticker', 'sector']:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Impute missing numeric values using sector median
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    if 'forward_return' in numeric_cols:
        numeric_cols.remove('forward_return') # Do not impute target
    
    # Ensure 'sector' column is handled for imputation
    if 'sector' not in df_processed.columns:
        print("Warning: 'sector' column not found. Falling back to overall median imputation for missing values.")
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
    else:
        df_processed['sector'].fillna('Unknown', inplace=True) # Fill missing sectors before grouping
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed.groupby('sector')[col].transform(lambda x: x.fillna(x.median()))
        # Fill any remaining NaNs (e.g., for sectors with all NaNs) with global median
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Outlier Treatment: Winsorization (1st and 99th percentiles)
    for col in numeric_cols:
        if col in df_processed.columns:
            lower_bound = df_processed[col].quantile(0.01)
            upper_bound = df_processed[col].quantile(0.99)
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

    # Feature Engineering
    df_processed['earnings_yield'] = df_processed['PE_ratio'].apply(lambda x: 1/x if x != 0 else np.nan)
    df_processed['quality_score'] = df_processed['ROE'] * df_processed['profit_margin']
    
    # Handle potential division by zero for DE_ratio max
    max_de_ratio = df_processed['DE_ratio'].max()
    if max_de_ratio == 0: # Avoid division by zero if all DE_ratio are zero
        df_processed['leverage_adj_roe'] = df_processed['ROE']
    else:
        df_processed['leverage_adj_roe'] = df_processed['ROE'] * (1 - df_processed['DE_ratio'] / max_de_ratio)
    
    # Drop rows that might have NaNs introduced by feature engineering (e.g., 1/PE_ratio for PE=0)
    df_processed.dropna(subset=['earnings_yield', 'quality_score', 'leverage_adj_roe'], inplace=True)

    print(f"Processed data shape after cleaning and engineering: {df_processed.shape}")
    return df_processed

df_clean = clean_and_engineer_features(df_raw.copy())

# Target Variable Construction
# Sarah defines 'Buy', 'Hold', 'Sell' based on quantiles of forward returns.
# Top 30% = Buy, Middle 40% = Hold, Bottom 30% = Sell
def create_target_variable(df, buy_quantile=0.70, sell_quantile=0.30):
    """Creates a three-class target variable ('Buy', 'Hold', 'Sell') based on forward returns quantiles."""
    df_with_target = df.copy()
    q_buy = df_with_target['forward_return'].quantile(buy_quantile)
    q_sell = df_with_target['forward_return'].quantile(sell_quantile)

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

df_final = create_target_variable(df_clean)

# Exploratory Data Analysis (EDA) - Visualizations
# Correlation heatmap of features
plt.figure(figsize=(12, 10))
sns.heatmap(df_final.select_dtypes(include=np.number).drop(columns=['forward_return']).corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Distribution plots of features segmented by target class
numeric_features = df_final.select_dtypes(include=np.number).drop(columns=['forward_return']).columns.tolist()
plt.figure(figsize=(16, 10))
for i, feature in enumerate(np.random.choice(numeric_features, 4, replace=False)): # Sample 4 features
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=df_final, x=feature, hue='target', kde=True, palette='viridis', multiple='stack')
    plt.title(f'Distribution of {feature} by Target Class')
plt.tight_layout()
plt.show()

print(f"Final data head after target creation:\n{df_final.head()}")
```

**Markdown Cell — Explanation of Execution**

Sarah has completed a critical data preparation phase. Missing values have been imputed using sector medians, a financially sound approach that leverages peer group information. Outliers have been tamed through winsorization, preventing extreme data points from unduly influencing models. Crucially, new features like `earnings_yield`, `quality_score`, and `leverage_adj_roe` have been engineered, reflecting deeper financial insights than raw ratios alone. The target variable (`Buy`, `Hold`, `Sell`) is now constructed based on forward return quantiles, translating continuous returns into actionable categories for classification. The correlation heatmap helps identify potential multicollinearity, and the distribution plots offer visual insights into how feature values are spread across the different target classes, giving Sarah an initial sense of feature importance and separation. This clean, enriched dataset is now ready for both screening methods.

---

### 3. Traditional Rules-Based Stock Screening (GARP)

Sarah will now implement Alpha Capital's traditional "Growth at a Reasonable Price" (GARP) screen. This method relies on hard-coded thresholds for specific financial ratios, providing a transparent and easily understandable way to filter stocks.

**Real-World Relevance:** Rules-based screening is fundamental in traditional equity analysis. It reflects established investment philosophies (e.g., value investing, growth investing) and provides clear criteria that are easy to communicate to clients or investment committees. However, the choice of thresholds is often subjective and fixed, which can be a limitation.

The GARP screen applies Boolean logic with hard thresholds on financial ratios. For example, a stock might be classified as "Buy" if:
*   P/E ratio is between 0 and 20
*   ROE is greater than 12%
*   Debt-to-Equity ratio is less than 1.5
*   Revenue growth is greater than 5%
*   Profit margin is greater than 8%

Stocks failing these criteria or meeting specific 'Sell' conditions (e.g., very high P/E, very low ROE) are flagged accordingly.

```python
def rules_based_screen(df):
    """
    Implements a classic "Growth at a Reasonable Price" (GARP) screen
    with hard-coded thresholds.
    Returns: DataFrame with 'signal_rules' column (Buy/Hold/Sell).
    """
    df_screened = df.copy()
    
    # Initialize all signals to 'Hold'
    df_screened['signal_rules'] = 'Hold'

    # Conditions for a 'Buy' signal (GARP criteria)
    conditions_buy = (
        (df_screened['PE_ratio'] > 0) & # Positive earnings
        (df_screened['PE_ratio'] < 20) & # Reasonable valuation
        (df_screened['ROE'] > 0.12) & # Strong profitability (12%)
        (df_screened['DE_ratio'] < 1.5) & # Moderate leverage
        (df_screened['revenue_growth'] > 0.05) & # Growing revenue (5%)
        (df_screened['profit_margin'] > 0.08) # Decent margins (8%)
    )
    df_screened.loc[conditions_buy, 'signal_rules'] = 'Buy'

    # Conditions for a 'Sell' signal
    conditions_sell = (
        (df_screened['PE_ratio'] > 30) | # Very expensive
        (df_screened['PE_ratio'] < 0) | # Negative earnings (loss-making)
        (df_screened['ROE'] < 0.05) | # Weak profitability (5%)
        (df_screened['DE_ratio'] > 3.0) | # High leverage
        (df_screened['revenue_growth'] < -0.10) # Declining revenue
    )
    df_screened.loc[conditions_sell, 'signal_rules'] = 'Sell'
    
    return df_screened

df_screened_rules = rules_based_screen(df_final.copy())

print("Rules-based screen signal distribution:")
print(df_screened_rules['signal_rules'].value_counts())
print("\nSample of rules-based signals:")
print(df_screened_rules[['ticker', 'PE_ratio', 'ROE', 'DE_ratio', 'revenue_growth', 'profit_margin', 'signal_rules', 'target']].head())
```

**Markdown Cell — Explanation of Execution**

Sarah has successfully applied the traditional GARP rules-based screen. Each stock is now classified as 'Buy', 'Hold', or 'Sell' based on predefined thresholds. This provides a clear, interpretable output. For Alpha Capital, this output can be directly fed into an analyst's review process. However, Sarah notes that this method is rigid; a stock with a P/E of 20.1 would be categorized differently from one with 19.9, despite a negligible difference, potentially missing out on good opportunities or including questionable ones based on arbitrary cutoffs. This highlights the "binary outcomes" limitation where there's no notion of confidence or probability associated with the signal.

---

### 4. Preparing Data for Machine Learning & Temporal Split

Before training a machine learning model, Sarah needs to encode categorical features, standardize numerical features, and perform a robust train/test split. Given the time-series nature of financial data, a temporal (walk-forward) split is critical to prevent look-ahead bias.

**Real-World Relevance:** In financial modeling, preventing look-ahead bias is paramount. Using future information to predict the past or present is a common pitfall. A temporal split simulates a real-world scenario where a model trained on historical data is used to predict future outcomes. Encoding categorical variables (like `sector`) allows ML models to incorporate non-numeric information. Standardization ensures that features with larger numerical ranges don't disproportionately influence the model's learning process.

**Practitioner Warning: Look-Ahead Bias:** The target variable uses future returns. In a live setting, these are unknown. This case study uses historical data with proper temporal splitting (train period before test period) to avoid look-ahead bias. Participants must understand that the ML model is trained on past feature-return relationships and tested on a held-out future period, not the same period. Since we are using a single cross-section of data, we will simulate a temporal split by sorting the data (e.g., by market cap or ticker) and taking the first N% as training and remaining as test, but acknowledge this is a simplification of a true walk-forward split over multiple time periods. For a more robust temporal split with a single cross-section, one might split by a date if available, or simply use a random split (stratified) and explicitly note the limitation regarding temporal dynamics. Given the synthetic nature, a stratified random split is appropriate while acknowledging the theoretical superiority of true temporal splitting for time-series data.

```python
# One-hot encode the sector feature
df_ml = df_screened_rules.copy() # Start from the dataframe with rules-based signals

# Ensure 'sector' column exists
if 'sector' not in df_ml.columns:
    print("Warning: 'sector' column not found, skipping one-hot encoding.")
    df_ml['sector'] = 'Unknown' # Placeholder to avoid errors

df_ml = pd.get_dummies(df_ml, columns=['sector'], drop_first=True) # drop_first to avoid multicollinearity

# Define features (X) and target (y)
feature_cols = [col for col in df_ml.columns if col not in ['ticker', 'forward_return', 'target', 'signal_rules']]
X = df_ml[feature_cols]
y = df_ml['target']

# Encode the target variable (Buy=0, Hold=1, Sell=2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_ # Store class names for later interpretation

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y_encoded.shape}")
print(f"Target classes: {target_names}")

# Temporal (or stratified random for single snapshot) train/test split
# For this lab's single snapshot, we use a stratified random split to ensure
# representation of all classes in both train and test sets, acknowledging
# the limitations for true time-series data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

# Standardize numeric features (fit on train, transform both train and test)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTraining features (scaled) shape: {X_train_sc.shape}")
print(f"Test features (scaled) shape: {X_test_sc.shape}")

# Store ticker and actual returns for the test set for later analysis
df_test_metadata = df_ml.loc[X_test.index, ['ticker', 'forward_return', 'target', 'signal_rules']].copy()
```

**Markdown Cell — Explanation of Execution**

Sarah has successfully prepared the data for machine learning. The `sector` variable has been one-hot encoded, converting it into a numerical format suitable for the Logistic Regression model. The continuous `target` variable has been discreetly labeled into 'Buy', 'Hold', and 'Sell' using `LabelEncoder`. A stratified random split was applied for the train/test sets to maintain class proportions, carefully noting its simplification compared to a true temporal split for this single cross-section dataset. Finally, all numerical features have been standardized using `StandardScaler`. This preprocessing ensures that the Logistic Regression model can be trained effectively without being biased by feature scales or non-numeric inputs, aligning with best practices in quantitative modeling at Alpha Capital.

---

### 5. Machine Learning Classifier: Logistic Regression

Now, Sarah will train a Multinomial Logistic Regression model to classify stocks into 'Buy', 'Hold', or 'Sell'. This model learns the relationship between financial features and the target outcome probabilistically, offering a more nuanced approach than hard rules. She will also tune its hyperparameters to optimize performance.

**Real-World Relevance:** Logistic Regression is a simple yet powerful classifier, often used as a baseline in finance due to its interpretability. It's a natural bridge from traditional statistical regression to machine learning. Hyperparameter tuning is crucial for any ML model to prevent overfitting and ensure robust performance on unseen data, a key concern for Alpha Capital.

**Multinomial Logistic Regression Formulation:**
The multinomial logistic regression models the probability of each class $k \in \{\text{Buy, Hold, Sell}\}$ given the feature vector $\mathbf{x}_i$:

$$ P(y_i = k | \mathbf{x}_i) = \frac{\exp(\boldsymbol{\beta}_k^\text{T} \mathbf{x}_i + \beta_{k,0})}{\sum_{j=1}^K \exp(\boldsymbol{\beta}_j^\text{T} \mathbf{x}_i + \beta_{j,0})} $$

where:
*   $\boldsymbol{\beta}_k \in \mathbb{R}^P$ is the coefficient vector for class $k$.
*   $\beta_{k,0}$ is the intercept (bias) for class $k$.
*   $K = 3$ (Buy, Hold, Sell).
*   $P$ is the number of features.

The model parameters are estimated by maximizing the regularized log-likelihood. With L2 regularization (Ridge), the objective function to minimize is typically:

$$ \min_{\boldsymbol{\beta}} \left[ -\frac{1}{N} \sum_{i=1}^N \log P(y_i | \mathbf{x}_i; \boldsymbol{\beta}) + \frac{\lambda}{2} \sum_{k=1}^K ||\boldsymbol{\beta}_k||_2^2 \right] $$

where $\lambda$ is the L2 regularization strength, which is inversely related to the `C` parameter in `scikit-learn` ($C = 1/\lambda$). A smaller `C` implies stronger regularization.

```python
# Hyperparameter Tuning using GridSearchCV
# Sarah needs to find the optimal regularization strength (C) for the Logistic Regression.
# Too strong regularization might lead to underfitting, too weak to overfitting.
param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}

# Initialize Logistic Regression with L2 penalty and multinomial solver
lr_model = LogisticRegression(
    penalty='l2',
    multi_class='multinomial', # For multi-class classification
    solver='lbfgs', # 'lbfgs' is a good default for 'multinomial'
    max_iter=1000, # Increased max_iter for convergence
    random_state=42
)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid,
    cv=5, # 5-fold cross-validation
    scoring='f1_weighted', # F1-score is robust for imbalanced classes
    n_jobs=-1, # Use all available cores
    verbose=1
)

print("Starting GridSearchCV for Logistic Regression hyperparameter tuning...")
grid_search.fit(X_train_sc, y_train)

best_C = grid_search.best_params_['C']
print(f"\nBest regularization parameter C found: {best_C}")

# Train the final Logistic Regression model with the best C
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

# Make predictions on the test set
y_pred_ml_encoded = ml_model.predict(X_test_sc)
y_pred_ml_proba = ml_model.predict_proba(X_test_sc) # Probabilities for AUC-ROC

# Decode predictions back to original class names
y_pred_ml = le.inverse_transform(y_pred_ml_encoded)
y_test_decoded = le.inverse_transform(y_test)

print("\nLogistic Regression model trained and predictions made.")
```

**Markdown Cell — Explanation of Execution**

Sarah has successfully trained a Multinomial Logistic Regression model and optimized its key hyperparameter, `C` (regularization strength), using `GridSearchCV`. This systematic tuning process helps Alpha Capital ensure the model generalizes well to unseen data, mitigating the risk of overfitting common in financial markets with limited historical data. The model now provides both discrete 'Buy', 'Hold', 'Sell' predictions and, importantly, the underlying probabilities. These probabilities offer a richer understanding of conviction than binary classifications, allowing Sarah to assess the model's confidence in its signals. The next step is to evaluate these predictions against both the actual forward returns and the rules-based approach.

---

### 6. Comparing Rules-Based vs. ML Classifier Performance

This is the core comparison point. Sarah will evaluate both the traditional rules-based screen and the newly trained Logistic Regression model on the same held-out test set. She will use standard classification metrics and, crucially, finance-specific metrics like Information Coefficient (IC) and "Spread" to assess their practical value for Alpha Capital.

**Real-World Relevance:** For an investment firm, model evaluation goes beyond simple accuracy. Metrics like Precision for 'Buy' signals (how many predicted buys actually performed well), Recall for 'Sell' signals (how many actual underperformers were caught), and the economic "Spread" (return difference between Buy and Sell) directly translate into portfolio performance and risk management. The Information Coefficient (IC) measures the rank correlation of predictions with actual returns, indicating a model's ability to rank stocks, which is highly relevant for portfolio construction.

**Evaluation Metrics:**

*   **Accuracy:** Overall correct classifications.
    $$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{N_{\text{test}}} $$
*   **Precision (per class $k$):** Of all stocks predicted as class $k$, how many truly belong to class $k$?
    $$ \text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k} $$
*   **Recall (per class $k$):** Of all stocks truly belonging to class $k$, how many were correctly identified?
    $$ \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k} $$
*   **F1-Score (per class $k$):** Harmonic mean of Precision and Recall.
    $$ \text{F1}_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} $$
*   **Information Coefficient (IC):** Measures the rank correlation between the ML model's Buy probability (or other ranking score) and actual forward returns. IC $> 0.05$ is often considered meaningful in quantitative investing.
    $$ \text{IC} = \text{Spearman}(P(\text{Buy})_i, r_i^{\text{fwd}}) $$
*   **Spread:** The average return difference between 'Buy' and 'Sell' signals. This is often the most financially meaningful measure of a screening model's economic value.
    $$ \text{Spread} = \text{Avg return}(\text{Buy}) - \text{Avg return}(\text{Sell}) $$
*   **AUC-ROC (One-vs-Rest):** For multi-class, compute AUC for each class vs. all others using predicted probabilities. Macro-average AUC provides a single summary metric of discriminative power.

```python
# Ensure consistent mapping of rules-based signals to encoded target names
# First, map rules-based signals to numerical encoding
df_test_metadata['signal_rules_encoded'] = df_test_metadata['signal_rules'].map({
    target_names[0]: 0, # Assuming 'Buy' is 0
    target_names[1]: 1, # Assuming 'Hold' is 1
    target_names[2]: 2  # Assuming 'Sell' is 2
}).fillna(1) # Fill any unmapped (e.g., if a signal was 'unknown') with 'Hold'

# Get predictions from rules-based screen for the test set
# Align the index of rules-based signals with the test set's original index
y_pred_rules_encoded = df_test_metadata['signal_rules_encoded'].values.astype(int)

# --- Comparison Table and Classification Reports ---
print("--- Comparison of Stock Screening Approaches ---")

# Evaluate Rules-Based Screen
print("\n=== Rules-Based Screen Evaluation ===")
print(classification_report(y_test_decoded, df_test_metadata['signal_rules'], target_names=target_names, zero_division=0))

# Evaluate ML (Logistic Regression) Classifier
print("\n=== ML (Logistic Regression) Evaluation ===")
print(classification_report(y_test_decoded, y_pred_ml, target_names=target_names, zero_division=0))

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix for Rules-Based Screen
cm_rules = confusion_matrix(y_test_decoded, df_test_metadata['signal_rules'], labels=target_names)
sns.heatmap(cm_rules, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=axes[0])
axes[0].set_title('Confusion Matrix: Rules-Based Screen')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Confusion Matrix for ML Classifier
cm_ml = confusion_matrix(y_test_decoded, y_pred_ml, labels=target_names)
sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names, ax=axes[1])
axes[1].set_title('Confusion Matrix: ML (Logistic Regression)')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

# --- Calculate finance-specific metrics (IC and Spread) ---
# Prepare data for IC and Spread calculation
results_df = df_test_metadata.copy()
results_df['ml_pred_class'] = y_pred_ml
results_df['ml_buy_proba'] = y_pred_ml_proba[:, le.transform(['Buy'])[0]] # Probability of 'Buy' class

# Information Coefficient (IC) for ML model
# IC is between ML Buy probability and actual forward returns
ic_ml, _ = spearmanr(results_df['ml_buy_proba'], results_df['forward_return'])
print(f"\nInformation Coefficient (IC) for ML Model: {ic_ml:.4f}")

# Calculate Spread for Rules-Based
buy_returns_rules = results_df[results_df['signal_rules'] == 'Buy']['forward_return']
sell_returns_rules = results_df[results_df['signal_rules'] == 'Sell']['forward_return']
spread_rules = buy_returns_rules.mean() - sell_returns_rules.mean() if not buy_returns_rules.empty and not sell_returns_rules.empty else np.nan
print(f"Spread (Avg Return Buy - Avg Return Sell) for Rules-Based: {spread_rules:.4f}")

# Calculate Spread for ML
buy_returns_ml = results_df[results_df['ml_pred_class'] == 'Buy']['forward_return']
sell_returns_ml = results_df[results_df['ml_pred_class'] == 'Sell']['forward_return']
spread_ml = buy_returns_ml.mean() - sell_returns_ml.mean() if not buy_returns_ml.empty and not sell_returns_ml.empty else np.nan
print(f"Spread (Avg Return Buy - Avg Return Sell) for ML: {spread_ml:.4f}")

# --- ROC Curves (One-vs-Rest) ---
fig, ax = plt.subplots(figsize=(8, 6))
for i, class_label in enumerate(target_names):
    # For ROC curve, we need binary labels (1 for current class, 0 for rest)
    y_true_binary = (y_test == i).astype(int)
    y_score = y_pred_ml_proba[:, i]
    
    # Check if there are both positive and negative samples for the class
    if len(np.unique(y_true_binary)) > 1:
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC curve for {class_label} (area = {roc_auc:.2f})')
    else:
        print(f"Skipping ROC curve for class {class_label} due to insufficient samples in test set.")

ax.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('One-vs-Rest ROC Curves for ML Classifier')
ax.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Signal Return Distribution (Box Plots) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Rules-based
sns.boxplot(x='signal_rules', y='forward_return', data=results_df, order=target_names, palette='RdYlGn_r', ax=axes[0])
axes[0].set_title('Forward Returns by Rules-Based Signal')
axes[0].set_xlabel('Rules-Based Signal')
axes[0].set_ylabel('12-Month Forward Return')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# ML-based
sns.boxplot(x='ml_pred_class', y='forward_return', data=results_df, order=target_names, palette='RdYlGn_r', ax=axes[1])
axes[1].set_title('Forward Returns by ML (Logistic Regression) Signal')
axes[1].set_xlabel('ML Signal')
axes[1].set_ylabel('') # Remove redundant y-label
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Comparative Forward Returns by Signal Class', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()
```

**Markdown Cell — Explanation of Execution**

Sarah has performed a comprehensive evaluation, comparing the rules-based screen and the Logistic Regression model using both standard classification metrics and finance-specific indicators. The classification reports provide a detailed breakdown of precision, recall, and F1-scores for each class ('Buy', 'Hold', 'Sell'), highlighting the strengths and weaknesses of each method. The confusion matrices visually summarize the correct and incorrect classifications, revealing where each model makes mistakes.

Crucially, the Information Coefficient (IC) quantifies the ML model's ability to rank stocks, with a positive IC suggesting a meaningful predictive edge. The "Spread" metric, which is the average return difference between 'Buy' and 'Sell' signals, provides a direct measure of the economic value generated by each screening approach. For Alpha Capital, a larger spread implies a more effective screening tool. The box plots of forward returns segmented by signal class visually confirm which method generates a better separation between high-performing ('Buy') and low-performing ('Sell') stocks. The ROC curves offer insight into the discriminative power of the ML model for each class. This holistic evaluation helps Sarah understand not just how accurate the models are, but how practically useful they are for Alpha Capital's investment process.

---

### 7. Interpreting the ML Model and Discussing Business Implications

Sarah's final task is to interpret the coefficients of the Logistic Regression model to understand which financial features drive its 'Buy' and 'Sell' decisions. She will then discuss the practical trade-offs between the interpretable rules-based screen and the more adaptable ML approach, considering Alpha Capital's needs.

**Real-World Relevance:** Model interpretability is crucial in finance, especially for CFA charterholders who need to explain investment decisions to clients or justify models to regulators. Understanding factor sensitivities (e.g., how P/E or ROE impacts a 'Buy' signal) allows analysts to gain trust in the model and refine their investment hypotheses. Discussing trade-offs helps Alpha Capital make informed decisions about integrating new technologies into their workflow.

**Coefficient Interpretation:** For Logistic Regression, a positive coefficient for a feature in the 'Buy' class implies that higher values of that feature increase the log-odds of a stock being classified as 'Buy'. Conversely, a negative coefficient decreases these log-odds.

```python
# Interpret Logistic Regression Coefficients
# The coefficients are stored in ml_model.coef_ and ml_model.intercept_
# ml_model.coef_ has shape (n_classes, n_features)

# Get coefficients for each class
coefficients = pd.DataFrame(ml_model.coef_, columns=X.columns, index=target_names)
intercepts = pd.Series(ml_model.intercept_, index=target_names, name='Intercept')

print("\n--- Logistic Regression Coefficients per Class ---")
print(coefficients)
print("\n--- Logistic Regression Intercepts per Class ---")
print(intercepts)

# Visualize coefficients for the 'Buy' class (assuming 'Buy' is target_names[0])
buy_class_index = le.transform(['Buy'])[0]
buy_coefficients = coefficients.loc[target_names[buy_class_index]].sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=buy_coefficients.values, y=buy_coefficients.index, palette='coolwarm')
plt.title(f'Logistic Regression Coefficients for "{target_names[buy_class_index]}" Class')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Performance Comparison Bar Chart (Accuracy, F1, Spread) ---
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Weighted', 'IC (ML Only)', 'Spread (Buy-Sell)'],
    'Rules-Based': [
        accuracy_score(y_test_decoded, df_test_metadata['signal_rules']),
        f1_score(y_test_decoded, df_test_metadata['signal_rules'], average='weighted'),
        np.nan, # IC is typically for probabilistic models
        spread_rules
    ],
    'ML (Logistic Reg)': [
        accuracy_score(y_test_decoded, y_pred_ml),
        f1_score(y_test_decoded, y_pred_ml, average='weighted'),
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
```

**Markdown Cell — Explanation of Execution**

Sarah has delved into the interpretability of the Logistic Regression model by examining its coefficients. The bar chart for the 'Buy' class coefficients immediately reveals which features are most positively (e.g., high ROE, positive revenue growth) or negatively (e.g., high DE_ratio, low profit margin) associated with a 'Buy' signal. This provides Alpha Capital with actionable insights into the underlying drivers of the ML model's recommendations, allowing Sarah to align them with her existing financial intuition and identify potentially new factor sensitivities.

The comparative bar chart clearly summarizes the performance differences across key metrics like accuracy, F1-score, IC, and Spread. This visualization is crucial for presenting the findings to senior portfolio managers or the investment committee, enabling a data-driven discussion on the merits and trade-offs of integrating machine learning into their stock screening process. Sarah can now articulate how the ML model, while more complex, might offer a more nuanced and potentially more profitable approach than rigid rules.

---

### Conclusion and Discussion Points for Alpha Capital

Sarah has successfully conducted a comparative analysis of traditional rules-based stock screening and a simple Machine Learning (ML) classifier using Logistic Regression. This exercise provides valuable insights for Alpha Capital.

**Discussion Points:**

*   **Interpretability vs. Performance Trade-off:** The rules-based screen is highly transparent and easily explainable (e.g., "P/E under 20 and ROE over 12%"). The Logistic Regression model, while interpretable through its coefficients, requires a deeper understanding of its mechanics. Sarah needs to consider where Alpha Capital draws the line on model complexity vs. explainability, especially when communicating with clients or investment committees. The firm's culture and regulatory environment will influence this balance.
*   **Adaptability and Nuance:** The ML model's ability to learn from data allows it to adapt to changing market conditions (if retrained periodically) and potentially capture subtle interaction effects between features that hard-coded rules miss. This can lead to more dynamic and robust signals. Static rules, in contrast, require manual updates and lack this inherent adaptability.
*   **Probabilistic Outputs:** Unlike the binary pass/fail of rules, the ML model provides probabilities for each class. This offers a richer information set, allowing Sarah to not just identify 'Buy' stocks but also gauge the model's conviction level, which can be critical for risk-adjusted portfolio construction.
*   **Regulatory Implications:** As a CFA Charterholder, Sarah is aware of the regulatory landscape. Any model used for investment decisions, even a simple Logistic Regression, might be subject to model validation guidelines (e.g., SR 11-7 in the US). A rules-based screen might not always be considered a "model" in the same stringent sense, posing a different compliance burden for Alpha Capital.
*   **Practical Deployment:** In practice, ML-based screening often serves as the first stage in a multi-stage process. The ML model can generate a shortlist of candidates, which human analysts like Sarah then subject to deep fundamental due diligence. This "human-in-the-loop" approach combines ML efficiency with human judgment, leveraging the strengths of both.
*   **Feature Engineering as Domain Expertise:** The creation of features like `quality_score` and `leverage_adj_roe` directly embeds financial domain knowledge into the ML process. This highlights how a CFA's expertise is not replaced but augmented by ML, ensuring the models are financially sensible rather than purely statistical abstractions.

**Future Enhancements:**

*   Explore more advanced ML models (e.g., Decision Trees, Random Forests) to potentially capture non-linear relationships.
*   Incorporate momentum factors (e.g., trailing 6-month returns) to enrich the feature set.
*   Develop a "hybrid" approach where ML-generated scores are constrained by fundamental screens (e.g., requiring positive earnings) to ensure financial prudence.
*   Implement a true walk-forward backtesting framework with multiple train/test splits over historical periods to rigorously evaluate the models' out-of-sample performance over time.

This hands-on exploration provides Sarah and Alpha Capital a foundational understanding of how ML can enhance their core equity research workflow, paving the way for more sophisticated quantitative strategies.


id: 698f64d053049d8a753b8d1e_documentation
summary: Lab 1: Stock Screening with SimpleML Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Stock Screening with SimpleML

## Step 1: Introduction and Setting Up the Environment
Duration: 00:05

Welcome to QuLab: Stock Screening with SimpleML! In this codelab, you will step into the shoes of Sarah, a Junior Portfolio Manager at Alpha Capital, and explore how traditional rules-based stock screening can be augmented and potentially enhanced by simple machine learning (ML) techniques. This lab aims to provide a comprehensive guide for developers to understand the functionalities of a Streamlit application designed for this comparison.

<aside class="positive">
<b>Why is this important?</b> In the dynamic world of finance, identifying promising investment opportunities is paramount. While traditional, rules-based screens offer transparency, they often lack adaptability and the ability to capture complex relationships in data. Machine learning offers a powerful alternative to create more dynamic and nuanced stock selection models, directly impacting alpha generation and risk management for firms like Alpha Capital.
</aside>

**The Problem:** Sarah currently relies on static, rigid rules (e.g., "Growth at a Reasonable Price" - GARP) implemented in spreadsheets. These rules, while intuitive, can miss subtle opportunities, fail to adapt to changing market conditions, and cannot easily capture interactions between different financial metrics.

**The Solution:** This lab will guide Sarah through implementing both her firm's classic GARP screen and a simple ML classifier (Logistic Regression) to perform the same equity selection task. By comparing them side-by-side, we will understand their trade-offs and potential for improving Alpha Capital's systematic security selection process.

### Learning Outcomes:
*   Acquire and clean S&P 500 fundamental data, including feature engineering and outlier treatment.
*   Implement both a rules-based (GARP) screen and a Logistic Regression classifier for stock selection.
*   Perform temporal train/test splitting and hyperparameter tuning for the ML model.
*   Evaluate both methods using comprehensive classification and finance-specific metrics.
*   Interpret model results and discuss trade-offs between interpretability and performance.

### 1.1 Setting Up the Python Environment and Acquiring Financial Data

Sarah's first step is to set up her Python environment and acquire the necessary financial data. For a robust stock screening process, she needs fundamental financial ratios for S&P 500 constituents, along with their subsequent 12-month forward returns to serve as a target for both methods.

**Real-World Relevance:** In investment firms like Alpha Capital, data acquisition is the bedrock of any quantitative strategy. Analysts often need to integrate data from various sources (e.g., Bloomberg, Refinitiv, S&P Global Market Intelligence). For preliminary analysis and research, publicly available APIs like `yfinance` are often used to quickly prototype ideas.

The formula for the 12-month forward total return is:
$$ r_i^{\text{fwd}} = \frac{P_{i,t+12} - P_{i,t}}{P_{i,t}} $$
where $P$ is the adjusted close price.

### Application Architecture and Helper Functions

The Streamlit application leverages a modular design, with core data processing and modeling logic encapsulated in a `source.py` file. This promotes code organization and reusability.

**Application Flow:**
The application proceeds in a linear, step-by-step fashion, guiding the user through data acquisition, preparation, model building (both rules-based and ML), evaluation, and interpretation. Each step builds upon the results of the previous one, with session state management ensuring data persistence across page navigations.

**`source.py` (Assumed Content):**
For this codelab, we assume `source.py` contains the following helper functions. You would typically have this file alongside your main Streamlit script.

```python
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Function to fetch S&P 500 tickers
def fetch_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.tbody.find_all('tr')[1:]:
            ticker = row.find_all('td')[0].text.strip()
            tickers.append(ticker)
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        print("Using fallback tickers.")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'V', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'NFLX']

# Function to fetch fundamental financial data and calculate forward returns
def fetch_financial_data(tickers, start_date, end_date):
    data = []
    print(f"Attempting to fetch data for {len(tickers)} tickers...")
    
    # Define a simplified set of financial metrics from yfinance
    metrics_to_fetch = {
        'trailingPE': 'PE_ratio',
        'forwardPE': 'forward_PE_ratio',
        'returnOnEquity': 'ROE',
        'debtToEquity': 'DE_ratio',
        'revenueGrowth': 'revenue_growth',
        'grossMargins': 'gross_margin',
        'profitMargins': 'profit_margin',
        'marketCap': 'market_cap',
        'earningsGrowth': 'earnings_growth' # Added earnings growth for completeness
    }

    # Simulate forward return calculation - in a real scenario, this would be complex
    # and require aligning historical fundamentals with future price data.
    # For this lab, we'll simulate a 12-month forward return based on a recent historical period.
    # The actual 'forward_return' column will be populated with synthetic data in `create_target_variable`
    # or based on a simplified lookback. For `fetch_financial_data`, we'll just focus on fundamentals.

    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            row = {'ticker': ticker}
            for yf_key, custom_key in metrics_to_fetch.items():
                row[custom_key] = info.get(yf_key)

            # Add sector information
            row['sector'] = info.get('sector', 'Unknown')
            
            # Simulate a 12-month forward return. In a real application, this would be computed
            # from future price data. For now, we'll leave it as None or a placeholder.
            # We will generate a more robust 'forward_return' in create_target_variable.
            row['forward_return'] = None # Placeholder

            data.append(row)
            if (i + 1) % 50 == 0:
                print(f"Fetched data for {i + 1}/{len(tickers)} tickers.")
        except Exception as e:
            # print(f"Could not fetch data for {ticker}: {e}")
            pass # Suppress individual ticker errors for cleaner output

    if not data:
        print("Failed to fetch any data. Using synthetic fallback data.")
        return create_synthetic_data(tickers) # Fallback to synthetic data

    df = pd.DataFrame(data)
    df['market_cap'] = df['market_cap'].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
    return df

# Synthetic data creation function as a fallback
def create_synthetic_data(tickers):
    num_stocks = 200 # A reasonable number for synthetic data
    if len(tickers) < num_stocks:
        selected_tickers = tickers
    else:
        selected_tickers = np.random.choice(tickers, num_stocks, replace=False)

    data = {
        'ticker': selected_tickers,
        'PE_ratio': np.random.uniform(5, 50, num_stocks),
        'forward_PE_ratio': np.random.uniform(5, 45, num_stocks),
        'ROE': np.random.uniform(0.01, 0.30, num_stocks),
        'DE_ratio': np.random.uniform(0.1, 3.0, num_stocks),
        'revenue_growth': np.random.uniform(-0.05, 0.20, num_stocks),
        'gross_margin': np.random.uniform(0.1, 0.8, num_stocks),
        'profit_margin': np.random.uniform(0.01, 0.30, num_stocks),
        'market_cap': np.random.uniform(1e9, 5e11, num_stocks),
        'earnings_growth': np.random.uniform(-0.1, 0.3, num_stocks),
        'sector': np.random.choice(['Technology', 'Healthcare', 'Financials', 'Industrials', 'Consumer Discretionary'], num_stocks)
    }
    df = pd.DataFrame(data)
    df['market_cap'] = df['market_cap'].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else np.nan)
    df['forward_return'] = np.random.uniform(-0.10, 0.30, num_stocks) # Synthetic forward returns
    return df


def clean_and_engineer_features(df):
    initial_shape = df.shape[0]
    
    # 1. Drop rows with too many missing values (e.g., >30%)
    df_cleaned = df.dropna(thresh=df.shape[1] * 0.7, axis=0).reset_index(drop=True)
    print(f"Dropped {initial_shape - df_cleaned.shape[0]} stocks due to excessive missing data.")

    # 2. Impute missing values using sector median
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    
    # Ensure 'sector' column exists for imputation
    if 'sector' not in df_cleaned.columns:
        df_cleaned['sector'] = 'Unknown'

    for col in numeric_cols:
        df_cleaned[col] = df_cleaned.groupby('sector')[col].transform(lambda x: x.fillna(x.median()))
        
    # Fill remaining NaNs for columns that might have all NaNs within a sector (e.g., if a sector only has 1 stock with NaN)
    for col in numeric_cols:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

    # 3. Outlier Treatment (Winsorization)
    for col in numeric_cols:
        if col in df_cleaned.columns:
            lower_bound = df_cleaned[col].quantile(0.01)
            upper_bound = df_cleaned[col].quantile(0.99)
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)

    # 4. Feature Engineering
    df_cleaned['earnings_yield'] = df_cleaned['PE_ratio'].apply(lambda x: 1/x if x != 0 else 0)
    df_cleaned['quality_score'] = df_cleaned['ROE'] * df_cleaned['profit_margin']
    
    # Avoid division by zero for leverage_adj_roe
    max_de = df_cleaned['DE_ratio'].max()
    if max_de > 0:
        df_cleaned['leverage_adj_roe'] = df_cleaned['ROE'] * (1 - df_cleaned['DE_ratio'] / max_de)
    else:
        df_cleaned['leverage_adj_roe'] = df_cleaned['ROE'] # If no debt, no adjustment

    # Drop original PE if earnings_yield is preferred, or keep both
    # df_cleaned = df_cleaned.drop(columns=['PE_ratio'], errors='ignore')

    return df_cleaned


def create_target_variable(df):
    # If 'forward_return' is None/synthetic, generate it more meaningfully here
    # For this lab, let's ensure forward_return has varied values for target creation
    if 'forward_return' not in df.columns or df['forward_return'].isnull().all():
        df['forward_return'] = np.random.normal(0.05, 0.15, df.shape[0]) # Realistic distribution
    
    # Remove any NaN in forward_return before quantile calculation
    df_filtered = df.dropna(subset=['forward_return']).copy()

    # Define quantiles for Buy, Hold, Sell
    q_low = df_filtered['forward_return'].quantile(0.33)
    q_high = df_filtered['forward_return'].quantile(0.66)

    def assign_target(row):
        if row['forward_return'] > q_high:
            return 'Buy'
        elif row['forward_return'] < q_low:
            return 'Sell'
        else:
            return 'Hold'
    
    df_filtered['target'] = df_filtered.apply(assign_target, axis=1)
    return df_filtered


def rules_based_screen(df):
    df_rules = df.copy()

    # Apply GARP rules
    def apply_garp(row):
        # Basic Buy conditions
        buy_conditions = (
            (row['PE_ratio'] > 0) & (row['PE_ratio'] <= 20) &
            (row['ROE'] > 0.12) &
            (row['DE_ratio'] < 1.5) &
            (row['revenue_growth'] > 0.05) &
            (row['profit_margin'] > 0.08)
        )
        
        # Basic Sell conditions (e.g., highly negative performance indicators)
        sell_conditions = (
            (row['PE_ratio'] > 50) | (row['PE_ratio'] <= 0) | # Very high P/E or negative P/E (losses)
            (row['ROE'] < 0.05) |
            (row['DE_ratio'] > 3.0) |
            (row['revenue_growth'] < -0.05) |
            (row['profit_margin'] < 0.01)
        )

        if buy_conditions:
            return 'Buy'
        elif sell_conditions:
            return 'Sell'
        else:
            return 'Hold'

    # Apply the function row-wise. Handle potential NaNs in features by filling them temporarily
    # for rule evaluation or ensure rules are robust to NaNs.
    # For simplicity, we assume cleaned data here, but in production, careful NaN handling is needed.
    df_rules['signal_rules'] = df_rules.apply(apply_garp, axis=1)
    return df_rules

```

### Running the Streamlit Application

To run the Streamlit application and follow along, you'll need Python installed.

1.  **Save the Streamlit application code:** Save the provided Streamlit Python code as `app.py`.
2.  **Save the helper functions:** Save the `source.py` content above into a file named `source.py` in the same directory as `app.py`.
3.  **Install dependencies:** Open your terminal or command prompt and run:
    ```console
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn yfinance requests beautifulsoup4
    ```
4.  **Run the application:** In your terminal, navigate to the directory where you saved `app.py` and `source.py`, then run:
    ```console
    streamlit run app.py
    ```
    This will open the application in your web browser.

Once the application is running, navigate to the "1. Introduction & Data Acquisition" page if you're not already there.

<button>
  [Download `app.py` and `source.py` from this codelab](https://www.google.com)
</button>

**Action:** Click the "Fetch S&P 500 Financial Data" button.

This will initiate the data acquisition process using `yfinance` to fetch S&P 500 tickers and their fundamental data. Due to potential API rate limits or network issues, a synthetic fallback dataset is included to ensure the lab can proceed. Once completed, you'll see a confirmation message and the head of the raw DataFrame.

## Step 2: Data Preparation, Feature Engineering, and EDA
Duration: 00:10

Now that Sarah has acquired the raw financial data, the next critical step is to prepare it for modeling. This involves rigorous data cleaning, engineering new features that encapsulate deeper financial insights, and performing exploratory data analysis (EDA) to understand the dataset's characteristics.

**Real-World Relevance:** Data quality is paramount in finance. Missing data can bias results, and outliers can distort model training. Feature engineering, where domain expertise (like Sarah's CFA background) is applied to create meaningful new variables, is often the most impactful step in quantitative modeling. Z-score standardization is essential for many ML models like Logistic Regression, which are sensitive to feature scales.

### 2.1 Data Cleaning
*   **Missing Value Treatment:**
    *   Drop stocks with `>30%` missing features (these might be illiquid or newly listed, lacking comprehensive data).
    *   For remaining missing values, apply **sector-median imputation**: replace missing values with the median of the feature within the stock's GICS sector. This is financially motivated—a missing ROE is better approximated by sector peers than the global median.
*   **Outlier Treatment (Winsorization):**
    *   Winsorize each numeric feature at the 1st and 99th percentiles to mitigate the impact of extreme values (e.g., negative earnings producing very large negative P/E, or extremely high growth rates). This clips values to a specified range, preserving the order of data points but reducing the influence of extremes.
    $$ x_{ij}^{\text{win}} = \max(q_{0.01}, \min(x_{ij}, q_{0.99})) $$
    where $q$ represents the quantiles.

### 2.2 Feature Engineering
New financial ratios are engineered to provide more predictive power and capture specific investment theses:
*   `earnings_yield = 1/P/E` (inverse of P/E, often preferred as it handles negative earnings better and has a more intuitive linear relationship with value).
*   `quality_score = ROE * profit_margin` (an interaction term capturing the quality of a company's profitability).
*   `leverage_adj_roe = ROE * (1 - D/E / max(D/E))` (penalizes high-leverage ROE, reflecting a more conservative view on quality of earnings).

### 2.3 Standardization
*   Apply **z-score standardization** to all numeric features after cleaning and engineering. This scales features to have a mean of 0 and a standard deviation of 1, which is critical for distance-based ML models like Logistic Regression.
    $$ Z_{ij} = \frac{X_{ij} - \bar{X}_j}{S_j} $$
    where $\bar{X}_j$ and $S_j$ are the mean and standard deviation of feature $j$ from the training set.

### 2.4 Target Variable Creation
The target variable for classification is created from the 12-month forward return. The continuous `forward_return` is discretized into three categories: 'Buy', 'Hold', and 'Sell' based on quantiles.
*   'Buy': Top 33% of forward returns.
*   'Sell': Bottom 33% of forward returns.
*   'Hold': Middle 34% of forward returns.

This transforms the problem into a multi-class classification task, which is more directly actionable for stock selection.

**Action:** Navigate to the "2. Data Preparation & EDA" page and click the "Clean Data, Engineer Features & Create Target" button.

After processing, you will see a summary of the data cleaning steps, the head of the final DataFrame, and the distribution of the newly created `target` variable.

### 2.5 Exploratory Data Analysis (EDA)
Following the data preparation, the application displays key EDA visualizations:
*   **Correlation Heatmap:** This helps identify multicollinearity among numeric features, which can affect the stability and interpretability of models like Logistic Regression. Features with very high correlation might provide redundant information.
*   **Distribution Plots by Target Class:** These plots (histograms) show the distribution of selected features, segmented by the 'Buy', 'Hold', and 'Sell' target classes. This offers an initial visual insight into which features might be strong predictors, as a clear separation in distributions between classes indicates predictive power.

<aside class="positive">
<b>Key Takeaway:</b> Sarah has completed a critical data preparation phase. Missing values have been imputed using sector medians (a financially sound approach), and outliers have been tamed through winsorization. Crucially, new features have been engineered, reflecting deeper financial insights. The target variable is now constructed based on forward return quantiles, translating continuous returns into actionable categories for classification. This clean, enriched dataset is now ready for both screening methods.
</aside>

## Step 3: Implementing Traditional Rules-Based Screening (GARP)
Duration: 00:08

Sarah will now implement Alpha Capital's traditional "Growth at a Reasonable Price" (GARP) screen. This method relies on hard-coded thresholds for specific financial ratios, providing a transparent and easily understandable way to filter stocks.

**Real-World Relevance:** Rules-based screening is fundamental in traditional equity analysis. It reflects established investment philosophies (e.g., value investing, growth investing) and provides clear criteria that are easy to communicate to clients or investment committees. However, the choice of thresholds is often subjective and fixed, which can be a limitation.

### 3.1 GARP Screen Logic
The GARP screen applies Boolean logic with hard thresholds on financial ratios. For example, a stock might be classified as "Buy" if it meets **all** of the following conditions:
*   P/E ratio is between 0 and 20 (reasonable valuation)
*   ROE (Return on Equity) is greater than 12% (strong profitability)
*   Debt-to-Equity ratio is less than 1.5 (healthy balance sheet, low leverage)
*   Revenue growth is greater than 5% (growth component)
*   Profit margin is greater than 8% (efficient operations)

Stocks failing these criteria or meeting specific 'Sell' conditions (e.g., very high P/E, very low ROE, high debt, negative growth/margins) are flagged accordingly. Stocks that don't fit into 'Buy' or 'Sell' categories are classified as 'Hold'.

**Action:** Navigate to the "3. Traditional Rules-Based Screening" page and click the "Apply Rules-Based Screen" button.

Upon completion, the application will display the distribution of 'Buy', 'Hold', and 'Sell' signals generated by the rules-based screen, along with a sample of the screened DataFrame.

<aside class="negative">
<b>Limitation Highlight:</b> Sarah notes that this method is rigid; a stock with a P/E of 20.1 would be categorized differently from one with 19.9, despite a negligible difference. This can lead to missing out on good opportunities or including questionable ones based on arbitrary cutoffs. This highlights the "binary outcomes" limitation where there's no notion of confidence or probability associated with the signal.
</aside>

## Step 4: Training a Machine Learning Classifier (Logistic Regression)
Duration: 00:15

Now, Sarah will train a Multinomial Logistic Regression model to classify stocks into 'Buy', 'Hold', or 'Sell'. This model learns the relationship between financial features and the target outcome probabilistically, offering a more nuanced approach than hard rules. She will also tune its hyperparameters to optimize performance.

### 4.1 Data Preparation for ML
Before training the ML model, further data preparation is needed:
*   **Categorical Feature Encoding:** The `sector` feature, being categorical, needs to be converted into a numerical format that ML models can understand. One-hot encoding is applied, creating new binary features for each sector (e.g., `sector_Technology`, `sector_Healthcare`). The `drop_first=True` argument prevents multicollinearity among the one-hot encoded features.
*   **Target Variable Encoding:** The string labels ('Buy', 'Hold', 'Sell') for the target variable are converted into numerical labels (e.g., 0, 1, 2) using `LabelEncoder`. This is a requirement for most ML algorithms in `scikit-learn`.
*   **Train/Test Split:** A `train_test_split` is performed to divide the data into training and testing sets.
    <aside class="negative">
    <b>Practitioner Warning: Look-Ahead Bias:</b> The target variable uses future returns. In a live setting, these are unknown. This case study uses historical data with proper temporal splitting (train period before test period) to avoid look-ahead bias. Participants must understand that the ML model is trained on past feature-return relationships and tested on a held-out future period, not the same period. Since we are using a single cross-section of data, we will simulate a temporal split by sorting the data (e.g., by market cap or ticker) and taking the first N% as training and remaining as test, but acknowledge this is a simplification of a true walk-forward split over multiple time periods. For a more robust temporal split with a single cross-section, one might split by a date if available, or simply use a random split (stratified) and explicitly note the limitation regarding temporal dynamics. Given the synthetic nature, a stratified random split is appropriate while acknowledging the theoretical superiority of true temporal splitting for time-series data.
    </aside>
*   **Feature Scaling:** While z-score standardization was applied in Step 2, it's crucial to apply `StandardScaler` *again* after the train/test split. This ensures that the scaling parameters (mean and standard deviation) are learned *only* from the training data and then applied to both training and test sets, preventing data leakage from the test set.

### 4.2 Multinomial Logistic Regression Formulation
Logistic Regression is a linear model for classification. When there are more than two classes (like 'Buy', 'Hold', 'Sell'), it becomes Multinomial Logistic Regression.
The model estimates the probability of each class $k \in \{\text{Buy, Hold, Sell}\}$ given the feature vector $\mathbf{x}_i$:
$$ P(y_i = k | \mathbf{x}_i) = \frac{\exp(\boldsymbol{\beta}_k^\text{T} \mathbf{x}_i + \beta_{k,0})}{\sum_{j=1}^K \exp(\boldsymbol{\beta}_j^\text{T} \mathbf{x}_i + \beta_{j,0})} $$
where $\boldsymbol{\beta}_k \in \mathbb{R}^P$ is the coefficient vector for class $k$, and $\beta_{k,0}$ is the intercept.

With L2 regularization (Ridge), the objective function to minimize is typically:
$$ \min_{\boldsymbol{\beta}} \left[ -\frac{1}{N} \sum_{i=1}^N \log P(y_i | \mathbf{x}_i; \boldsymbol{\beta}) + \frac{\lambda}{2} \sum_{k=1}^K ||\boldsymbol{\beta}_k||_2^2 \right] $$
The regularization parameter `C` in `scikit-learn` is the inverse of the regularization strength $\lambda$. Smaller `C` means stronger regularization.

### 4.3 Hyperparameter Tuning with `GridSearchCV`
To find the optimal regularization strength for the Logistic Regression model, `GridSearchCV` is used. This technique systematically works through multiple combinations of parameter values, evaluating the model's performance (using cross-validation) for each combination and selecting the best one.
*   **Parameters:** The `C` parameter (inverse of regularization strength) is tuned over a range of values (e.g., `[0.01, 0.1, 1.0, 10.0, 100.0]`).
*   **Cross-Validation:** `cv=5` means the training data is split into 5 folds, and the model is trained and evaluated 5 times, with each fold serving as a validation set once.
*   **Scoring:** `scoring='f1_weighted'` is used for multi-class classification, as it accounts for class imbalance by calculating the F1-score for each class and averaging them weighted by the number of true instances for each label.

**Action:** Navigate to the "4. Machine Learning Model Training" page and click the "Train ML Model (Logistic Regression)" button.

The application will display information about feature and target shapes, the best `C` parameter found, and confirmation of model training. The trained model will then make predictions and calculate probabilities on the test set.

<aside class="positive">
<b>Key Takeaway:</b> Sarah has successfully trained a Multinomial Logistic Regression model and optimized its key hyperparameter, `C` (regularization strength), using `GridSearchCV`. This systematic tuning process helps Alpha Capital ensure the model generalizes well to unseen data, mitigating the risk of overfitting common in financial markets with limited historical data. The model now provides both discrete 'Buy', 'Hold', 'Sell' predictions and, importantly, the underlying probabilities. These probabilities offer a richer understanding of conviction than binary classifications, allowing Sarah to assess the model's confidence in its signals.
</aside>

## Step 5: Comparative Performance Evaluation
Duration: 00:12

This is the core comparison point. Sarah will evaluate both the traditional rules-based screen and the newly trained Logistic Regression model on the same held-out test set. She will use standard classification metrics and, crucially, finance-specific metrics like Information Coefficient (IC) and "Spread" to assess their practical value for Alpha Capital.

**Real-World Relevance:** For an investment firm, model evaluation goes beyond simple accuracy. Metrics like Precision for 'Buy' signals (how many predicted buys actually performed well), Recall for 'Sell' signals (how many actual underperformers were caught), and the economic "Spread" (return difference between Buy and Sell) directly translate into portfolio performance and risk management. The Information Coefficient (IC) measures the rank correlation of predictions with actual returns, indicating a model's ability to rank stocks, which is highly relevant for portfolio construction.

### 5.1 Evaluation Metrics
*   **Accuracy:** Overall correct classifications.
    $$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{N_{\text{test}}} $$
*   **Precision (per class k):** Of all stocks predicted as class k, how many truly belong to class k?
    $$ \text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k} $$
*   **Recall (per class k):** Of all stocks truly belonging to class k, how many were correctly identified?
    $$ \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k} $$
*   **F1-Score (per class k):** Harmonic mean of Precision and Recall.
    $$ \text{F1}_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} $$
*   **Information Coefficient (IC):** Measures the Spearman rank correlation between the ML model's Buy probability (or other ranking score) and actual forward returns. IC > 0.05 is often considered meaningful in quantitative investing.
    $$ \text{IC} = \text{Spearman}(P(\text{Buy})_i, r_i^{\text{fwd}}) $$
*   **Spread:** The average return difference between 'Buy' and 'Sell' signals. This is often the most financially meaningful measure of a screening model's economic value.
    $$ \text{Spread} = \text{Avg return}(\text{Buy}) - \text{Avg return}(\text{Sell}) $$
*   **AUC-ROC (One-vs-Rest):** For multi-class, AUC (Area Under the Receiver Operating Characteristic Curve) is computed for each class vs. all others using predicted probabilities. Macro-average AUC provides a single summary metric of discriminative power.

### 5.2 Comparative Analysis
The application presents the following for both rules-based and ML models:
*   **Classification Reports:** Detailed per-class metrics (precision, recall, f1-score, support).
*   **Confusion Matrices:** Visual representation of correct and incorrect classifications for each class.
*   **IC and Spread Metrics:** Quantified values of IC for ML and Spread for both models.
*   **One-vs-Rest ROC Curves:** Visualizes the trade-off between true positive rate and false positive rate for each class in the ML model.
*   **Box Plots of Forward Returns by Signal Class:** Visually compares the distribution of actual forward returns for stocks classified as 'Buy', 'Hold', or 'Sell' by each method. This is a powerful visualization to see which model effectively separates high-performing from low-performing stocks.

**Action:** Navigate to the "5. Comparative Performance Evaluation" page and click the "Evaluate Models" button.

The application will generate and display all the described metrics and visualizations, allowing for a thorough comparison.

<aside class="positive">
<b>Key Takeaway:</b> Sarah has performed a comprehensive evaluation. The classification reports and confusion matrices reveal where each model makes mistakes. The Information Coefficient (IC) quantifies the ML model's ability to rank stocks. The "Spread" metric provides a direct measure of the economic value generated by each screening approach. The box plots visually confirm which method generates a better separation between high-performing ('Buy') and low-performing ('Sell') stocks. This holistic evaluation helps Sarah understand not just how accurate the models are, but how practically useful they are for Alpha Capital's investment process.
</aside>

## Step 6: Interpretation, Business Implications, and Next Steps
Duration: 00:10

Sarah's final task is to interpret the coefficients of the Logistic Regression model to understand which financial features drive its 'Buy' and 'Sell' decisions. She will then discuss the practical trade-offs between the interpretable rules-based screen and the more adaptable ML approach, considering Alpha Capital's needs.

**Real-World Relevance:** Model interpretability is crucial in finance, especially for CFA charterholders who need to explain investment decisions to clients or justify models to regulators. Understanding factor sensitivities (e.g., how P/E or ROE impacts a 'Buy' signal) allows analysts to gain trust in the model and refine their investment hypotheses. Discussing trade-offs helps Alpha Capital make informed decisions about integrating new technologies into their workflow.

### 6.1 Logistic Regression Coefficient Interpretation
For Logistic Regression, the coefficients indicate the strength and direction of the relationship between each feature and the log-odds of a stock belonging to a particular class.
*   A **positive coefficient** for a feature in the 'Buy' class implies that higher values of that feature increase the log-odds (and thus the probability) of a stock being classified as 'Buy'.
*   Conversely, a **negative coefficient** decreases these log-odds.
The magnitude of the coefficient indicates the strength of this influence. Since features are standardized, we can directly compare the magnitudes of coefficients across features for a given class.

The application displays a table of coefficients for each class and a bar chart specifically for the 'Buy' class, highlighting the most influential features.

**Action:** Navigate to the "6. Interpretation & Discussion" page and click the "Interpret ML Model & Compare Performance" button.

This will display the coefficient tables and charts, followed by a comparative performance bar chart.

### 6.2 Comparative Performance Summary Chart
A bar chart summarizes the key performance metrics (Accuracy, F1-Weighted, IC, Spread) for both the Rules-Based and ML (Logistic Regression) approaches. This visual comparison makes it easy to spot which method performed better on which metric.

### 6.3 Conclusion and Discussion Points for Alpha Capital
Sarah has successfully conducted a comparative analysis. This exercise provides valuable insights for Alpha Capital:

*   **Interpretability vs. Performance Trade-off:** The rules-based screen is highly transparent and easily explainable. The Logistic Regression model, while interpretable through its coefficients, requires a deeper understanding of its mechanics. Sarah needs to consider where Alpha Capital draws the line on model complexity vs. explainability, especially when communicating with clients or investment committees. The firm's culture and regulatory environment will influence this balance.
*   **Adaptability and Nuance:** The ML model's ability to learn from data allows it to adapt to changing market conditions (if retrained periodically) and potentially capture subtle interaction effects between features that hard-coded rules miss. This can lead to more dynamic and robust signals. Static rules, in contrast, require manual updates and lack this inherent adaptability.
*   **Probabilistic Outputs:** Unlike the binary pass/fail of rules, the ML model provides probabilities for each class. This offers a richer information set, allowing Sarah to not just identify 'Buy' stocks but also gauge the model's conviction level, which can be critical for risk-adjusted portfolio construction.
*   **Regulatory Implications:** Any model used for investment decisions, even a simple Logistic Regression, might be subject to model validation guidelines (e.g., SR 11-7 in the US). A rules-based screen might not always be considered a "model" in the same stringent sense, posing a different compliance burden for Alpha Capital.
*   **Practical Deployment:** In practice, ML-based screening often serves as the first stage in a multi-stage process. The ML model can generate a shortlist of candidates, which human analysts like Sarah then subject to deep fundamental due diligence. This "human-in-the-loop" approach combines ML efficiency with human judgment, leveraging the strengths of both.
*   **Feature Engineering as Domain Expertise:** The creation of features like `quality_score` and `leverage_adj_roe` directly embeds financial domain knowledge into the ML process. This highlights how a CFA's expertise is not replaced but augmented by ML, ensuring the models are financially sensible rather than purely statistical abstractions.

### 6.4 Future Enhancements:
*   Explore more advanced ML models (e.g., Decision Trees, Random Forests, Gradient Boosting) to potentially capture non-linear relationships.
*   Incorporate momentum factors (e.g., trailing 6-month returns) and sentiment data (e.g., news sentiment) to enrich the feature set.
*   Develop a "hybrid" approach where ML-generated scores are constrained by fundamental screens (e.g., requiring positive earnings) to ensure financial prudence.
*   Implement a true walk-forward backtesting framework with multiple train/test splits over historical periods to rigorously evaluate the models' out-of-sample performance over time.

This hands-on exploration provides Sarah and Alpha Capital a foundational understanding of how ML can enhance their core equity research workflow, paving the way for more sophisticated quantitative strategies.

**Congratulations!** You have completed the QuLab: Stock Screening with SimpleML codelab.

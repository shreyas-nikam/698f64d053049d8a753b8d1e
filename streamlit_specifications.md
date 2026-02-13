
# Streamlit Application Specification: Stock Screening with Simple ML at Alpha Capital

## 1. Application Overview

This Streamlit application provides CFA Charterholders and Investment Professionals with a hands-on, interactive lab experience comparing traditional rules-based stock screening with a simple Machine Learning (ML) classifier (Logistic Regression). The app simulates a real-world workflow at Alpha Capital, guiding the user through data acquisition, cleaning, feature engineering, model training, evaluation, and interpretation.

**High-Level Story Flow:**

1.  **Introduction & Data Acquisition:** Sarah, a Junior Portfolio Manager at Alpha Capital, begins by understanding the lab's purpose and initiating the process to acquire S&P 500 fundamental data.
2.  **Data Preparation & EDA:** Sarah then cleans the raw data, handles missing values and outliers, engineers new financial features, and constructs a multi-class target variable ('Buy', 'Hold', 'Sell'). She performs exploratory data analysis to understand feature distributions and correlations.
3.  **Rules-Based Screening:** Next, Sarah implements Alpha Capital's traditional "Growth at a Reasonable Price" (GARP) rules-based screen, applying hard-coded thresholds to identify investment opportunities.
4.  **ML Model Training:** She then prepares the data for machine learning, performs a train/test split, standardizes features, and trains a Multinomial Logistic Regression model with hyperparameter tuning.
5.  **Comparative Performance:** Sarah evaluates both the rules-based screen and the ML classifier on the same held-out test set, comparing their performance using standard classification metrics and finance-specific indicators like Information Coefficient (IC) and "Spread."
6.  **Interpretation & Discussion:** Finally, Sarah interprets the ML model's coefficients to understand its drivers and engages in a critical discussion about the trade-offs, limitations, and business implications of integrating ML into Alpha Capital's systematic security selection process.

## 2. Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings

# Suppress specific warnings for cleaner output in the Streamlit app
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import functions from source.py
from source import (
    fetch_sp500_tickers,
    fetch_financial_data,
    clean_and_engineer_features,
    create_target_variable,
    rules_based_screen
)
```

### `st.session_state` Design

The following `st.session_state` keys will be initialized, updated, and read to preserve state across user interactions and simulated pages:

*   `st.session_state.page`: Stores the current page selected in the sidebar. Initialized to `'1. Introduction & Data Acquisition'`.
*   `st.session_state.df_raw`: Stores the raw DataFrame after data acquisition. Initialized to `None`.
*   `st.session_state.df_final`: Stores the DataFrame after cleaning, feature engineering, and target variable creation. Initialized to `None`.
*   `st.session_state.df_screened_rules`: Stores the DataFrame with rules-based signals. Initialized to `None`.
*   `st.session_state.X_train_sc`, `st.session_state.X_test_sc`, `st.session_state.y_train`, `st.session_state.y_test`: Scaled training and test features/targets for ML. Initialized to `None`.
*   `st.session_state.le`: `LabelEncoder` object used for target variable encoding. Initialized to `None`.
*   `st.session_state.target_names`: List of target class names (e.g., `['Buy', 'Hold', 'Sell']`). Initialized to `None`.
*   `st.session_state.scaler`: `StandardScaler` object used for feature standardization. Initialized to `None`.
*   `st.session_state.ml_model`: Trained `LogisticRegression` model. Initialized to `None`.
*   `st.session_state.df_test_metadata`: DataFrame containing ticker, actual returns, and rules-based signals for the test set, used for ML evaluation. Initialized to `None`.
*   `st.session_state.y_pred_ml_proba`: ML model predicted probabilities for the test set. Initialized to `None`.
*   `st.session_state.y_pred_ml`: ML model predicted classes for the test set (decoded). Initialized to `None`.
*   `st.session_state.y_test_decoded`: Actual decoded target classes for the test set. Initialized to `None`.
*   `st.session_state.ic_ml`: Calculated Information Coefficient for the ML model. Initialized to `None`.
*   `st.session_state.spread_rules`: Calculated 'Spread' for the rules-based model. Initialized to `None`.
*   `st.session_state.spread_ml`: Calculated 'Spread' for the ML model. Initialized to `None`.
*   `st.session_state.coefficients`: DataFrame of Logistic Regression coefficients. Initialized to `None`.
*   `st.session_state.metrics_df`: DataFrame summarizing comparative performance metrics. Initialized to `None`.

### Application Structure and Flow

The application will use `st.sidebar.selectbox` to navigate between the following pages. Each page will conditionally render its content.

#### Common Application Setup

```python
st.set_page_config(layout="wide", page_title="ML Stock Screening for CFAs")

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = '1. Introduction & Data Acquisition'
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'df_screened_rules' not in st.session_state:
    st.session_state.df_screened_rules = None
if 'X_train_sc' not in st.session_state:
    st.session_state.X_train_sc = None
if 'X_test_sc' not in st.session_state:
    st.session_state.X_test_sc = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'le' not in st.session_state:
    st.session_state.le = None
if 'target_names' not in st.session_state:
    st.session_state.target_names = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'df_test_metadata' not in st.session_state:
    st.session_state.df_test_metadata = None
if 'y_pred_ml_proba' not in st.session_state:
    st.session_state.y_pred_ml_proba = None
if 'y_pred_ml' not in st.session_state:
    st.session_state.y_pred_ml = None
if 'y_test_decoded' not in st.session_state:
    st.session_state.y_test_decoded = None
if 'ic_ml' not in st.session_state:
    st.session_state.ic_ml = None
if 'spread_rules' not in st.session_state:
    st.session_state.spread_rules = None
if 'spread_ml' not in st.session_state:
    st.session_state.spread_ml = None
if 'coefficients' not in st.session_state:
    st.session_state.coefficients = None
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None


# Sidebar Navigation
st.sidebar.title("Navigation")
page_options = [
    '1. Introduction & Data Acquisition',
    '2. Data Preparation & EDA',
    '3. Traditional Rules-Based Screening',
    '4. Machine Learning Model Training',
    '5. Comparative Performance Evaluation',
    '6. Interpretation & Discussion'
]
st.session_state.page = st.sidebar.selectbox("Go to page", page_options, index=page_options.index(st.session_state.page))
```

---

#### Page 1: Introduction & Data Acquisition

**Markdown:**

```python
st.title("Stock Screening with Simple ML: Augmenting Equity Selection at Alpha Capital")

st.markdown(f"Sarah, a CFA Charterholder and Junior Portfolio Manager at Alpha Capital, traditionally relies on well-established, rules-based screens to identify investment opportunities. These screens, often implemented in spreadsheets, apply rigid thresholds to financial ratios like Price-to-Earnings (P/E) and Return on Equity (ROE). While intuitive and transparent, Sarah recognizes their limitations: static rules don't adapt to changing market conditions, arbitrary thresholds can miss subtly attractive stocks, and they often fail to capture complex interaction effects between financial metrics.")

st.markdown(f"Alpha Capital is exploring how basic machine learning (ML) can augment Sarah's workflow, providing more dynamic, nuanced, and data-driven insights. This lab will guide Sarah through a hands-on comparison: she will first replicate her firm's classic 'Growth at a Reasonable Price' (GARP) rules-based screen in Python and then implement a simple ML classifier (Logistic Regression) to perform the same equity selection task. By evaluating both approaches side-by-side, Sarah aims to understand the trade-offs, identify potential for improved alpha generation, and ultimately, enhance Alpha Capital's systematic security selection process. This foundational exercise will bridge traditional quantitative methods with the ML paradigm, setting the stage for more advanced applications.")

st.header("Learning Outcomes:")
st.markdown(f"- Acquire and clean S&P 500 fundamental data, including feature engineering and outlier treatment.")
st.markdown(f"- Implement both a rules-based (GARP) screen and a Logistic Regression classifier for stock selection.")
st.markdown(f"- Perform temporal train/test splitting and hyperparameter tuning for the ML model.")
st.markdown(f"- Evaluate both methods using comprehensive classification and finance-specific metrics.")
st.markdown(f"- Interpret model results and discuss trade-offs between interpretability and performance.")

st.header("1. Setting Up the Environment and Acquiring Financial Data")
st.markdown(f"Sarah's first step is to set up her Python environment and acquire the necessary financial data. For a robust stock screening process, she needs fundamental financial ratios for S&P 500 constituents, along with their subsequent 12-month forward returns to serve as a target for both methods. She'll use `yfinance` for fetching live data and `pandas` for data manipulation. A synthetic fallback is also available if `yfinance` encounters rate limits or connectivity issues.")
st.markdown(f"**Real-World Relevance:** In investment firms like Alpha Capital, data acquisition is the bedrock of any quantitative strategy. Analysts often need to integrate data from various sources (e.g., Bloomberg, Refinitiv, S&P Global Market Intelligence). For preliminary analysis and research, publicly available APIs like `yfinance` are often used to quickly prototype ideas.")
st.markdown(f"Now, Sarah will retrieve the S&P 500 ticker list and fetch the required fundamental data and historical prices. This process can be time-consuming and prone to API rate limits, so handling exceptions is crucial. The `forward_return` needs to be calculated from historical price data, representing the total return over the subsequent 12 months.")

st.markdown(r"The formula for the 12-month forward total return $r_i^{{\text{{fwd}}}}$ for stock $i$ at time $t$ is:")
st.markdown(r"$$ r_i^{{\text{{fwd}}}} = \frac{{P_{{i,t+12}} - P_{{i,t}}}}{{P_{{i,t}}}} $$")
st.markdown(r"where $P_{{i,t}}$ is the adjusted close price of stock $i$ at formation date $t$, and $P_{{i,t+12}}$ is the adjusted close price 12 months later. This forward return will be used to define the target variable.")
```

**UI Interactions and Function Calls:**

```python
if st.button("Fetch S&P 500 Financial Data"):
    with st.spinner("Fetching data... This may take a moment or use synthetic data if API limits are hit."):
        current_date = pd.to_datetime('2023-12-31')
        past_date = current_date - pd.DateOffset(years=2)
        sp500_tickers = fetch_sp500_tickers()
        st.session_state.df_raw = fetch_financial_data(sp500_tickers, start_date=past_date, end_date=current_date)
    
    if st.session_state.df_raw is not None and not st.session_state.df_raw.empty:
        st.success(f"Data acquired successfully! Raw data shape: {st.session_state.df_raw.shape[0]} stocks, {st.session_state.df_raw.shape[1]} features.")
        st.dataframe(st.session_state.df_raw.head())
        st.markdown(f"Sarah has successfully acquired a dataset of S&P 500 stocks, including various fundamental financial ratios and a simulated 12-month forward return. The log-transformation of `market_cap` is a common practice to normalize its skewed distribution. While `yfinance` provides current fundamental data, the forward return was simulated to ensure a consistent target variable for this lab, as precise historical forward returns for a single point in time are complex to align with current fundamentals using this simple API. For Alpha Capital, obtaining precisely time-aligned fundamental data and future returns from a data vendor would be standard practice. This step ensures she has the raw material for both her traditional and ML-driven screening.")
    else:
        st.error("Failed to fetch data. Please try again or check your internet connection.")
else:
    st.info("Click 'Fetch S&P 500 Financial Data' to begin.")
```

---

#### Page 2: Data Preparation & EDA

**Markdown:**

```python
st.header("2. Data Cleaning, Feature Engineering, and Exploratory Data Analysis")
st.markdown(f"Before Sarah can apply any screening method, she needs to clean the data and engineer additional features that might provide more predictive power. This includes handling missing values, treating outliers, and creating new financial ratios that capture specific investment theses.")
st.markdown(f"**Real-World Relevance:** Data quality is paramount in finance. Missing data can bias results, and outliers can distort model training. Feature engineering, where domain expertise (like Sarah's CFA background) is applied to create meaningful new variables, is often the most impactful step in quantitative modeling. Z-score standardization is essential for many ML models like Logistic Regression, which are sensitive to feature scales.")

st.subheader("Missing Value Treatment:")
st.markdown(f"*   Drop stocks with `>30%` missing features (illiquid, new listing).")
st.markdown(f"*   For remaining missing values, apply **sector-median imputation**: replace missing $x_{{ij}}$ with the median of feature $j$ within the stock's GICS sector. This is financially motivated—a missing ROE is better approximated by sector peers than the global median.")

st.subheader("Outlier Treatment (Winsorization):")
st.markdown(f"*   Winsorize each numeric feature at the 1st and 99th percentiles to mitigate the impact of extreme values (e.g., negative earnings producing very large negative P/E).")
st.markdown(r"*   The winsorization formula for a feature $x_{{ij}}$ for stock $i$ and feature $j$ is:")
st.markdown(r"$$ x_{{ij}}^{{\text{{win}}}} = \max(q_{{0.01}}, \min(x_{{ij}}, q_{{0.99}})) $$")
st.markdown(r"where $q_{{0.01}}$ and $q_{{0.99}}$ are the 1st and 99th percentiles of feature $j$'s distribution, respectively.")

st.subheader("Feature Engineering:")
st.markdown(f"*   `earnings_yield = 1/P/E` (inverse of P/E, handles negatives better).")
st.markdown(f"*   `quality_score = ROE * profit_margin` (interaction term capturing profitability quality).")
st.markdown(f"*   `leverage_adj_roe = ROE * (1 - D/E / \\max(D/E))` (penalizes high-leverage ROE).")

st.subheader("Standardization:")
st.markdown(f"*   Apply **z-score standardization** to all numeric features after cleaning and engineering. This scales features to have a mean of 0 and a standard deviation of 1, which is critical for Logistic Regression.")
st.markdown(r"*   The z-score for a feature $x_{{ij}}$ is:")
st.markdown(r"$$ Z_{{ij}} = \frac{{X_{{ij}} - \bar{{X}}_j}}{{S_j}} $$")
st.markdown(r"where $\bar{{X}}_j$ and $S_j$ are the training-set mean and standard deviation of feature $j$.")
```

**UI Interactions and Function Calls:**

```python
if st.session_state.df_raw is not None:
    if st.button("Clean Data, Engineer Features & Create Target"):
        with st.spinner("Processing data..."):
            df_clean = clean_and_engineer_features(st.session_state.df_raw.copy())
            st.session_state.df_final = create_target_variable(df_clean.copy())
        
        st.success(f"Data cleaned, features engineered, and target variable created. Final data shape: {st.session_state.df_final.shape[0]} stocks, {st.session_state.df_final.shape[1]} features.")
        st.dataframe(st.session_state.df_final.head())
        
        st.subheader("Target Variable Distribution:")
        st.dataframe(st.session_state.df_final['target'].value_counts())

        st.subheader("Exploratory Data Analysis (EDA)")

        # Correlation heatmap of features (replicate source.py procedural code)
        st.markdown(f"**Correlation Heatmap of Numeric Features:**")
        numeric_cols_for_corr = st.session_state.df_final.select_dtypes(include=np.number).drop(columns=['forward_return'], errors='ignore')
        if not numeric_cols_for_corr.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(numeric_cols_for_corr.corr(), annot=False, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap of Numeric Features')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No numeric features to plot correlation heatmap.")

        # Distribution plots of features segmented by target class (replicate source.py procedural code)
        st.markdown(f"**Distribution Plots of Features by Target Class:**")
        numeric_features = st.session_state.df_final.select_dtypes(include=np.number).drop(columns=['forward_return'], errors='ignore').columns.tolist()
        
        # Sample 4 features for visualization, ensuring there are enough features
        if len(numeric_features) >= 4:
            sampled_features = np.random.choice(numeric_features, 4, replace=False)
        else:
            sampled_features = numeric_features
        
        if sampled_features:
            num_plots = len(sampled_features)
            n_rows = (num_plots + 1) // 2
            fig_dist, axes_dist = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows))
            axes_dist = axes_dist.flatten() # Flatten for easy iteration
            
            for i, feature in enumerate(sampled_features):
                if i < len(axes_dist):
                    sns.histplot(data=st.session_state.df_final, x=feature, hue='target', kde=True, palette='viridis', multiple='stack', ax=axes_dist[i])
                    axes_dist[i].set_title(f'Distribution of {feature} by Target Class')
            
            # Hide any unused subplots
            for j in range(num_plots, len(axes_dist)):
                fig_dist.delaxes(axes_dist[j])

            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)
        else:
            st.warning("Not enough numeric features to plot distributions.")

        st.markdown(f"Sarah has completed a critical data preparation phase. Missing values have been imputed using sector medians, a financially sound approach that leverages peer group information. Outliers have been tamed through winsorization, preventing extreme data points from unduly influencing models. Crucially, new features like `earnings_yield`, `quality_score`, and `leverage_adj_roe` have been engineered, reflecting deeper financial insights than raw ratios alone. The target variable (`Buy`, `Hold`, `Sell`) is now constructed based on forward return quantiles, translating continuous returns into actionable categories for classification. The correlation heatmap helps identify potential multicollinearity, and the distribution plots offer visual insights into how feature values are spread across the different target classes, giving Sarah an initial sense of feature importance and separation. This clean, enriched dataset is now ready for both screening methods.")

else:
    st.info("Please complete '1. Introduction & Data Acquisition' first.")
```

---

#### Page 3: Traditional Rules-Based Stock Screening

**Markdown:**

```python
st.header("3. Traditional Rules-Based Stock Screening (GARP)")
st.markdown(f"Sarah will now implement Alpha Capital's traditional \"Growth at a Reasonable Price\" (GARP) screen. This method relies on hard-coded thresholds for specific financial ratios, providing a transparent and easily understandable way to filter stocks.")
st.markdown(f"**Real-World Relevance:** Rules-based screening is fundamental in traditional equity analysis. It reflects established investment philosophies (e.g., value investing, growth investing) and provides clear criteria that are easy to communicate to clients or investment committees. However, the choice of thresholds is often subjective and fixed, which can be a limitation.")
st.markdown(f"The GARP screen applies Boolean logic with hard thresholds on financial ratios. For example, a stock might be classified as \"Buy\" if:")
st.markdown(f"*   P/E ratio is between 0 and 20")
st.markdown(f"*   ROE is greater than 12%")
st.markdown(f"*   Debt-to-Equity ratio is less than 1.5")
st.markdown(f"*   Revenue growth is greater than 5%")
st.markdown(f"*   Profit margin is greater than 8%")
st.markdown(f"Stocks failing these criteria or meeting specific 'Sell' conditions (e.g., very high P/E, very low ROE) are flagged accordingly.")
```

**UI Interactions and Function Calls:**

```python
if st.session_state.df_final is not None:
    if st.button("Apply Rules-Based Screen"):
        with st.spinner("Applying GARP screen..."):
            st.session_state.df_screened_rules = rules_based_screen(st.session_state.df_final.copy())
        
        st.success("Rules-based screen applied successfully!")
        st.subheader("Rules-Based Screen Signal Distribution:")
        st.dataframe(st.session_state.df_screened_rules['signal_rules'].value_counts())
        st.subheader("Sample of Rules-Based Signals:")
        st.dataframe(st.session_state.df_screened_rules[['ticker', 'PE_ratio', 'ROE', 'DE_ratio', 'revenue_growth', 'profit_margin', 'signal_rules', 'target']].head())
        st.markdown(f"Sarah has successfully applied the traditional GARP rules-based screen. Each stock is now classified as 'Buy', 'Hold', or 'Sell' based on predefined thresholds. This provides a clear, interpretable output. For Alpha Capital, this output can be directly fed into an analyst's review process. However, Sarah notes that this method is rigid; a stock with a P/E of 20.1 would be categorized differently from one with 19.9, despite a negligible difference, potentially missing out on good opportunities or including questionable ones based on arbitrary cutoffs. This highlights the \"binary outcomes\" limitation where there's no notion of confidence or probability associated with the signal.")
else:
    st.info("Please complete '2. Data Preparation & EDA' first.")
```

---

#### Page 4: Machine Learning Model Training

**Markdown:**

```python
st.header("4. Preparing Data for Machine Learning & Temporal Split")
st.markdown(f"Before training a machine learning model, Sarah needs to encode categorical features, standardize numerical features, and perform a robust train/test split. Given the time-series nature of financial data, a temporal (walk-forward) split is critical to prevent look-ahead bias.")
st.markdown(f"**Real-World Relevance:** In financial modeling, preventing look-ahead bias is paramount. Using future information to predict the past or present is a common pitfall. A temporal split simulates a real-world scenario where a model trained on historical data is used to predict future outcomes. Encoding categorical variables (like `sector`) allows ML models to incorporate non-numeric information. Standardization ensures that features with larger numerical ranges don't disproportionately influence the model's learning process.")

st.markdown(f"**Practitioner Warning: Look-Ahead Bias:** The target variable uses future returns. In a live setting, these are unknown. This case study uses historical data with proper temporal splitting (train period before test period) to avoid look-ahead bias. Participants must understand that the ML model is trained on past feature-return relationships and tested on a held-out future period, not the same period. Since we are using a single cross-section of data, we will simulate a temporal split by sorting the data (e.g., by market cap or ticker) and taking the first N% as training and remaining as test, but acknowledge this is a simplification of a true walk-forward split over multiple time periods. For a more robust temporal split with a single cross-section, one might split by a date if available, or simply use a random split (stratified) and explicitly note the limitation regarding temporal dynamics. Given the synthetic nature, a stratified random split is appropriate while acknowledging the theoretical superiority of true temporal splitting for time-series data.")

st.header("5. Machine Learning Classifier: Logistic Regression")
st.markdown(f"Now, Sarah will train a Multinomial Logistic Regression model to classify stocks into 'Buy', 'Hold', or 'Sell'. This model learns the relationship between financial features and the target outcome probabilistically, offering a more nuanced approach than hard rules. She will also tune its hyperparameters to optimize performance.")
st.markdown(f"**Real-World Relevance:** Logistic Regression is a simple yet powerful classifier, often used as a baseline in finance due to its interpretability. It's a natural bridge from traditional statistical regression to machine learning. Hyperparameter tuning is crucial for any ML model to prevent overfitting and ensure robust performance on unseen data, a key concern for Alpha Capital.")
st.subheader("Multinomial Logistic Regression Formulation:")
st.markdown(r"The multinomial logistic regression models the probability of each class $k \in \{{\text{{Buy, Hold, Sell}}\}\}$ given the feature vector $\mathbf{{x}}_i$:")
st.markdown(r"$$ P(y_i = k | \mathbf{{x}}_i) = \frac{{\exp(\boldsymbol{{\beta}}_k^\text{{T}} \mathbf{{x}}_i + \beta_{{k,0}})}}{{\sum_{{j=1}}^K \exp(\boldsymbol{{\beta}}_j^\text{{T}} \mathbf{{x}}_i + \beta_{{j,0}})}} $$")
st.markdown(r"where:")
st.markdown(r"*   $\boldsymbol{{\beta}}_k \in \mathbb{{R}}^P$ is the coefficient vector for class $k$.")
st.markdown(r"*   $\beta_{{k,0}}$ is the intercept (bias) for class $k$.")
st.markdown(r"*   $K = 3$ (Buy, Hold, Sell).")
st.markdown(r"*   $P$ is the number of features.")
st.markdown(r"The model parameters are estimated by maximizing the regularized log-likelihood. With L2 regularization (Ridge), the objective function to minimize is typically:")
st.markdown(r"$$ \min_{{\boldsymbol{{\beta}}}} \left[ -\frac{{1}}{{N}} \sum_{{i=1}}^N \log P(y_i | \mathbf{{x}}_i; \boldsymbol{{\beta}}) + \frac{{\lambda}}{{2}} \sum_{{k=1}}^K ||\boldsymbol{{\beta}}_k||_2^2 \right] $$")
st.markdown(r"where $\lambda$ is the L2 regularization strength, which is inversely related to the `C` parameter in `scikit-learn` ($C = 1/\lambda$). A smaller `C` implies stronger regularization.")
```

**UI Interactions and Function Calls:**

```python
if st.session_state.df_screened_rules is not None:
    if st.button("Train ML Model (Logistic Regression)"):
        with st.spinner("Preparing data and training Logistic Regression model... This includes hyperparameter tuning via GridSearchCV."):
            df_ml = st.session_state.df_screened_rules.copy()

            # One-hot encode the sector feature (replicate source.py procedural code)
            if 'sector' not in df_ml.columns:
                df_ml['sector'] = 'Unknown' # Placeholder to avoid errors
            df_ml = pd.get_dummies(df_ml, columns=['sector'], drop_first=True, dtype=int)

            # Define features (X) and target (y)
            feature_cols = [col for col in df_ml.columns if col not in ['ticker', 'forward_return', 'target', 'signal_rules']]
            X = df_ml[feature_cols]
            y = df_ml['target']

            # Encode the target variable (replicate source.py procedural code)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.session_state.le = le
            st.session_state.target_names = le.classes_

            st.write(f"Features (X) shape: {X.shape}")
            st.write(f"Target (y) shape: {y_encoded.shape}")
            st.write(f"Target classes: {st.session_state.target_names}")

            # Temporal (or stratified random for single snapshot) train/test split (replicate source.py procedural code)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
            )
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            # Standardize numeric features (fit on train, transform both train and test) (replicate source.py procedural code)
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            st.session_state.X_train_sc = X_train_sc
            st.session_state.X_test_sc = X_test_sc
            st.session_state.scaler = scaler

            st.write(f"\nTraining features (scaled) shape: {X_train_sc.shape}")
            st.write(f"Test features (scaled) shape: {X_test_sc.shape}")

            # Store ticker and actual returns for the test set for later analysis
            st.session_state.df_test_metadata = df_ml.loc[X_test.index, ['ticker', 'forward_return', 'target', 'signal_rules']].copy()

            # Hyperparameter Tuning using GridSearchCV (replicate source.py procedural code)
            param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
            lr_model = LogisticRegression(
                penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42
            )
            grid_search = GridSearchCV(
                estimator=lr_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_sc, y_train)
            best_C = grid_search.best_params_['C']
            st.write(f"\nBest regularization parameter C found: {best_C}")

            # Train the final Logistic Regression model with the best C
            ml_model = LogisticRegression(
                C=best_C, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42
            )
            ml_model.fit(X_train_sc, y_train)
            st.session_state.ml_model = ml_model

            # Make predictions on the test set
            y_pred_ml_encoded = ml_model.predict(X_test_sc)
            st.session_state.y_pred_ml_proba = ml_model.predict_proba(X_test_sc)

            # Decode predictions back to original class names
            st.session_state.y_pred_ml = le.inverse_transform(y_pred_ml_encoded)
            st.session_state.y_test_decoded = le.inverse_transform(y_test)

            st.success("Logistic Regression model trained and predictions made.")
            st.markdown(f"Sarah has successfully trained a Multinomial Logistic Regression model and optimized its key hyperparameter, `C` (regularization strength), using `GridSearchCV`. This systematic tuning process helps Alpha Capital ensure the model generalizes well to unseen data, mitigating the risk of overfitting common in financial markets with limited historical data. The model now provides both discrete 'Buy', 'Hold', 'Sell' predictions and, importantly, the underlying probabilities. These probabilities offer a richer understanding of conviction than binary classifications, allowing Sarah to assess the model's confidence in its signals. The next step is to evaluate these predictions against both the actual forward returns and the rules-based approach.")
else:
    st.info("Please complete '3. Traditional Rules-Based Screening' first.")
```

---

#### Page 5: Comparative Performance Evaluation

**Markdown:**

```python
st.header("6. Comparing Rules-Based vs. ML Classifier Performance")
st.markdown(f"This is the core comparison point. Sarah will evaluate both the traditional rules-based screen and the newly trained Logistic Regression model on the same held-out test set. She will use standard classification metrics and, crucially, finance-specific metrics like Information Coefficient (IC) and \"Spread\" to assess their practical value for Alpha Capital.")
st.markdown(f"**Real-World Relevance:** For an investment firm, model evaluation goes beyond simple accuracy. Metrics like Precision for 'Buy' signals (how many predicted buys actually performed well), Recall for 'Sell' signals (how many actual underperformers were caught), and the economic \"Spread\" (return difference between Buy and Sell) directly translate into portfolio performance and risk management. The Information Coefficient (IC) measures the rank correlation of predictions with actual returns, indicating a model's ability to rank stocks, which is highly relevant for portfolio construction.")

st.subheader("Evaluation Metrics:")
st.markdown(f"*   **Accuracy:** Overall correct classifications.")
st.markdown(r"$$ \text{{Accuracy}} = \frac{{\text{{Number of correct predictions}}}}{{N_{{\text{{test}}}}}} $$")
st.markdown(r"where $N_{\text{test}}$ is the number of samples in the test set.")
st.markdown(f"*   **Precision (per class $k$):** Of all stocks predicted as class $k$, how many truly belong to class $k$?")
st.markdown(r"$$ \text{{Precision}}_k = \frac{{\text{{TP}}_k}}{{\text{{TP}}_k + \text{{FP}}_k}} $$")
st.markdown(r"where $\text{TP}_k$ is the number of true positives for class $k$ and $\text{FP}_k$ is the number of false positives for class $k$.")
st.markdown(f"*   **Recall (per class $k$):** Of all stocks truly belonging to class $k$, how many were correctly identified?")
st.markdown(r"$$ \text{{Recall}}_k = \frac{{\text{{TP}}_k}}{{\text{{TP}}_k + \text{{FN}}_k}} $$")
st.markdown(r"where $\text{FN}_k$ is the number of false negatives for class $k$.")
st.markdown(f"*   **F1-Score (per class $k$):** Harmonic mean of Precision and Recall.")
st.markdown(r"$$ \text{{F1}}_k = 2 \cdot \frac{{\text{{Precision}}_k \cdot \text{{Recall}}_k}}{{\text{{Precision}}_k + \text{{Recall}}_k}} $$")
st.markdown(f"*   **Information Coefficient (IC):** Measures the rank correlation between the ML model's Buy probability (or other ranking score) and actual forward returns. IC $> 0.05$ is often considered meaningful in quantitative investing.")
st.markdown(r"$$ \text{{IC}} = \text{{Spearman}}(P(\text{{Buy}})_i, r_i^{{\text{{fwd}}}}) $$")
st.markdown(r"where $P(\text{Buy})_i$ is the predicted probability of a 'Buy' signal for stock $i$ and $r_i^{\text{fwd}}$ is the actual forward return for stock $i$.")
st.markdown(f"*   **Spread:** The average return difference between 'Buy' and 'Sell' signals. This is often the most financially meaningful measure of a screening model's economic value.")
st.markdown(r"$$ \text{{Spread}} = \text{{Avg return}}(\text{{Buy}}) - \text{{Avg return}}(\text{{Sell}}) $$")
st.markdown(f"*   **AUC-ROC (One-vs-Rest):** For multi-class, compute AUC for each class vs. all others using predicted probabilities. Macro-average AUC provides a single summary metric of discriminative power.")
```

**UI Interactions and Function Calls:**

```python
if st.session_state.ml_model is not None and st.session_state.df_test_metadata is not None:
    if st.button("Evaluate Models"):
        # Ensure consistent mapping of rules-based signals to encoded target names (replicate source.py procedural code)
        # Assuming target_names are ['Buy', 'Hold', 'Sell'] due to alphabetical sorting by LabelEncoder
        target_names = st.session_state.target_names
        le = st.session_state.le
        df_test_metadata = st.session_state.df_test_metadata.copy()

        # Map rules-based signals to numerical encoding
        signal_map = {name: le.transform([name])[0] for name in target_names}
        df_test_metadata['signal_rules_encoded'] = df_test_metadata['signal_rules'].map(signal_map).fillna(le.transform(['Hold'])[0])

        # Get predictions from rules-based screen for the test set
        y_pred_rules_encoded = df_test_metadata['signal_rules_encoded'].values.astype(int)

        st.subheader("--- Comparison of Stock Screening Approaches ---")

        # Evaluate Rules-Based Screen (replicate source.py procedural code)
        st.markdown("\n=== Rules-Based Screen Evaluation ===")
        st.text(classification_report(st.session_state.y_test_decoded, df_test_metadata['signal_rules'], target_names=target_names, zero_division=0))

        # Evaluate ML (Logistic Regression) Classifier (replicate source.py procedural code)
        st.markdown("\n=== ML (Logistic Regression) Evaluation ===")
        st.text(classification_report(st.session_state.y_test_decoded, st.session_state.y_pred_ml, target_names=target_names, zero_division=0))

        # --- Confusion Matrices --- (replicate source.py procedural code)
        st.subheader("Confusion Matrices:")
        fig_cm, axes_cm = plt.subplots(1, 2, figsize=(16, 6))

        cm_rules = confusion_matrix(st.session_state.y_test_decoded, df_test_metadata['signal_rules'], labels=target_names)
        sns.heatmap(cm_rules, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=axes_cm[0])
        axes_cm[0].set_title('Confusion Matrix: Rules-Based Screen')
        axes_cm[0].set_xlabel('Predicted Label')
        axes_cm[0].set_ylabel('True Label')

        cm_ml = confusion_matrix(st.session_state.y_test_decoded, st.session_state.y_pred_ml, labels=target_names)
        sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names, ax=axes_cm[1])
        axes_cm[1].set_title('Confusion Matrix: ML (Logistic Regression)')
        axes_cm[1].set_xlabel('Predicted Label')
        axes_cm[1].set_ylabel('True Label')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # --- Calculate finance-specific metrics (IC and Spread) --- (replicate source.py procedural code)
        results_df = df_test_metadata.copy()
        results_df['ml_pred_class'] = st.session_state.y_pred_ml
        # Ensure 'Buy' class index is correct for probability extraction
        buy_class_index = le.transform(['Buy'])[0] if 'Buy' in target_names else 0 
        results_df['ml_buy_proba'] = st.session_state.y_pred_ml_proba[:, buy_class_index] 

        # Information Coefficient (IC) for ML model
        ic_ml, _ = spearmanr(results_df['ml_buy_proba'], results_df['forward_return'])
        st.session_state.ic_ml = ic_ml
        st.write(f"\nInformation Coefficient (IC) for ML Model: {ic_ml:.4f}")

        # Calculate Spread for Rules-Based
        buy_returns_rules = results_df[results_df['signal_rules'] == 'Buy']['forward_return']
        sell_returns_rules = results_df[results_df['signal_rules'] == 'Sell']['forward_return']
        spread_rules = buy_returns_rules.mean() - sell_returns_rules.mean() if not buy_returns_rules.empty and not sell_returns_rules.empty else np.nan
        st.session_state.spread_rules = spread_rules
        st.write(f"Spread (Avg Return Buy - Avg Return Sell) for Rules-Based: {spread_rules:.4f}")

        # Calculate Spread for ML
        buy_returns_ml = results_df[results_df['ml_pred_class'] == 'Buy']['forward_return']
        sell_returns_ml = results_df[results_df['ml_pred_class'] == 'Sell']['forward_return']
        spread_ml = buy_returns_ml.mean() - sell_returns_ml.mean() if not buy_returns_ml.empty and not sell_returns_ml.empty else np.nan
        st.session_state.spread_ml = spread_ml
        st.write(f"Spread (Avg Return Buy - Avg Return Sell) for ML: {spread_ml:.4f}")

        # --- ROC Curves (One-vs-Rest) --- (replicate source.py procedural code)
        st.subheader("One-vs-Rest ROC Curves for ML Classifier:")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i, class_label in enumerate(target_names):
            y_true_binary = (st.session_state.y_test == i).astype(int)
            y_score = st.session_state.y_pred_ml_proba[:, i]
            
            # Check if there are at least two unique class labels in y_true_binary
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f'ROC curve for {class_label} (area = {roc_auc:.2f})')
            else:
                st.warning(f"Skipping ROC curve for class {class_label} due to insufficient samples in test set.")

        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.50)')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('One-vs-Rest ROC Curves for ML Classifier')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

        # --- Signal Return Distribution (Box Plots) --- (replicate source.py procedural code)
        st.subheader("Comparative Forward Returns by Signal Class:")
        fig_box, axes_box = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        sns.boxplot(x='signal_rules', y='forward_return', data=results_df, order=target_names, palette='RdYlGn_r', ax=axes_box[0])
        axes_box[0].set_title('Forward Returns by Rules-Based Signal')
        axes_box[0].set_xlabel('Rules-Based Signal')
        axes_box[0].set_ylabel('12-Month Forward Return')
        axes_box[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        sns.boxplot(x='ml_pred_class', y='forward_return', data=results_df, order=target_names, palette='RdYlGn_r', ax=axes_box[1])
        axes_box[1].set_title('Forward Returns by ML (Logistic Regression) Signal')
        axes_box[1].set_xlabel('ML Signal')
        axes_box[1].set_ylabel('')
        axes_box[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        fig_box.suptitle('Comparative Forward Returns by Signal Class', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig_box)
        plt.close(fig_box)

        st.markdown(f"Sarah has performed a comprehensive evaluation, comparing the rules-based screen and the Logistic Regression model using both standard classification metrics and finance-specific indicators. The classification reports provide a detailed breakdown of precision, recall, and F1-scores for each class ('Buy', 'Hold', 'Sell'), highlighting the strengths and weaknesses of each method. The confusion matrices visually summarize the correct and incorrect classifications, revealing where each model makes mistakes.")
        st.markdown(f"Crucially, the Information Coefficient (IC) quantifies the ML model's ability to rank stocks, with a positive IC suggesting a meaningful predictive edge. The \"Spread\" metric, which is the average return difference between 'Buy' and 'Sell' signals, provides a direct measure of the economic value generated by each screening approach. For Alpha Capital, a larger spread implies a more effective screening tool. The box plots of forward returns segmented by signal class visually confirm which method generates a better separation between high-performing ('Buy') and low-performing ('Sell') stocks. The ROC curves offer insight into the discriminative power of the ML model for each class. This holistic evaluation helps Sarah understand not just how accurate the models are, but how practically useful they are for Alpha Capital's investment process.")

else:
    st.info("Please complete '4. Machine Learning Model Training' first.")
```

---

#### Page 6: Interpretation & Discussion

**Markdown:**

```python
st.header("7. Interpreting the ML Model and Discussing Business Implications")
st.markdown(f"Sarah's final task is to interpret the coefficients of the Logistic Regression model to understand which financial features drive its 'Buy' and 'Sell' decisions. She will then discuss the practical trade-offs between the interpretable rules-based screen and the more adaptable ML approach, considering Alpha Capital's needs.")
st.markdown(f"**Real-World Relevance:** Model interpretability is crucial in finance, especially for CFA charterholders who need to explain investment decisions to clients or justify models to regulators. Understanding factor sensitivities (e.g., how P/E or ROE impacts a 'Buy' signal) allows analysts to gain trust in the model and refine their investment hypotheses. Discussing trade-offs helps Alpha Capital make informed decisions about integrating new technologies into their workflow.")
st.subheader("Coefficient Interpretation:")
st.markdown(f"For Logistic Regression, a positive coefficient for a feature in the 'Buy' class implies that higher values of that feature increase the log-odds of a stock being classified as 'Buy'. Conversely, a negative coefficient decreases these log-odds.")
```

**UI Interactions and Function Calls:**

```python
if st.session_state.ml_model is not None and st.session_state.X_train_sc is not None:
    if st.button("Interpret ML Model & Compare Performance"):
        # Interpret Logistic Regression Coefficients (replicate source.py procedural code)
        ml_model = st.session_state.ml_model
        target_names = st.session_state.target_names
        le = st.session_state.le

        # Need feature column names for coefficients, which were from X before scaling
        # Reconstruct X to get column names:
        df_ml_temp = st.session_state.df_screened_rules.copy()
        if 'sector' not in df_ml_temp.columns:
            df_ml_temp['sector'] = 'Unknown'
        df_ml_temp = pd.get_dummies(df_ml_temp, columns=['sector'], drop_first=True, dtype=int)
        feature_cols_X = [col for col in df_ml_temp.columns if col not in ['ticker', 'forward_return', 'target', 'signal_rules']]
        
        coefficients = pd.DataFrame(ml_model.coef_, columns=feature_cols_X, index=target_names)
        intercepts = pd.Series(ml_model.intercept_, index=target_names, name='Intercept')
        st.session_state.coefficients = coefficients

        st.markdown("\n--- Logistic Regression Coefficients per Class ---")
        st.dataframe(coefficients)
        st.markdown("\n--- Logistic Regression Intercepts per Class ---")
        st.dataframe(intercepts)

        # Visualize coefficients for the 'Buy' class (replicate source.py procedural code)
        st.subheader(f'Logistic Regression Coefficients for "Buy" Class')
        if 'Buy' in target_names:
            buy_class_index = le.transform(['Buy'])[0]
            buy_coefficients = coefficients.loc[target_names[buy_class_index]].sort_values(ascending=False)
            fig_coef, ax_coef = plt.subplots(figsize=(10, 8))
            sns.barplot(x=buy_coefficients.values, y=buy_coefficients.index, palette='coolwarm', ax=ax_coef)
            ax_coef.set_title(f'Logistic Regression Coefficients for "{target_names[buy_class_index]}" Class')
            ax_coef.set_xlabel('Coefficient Value')
            ax_coef.set_ylabel('Feature')
            ax_coef.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_coef)
            plt.close(fig_coef)
        else:
            st.warning("Could not find 'Buy' class for coefficient visualization.")

        st.markdown(f"Sarah has delved into the interpretability of the Logistic Regression model by examining its coefficients. The bar chart for the 'Buy' class coefficients immediately reveals which features are most positively (e.g., high ROE, positive revenue growth) or negatively (e.g., high DE_ratio, low profit margin) associated with a 'Buy' signal. This provides Alpha Capital with actionable insights into the underlying drivers of the ML model's recommendations, allowing Sarah to align them with her existing financial intuition and identify potentially new factor sensitivities.")

        # --- Performance Comparison Bar Chart (Accuracy, F1, Spread) --- (replicate source.py procedural code)
        y_test_decoded = st.session_state.y_test_decoded
        df_test_metadata = st.session_state.df_test_metadata
        y_pred_ml = st.session_state.y_pred_ml
        ic_ml = st.session_state.ic_ml
        spread_rules = st.session_state.spread_rules
        spread_ml = st.session_state.spread_ml

        # Calculate metrics needed for the comparison DataFrame
        acc_rules = accuracy_score(y_test_decoded, df_test_metadata['signal_rules'])
        f1_rules = f1_score(y_test_decoded, df_test_metadata['signal_rules'], average='weighted', zero_division=0)
        acc_ml = accuracy_score(y_test_decoded, y_pred_ml)
        f1_ml = f1_score(y_test_decoded, y_pred_ml, average='weighted', zero_division=0)

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Weighted', 'IC (ML Only)', 'Spread (Buy-Sell)'],
            'Rules-Based': [
                acc_rules,
                f1_rules,
                np.nan,
                spread_rules
            ],
            'ML (Logistic Reg)': [
                acc_ml,
                f1_ml,
                ic_ml,
                spread_ml
            ]
        })
        st.session_state.metrics_df = metrics_df

        st.subheader('Comparative Performance: Rules-Based vs. ML Classifier')
        fig_metrics, ax_metrics = plt.subplots(figsize=(12, 7))
        melted_metrics = metrics_df.melt(id_vars='Metric', var_name='Method', value_name='Score')
        sns.barplot(x='Metric', y='Score', hue='Method', data=melted_metrics, palette='viridis', ax=ax_metrics)
        ax_metrics.set_title('Comparative Performance: Rules-Based vs. ML Classifier')
        ax_metrics.set_ylabel('Score')
        ax_metrics.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_metrics)
        plt.close(fig_metrics)
        st.markdown(f"The comparative bar chart clearly summarizes the performance differences across key metrics like accuracy, F1-score, IC, and Spread. This visualization is crucial for presenting the findings to senior portfolio managers or the investment committee, enabling a data-driven discussion on the merits and trade-offs of integrating machine learning into their stock screening process. Sarah can now articulate how the ML model, while more complex, might offer a more nuanced and potentially more profitable approach than rigid rules.")

        st.header("Conclusion and Discussion Points for Alpha Capital")
        st.markdown(f"Sarah has successfully conducted a comparative analysis of traditional rules-based stock screening and a simple Machine Learning (ML) classifier using Logistic Regression. This exercise provides valuable insights for Alpha Capital.")

        st.subheader("Discussion Points:")
        st.markdown(f"*   **Interpretability vs. Performance Trade-off:** The rules-based screen is highly transparent and easily explainable (e.g., \"P/E under 20 and ROE over 12%\"). The Logistic Regression model, while interpretable through its coefficients, requires a deeper understanding of its mechanics. Sarah needs to consider where Alpha Capital draws the line on model complexity vs. explainability, especially when communicating with clients or investment committees. The firm's culture and regulatory environment will influence this balance.")
        st.markdown(f"*   **Adaptability and Nuance:** The ML model's ability to learn from data allows it to adapt to changing market conditions (if retrained periodically) and potentially capture subtle interaction effects between features that hard-coded rules miss. This can lead to more dynamic and robust signals. Static rules, in contrast, require manual updates and lack this inherent adaptability.")
        st.markdown(f"*   **Probabilistic Outputs:** Unlike the binary pass/fail of rules, the ML model provides probabilities for each class. This offers a richer information set, allowing Sarah to not just identify 'Buy' stocks but also gauge the model's conviction level, which can be critical for risk-adjusted portfolio construction.")
        st.markdown(f"*   **Regulatory Implications:** As a CFA Charterholder, Sarah is aware of the regulatory landscape. Any model used for investment decisions, even a simple Logistic Regression, might be subject to model validation guidelines (e.g., SR 11-7 in the US). A rules-based screen might not always be considered a \"model\" in the same stringent sense, posing a different compliance burden for Alpha Capital.")
        st.markdown(f"*   **Practical Deployment:** In practice, ML-based screening often serves as the first stage in a multi-stage process. The ML model can generate a shortlist of candidates, which human analysts like Sarah then subject to deep fundamental due diligence. This \"human-in-the-loop\" approach combines ML efficiency with human judgment, leveraging the strengths of both.")
        st.markdown(f"*   **Feature Engineering as Domain Expertise:** The creation of features like `quality_score` and `leverage_adj_roe` directly embeds financial domain knowledge into the ML process. This highlights how a CFA's expertise is not replaced but augmented by ML, ensuring the models are financially sensible rather than purely statistical abstractions.")

        st.subheader("Future Enhancements:")
        st.markdown(f"*   Explore more advanced ML models (e.g., Decision Trees, Random Forests) to potentially capture non-linear relationships.")
        st.markdown(f"*   Incorporate momentum factors (e.g., trailing 6-month returns) to enrich the feature set.")
        st.markdown(f"*   Develop a \"hybrid\" approach where ML-generated scores are constrained by fundamental screens (e.g., requiring positive earnings) to ensure financial prudence.")
        st.markdown(f"*   Implement a true walk-forward backtesting framework with multiple train/test splits over historical periods to rigorously evaluate the models' out-of-sample performance over time.")
        st.markdown(f"This hands-on exploration provides Sarah and Alpha Capital a foundational understanding of how ML can enhance their core equity research workflow, paving the way for more sophisticated quantitative strategies.")

else:
    st.info("Please complete '5. Comparative Performance Evaluation' first.")
```

This comprehensive `README.md` is designed for your Streamlit application lab project, providing detailed information for developers and users alike.

---

# QuLab: Lab 1: Stock Screening with SimpleML

## Stock Screening with Simple ML: Augmenting Equity Selection at Alpha Capital

This Streamlit application is the first lab project in the QuLab series, focusing on a hands-on comparison between traditional rules-based stock screening and a simple Machine Learning (ML) classifier for equity selection. Developed for Sarah, a CFA Charterholder and Junior Portfolio Manager at Alpha Capital, the lab guides users through the entire process from data acquisition to model interpretation and performance evaluation.

The project addresses the limitations of static rules-based screens by introducing a dynamic, data-driven ML approach using Logistic Regression. By evaluating both methods side-by-side, it aims to highlight trade-offs, identify potential for improved alpha generation, and enhance systematic security selection processes within an investment firm context.

### Learning Outcomes:
*   Acquire and clean S&P 500 fundamental data, including feature engineering and outlier treatment.
*   Implement both a rules-based ("Growth at a Reasonable Price" - GARP) screen and a Logistic Regression classifier for stock selection.
*   Perform temporal train/test splitting and hyperparameter tuning for the ML model.
*   Evaluate both methods using comprehensive classification and finance-specific metrics (e.g., Information Coefficient, Spread).
*   Interpret model results and discuss trade-offs between interpretability and performance, along with business implications.

## ✨ Features

This application offers a multi-step, interactive journey through the stock screening process:

*   **Interactive Streamlit UI**: A multi-page navigation system (sidebar) for a structured lab experience.
*   **Data Acquisition**: Fetches real-time S&P 500 ticker lists and fundamental financial data using `yfinance` (with a synthetic fallback for robustness). Calculates 12-month forward returns.
*   **Robust Data Preparation**:
    *   **Missing Value Imputation**: Employs sector-median imputation, a financially motivated approach.
    *   **Outlier Treatment**: Uses Winsorization at the 1st and 99th percentiles to handle extreme values.
    *   **Feature Engineering**: Creates new insightful financial ratios like `earnings_yield`, `quality_score`, and `leverage_adj_roe`.
    *   **Target Variable Creation**: Categorizes stocks into 'Buy', 'Hold', 'Sell' based on forward return quantiles.
    *   **Data Standardization**: Applies z-score standardization to numeric features, crucial for ML models.
*   **Exploratory Data Analysis (EDA)**: Visualizes data through correlation heatmaps and feature distribution plots by target class.
*   **Traditional Rules-Based Screening**: Implements a "Growth at a Reasonable Price" (GARP) strategy with predefined thresholds for financial ratios.
*   **Machine Learning Model Training**:
    *   **Temporal Train/Test Split**: Simulates a walk-forward split to prevent look-ahead bias.
    *   **Multinomial Logistic Regression**: Trains a classification model to predict 'Buy', 'Hold', or 'Sell' signals.
    *   **Hyperparameter Tuning**: Utilizes `GridSearchCV` to optimize the model's regularization strength.
*   **Comparative Performance Evaluation**:
    *   Evaluates both rules-based and ML models using a comprehensive suite of metrics:
        *   Standard classification metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrices, ROC Curves.
        *   Finance-specific metrics: Information Coefficient (IC) and "Spread" (return difference between Buy and Sell signals).
    *   Visualizes performance comparisons through various plots (confusion matrices, ROC curves, box plots of returns, bar charts of metrics).
*   **Model Interpretation**: Analyzes and visualizes the coefficients of the Logistic Regression model to understand feature importance and drivers of 'Buy'/'Sell' decisions.
*   **Discussion of Business Implications**: Provides a framework for discussing interpretability, adaptability, probabilistic outputs, regulatory implications, and practical deployment within an investment firm like Alpha Capital.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need Python 3.8+ installed on your system.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/quslab-stock-screening.git
    cd quslab-stock-screening
    ```
    (Replace `your-username/quslab-stock-screening` with the actual repository URL.)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    scipy
    yfinance
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **`source.py` module**:
    The application depends on a `source.py` file containing helper functions. For a lab project, you might be provided this file, or you might be expected to implement it yourself. Ensure a `source.py` file exists in the same directory as your main Streamlit application, containing the definitions for functions like:
    *   `fetch_sp500_tickers()`
    *   `fetch_financial_data(tickers, start_date, end_date)`
    *   `clean_and_engineer_features(df_raw)`
    *   `create_target_variable(df_clean)`
    *   `rules_based_screen(df_final)`

    *Example `source.py` structure (placeholders):*
    ```python
    # source.py
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import winsorize

    def fetch_sp500_tickers():
        # Implementation to fetch S&P 500 tickers
        # ... (e.g., from a CSV, Wikipedia, or an API)
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'] # Example

    def fetch_financial_data(tickers, start_date, end_date):
        # Implementation to fetch fundamental data and calculate forward returns
        # ... (This is a complex function involving multiple yfinance calls and return calculations)
        # For simplicity, returning a synthetic dataframe for demonstration
        print("Fetching financial data (using synthetic data for demonstration)...")
        data = {
            'ticker': tickers,
            'PE_ratio': np.random.rand(len(tickers)) * 50 + 5,
            'ROE': np.random.rand(len(tickers)) * 0.3 + 0.05,
            'DE_ratio': np.random.rand(len(tickers)) * 3 + 0.1,
            'revenue_growth': np.random.rand(len(tickers)) * 0.2 - 0.05,
            'profit_margin': np.random.rand(len(tickers)) * 0.15 + 0.03,
            'market_cap': np.exp(np.random.rand(len(tickers)) * 5 + 20), # Log-normal distribution
            'sector': np.random.choice(['Technology', 'Healthcare', 'Financials', 'Industrials'], len(tickers)),
            'forward_return': np.random.rand(len(tickers)) * 0.6 - 0.3 # Simulated 12-month forward return
        }
        df = pd.DataFrame(data)
        df['log_market_cap'] = np.log(df['market_cap'])
        return df.set_index('ticker')

    def clean_and_engineer_features(df_raw):
        # Implementation for missing value, outlier treatment, feature engineering
        df = df_raw.copy()
        # Drop rows with >30% missing
        initial_rows = df.shape[0]
        df.dropna(thresh=df.shape[1] * 0.7, inplace=True)
        print(f"Dropped {initial_rows - df.shape[0]} stocks with >30% missing features.")

        # Sector-median imputation
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df.groupby('sector')[col].transform(lambda x: x.fillna(x.median()))
        # Fill any remaining NaNs with global median if sector median imputation fails
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # Winsorization
        for col in numeric_cols:
            df[col] = winsorize(df[col], limits=[0.01, 0.01])

        # Feature Engineering
        df['earnings_yield'] = df['PE_ratio'].apply(lambda x: 1/x if x != 0 else np.nan)
        df['quality_score'] = df['ROE'] * df['profit_margin']
        max_de = df['DE_ratio'].replace([np.inf, -np.inf], np.nan).max()
        df['leverage_adj_roe'] = df['ROE'] * (1 - df['DE_ratio'] / (max_de if max_de > 0 else 1)) # Handle max_de = 0 or inf

        # Drop original market_cap if log_market_cap is used
        if 'market_cap' in df.columns and 'log_market_cap' in df.columns:
            df = df.drop(columns=['market_cap'])

        return df

    def create_target_variable(df_clean):
        # Implementation to create 'Buy', 'Hold', 'Sell' target
        df = df_clean.copy()
        df['target'] = pd.qcut(df['forward_return'], q=[0, 0.3, 0.7, 1], labels=['Sell', 'Hold', 'Buy'], duplicates='drop')
        return df

    def rules_based_screen(df_final):
        # Implementation for GARP rules-based screening
        df = df_final.copy()
        df['signal_rules'] = 'Hold' # Default
        
        # Buy rules
        buy_condition = (df['PE_ratio'] > 0) & (df['PE_ratio'] <= 20) & \
                        (df['ROE'] > 0.12) & \
                        (df['DE_ratio'] < 1.5) & \
                        (df['revenue_growth'] > 0.05) & \
                        (df['profit_margin'] > 0.08)
        df.loc[buy_condition, 'signal_rules'] = 'Buy'
        
        # Sell rules (simple example)
        sell_condition = (df['PE_ratio'] > 50) | (df['ROE'] < 0.02) | (df['DE_ratio'] > 5)
        df.loc[sell_condition, 'signal_rules'] = 'Sell'

        return df
    ```

## 🏃‍♀️ Usage

1.  **Run the Streamlit application**:
    From your project root directory, run:
    ```bash
    streamlit run app.py
    ```
    (Assuming your main application file is named `app.py`)

2.  **Interact with the UI**:
    *   The application will open in your default web browser (or provide a URL to open).
    *   Navigate through the lab steps using the "Navigation" selectbox in the sidebar.
    *   Click the buttons on each page to trigger data fetching, processing, model training, and evaluation.
    *   Review the displayed dataframes, plots, and analysis results as you progress.

## 📁 Project Structure

```
.
├── README.md
├── app.py                  # Main Streamlit application file
├── source.py               # Helper functions for data processing, feature engineering, etc.
├── requirements.txt        # List of Python dependencies
└── venv/                   # Python virtual environment (optional but recommended)
```

## 🛠️ Technology Stack

*   **Python**: Programming language
*   **Streamlit**: For building interactive web applications
*   **Pandas**: Data manipulation and analysis
*   **NumPy**: Numerical computing
*   **Scikit-learn**: Machine learning (model training, preprocessing, evaluation)
*   **Matplotlib**: Static plotting
*   **Seaborn**: Statistical data visualization
*   **SciPy**: Scientific computing (e.g., for `spearmanr` and `winsorize`)
*   **yfinance**: For fetching financial data (though synthetic data is used in the provided `source.py` example for robustness and quick testing).

## 👋 Contributing

As this is a lab project, direct contributions via pull requests might not be the primary focus. However, if you find any bugs, have suggestions for improvements, or want to propose alternative implementations for the lab steps, feel free to open an issue in the repository.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any questions or inquiries regarding this project, please contact:

*   **Quant University**
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)

---

## License

## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)


import pandas as pd
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import MagicMock, patch

# Dummy data for mocking
@pytest.fixture
def dummy_df_raw():
    return pd.DataFrame({
        'ticker': ['AAPL', 'GOOG', 'MSFT'],
        'market_cap': [2000e9, 1500e9, 1800e9],
        'PE_ratio': [30.0, 25.0, 35.0],
        'ROE': [0.4, 0.3, 0.5],
        'DE_ratio': [0.5, 0.6, 0.4],
        'revenue_growth': [0.15, 0.12, 0.18],
        'profit_margin': [0.2, 0.18, 0.22],
        'sector': ['Technology', 'Technology', 'Technology'],
        'forward_return': [0.1, 0.05, 0.12]
    })

@pytest.fixture
def dummy_df_final(dummy_df_raw):
    df = dummy_df_raw.copy()
    df['target'] = pd.cut(df['forward_return'], bins=[-np.inf, 0.06, 0.11, np.inf], labels=['Sell', 'Hold', 'Buy'])
    df['earnings_yield'] = 1 / df['PE_ratio']
    df['quality_score'] = df['ROE'] * df['profit_margin']
    df['leverage_adj_roe'] = df['ROE'] * (1 - df['DE_ratio'] / df['DE_ratio'].max())
    return df

@pytest.fixture
def dummy_df_screened_rules(dummy_df_final):
    df = dummy_df_final.copy()
    df['signal_rules'] = 'Hold'
    df.loc[df['ROE'] > 0.4, 'signal_rules'] = 'Buy'
    df.loc[df['ROE'] < 0.35, 'signal_rules'] = 'Sell'
    return df

@pytest.fixture
def dummy_ml_data():
    X_train_sc = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_test_sc = np.array([[5.0, 6.0], [7.0, 8.0]])
    y_train = np.array([0, 1])
    y_test = np.array([0, 1])
    y_pred_ml_proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    return X_train_sc, X_test_sc, y_train, y_test, y_pred_ml_proba

# --- Mocking external functions from 'source.py' ---
@pytest.fixture(autouse=True)
def mock_source_functions(monkeypatch, dummy_df_raw, dummy_df_final, dummy_df_screened_rules):
    monkeypatch.setattr("source.fetch_sp500_tickers", lambda: ['AAPL', 'GOOG', 'MSFT'])
    monkeypatch.setattr("source.fetch_financial_data", lambda tickers, start_date, end_date: dummy_df_raw)
    monkeypatch.setattr("source.clean_and_engineer_features", lambda df: dummy_df_final)
    monkeypatch.setattr("source.create_target_variable", lambda df: dummy_df_final)
    monkeypatch.setattr("source.rules_based_screen", lambda df: dummy_df_screened_rules)

# --- Mocking sklearn components ---
@pytest.fixture(autouse=True)
def mock_sklearn(monkeypatch, dummy_ml_data):
    X_train_sc, X_test_sc, y_train, y_test, y_pred_ml_proba = dummy_ml_data

    # Mock LabelEncoder
    mock_le = MagicMock()
    mock_le.fit_transform.return_value = np.array([0, 1, 2]) # Assuming 'Sell', 'Hold', 'Buy'
    mock_le.inverse_transform.side_effect = lambda x: np.array(['Sell', 'Hold', 'Buy'])[x]
    mock_le.classes_ = np.array(['Buy', 'Hold', 'Sell'])
    mock_le.transform.side_effect = lambda x: np.array([0 if val == 'Buy' else (1 if val == 'Hold' else 2) for val in x])
    monkeypatch.setattr("sklearn.preprocessing.LabelEncoder", MagicMock(return_value=mock_le))

    # Mock StandardScaler
    mock_scaler = MagicMock()
    mock_scaler.fit_transform.return_value = X_train_sc
    mock_scaler.transform.return_value = X_test_sc
    monkeypatch.setattr("sklearn.preprocessing.StandardScaler", MagicMock(return_value=mock_scaler))

    # Mock LogisticRegression
    mock_lr_model = MagicMock()
    mock_lr_model.fit.return_value = None
    mock_lr_model.predict.return_value = np.array([2, 0]) # Corresponds to 'Sell', 'Buy'
    mock_lr_model.predict_proba.return_value = y_pred_ml_proba
    mock_lr_model.coef_ = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) # For 3 classes, 2 features
    mock_lr_model.intercept_ = np.array([0.01, 0.02, 0.03])
    monkeypatch.setattr("sklearn.linear_model.LogisticRegression", MagicMock(return_value=mock_lr_model))

    # Mock GridSearchCV
    mock_grid_search = MagicMock()
    mock_grid_search.fit.return_value = None
    mock_grid_search.best_params_ = {'C': 1.0}
    monkeypatch.setattr("sklearn.model_selection.GridSearchCV", MagicMock(return_value=mock_grid_search))

    # Mock train_test_split
    monkeypatch.setattr("sklearn.model_selection.train_test_split", lambda X, y, test_size, random_state, stratify: (X_train_sc, X_test_sc, y_train, y_test))

    # Mock matplotlib and seaborn to prevent plotting in tests
    monkeypatch.setattr("matplotlib.pyplot.subplots", MagicMock(return_value=(MagicMock(), [MagicMock(), MagicMock(), MagicMock(), MagicMock()])))
    monkeypatch.setattr("matplotlib.pyplot.close", MagicMock())
    monkeypatch.setattr("seaborn.heatmap", MagicMock())
    monkeypatch.setattr("seaborn.histplot", MagicMock())
    monkeypatch.setattr("seaborn.boxplot", MagicMock())
    monkeypatch.setattr("seaborn.barplot", MagicMock())

    # Mock scipy.stats.spearmanr
    monkeypatch.setattr("scipy.stats.spearmanr", MagicMock(return_value=(0.1, 0.05))) # Example IC and p-value

    # Mock sklearn.metrics functions
    monkeypatch.setattr("sklearn.metrics.classification_report", MagicMock(return_value="Mock Classification Report"))
    monkeypatch.setattr("sklearn.metrics.confusion_matrix", MagicMock(return_value=np.array([[10,1,0],[0,10,1],[1,0,10]])))
    monkeypatch.setattr("sklearn.metrics.roc_curve", MagicMock(return_value=(np.array([0,0.5,1]), np.array([0,0.5,1]), np.array([0,1,2]))))
    monkeypatch.setattr("sklearn.metrics.auc", MagicMock(return_value=0.75))
    monkeypatch.setattr("sklearn.metrics.accuracy_score", MagicMock(return_value=0.85))
    monkeypatch.setattr("sklearn.metrics.f1_score", MagicMock(return_value=0.82))


# --- Tests for the Streamlit App ---

def test_page_navigation_and_initial_state():
    at = AppTest.from_file("app.py").run()
    
    # Check initial page
    assert at.session_state["page"] == '1. Introduction & Data Acquisition'
    assert "QuLab: Lab 1: Stock Screening with SimpleML" in at.title[0].value
    assert "Stock Screening with Simple ML: Augmenting Equity Selection at Alpha Capital" in at.title[1].value

    # Test navigation to Page 2
    at.selectbox[0].set_value('2. Data Preparation & EDA').run()
    assert at.session_state["page"] == '2. Data Preparation & EDA'
    assert "2. Data Cleaning, Feature Engineering, and Exploratory Data Analysis" in at.header[0].value

    # Test navigation to Page 3
    at.selectbox[0].set_value('3. Traditional Rules-Based Screening').run()
    assert at.session_state["page"] == '3. Traditional Rules-Based Screening'
    assert "3. Traditional Rules-Based Stock Screening (GARP)" in at.header[0].value

    # Test navigation to Page 4
    at.selectbox[0].set_value('4. Machine Learning Model Training').run()
    assert at.session_state["page"] == '4. Machine Learning Model Training'
    assert "4. Preparing Data for Machine Learning & Temporal Split" in at.header[0].value

    # Test navigation to Page 5
    at.selectbox[0].set_value('5. Comparative Performance Evaluation').run()
    assert at.session_state["page"] == '5. Comparative Performance Evaluation'
    assert "6. Comparing Rules-Based vs. ML Classifier Performance" in at.header[0].value

    # Test navigation to Page 6
    at.selectbox[0].set_value('6. Interpretation & Discussion').run()
    assert at.session_state["page"] == '6. Interpretation & Discussion'
    assert "7. Interpreting the ML Model and Discussing Business Implications" in at.header[0].value


def test_page1_data_acquisition(dummy_df_raw):
    at = AppTest.from_file("app.py").run()
    assert at.session_state["page"] == '1. Introduction & Data Acquisition'

    # Click the "Fetch S&P 500 Financial Data" button
    at.button[0].click().run()

    # Verify data is fetched and stored in session state
    assert at.session_state["df_raw"] is not None
    pd.testing.assert_frame_equal(at.session_state["df_raw"], dummy_df_raw)
    assert at.success[0].value == f"Data acquired successfully! Raw data shape: {dummy_df_raw.shape[0]} stocks, {dummy_df_raw.shape[1]} features."
    assert at.dataframe[0].value.equals(dummy_df_raw.head())


def test_page2_data_preparation(dummy_df_final):
    at = AppTest.from_file("app.py").run()
    # Set df_raw in session state to enable this page's functionality
    at.session_state["df_raw"] = dummy_df_final # Use final as raw to simplify mock, it won't be modified by clean_and_engineer_features mock
    at.selectbox[0].set_value('2. Data Preparation & EDA').run()

    # Click the "Clean Data, Engineer Features & Create Target" button
    at.button[0].click().run()

    # Verify data is processed and stored in session state
    assert at.session_state["df_final"] is not None
    pd.testing.assert_frame_equal(at.session_state["df_final"], dummy_df_final)
    assert at.success[0].value.startswith("Data cleaned, features engineered, and target variable created.")
    assert at.dataframe[0].value.equals(dummy_df_final.head())
    assert at.dataframe[1].value.equals(dummy_df_final['target'].value_counts().to_frame())
    # Check for presence of plots (mocked away, so just check for components)
    assert at.markdown[6].value == "**Correlation Heatmap of Numeric Features:**"
    assert at.markdown[7].value == "**Distribution Plots of Features by Target Class:**"


def test_page3_rules_based_screening(dummy_df_screened_rules):
    at = AppTest.from_file("app.py").run()
    # Set df_final in session state
    at.session_state["df_final"] = dummy_df_screened_rules # Use screened as final to simplify mock
    at.selectbox[0].set_value('3. Traditional Rules-Based Screening').run()

    # Click the "Apply Rules-Based Screen" button
    at.button[0].click().run()

    # Verify rules are applied and stored in session state
    assert at.session_state["df_screened_rules"] is not None
    pd.testing.assert_frame_equal(at.session_state["df_screened_rules"], dummy_df_screened_rules)
    assert at.success[0].value == "Rules-based screen applied successfully!"
    assert at.dataframe[0].value.equals(dummy_df_screened_rules['signal_rules'].value_counts().to_frame())
    assert at.dataframe[1].value.equals(dummy_df_screened_rules[['ticker', 'PE_ratio', 'ROE', 'DE_ratio', 'revenue_growth', 'profit_margin', 'signal_rules', 'target']].head())


def test_page4_ml_model_training(dummy_df_screened_rules, dummy_ml_data):
    X_train_sc, X_test_sc, y_train, y_test, y_pred_ml_proba = dummy_ml_data
    at = AppTest.from_file("app.py").run()
    # Set df_screened_rules in session state
    at.session_state["df_screened_rules"] = dummy_df_screened_rules
    at.selectbox[0].set_value('4. Machine Learning Model Training').run()

    # Click the "Train ML Model (Logistic Regression)" button
    at.button[0].click().run()

    # Verify ML model training and related data are stored in session state
    assert at.session_state["ml_model"] is not None
    assert at.session_state["X_train_sc"] is not None
    assert at.session_state["X_test_sc"] is not None
    assert at.session_state["y_train"] is not None
    assert at.session_state["y_test"] is not None
    assert at.session_state["le"] is not None
    assert at.session_state["target_names"] is not None
    assert at.session_state["scaler"] is not None
    assert at.session_state["df_test_metadata"] is not None
    assert at.session_state["y_pred_ml_proba"] is not None
    assert at.session_state["y_pred_ml"] is not None
    assert at.session_state["y_test_decoded"] is not None

    assert "Best regularization parameter C found: 1.0" in at.markdown[6].value
    assert at.success[0].value == "Logistic Regression model trained and predictions made."


def test_page5_comparative_performance_evaluation(dummy_df_screened_rules, dummy_ml_data):
    X_train_sc, X_test_sc, y_train, y_test, y_pred_ml_proba = dummy_ml_data
    at = AppTest.from_file("app.py").run()
    # Set necessary session state variables
    at.session_state["df_screened_rules"] = dummy_df_screened_rules
    at.session_state["le"] = MagicMock()
    at.session_state["le"].transform.side_effect = lambda x: np.array([0 if val == 'Buy' else (1 if val == 'Hold' else 2) for val in x])
    at.session_state["le"].classes_ = np.array(['Buy', 'Hold', 'Sell'])
    at.session_state["target_names"] = np.array(['Buy', 'Hold', 'Sell'])
    at.session_state["ml_model"] = MagicMock()
    at.session_state["y_test"] = y_test
    at.session_state["y_pred_ml_proba"] = y_pred_ml_proba
    at.session_state["y_test_decoded"] = np.array(['Buy', 'Hold']) # Example decoded values
    at.session_state["y_pred_ml"] = np.array(['Sell', 'Buy']) # Example decoded predictions
    
    # Create a dummy df_test_metadata consistent with the test data size
    df_test_metadata_dummy = pd.DataFrame({
        'ticker': ['MSFT', 'GOOG'],
        'forward_return': [0.12, 0.05],
        'target': ['Buy', 'Hold'],
        'signal_rules': ['Buy', 'Sell']
    })
    at.session_state["df_test_metadata"] = df_test_metadata_dummy

    at.selectbox[0].set_value('5. Comparative Performance Evaluation').run()

    # Click the "Evaluate Models" button
    at.button[0].click().run()

    # Verify evaluation metrics and plots are generated and stored
    assert at.session_state["ic_ml"] is not None
    assert at.session_state["spread_rules"] is not None
    assert at.session_state["spread_ml"] is not None

    assert "Information Coefficient (IC) for ML Model: 0.1000" in at.markdown[5].value
    assert "Spread (Avg Return Buy - Avg Return Sell) for Rules-Based:" in at.markdown[6].value
    assert "Spread (Avg Return Buy - Avg Return Sell) for ML:" in at.markdown[7].value

    assert at.text[0].value == "Mock Classification Report"
    assert at.text[1].value == "Mock Classification Report"

    # Check for presence of plots (mocked away, so just check for components)
    assert at.subheader[2].value == "Confusion Matrices:"
    assert at.subheader[3].value == "One-vs-Rest ROC Curves for ML Classifier:"
    assert at.subheader[4].value == "Comparative Forward Returns by Signal Class:"


def test_page6_interpretation_discussion(dummy_df_screened_rules, dummy_ml_data):
    X_train_sc, X_test_sc, y_train, y_test, y_pred_ml_proba = dummy_ml_data
    at = AppTest.from_file("app.py").run()
    # Set necessary session state variables
    at.session_state["df_screened_rules"] = dummy_df_screened_rules
    at.session_state["ml_model"] = MagicMock()
    at.session_state["ml_model"].coef_ = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    at.session_state["ml_model"].intercept_ = np.array([0.01, 0.02, 0.03])
    at.session_state["X_train_sc"] = X_train_sc # Used to reconstruct feature names
    at.session_state["le"] = MagicMock()
    at.session_state["le"].transform.side_effect = lambda x: np.array([0 if val == 'Buy' else (1 if val == 'Hold' else 2) for val in x])
    at.session_state["le"].classes_ = np.array(['Buy', 'Hold', 'Sell'])
    at.session_state["target_names"] = np.array(['Buy', 'Hold', 'Sell'])
    
    # Required for metrics_df
    at.session_state["y_test_decoded"] = np.array(['Buy', 'Hold'])
    df_test_metadata_dummy = pd.DataFrame({
        'ticker': ['MSFT', 'GOOG'],
        'forward_return': [0.12, 0.05],
        'target': ['Buy', 'Hold'],
        'signal_rules': ['Buy', 'Sell']
    })
    at.session_state["df_test_metadata"] = df_test_metadata_dummy
    at.session_state["y_pred_ml"] = np.array(['Sell', 'Buy'])
    at.session_state["ic_ml"] = 0.1
    at.session_state["spread_rules"] = 0.02
    at.session_state["spread_ml"] = 0.03

    at.selectbox[0].set_value('6. Interpretation & Discussion').run()

    # Click the "Interpret ML Model & Compare Performance" button
    at.button[0].click().run()

    # Verify coefficients and performance comparison are displayed and stored
    assert at.session_state["coefficients"] is not None
    assert at.session_state["metrics_df"] is not None

    assert at.markdown[3].value == "--- Logistic Regression Coefficients per Class ---"
    assert at.dataframe[0].value is not None # Coefficients dataframe
    assert at.markdown[4].value == "--- Logistic Regression Intercepts per Class ---"
    assert at.dataframe[1].value is not None # Intercepts dataframe

    # Check for presence of plots (mocked away, so just check for components)
    assert at.subheader[2].value == 'Comparative Performance: Rules-Based vs. ML Classifier'

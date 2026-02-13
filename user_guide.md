id: 698f64d053049d8a753b8d1e_user_guide
summary: Lab 1: Stock Screening with SimpleML User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Lab 1: Stock Screening with SimpleML

## 1. Introduction to Stock Screening and Data Acquisition
Duration: 0:05:00

Welcome to **QuLab: Lab 1: Stock Screening with SimpleML**! In this codelab, you'll step into the shoes of Sarah, a Junior Portfolio Manager at Alpha Capital, to explore how traditional stock screening methods can be augmented by simple machine learning.

Sarah currently relies on rigid, rules-based screens, similar to filtering spreadsheets, to find investment opportunities. While these methods are clear and transparent, they often struggle to adapt to dynamic market conditions or capture complex relationships between financial metrics. This lab will guide you through comparing a classic "Growth at a Reasonable Price" (GARP) rules-based screen with a simple Machine Learning (ML) classifier (Logistic Regression) for equity selection.

This exercise is crucial for understanding the trade-offs between interpretability and performance, identifying avenues for improved investment alpha, and bridging traditional quantitative finance with modern data science techniques.

### Learning Outcomes:
- Acquire and prepare S&P 500 fundamental financial data.
- Implement both a rules-based (GARP) screen and a Logistic Regression classifier for stock selection.
- Evaluate both methods using comprehensive financial and statistical metrics.
- Interpret ML model results and discuss their practical implications.

### Setting Up and Acquiring Financial Data

Your first task is to set up the environment and acquire the necessary financial data. For a robust stock screening process, you need fundamental financial ratios for S&P 500 constituents, alongside their subsequent 12-month forward returns. The forward return will serve as the target variable that both screening methods will try to predict.

<aside class="positive">
<b>Real-World Relevance:</b> In investment firms like Alpha Capital, data acquisition is foundational. Analysts often integrate data from various sources (e.g., Bloomberg, Refinitiv). For preliminary research, publicly available APIs like <code>yfinance</code> are invaluable for quick prototyping.
</aside>

The 12-month forward total return is calculated using the following formula:
$$ r_i^{\text{fwd}} = \frac{P_{i,t+12} - P_{i,t}}{P_{i,t}} $$
where $P$ is the adjusted close price of a stock $i$ at time $t$ and $t+12$ months respectively.

Click the button below to fetch the S&P 500 financial data. This process involves retrieving a list of S&P 500 tickers and then fetching their fundamental data and historical prices.

<button type="button">Fetch S&P 500 Financial Data</button>

After clicking the button, you'll see a confirmation message and the head of the acquired raw data. This dataset includes various financial ratios and a simulated 12-month forward return. The `market_cap` feature is log-transformed, a common practice to normalize its typically skewed distribution. This step provides the raw material for both your traditional and ML-driven screening.

## 2. Preparing Your Data: Cleaning, Engineering, and Understanding
Duration: 0:08:00

Before any screening method can be applied, the raw data needs meticulous cleaning and enhancement. This step ensures data quality and extracts more predictive power from the existing information.

<aside class="positive">
<b>Real-World Relevance:</b> Data quality is paramount in finance. Missing data can bias results, and outliers can distort model training. Feature engineering, where domain expertise (like Sarah's CFA background) is applied, is often the most impactful step in quantitative modeling.
</aside>

### Missing Value Treatment:
- Stocks with a significant percentage (e.g., >30%) of missing features are typically dropped, as they might represent illiquid or newly listed companies.
- For remaining missing values, **sector-median imputation** is applied. This means a missing value is replaced by the median value of that feature within the stock's respective GICS sector. This approach is financially motivated, as peers within the same sector often share similar characteristics.

### Outlier Treatment (Winsorization):
- To mitigate the impact of extreme values (outliers), each numeric feature is **Winsorized** at the 1st and 99th percentiles. This means any value below the 1st percentile is set to the 1st percentile, and any value above the 99th percentile is set to the 99th percentile. This clamps extreme values without removing them entirely.
The Winsorization formula is:
$$ x_{ij}^{\text{win}} = \max(q_{0.01}, \min(x_{ij}, q_{0.99})) $$
where $x_{ij}$ is the $j$-th feature for stock $i$, and $q_{0.01}$ and $q_{0.99}$ are the 1st and 99th quantiles of that feature, respectively.

### Feature Engineering:
New features are created to capture more nuanced financial insights:
- `earnings_yield = 1/P/E`: The inverse of the Price-to-Earnings ratio, often preferred as it handles negative P/E values better and provides a direct measure of earnings power relative to price.
- `quality_score = ROE * profit_margin`: An interaction term that aims to capture the quality of a company's profitability.
- `leverage_adj_roe = ROE * (1 - D/E / max(D/E))`: This metric penalizes Return on Equity (ROE) for companies with high Debt-to-Equity (D/E) ratios, reflecting a more conservative view of profitability.

### Standardization:
- All numeric features are then scaled using **z-score standardization**. This process transforms features to have a mean of 0 and a standard deviation of 1. This is crucial for many machine learning models, like Logistic Regression, which are sensitive to the scale of input features.
The Z-score standardization formula is:
$$ Z_{ij} = \frac{X_{ij} - \bar{X}_j}{S_j} $$
where $X_{ij}$ is the value of feature $j$ for stock $i$, $\bar{X}_j$ is the mean of feature $j$, and $S_j$ is the standard deviation of feature $j$.

Finally, a `target` variable ('Buy', 'Hold', 'Sell') is created based on the quantiles of the `forward_return`. For instance, stocks in the top quantile might be labeled 'Buy', middle quantiles 'Hold', and bottom quantiles 'Sell'. This transforms a continuous return into actionable categories for classification.

Click the button below to perform these data preparation steps and visualize the initial data insights.

<button type="button">Clean Data, Engineer Features & Create Target</button>

After processing, you will see a summary of the `target` variable distribution, a **Correlation Heatmap** of numeric features, and **Distribution Plots** of key features by target class. The correlation heatmap helps identify relationships between variables and potential multicollinearity. The distribution plots offer visual insights into how different financial features are spread across 'Buy', 'Hold', and 'Sell' categories, giving an initial sense of their predictive power.

## 3. Implementing Traditional Rules-Based Stock Screening
Duration: 0:04:00

Now that the data is clean and prepared, you will implement Alpha Capital's traditional "Growth at a Reasonable Price" (GARP) screen. This method relies on predefined, hard-coded thresholds for specific financial ratios to filter stocks.

<aside class="positive">
<b>Real-World Relevance:</b> Rules-based screening is fundamental in traditional equity analysis. It reflects established investment philosophies (e.g., value investing, growth investing) and provides clear, easily communicable criteria. However, the choice of thresholds is often subjective and fixed, leading to potential limitations.
</aside>

The GARP screen applies Boolean logic to financial ratios. For example, a stock might be classified as a "Buy" if it meets several conditions simultaneously, such as:
*   P/E ratio is between 0 and 20 (reasonable price)
*   ROE is greater than 12% (quality growth)
*   Debt-to-Equity ratio is less than 1.5 (financial health)
*   Revenue growth is greater than 5% (growth aspect)
*   Profit margin is greater than 8% (profitability)

Stocks that fail these "Buy" criteria but don't meet explicit "Sell" conditions (e.g., very high P/E, very low ROE) are typically classified as "Hold."

Click the button below to apply this rules-based screen to your prepared dataset.

<button type="button">Apply Rules-Based Screen</button>

Upon completion, you'll see the distribution of signals (`Buy`, `Hold`, `Sell`) from the rules-based screen and a sample of stocks with their relevant financial ratios and the assigned `signal_rules`. This output is highly interpretable, showing exactly why each stock received its signal. However, notice the rigidity: a stock with a P/E of 20.1 might be treated vastly differently from one with 19.9, despite a negligible actual difference. This illustrates a limitation where there's no notion of confidence or probability associated with the signal.

## 4. Training a Machine Learning Model for Stock Selection
Duration: 0:10:00

The next step is to prepare the data specifically for a machine learning model and then train a Logistic Regression classifier. This model learns the relationships between financial features and the target outcome probabilistically, offering a more nuanced approach than rigid rules.

### Preparing Data for Machine Learning & Temporal Split

Before training, it's crucial to encode categorical features (like `sector`), standardize numerical features, and perform a robust train/test split.

<aside class="negative">
<b>Practitioner Warning: Look-Ahead Bias:</b> In financial modeling, preventing look-ahead bias is paramount. Using future information to predict the past or present is a common pitfall. A true temporal split simulates a real-world scenario where a model trained on historical data is used to predict future outcomes. For this specific codelab, given a single cross-section of data, we simulate a temporal split using a stratified random split. While acknowledging the theoretical superiority of true walk-forward temporal splitting for time-series data, this approach ensures a fair evaluation for this synthetic dataset.
</aside>

Categorical features (like `sector`) are converted into a numerical format using **one-hot encoding**, allowing the ML model to process them. The target variable ('Buy', 'Hold', 'Sell') is also numerically encoded. The data is then split into training and testing sets, ensuring that the model is evaluated on unseen data. Numerical features are standardized (as explained in Step 2) using the `StandardScaler`, fitted only on the training data to avoid data leakage.

### Machine Learning Classifier: Logistic Regression

You will now train a Multinomial Logistic Regression model to classify stocks into 'Buy', 'Hold', or 'Sell'. This model is a simple yet powerful classifier, often used as a baseline in finance due to its interpretability. It's a natural bridge from traditional statistical regression to machine learning.

The multinomial logistic regression models the probability of each class $k \in \{\text{Buy, Hold, Sell}\}$ given the feature vector $\mathbf{x}_i$:
$$ P(y_i = k | \mathbf{x}_i) = \frac{\exp(\boldsymbol{\beta}_k^\text{T} \mathbf{x}_i + \beta_{k,0})}{\sum_{j=1}^K \exp(\boldsymbol{\beta}_j^\text{T} \mathbf{x}_i + \beta_{j,0})} $$
where $\boldsymbol{\beta}_k \in \mathbb{R}^P$ is the coefficient vector for class $k$, and $\beta_{k,0}$ is the intercept. This formula estimates the probability of a stock belonging to a certain class based on its features.

With L2 regularization (Ridge), which helps prevent overfitting by penalizing large coefficients, the objective function to minimize is typically:
$$ \min_{\boldsymbol{\beta}} \left[ -\frac{1}{N} \sum_{i=1}^N \log P(y_i | \mathbf{x}_i; \boldsymbol{\beta}) + \frac{\lambda}{2} \sum_{k=1}^K ||\boldsymbol{\beta}_k||_2^2 \right] $$
Here, $\lambda$ controls the strength of the regularization.

**Hyperparameter tuning** is also performed using `GridSearchCV`. This systematic process searches for the best values for the model's parameters (like `C`, the inverse of regularization strength) to optimize performance and prevent overfitting on unseen data.

Click the button below to prepare the data, train the Logistic Regression model, and make predictions on the test set.

<button type="button">Train ML Model (Logistic Regression)</button>

After training, you'll see details about the data shapes, the best regularization parameter `C` found during tuning, and a confirmation message. The model will provide both discrete 'Buy', 'Hold', 'Sell' predictions and the underlying probabilities for each class. These probabilities offer a richer understanding of conviction than binary classifications, allowing you to assess the model's confidence in its signals.

## 5. Comparing Performance: Rules-Based vs. Machine Learning
Duration: 0:12:00

This is the core comparison point of the codelab. You will now evaluate both the traditional rules-based screen and the Logistic Regression model on the same held-out test set. You'll use standard classification metrics and, crucially, finance-specific metrics to assess their practical value for Alpha Capital.

<aside class="positive">
<b>Real-World Relevance:</b> For an investment firm, model evaluation extends beyond simple accuracy. Metrics like Precision for 'Buy' signals (how many predicted buys actually performed well), Recall for 'Sell' signals (how many actual underperformers were caught), and the economic "Spread" (return difference between Buy and Sell) directly translate into portfolio performance and risk management. The Information Coefficient (IC) measures the rank correlation of predictions with actual returns, indicating a model's ability to rank stocks, which is highly relevant for portfolio construction.
</aside>

### Evaluation Metrics:
*   **Accuracy:** The overall proportion of correct classifications.
    $$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{N_{\text{test}}} $$
*   **Precision (per class k):** Of all stocks predicted as class $k$, how many truly belonged to class $k$?
    $$ \text{Precision}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k} $$
    (TP = True Positives, FP = False Positives)
*   **Recall (per class k):** Of all stocks truly belonging to class $k$, how many were correctly identified?
    $$ \text{Recall}_k = \frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k} $$
    (FN = False Negatives)
*   **F1-Score (per class k):** The harmonic mean of Precision and Recall, providing a balanced measure.
    $$ \text{F1}_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k} $$
*   **Information Coefficient (IC):** Measures the rank correlation (Spearman's rho) between the ML model's 'Buy' probability (or other ranking score) and actual forward returns. An IC > 0.05 is often considered meaningful in quantitative investing.
    $$ \text{IC} = \text{Spearman}(P(\text{Buy})_i, r_i^{\text{fwd}}) $$
*   **Spread:** The average return difference between 'Buy' and 'Sell' signals. This is often the most financially meaningful measure of a screening model's economic value.
    $$ \text{Spread} = \text{Avg return}(\text{Buy}) - \text{Avg return}(\text{Sell}) $$
*   **AUC-ROC (One-vs-Rest):** For multi-class classification, this is computed for each class against all others using predicted probabilities, summarizing the model's discriminative power.

Click the button below to evaluate both models and view their comparative performance.

<button type="button">Evaluate Models</button>

After evaluation, you will see detailed **Classification Reports** for both approaches, **Confusion Matrices** visualizing correct and incorrect classifications, the **Information Coefficient (IC)** for the ML model, and the crucial **Spread** for both methods. You will also see **ROC Curves** for the ML model and **Box Plots** of forward returns segmented by signal class, visually confirming which method generates a better separation between high-performing ('Buy') and low-performing ('Sell') stocks. This holistic evaluation helps you understand not just how accurate the models are, but how practically useful they are for Alpha Capital's investment process.

## 6. Interpreting ML Insights and Business Implications
Duration: 0:08:00

Your final task is to interpret the coefficients of the Logistic Regression model to understand which financial features drive its 'Buy' and 'Sell' decisions. Then, you'll discuss the practical trade-offs between the interpretable rules-based screen and the more adaptable ML approach, considering Alpha Capital's strategic needs.

<aside class="positive">
<b>Real-World Relevance:</b> Model interpretability is crucial in finance. Understanding factor sensitivities (e.g., how P/E or ROE impacts a 'Buy' signal) allows analysts to gain trust in the model, refine investment hypotheses, and explain decisions to clients or regulators.
</aside>

### Coefficient Interpretation:
For Logistic Regression, a positive coefficient for a feature in the 'Buy' class implies that higher values of that feature increase the log-odds of a stock being classified as 'Buy'. Conversely, a negative coefficient decreases these log-odds. The magnitude of the coefficient indicates the strength of this relationship.

Click the button below to interpret the ML model's coefficients and view a comparative performance chart.

<button type="button">Interpret ML Model & Compare Performance</button>

You will now see the **Logistic Regression Coefficients per Class** in a table, showing the influence of each feature on the 'Buy', 'Hold', and 'Sell' predictions. A bar chart specifically for the 'Buy' class coefficients will highlight which features are most positively (e.g., high ROE, positive revenue growth) or negatively (e.g., high D/E ratio, low profit margin) associated with a 'Buy' signal. This provides Alpha Capital with actionable insights into the underlying drivers of the ML model's recommendations.

Finally, a **Comparative Performance** bar chart summarizes the differences across key metrics like accuracy, F1-score, IC, and Spread. This visualization is crucial for presenting findings to senior portfolio managers, enabling a data-driven discussion on integrating machine learning into their stock screening process.

### Conclusion and Discussion Points for Alpha Capital

You have successfully conducted a comparative analysis of traditional rules-based stock screening and a simple Machine Learning (ML) classifier using Logistic Regression. This exercise provides valuable insights for Alpha Capital:

### Discussion Points:
*   **Interpretability vs. Performance Trade-off:** The rules-based screen is highly transparent, while the Logistic Regression model, though interpretable through its coefficients, requires a deeper understanding. Alpha Capital needs to decide on the acceptable level of model complexity versus explainability for their firm.
*   **Adaptability and Nuance:** The ML model's ability to learn from data allows it to adapt to changing market conditions (if retrained periodically) and potentially capture subtle interaction effects that hard-coded rules miss. This can lead to more dynamic and robust signals.
*   **Probabilistic Outputs:** Unlike the binary pass/fail of rules, the ML model provides probabilities for each class. This allows you to not just identify 'Buy' stocks but also gauge the model's conviction level, which is critical for risk-adjusted portfolio construction.
*   **Regulatory Implications:** Any model used for investment decisions might be subject to model validation guidelines. A rules-based screen may have a different compliance burden than an ML model.
*   **Practical Deployment:** In practice, ML-based screening often serves as the first stage in a multi-stage process. The ML model can generate a shortlist of candidates, which human analysts then subject to deep fundamental due diligence. This "human-in-the-loop" approach combines ML efficiency with human judgment.
*   **Feature Engineering as Domain Expertise:** The creation of features like `quality_score` and `leverage_adj_roe` directly embeds financial domain knowledge into the ML process. This highlights how a CFA's expertise is augmented, not replaced, by ML.

### Future Enhancements:
*   Explore more advanced ML models (e.g., Decision Trees, Random Forests) to potentially capture non-linear relationships.
*   Incorporate momentum factors (e.g., trailing 6-month returns) to enrich the feature set.
*   Develop a "hybrid" approach where ML-generated scores are constrained by fundamental screens (e.g., requiring positive earnings) to ensure financial prudence.
*   Implement a true walk-forward backtesting framework with multiple train/test splits over historical periods to rigorously evaluate the models' out-of-sample performance over time.

This hands-on exploration provides a foundational understanding of how ML can enhance Alpha Capital's core equity research workflow, paving the way for more sophisticated quantitative strategies.

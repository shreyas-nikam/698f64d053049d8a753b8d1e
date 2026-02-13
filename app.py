import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
)

from source import *

# -----------------------------
# App Settings
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="QuLab — Stock Screening: Rules vs Probabilistic Scoring", layout="wide")

# -----------------------------
# Session State
# -----------------------------
def _init_state():
    defaults = {
        "page": "1. Overview",
        "mode": "3-minute insight",
        "df_raw": None,
        "df_final": None,
        "df_screened_rules": None,
        "ml_model": None,
        "scaler": None,
        "le": None,
        "target_names": None,
        "X_train_sc": None,
        "X_test_sc": None,
        "y_train": None,
        "y_test": None,
        "y_test_decoded": None,
        "y_pred_ml": None,
        "y_pred_ml_proba": None,
        "df_test_metadata": None,
        "ic_ml": None,
        "spread_rules": None,
        "spread_ml": None,
        "coefficients": None,
        "metrics_df": None,
        # Learning/UX controls
        "bucket_pct": 30,            # target construction: top/bottom % (if supported by source)
        "confidence_threshold": 0.60, # act on ML Buy only above this probability
        "show_diagnostics": False,   # ROC, confusion matrices, etc.
        "show_math": False,          # formulas
        "assumptions_ack": False,    # integrity/assumptions acknowledgement
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# -----------------------------
# Helpers (Pedagogy-first UI)
# -----------------------------
def info_card(title: str, body: str):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(body)

def why_this_matters(text: str):
    st.info(f"**Why this matters:** {text}")

def checkpoint(question: str, options: list, correct_index: int, explanation: str, key: str):
    st.markdown("#### Checkpoint")
    choice = st.radio(question, options, key=key, horizontal=False)
    if choice:
        idx = options.index(choice)
        if idx == correct_index:
            st.success("✅ Correct.")
        else:
            st.warning("⚠️ Not quite.")
        st.markdown(explanation)

def require_step(condition: bool, message: str):
    if not condition:
        st.warning(message)
        st.stop()

def short_assumptions_panel():
    st.sidebar.markdown("### Assumptions & Integrity")
    st.sidebar.caption("Keep these in mind while you interpret outputs.")
    items = [
        "This is a **learning app**: outputs illustrate decision logic, not a tradable strategy.",
        "‘Buy/Hold/Sell’ are **buckets** based on forward returns (a simplified teaching device).",
        "Economic usefulness is judged primarily by **Buy–Sell Spread** and **IC**, not just accuracy.",
        "Coefficients describe **associations**, not causality.",
    ]
    for it in items:
        st.sidebar.write(f"• {it}")

def metric_hierarchy_box():
    with st.container(border=True):
        st.markdown("### How to judge the screen (priority order)")
        st.markdown(
            "1) **Spread (economic value):** Do ‘Buy’ names outperform ‘Sell’ names on average?\n\n"
            "2) **IC (ranking usefulness):** Do higher ‘Buy’ probabilities align with higher returns in rank order?\n\n"
            "3) **Operational reliability:** Precision/recall by class (how many false positives / missed risks).\n\n"
            "4) **Accuracy/F1:** Secondary summaries; don’t confuse them with alpha."
        )

def decision_translation_lines():
    st.markdown(
        "- **If Spread increases:** consider the screen as a first-pass shortlist (still requires analyst review).\n"
        "- **If IC increases:** consider using the score for **ranking/weighting** rather than hard Buy/Sell calls.\n"
        "- **If Buy precision is low:** expect analyst time wasted on false positives.\n"
        "- **If Sell recall is low:** risk flags may be missed."
    )

def format_signal_counts(df, col):
    vc = df[col].value_counts(dropna=False)
    out = pd.DataFrame({"Signal": vc.index.astype(str), "Count": vc.values})
    return out

def safe_mean(x):
    return float(np.nanmean(x)) if x is not None and len(x) else np.nan

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")

st.sidebar.divider()
st.sidebar.title("Navigation")
page_options = [
    "1. Overview",
    "2. Load Data",
    "3. Make Inputs Investable",
    "4. Run Classic GARP Rules",
    "5. Create Probability-Based Signals",
    "6. Compare Economic Usefulness",
    "7. Explain Drivers & Discuss"
]
if st.session_state.page not in page_options:
    st.session_state.page = page_options[0]
st.session_state.page = st.sidebar.selectbox("Go to page", page_options, index=page_options.index(st.session_state.page))

st.sidebar.divider()

st.sidebar.title("Learning Mode")
st.session_state.mode = st.sidebar.radio(
    "Choose your time-to-value path",
    ["3-minute insight", "10-minute understanding", "20-minute mastery"],
    index=["3-minute insight", "10-minute understanding", "20-minute mastery"].index(st.session_state.mode),
    help="A faster path shows only the most decision-relevant outputs. Longer paths add diagnostics and deeper intuition builders."
)

short_assumptions_panel()

# Controls that matter for decisions
st.sidebar.divider()
st.sidebar.title("Decision Controls")
st.session_state.confidence_threshold = st.sidebar.slider(
    "Confidence threshold for acting on ML ‘Buy’",
    min_value=0.50,
    max_value=0.90,
    value=float(st.session_state.confidence_threshold),
    step=0.01,
    help="Higher threshold = fewer ‘Buy’ picks, typically higher conviction."
)

st.session_state.show_diagnostics = st.sidebar.toggle(
    "Show diagnostics (confusion matrices, ROC)",
    value=bool(st.session_state.show_diagnostics),
    help="Turn on only if you want deeper diagnostics beyond economic usefulness."
)

st.session_state.show_math = st.sidebar.toggle(
    "Show optional formulas",
    value=bool(st.session_state.show_math),
    help="Keep off for a decision-first experience."
)

# -----------------------------
# Header
# -----------------------------
st.title("QuLab — Stock Screening: Rules vs Probabilistic Scoring")
st.caption("Designed for CFA charterholders and investment professionals: clarity, intuition, and decision usefulness.")
st.divider()

# -----------------------------
# Page 1 — Overview
# -----------------------------
if st.session_state.page == "1. Overview":
    info_card(
        "What you’ll learn",
        "You’ll compare two ways to build a stock shortlist:\n\n"
        "• **Classic rules (GARP):** transparent thresholds you can explain in 30 seconds.\n"
        "• **Probabilistic scoring:** a confidence-weighted signal that can reduce cliff effects.\n\n"
        "You’ll judge them by **economic usefulness** (Spread, IC), not just accuracy."
    )

    why_this_matters(
        "Most investment screens fail not because they’re ‘wrong’, but because users confuse technical metrics "
        "with decision value. This lab keeps the focus on what improves a real workflow."
    )

    st.markdown("### The case context")
    st.markdown(
        "Sarah (CFA, junior PM at Alpha Capital) uses spreadsheet-style screens to find candidates. "
        "They’re committee-friendly, but rigid. Alpha Capital wants to test whether a **probability-based score** "
        "can add nuance without becoming a black box."
    )

    checkpoint(
        "A stock barely misses a P/E cutoff (20.1 vs 19.9). What’s the real issue with hard rules?",
        [
            "There is no issue; cutoffs are precise.",
            "Hard rules create cliff effects: tiny differences can flip decisions.",
            "Hard rules always outperform probability scores."
        ],
        correct_index=1,
        explanation=(
            "**Cliff effects** make screens brittle. A probability score can treat ‘nearby’ stocks more smoothly.\n\n"
            "In practice, PMs often use rules as **guardrails**, then rank within the allowed universe."
        ),
        key="cp_overview"
    )

    st.markdown("### Quick start")
    st.markdown(
        "- If you want a **fast takeaway**, go to **Load Data → Compare Economic Usefulness**.\n"
        "- If you want deeper intuition, follow the pages in order."
    )

# -----------------------------
# Page 2 — Load Data
# -----------------------------
elif st.session_state.page == "2. Load Data":
    st.header("Load a learning dataset (S&P 500 fundamentals)")
    st.markdown(
        "This step gives you a consistent dataset so you can focus on **decision logic** rather than data plumbing."
    )
    why_this_matters("If your inputs are inconsistent or missing, your screen becomes a hidden sector or quality bet.")

    with st.container(border=True):
        st.markdown("#### Integrity check: No peeking")
        st.markdown(
            "We will **train** on one subset and **test** on a separate subset to approximate how a screen behaves "
            "on names it hasn’t seen."
        )
        st.session_state.assumptions_ack = st.checkbox(
            "I understand this is a learning setup and not a live trading backtest.",
            value=bool(st.session_state.assumptions_ack)
        )

    if st.button("Load sample S&P 500 fundamentals (for learning)"):
        require_step(st.session_state.assumptions_ack, "Please acknowledge the integrity check before proceeding.")
        with st.spinner("Loading data..."):
            current_date = pd.to_datetime("2023-12-31")
            past_date = current_date - pd.DateOffset(years=2)
            sp500_tickers = fetch_sp500_tickers()
            st.session_state.df_raw = fetch_financial_data(sp500_tickers, start_date=past_date, end_date=current_date)

    if st.session_state.df_raw is not None and not st.session_state.df_raw.empty:
        st.success(f"Loaded {st.session_state.df_raw.shape[0]} stocks with {st.session_state.df_raw.shape[1]} columns.")
        with st.expander("Preview the dataset (first 10 rows)"):
            st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

        if st.session_state.show_math:
            st.markdown("**Optional formula:** 12-month forward return (teaching target)")
            st.markdown(r"$$ r^{fwd} = \frac{P_{t+12} - P_t}{P_t} $$")
    else:
        st.info("Click the button above to load the dataset.")

# -----------------------------
# Page 3 — Make Inputs Investable
# -----------------------------
elif st.session_state.page == "3. Make Inputs Investable":
    st.header("Make the inputs investable")
    require_step(st.session_state.df_raw is not None, "Please complete **Load Data** first.")

    st.markdown(
        "Before trusting any screen, you need to ensure ratios aren’t misleading due to missing data, outliers, "
        "or peer comparability."
    )

    why_this_matters(
        "Finance ratios can break (negative earnings, extreme leverage). Cleaning reduces the chance you’re ‘learning’ from artifacts."
    )

    with st.expander("What we do (plain English)", expanded=True):
        st.markdown(
            "• **Peer-based fill:** If a metric is missing, use the **sector median** (peer anchor).\n"
            "• **Cap extremes:** Limit outliers so a few broken ratios don’t dominate.\n"
            "• **Create investable composites:** simple features like earnings yield, quality, and leverage-adjusted quality."
        )

    if st.button("Create cleaned inputs + Buy/Hold/Sell buckets"):
        with st.spinner("Preparing investable inputs..."):
            df_clean = clean_and_engineer_features(st.session_state.df_raw.copy())
            # If your source supports bucket sizing, use it; otherwise fall back to existing behavior.
            try:
                st.session_state.df_final = create_target_variable(df_clean.copy(), bucket_pct=st.session_state.bucket_pct)
            except TypeError:
                st.session_state.df_final = create_target_variable(df_clean.copy())

    if st.session_state.df_final is not None and not st.session_state.df_final.empty:
        st.success("Inputs are ready.")
        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown("#### Bucket distribution (Buy/Hold/Sell)")
            st.dataframe(
                st.session_state.df_final["target"].value_counts().rename_axis("Bucket").reset_index(name="Count"),
                use_container_width=True
            )
        with colB:
            st.markdown("#### Missingness quick check (top 10 columns)")
            miss = st.session_state.df_raw.isna().mean().sort_values(ascending=False).head(10)
            st.dataframe(miss.rename("Missing %").reset_index(name="Column"), use_container_width=True)

        if st.session_state.mode != "3-minute insight":
            # EDA as optional / lower priority
            st.subheader("Diagnostics: are inputs behaving sensibly?")
            st.caption("Use these as *sanity checks*, not proof.")

            with st.expander("Redundancy map (correlations)", expanded=False):
                numeric_cols_for_corr = (
                    st.session_state.df_final
                    .select_dtypes(include=np.number)
                    .drop(columns=["forward_return"], errors="ignore")
                )
                if not numeric_cols_for_corr.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_cols_for_corr.corr(), annot=False, cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation Heatmap (Redundancy Map)")
                    st.pyplot(fig)
                    plt.close(fig)
                st.markdown("**Interpretation:** clusters suggest metrics telling the same story (risk of double counting).")

            with st.expander("How do Buy vs Sell differ on key features?", expanded=False):
                # Prefer consistent “investor-relevant” features if present
                preferred = ["earnings_yield", "ROE", "DE_ratio", "profit_margin", "revenue_growth"]
                available = [c for c in preferred if c in st.session_state.df_final.columns]
                if not available:
                    available = (
                        st.session_state.df_final
                        .select_dtypes(include=np.number)
                        .drop(columns=["forward_return"], errors="ignore")
                        .columns.tolist()[:4]
                    )
                dfp = st.session_state.df_final.copy()
                fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
                if len(available) == 1:
                    axes = [axes]
                for i, feature in enumerate(available):
                    sns.histplot(data=dfp, x=feature, hue="target", kde=True, multiple="stack", ax=axes[i])
                    axes[i].set_title(feature)
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("**Interpretation:** heavy overlap suggests weak separation from fundamentals alone in this sample.")

        checkpoint(
            "A high accuracy score always means the screen will generate alpha.",
            ["True", "False"],
            correct_index=1,
            explanation=(
                "**False.** Accuracy can improve even when economic usefulness does not. "
                "Always prioritize **Spread** (economic value) and **IC** (ranking usefulness)."
            ),
            key="cp_inputs"
        )
    else:
        st.info("Click the button above to prepare inputs.")

# -----------------------------
# Page 4 — Classic GARP Rules
# -----------------------------
elif st.session_state.page == "4. Run Classic GARP Rules":
    st.header("Run the classic GARP screen (committee-friendly rules)")
    require_step(st.session_state.df_final is not None, "Please complete **Make Inputs Investable** first.")

    st.markdown(
        "These are simple thresholds you can explain quickly (e.g., valuation, quality, leverage, growth). "
        "They’re transparent—but can be brittle near cutoffs."
    )
    why_this_matters(
        "Rules are great as guardrails, but they create cliff effects and don’t express confidence."
    )

    with st.expander("Illustrative rules (plain English)", expanded=True):
        st.markdown(
            "Example ‘Buy’ logic might require:\n"
            "• Reasonable valuation\n"
            "• Solid profitability/ROE\n"
            "• Controlled leverage\n"
            "• Positive growth and margins\n\n"
            "This lab uses an illustrative GARP template; firms often tune thresholds by sector and market regime."
        )

    if st.button("Apply GARP rules"):
        with st.spinner("Applying rules..."):
            st.session_state.df_screened_rules = rules_based_screen(st.session_state.df_final.copy())

    if st.session_state.df_screened_rules is not None and not st.session_state.df_screened_rules.empty:
        st.success("Rules applied.")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Signal counts")
            st.dataframe(format_signal_counts(st.session_state.df_screened_rules, "signal_rules"), use_container_width=True)
        with c2:
            st.markdown("#### Boundary cases (near common cutoffs)")
            df = st.session_state.df_screened_rules.copy()
            # Show near P/E = 20 if available
            if "PE_ratio" in df.columns:
                near = df.loc[df["PE_ratio"].between(18, 22), ["ticker", "PE_ratio", "signal_rules", "target"]].sort_values("PE_ratio")
                st.dataframe(near.head(15), use_container_width=True)
                st.caption("These highlight cliff effects: tiny differences can flip labels.")
            else:
                st.caption("P/E ratio not available in this dataset snapshot.")

        with st.expander("Preview sample signals (first 15)", expanded=False):
            cols = ["ticker", "signal_rules", "target", "forward_return"]
            for c in ["PE_ratio", "ROE", "DE_ratio", "revenue_growth", "profit_margin"]:
                if c in st.session_state.df_screened_rules.columns:
                    cols.insert(-2, c)
            st.dataframe(st.session_state.df_screened_rules[cols].head(15), use_container_width=True)

        checkpoint(
            "If most names end up as ‘Hold’, what is the most likely issue?",
            ["The market has no opportunities", "Thresholds may be too strict for this universe", "Rules are always wrong"],
            correct_index=1,
            explanation=(
                "Often the screen is **over-restrictive** or poorly tuned to the universe/sector mix. "
                "In practice, PMs calibrate screens to produce a manageable shortlist."
            ),
            key="cp_rules"
        )
    else:
        st.info("Click the button above to apply rules.")

# -----------------------------
# Page 5 — Probability-Based Signals
# -----------------------------
elif st.session_state.page == "5. Create Probability-Based Signals":
    st.header("Create probability-based signals (not a black box)")
    require_step(st.session_state.df_screened_rules is not None, "Please complete **Run Classic GARP Rules** first.")

    st.markdown(
        "Instead of hard cutoffs, this step creates a **confidence-weighted signal**: "
        "the app estimates the odds a stock falls into Buy/Hold/Sell buckets given its fundamentals."
    )

    why_this_matters(
        "A probability score reduces cliff effects and lets you set a conviction bar (e.g., act only on high-confidence buys)."
    )

    with st.container(border=True):
        st.markdown("#### What you get")
        st.markdown(
            "• A **label** (Buy/Hold/Sell)\n"
            "• A **confidence score** (probability) you can use for ranking or sizing\n"
            "• A simple explanation of **what tends to drive Buy vs Sell**"
        )

    if st.session_state.mode == "20-minute mastery":
        st.caption("Mastery mode: we keep more details visible so you can build intuition.")
    else:
        st.caption("Decision-first mode: we keep the focus on outputs you can use.")

    if st.button("Create probability-based signals"):
        with st.spinner("Training a simple probability model and generating out-of-sample predictions..."):
            df_ml = st.session_state.df_screened_rules.copy()

            # One-hot encode sector (if present)
            if "sector" not in df_ml.columns:
                df_ml["sector"] = "Unknown"
            df_ml = pd.get_dummies(df_ml, columns=["sector"], drop_first=True, dtype=int)

            feature_cols = [c for c in df_ml.columns if c not in ["ticker", "forward_return", "target", "signal_rules"]]
            X = df_ml[feature_cols]
            y = df_ml["target"]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.session_state.le = le
            st.session_state.target_names = le.classes_

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
            )
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            st.session_state.scaler = scaler
            st.session_state.X_train_sc = X_train_sc
            st.session_state.X_test_sc = X_test_sc

            st.session_state.df_test_metadata = df_ml.loc[X_test.index, ["ticker", "forward_return", "target", "signal_rules"]].copy()

            # Hyperparameter selection (kept behind the scenes; outcome is what matters)
            param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
            base = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
            gs = GridSearchCV(base, param_grid=param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0)
            gs.fit(X_train_sc, y_train)

            ml_model = LogisticRegression(C=gs.best_params_["C"], penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
            ml_model.fit(X_train_sc, y_train)
            st.session_state.ml_model = ml_model

            y_pred_encoded = ml_model.predict(X_test_sc)
            st.session_state.y_pred_ml_proba = ml_model.predict_proba(X_test_sc)

            st.session_state.y_pred_ml = le.inverse_transform(y_pred_encoded)
            st.session_state.y_test_decoded = le.inverse_transform(y_test)

        st.success("Probability-based signals created.")

    if st.session_state.ml_model is not None:
        # Show results in decision terms
        le = st.session_state.le
        target_names = st.session_state.target_names
        df_test = st.session_state.df_test_metadata.copy()
        df_test["ml_pred_class"] = st.session_state.y_pred_ml

        # Buy probability
        buy_idx = le.transform(["Buy"])[0] if "Buy" in target_names else 0
        df_test["ml_buy_proba"] = st.session_state.y_pred_ml_proba[:, buy_idx]

        # Apply confidence threshold for “actionable buys”
        df_test["ml_actionable_buy"] = np.where(df_test["ml_buy_proba"] >= st.session_state.confidence_threshold, "Actionable Buy", "Not actionable")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Actionable buys (by your confidence threshold)")
            st.dataframe(df_test["ml_actionable_buy"].value_counts().rename_axis("Bucket").reset_index(name="Count"), use_container_width=True)
            st.caption("Use this as a shortlist filter: higher threshold = fewer, higher-conviction candidates.")
        with c2:
            st.markdown("#### Top candidates by Buy probability")
            top = df_test.sort_values("ml_buy_proba", ascending=False).head(15)[
                ["ticker", "ml_buy_proba", "ml_pred_class", "ml_actionable_buy", "signal_rules", "target", "forward_return"]
            ]
            st.dataframe(top, use_container_width=True)

        checkpoint(
            "A probability score is most useful when you…",
            ["Treat it as an oracle and trade it blindly", "Use it to rank and set conviction thresholds", "Ignore it and rely only on cutoffs"],
            correct_index=1,
            explanation=(
                "Probability scores are best as **ranking/conviction tools**: you can shortlist, size, or require a confidence bar."
            ),
            key="cp_ml"
        )
    else:
        st.info("Click the button above to create probability-based signals.")

# -----------------------------
# Page 6 — Compare Economic Usefulness
# -----------------------------
elif st.session_state.page == "6. Compare Economic Usefulness":
    st.header("Compare economic usefulness (what matters to a PM)")
    require_step(st.session_state.ml_model is not None, "Please complete **Create Probability-Based Signals** first.")

    metric_hierarchy_box()

    st.markdown("### Run the comparison")
    if st.button("Evaluate rules vs probability-based signals"):
        le = st.session_state.le
        target_names = st.session_state.target_names
        df_test = st.session_state.df_test_metadata.copy()
        y_true = st.session_state.y_test_decoded
        y_pred_ml = st.session_state.y_pred_ml

        # Map rules signals to ensure alignment
        df_test["ml_pred_class"] = y_pred_ml

        # Buy probability for IC
        buy_idx = le.transform(["Buy"])[0] if "Buy" in target_names else 0
        df_test["ml_buy_proba"] = st.session_state.y_pred_ml_proba[:, buy_idx]

        # IC (rank correlation)
        ic_ml, _ = spearmanr(df_test["ml_buy_proba"], df_test["forward_return"])
        st.session_state.ic_ml = float(ic_ml) if ic_ml is not None else np.nan

        # Spread
        buy_rules = df_test.loc[df_test["signal_rules"] == "Buy", "forward_return"]
        sell_rules = df_test.loc[df_test["signal_rules"] == "Sell", "forward_return"]
        spread_rules = safe_mean(buy_rules) - safe_mean(sell_rules)
        st.session_state.spread_rules = spread_rules

        buy_ml = df_test.loc[df_test["ml_pred_class"] == "Buy", "forward_return"]
        sell_ml = df_test.loc[df_test["ml_pred_class"] == "Sell", "forward_return"]
        spread_ml = safe_mean(buy_ml) - safe_mean(sell_ml)
        st.session_state.spread_ml = spread_ml

        # Summary metrics
        acc_rules = accuracy_score(y_true, df_test["signal_rules"])
        f1_rules = f1_score(y_true, df_test["signal_rules"], average="weighted", zero_division=0)
        acc_ml = accuracy_score(y_true, y_pred_ml)
        f1_ml = f1_score(y_true, y_pred_ml, average="weighted", zero_division=0)

        metrics_df = pd.DataFrame({
            "Metric": ["Spread (Buy–Sell)", "IC (ranking)", "Accuracy", "F1 (weighted)"],
            "Rules": [spread_rules, np.nan, acc_rules, f1_rules],
            "Probability-based": [spread_ml, st.session_state.ic_ml, acc_ml, f1_ml],
        })
        st.session_state.metrics_df = metrics_df

    if st.session_state.metrics_df is not None:
        # Topline decision view
        st.subheader("Decision-first summary")
        st.dataframe(st.session_state.metrics_df, use_container_width=True)

        decision_translation_lines()

        # Guardrails
        with st.expander("Guardrails to prevent misinterpretation", expanded=True):
            st.markdown(
                "• Always display **counts** in Buy/Sell buckets (small samples can exaggerate Spread).\n"
                "• Treat **IC and Spread as noisy**: one split is not proof of persistence.\n"
                "• If accuracy improves but Spread doesn’t, treat the model as **diagnostic**, not investable.\n"
                "• Remember: these results ignore costs, constraints, and risk model effects."
            )

        # Visual: boxplots (high value)
        st.subheader("Economic separation view (most investable chart)")
        df_test = st.session_state.df_test_metadata.copy()
        df_test["ml_pred_class"] = st.session_state.y_pred_ml

        fig_box, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        sns.boxplot(x="signal_rules", y="forward_return", data=df_test, ax=axes[0])
        axes[0].set_title("Forward returns by Rules signal")
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        sns.boxplot(x="ml_pred_class", y="forward_return", data=df_test, ax=axes[1])
        axes[1].set_title("Forward returns by Probability-based signal")
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        st.pyplot(fig_box)
        plt.close(fig_box)

        # Optional diagnostics
        if st.session_state.show_diagnostics:
            st.subheader("Diagnostics (optional)")

            # Classification tables
            st.markdown("#### Operational reliability (precision/recall by class)")
            rules_rep = classification_report(y_true, df_test["signal_rules"], output_dict=True, zero_division=0)
            ml_rep = classification_report(y_true, st.session_state.y_pred_ml, output_dict=True, zero_division=0)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rules**")
                st.dataframe(pd.DataFrame(rules_rep).transpose(), use_container_width=True)
            with c2:
                st.markdown("**Probability-based**")
                st.dataframe(pd.DataFrame(ml_rep).transpose(), use_container_width=True)

            # Confusion matrices
            st.markdown("#### Where mistakes happen (confusion matrices)")
            fig_cm, axes_cm = plt.subplots(1, 2, figsize=(14, 5))
            cm_rules = confusion_matrix(y_true, df_test["signal_rules"], labels=st.session_state.target_names)
            sns.heatmap(cm_rules, annot=True, fmt="d", xticklabels=st.session_state.target_names, yticklabels=st.session_state.target_names, ax=axes_cm[0])
            axes_cm[0].set_title("Rules confusion matrix")

            cm_ml = confusion_matrix(y_true, st.session_state.y_pred_ml, labels=st.session_state.target_names)
            sns.heatmap(cm_ml, annot=True, fmt="d", xticklabels=st.session_state.target_names, yticklabels=st.session_state.target_names, ax=axes_cm[1])
            axes_cm[1].set_title("Probability-based confusion matrix")

            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # ROC curves (kept optional)
            st.markdown("#### Probability separation (ROC) — diagnostic only")
            fig_roc, ax = plt.subplots(figsize=(7, 5))
            for i, lbl in enumerate(st.session_state.target_names):
                y_true_bin = (st.session_state.y_test == i).astype(int)
                y_score = st.session_state.y_pred_ml_proba[:, i]
                if len(np.unique(y_true_bin)) > 1:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
                    ax.plot(fpr, tpr, label=f"{lbl} (AUC={auc(fpr, tpr):.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random (0.50)")
            ax.set_title("ROC curves (diagnostic)")
            ax.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)

# -----------------------------
# Page 7 — Explain Drivers & Discuss
# -----------------------------
elif st.session_state.page == "7. Explain Drivers & Discuss":
    st.header("Explain drivers (build trust without black-boxing)")
    require_step(st.session_state.ml_model is not None, "Please complete **Compare Economic Usefulness** first (or at least train the model).")

    st.markdown(
        "This section translates the model into **factor intuition**: what tends to push a stock toward ‘Buy’ vs away from it."
    )
    why_this_matters(
        "For investment use, you need to defend a shortlist in an IC meeting: what story is the model actually rewarding?"
    )

    if st.button("Show top drivers (simple, committee-ready view)"):
        ml_model = st.session_state.ml_model
        le = st.session_state.le
        target_names = st.session_state.target_names

        # Rebuild feature names the same way as training
        df_tmp = st.session_state.df_screened_rules.copy()
        if "sector" not in df_tmp.columns:
            df_tmp["sector"] = "Unknown"
        df_tmp = pd.get_dummies(df_tmp, columns=["sector"], drop_first=True, dtype=int)
        feature_cols = [c for c in df_tmp.columns if c not in ["ticker", "forward_return", "target", "signal_rules"]]

        coefs = pd.DataFrame(ml_model.coef_, columns=feature_cols, index=target_names)
        st.session_state.coefficients = coefs

        # Committee-ready summaries
        if "Buy" in target_names:
            buy_row = "Buy"
        else:
            buy_row = target_names[0]

        top_pos = coefs.loc[buy_row].sort_values(ascending=False).head(8)
        top_neg = coefs.loc[buy_row].sort_values(ascending=True).head(8)

        c1, c2 = st.columns(2)
        with c1:
            info_card(
                "Top drivers *toward* Buy (associations)",
                "\n".join([f"• **{k}**" for k in top_pos.index])
            )
        with c2:
            info_card(
                "Top drivers *against* Buy (associations)",
                "\n".join([f"• **{k}**" for k in top_neg.index])
            )

        st.warning(
            "Guardrail: these are **associations in this dataset**, not proof of causality. "
            "Drivers can change across regimes and sectors."
        )

        if st.session_state.mode == "20-minute mastery":
            with st.expander("Full coefficient table (advanced)", expanded=False):
                st.dataframe(coefs, use_container_width=True)

            with st.expander("Visual: Buy driver magnitudes (advanced)", expanded=False):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=top_pos.values, y=top_pos.index, ax=ax)
                ax.set_title(f"Top positive Buy associations ({buy_row})")
                st.pyplot(fig)
                plt.close(fig)

    # Discussion prompts (learning-focused)
    st.subheader("Discussion prompts (investment-first)")
    st.markdown(
        "1) Does the probability-based screen behave like a **quality tilt**, a **value tilt**, or something else?\n"
        "2) If you already run a quality factor sleeve, does this screen add diversification—or duplicate exposure?\n"
        "3) Where would you insert human judgment (analyst review) in this workflow?"
    )

    checkpoint(
        "A large positive coefficient means the feature causes higher returns.",
        ["True", "False"],
        correct_index=1,
        explanation=(
            "**False.** Coefficients show what the model associates with the Buy bucket in this dataset. "
            "Causality requires stronger evidence and design."
        ),
        key="cp_drivers"
    )

# -----------------------------
# Footer / License
# -----------------------------
st.caption(
    """
---
## QuantUniversity License

© QuantUniversity 2025  
This app is for **educational purposes only** and is **not intended for commercial use**.

- You **may not copy, share, or redistribute** without explicit permission from QuantUniversity.  
- You **may not delete or modify** this license without authorization.  
- This content may include errors. Please **verify before using**.

All rights reserved. For permissions or commercial licensing, contact: info@qusandbox.com
"""
)

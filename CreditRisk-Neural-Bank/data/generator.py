"""
data/generator.py
Specialized Banking & Credit Risk Portfolio Generator and CSV Preprocessor.
Generates domain-calibrated credit portfolios with realistic financial ratios,
non-linear default dynamics, and macroeconomic stress shocks.
"""
import io
import numpy as np
import pandas as pd


PORTFOLIO_METADATA = {
    "retail_unsecured": {
        "title": "Retail Unsecured Personal Loans",
        "default_lgd": 0.65,
        "avg_ead_k": 18.5,
        "base_rate_pct": 6.50,
        "feature_names": [
            "Debt_to_Income_Pct",
            "Credit_Utilization_Pct",
            "Delinquencies_2Y",
            "Credit_History_Yrs",
            "Loan_to_Income_Ratio",
            "Bureau_Score_Proxy"
        ],
        "target_name": "Default_12M",
        "class_names": ["Performing (0)", "Defaulted (1)"]
    },
    "mortgage_hl": {
        "title": "Residential Mortgage & Home Equity (Basel III)",
        "default_lgd": 0.25,
        "avg_ead_k": 245.0,
        "base_rate_pct": 4.85,
        "feature_names": [
            "LTV_Ratio_Pct",
            "Debt_Service_DSCR",
            "Borrower_FICO_Score",
            "Rate_Shock_Bps",
            "Employment_Tenure_Yrs",
            "Collateral_Haircut_Pct"
        ],
        "target_name": "Default_12M",
        "class_names": ["Performing (0)", "Defaulted (1)"]
    },
    "sme_commercial": {
        "title": "SME & Corporate Commercial Credit Book",
        "default_lgd": 0.45,
        "avg_ead_k": 420.0,
        "base_rate_pct": 5.75,
        "feature_names": [
            "Net_Debt_to_EBITDA",
            "Interest_Coverage_ICR",
            "Current_Liquidity_Ratio",
            "EBITDA_Margin_Pct",
            "Payable_Delays_Days",
            "Asset_Turnover_Ratio"
        ],
        "target_name": "Default_12M",
        "class_names": ["Performing (0)", "Defaulted (1)"]
    },
    "credit_cards": {
        "title": "Revolving Credit Card & Overdraft Facilities",
        "default_lgd": 0.78,
        "avg_ead_k": 8.2,
        "base_rate_pct": 11.20,
        "feature_names": [
            "Revolving_Utilization_Pct",
            "Min_Payment_Freq_Pct",
            "Cash_Advance_Ratio_Pct",
            "Hard_Inquiries_6M",
            "Seasoning_Months",
            "Overlimit_Count_12M"
        ],
        "target_name": "Default_12M",
        "class_names": ["Performing (0)", "Defaulted (1)"]
    }
}


def generate_credit_portfolio(
    preset: str = "retail_unsecured",
    n_samples: int = 320,
    stress_noise: float = 0.16,
    random_state: int = 42
):
    """
    Generates a specialized banking loan portfolio with domain-calibrated financial
    ratios and structural default probabilities (PD).
    """
    rng = np.random.RandomState(random_state)
    if preset not in PORTFOLIO_METADATA:
        preset = "retail_unsecured"

    meta = PORTFOLIO_METADATA[preset]
    feature_names = meta["feature_names"]
    target_name = meta["target_name"]
    class_names = meta["class_names"]

    if preset == "retail_unsecured":
        dti = np.clip(rng.normal(34.0, 13.5, n_samples) + stress_noise * 12.0 * rng.randn(n_samples), 6.0, 75.0)
        util = np.clip(rng.beta(2.2, 2.5, n_samples) * 100.0 + stress_noise * 10.0 * rng.randn(n_samples), 1.0, 99.5)
        delinq = np.clip(np.round(rng.exponential(0.65 + stress_noise * 1.1, n_samples)), 0, 7)
        hist_yrs = np.clip(rng.gamma(3.2, 2.6, n_samples), 0.5, 30.0)
        lti = np.clip(0.12 + 0.006 * dti + rng.uniform(-0.08, 0.25, n_samples), 0.04, 0.95)
        bureau = np.clip(
            760.0 - 1.45 * util - 1.25 * dti - 26.0 * delinq + 2.8 * hist_yrs + rng.normal(0, 18.0, n_samples),
            480.0, 850.0
        )

        # Structural non-linear log-odds of default (including DTI x Utilization interaction)
        z = (
            -3.35
            + 0.052 * (dti - 30.0)
            + 0.038 * (util - 45.0)
            + 0.00075 * np.maximum(0, dti - 32.0) * np.maximum(0, util - 50.0)
            + 0.68 * delinq
            - 0.11 * hist_yrs
            + 2.45 * (lti - 0.30)
            - 0.012 * (bureau - 660.0)
            + rng.normal(0, stress_noise * 2.1, n_samples)
        )
        X = np.column_stack([dti, util, delinq, hist_yrs, lti, bureau])

    elif preset == "mortgage_hl":
        ltv = np.clip(rng.normal(74.0, 14.0, n_samples) + stress_noise * 10.0 * rng.randn(n_samples), 35.0, 108.0)
        dscr = np.clip(rng.normal(1.52, 0.42, n_samples) - stress_noise * 0.25 * rng.randn(n_samples), 0.65, 3.10)
        fico = np.clip(785.0 - 1.3 * np.maximum(0, ltv - 60.0) + 38.0 * (dscr - 1.2) + rng.normal(0, 24.0, n_samples), 500.0, 850.0)
        rate_shock = np.clip(rng.uniform(25.0, 350.0, n_samples) + stress_noise * 80.0 * rng.randn(n_samples), 0.0, 500.0)
        tenure = np.clip(rng.gamma(2.8, 2.5, n_samples), 0.5, 28.0)
        haircut = np.clip(18.0 + 0.22 * np.maximum(0, ltv - 65.0) + rng.normal(0, 4.5, n_samples), 10.0, 58.0)

        z = (
            -3.10
            + 0.068 * (ltv - 75.0)
            - 2.35 * (dscr - 1.35)
            + 0.042 * np.maximum(0, ltv - 80.0) * np.maximum(0, 1.30 - dscr)
            - 0.014 * (fico - 680.0)
            + 0.0065 * (rate_shock - 150.0)
            - 0.095 * tenure
            + 0.035 * (haircut - 25.0)
            + rng.normal(0, stress_noise * 2.0, n_samples)
        )
        X = np.column_stack([ltv, dscr, fico, rate_shock, tenure, haircut])

    elif preset == "sme_commercial":
        debt_ebitda = np.clip(rng.gamma(2.6, 1.25, n_samples) + stress_noise * 0.8 * rng.randn(n_samples), 0.2, 8.5)
        icr = np.clip(6.2 - 0.72 * debt_ebitda + rng.normal(0, 1.1, n_samples), 0.4, 11.0)
        curr_ratio = np.clip(rng.normal(1.55, 0.52, n_samples), 0.45, 3.60)
        ebitda_margin = np.clip(rng.normal(14.5, 7.5, n_samples) - stress_noise * 4.0 * rng.randn(n_samples), -8.0, 42.0)
        pay_delays = np.clip(np.round(rng.exponential(18.0, n_samples) + 4.5 * debt_ebitda), 0.0, 120.0)
        turnover = np.clip(rng.normal(2.4, 0.85, n_samples), 0.5, 6.0)

        z = (
            -2.85
            + 0.68 * (debt_ebitda - 3.2)
            - 0.58 * (icr - 2.5)
            + 0.35 * np.maximum(0, debt_ebitda - 3.5) * np.maximum(0, 2.2 - icr)
            - 0.85 * (curr_ratio - 1.35)
            - 0.065 * (ebitda_margin - 12.0)
            + 0.028 * (pay_delays - 22.0)
            - 0.25 * (turnover - 2.0)
            + rng.normal(0, stress_noise * 2.1, n_samples)
        )
        X = np.column_stack([debt_ebitda, icr, curr_ratio, ebitda_margin, pay_delays, turnover])

    else:  # credit_cards
        rev_util = np.clip(rng.beta(2.0, 2.1, n_samples) * 102.0, 1.0, 108.0)
        min_pay = np.clip(0.18 * rev_util + rng.uniform(0, 65, n_samples), 0.0, 100.0)
        cash_adv = np.clip(rng.exponential(11.0, n_samples) + 0.12 * rev_util, 0.0, 75.0)
        inquiries = np.clip(np.round(rng.poisson(1.6 + stress_noise * 2.0, n_samples)), 0, 10)
        seasoning = np.clip(rng.gamma(3.0, 16.0, n_samples), 3.0, 150.0)
        overlimit = np.clip(np.round(rng.exponential(0.45, n_samples) + (rev_util > 85.0) * 1.2), 0, 6)

        z = (
            -3.20
            + 0.042 * (rev_util - 50.0)
            + 0.028 * (min_pay - 40.0)
            + 0.036 * (cash_adv - 15.0)
            + 0.42 * (inquiries - 1.5)
            - 0.018 * (seasoning - 36.0)
            + 0.72 * overlimit
            + rng.normal(0, stress_noise * 2.0, n_samples)
        )
        X = np.column_stack([rev_util, min_pay, cash_adv, inquiries, seasoning, overlimit])

    probs = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    # Calibrate threshold so portfolio has a realistic stress default distribution (~22%-35% in training sample for balanced evaluation)
    cutoff = np.percentile(probs, 72)
    y = (probs >= min(0.48, cutoff)).astype(int)

    perm = rng.permutation(len(y))
    X = np.round(X[perm], 3)
    y = y[perm]

    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y
    return df, feature_names, target_name, class_names


def load_and_preprocess_csv(csv_content: str, target_col: str = None, feature_cols: list = None):
    """
    Parses a banking credit CSV dataset, imputes missing values with column medians,
    and encodes binary default targets {0: Performing, 1: Defaulted}.
    """
    df = pd.read_csv(io.StringIO(csv_content))
    df = df.dropna(how="all").reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError("Credit portfolio CSV must contain at least 1 financial feature column and 1 target column.")

    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]

    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]
    else:
        feature_cols = [c for c in feature_cols if c in df.columns and c != target_col]

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

    raw_target = df[target_col].astype(str).str.strip()
    unique_vals = sorted(raw_target.unique())

    if len(unique_vals) >= 2:
        pos_label = unique_vals[1]
        y = (raw_target == pos_label).astype(int).values
        if set(unique_vals) == {"0", "1"}:
            class_names = ["Performing (0)", "Defaulted (1)"]
        else:
            class_names = [f"{unique_vals[0]} (0)", f"{unique_vals[1]} (1)"]
    else:
        y = pd.to_numeric(raw_target, errors="coerce").fillna(0).astype(int).values
        class_names = ["Performing (0)", "Defaulted (1)"]

    clean_df = X_df.copy()
    clean_df[target_col] = y

    return clean_df, list(X_df.columns), target_col, class_names

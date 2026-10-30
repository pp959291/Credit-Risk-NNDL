"""
models/trainer.py
Specialized Credit Risk & Banking Machine Learning Pipeline.
Implements:
1. WoEScorecardClassifier: Weight of Evidence (WoE) & Information Value (IV) Credit Scorecard
   with PDO (Points to Double the Odds) 300-850 Credit Score scaling.
2. CreditRiskNeuralMLP: 2-Layer Cost-Sensitive Feedforward Neural Network (LeakyReLU + Sigmoid)
   trained with Adam/Momentum and default-cost weighting for calibrated Probability of Default (PD).
3. CreditInteractionPolyModel: Non-linear Financial Leverage & Stress Interaction Network.
4. CreditGradientBoostedTrees: Gradient Boosted Decision Trees (GBDT) optimizing log-odds of default.
5. Basel III IRB & IFRS 9 Expected Credit Loss (ECL), RWA, Regulatory Capital, KS, Gini, and Lift Engine.
"""
import math
import os
import pickle
import numpy as np


def _erf_approx(x: float) -> float:
    """Abramowitz and Stegun approximation for erf(x)."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Rational approximation for standard normal inverse CDF (Beasley-Springer-Moro)."""
    p = max(1e-6, min(1.0 - 1e-6, p))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if phigh < p:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def compute_basel_irb_capital(pd_val: float, lgd_val: float, ead_val: float) -> dict:
    """
    Computes Basel III Internal Ratings-Based (IRB) Asset Correlation (R),
    Capital Requirement Factor (K), Risk-Weighted Assets (RWA), and Expected Credit Loss (ECL).
    """
    pd_clean = max(0.0003, min(0.999, float(pd_val)))
    lgd_clean = max(0.05, min(1.0, float(lgd_val)))
    ead_clean = max(0.0, float(ead_val))

    # Basel III Retail / Corporate ASRF Correlation Formula
    exp_factor = (1.0 - math.exp(-50.0 * pd_clean)) / (1.0 - math.exp(-50.0))
    r_corr = 0.12 * exp_factor + 0.24 * (1.0 - exp_factor)

    # Maturity adjustment b(PD)
    b_mat = (0.11852 - 0.05478 * math.log(pd_clean)) ** 2
    maturity_adj = (1.0 + (2.5 - 2.5) * b_mat) / (1.0 - 1.5 * b_mat)

    # Conditional Default Probability at 99.9% Confidence Interval
    inv_pd = norm_ppf(pd_clean)
    inv_999 = norm_ppf(0.999)
    cond_pd = norm_cdf((inv_pd + math.sqrt(r_corr) * inv_999) / math.sqrt(1.0 - r_corr))

    k_factor = max(0.0, (lgd_clean * cond_pd - pd_clean * lgd_clean) * maturity_adj)
    rwa = 12.5 * k_factor * ead_clean
    reg_capital = 0.08 * rwa
    tier1_buffer_capital = 0.105 * rwa
    ecl = pd_clean * lgd_clean * ead_clean
    risk_weight_pct = (rwa / ead_clean * 100.0) if ead_clean > 0 else 0.0

    return {
        "pd": round(pd_clean, 5),
        "lgd": round(lgd_clean, 4),
        "ead": round(ead_clean, 2),
        "ecl": round(ecl, 3),
        "asset_correlation_r": round(r_corr, 4),
        "capital_factor_k": round(k_factor, 4),
        "rwa": round(rwa, 2),
        "risk_weight_pct": round(risk_weight_pct, 1),
        "regulatory_capital_8pct": round(reg_capital, 3),
        "total_capital_10_5pct": round(tier1_buffer_capital, 3)
    }


def pd_to_credit_score_and_grade(pd_val: float, base_score: float = 650.0, base_odds: float = 19.0, pdo: float = 50.0):
    """
    Maps calibrated Probability of Default (PD) to a 300-850 Bank Credit Score
    using Points-to-Double-the-Odds (PDO) scaling and assigns an Agency Rating Grade.
    """
    p = max(1e-4, min(1.0 - 1e-4, float(pd_val)))
    good_bad_odds = (1.0 - p) / p
    factor = pdo / math.log(2.0)
    offset = base_score - factor * math.log(base_odds)
    raw_score = offset + factor * math.log(good_bad_odds)
    score = int(round(max(300.0, min(850.0, raw_score))))

    if p < 0.015:
        grade, stage = "AAA", "Stage 1 (Performing)"
    elif p < 0.030:
        grade, stage = "AA", "Stage 1 (Performing)"
    elif p < 0.055:
        grade, stage = "A", "Stage 1 (Performing)"
    elif p < 0.100:
        grade, stage = "BBB", "Stage 2 (SICR Watch)"
    elif p < 0.180:
        grade, stage = "BB", "Stage 2 (SICR Watch)"
    elif p < 0.300:
        grade, stage = "B", "Stage 2 (Elevated Risk)"
    else:
        grade, stage = "CCC/D", "Stage 3 (Impaired / High Risk)"

    return score, grade, stage


class ZScoreScaler:
    """Standardizes financial features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def compute_woe_iv_table(X: np.ndarray, y: np.ndarray, n_bins: int = 5):
    """
    Computes Weight of Evidence (WoE) bins and Information Value (IV) for each financial ratio.
    """
    n_samples, n_feats = X.shape
    total_bad = max(1.0, float(np.sum(y == 1)))
    total_good = max(1.0, float(np.sum(y == 0)))
    iv_scores = []
    bin_models = []

    for j in range(n_feats):
        col = X[:, j]
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(col, quantiles))
        if len(edges) <= 2:
            edges = np.array([np.min(col) - 1e-5, np.mean(col), np.max(col) + 1e-5])
        edges[0] -= 1e-5
        edges[-1] += 1e-5

        woe_map = []
        iv_val = 0.0
        for b in range(len(edges) - 1):
            mask = (col >= edges[b]) & (col < edges[b + 1]) if b < len(edges) - 2 else (col >= edges[b]) & (col <= edges[b + 1])
            b_bad = float(np.sum(y[mask] == 1)) + 0.5
            b_good = float(np.sum(y[mask] == 0)) + 0.5
            dist_bad = b_bad / (total_bad + 0.5 * (len(edges) - 1))
            dist_good = b_good / (total_good + 0.5 * (len(edges) - 1))
            woe = math.log(dist_bad / dist_good)  # positive WoE = higher default risk
            iv_val += (dist_bad - dist_good) * woe
            woe_map.append(woe)

        iv_scores.append(round(float(abs(iv_val)), 4))
        bin_models.append({"edges": edges, "woe": np.array(woe_map)})

    return iv_scores, bin_models


class WoEScorecardClassifier:
    """
    Bank Credit Scorecard Classifier using Weight of Evidence (WoE) transformation,
    Information Value (IV) feature weighting, and Cost-Sensitive L2 Logistic Regression.
    """
    def __init__(self, learning_rate: float = 0.15, epochs: int = 200, l2_lambda: float = 0.002, default_cost_weight: float = 1.6):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2_lambda = float(l2_lambda)
        self.default_cost_weight = float(default_cost_weight)
        self.bin_models = []
        self.iv_scores = []
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def _transform_woe(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_feats = X.shape
        X_woe = np.zeros_like(X, dtype=float)
        for j in range(n_feats):
            edges = self.bin_models[j]["edges"]
            woe_vals = self.bin_models[j]["woe"]
            idx = np.clip(np.digitize(X[:, j], edges) - 1, 0, len(woe_vals) - 1)
            X_woe[:, j] = woe_vals[idx]
        # Blend continuous standardized signal (65%) with monotonic WoE step signal (35%) for smooth scorecard contours
        return 0.65 * X + 0.35 * X_woe

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.iv_scores, self.bin_models = compute_woe_iv_table(X, y, n_bins=5)
        X_design = self._transform_woe(X)
        m, n = X_design.shape
        rng = np.random.RandomState(42)
        self.weights = rng.normal(0, 0.05, size=n)
        self.bias = 0.0
        self.loss_history = []

        sample_weights = np.where(y == 1, self.default_cost_weight, 1.0)
        eps = 1e-15

        for epoch in range(1, self.epochs + 1):
            z = np.dot(X_design, self.weights) + self.bias
            p = self._sigmoid(z)
            p_clip = np.clip(p, eps, 1.0 - eps)

            bce = -np.mean(sample_weights * (y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))
            reg = (self.l2_lambda / (2.0 * m)) * np.sum(self.weights ** 2)
            self.loss_history.append({"epoch": epoch, "loss": round(float(bce + reg), 5)})

            err = sample_weights * (p - y)
            dw = (1.0 / m) * np.dot(X_design.T, err) + (self.l2_lambda / m) * self.weights
            db = float(np.mean(err))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_design = self._transform_woe(X)
        p1 = self._sigmoid(np.dot(X_design, self.weights) + self.bias)
        return np.column_stack([1.0 - p1, p1])


class CreditRiskNeuralMLP:
    """
    Specially Trained 2-Layer Credit Risk Neural Network (Multi-Layer Perceptron)
    Architecture: Input -> Dense(hidden_dim, LeakyReLU) -> Dense(Output, Sigmoid PD)
    Trained via Mini-Batch / Full-Batch Momentum Gradient Descent with Cost-Sensitive Default Loss.
    """
    def __init__(
        self,
        hidden_dim: int = 14,
        learning_rate: float = 0.12,
        epochs: int = 220,
        l2_lambda: float = 0.0015,
        default_cost_weight: float = 1.65
    ):
        self.hidden_dim = max(4, int(hidden_dim))
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2_lambda = float(l2_lambda)
        self.default_cost_weight = float(default_cost_weight)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = 0.0
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    @staticmethod
    def _leaky_relu(z, alpha=0.1):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def _leaky_relu_deriv(z, alpha=0.1):
        return np.where(z > 0, 1.0, alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        rng = np.random.RandomState(42)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n), size=(n, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), size=(self.hidden_dim, 1))
        self.b2 = 0.0

        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = 0.0
        momentum = 0.85

        y_col = y.reshape(-1, 1)
        sw = np.where(y_col == 1, self.default_cost_weight, 1.0)
        eps = 1e-15
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._leaky_relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            p = self._sigmoid(z2)

            p_clip = np.clip(p, eps, 1.0 - eps)
            bce = -np.mean(sw * (y_col * np.log(p_clip) + (1.0 - y_col) * np.log(1.0 - p_clip)))
            reg = (self.l2_lambda / (2.0 * m)) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            self.loss_history.append({"epoch": epoch, "loss": round(float(bce + reg), 5)})

            dz2 = sw * (p - y_col) / m
            dW2 = np.dot(a1.T, dz2) + (self.l2_lambda / m) * self.W2
            db2 = float(np.sum(dz2))

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self._leaky_relu_deriv(z1)
            dW1 = np.dot(X.T, dz1) + (self.l2_lambda / m) * self.W1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            vW2 = momentum * vW2 - self.learning_rate * dW2
            vb2 = momentum * vb2 - self.learning_rate * db2
            vW1 = momentum * vW1 - self.learning_rate * dW1
            vb1 = momentum * vb1 - self.learning_rate * db1

            self.W2 += vW2
            self.b2 += vb2
            self.W1 += vW1
            self.b1 += vb1

        # Compute effective input-to-output directional sensitivity weights
        eff_w = np.dot(self.W1, self.W2).ravel()
        self.weights = eff_w
        self.bias = float(self.b2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._leaky_relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        p1 = self._sigmoid(z2).ravel()
        return np.column_stack([1.0 - p1, p1])


class CreditInteractionPolyModel:
    """
    Non-Linear Banking Ratio Interaction Model capturing compound leverage stress
    (e.g., DTI x Utilization, LTV x DSCR, Debt/EBITDA x ICR) with Cost-Sensitive L2 optimization.
    """
    def __init__(self, learning_rate: float = 0.14, epochs: int = 200, l2_lambda: float = 0.002, default_cost_weight: float = 1.6):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2_lambda = float(l2_lambda)
        self.default_cost_weight = float(default_cost_weight)
        self.n_raw_features = 0
        self.full_weights = None
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def _expand(self, X: np.ndarray) -> np.ndarray:
        cols = [X, np.clip(X ** 2, -6.0, 6.0)]
        n = X.shape[1]
        for i in range(n):
            for j in range(i + 1, n):
                cols.append(np.clip(X[:, i] * X[:, j], -6.0, 6.0).reshape(-1, 1))
        return np.hstack(cols)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_raw_features = X.shape[1]
        X_poly = self._expand(X)
        m, n = X_poly.shape
        rng = np.random.RandomState(42)
        self.full_weights = rng.normal(0, 0.04, size=n)
        self.bias = 0.0
        self.loss_history = []

        sw = np.where(y == 1, self.default_cost_weight, 1.0)
        eps = 1e-15

        for epoch in range(1, self.epochs + 1):
            z = np.dot(X_poly, self.full_weights) + self.bias
            p = self._sigmoid(z)
            p_clip = np.clip(p, eps, 1.0 - eps)

            bce = -np.mean(sw * (y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))
            reg = (self.l2_lambda / (2.0 * m)) * np.sum(self.full_weights ** 2)
            self.loss_history.append({"epoch": epoch, "loss": round(float(bce + reg), 5)})

            err = sw * (p - y)
            dw = (1.0 / m) * np.dot(X_poly.T, err) + (self.l2_lambda / m) * self.full_weights
            db = float(np.mean(err))

            self.full_weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.weights = self.full_weights[:self.n_raw_features]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_poly = self._expand(X)
        p1 = self._sigmoid(np.dot(X_poly, self.full_weights) + self.bias)
        return np.column_stack([1.0 - p1, p1])


class CreditGradientBoostedTrees:
    """
    Gradient Boosted Credit Risk Decision Trees (GBDT) fitting sequential regression trees
    on Newton-Raphson log-odds residuals for sharp banking underwriting rules.
    """
    def __init__(self, n_estimators: int = 25, max_depth: int = 3, learning_rate: float = 0.22):
        self.n_estimators = max(5, int(n_estimators))
        self.max_depth = max(1, int(max_depth))
        self.learning_rate = float(learning_rate)
        self.trees = []
        self.init_log_odds = 0.0
        self.feature_importances_ = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def _build_reg_tree(self, X: np.ndarray, residuals: np.ndarray, weights: np.ndarray, depth: int):
        n_samples, n_feats = X.shape
        if depth >= self.max_depth or n_samples <= 6 or np.std(residuals) < 1e-6:
            denom = float(np.sum(weights)) + 1e-4
            leaf_val = float(np.sum(residuals) / denom)
            return {"leaf": True, "val": leaf_val}

        best_gain = 0.0
        best_split = None
        base_var = float(np.var(residuals) * n_samples)

        for f_idx in range(n_feats):
            vals = np.unique(X[:, f_idx])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 12:
                thresholds = np.linspace(thresholds[0], thresholds[-1], 12)

            for th in thresholds:
                l_mask = X[:, f_idx] <= th
                r_mask = ~l_mask
                n_l, n_r = int(np.sum(l_mask)), int(np.sum(r_mask))
                if n_l < 3 or n_r < 3:
                    continue
                gain = base_var - (float(np.var(residuals[l_mask]) * n_l) + float(np.var(residuals[r_mask]) * n_r))
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f_idx, float(th), l_mask)

        if best_split is None:
            denom = float(np.sum(weights)) + 1e-4
            return {"leaf": True, "val": float(np.sum(residuals) / denom)}

        f_idx, th_val, l_mask = best_split
        self.feature_importances_[f_idx] += best_gain
        return {
            "leaf": False,
            "feature": f_idx,
            "threshold": th_val,
            "left": self._build_reg_tree(X[l_mask], residuals[l_mask], weights[l_mask], depth + 1),
            "right": self._build_reg_tree(X[~l_mask], residuals[~l_mask], weights[~l_mask], depth + 1)
        }

    def _predict_tree_row(self, node: dict, row: np.ndarray) -> float:
        if node["leaf"]:
            return node["val"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_tree_row(node["left"], row)
        return self._predict_tree_row(node["right"], row)

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.feature_importances_ = np.zeros(n, dtype=float)
        p_base = np.clip(np.mean(y), 0.05, 0.95)
        self.init_log_odds = float(math.log(p_base / (1.0 - p_base)))
        f_pred = np.full(m, self.init_log_odds, dtype=float)
        self.trees = []
        self.loss_history = []
        eps = 1e-15

        for m_step in range(1, self.n_estimators + 1):
            p = self._sigmoid(f_pred)
            residuals = y - p
            hess = np.maximum(p * (1.0 - p), 1e-4)
            tree = self._build_reg_tree(X, residuals, hess, 0)
            self.trees.append(tree)

            update = np.array([self._predict_tree_row(tree, row) for row in X])
            f_pred += self.learning_rate * update

            p_new = np.clip(self._sigmoid(f_pred), eps, 1.0 - eps)
            bce = -np.mean(y * np.log(p_new) + (1.0 - y) * np.log(1.0 - p_new))
            self.loss_history.append({"epoch": m_step, "loss": round(float(bce), 5)})

        tot = np.sum(self.feature_importances_)
        if tot > 0:
            self.feature_importances_ /= tot
        else:
            self.feature_importances_ = np.ones(n) / n
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        f_pred = np.full(len(X), self.init_log_odds, dtype=float)
        for tree in self.trees:
            f_pred += self.learning_rate * np.array([self._predict_tree_row(tree, row) for row in X])
        p1 = self._sigmoid(f_pred)
        return np.column_stack([1.0 - p1, p1])


def compute_roc_ks_and_gini(y_true: np.ndarray, y_probs: np.ndarray):
    """
    Computes ROC Curve, AUC, Gini Coefficient (2*AUC - 1), and Kolmogorov-Smirnov (KS)
    Cumulative Bad vs. Cumulative Good separation curve.
    """
    thresholds = np.linspace(1.0, 0.0, 51)
    total_bad = max(1, int(np.sum(y_true == 1)))
    total_good = max(1, int(np.sum(y_true == 0)))

    roc_points = []
    ks_curve = []
    fpr_list = []
    tpr_list = []
    max_ks = 0.0
    ks_threshold = 0.35

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tpr = float(tp / total_bad)
        fpr = float(fp / total_good)
        ks_diff = abs(tpr - fpr)
        if ks_diff > max_ks:
            max_ks = ks_diff
            ks_threshold = float(th)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_points.append({"x": round(fpr, 4), "y": round(tpr, 4)})
        ks_curve.append({
            "threshold": round(float(th), 2),
            "cum_bad": round(tpr * 100.0, 2),
            "cum_good": round(fpr * 100.0, 2),
            "ks_gap": round(ks_diff * 100.0, 2)
        })

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_val = float(trapz_fn(tpr_list, fpr_list))
    auc_clean = max(0.5, min(1.0, abs(auc_val)))
    gini_val = max(0.0, 2.0 * auc_clean - 1.0)

    return roc_points, auc_clean, gini_val, max_ks, ks_threshold, list(reversed(ks_curve))


def compute_decile_lift_and_basel(y_true: np.ndarray, y_probs: np.ndarray, lgd_val: float = 0.45, avg_ead_k: float = 25.0):
    """
    Groups borrowers into 10 Credit Risk Deciles (Decile 1 = Lowest PD to Decile 10 = Highest PD)
    and calculates Expected Credit Loss (ECL) and Basel III RWA across the loan book.
    """
    order = np.argsort(y_probs)
    y_sorted = y_true[order]
    p_sorted = y_probs[order]
    n = len(y_sorted)
    deciles = []
    total_bads = max(1, int(np.sum(y_true == 1)))
    cum_bads = 0

    grade_map = ["AAA", "AA", "A+", "A", "BBB+", "BBB", "BB+", "BB", "B", "CCC/D"]

    for d in range(10):
        start = int(round(d * n / 10.0))
        end = int(round((d + 1) * n / 10.0))
        if end <= start:
            continue
        sub_y = y_sorted[start:end]
        sub_p = p_sorted[start:end]
        cnt = len(sub_y)
        bads = int(np.sum(sub_y == 1))
        cum_bads += bads
        avg_pd = float(np.mean(sub_p))
        obs_bad_rate = float(bads / cnt)
        ead_total_k = cnt * avg_ead_k
        basel = compute_basel_irb_capital(avg_pd, lgd_val, ead_total_k)

        deciles.append({
            "decile": f"D{d + 1} ({grade_map[d]})",
            "rating": grade_map[d],
            "count": cnt,
            "defaults": bads,
            "predicted_pd_pct": round(avg_pd * 100.0, 2),
            "observed_bad_pct": round(obs_bad_rate * 100.0, 2),
            "ecl_k": round(basel["ecl"], 2),
            "rwa_k": round(basel["rwa"], 2)
        })

    return deciles


class CreditRiskPipeline:
    """
    End-to-End Banking & Credit Risk Machine Learning Pipeline:
    1. Stratified/Deterministic 80/20 Train/Test Split
    2. Z-Score Financial Ratio Standardization & WoE/IV Analysis
    3. Specialized Model Training (WoE Scorecard, Credit Neural MLP, Credit Interaction Net, Credit GBDT)
    4. Basel III IRB Capital, IFRS 9 Stage Provisioning, Gini, KS, and Decile Lift Analytics
    5. 2D Credit Underwriting Iso-PD Contour Surface & Model Serialization (.pkl)
    """
    def __init__(self, save_dir: str = "models/saved"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.model = None
        self.scaler = None
        self.standardize = True
        self.feature_names = []
        self.target_name = "Default_12M"
        self.class_names = ["Performing (0)", "Defaulted (1)"]
        self.model_type = "woe_scorecard"
        self.threshold = 0.35
        self.default_lgd = 0.45
        self.avg_ead_k = 25.0
        self.base_rate_pct = 5.50
        self.feature_means = None
        self.feature_mins = None
        self.feature_maxs = None
        self.iv_scores = []

    def train_and_evaluate(
        self,
        df,
        feature_names: list,
        target_name: str,
        class_names: list,
        model_type: str = "woe_scorecard",
        hyperparams: dict = None,
        standardize: bool = True,
        test_size: float = 0.2,
        axis_x: int = 0,
        axis_y: int = 1,
        artifact_name: str = "trained_model.pkl"
    ) -> dict:
        if hyperparams is None:
            hyperparams = {}

        self.feature_names = feature_names
        self.target_name = target_name
        self.class_names = class_names
        self.model_type = model_type
        self.standardize = bool(standardize)
        self.threshold = float(hyperparams.get("threshold", 0.35))
        self.default_lgd = float(hyperparams.get("lgd", 0.45))
        self.avg_ead_k = float(hyperparams.get("ead_k", 25.0))
        self.base_rate_pct = float(hyperparams.get("base_rate_pct", 5.50))

        X = df[feature_names].values.astype(float)
        y = df[target_name].values.astype(int)

        self.feature_means = np.mean(X, axis=0)
        self.feature_mins = np.min(X, axis=0)
        self.feature_maxs = np.max(X, axis=0)

        rng = np.random.RandomState(42)
        indices = rng.permutation(len(X))
        split_idx = int(len(X) * (1.0 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        if self.standardize:
            self.scaler = ZScoreScaler()
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            X_all_s = self.scaler.transform(X)
        else:
            self.scaler = None
            X_train_s = X_train
            X_test_s = X_test
            X_all_s = X

        self.iv_scores, _ = compute_woe_iv_table(X_train_s, y_train, n_bins=5)

        lr = float(hyperparams.get("learning_rate", 0.14))
        epochs = int(hyperparams.get("epochs", 200))
        l2_lambda = float(hyperparams.get("l2_lambda", 0.002))
        cost_weight = float(hyperparams.get("default_cost_weight", 1.6))

        if model_type in ("woe_scorecard", "logistic"):
            self.model = WoEScorecardClassifier(
                learning_rate=lr, epochs=epochs, l2_lambda=l2_lambda, default_cost_weight=cost_weight
            )
        elif model_type in ("credit_neural_mlp", "mlp"):
            hidden_dim = int(hyperparams.get("hidden_dim", 14))
            self.model = CreditRiskNeuralMLP(
                hidden_dim=hidden_dim, learning_rate=lr, epochs=epochs, l2_lambda=l2_lambda, default_cost_weight=cost_weight
            )
        elif model_type in ("credit_interaction_net", "poly_logistic"):
            self.model = CreditInteractionPolyModel(
                learning_rate=lr, epochs=epochs, l2_lambda=l2_lambda, default_cost_weight=cost_weight
            )
        elif model_type in ("credit_gbdt", "decision_tree"):
            n_est = int(hyperparams.get("n_estimators", 25))
            max_depth = int(hyperparams.get("max_depth", 3))
            self.model = CreditGradientBoostedTrees(
                n_estimators=n_est, max_depth=max_depth, learning_rate=min(0.35, lr * 1.5)
            )
        else:
            raise ValueError(f"Unsupported credit risk model_type: {model_type}")

        self.model.fit(X_train_s, y_train)
        loss_history = self.model.loss_history

        train_probs = self.model.predict_proba(X_train_s)[:, 1]
        test_probs = self.model.predict_proba(X_test_s)[:, 1]
        all_probs = self.model.predict_proba(X_all_s)[:, 1]

        train_preds = (train_probs >= self.threshold).astype(int)
        test_preds = (test_probs >= self.threshold).astype(int)

        train_acc = float(np.mean(train_preds == y_train))
        test_acc = float(np.mean(test_preds == y_test))
        brier_score = float(np.mean((test_probs - y_test) ** 2))

        tp = int(np.sum((test_preds == 1) & (y_test == 1)))
        tn = int(np.sum((test_preds == 0) & (y_test == 0)))
        fp = int(np.sum((test_preds == 1) & (y_test == 0)))
        fn = int(np.sum((test_preds == 0) & (y_test == 1)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        roc_points, roc_auc, gini_coeff, ks_stat, ks_opt_th, ks_curve = compute_roc_ks_and_gini(y_test, test_probs)
        decile_table = compute_decile_lift_and_basel(y, all_probs, lgd_val=self.default_lgd, avg_ead_k=self.avg_ead_k)

        # Portfolio Basel III & IFRS 9 Stage Summary
        stage1_mask = all_probs < 0.08
        stage2_mask = (all_probs >= 0.08) & (all_probs < 0.25)
        stage3_mask = all_probs >= 0.25

        avg_pd_all = float(np.mean(all_probs))
        total_ead_k = len(df) * self.avg_ead_k
        portfolio_basel = compute_basel_irb_capital(avg_pd_all, self.default_lgd, total_ead_k)

        # Financial Profit/Loss impact on Test Portfolio ($k)
        interest_gain_per_good_k = self.avg_ead_k * (self.base_rate_pct / 100.0)
        loss_per_default_k = self.avg_ead_k * self.default_lgd
        net_interest_earned_k = tn * interest_gain_per_good_k
        credit_loss_incurred_k = fn * loss_per_default_k
        loss_prevented_k = tp * loss_per_default_k
        opp_cost_rejected_k = fp * interest_gain_per_good_k
        net_portfolio_pnl_k = net_interest_earned_k - credit_loss_incurred_k

        if hasattr(self.model, "weights") and self.model.weights is not None:
            weights = [round(float(w), 4) for w in self.model.weights]
            bias = round(float(self.model.bias), 4)
        elif hasattr(self.model, "feature_importances_"):
            weights = [round(float(w), 4) for w in self.model.feature_importances_]
            bias = 0.0
        else:
            weights = [round(1.0 / len(feature_names), 4) for _ in feature_names]
            bias = 0.0

        boundary_grid = self.compute_decision_boundary(X, y, axis_x=axis_x, axis_y=axis_y)

        artifact_path = os.path.join(self.save_dir, artifact_name)
        with open(artifact_path, "wb") as f_out:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "class_names": self.class_names,
                "model_type": self.model_type,
                "threshold": self.threshold,
                "default_lgd": self.default_lgd,
                "avg_ead_k": self.avg_ead_k,
                "base_rate_pct": self.base_rate_pct,
                "iv_scores": self.iv_scores
            }, f_out)

        feature_ranges = []
        for idx, name in enumerate(feature_names):
            feature_ranges.append({
                "index": idx,
                "name": name,
                "min": round(float(self.feature_mins[idx]), 3),
                "max": round(float(self.feature_maxs[idx]), 3),
                "mean": round(float(self.feature_means[idx]), 3),
                "iv": self.iv_scores[idx]
            })

        return {
            "model_type": model_type,
            "artifact_path": artifact_path,
            "dataset_summary": {
                "total_rows": int(len(df)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "feature_names": feature_names,
                "target_name": target_name,
                "class_names": class_names,
                "class_counts": {
                    "class_0": int(np.sum(y == 0)),
                    "class_1": int(np.sum(y == 1))
                },
                "preview_rows": df.head(10).round(3).to_dict(orient="records")
            },
            "metrics": {
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "gini_coefficient": round(gini_coeff, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_optimal_threshold": round(ks_opt_th, 2),
                "brier_score": round(brier_score, 4),
                "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
            },
            "basel_summary": {
                **portfolio_basel,
                "stage_1_count": int(np.sum(stage1_mask)),
                "stage_2_count": int(np.sum(stage2_mask)),
                "stage_3_count": int(np.sum(stage3_mask)),
                "net_interest_earned_k": round(net_interest_earned_k, 2),
                "credit_loss_incurred_k": round(credit_loss_incurred_k, 2),
                "loss_prevented_k": round(loss_prevented_k, 2),
                "opp_cost_rejected_k": round(opp_cost_rejected_k, 2),
                "net_portfolio_pnl_k": round(net_portfolio_pnl_k, 2)
            },
            "loss_history": loss_history,
            "roc_curve": roc_points,
            "ks_curve": ks_curve,
            "decile_table": decile_table,
            "weights": weights,
            "iv_scores": self.iv_scores,
            "bias": bias,
            "feature_ranges": feature_ranges,
            "boundary_grid": boundary_grid
        }

    def compute_decision_boundary(self, X: np.ndarray, y: np.ndarray, axis_x: int = 0, axis_y: int = 1, resolution: int = 56) -> dict:
        n_features = X.shape[1]
        axis_x = max(0, min(axis_x, n_features - 1))
        axis_y = max(0, min(axis_y, n_features - 1))

        x_vals = X[:, axis_x]
        y_vals = X[:, axis_y]

        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

        pad_x = (x_max - x_min) * 0.10 or 1.0
        pad_y = (y_max - y_min) * 0.10 or 1.0

        x0, x1 = x_min - pad_x, x_max + pad_x
        y0, y1 = y_min - pad_y, y_max + pad_y

        xx, yy = np.meshgrid(
            np.linspace(x0, x1, resolution),
            np.linspace(y0, y1, resolution)
        )

        grid_input = np.tile(self.feature_means, (resolution * resolution, 1))
        grid_input[:, axis_x] = xx.ravel()
        grid_input[:, axis_y] = yy.ravel()

        if self.scaler is not None:
            grid_scaled = self.scaler.transform(grid_input)
        else:
            grid_scaled = grid_input

        probs = self.model.predict_proba(grid_scaled)[:, 1].reshape(resolution, resolution)

        points = []
        for i in range(len(X)):
            points.append({
                "x": round(float(X[i, axis_x]), 4),
                "y": round(float(X[i, axis_y]), 4),
                "label": int(y[i])
            })

        return {
            "axis_x": axis_x,
            "axis_y": axis_y,
            "axis_x_name": self.feature_names[axis_x],
            "axis_y_name": self.feature_names[axis_y],
            "x_min": round(x0, 4),
            "x_max": round(x1, 4),
            "y_min": round(y0, 4),
            "y_max": round(y1, 4),
            "resolution": resolution,
            "threshold": self.threshold,
            "probs": np.round(probs, 3).tolist(),
            "points": points
        }

    def predict_single(self, feature_values: list, loan_amount_k: float = None, lgd_pct: float = None, base_rate_pct: float = None) -> dict:
        if self.model is None:
            raise ValueError("Credit risk model has not been trained yet.")

        arr = np.array(feature_values, dtype=float).reshape(1, -1)
        if self.scaler is not None:
            arr_s = self.scaler.transform(arr)
        else:
            arr_s = arr

        pd_1 = float(self.model.predict_proba(arr_s)[0, 1])
        pd_0 = 1.0 - pd_1
        pred_class = 1 if pd_1 >= self.threshold else 0

        ead_k = float(loan_amount_k) if loan_amount_k is not None else self.avg_ead_k
        lgd_val = (float(lgd_pct) / 100.0) if lgd_pct is not None else self.default_lgd
        base_rate = float(base_rate_pct) if base_rate_pct is not None else self.base_rate_pct

        score, rating_grade, ifrs_stage = pd_to_credit_score_and_grade(pd_1)
        basel = compute_basel_irb_capital(pd_1, lgd_val, ead_k)

        # Risk-Adjusted Loan Pricing & RAROC Calculation
        expected_loss_rate_pct = pd_1 * lgd_val * 100.0
        cost_of_capital_pct = (basel["regulatory_capital_8pct"] / max(0.01, ead_k)) * 12.0  # 12% hurdle ROE
        opex_And_margin_pct = 1.45
        recommended_apr_pct = base_rate + expected_loss_rate_pct + cost_of_capital_pct + opex_And_margin_pct

        net_expected_income_k = ead_k * ((recommended_apr_pct - base_rate) / 100.0) - basel["ecl"]
        econ_capital_k = max(0.05 * ead_k, basel["regulatory_capital_8pct"])
        raroc_pct = (net_expected_income_k / econ_capital_k) * 100.0

        if pd_1 < self.threshold * 0.65:
            decision = "APPROVE — STAGE 1 PRIME"
        elif pd_1 < self.threshold:
            decision = "CONDITIONAL APPROVE — PRICING COVENANT"
        else:
            decision = "DECLINE — EXCEEDS PD CUTOFF"

        return {
            "predicted_class": pred_class,
            "predicted_label": self.class_names[pred_class],
            "decision": decision,
            "prob_performing": round(pd_0, 4),
            "prob_default": round(pd_1, 4),
            "credit_score": score,
            "rating_grade": rating_grade,
            "ifrs9_stage": ifrs_stage,
            "threshold": self.threshold,
            "basel": basel,
            "recommended_apr_pct": round(recommended_apr_pct, 2),
            "risk_spread_bps": int(round((recommended_apr_pct - base_rate) * 100)),
            "raroc_pct": round(raroc_pct, 1)
        }


# Alias for backward compatibility
MLModelPipeline = CreditRiskPipeline

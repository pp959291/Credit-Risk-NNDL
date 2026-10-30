# CreditRisk-Neural-Bank: Specialized Credit Risk & Basel III Banking ML Suite

An institutional-grade, modular Credit Risk & Banking Machine Learning project modeled on transparent, dependency-free vectorized NumPy + Pandas and browser-ready interactive visualizations. Specifically engineered for **Probability of Default (PD) Modeling**, **Weight of Evidence (WoE) & Information Value (IV) Credit Scorecards**, **Cost-Sensitive Credit Neural Networks (MLP)**, **Gradient Boosted Credit Trees (GBDT)**, and **Basel III IRB / IFRS 9 Expected Credit Loss (ECL) & Capital Requirement Calculations**.

## Project Structure

```text
CreditRisk-Neural-Bank/
├── index.html                        # Main Credit Risk & Basel III Dashboard UI
├── app.py                            # Python HTTP REST Server for Credit Modeling & Underwriting
├── train_cli.py                      # Standalone CLI Trainer & Basel III Portfolio Report Generator
├── requirements.txt                  # Python dependencies (numpy, pandas)
├── data/
│   ├── generator.py                  # Banking Loan Portfolio Generator & CSV Preprocessor
│   └── sample_datasets/              # Pre-built Banking Credit CSV Datasets
│       ├── retail_unsecured_loans.csv
│       ├── mortgage_hl_portfolio.csv
│       ├── sme_commercial_credit.csv
│       └── credit_card_revolving.csv
├── models/
│   ├── trainer.py                    # Specialized Credit Risk Models (WoE Scorecard, Neural MLP, Interaction Net, GBDT) & Basel III Engine
│   └── saved/                        # Pre-trained .pkl Credit Models & Basel III Evaluation JSONs
│       ├── trained_model.pkl
│       ├── retail_woe_scorecard.pkl
│       ├── mortgage_neural_mlp.pkl
│       ├── sme_interaction_net.pkl
│       ├── cards_credit_gbdt.pkl
│       └── latest_metrics.json
└── static/
    ├── css/styles.css                # Custom stylesheet
    └── js/
        ├── data_generator.js         # Client-side Banking Loan Book Generator & CSV Parser
        ├── model_trainer.js          # Client-side Credit Risk Models & Basel III / IFRS 9 Calculator
        ├── visualizer.js             # 2D Iso-PD Surface, KS Curve, Decile Lift, ROC/Gini, IV & Loss Plots
        └── dashboard.js              # Interactive Underwriting & Portfolio Stress Controller
```

## Specialized Credit Risk Models Included

1. **WoE & IV Bank Credit Scorecard (`woe_scorecard`)**:
   - Computes quantile Weight of Evidence ($\text{WoE}_i = \ln(\%\text{Bad}_i / \%\text{Good}_i)$) and Information Value ($\text{IV} = \sum (\%\text{Bad}_i - \%\text{Good}_i)\cdot \text{WoE}_i$) for each financial ratio, fits a Cost-Sensitive L2 Logistic Scorecard, and scales log-odds into a **300–850 Bank Credit Score** using **PDO = 50 (Points to Double the Odds)** and Agency Rating Grades (`AAA` to `CCC/D`).
2. **Cost-Sensitive Credit Risk Neural Network (`credit_neural_mlp`)**:
   - 2-Layer Feedforward Neural Network (`Input -> Dense(LeakyReLU) -> Dense(Sigmoid PD)`) trained via Momentum Gradient Descent with asymmetric default loss weighting ($\alpha_{\text{default}} = 1.65$) to penalize missed defaults (`False Negatives`).
3. **Leverage & Stress Non-Linear Interaction Network (`credit_interaction_net`)**:
   - Captures compound credit distress interactions (such as $\text{DTI} \times \text{Credit Utilization}$, $\text{LTV} \times \text{DSCR}^{-1}$, and $\text{Net Debt/EBITDA} \times \text{ICR}^{-1}$).
4. **Gradient Boosted Credit Decision Trees (`credit_gbdt`)**:
   - Fits sequential regression trees on Newton-Raphson log-odds residuals to isolate sharp underwriting cutoff rules.

## Basel III IRB & IFRS 9 Calculations

- **Expected Credit Loss (ECL)**: $\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}$
- **IFRS 9 Stage Classification**: Stage 1 Performing (12M ECL), Stage 2 SICR Watch (Lifetime ECL), Stage 3 Credit-Impaired
- **Basel III IRB Risk-Weighted Assets (RWA)**: Computes Asset Correlation $R(\text{PD})$, 99.9% Confidence Capital Factor $K(\text{PD}, \text{LGD})$, $\text{RWA} = 12.5 \times K \times \text{EAD}$, and 8% Tier 1 Regulatory Capital
- **Risk-Adjusted Pricing & RAROC**: Computes recommended loan APR (Base Rate + Expected Loss Spread + Capital Cost + OpEx) and Risk-Adjusted Return on Capital (RAROC %).

## How to Run

1. **Instant Browser Mode**:
   Open `index.html` directly in any browser or via VS Code Live Server.
2. **Python Backend Server Mode**:
   ```bash
   pip install -r requirements.txt
   python3 app.py
   ```
   Then open `http://127.0.0.1:5000`.
3. **Standalone CLI Training & Basel III Portfolio Report**:
   ```bash
   python3 train_cli.py --preset retail_unsecured --samples 320 --model woe_scorecard
   python3 train_cli.py --preset mortgage_hl --samples 320 --model credit_neural_mlp
   python3 train_cli.py --csv data/sample_datasets/sme_commercial_credit.csv --model credit_gbdt
   ```
 Author - Piyush (25MDA10036)

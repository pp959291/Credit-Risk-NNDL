#!/usr/bin/env python3
"""
train_cli.py
Standalone CLI for Training Specialized Credit Risk & Basel III Banking Models.

Usage Examples:
  1. Train WoE Credit Scorecard on Retail Unsecured Loans:
     python3 train_cli.py --preset retail_unsecured --samples 320 --model woe_scorecard

  2. Train Credit Risk Neural MLP on Residential Mortgage Book:
     python3 train_cli.py --preset mortgage_hl --samples 350 --model credit_neural_mlp --epochs 240

  3. Train Gradient Boosted Credit Trees on Custom CSV Portfolio:
     python3 train_cli.py --csv data/sample_datasets/sme_commercial_credit.csv --model credit_gbdt
"""
import argparse
import json
import os
from data.generator import generate_credit_portfolio, load_and_preprocess_csv, PORTFOLIO_METADATA
from models.trainer import CreditRiskPipeline


def main():
    parser = argparse.ArgumentParser(description="Credit Risk & Basel III Banking Model Trainer CLI")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom credit portfolio CSV file")
    parser.add_argument("--target", type=str, default=None, help="Target column name (default: Default_12M)")
    parser.add_argument(
        "--preset",
        type=str,
        default="retail_unsecured",
        choices=["retail_unsecured", "mortgage_hl", "sme_commercial", "credit_cards"],
        help="Banking loan portfolio preset"
    )
    parser.add_argument("--samples", type=int, default=320, help="Number of loan facilities to generate")
    parser.add_argument("--stress", type=float, default=0.16, help="Macroeconomic stress / volatility level")
    parser.add_argument(
        "--model",
        type=str,
        default="woe_scorecard",
        choices=["woe_scorecard", "credit_neural_mlp", "credit_interaction_net", "credit_gbdt"],
        help="Specialized credit risk model architecture"
    )
    parser.add_argument("--lr", type=float, default=0.14, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--l2", type=float, default=0.002, help="L2 regularization strength")
    parser.add_argument("--hidden", type=int, default=14, help="Hidden neurons for Credit Neural MLP")
    parser.add_argument("--trees", type=int, default=25, help="Boosting trees for Credit GBDT")
    parser.add_argument("--depth", type=int, default=3, help="Max tree depth for Credit GBDT")
    parser.add_argument("--threshold", type=float, default=0.35, help="Underwriting PD cutoff threshold")
    parser.add_argument("--artifact-name", type=str, default="trained_model.pkl", help="Saved .pkl filename")
    parser.add_argument("--output-json", type=str, default="models/saved/latest_metrics.json", help="Metrics JSON output path")

    args = parser.parse_args()

    print("=" * 76)
    print("  CREDIT RISK & BASEL III BANKING NEURAL ANALYTICS TRAINING PIPELINE")
    print("=" * 76)

    meta = PORTFOLIO_METADATA.get(args.preset, PORTFOLIO_METADATA["retail_unsecured"])

    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        with open(args.csv, "r", encoding="utf-8") as f:
            csv_text = f.read()
        df, feature_names, target_name, class_names = load_and_preprocess_csv(csv_text, target_col=args.target)
        print(f"[PORTFOLIO] Loaded CSV '{args.csv}' -> {len(df)} facilities, Target: '{target_name}'")
    else:
        df, feature_names, target_name, class_names = generate_credit_portfolio(
            preset=args.preset, n_samples=args.samples, stress_noise=args.stress
        )
        print(f"[PORTFOLIO] Generated '{meta['title']}' ({args.preset}) -> {len(df)} facilities")

    pipeline = CreditRiskPipeline(save_dir="models/saved")
    hyperparams = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "l2_lambda": args.l2,
        "hidden_dim": args.hidden,
        "n_estimators": args.trees,
        "max_depth": args.depth,
        "threshold": args.threshold,
        "lgd": meta["default_lgd"],
        "ead_k": meta["avg_ead_k"],
        "base_rate_pct": meta["base_rate_pct"]
    }

    results = pipeline.train_and_evaluate(
        df=df,
        feature_names=feature_names,
        target_name=target_name,
        class_names=class_names,
        model_type=args.model,
        hyperparams=hyperparams,
        standardize=True,
        artifact_name=args.artifact_name
    )

    m = results["metrics"]
    b = results["basel_summary"]
    cm = m["confusion_matrix"]

    print("-" * 76)
    print(f"[MODEL] Architecture          : {args.model.upper()}")
    print(f"[METRIC] ROC AUC / Gini       : AUC = {m['roc_auc']:.4f} | Gini = {m['gini_coefficient']:.4f}")
    print(f"[METRIC] KS Statistic         : {m['ks_statistic'] * 100:.2f}% (Optimal PD Cutoff: {m['ks_optimal_threshold'] * 100:.0f}%)")
    print(f"[METRIC] Train / Test Acc     : {m['train_accuracy'] * 100:.2f}% / {m['test_accuracy'] * 100:.2f}%")
    print(f"[METRIC] Default Recall / F1  : Recall = {m['recall']:.4f} | F1 = {m['f1_score']:.4f} | Brier = {m['brier_score']:.4f}")
    print(f"[MATRIX] Underwriting Matrix  : Performing Approved (TN)={cm['tn']}, False Alarm (FP)={cm['fp']}, Missed Default (FN)={cm['fn']}, Default Caught (TP)={cm['tp']}")
    print(f"[BASEL] Portfolio EAD / ECL   : EAD = ${b['ead']:,.1f}k | Expected Credit Loss (ECL) = ${b['ecl']:,.2f}k")
    print(f"[BASEL] RWA & Reg Capital     : RWA = ${b['rwa']:,.1f}k (RW: {b['risk_weight_pct']}%) | Tier 1 Cap (8%) = ${b['regulatory_capital_8pct']:,.2f}k")
    print(f"[IFRS 9] Stage Distribution   : Stage 1 = {b['stage_1_count']} | Stage 2 (SICR) = {b['stage_2_count']} | Stage 3 = {b['stage_3_count']}")
    print(f"[IV] Feature Info Value (IV)  : {dict(zip(feature_names, results['iv_scores']))}")
    print("-" * 76)
    print(f"[SAVED] Trained Credit Model  -> {results['artifact_path']}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f_json:
        summary_out = {k: v for k, v in results.items() if k != "boundary_grid"}
        json.dump(summary_out, f_json, indent=2)
    print(f"[SAVED] Basel & Risk Metrics  -> {args.output_json}")
    print("=" * 76)


if __name__ == "__main__":
    main()

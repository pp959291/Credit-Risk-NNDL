#!/usr/bin/env python3
"""
app.py
Modular Python Banking & Credit Risk Analytics Backend Server connecting
`models/trainer.py` and `data/generator.py` with `index.html` and `static/js/`.
"""
import json
import mimetypes
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from data.generator import generate_credit_portfolio, load_and_preprocess_csv, PORTFOLIO_METADATA
from models.trainer import CreditRiskPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATE = {
    "preset": "retail_unsecured",
    "df": None,
    "feature_names": [],
    "target_name": "Default_12M",
    "class_names": ["Performing (0)", "Defaulted (1)"],
    "pipeline": CreditRiskPipeline(save_dir=os.path.join(BASE_DIR, "models", "saved")),
    "last_results": None
}


def init_default_portfolio():
    df, feat_names, target_name, class_names = generate_credit_portfolio("retail_unsecured", 320, 0.16)
    STATE["preset"] = "retail_unsecured"
    STATE["df"] = df
    STATE["feature_names"] = feat_names
    STATE["target_name"] = target_name
    STATE["class_names"] = class_names
    meta = PORTFOLIO_METADATA["retail_unsecured"]
    STATE["last_results"] = STATE["pipeline"].train_and_evaluate(
        df, feat_names, target_name, class_names,
        model_type="woe_scorecard",
        hyperparams={
            "threshold": 0.35,
            "lgd": meta["default_lgd"],
            "ead_k": meta["avg_ead_k"],
            "base_rate_pct": meta["base_rate_pct"]
        }
    )


class CreditRiskDashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath: str, content_type: str = None):
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        if not content_type:
            content_type, _ = mimetypes.guess_type(filepath)
            content_type = content_type or "application/octet-stream"
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/templates/index.html"):
            if STATE["df"] is None:
                init_default_portfolio()
            self._serve_file(os.path.join(BASE_DIR, "index.html"), "text/html; charset=utf-8")

        elif path.startswith("/static/"):
            rel_path = path.lstrip("/")
            self._serve_file(os.path.join(BASE_DIR, rel_path))

        elif path == "/api/export_csv":
            if STATE["df"] is None:
                init_default_portfolio()
            csv_bytes = STATE["df"].to_csv(index=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=credit_risk_portfolio.csv")
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)

        elif path.startswith("/api/sample_csv/"):
            name = path.split("/api/sample_csv/")[-1]
            mapping = {
                "retail_unsecured": "retail_unsecured_loans.csv",
                "mortgage_hl": "mortgage_hl_portfolio.csv",
                "sme_commercial": "sme_commercial_credit.csv",
                "credit_cards": "credit_card_revolving.csv"
            }
            filename = mapping.get(name)
            if not filename:
                self._send_json({"status": "error", "message": "Unknown banking sample dataset"}, 404)
                return

            csv_path = os.path.join(BASE_DIR, "data", "sample_datasets", filename)
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_text = f.read()

            df, feat_names, target_name, class_names = load_and_preprocess_csv(csv_text)
            STATE["preset"] = name
            STATE["df"] = df
            STATE["feature_names"] = feat_names
            STATE["target_name"] = target_name
            STATE["class_names"] = class_names

            meta = PORTFOLIO_METADATA.get(name, PORTFOLIO_METADATA["retail_unsecured"])
            results = STATE["pipeline"].train_and_evaluate(
                df=df,
                feature_names=feat_names,
                target_name=target_name,
                class_names=class_names,
                model_type="woe_scorecard",
                hyperparams={
                    "threshold": 0.35,
                    "lgd": meta["default_lgd"],
                    "ead_k": meta["avg_ead_k"],
                    "base_rate_pct": meta["base_rate_pct"]
                }
            )
            STATE["last_results"] = results
            self._send_json({"status": "ok", "data": results})

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(raw_body or "{}")

        try:
            if path == "/api/generate":
                preset = payload.get("preset", "retail_unsecured")
                n_samples = int(payload.get("n_samples", 320))
                stress = float(payload.get("noise", 0.16))
                seed = int(payload.get("seed", 42))

                df, feat_names, target_name, class_names = generate_credit_portfolio(
                    preset=preset, n_samples=n_samples, stress_noise=stress, random_state=seed
                )
                STATE["preset"] = preset
                STATE["df"] = df
                STATE["feature_names"] = feat_names
                STATE["target_name"] = target_name
                STATE["class_names"] = class_names

                results = STATE["pipeline"].train_and_evaluate(
                    df=df,
                    feature_names=feat_names,
                    target_name=target_name,
                    class_names=class_names,
                    model_type=payload.get("model_type", "woe_scorecard"),
                    hyperparams=payload.get("hyperparams", {}),
                    standardize=payload.get("standardize", True)
                )
                STATE["last_results"] = results
                self._send_json({"status": "ok", "data": results})

            elif path == "/api/upload_csv":
                csv_text = payload.get("csv_text", "")
                target_col = payload.get("target_col")
                feature_cols = payload.get("feature_cols")
                df, feat_names, target_name, class_names = load_and_preprocess_csv(
                    csv_text, target_col=target_col, feature_cols=feature_cols
                )
                STATE["df"] = df
                STATE["feature_names"] = feat_names
                STATE["target_name"] = target_name
                STATE["class_names"] = class_names

                results = STATE["pipeline"].train_and_evaluate(
                    df=df,
                    feature_names=feat_names,
                    target_name=target_name,
                    class_names=class_names,
                    model_type=payload.get("model_type", "woe_scorecard"),
                    hyperparams=payload.get("hyperparams", {}),
                    standardize=payload.get("standardize", True)
                )
                STATE["last_results"] = results
                self._send_json({"status": "ok", "data": results})

            elif path == "/api/train":
                if STATE["df"] is None:
                    init_default_portfolio()

                model_type = payload.get("model_type", "woe_scorecard")
                hyperparams = payload.get("hyperparams", {})
                standardize = bool(payload.get("standardize", True))
                axis_x = int(payload.get("axis_x", 0))
                axis_y = int(payload.get("axis_y", 1))

                results = STATE["pipeline"].train_and_evaluate(
                    df=STATE["df"],
                    feature_names=STATE["feature_names"],
                    target_name=STATE["target_name"],
                    class_names=STATE["class_names"],
                    model_type=model_type,
                    hyperparams=hyperparams,
                    standardize=standardize,
                    axis_x=axis_x,
                    axis_y=axis_y
                )
                STATE["last_results"] = results
                self._send_json({"status": "ok", "data": results})

            elif path == "/api/predict":
                features = payload.get("features", [])
                loan_amount_k = payload.get("loan_amount_k")
                lgd_pct = payload.get("lgd_pct")
                base_rate_pct = payload.get("base_rate_pct")
                prediction = STATE["pipeline"].predict_single(
                    features, loan_amount_k=loan_amount_k, lgd_pct=lgd_pct, base_rate_pct=base_rate_pct
                )
                self._send_json({"status": "ok", "prediction": prediction})

            else:
                self.send_error(404, "API endpoint not found")
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, 400)


def run_server(host: str = "127.0.0.1", port: int = 5000):
    init_default_portfolio()
    server = HTTPServer((host, port), CreditRiskDashboardHandler)
    print(f"Credit Risk & Banking Neural Suite running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()

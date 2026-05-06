"""Flask API for counterfactual simulation queries."""

from __future__ import annotations

from flask import Flask, jsonify, request

from causal_forecasting.counterfactuals import CounterfactualEngine


def create_counterfactual_app(
    counterfactual_engine: CounterfactualEngine,
    current_state_provider,
) -> Flask:
    """Create a Flask app exposing `/counterfactual`."""

    app = Flask(__name__)

    @app.post("/counterfactual")
    def run_counterfactual():
        body = request.get_json(force=True) or {}
        intervention_var = body["intervention_variable"]
        horizon = int(body.get("forecast_horizon", 10))
        n_bootstrap = int(body.get("n_bootstrap", 500))
        current_state = current_state_provider()

        if "intervention_value" in body:
            result = counterfactual_engine.intervene(
                current_state,
                intervention_var,
                float(body["intervention_value"]),
                forecast_horizon=horizon,
                n_bootstrap=n_bootstrap,
            )
        else:
            result = counterfactual_engine.intervene_relative(
                current_state,
                intervention_var,
                float(body["intervention_magnitude"]),
                forecast_horizon=horizon,
                n_bootstrap=n_bootstrap,
            )

        return jsonify(result.to_dict())

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


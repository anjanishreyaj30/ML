"""
app.py
------
Flask web server. Serves the page and exposes JSON APIs the front-end calls.
No database; each browser session gets its own CSV file inside uploads/.
"""

import os
import uuid
import json     

import pandas as pd
from flask import (
    Flask, render_template, request, jsonify, session, send_file
)

import ml_utils as ml
from sample_data import generate_sample_data

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-string-before-you-deploy"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_session_id() -> str:
    """Give each browser a random id, stored in a signed cookie."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def input_csv_path(sid: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{sid}_input.csv")


def output_csv_path(sid: str) -> str:
    return os.path.join(UPLOAD_DIR, f"{sid}_clustered.csv")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample", methods=["POST"])
def api_sample():
    sid = get_session_id()
    df = generate_sample_data(100)
    df.to_csv(input_csv_path(sid), index=False)
    return jsonify(ml.basic_stats(df))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    sid = get_session_id()

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No file was selected."}), 400

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        return jsonify({"error": f"Could not read that CSV: {exc}"}), 400

    required = [ml.PLAYER_COL] + ml.FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({
            "error": f"CSV is missing these columns: {missing}. "
                     f"Required columns: {required}"
        }), 400

    df.to_csv(input_csv_path(sid), index=False)
    return jsonify(ml.basic_stats(df))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    sid = get_session_id()
    path = input_csv_path(sid)

    if not os.path.exists(path):
        return jsonify({"error": "Please upload a CSV or load the sample dataset first."}), 400

    payload = request.get_json(silent=True) or {}
    try:
        k = int(payload.get("k", 4))
    except (TypeError, ValueError):
        k = 4
    k = max(2, min(10, k))           # clamp to the range we plotted

    raw_df = pd.read_csv(path)

    try:
        result, final_df = ml.run_full_pipeline(raw_df, k=k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    final_df.to_csv(output_csv_path(sid), index=False)
        

    # Send the full clustered table to the browser as JSON
    # (json.loads + to_json handles NaN cleanly, unlike to_dict)
    result["final_data"] = json.loads(final_df.to_json(orient="records"))
    result["k_used"] = k
    result["download_url"] = "/download"

    return jsonify(result)


@app.route("/download")
def download():
    sid = get_session_id()
    path = output_csv_path(sid)

    if not os.path.exists(path):
        return "No results yet - run the analysis first.", 404

    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="players_clustered.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
"""
PediAppend – Flask Web Application
===================================
Pediatric Appendicitis Diagnosis Support with SHAP explainability.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime

import numpy as np
import joblib
import shap
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Flask app ──
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

# ── Load model artefacts ──
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

# ── SHAP explainer ──
# VotingClassifier is not supported by TreeExplainer directly;
# use the best tree-based sub-estimator for explanations.
_model_type = type(model).__name__
if _model_type == "VotingClassifier":
    # Pick a tree-based sub-estimator for SHAP
    for _name in ("lgbm", "cat", "rf"):
        _sub = model.named_estimators_.get(_name)
        if _sub is not None:
            explainer = shap.TreeExplainer(_sub)
            logger.info("SHAP explainer using sub-estimator '%s' (%s)", _name, type(_sub).__name__)
            break
    else:
        explainer = shap.TreeExplainer(model.estimators_[0])
elif _model_type in ("RandomForestClassifier", "LGBMClassifier", "CatBoostClassifier"):
    explainer = shap.TreeExplainer(model)
else:
    explainer = None  # will fall back to KernelExplainer at predict time

# ── Database ──
DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age REAL,
            sex TEXT,
            prediction INTEGER,
            confidence REAL,
            probability REAL,
            form_data TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ═══════════════════════════════════════════════════════════
#  Feature-vector builder
# ═══════════════════════════════════════════════════════════

def build_feature_vector(form_data):
    """Convert raw form data into a dict aligned with model feature_names."""
    vec = {}

    # ── Continuous / numeric ──
    for col in ["Age", "BMI", "Height", "Weight", "Appendix_Diameter",
                "Body_Temperature", "WBC_Count", "Neutrophil_Percentage",
                "RBC_Count", "Hemoglobin", "RDW", "Thrombocyte_Count", "CRP"]:
        val = form_data.get(col, "0")
        vec[col] = float(val) if val else 0.0

    # Features the form doesn't supply – default to 0
    vec.setdefault("Length_of_Stay", 0)
    vec.setdefault("Alvarado_Score", 0)
    vec.setdefault("Paedriatic_Appendicitis_Score", 0)

    # ── Engineered features ──
    vec["WBC_CRP_Ratio"] = vec["WBC_Count"] / (vec["CRP"] + 0.1)
    vec["Neutrophil_WBC_Interaction"] = vec["Neutrophil_Percentage"] * vec["WBC_Count"]

    # ── Binary _yes features ──
    binary_map = {
        "Sex":        ("Sex_male", "male"),
        "Migratory_Pain":   ("Migratory_Pain_yes", "yes"),
        "Lower_Right_Abd_Pain": ("Lower_Right_Abd_Pain_yes", "yes"),
        "Contralateral_Rebound_Tenderness": ("Contralateral_Rebound_Tenderness_yes", "yes"),
        "Coughing_Pain": ("Coughing_Pain_yes", "yes"),
        "Nausea":   ("Nausea_yes", "yes"),
        "Loss_of_Appetite": ("Loss_of_Appetite_yes", "yes"),
        "Neutrophilia": ("Neutrophilia_yes", "yes"),
        "Dysuria":  ("Dysuria_yes", "yes"),
        "Psoas_Sign": ("Psoas_Sign_yes", "yes"),
        "Ipsilateral_Rebound_Tenderness": ("Ipsilateral_Rebound_Tenderness_yes", "yes"),
        "US_Performed": ("US_Performed_yes", "yes"),
        "Appendix_on_US": ("Appendix_on_US_yes", "yes"),
        "Free_Fluids": ("Free_Fluids_yes", "yes"),
    }
    for form_key, (feat, positive) in binary_map.items():
        vec[feat] = 1 if form_data.get(form_key, "").lower() == positive else 0

    # ── Multi-class one-hot ──
    # Ketones_in_Urine: ++, +++, no  (drop_first removes +)
    ketones = form_data.get("Ketones_in_Urine", "no")
    vec["Ketones_in_Urine_++"] = 1 if ketones == "++" else 0
    vec["Ketones_in_Urine_+++"] = 1 if ketones == "+++" else 0
    vec["Ketones_in_Urine_no"] = 1 if ketones == "no" else 0

    # RBC_in_Urine
    rbc_u = form_data.get("RBC_in_Urine", "no")
    vec["RBC_in_Urine_++"] = 1 if rbc_u == "++" else 0
    vec["RBC_in_Urine_+++"] = 1 if rbc_u == "+++" else 0
    vec["RBC_in_Urine_no"] = 1 if rbc_u == "no" else 0

    # WBC_in_Urine
    wbc_u = form_data.get("WBC_in_Urine", "no")
    vec["WBC_in_Urine_++"] = 1 if wbc_u == "++" else 0
    vec["WBC_in_Urine_+++"] = 1 if wbc_u == "+++" else 0
    vec["WBC_in_Urine_no"] = 1 if wbc_u == "no" else 0

    # Stool: constipation/diarrhea, diarrhea, normal  (drop_first removes constipation)
    stool = form_data.get("Stool", "normal").lower()
    vec["Stool_constipation, diarrhea"] = 1 if stool == "constipation" else 0
    vec["Stool_diarrhea"] = 1 if stool == "diarrhea" else 0
    vec["Stool_normal"] = 1 if stool == "normal" else 0

    # Peritonitis: local, no  (drop_first removes generalized)
    perit = form_data.get("Peritonitis", "no").lower()
    vec["Peritonitis_local"] = 1 if perit == "local" else 0
    vec["Peritonitis_no"] = 1 if perit == "no" else 0

    return vec


def prepare_input(feature_vector):
    """Align to model features, fill missing with 0, scale."""
    import pandas as pd
    df = pd.DataFrame([feature_vector])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    X_scaled = scaler.transform(df.values)
    return X_scaled


def compute_shap_values(X_scaled):
    """Return top-10 SHAP features as list of dicts."""
    global explainer
    if explainer is None:
        bg = shap.sample(X_scaled, min(50, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
    sv = explainer.shap_values(X_scaled)
    if isinstance(sv, list):
        sv = sv[1]
    vals = sv[0]
    pairs = list(zip(feature_names, X_scaled[0], vals))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return [{"feature": f, "value": round(float(v), 4), "shap": round(float(s), 4)}
            for f, v, s in pairs[:10]]


# ═══════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        vec = build_feature_vector(form_data)
        X_scaled = prepare_input(vec)

        prediction = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
        probability = float(proba[1])
        confidence = float(max(proba))

        shap_values = compute_shap_values(X_scaled)

        # Save to DB
        import json
        conn = get_db()
        conn.execute(
            "INSERT INTO history (timestamp, age, sex, prediction, confidence, probability, form_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             form_data.get("Age", ""),
             form_data.get("Sex", ""),
             prediction,
             confidence,
             probability,
             json.dumps(form_data))
        )
        conn.commit()
        conn.close()

        return render_template("result.html",
                               prediction=prediction,
                               probability=probability,
                               confidence=confidence,
                               shap_values=shap_values,
                               form_data=form_data)
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        flash(f"Error: {e}", "error")
        return redirect(url_for("diagnosis"))


@app.route("/history")
def history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    conn.close()
    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "age": r["age"],
            "sex": r["sex"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
        })
    return render_template("history.html", records=records)


@app.route("/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    conn = get_db()
    conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


@app.route("/history/clear", methods=["POST"])
def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

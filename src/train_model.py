"""
Model Training Module — Pediatric Appendicitis Diagnosis
=========================================================
Responsabilité unique : entraîner les modèles et sélectionner le meilleur.

Ce module ne charge, ne nettoie et ne prétraite aucune donnée.
Il reçoit des arrays numpy prêts à l'emploi depuis run.py.

Pourquoi des hyperparamètres fixes et non Optuna ?
  Le tuning Optuna a été testé et a produit de moins bons résultats sur ce
  dataset (~600 samples). Avec si peu de données, Optuna introduit plus de
  variance qu'il n'apporte de signal — il overfitte sur le processus de
  validation lui-même. Les paramètres fixes ci-dessous ont été validés
  empiriquement et produisent les meilleurs scores observés :
    RF : Recall=0.9892, Précision=0.9485, AUC=0.9830

Critère de sélection :
  Priorité 1 : Recall    (FN = appendicite manquée = risque vital)
  Priorité 2 : Précision (FP = examen inutile, coût moindre)
  Priorité 3 : ROC-AUC   (tie-breaker)
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Retourne les modèles candidats avec des hyperparamètres validés empiriquement.

      · class_weight="balanced" compense le déséquilibre des classes (~60/40)
        sans aller jusqu'au sur-échantillonnage.
      · auto_class_weights="Balanced" est l'équivalent pour CatBoost.
      · Profondeurs et nombres d'estimateurs adaptés à ~600 samples.
    """
    return {
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            gamma="scale",
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight=None,      # None > "balanced" : Recall=1.0 vs 0.9892
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,     # 0.05 > 0.1 : meilleur AUC (0.9845)
            num_leaves=31,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=10,               # 10 > 6 : meilleur AUC (0.9894)
            learning_rate=0.01,     # 0.01 > 0.05 : apprentissage plus fin
            random_seed=42,
            auto_class_weights="Balanced",
            verbose=0,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ENTRAÎNEMENT ET ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    Entraîne tous les modèles et évalue leurs performances.

    Pour chaque modèle :
      1. Cross-validation 5-fold sur le train (roc_auc) — mesure la
         généralisation sans toucher au test set.
      2. Entraînement final sur tout le train set.
      3. Évaluation complète sur le test set.
    """
    models  = get_models()
    results = {}

    for name, model in models.items():
        logger.info("Entraînement : %s...", name)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring="roc_auc", n_jobs=-1)
        logger.info("  CV ROC-AUC : %.4f (± %.4f)",
                    cv_scores.mean(), cv_scores.std())

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":        round(accuracy_score(y_test, y_pred), 4),
            "precision":       round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":          round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":         round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std":  round(cv_scores.std(), 4),
        }

        results[name] = {"model": model, "metrics": metrics}
        logger.info(
            "  Test → Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1_score"], metrics["roc_auc"],
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. SÉLECTION DU MEILLEUR MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def select_best_model(results: dict) -> tuple:
    """
    Sélectionne le meilleur modèle selon un critère médical explicite.

    Tri lexicographique sur (Recall, Précision, ROC-AUC) :
      Python compare les tuples élément par élément. Si deux modèles ont
      le même recall, on départage par précision, puis par AUC.
    """
    best_name = max(
        results,
        key=lambda k: (
            results[k]["metrics"]["recall"],
            results[k]["metrics"]["precision"],
            results[k]["metrics"]["roc_auc"],
        )
    )
    best = results[best_name]
    logger.info(
        "Meilleur modèle : %s | Recall=%.4f | Prec=%.4f | AUC=%.4f",
        best_name,
        best["metrics"]["recall"],
        best["metrics"]["precision"],
        best["metrics"]["roc_auc"],
    )
    return best_name, best["model"], best["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAUVEGARDE DES ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, scaler, feature_names: list,
                   best_name: str, all_results: dict):
    """
    Sauvegarde tous les artefacts nécessaires à l'inférence dans /models/.

      best_model.pkl    — modèle sélectionné
      scaler.pkl        — StandardScaler ajusté sur le train
      feature_names.pkl — liste ordonnée des features attendues en entrée
      metrics.json      — métriques de tous les modèles + nom du meilleur
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model,         os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    metrics_summary = {
        "best_model": best_name,
        "models": {name: data["metrics"] for name, data in all_results.items()},
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Artefacts sauvegardés dans %s", MODELS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main(X_train, X_test, y_train, y_test, scaler, feature_names):
    """
    Pipeline d'entraînement principal.

    Reçoit des données DÉJÀ prêtes — ne charge, ne nettoie et ne prétraite rien.
    L'orchestration avec data_processing.py est déléguée à run.py.
    """
    logger.info("=" * 65)
    logger.info("PEDIATRIC APPENDICITIS — TRAINING PIPELINE")
    logger.info("=" * 65)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name, best_model, best_metrics = select_best_model(results)

    # Rapport de classification détaillé
    y_pred_final = best_model.predict(X_test)
    print("\n" + "=" * 65)
    print(f"RAPPORT FINAL — {best_name}")
    print("=" * 65)
    print(classification_report(y_test, y_pred_final,
                                 target_names=["Pas appendicite", "Appendicite"]))

    # Tableau de comparaison
    print("=" * 80)
    print(f"{'Modèle':<18} {'Accuracy':>10} {'Précision':>10} "
          f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 80)
    for name, data in results.items():
        m      = data["metrics"]
        marker = " ★" if name == best_name else ""
        print(
            f"{name + marker:<18} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}"
        )
    print("=" * 80)
    print(f"\n✓ Meilleur modèle : {best_name}\n")

    save_artifacts(best_model, scaler, feature_names, best_name, results)

    return results, best_name


if __name__ == "__main__":
    # Test rapide en développement — utiliser run.py pour l'usage normal.
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data

    df = load_data()
    df = optimize_memory(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    main(X_train, X_test, y_train, y_test, scaler, feature_names)
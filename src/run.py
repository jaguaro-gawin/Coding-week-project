"""
Pipeline d'orchestration — Pediatric Appendicitis Diagnosis
============================================================
Ce script est le seul endroit où les deux modules communiquent.

Responsabilité : coordonner data_processing.py et train_model.py
sans que l'un connaisse les détails internes de l'autre.

          ┌─────────────────────┐
          │   data_processing   │  ← chargement, nettoyage, prétraitement
          └────────┬────────────┘
                   │ X_train, X_test, y_train, y_test, scaler, feature_names
          ┌────────▼────────────┐
          │    train_model      │  ← entraînement, sélection, sauvegarde
          └─────────────────────┘

Structure attendue :
  projet/
    ├── src/
    │   ├── run.py              ← ce fichier
    │   ├── data_processing.py
    │   └── train_model.py
    ├── data/
    │   └── appendicitis.csv
    └── models/                 ← créé automatiquement après l'entraînement
"""

import sys
import os

# On remonte d'un niveau (src/ → projet/) pour que Python trouve
# les autres modules du projet si nécessaire.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Imports directs sans préfixe "src." — on est déjà dans le dossier src/
from data_processing import load_data, optimize_memory, clean_data, preprocess_data
from train_model import main as train


def run_pipeline():
    # ── Étape 1 : préparation des données (data_processing.py) ───────────────
    df = load_data()
    df = optimize_memory(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # ── Étape 2 : entraînement (train_model.py) ───────────────────────────────
    # On passe les données prêtes — train_model ne sait pas d'où elles viennent.
    results, best_name = train(X_train, X_test, y_train, y_test, scaler, feature_names)

    return results, best_name


if __name__ == "__main__":
    run_pipeline()
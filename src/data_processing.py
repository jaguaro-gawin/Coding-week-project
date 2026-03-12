"""
Data Processing Module — Pediatric Appendicitis Diagnosis  (v3)
===============================================================
Responsabilité : transformer les données brutes en données ML-ready.

Pipeline de nettoyage (clean_data) — stratégie d'imputation en cascade :
  1. Suppression des colonnes cibles parasites (Management, Severity)
  2. Suppression des colonnes avec > 50% de NaN
  3. Reconstruction du BMI depuis poids et taille (formule exacte), puis
     suppression de poids et taille pour éviter la redondance
  4. Imputation par corrélation pour les paires fortement liées (|r| > 0.69) :
     on ajuste une régression linéaire sur les lignes complètes, puis on
     prédit les NaN — bien plus précis que la médiane globale
  5. Médiane en dernier recours pour les NaN résiduels
  6. Suppression des doublons
  7. Capping des outliers par IQR (on ne supprime pas, on ramène aux bornes)

Feature Engineering dans clean_data :
  - WBC_CRP_Ratio             : ratio WBC / (CRP + 0.1)
  - Neutrophil_WBC_Interaction : Neutrophil_Percentage × WBC_Count
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Charge le dataset Regensburg Pediatric Appendicitis.
    Télécharge depuis l'UCI ML Repository si le fichier local est absent.
    """
    csv_path = os.path.join(data_dir, "appendicitis.csv")

    if os.path.exists(csv_path):
        logger.info("Chargement depuis le fichier local : %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        logger.info("Téléchargement depuis UCI ML Repository...")
        os.makedirs(data_dir, exist_ok=True)
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=938)
            X = dataset.data.features
            y = dataset.data.targets
            if isinstance(y, pd.DataFrame):
                target = y["Diagnosis"] if "Diagnosis" in y.columns else y.iloc[:, 0]
            else:
                target = y
            df = pd.concat([X, target.rename("Diagnosis")], axis=1)
            df.to_csv(csv_path, index=False)
            logger.info("Dataset sauvegardé : %s (%d lignes, %d colonnes)",
                        csv_path, len(df), len(df.columns))
        except Exception as e:
            logger.error("Échec du téléchargement : %s", e)
            raise

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTIMISATION MÉMOIRE
# ─────────────────────────────────────────────────────────────────────────────

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast des types numériques, colonnes textuelles → category."""
    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Mémoire AVANT : %.2f MB", mem_before)
    df_opt = df.copy()

    for col in df_opt.select_dtypes(include=["float64"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
    for col in df_opt.select_dtypes(include=["int64", "int32", "int16"]).columns:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
    for col in df_opt.select_dtypes(include=["object"]).columns:
        if df_opt[col].nunique() / len(df_opt) < 0.5:
            df_opt[col] = df_opt[col].astype("category")

    mem_after = df_opt.memory_usage(deep=True).sum() / 1024**2
    logger.info("Mémoire APRÈS : %.2f MB (réduction %.1f%%)",
                mem_after, (1 - mem_after / mem_before) * 100)
    return df_opt


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPERS D'IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _impute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruit le BMI depuis les colonnes poids et taille quand elles existent,
    puis supprime poids et taille pour éliminer la redondance parfaite.

    Formule exacte : BMI = poids_kg / taille_m²
    Si la taille est en cm (valeur médiane > 3), on divise d'abord par 100.

    Trois cas possibles pour chaque ligne :
      a) BMI déjà connu → on le garde intact, rien n'est modifié.
      b) BMI manquant, poids ET taille disponibles → calcul exact.
      c) BMI manquant, poids ou taille également manquants → NaN conservé,
         sera traité par imputation par corrélation ou médiane à l'étape suivante.

    Pourquoi supprimer poids et taille ensuite ?
    Garder les trois introduirait une colinéarité parfaite (BMI est une
    fonction déterministe de poids et taille), ce qui perturbe les modèles
    linéaires et gonfle artificiellement l'importance de ces variables dans
    les arbres de décision.
    """
    df = df.copy()

    # Recherche insensible à la casse pour trouver les bonnes colonnes
    col_map    = {c.lower(): c for c in df.columns}
    bmi_col    = col_map.get("bmi")
    weight_col = col_map.get("weight") or col_map.get("weight_kg")
    height_col = (col_map.get("height") or col_map.get("height_cm")
                  or col_map.get("height_m"))

    if bmi_col is None:
        logger.info("Colonne BMI non trouvée — étape BMI ignorée.")
        return df

    if weight_col is None or height_col is None:
        logger.info("Colonnes poids/taille non trouvées — BMI non recalculable par formule.")
        return df

    # Détection automatique de l'unité : médiane > 3 → probable cm
    height_median = df[height_col].median()
    height_in_cm  = height_median > 3.0
    height_m      = df[height_col] / 100.0 if height_in_cm else df[height_col]
    if height_in_cm:
        logger.info("Taille détectée en cm (médiane=%.1f) — conversion en mètres.", height_median)

    # On ne calcule que les BMI manquants où poids ET taille sont disponibles
    mask_missing  = df[bmi_col].isna()
    mask_has_data = df[weight_col].notna() & df[height_col].notna() & (height_m > 0)
    mask_calc     = mask_missing & mask_has_data

    n_calc = mask_calc.sum()
    if n_calc > 0:
        df.loc[mask_calc, bmi_col] = (
            df.loc[mask_calc, weight_col] / (height_m[mask_calc] ** 2)
        )
        logger.info("BMI calculé par formule pour %d lignes.", n_calc)

    n_still = df[bmi_col].isna().sum()
    if n_still > 0:
        logger.info("%d BMI toujours manquants → traités à l'étape suivante.", n_still)

    # Suppression poids + taille
    cols_to_drop = [c for c in [weight_col, height_col] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Colonnes supprimées après reconstruction BMI : %s", cols_to_drop)
    return df


def _impute_by_correlation(df: pd.DataFrame,
                            corr_threshold: float = 0.69,
                            target_col: str = "Diagnosis") -> pd.DataFrame:
    """
    Imputation par régression linéaire pour les paires fortement corrélées.

    Pourquoi est-ce mieux que la médiane ?
    La médiane utilise uniquement la distribution marginale d'une variable,
    en ignorant totalement le profil du patient. La régression utilise la
    relation conditionnelle : si WBC_Count = 15 000 et que Neutrophil_%
    lui est corrélé à 0.85, on peut prédire une valeur de Neutrophil_%
    qui respecte ce lien biologique, bien plus réaliste que la médiane globale.

    Algorithme pour chaque paire (A, B) avec |r| ≥ seuil :
      - Si A a des NaN et B est connu pour ces lignes → régression A ~ B
        ajustée sur les lignes complètes, puis prédiction des NaN de A.
      - Si B a des NaN et A est connu → régression B ~ A, même logique.
      - Si A et B sont simultanément NaN → impossible, on laisse pour le fallback.

    On trie les paires par corrélation décroissante pour exploiter en premier
    les relations les plus solides — une valeur prédite par r=0.95 sera
    elle-même disponible pour aider une prédiction suivante.
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col]

    # Matrice de corrélation calculée sur les lignes sans NaN (comportement par défaut)
    corr_matrix = df[num_cols].corr().abs()

    # Collecte des paires du triangle supérieur uniquement (évite les doublons)
    pairs = [
        (col_a, col_b, corr_matrix.loc[col_a, col_b])
        for i, col_a in enumerate(num_cols)
        for col_b in num_cols[i+1:]
        if corr_matrix.loc[col_a, col_b] >= corr_threshold
    ]

    if not pairs:
        logger.info("Aucune paire avec |r| ≥ %.2f — imputation par corrélation ignorée.",
                    corr_threshold)
        return df

    logger.info("%d paire(s) identifiée(s) pour imputation (|r| ≥ %.2f).",
                len(pairs), corr_threshold)

    total_imputed = 0
    # Trier par corrélation décroissante : les meilleures paires d'abord
    for col_a, col_b, r in sorted(pairs, key=lambda x: -x[2]):
        for target_var, predictor_var in [(col_a, col_b), (col_b, col_a)]:
            nan_target    = df[target_var].isna()
            nan_predictor = df[predictor_var].isna()

            # Lignes à prédire : target_var manquant, predictor_var connu
            mask_predict = nan_target & ~nan_predictor
            if mask_predict.sum() == 0:
                continue

            # Lignes d'entraînement : les deux colonnes sont connues
            train_mask = ~nan_target & ~nan_predictor
            if train_mask.sum() < 5:
                # Pas assez de données pour une régression fiable
                continue

            reg = LinearRegression()
            reg.fit(df.loc[train_mask, [predictor_var]],
                    df.loc[train_mask, target_var])
            df.loc[mask_predict, target_var] = reg.predict(
                df.loc[mask_predict, [predictor_var]]
            )
            total_imputed += mask_predict.sum()
            logger.info("  %s ← %s (r=%.3f) : %d valeurs imputées.",
                        target_var, predictor_var, r, mask_predict.sum())

            # Relire les masques après chaque imputation :
            # target_var a peut-être été complété et peut maintenant servir
            # de prédicteur pour une autre variable dans les itérations suivantes.

    logger.info("Imputation par corrélation : %d valeurs imputées.", total_imputed)
    return df


def _drop_correlated_features(df: pd.DataFrame,
                               corr_threshold: float = 0.69,
                               target_col: str = "Diagnosis",
                               original_nan_counts: pd.Series = None) -> tuple:
    """
    Supprime les colonnes redondantes dans les paires fortement corrélées.

    Cette fonction doit être appelée APRÈS l'imputation et le feature engineering,
    car on a besoin des deux colonnes d'une paire pour :
      a) imputer l'une à partir de l'autre (étape 4)
      b) créer des features d'interaction (étape 5)
    On ne supprime qu'une fois ces deux utilisations terminées.

    Algorithme glouton (greedy) :
      On utilise une approche gloutonne plutôt que paire par paire pour gérer
      les clusters de corrélation (A-B et B-C fortement corrélées : une approche
      naïve pourrait garder A et C qui sont peut-être elles aussi corrélées).

      À chaque itération :
        1. Calculer la matrice de corrélation sur les colonnes restantes.
        2. Compter pour chaque colonne combien d'autres colonnes elle dépasse
           le seuil (son "score de redondance").
        3. Parmi les colonnes avec score > 0, identifier celle à supprimer :
             → Priorité 1 : garder celle avec la plus forte corrélation à la cible.
               La colonne la plus discriminante pour le diagnostic est la plus précieuse.
             → Priorité 2 (ex æquo) : garder celle qui avait le moins de NaN à
               l'origine. Ses valeurs sont "plus vraies" que des valeurs imputées.
             → Priorité 3 (ex æquo) : garder la première alphabétiquement
               (critère stable et reproductible).
        4. Supprimer cette colonne et recommencer jusqu'à ce qu'aucune paire
           ne dépasse le seuil.

    Args:
        df                 : DataFrame après imputation et feature engineering.
        corr_threshold     : Seuil de corrélation (défaut 0.69).
        target_col         : Colonne cible à exclure de la suppression.
        original_nan_counts: pd.Series avec le nombre de NaN par colonne
                             AVANT imputation, pour guider le choix.

    Returns:
        (df_reduced, dropped_columns) — DataFrame allégé et liste des colonnes supprimées.
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col]

    # Corrélation de chaque feature avec la cible — plus c'est élevé, plus on veut garder
    # La colonne cible est encore au format texte ("appendicitis" / "no appendicitis")
    # à ce stade — preprocess_data n'a pas encore été appelée.
    # On l'encode temporairement en 0/1 uniquement pour le calcul de corrélation,
    # sans modifier le DataFrame réel.
    target_numeric = df[target_col].copy()
    if target_numeric.dtype.name in ("object", "category"):
        target_numeric = target_numeric.astype(str).str.strip().str.lower()
        target_numeric = target_numeric.map(
            lambda v: 0 if "no" in v else (1 if "appendicitis" in v else np.nan)
        )

    # On construit un DataFrame temporaire numérique pour la corrélation
    df_for_corr = df[num_cols].copy()
    df_for_corr[target_col] = target_numeric.values

    target_corr = df_for_corr.corr()[target_col].abs().drop(target_col)

    dropped = []

    # Boucle gloutonne : on continue tant qu'il reste des paires au-dessus du seuil
    while True:
        # Recalculer sur les colonnes restantes à chaque itération
        remaining = [c for c in num_cols if c not in dropped]
        if len(remaining) < 2:
            break

        corr_matrix = df[remaining].corr().abs()

        # Score de redondance : combien d'autres colonnes dépassent le seuil
        # (on exclut la diagonale qui vaut toujours 1.0)
        np.fill_diagonal(corr_matrix.values, 0)
        redundancy_score = (corr_matrix >= corr_threshold).sum()

        if redundancy_score.max() == 0:
            # Plus aucune paire au-dessus du seuil : on a terminé
            break

        # Identifier la colonne la plus redondante
        # En cas d'égalité du score, on départage par corrélation avec la cible
        # (on veut SUPPRIMER celle qui est la MOINS corrélée à la cible)
        candidates = redundancy_score[redundancy_score > 0]

        # Trouver la colonne à supprimer :
        # parmi les candidates, celle avec la plus faible corrélation à la cible
        # — car c'est la moins utile pour prédire le diagnostic.
        col_to_drop = min(
            candidates.index,
            key=lambda c: (
                target_corr.get(c, 0),          # priorité 1 : garder les + corrélées à la cible
                -(original_nan_counts.get(c, 0)  # priorité 2 : garder celles avec moins de NaN
                  if original_nan_counts is not None else 0),
                c                                # priorité 3 : ordre alphabétique (stable)
            )
        )

        # Identifier avec quelles colonnes elle était en conflit (pour le log)
        conflicting = corr_matrix[col_to_drop][corr_matrix[col_to_drop] >= corr_threshold].index.tolist()
        logger.info(
            "Étape 9 — Suppression '%s' (r ≥ %.2f avec : %s | corr_cible=%.3f)",
            col_to_drop, corr_threshold, conflicting,
            target_corr.get(col_to_drop, 0)
        )
        dropped.append(col_to_drop)

    if dropped:
        df = df.drop(columns=dropped)
        logger.info(
            "Étape 9 — %d colonne(s) supprimées pour redondance : %s",
            len(dropped), dropped
        )
    else:
        logger.info("Étape 9 — Aucune colonne redondante détectée (seuil %.2f).", corr_threshold)

    return df, dropped


# ─────────────────────────────────────────────────────────────────────────────
# 4. NETTOYAGE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame,
               missing_threshold: float = 0.50,
               corr_threshold: float = 0.69) -> pd.DataFrame:
    """
    Nettoie et enrichit le dataset en 9 étapes ordonnées.

    Étape 1 — Suppression des colonnes parasites (Management, Severity).
    Étape 2 — Suppression des colonnes features avec > missing_threshold de NaN.
    Étape 3 — Reconstruction du BMI par formule, suppression poids et taille.
    Étape 4 — Imputation par régression pour les paires |r| > corr_threshold.
    Étape 5 — Feature engineering (WBC_CRP_Ratio, Neutrophil_WBC_Interaction).
               Créé APRÈS imputation pour être calculé sur données complètes.
    Étape 6 — Médiane/mode en dernier recours pour les NaN résiduels.
    Étape 7 — Suppression des doublons.
    Étape 8 — Capping IQR des outliers.
    Étape 9 — Suppression des colonnes redondantes (paires |r| > corr_threshold).
               Exécutée EN DERNIER car les étapes 4 et 5 ont besoin des deux
               colonnes d'une paire pour imputer et créer les features d'interaction.
               On garde la colonne la plus corrélée à la cible (Diagnosis).
    """
    logger.info("Nettoyage : %d lignes, %d colonnes.", *df.shape)
    df_clean   = df.copy()
    target_col = "Diagnosis"

    # ── Étape 1 ──────────────────────────────────────────────────────────────
    # Quatre catégories de suppressions, toutes justifiées :
    #
    # . Variables cibles UCI — ne sont jamais des prédicteurs :
    #     - Management, Severity
    #
    # . Data leakage post-hospitalisation :
    #     - Length_of_Stay : connue uniquement à la sortie du patient
    #
    # . Biais circulaire — ont participé à construire la cible Diagnosis :
    #     (doc UCI : label conservatif = Alvarado >= 4 ET Appendix_Diameter >= 6mm)
    #     - Alvarado_Score, Paedriatic_Appendicitis_Score
    #
    # . Non validées par la littérature comme prédicteurs indépendants :
    #     - Ketones_in_Urine  : non spécifique, absent des modèles multivariés
    #     - RBC_in_Urine      : idem
    #     - WBC_in_Urine      : idem
    #     - Dysuria           : davantage associé à l'infection urinaire
    #     - Stool             : peu mentionné comme prédicteur indépendant
    #     - US_Performed      : décision clinique, pas caractéristique du patient
    #     - Hemoglobin, RDW, Thrombocyte_Count, RBC_Count : marqueurs hématologiques
    #       généraux, peu spécifiques à l'appendicite, non retenus dans les modèles multivariés
    TARGET_COLS    = ["Management", "Severity"]
    LEAKAGE_COLS   = ["Length_of_Stay"]
    CIRCULAR_COLS  = ["Alvarado_Score", "Paedriatic_Appendicitis_Score"]
    WEAK_COLS      = ["Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine",
                      "Dysuria", "Stool", "US_Performed",
                      "Hemoglobin", "RDW", "Thrombocyte_Count", "RBC_Count"]

    cols_to_drop = [c for c in TARGET_COLS + LEAKAGE_COLS + CIRCULAR_COLS + WEAK_COLS
                    if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info("Étape 1 — %d colonne(s) supprimées : %s", len(cols_to_drop), cols_to_drop)

    # ── Étape 2 ──────────────────────────────────────────────────────────────
    feature_cols = [c for c in df_clean.columns if c != target_col]
    missing_rate = df_clean[feature_cols].isnull().mean()
    high_missing = missing_rate[missing_rate > missing_threshold].index.tolist()
    if high_missing:
        logger.info(
            "Étape 2 — %d colonne(s) supprimées (NaN > %.0f%%) :\n%s",
            len(high_missing), missing_threshold * 100,
            missing_rate[high_missing].sort_values(ascending=False).to_string()
        )
        df_clean = df_clean.drop(columns=high_missing)
    else:
        logger.info("Étape 2 — Aucune colonne > %.0f%% NaN.", missing_threshold * 100)

    # Mémoriser le nombre de NaN par colonne AVANT imputation.
    # Ce comptage sera utilisé à l'étape 9 pour départager les colonnes
    # à supprimer : on préfère garder celles qui avaient le moins de NaN
    # (leurs valeurs sont "plus réelles" que des valeurs imputées).
    original_nan_counts = df_clean.isnull().sum()

    # ── Étape 3 ──────────────────────────────────────────────────────────────
    logger.info("Étape 3 — Reconstruction BMI...")
    df_clean = _impute_bmi(df_clean)

    # ── Étape 4 ──────────────────────────────────────────────────────────────
    logger.info("Étape 4 — Imputation par corrélation (|r| ≥ %.2f)...", corr_threshold)
    df_clean = _impute_by_correlation(df_clean, corr_threshold, target_col)

    # ── Étape 5 ──────────────────────────────────────────────────────────────
    if "WBC_Count" in df_clean.columns and "CRP" in df_clean.columns:
        df_clean["WBC_CRP_Ratio"] = df_clean["WBC_Count"] / (df_clean["CRP"] + 0.1)
        logger.info("Étape 5 — Feature 'WBC_CRP_Ratio' créée.")

    if "Neutrophil_Percentage" in df_clean.columns and "WBC_Count" in df_clean.columns:
        df_clean["Neutrophil_WBC_Interaction"] = (
            df_clean["Neutrophil_Percentage"] * df_clean["WBC_Count"]
        )
        logger.info("Étape 5 — Feature 'Neutrophil_WBC_Interaction' créée.")

    # ── Étape 6 ──────────────────────────────────────────────────────────────
    num_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                if c != target_col]
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

    nan_num = df_clean[num_cols].isnull().sum().sum()
    nan_cat = df_clean[cat_cols].isnull().sum().sum() if cat_cols else 0

    if nan_num > 0:
        logger.info("Étape 6 — %d NaN numériques résiduels → médiane.", nan_num)
        df_clean[num_cols] = SimpleImputer(strategy="median").fit_transform(
            df_clean[num_cols])
    if nan_cat > 0:
        logger.info("Étape 6 — %d NaN catégoriels résiduels → mode.", nan_cat)
        df_clean[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            df_clean[cat_cols])
    if nan_num == 0 and nan_cat == 0:
        logger.info("Étape 6 — Aucun NaN résiduel.")

    # ── Étape 7 ──────────────────────────────────────────────────────────────
    n_dupes = df_clean.duplicated().sum()
    if n_dupes > 0:
        df_clean = df_clean.drop_duplicates()
        logger.info("Étape 7 — %d doublon(s) supprimé(s).", n_dupes)

    # ── Étape 8 ──────────────────────────────────────────────────────────────
    numeric_features = [c for c in df_clean.select_dtypes(include=[np.number]).columns
                        if c != target_col]
    total_capped = 0
    for col in numeric_features:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out  = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        if n_out > 0:
            df_clean[col]  = df_clean[col].clip(lower, upper)
            total_capped  += n_out
    if total_capped > 0:
        logger.info("Étape 8 — %d valeurs cappées (IQR).", total_capped)

    # ── Étape 9 : suppression des colonnes redondantes ───────────────────────
    # On passe les NaN originaux (avant imputation) pour guider le choix :
    # en cas d'égalité sur la corrélation avec la cible, on préfère garder
    # la colonne qui avait le moins de NaN à l'origine.
    logger.info("Étape 9 — Suppression colonnes redondantes (|r| ≥ %.2f)...", corr_threshold)
    df_clean, _ = _drop_correlated_features(
        df_clean,
        corr_threshold=corr_threshold,
        target_col=target_col,
        original_nan_counts=original_nan_counts,
    )

    logger.info("Nettoyage terminé : %d lignes, %d colonnes.", *df_clean.shape)
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRÉTRAITEMENT ML
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "Diagnosis",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Prépare les données nettoyées pour l'entraînement ML.

    Ordre des opérations (ne pas inverser — risque de data leakage) :
      1. Encodage de la cible et one-hot des catégorielles
      2. Split stratifié train/test  ← avant le scaling
      3. Scaling (StandardScaler)    ← fit sur train uniquement

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    logger.info("Prétraitement ML...")
    df_proc = df.copy()

    # ── Encodage de la cible ─────────────────────────────────────────────────
    if df_proc[target_col].dtype.name == "category":
        df_proc[target_col] = df_proc[target_col].astype(str)

    if df_proc[target_col].dtype == object or isinstance(df_proc[target_col].iloc[0], str):
        unique_vals = df_proc[target_col].unique()
        mapping = {}
        for val in unique_vals:
            val_str = str(val).strip().lower()
            if "no" in val_str:
                mapping[val] = 0
            elif "appendicitis" in val_str:
                mapping[val] = 1
            else:
                try:
                    mapping[val] = int(val)
                except (ValueError, TypeError):
                    mapping[val] = 0
        df_proc[target_col] = df_proc[target_col].map(mapping)
        logger.info("Mapping cible : %s", mapping)

    df_proc = df_proc.dropna(subset=[target_col])
    y = df_proc[target_col].astype(int)
    X = df_proc.drop(columns=[target_col])

    # ── One-hot encoding ─────────────────────────────────────────────────────
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        logger.info("One-hot encoding : %s", cat_cols)

    feature_names = X.columns.tolist()

    # ── Split stratifié AVANT scaling (évite le data leakage) ────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ── Scaling : fit sur train uniquement, transform sur les deux ────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    logger.info(
        "Train : %d | Test : %d | Taux positifs — train : %.1f%%, test : %.1f%%",
        len(X_train), len(X_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def get_class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts()
    return {
        "counts":      counts.to_dict(),
        "percentages": (counts / len(y) * 100).round(1).to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"\nShape brut : {df.shape}")
    df_opt   = optimize_memory(df)
    df_clean = clean_data(df_opt)
    print(f"Shape après nettoyage : {df_clean.shape}")
    print(f"Colonnes : {list(df_clean.columns)}")
    print(f"NaN résiduels : {df_clean.isnull().sum().sum()}")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df_clean)
    print(f"\nFeatures ({len(features)}) : {features}")
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"Distribution : {get_class_distribution(pd.Series(y_train))}")
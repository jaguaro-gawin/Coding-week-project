"""
Automated Tests for Pediatric Appendicitis Diagnosis
=====================================================
Tests for data processing, memory optimization, and model prediction.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import load_data, optimize_memory, clean_data, preprocess_data


class TestDataProcessing:
    """Tests for the data processing pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load dataset once for all tests."""
        self.df = load_data()

    def test_data_loads_successfully(self):
        """Verify dataset loads and has expected shape."""
        assert self.df is not None
        assert len(self.df) > 0
        assert len(self.df.columns) > 10
        assert "Diagnosis" in self.df.columns

    def test_data_shape(self):
        """Verify expected number of rows and columns."""
        assert self.df.shape[0] == 782, f"Expected 782 rows, got {self.df.shape[0]}"
        assert self.df.shape[1] >= 50, f"Expected >= 50 columns, got {self.df.shape[1]}"

    def test_optimize_memory_reduces_usage(self):
        """Verify optimize_memory reduces DataFrame memory usage."""
        mem_before = self.df.memory_usage(deep=True).sum()
        df_opt = optimize_memory(self.df)
        mem_after = df_opt.memory_usage(deep=True).sum()

        assert mem_after < mem_before, "Memory was not reduced"
        reduction = (1 - mem_after / mem_before) * 100
        assert reduction > 30, f"Expected >30% reduction, got {reduction:.1f}%"

    def test_optimize_memory_preserves_data(self):
        """Verify optimize_memory doesn't change data values."""
        df_opt = optimize_memory(self.df)
        assert len(df_opt) == len(self.df)
        assert list(df_opt.columns) == list(self.df.columns)

    def test_missing_values_handling(self):
        """Verify clean_data handles missing values."""
        df_opt = optimize_memory(self.df)
        df_clean = clean_data(df_opt)

        # After cleaning, numeric columns should have no NaNs
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df_clean[col].isnull().sum() == 0, f"Column {col} still has NaN values"

    def test_clean_data_preserves_rows(self):
        """Verify cleaning doesn't excessively reduce data."""
        df_opt = optimize_memory(self.df)
        df_clean = clean_data(df_opt)
        # Should keep at least 90% of rows
        assert len(df_clean) >= len(self.df) * 0.9, "Too many rows removed"

    def test_preprocess_produces_valid_splits(self):
        """Verify preprocessing produces valid train/test splits."""
        df_opt = optimize_memory(self.df)
        df_clean = clean_data(df_opt)
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df_clean)

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert len(feature_names) == X_train.shape[1]

    def test_target_encoding(self):
        """Verify target is encoded as binary 0/1."""
        df_opt = optimize_memory(self.df)
        df_clean = clean_data(df_opt)
        _, _, y_train, y_test, _, _ = preprocess_data(df_clean)

        y_all = np.concatenate([y_train, y_test])
        unique_values = set(y_all)
        assert unique_values == {0, 1}, f"Expected {{0, 1}}, got {unique_values}"


class TestModelPrediction:
    """Tests for model loading and prediction."""

    def test_model_loading_and_prediction(self):
        """Verify trained model loads and produces valid predictions."""
        import joblib

        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        # Check model files exist
        assert os.path.exists(os.path.join(models_dir, "best_model.pkl")), "Model file not found"
        assert os.path.exists(os.path.join(models_dir, "scaler.pkl")), "Scaler file not found"
        assert os.path.exists(os.path.join(models_dir, "feature_names.pkl")), "Feature names not found"

        # Load model
        model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))

        # Create a dummy input
        dummy_input = np.zeros((1, len(feature_names)))
        dummy_scaled = scaler.transform(dummy_input)

        # Predict
        prediction = model.predict(dummy_scaled)
        probabilities = model.predict_proba(dummy_scaled)

        assert prediction[0] in [0, 1], "Invalid prediction"
        assert probabilities.shape == (1, 2), "Invalid probabilities shape"
        assert 0 <= probabilities[0][0] <= 1, "Probability out of range"
        assert 0 <= probabilities[0][1] <= 1, "Probability out of range"
        assert abs(probabilities[0].sum() - 1.0) < 1e-5, "Probabilities don't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

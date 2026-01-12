"""Tests for eyefeatures/features/shift.py - IndividualNormalization transformer."""

import numpy as np
import pytest

from eyefeatures.features.shift import IndividualNormalization


class TestIndividualNormalization:
    """Tests for IndividualNormalization transformer."""

    def test_init_single_pk(self):
        """Test initialization with single pk (list format)."""
        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={"sac_length": ["mean"]},
            dependent_features={"fix_duration": ["mean"]},
        )
        # Single pk should be converted to tuple internally
        assert isinstance(norm.pk, tuple)
        assert norm.pk[0] == ["participant"]

    def test_init_multiple_pk(self):
        """Test initialization with multiple pk levels (tuple format)."""
        norm = IndividualNormalization(
            pk=(["participant"], ["stimulus"]),
            independent_features=({"sac_length": ["mean"]}, {"fix_duration": ["mean"]}),
            dependent_features=({"sac_length": ["std"]}, {"fix_duration": ["std"]}),
        )
        assert len(norm.pk) == 2

    def test_init_validation_error(self):
        """Test that mismatched types raise assertion error."""
        with pytest.raises(AssertionError):
            IndividualNormalization(
                pk=["participant"],  # list
                independent_features=({"sac_length": ["mean"]},),  # tuple - mismatch
                dependent_features={"fix_duration": ["mean"]},  # dict
            )

    def test_fit_calculates_stats(self, extracted_features_df):
        """Test that fit calculates mean/std for dependent features."""
        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={},
            dependent_features={"fix_duration": ["mean"]},
        )
        norm.fit(extracted_features_df)

        assert norm.features_stats is not None
        assert len(norm.features_stats) == 1
        # Should have stats for each group
        assert "p1" in norm.features_stats[0]
        assert "p2" in norm.features_stats[0]

    def test_fit_returns_self(self, extracted_features_df):
        """Test that fit returns self for chaining."""
        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={},
            dependent_features={"fix_duration": ["mean"]},
        )
        assert norm.fit(extracted_features_df) is norm

    def test_transform_creates_new_columns(self, extracted_features_df):
        """Test transform creates new columns when inplace=False."""
        df = extracted_features_df.copy()

        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={"fix_duration": ["mean"]},
            dependent_features={},
            inplace=False,
        )
        result = norm.fit(df).transform(df)

        assert "fix_duration_mean_norm" in result.columns
        # Original column should be unchanged
        assert "fix_duration_mean" in result.columns

    def test_transform_returns_array(self, extracted_features_df):
        """Test that return_df=False returns numpy array."""
        df = extracted_features_df.copy()

        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={"fix_duration": ["mean"]},
            dependent_features={},
            return_df=False,
        )
        result = norm.fit(df).transform(df)

        assert isinstance(result, np.ndarray)

    def test_normalization_per_group(self, extracted_features_df):
        """Test that normalization is applied per group correctly."""
        df = extracted_features_df.copy()

        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={"sac_length": ["mean"]},
            dependent_features={},
            inplace=True,
        )
        norm.fit(df)
        result = norm.transform(df)

        # Each group should have mean ~0 after normalization
        p1_mean = result[result["participant"] == "p1"]["sac_length_mean"].mean()
        p2_mean = result[result["participant"] == "p2"]["sac_length_mean"].mean()

        assert pytest.approx(p1_mean, abs=1e-10) == 0.0
        assert pytest.approx(p2_mean, abs=1e-10) == 0.0

    def test_use_custom_mean_std(self, extracted_features_df):
        """Test using custom mean/std values."""
        df = extracted_features_df.copy()

        norm = IndividualNormalization(
            pk=["participant"],
            independent_features={},
            dependent_features={"fix_duration": ["mean"]},
            use_mean={"p1": 150.0, "p2": 200.0},
            use_std={"p1": 50.0, "p2": 50.0},
        )
        norm.fit(df)

        # Custom stats should be used instead of calculated
        assert norm.features_stats[0]["p1"]["fix_duration_mean"]["mean"] == 150.0
        assert norm.features_stats[0]["p2"]["fix_duration_mean"]["std"] == 50.0

"""Tests for eyefeatures/features/stats.py - Statistical feature transformers."""

import pandas as pd
import pytest

from eyefeatures.features.stats import (
    FixationFeatures,
    MicroSaccadeFeatures,
    RegressionFeatures,
    SaccadeFeatures,
)


class TestSaccadeFeatures:
    """Tests for SaccadeFeatures transformer."""

    def test_available_features_and_prefix(self):
        """Test available features include all expected and prefix is correct."""
        sac = SaccadeFeatures(features_stats={"length": ["mean"]})
        assert set(sac.available_feats) == {"length", "speed", "acceleration", "angle"}
        assert sac._fp == "sac"

    def test_transform_all_features(self, sample_df):
        """Test extraction of length, speed, and angle features."""
        sac = SaccadeFeatures(
            features_stats={
                "length": ["mean", "max", "sum"],
                "speed": ["mean"],
                "angle": ["mean", "std"],
            },
            x="x",
            y="y",
            t="t",
            duration="duration",
            pk=["participant", "stimulus"],
        )
        result = sac.fit(sample_df).transform(sample_df)

        assert isinstance(result, pd.DataFrame)
        for feat in [
            "sac_length_mean",
            "sac_length_max",
            "sac_length_sum",
            "sac_speed_mean",
            "sac_angle_mean",
            "sac_angle_std",
        ]:
            assert feat in result.columns

    def test_transform_with_aoi(self, sample_df):
        """Test saccade features with AOI grouping."""
        sac = SaccadeFeatures(
            features_stats={"length": ["mean"]},
            x="x",
            y="y",
            pk=["participant", "stimulus"],
            aoi="aoi",
        )
        result = sac.fit(sample_df).transform(sample_df)

        assert any("aoi[A]" in col for col in result.columns)
        assert any("aoi[B]" in col for col in result.columns)

    def test_multiple_groups(self, sample_df):
        """Test features with multiple participant groups."""
        sac = SaccadeFeatures(
            features_stats={"length": ["min", "max", "mean", "median", "std", "sum"]},
            x="x",
            y="y",
            pk=["participant", "stimulus"],
        )
        result = sac.fit(sample_df).transform(sample_df)

        # Should have rows for each unique pk combination
        assert len(result) > 1
        for stat in ["min", "max", "mean", "median", "std", "sum"]:
            assert f"sac_length_{stat}" in result.columns

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        # Missing x/y
        sac_no_x = SaccadeFeatures(features_stats={"length": ["mean"]}, x=None, y="y")
        with pytest.raises(AssertionError):
            sac_no_x._check_features_stats()

        # Invalid feature name
        sac_invalid = SaccadeFeatures(
            features_stats={"invalid": ["mean"]}, x="x", y="y"
        )
        with pytest.raises(AssertionError):
            sac_invalid._check_features_stats()


class TestRegressionFeatures:
    """Tests for RegressionFeatures transformer."""

    def test_available_features_and_prefix(self):
        """Test available features and prefix."""
        reg = RegressionFeatures(features_stats={"length": ["mean"]}, rule=(2,))
        assert set(reg.available_feats) == {
            "length",
            "speed",
            "acceleration",
            "angle",
            "mask",
        }
        assert reg._fp == "reg"

    def test_quadrant_and_angle_rules(self, sample_df):
        """Test regression with quadrant-based and angle-based rules."""
        # Quadrant rule
        reg_quad = RegressionFeatures(
            features_stats={"length": ["mean"], "angle": ["mean"]},
            rule=(2, 3),
            x="x",
            y="y",
            t="t",
            duration="duration",
            pk=["participant", "stimulus"],
        )
        result_quad = reg_quad.fit(sample_df).transform(sample_df)
        assert "reg_length_mean" in result_quad.columns
        assert "reg_angle_mean" in result_quad.columns

        # Angle rule with deviation
        reg_angle = RegressionFeatures(
            features_stats={"length": ["mean"]},
            rule=(180,),
            deviation=20,
            x="x",
            y="y",
            pk=["participant", "stimulus"],
        )
        result_angle = reg_angle.fit(sample_df).transform(sample_df)
        assert "reg_length_mean" in result_angle.columns

    def test_validation_errors(self):
        """Test parameter validation."""
        # Invalid quadrant
        reg_quad = RegressionFeatures(
            features_stats={"length": ["mean"]}, rule=(5,), x="x", y="y"
        )
        with pytest.raises(AssertionError):
            reg_quad._check_features_stats()

        # Invalid angle
        reg_angle = RegressionFeatures(
            features_stats={"length": ["mean"]}, rule=(400,), deviation=10, x="x", y="y"
        )
        with pytest.raises(AssertionError):
            reg_angle._check_features_stats()

        # Invalid deviation
        reg_dev = RegressionFeatures(
            features_stats={"length": ["mean"]},
            rule=(180,),
            deviation=200,
            x="x",
            y="y",
        )
        with pytest.raises(AssertionError):
            reg_dev._check_features_stats()


class TestFixationFeatures:
    """Tests for FixationFeatures transformer."""

    def test_available_features_and_prefix(self):
        """Test available features and prefix."""
        fix = FixationFeatures(features_stats={"duration": ["mean"]})
        assert set(fix.available_feats) == {"duration", "vad"}
        assert fix._fp == "fix"

    def test_transform_duration_and_vad(self, sample_df):
        """Test fixation duration and vad extraction."""
        fix = FixationFeatures(
            features_stats={"duration": ["mean", "sum", "max"], "vad": ["mean"]},
            duration="duration",
            dispersion="dispersion",
            pk=["participant", "stimulus"],
        )
        result = fix.fit(sample_df).transform(sample_df)

        for col in [
            "fix_duration_mean",
            "fix_duration_sum",
            "fix_duration_max",
            "fix_vad_mean",
        ]:
            assert col in result.columns


class TestMicroSaccadeFeatures:
    """Tests for MicroSaccadeFeatures transformer."""

    def test_init_and_prefix(self):
        """Test initialization parameters and prefix."""
        micro = MicroSaccadeFeatures(
            features_stats={"length": ["mean"]},
            min_dispersion=5.0,
            max_speed=100.0,
        )
        assert micro.min_dispersion == 5.0
        assert micro.max_speed == 100.0
        assert micro._fp == "microsac"

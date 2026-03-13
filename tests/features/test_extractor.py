"""Tests for eyefeatures/features/extractor.py - Transformers."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.features.extractor import BaseTransformer, Extractor
from eyefeatures.features.measures import RQAMeasures, ShannonEntropy
from eyefeatures.features.scanpath_dist import SimpleDistances
from eyefeatures.features.stats import FixationFeatures, SaccadeFeatures


class TestBaseTransformer:
    """Tests for BaseTransformer class."""

    def test_init_and_set_data(self):
        """Test initialization and set_data update attributes correctly."""
        transformer = BaseTransformer(
            x="x_coord",
            y="y_coord",
            t="timestamp",
            duration="dur",
            pk=["participant", "stimulus"],
            return_df=False,
        )
        assert transformer.x == "x_coord"
        assert transformer.pk == ["participant", "stimulus"]
        assert transformer.return_df is False

        # set_data should update attributes
        transformer.set_data(x="new_x", y="new_y", t="new_t", pk=["id"])
        assert transformer.x == "new_x"
        assert transformer.pk == ["id"]

    def test_fit_and_transform(self, sample_df):
        """Test fit returns self and transform returns correct output types."""
        transformer = BaseTransformer(return_df=True)
        result = transformer.fit(sample_df)
        assert result is transformer
        assert isinstance(transformer.transform(sample_df), pd.DataFrame)

        transformer_array = BaseTransformer(return_df=False)
        assert isinstance(transformer_array.transform(sample_df), np.ndarray)

    def test_check_init_validation(self):
        """Test _check_init passes for initialized values and raises for None."""
        transformer = BaseTransformer(x="x", y="y")
        transformer._check_init([("x", "x"), ("y", "y")])  # Should not raise

        transformer_none = BaseTransformer()
        with pytest.raises(RuntimeError, match="x is not initialized"):
            transformer_none._check_init([(None, "x")])


class TestExtractor:
    """Tests for Extractor class."""

    def test_fit_and_transform_workflow(self, sample_df):
        """Test complete fit/transform workflow including is_fitted flag."""
        extractor = Extractor(pk=["participant", "stimulus"])

        # Should not be fitted initially
        assert extractor.is_fitted is False
        with pytest.raises(RuntimeError, match="Class is not fitted"):
            extractor.transform(sample_df)

        # Fit should set flag and return self
        result = extractor.fit(sample_df)
        assert result is extractor
        assert extractor.is_fitted is True

        # Transform should return DataFrame or array based on return_df
        assert isinstance(extractor.transform(sample_df), pd.DataFrame)

        extractor_array = Extractor(pk=["participant", "stimulus"], return_df=False)
        extractor_array.fit(sample_df)
        assert isinstance(extractor_array.transform(sample_df), np.ndarray)

    def test_process_input_handles_na(self, sample_df):
        """Test _process_input drops NA values and raises on pk NA."""
        df_with_na = sample_df.copy()
        df_with_na.loc[0, "x"] = np.nan
        extractor = Extractor(pk=["participant", "stimulus"], warn=False)
        X, y = extractor._process_input(df_with_na)
        assert len(X) == len(sample_df) - 1

        # NA in pk should raise
        df_pk_na = sample_df.copy()
        df_pk_na.loc[0, "participant"] = np.nan
        with pytest.raises(ValueError, match="Found missing values in pk"):
            Extractor(pk=["participant", "stimulus"])._process_input(df_pk_na)

    def test_leave_pk_and_set_data_on_transformers(self, sample_df):
        """Test leave_pk includes pk columns and Extractor sets data on transformers."""
        # leave_pk test
        extractor = Extractor(pk=["participant", "stimulus"], leave_pk=True)
        extractor.fit(sample_df)
        result = extractor.transform(sample_df)
        assert "participant" in result.columns
        assert "stimulus" in result.columns

        # Extractor should call set_data on transformers
        transformer = BaseTransformer()
        transformer.get_feature_names_out = lambda: []  # define abstract method
        extractor_with_trans = Extractor(
            features=[transformer],
            x="x",
            y="y",
            t="t",
            duration="duration",
            pk=["participant", "stimulus"],
        )
        extractor_with_trans.fit(sample_df)
        assert transformer.x == "x"
        assert transformer.pk == ["participant", "stimulus"]

    def test_extractor_comprehensive(self, sample_df):
        """Test Extractor with transformers from multiple submodules."""
        features = [
            SaccadeFeatures(features_stats={"length": ["mean"]}, calc_without_aoi=True),
            FixationFeatures(
                features_stats={"duration": ["mean"]}, calc_without_aoi=True
            ),
            ShannonEntropy(),
            RQAMeasures(measures=["rec"]),
            SimpleDistances(methods=["euc"]),
        ]

        extractor = Extractor(
            features=features,
            pk=["participant", "stimulus"],
            path_pk=["participant"],
            x="x",
            y="y",
            duration="duration",
            t="t",
            aoi="aoi",
        )

        extractor.fit(sample_df)
        result = extractor.transform(sample_df)

        assert isinstance(result, pd.DataFrame)
        # Check for expected columns from each transformer type
        assert any("sac_length" in c for c in result.columns)
        assert any("fix_duration" in c for c in result.columns)
        assert "entropy" in result.columns
        assert any("rec" in c for c in result.columns)
        assert any("euc_dist" in c for c in result.columns)

        # Check grouping
        # sample_df has (p1, s1), (p1, s2), (p2, s1) -> 3 groups
        assert len(result) == 3

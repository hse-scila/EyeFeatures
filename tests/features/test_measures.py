"""Tests for eyefeatures/features/measures.py - Measure transformers."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.features.measures import (
    CorrelationDimension,
    FractalDimension,
    FuzzyEntropy,
    GriddedDistributionEntropy,
    HHTFeatures,
    HurstExponent,
    IncrementalEntropy,
    LyapunovExponent,
    PhaseEntropy,
    RQAMeasures,
    SaccadeUnlikelihood,
    SampleEntropy,
    ShannonEntropy,
    SpectralEntropy,
)


class TestMeasureTransformer:
    """Tests for MeasureTransformer (via ShannonEntropy)."""

    def test_init_and_fit(self, sample_df):
        """Test initialization and fit."""
        mt = ShannonEntropy(aoi="aoi", pk=["participant"])
        assert mt.fit(sample_df) == mt

    def test_transform_execution(self, sample_df):
        """Test basic transform execution."""
        mt = ShannonEntropy(aoi="aoi", pk=["participant"])
        res = mt.transform(sample_df)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 2  # 2 participants

    def test_transform_returns_dataframe(self, sample_df):
        """Test transform returns DataFrame when return_df=True."""
        mt = ShannonEntropy(aoi="aoi", pk=["participant"], return_df=True)
        res = mt.transform(sample_df)
        assert isinstance(res, pd.DataFrame)
        # assert list(res.index.names) == ["participant"]

    def test_transform_returns_ndarray(self, sample_df):
        """Test transform returns ndarray when return_df=False."""
        mt = ShannonEntropy(aoi="aoi", pk=["participant"], return_df=False)
        res = mt.transform(sample_df)
        assert isinstance(res, np.ndarray)

    def test_transform_splitting_by_pk(self, sample_df):
        """Test that transformation respects the grouping key."""
        mt = ShannonEntropy(aoi="aoi", pk=["participant"])
        res = mt.transform(sample_df)
        assert len(res) == 2


class TestEntropies:
    """Tests for all entropy measures."""

    def test_shannon_calculation(self, sample_df):
        """Test Shannon Entropy basic execution."""
        se = ShannonEntropy(aoi="aoi", pk=["participant"])
        res = se.fit(sample_df).transform(sample_df)
        assert "entropy" in res.columns
        assert (res["entropy"] >= 0).all()

    def test_shannon_constant_sequence(self):
        """Test Shannon Entropy on constant sequence (should be 0)."""
        df = pd.DataFrame({"aoi": ["A"] * 10, "p": ["1"] * 10})
        se = ShannonEntropy(aoi="aoi", pk=["p"])
        res = se.fit(df).transform(df)
        assert res.iloc[0, 0] == 0.0

    def test_spectral_calculation(self):
        """Test Spectral Entropy on sine wave."""
        t = np.linspace(0, 1, 100)
        x = np.sin(2 * np.pi * 5 * t)
        df = pd.DataFrame({"x": x, "y": x, "p": ["1"] * 100})
        se = SpectralEntropy(x="x", y="y", pk=["p"])
        res = se.fit(df).transform(df)
        assert "spectral_entropy" in res.columns
        val = res.iloc[0, 0]
        assert val >= 0

    def test_fuzzy_entropy_execution(self):
        """Test Fuzzy Entropy execution."""
        x = np.random.rand(50)
        df = pd.DataFrame({"x": x, "y": x, "participant": ["1"] * 50})
        fe = FuzzyEntropy(x="x", y="y", m=2, r=0.2, pk=["participant"])
        result = fe.fit(df).transform(df)
        val = result.iloc[0, 0]
        assert isinstance(val, (float, np.floating))

    def test_sample_entropy(self, sample_df):
        """Test Sample Entropy execution."""
        se = SampleEntropy(m=2, r=0.2, x="x", y="y", pk=["participant"])
        result = se.fit(sample_df).transform(sample_df)
        assert "sample_entropy_m=2_r=0.2" in result.columns
        assert isinstance(result.iloc[0, 0], (float, np.floating))

    def test_incremental_entropy(self, sample_df):
        """Test Incremental Entropy execution."""
        ie = IncrementalEntropy(x="x", y="y", pk=["participant"])
        result = ie.fit(sample_df).transform(sample_df)
        assert "incremental_entropy" in result.columns
        assert (result["incremental_entropy"] >= 0).all()

    def test_gridded_distribution_entropy(self, sample_df):
        """Test Gridded Distribution Entropy execution."""
        ge = GriddedDistributionEntropy(grid_size=5, x="x", y="y", pk=["participant"])
        result = ge.fit(sample_df).transform(sample_df)
        assert "gridded_entropy_grid_size_5" in result.columns
        assert (result.iloc[:, 0] >= 0).all()

    def test_phase_entropy(self, sample_df):
        """Test Phase Entropy execution."""
        pe = PhaseEntropy(m=2, tau=1, x="x", y="y", pk=["participant"])
        result = pe.fit(sample_df).transform(sample_df)
        assert "phase_entropy_m_2_tau_1" in result.columns
        assert (result.iloc[:, 0] >= 0).all()


class TestDynamics:
    """Tests for nonlinear dynamics measures (Hurst, Lyapunov, Fractal, CorrDim)."""

    def test_hurst_calculation_white_noise(self):
        """Test Hurst on white noise."""
        np.random.seed(42)
        n = 2000  # Increased length to satisfy 2^n_iters check (1024)
        x = np.random.randn(n)
        df = pd.DataFrame({"x": x, "p": ["1"] * n})

        he = HurstExponent(x="x", pk=["p"], return_df=True)
        res = he.fit(df).transform(df)
        val = res.iloc[0, 0]
        assert isinstance(val, (float, np.floating))

    def test_hurst_calculation_trend(self):
        """Test Hurst on trending data."""
        n = 2000
        x = np.linspace(0, 10, n)
        df = pd.DataFrame({"x": x, "p": ["1"] * n})

        he = HurstExponent(x="x", pk=["p"])
        res = he.fit(df).transform(df)
        val = res.iloc[0, 0]
        assert isinstance(val, (float, np.floating))

    def test_check_init_validates_length(self, sample_df):
        """Test initialization check for Hurst (too short sequence)."""
        he = HurstExponent(
            x="x", pk=["participant"], n_iters=10
        )  # Requires 1024 points

        # sample_df length is 10.
        with pytest.raises(AssertionError, match="must be of length more than"):
            he.fit(sample_df).transform(sample_df)

    def test_lyapunov_exponent(self, sample_df):
        """Test Lyapunov Exponent execution."""
        le = LyapunovExponent(m=2, tau=1, T=1, x="x", y="y", pk=["participant"])
        result = le.fit(sample_df).transform(sample_df)
        col = "lyapunov_exponent_m_2_tau_1_T_1"
        assert col in result.columns
        assert isinstance(result.iloc[0, 0], (float, np.floating))

    def test_fractal_dimension(self, sample_df):
        """Test Fractal Dimension execution."""
        fd = FractalDimension(m=2, tau=1, x="x", y="y", pk=["participant"])
        result = fd.fit(sample_df).transform(sample_df)
        col = "fractal_dim_m_2_tau_1"
        assert col in result.columns
        assert (result[col] >= 0).all()

    def test_correlation_dimension(self, sample_df):
        """Test Correlation Dimension execution."""
        cd = CorrelationDimension(m=2, tau=1, r=0.5, x="x", y="y", pk=["participant"])
        result = cd.fit(sample_df).transform(sample_df)
        col = "corr_dim_m_2_tau_1_r_0.5"
        assert col in result.columns
        assert (result[col] >= 0).all()


class TestAdvancedMeasures:
    """Tests for advanced measures like RQA, Saccade Unlikelihood, HHT."""

    def test_rqa_measures(self, sample_df):
        """Test RQA execution (returns multiple columns)."""
        rqa = RQAMeasures(
            x="x",
            y="y",
            pk=["participant"],
            rho=0.1,
            min_length=2,
            measures=["rec", "det"],
        )
        result = rqa.fit(sample_df).transform(sample_df)

        cols = result.columns.tolist()
        assert any("rec" in c for c in cols)
        assert any("det" in c for c in cols)
        for c in cols:
            if "participant" not in c and "stimulus" not in c:
                assert (result[c] >= 0).all()

    def test_saccade_unlikelihood(self, sample_df):
        """Test Saccade Unlikelihood execution."""
        su = SaccadeUnlikelihood(
            x="x",
            y="y",
            pk=["participant"],
            mu_p=100,
            sigma_p1=10,
            sigma_p2=10,
            mu_r=100,
            sigma_r1=10,
            sigma_r2=10,
            psi=0.9,
        )
        result = su.fit(sample_df).transform(sample_df)
        assert "saccade_nll" in result.columns
        assert isinstance(result.iloc[0, 0], (float, np.floating))
        assert pytest.approx(result.loc["p1", "saccade_nll"], 1.0) == 65.0
        assert pytest.approx(result.loc["p2", "saccade_nll"], 1.0) == 25.0

    def test_hht_features(self, sample_df):
        """Test HHT features execution."""
        # Create longer signal
        t = np.linspace(0, 1, 100)
        x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
        df = pd.DataFrame({"x": x, "y": x, "participant": ["p1"] * 100})

        hht = HHTFeatures(x="x", y="y", pk=["participant"], max_imfs=2)
        result = hht.fit(df).transform(df)

        assert len(result.columns) > 0
        assert any("imf" in c for c in result.columns)

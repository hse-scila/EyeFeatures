"""Tests for eyefeatures/features/complex.py."""

import numpy as np

from eyefeatures.features.complex import (
    calculate_topological_features,
    get_gaf,
    get_heatmap,
    get_heatmaps,
    get_hilbert_curve,
    get_hilbert_curve_enc,
    get_mtf,
    get_pca,
    get_rqa,
    hilbert_huang_transform,
    lower_star_filtration,
    persistence_curve,
    persistence_entropy_curve,
    vietoris_rips_filtration,
)


class TestComplexFeatures:
    """Tests for advanced feature extraction functions."""

    def test_heatmaps(self, sample_df):
        """Test get_heatmap and get_heatmaps."""
        x_vals = sample_df["x"].values
        y_vals = sample_df["y"].values
        # Use smaller shape to speed up
        shape = (10, 10)

        # Test single heatmap
        hm = get_heatmap(x_vals, y_vals, shape=shape)
        assert hm.shape == shape
        assert hm.sum() > 0

        # Test multiple heatmaps via pk
        hms = get_heatmaps(sample_df, x="x", y="y", shape=shape, pk=["participant"])
        # Expected shape: [N_groups, H, W]
        assert hms.ndim == 3
        # sample_df has 2 participants
        assert hms.shape[0] == 2
        assert hms.shape[1] == 10
        assert hms.shape[2] == 10

    def test_pca(self):
        """Test PCA function."""
        p = 2  # number of principal components to take
        n = 10  # number of rows in data
        m = 5  # number of columns in data (dimension)
        matrix = np.random.rand(n, m)

        eigenvectors, projection, row_means = get_pca(matrix, p=p)

        assert eigenvectors.shape == (m, p)  # m eigenvectors of size p
        assert projection.shape == (n, p)  # n vectors with reduced dimension p < m
        assert row_means.shape == (n,)  # means of each row

        # Test with cum_sum
        ev, proj, means = get_pca(matrix, reserve_info=0.9)
        assert proj.shape[0] >= 1

    def test_rqa(self, sample_df):
        """Test get_rqa function."""
        rqa_mat = get_rqa(
            sample_df.iloc[:4],
            x="x",
            y="y",
            metric=lambda p1, p2: np.linalg.norm(p1 - p2),
            rho=100.0,
        )
        assert rqa_mat.shape == (4, 4)
        # Current implementation fills only off-diagonals with 0 or 1, diag is 0
        for i in range(4):
            assert rqa_mat[i, i] == 0

    def test_mtf(self, sample_df):
        """Test get_mtf function."""
        # MTF returns (2, len, len)
        mtf_mat = get_mtf(sample_df.iloc[:4], x="x", y="y", n_bins=2, output_size=4)
        assert mtf_mat.ndim == 3
        assert mtf_mat.shape[0] == 2
        assert mtf_mat.shape[1] == 4
        assert mtf_mat.shape[2] == 4

    def test_gaf(self, sample_df):
        """Test get_gaf function (Gramian Angular Field)."""
        gaf_mat = get_gaf(sample_df.iloc[:4], x="x", y="y", field_type="difference")
        # Returns [2, N, N]
        assert gaf_mat.ndim == 3
        assert gaf_mat.shape[0] == 2
        assert gaf_mat.shape[1] == 4
        assert gaf_mat.shape[2] == 4

    def test_hilbert_curve(self, sample_df):
        """Test Hilbert curve mapping."""
        # Mapping to 1D
        h_vals = get_hilbert_curve(sample_df.iloc[:4], x="x", y="y", p=4)
        assert len(h_vals) == 4
        assert np.all(h_vals >= 0)

        # Encoding to feature vector
        h_enc = get_hilbert_curve_enc(sample_df.iloc[:4], x="x", y="y", p=4)
        assert len(h_enc) == 256

    def test_hilbert_huang_transform(self):
        """Test HHT (EMD)."""
        # EMD2D needs 2D input
        data = np.random.rand(10, 10)
        imfs = hilbert_huang_transform(data, max_imf=1)
        assert isinstance(imfs, np.ndarray)

    def test_topological_features(self):
        """Test TDA features using gudhi."""
        scanpath = np.random.rand(10, 2)
        time_series = np.random.rand(10)

        # Basic filtration
        diag, st = vietoris_rips_filtration(scanpath, max_dim=1, max_radius=0.5)
        assert isinstance(diag, list)

        ls_diag, ls_st = lower_star_filtration(time_series)
        assert ls_diag is not None

        # Curves
        if len(diag) > 0:
            pure_diag = [p[1] for p in diag]
            val = persistence_curve(pure_diag, t=0.1)
            assert isinstance(val, (float, np.float64, np.float32))

            ent = persistence_entropy_curve(pure_diag, t=0.1)
            assert isinstance(ent, (float, np.float64, np.float32))

        # Overall features
        pc, pe = calculate_topological_features(
            scanpath, time_series, max_dim=1, time_steps=5
        )
        assert len(pc) == 5
        assert len(pe) == 5

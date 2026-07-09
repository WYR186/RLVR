"""Unit tests for the pure-numpy Q metrics (briefing §6 Phase 0)."""
import numpy as np
import pytest

from src.metrics import anisotropy_metrics, dormant_metrics, spectrum_metrics

RNG = np.random.default_rng(0)


class TestSpectrumMetrics:
    def test_rank1_matrix_has_erank_1(self):
        u = RNG.normal(size=(200, 1))
        v = RNG.normal(size=(1, 64))
        m = spectrum_metrics(u @ v, center=False)
        assert m["erank"] == pytest.approx(1.0, abs=1e-6)
        assert m["participation_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert m["top1_var_share"] == pytest.approx(1.0, abs=1e-9)

    def test_equal_singular_values_give_full_erank(self):
        # orthogonal columns scaled equally -> all d singular values equal
        d = 32
        A = np.eye(d) * 5.0
        m = spectrum_metrics(A, center=False)
        assert m["erank"] == pytest.approx(d, rel=1e-6)
        assert m["erank_norm"] == pytest.approx(1.0, rel=1e-6)
        assert m["participation_ratio"] == pytest.approx(d, rel=1e-6)

    def test_isotropic_gaussian_near_full_rank(self):
        A = RNG.normal(size=(4096, 32))
        m = spectrum_metrics(A, center=True)
        assert m["erank"] > 30  # close to d=32
        assert 0.9 < m["erank_norm"] <= 1.0

    def test_erank_decreases_with_anisotropy(self):
        # same size, one direction inflated -> lower erank
        A = RNG.normal(size=(512, 64))
        B = A.copy()
        B[:, 0] *= 50
        assert spectrum_metrics(B)["erank"] < spectrum_metrics(A)["erank"]

    def test_centering_removes_mean_direction(self):
        # huge common mean vector: uncentered spectrum is dominated by it
        A = RNG.normal(size=(512, 64)) + 100.0
        centered = spectrum_metrics(A, center=True)
        uncentered = spectrum_metrics(A, center=False)
        assert uncentered["top1_var_share"] > 0.9
        assert centered["erank"] > uncentered["erank"]

    def test_zero_matrix(self):
        m = spectrum_metrics(np.zeros((16, 8)), center=False)
        assert m["erank"] == 0.0

    def test_topk_shares_monotone(self):
        A = RNG.normal(size=(512, 64))
        m = spectrum_metrics(A)
        assert m["top1_var_share"] <= m["top8_var_share"] <= m["top32_var_share"] <= 1.0

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            spectrum_metrics(np.zeros(10))


class TestAnisotropy:
    def test_identical_rows_have_cos_1(self):
        A = np.tile(RNG.normal(size=(1, 32)), (50, 1))
        m = anisotropy_metrics(A)
        assert m["anisotropy_uncentered"] == pytest.approx(1.0, abs=1e-9)

    def test_random_rows_near_zero(self):
        A = RNG.normal(size=(500, 256))
        m = anisotropy_metrics(A)
        assert abs(m["anisotropy_uncentered"]) < 0.05
        assert abs(m["anisotropy_centered"]) < 0.05

    def test_mean_shift_artifact_detected_by_pair(self):
        # random directions + large common offset: uncentered high, centered low
        A = RNG.normal(size=(200, 64)) + 20.0
        m = anisotropy_metrics(A)
        assert m["anisotropy_uncentered"] > 0.9
        assert abs(m["anisotropy_centered"]) < 0.1


class TestDormant:
    def test_no_dormant_when_uniform(self):
        m = dormant_metrics(np.ones(100))
        assert m["dormant_frac_tau0.025"] == 0.0
        assert m["dormant_frac_tau0.1"] == 0.0

    def test_exact_fraction(self):
        # 10 of 100 units silent, rest at 1.0 -> mean 0.9, silent score 0 < tau
        a = np.ones(100)
        a[:10] = 0.0
        m = dormant_metrics(a)
        assert m["dormant_frac_tau0.025"] == pytest.approx(0.1)
        assert m["dormant_frac_tau0.1"] == pytest.approx(0.1)

    def test_tau_ordering(self):
        a = RNG.uniform(0, 1, size=1000) ** 4  # heavy low tail
        m = dormant_metrics(a)
        assert m["dormant_frac_tau0.025"] <= m["dormant_frac_tau0.1"]

    def test_all_silent_layer(self):
        m = dormant_metrics(np.zeros(64))
        assert m["dormant_frac_tau0.1"] == 1.0

    def test_scale_invariance(self):
        a = RNG.uniform(0, 1, size=512)
        m1 = dormant_metrics(a)
        m2 = dormant_metrics(a * 1000)
        assert m1["dormant_frac_tau0.1"] == m2["dormant_frac_tau0.1"]

"""DPM++ 2M SDE 수식 검증.

MLX가 없어도 순수 numpy로 돌 수 있는 수학만 테스트한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stage1_layerdiff.configs import DiffusionConfig
from src.stage1_layerdiff.schedulers import (
    DPMSolverMultistep,
    DPMSolverState,
    build_sigma_schedule,
    dpmpp_2m_sde_step,
    karras_sigmas,
    make_alphas_cumprod,
    make_betas,
    sigma_from_alphas_cumprod,
)

# ============================================================
# Beta / sigma schedule
# ============================================================


class TestBetaSchedule:
    def test_scaled_linear_shape(self) -> None:
        cfg = DiffusionConfig()
        betas = make_betas(cfg)
        assert betas.shape == (cfg.num_train_steps,)
        assert betas.dtype == np.float64

    def test_scaled_linear_bounds(self) -> None:
        cfg = DiffusionConfig()
        betas = make_betas(cfg)
        # scaled_linear = (linspace(√b0, √b1, N))²
        # → first == beta_start, last == beta_end
        np.testing.assert_allclose(betas[0], cfg.beta_start, rtol=1e-10)
        np.testing.assert_allclose(betas[-1], cfg.beta_end, rtol=1e-10)

    def test_linear_schedule(self) -> None:
        cfg = DiffusionConfig(beta_schedule="linear")
        betas = make_betas(cfg)
        # Equal spacing
        deltas = np.diff(betas)
        np.testing.assert_allclose(deltas, deltas[0], rtol=1e-10)

    def test_unknown_schedule_raises(self) -> None:
        from dataclasses import replace

        cfg = replace(DiffusionConfig(), beta_schedule="nonsense")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown beta_schedule"):
            make_betas(cfg)


class TestAlphasCumprod:
    def test_is_monotone_decreasing(self) -> None:
        betas = make_betas(DiffusionConfig())
        alphas_cumprod = make_alphas_cumprod(betas)
        # α̅ starts close to 1, ends close to 0
        assert alphas_cumprod[0] > 0.99
        assert alphas_cumprod[-1] < 0.05
        assert (np.diff(alphas_cumprod) <= 0).all()

    def test_sigma_monotone(self) -> None:
        betas = make_betas(DiffusionConfig())
        alphas_cumprod = make_alphas_cumprod(betas)
        sigmas = sigma_from_alphas_cumprod(alphas_cumprod)
        # σ monotonically increasing in t
        assert (np.diff(sigmas) > 0).all()


# ============================================================
# Karras sigmas
# ============================================================


class TestKarrasSigmas:
    def test_shape_and_trailing_zero(self) -> None:
        sig = karras_sigmas(30, sigma_min=0.01, sigma_max=14.0)
        assert sig.shape == (31,)
        assert sig[-1] == 0.0

    def test_bounds(self) -> None:
        sig = karras_sigmas(30, sigma_min=0.01, sigma_max=14.0)
        # sig[0] should be sigma_max, sig[-2] should be sigma_min
        np.testing.assert_allclose(sig[0], 14.0, rtol=1e-6)
        np.testing.assert_allclose(sig[-2], 0.01, rtol=1e-6)

    def test_monotone_decreasing(self) -> None:
        sig = karras_sigmas(30, sigma_min=0.01, sigma_max=14.0)
        assert (np.diff(sig) < 0).all()

    def test_invalid_steps(self) -> None:
        with pytest.raises(ValueError, match=">= 2"):
            karras_sigmas(1, sigma_min=0.01, sigma_max=14.0)


class TestBuildSigmaSchedule:
    def test_length(self) -> None:
        cfg = DiffusionConfig()
        sig = build_sigma_schedule(cfg, num_inference_steps=30)
        assert sig.shape == (31,)
        assert sig[-1] == 0.0

    def test_without_karras(self) -> None:
        from dataclasses import replace

        cfg = replace(DiffusionConfig(), use_karras_sigmas=False)
        sig = build_sigma_schedule(cfg, num_inference_steps=20)
        assert sig.shape == (21,)
        assert sig[-1] == 0.0


# ============================================================
# DPM++ 2M SDE step
# ============================================================


class TestDPMSolverStep:
    def _zero_eps(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    def test_final_step_is_denoised(self) -> None:
        """sigma_next=0 일 때 eps=0 이면 x가 그대로 반환."""
        x = np.ones((1, 4, 8, 8), dtype=np.float32)
        state = DPMSolverState()
        out = dpmpp_2m_sde_step(
            eps_pred=self._zero_eps(x.shape),
            x_t=x,
            sigma=0.5,
            sigma_next=0.0,
            state=state,
        )
        np.testing.assert_allclose(out, x)
        assert state.last_denoised is not None

    def test_zero_eps_leaves_x_unchanged(self) -> None:
        """eps=0 → denoised == x_t → x_next == x_t (수학적 identity).

        Derivation with h = log(sigma/sigma_next), exp(-h) = sigma_next/sigma:
            x_next = (sigma_next/sigma) * x_t - (exp(-h) - 1) * denoised
                   = (sigma_next/sigma) * x_t + (1 - sigma_next/sigma) * x_t
                   = x_t
        """
        x = np.ones((1, 4, 8, 8), dtype=np.float32)
        state = DPMSolverState()
        out = dpmpp_2m_sde_step(
            eps_pred=self._zero_eps(x.shape),
            x_t=x,
            sigma=0.5,
            sigma_next=0.3,
            state=state,
            noise=np.zeros_like(x),
        )
        np.testing.assert_allclose(out, x, rtol=1e-5)

    def test_nonzero_eps_changes_sample(self) -> None:
        """eps != 0이면 x_next는 x_t와 달라진다."""
        x = np.ones((1, 4, 8, 8), dtype=np.float32)
        eps = 0.5 * np.ones_like(x)
        state = DPMSolverState()
        out = dpmpp_2m_sde_step(
            eps_pred=eps,
            x_t=x,
            sigma=0.5,
            sigma_next=0.3,
            state=state,
            noise=np.zeros_like(x),
        )
        # denoised = 1 - 0.5*0.5 = 0.75
        # x_next = (0.3/0.5)*1 - (0.6 - 1)*0.75 = 0.6 + 0.4*0.75 = 0.9
        np.testing.assert_allclose(out, 0.9 * np.ones_like(x), rtol=1e-5)

    def test_state_updates_after_step(self) -> None:
        x = np.random.randn(1, 4, 4, 4).astype(np.float32)
        state = DPMSolverState()
        assert state.last_denoised is None
        assert state.sigma_history == []

        dpmpp_2m_sde_step(
            eps_pred=self._zero_eps(x.shape),
            x_t=x,
            sigma=0.5,
            sigma_next=0.3,
            state=state,
            noise=np.zeros_like(x),
        )
        assert state.last_denoised is not None
        assert state.sigma_history == [0.5]

    def test_second_step_uses_multistep_formula(self) -> None:
        """2nd step (state.last_denoised != None) 이 에러 없이 실행."""
        x = np.ones((1, 4, 4, 4), dtype=np.float32)
        state = DPMSolverState()
        # First step
        x1 = dpmpp_2m_sde_step(
            eps_pred=self._zero_eps(x.shape),
            x_t=x,
            sigma=1.0,
            sigma_next=0.6,
            state=state,
            noise=np.zeros_like(x),
        )
        # Second step
        x2 = dpmpp_2m_sde_step(
            eps_pred=self._zero_eps(x1.shape),
            x_t=x1,
            sigma=0.6,
            sigma_next=0.3,
            state=state,
            noise=np.zeros_like(x1),
        )
        assert x2.shape == x.shape
        assert not np.allclose(x2, 0)


class TestDPMSolverMultistep:
    def test_init_builds_sigmas(self) -> None:
        sampler = DPMSolverMultistep(config=DiffusionConfig(), num_inference_steps=30)
        assert sampler.sigmas.shape == (31,)
        assert sampler.sigmas[-1] == 0.0

    def test_initial_state_fresh(self) -> None:
        state = DPMSolverMultistep.initial_state()
        assert state.last_denoised is None
        assert state.sigma_history == []

    def test_roundtrip_run(self) -> None:
        """전체 루프를 eps=0 + noise=0으로 돌려본다 (수렴 경로만 검증)."""
        sampler = DPMSolverMultistep(config=DiffusionConfig(), num_inference_steps=10)
        x = np.random.randn(1, 4, 8, 8).astype(np.float32)
        state = sampler.initial_state()
        for i in range(len(sampler.sigmas) - 1):
            sigma = float(sampler.sigmas[i])
            sigma_next = float(sampler.sigmas[i + 1])
            eps = np.zeros_like(x)
            x = sampler.step(eps, x, sigma, sigma_next, state, noise=np.zeros_like(x))
        # 최종 step 이후 last_denoised는 존재해야 함
        assert state.last_denoised is not None
        assert len(state.sigma_history) == len(sampler.sigmas) - 1

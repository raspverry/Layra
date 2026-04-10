"""k-diffusion 스타일 DPM++ 2M SDE 샘플러.

See-Through의 `KDiffusionStableDiffusionXLPipeline`이 사용하는 기본 샘플러.
diffusers 0.37+의 `DPMSolverMultistepScheduler`의 DPM++ 2M SDE variant와
수식적으로 동등하다.

이 모듈은 **numpy 기반 수학 부분**을 먼저 구현한다:
- beta schedule (scaled_linear)
- alphas_cumprod
- Karras sigmas
- DPM++ 2M SDE 단일 스텝 업데이트

MLX 배열 버전은 같은 수식을 `mlx.core`로 치환하면 된다 — 내부 수치 로직은
동일. numpy 버전은 unit test가 가능해 맥북 없이도 수식 검증할 수 있다.

참조:
- https://github.com/crowsonkb/k-diffusion (원본 k-diffusion)
- diffusers.schedulers.scheduling_dpmsolver_multistep
- Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based
  Generative Models" (Karras sigmas)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.stage1_layerdiff.configs import DiffusionConfig

if TYPE_CHECKING:
    pass


# ============================================================
# Beta schedule + alphas_cumprod
# ============================================================


def make_betas(config: DiffusionConfig) -> np.ndarray:
    """Compute `betas` for the configured schedule.

    Returns a `(num_train_steps,)` float64 array.
    """
    if config.beta_schedule == "linear":
        return np.linspace(
            config.beta_start,
            config.beta_end,
            config.num_train_steps,
            dtype=np.float64,
        )
    if config.beta_schedule == "scaled_linear":
        return (
            np.linspace(
                config.beta_start**0.5,
                config.beta_end**0.5,
                config.num_train_steps,
                dtype=np.float64,
            )
            ** 2
        )
    raise ValueError(f"Unknown beta_schedule: {config.beta_schedule}")


def make_alphas_cumprod(betas: np.ndarray) -> np.ndarray:
    """α̅_t = ∏_{i≤t}(1 - β_i)."""
    return np.cumprod(1.0 - betas, dtype=np.float64)


def sigma_from_alphas_cumprod(alphas_cumprod: np.ndarray) -> np.ndarray:
    """σ_t = sqrt((1 - α̅_t) / α̅_t) — k-diffusion의 training sigmas."""
    result: np.ndarray = np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
    return result


# ============================================================
# Karras sigmas
# ============================================================


def karras_sigmas(
    num_inference_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
) -> np.ndarray:
    """Karras et al. 2022, Section 5.

    σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
    i = 0..N-1, with σ_N = 0 appended.

    Returns a `(num_inference_steps + 1,)` array ending in 0.
    """
    if num_inference_steps < 2:
        raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")

    ramp = np.linspace(0.0, 1.0, num_inference_steps, dtype=np.float64)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return np.concatenate([sigmas, np.zeros(1, dtype=np.float64)])


def build_sigma_schedule(
    config: DiffusionConfig,
    num_inference_steps: int,
) -> np.ndarray:
    """Build the actual sigma schedule for sampling.

    - Compute training sigmas from betas/alphas_cumprod.
    - If `use_karras_sigmas`, re-space them via Karras ρ=7.
    - Append final 0 (final_sigmas_type="zero").
    """
    betas = make_betas(config)
    alphas_cumprod = make_alphas_cumprod(betas)
    training_sigmas = sigma_from_alphas_cumprod(alphas_cumprod)
    sigma_min = float(training_sigmas[0])
    sigma_max = float(training_sigmas[-1])

    if config.use_karras_sigmas:
        return karras_sigmas(num_inference_steps, sigma_min, sigma_max, rho=7.0)

    # Non-Karras: simple log-linear spacing
    log_sigmas = np.log(training_sigmas)
    idx = np.linspace(0, len(log_sigmas) - 1, num_inference_steps)
    sig = np.exp(np.interp(idx, np.arange(len(log_sigmas)), log_sigmas))
    return np.concatenate([sig[::-1], np.zeros(1, dtype=np.float64)])


# ============================================================
# DPM++ 2M SDE step
# ============================================================


@dataclass(slots=True)
class DPMSolverState:
    """Multi-step solver state.

    DPM++ 2M needs the *previous* step's denoised prediction.
    """

    last_denoised: np.ndarray | None = None
    sigma_history: list[float] = field(default_factory=list)


def dpmpp_2m_sde_step(
    eps_pred: np.ndarray,
    x_t: np.ndarray,
    sigma: float,
    sigma_next: float,
    state: DPMSolverState,
    *,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Single DPM++ 2M SDE step.

    The "SDE" variant injects noise proportional to `sigma_next * sqrt(1 - exp(-2h))`
    where `h = log(sigma / sigma_next)`. When `sigma_next == 0`, this becomes a
    deterministic final step.

    Args:
        eps_pred: noise prediction ε_θ(x_t, σ_t).
        x_t: current sample.
        sigma: current noise level σ_t.
        sigma_next: next noise level σ_{t-1}.
        state: persistent state across steps (last denoised sample).
        noise: optional user-provided Gaussian noise. None = np.random.randn.

    Returns:
        x_{t-1} same shape as x_t.
    """
    # x_0 estimate via ε-prediction
    denoised = x_t - sigma * eps_pred

    # Final step: no noise, direct jump to 0
    if sigma_next == 0.0:
        state.last_denoised = denoised
        state.sigma_history.append(sigma)
        return denoised.astype(x_t.dtype)

    h = np.log(sigma / sigma_next)

    if state.last_denoised is None:
        # 1st-order update (first step)
        x_next = (sigma_next / sigma) * x_t - (np.exp(-h) - 1.0) * denoised
    else:
        # 2nd-order multi-step (DPM++ 2M)
        last_sigma = state.sigma_history[-1]
        h_last = np.log(last_sigma / sigma)
        r = h_last / h
        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (
            1.0 / (2.0 * r)
        ) * state.last_denoised
        x_next = (sigma_next / sigma) * x_t - (np.exp(-h) - 1.0) * denoised_d

    # SDE churn: add noise scaled by sqrt(sigma_next^2 * (1 - exp(-2h)))
    noise_scale = sigma_next * np.sqrt(1.0 - np.exp(-2.0 * h))
    if noise is None:
        noise = np.random.randn(*x_t.shape).astype(x_t.dtype)
    x_next = x_next + noise_scale * noise

    state.last_denoised = denoised
    state.sigma_history.append(sigma)
    result: np.ndarray = x_next.astype(x_t.dtype)
    return result


# ============================================================
# High-level sampler
# ============================================================


@dataclass(slots=True)
class DPMSolverMultistep:
    """k-diffusion DPM++ 2M SDE sampler.

    Usage::

        sampler = DPMSolverMultistep(config=DiffusionConfig(), num_inference_steps=30)
        sigmas = sampler.sigmas  # (N+1,) numpy array
        state = sampler.initial_state()
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            eps = model(x, timestep_from_sigma(sigma), encoder_hidden)
            x = sampler.step(eps, x, sigma, sigma_next, state)
    """

    config: DiffusionConfig
    num_inference_steps: int
    sigmas: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sigmas = build_sigma_schedule(self.config, self.num_inference_steps)

    @staticmethod
    def initial_state() -> DPMSolverState:
        return DPMSolverState()

    def step(
        self,
        eps_pred: np.ndarray,
        x_t: np.ndarray,
        sigma: float,
        sigma_next: float,
        state: DPMSolverState,
        *,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        return dpmpp_2m_sde_step(eps_pred, x_t, sigma, sigma_next, state, noise=noise)

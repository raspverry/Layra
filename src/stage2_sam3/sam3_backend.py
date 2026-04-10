"""SAM3 text-prompt backend.

PachiPakuGen의 `scripts/extract_neck_mask.py` 분석 결과를 반영한
공용 Sam3 래퍼. `build_sam3_image_model` + `Sam3Processor.set_text_prompt`
패턴으로 자연어 프롬프트("neck", "mouth", "eye") 기반 마스크를 추출한다.

설계 포인트:
    - Lazy loading: `_ensure_loaded()`가 처음 호출될 때만 SAM3 가중치 로드
    - Processor 주입 가능: 테스트에서 `_processor`를 직접 세팅하면 로드 스킵
    - `extract_logic`은 processor 인스턴스를 받는 순수 함수로 분리해
      mock 객체로 쉽게 단위 테스트 가능
    - 후처리(dilate + Gaussian blur)는 `postprocess_mask`로 분리

PachiPakuGen 원본 패턴:

    model = build_sam3_image_model(
        bpe_path=bpe_path, device=device, eval_mode=True,
        checkpoint_path=str(checkpoint_path), load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.3)
    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=text_prompt)
    masks = state.get("masks") or state.get("pred_masks")
    combined = np.maximum.reduce(masks)
    dilated = cv2.dilate(combined, kernel, iterations=2)
    blurred = cv2.GaussianBlur(dilated, (7, 7), 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from src.common.config import Stage2Config
from src.common.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


# ============================================================
# Protocol for typing / mocking
# ============================================================


class Sam3ProcessorLike(Protocol):
    """Sam3Processor와 호환되는 최소 인터페이스.

    테스트에서 이 프로토콜을 만족하는 mock 객체를 주입할 수 있다.
    """

    def set_image(self, image: np.ndarray) -> dict[str, Any]: ...
    def reset_all_prompts(self, state: dict[str, Any]) -> None: ...
    def set_text_prompt(self, state: dict[str, Any], prompt: str) -> dict[str, Any]: ...


# ============================================================
# Pure mask postprocessing (testable without SAM3)
# ============================================================


def combine_masks(masks_array: np.ndarray) -> np.ndarray:
    """여러 마스크를 OR로 결합 (Sam3Processor가 여러 detection을 반환할 수 있음).

    Args:
        masks_array: `(N, H, W)` 또는 `(H, W)` — numpy 배열.

    Returns:
        `(H, W)` uint8 {0, 1} 배열.
    """
    import numpy as np

    arr = np.asarray(masks_array)
    if arr.ndim == 2:
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 3:
        combined: np.ndarray = np.maximum.reduce(arr)
        return (combined > 0).astype(np.uint8)
    raise ValueError(f"Unexpected mask ndim: {arr.ndim} (shape={arr.shape})")


def postprocess_mask(
    binary_mask: np.ndarray,
    *,
    dilate_iterations: int,
    blur_kernel: int,
) -> np.ndarray:
    """바이너리 마스크 → dilate + Gaussian blur → uint8 0..255.

    PachiPakuGen과 동일한 후처리:
        dilate (3x3 kernel, N iterations) → GaussianBlur ((K, K), σ=0)

    Args:
        binary_mask: `(H, W)` bool / uint8 (0 또는 1) 배열.
        dilate_iterations: dilate 반복 수. 0이면 skip.
        blur_kernel: Gaussian kernel 크기 (홀수). 0이면 skip.

    Returns:
        `(H, W)` uint8 0..255 배열. 그대로 PNG로 저장 가능.
    """
    import cv2
    import numpy as np

    mask_u8: np.ndarray = (np.asarray(binary_mask) > 0).astype(np.uint8) * 255

    if dilate_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=dilate_iterations)

    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be odd, got {blur_kernel}")
        mask_u8 = cv2.GaussianBlur(mask_u8, (blur_kernel, blur_kernel), 0)

    return mask_u8


def extract_with_processor(
    processor: Sam3ProcessorLike,
    image: np.ndarray,
    prompt: str,
) -> np.ndarray:
    """Processor 기반 mask 추출 — SAM3 API에 의존하는 최소 로직.

    이 함수는 `Sam3Processor`의 실제 인스턴스나 mock 객체 모두 받는다.

    Args:
        processor: Sam3Processor 또는 호환 mock.
        image: HxWx3 uint8 RGB numpy 배열.
        prompt: 텍스트 프롬프트 (예: "neck", "mouth", "eye,eyelid").

    Returns:
        `(H, W)` uint8 {0, 1} binary mask (후처리 전).
    """
    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    masks = state.get("masks")
    if masks is None:
        masks = state.get("pred_masks")
    if masks is None:
        raise RuntimeError(
            f"Sam3Processor returned no masks for prompt={prompt!r}. "
            f"state keys: {list(state.keys())}"
        )
    return combine_masks(masks)


# ============================================================
# High-level extractor
# ============================================================


@dataclass(slots=True)
class Sam3TextExtractor:
    """Text-prompt based SAM3 wrapper.

    This is the primary interface for neck/mouth/eye mask extraction in
    Layra. NeckExtractor and MouthExtractor are thin convenience wrappers
    around this class.
    """

    config: Stage2Config
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Sam3ProcessorLike | None = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        """Load SAM3 weights on first use.

        Skipped if `_processor` was already injected (e.g. by a test).
        """
        if self._processor is not None:
            return

        weights = self.config.sam3_weights
        if not weights.exists():
            raise FileNotFoundError(
                f"SAM3 weights not found: {weights}\n"
                "Download from "
                "https://github.com/facebookresearch/segment-anything-3"
            )

        try:
            from sam3.model.sam3_image_processor import (
                Sam3Processor,  # type: ignore[import-not-found]
            )
            from segment_anything_3 import (
                build_sam3_image_model,  # type: ignore[import-not-found]
            )
        except ImportError as e:
            raise ImportError(
                "segment_anything_3 not installed. Run:\n"
                "  pip install git+https://github.com/"
                "facebookresearch/segment-anything-3.git"
            ) from e

        logger.info(f"Loading SAM3 (device={self.config.device}) from {weights}")
        bpe_path = self.config.bpe_path
        build_kwargs: dict[str, Any] = {
            "device": self.config.device,
            "eval_mode": True,
            "checkpoint_path": str(weights),
            "load_from_HF": False,
        }
        if bpe_path is not None:
            build_kwargs["bpe_path"] = str(bpe_path)
        self._model = build_sam3_image_model(**build_kwargs)
        self._processor = Sam3Processor(
            self._model,
            confidence_threshold=self.config.confidence_threshold,
        )

    def inject_processor(self, processor: Sam3ProcessorLike) -> None:
        """Override the processor (used by tests and offline workflows)."""
        self._processor = processor

    def extract(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Return an 8-bit (0..255) mask for a single text prompt.

        Args:
            image: HxWx3 uint8 numpy array.
            prompt: SAM3 text prompt, e.g. "neck", "mouth", "eye,eyelid".

        Returns:
            HxW uint8 mask with post-processing applied (dilate + blur).
        """
        self._ensure_loaded()
        assert self._processor is not None
        binary = extract_with_processor(self._processor, image, prompt)
        return postprocess_mask(
            binary,
            dilate_iterations=self.config.postprocess_dilate_iterations,
            blur_kernel=self.config.postprocess_blur_kernel,
        )

    def extract_named(
        self,
        image: np.ndarray,
        part_name: str,
    ) -> np.ndarray:
        """Resolve the prompt via config.text_prompts and delegate to extract.

        Args:
            image: HxWx3 uint8 numpy array.
            part_name: Logical part name (e.g. "neck"), resolved via
                ``config.text_prompts``. Falls back to using part_name
                directly as the prompt if not present in the map.
        """
        prompt = self.config.text_prompts.get(part_name, part_name)
        return self.extract(image, prompt)

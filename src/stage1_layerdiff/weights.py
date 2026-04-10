"""PyTorch safetensors → MLX weights 변환.

See-Through의 가중치는 HuggingFace Hub에서 safetensors로 배포된다::

    layerdifforg/seethroughv0.0.2_layerdiff3d/
    ├── unet/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors
    ├── vae/
    │   └── diffusion_pytorch_model.safetensors
    ├── text_encoder/
    │   └── model.safetensors
    ├── text_encoder_2/
    │   └── model.safetensors
    └── scheduler/
        └── scheduler_config.json

이 모듈은:
    1. safetensors 파일을 PyTorch state_dict로 로드
    2. key name을 Layra MLX 모듈 계층에 맞게 변환
    3. `mlx.core.array(tensor.numpy())`로 변환
    4. `mlx.core.savez`로 저장

사용 예::

    python -m src.stage1_layerdiff.weights \\
        --src models/hf_cache/seethroughv0.0.2_layerdiff3d/unet \\
        --dst models/layerdiff/unet.npz \\
        --component unet

key mapping은 `KEY_MAPS` dict에 있다. diffusers의 네이밍과 MLX SD 예제의
네이밍이 거의 1:1로 겹치지만, LayerDiffuse의 frame/cross-frame 블록은
별도 매핑이 필요하다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.common.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


Component = Literal["unet", "vae", "text_encoder", "text_encoder_2", "marigold"]


# ============================================================
# Key mapping rules
# ============================================================
#
# Each entry: (diffusers_prefix, mlx_prefix). `None` means "drop this key".
# Order matters — first match wins. We strip the diffusers prefix and
# prepend the mlx prefix. Inside the stripped key, a few substring
# substitutions are applied (see REWRITE_RULES).


UNET_KEY_MAP: tuple[tuple[str, str | None], ...] = (
    # top-level conv_in / conv_out
    ("conv_in.", "conv_in."),
    ("conv_out.", "conv_out."),
    ("conv_norm_out.", "conv_norm_out."),
    # Time embedding
    ("time_embedding.linear_1.", "time_embedding.linear_1."),
    ("time_embedding.linear_2.", "time_embedding.linear_2."),
    # SDXL addition embedding
    ("add_embedding.linear_1.", "add_embedding.linear_1."),
    ("add_embedding.linear_2.", "add_embedding.linear_2."),
    # Down blocks
    ("down_blocks.", "down_blocks."),
    # Mid block
    ("mid_block.resnets.0.", "mid_blocks.0."),
    ("mid_block.attentions.0.", "mid_blocks.1."),
    ("mid_block.resnets.1.", "mid_blocks.2."),
    # Up blocks
    ("up_blocks.", "up_blocks."),
)


VAE_KEY_MAP: tuple[tuple[str, str | None], ...] = (
    ("encoder.", "encoder."),
    ("decoder.", "decoder."),
    ("quant_conv.", "quant_proj."),
    ("post_quant_conv.", "post_quant_proj."),
    # LayerDiffuse alpha head
    ("alpha_head.", "alpha_head."),
)


TEXT_ENCODER_KEY_MAP: tuple[tuple[str, str | None], ...] = (
    ("text_model.embeddings.", "embeddings."),
    ("text_model.encoder.", "encoder."),
    ("text_model.final_layer_norm.", "final_layer_norm."),
    # SDXL text_encoder_2 projection head
    ("text_projection.", "text_projection."),
)


MARIGOLD_KEY_MAP: tuple[tuple[str, str | None], ...] = (
    # Marigold는 표준 SD 1.5 UNet 구조를 따름
    ("conv_in.", "conv_in."),
    ("conv_out.", "conv_out."),
    ("time_embedding.", "time_embedding."),
    ("down_blocks.", "down_blocks."),
    ("mid_block.", "mid_blocks."),
    ("up_blocks.", "up_blocks."),
)


KEY_MAPS: dict[Component, tuple[tuple[str, str | None], ...]] = {
    "unet": UNET_KEY_MAP,
    "vae": VAE_KEY_MAP,
    "text_encoder": TEXT_ENCODER_KEY_MAP,
    "text_encoder_2": TEXT_ENCODER_KEY_MAP,
    "marigold": MARIGOLD_KEY_MAP,
}


# Substring rewrites applied after prefix mapping.
# e.g. PyTorch uses "to_k" / "to_q" / "to_v", MLX SD uses the same so this
# is a no-op for most cases but we keep the hook for LayerDiffuse specifics.
REWRITE_RULES: tuple[tuple[str, str], ...] = (
    # PyTorch LayerNorm stores weight/bias as gamma/beta under some names
    (".layer_norm.", ".norm."),
    # diffusers "attentions.0" inside transformer3d → "attentions.0"
    # (no-op today; placeholder for future transformer3d remaps)
)


def rewrite_key(
    original: str,
    mapping: tuple[tuple[str, str | None], ...],
) -> str | None:
    """Apply prefix mapping + substring rewrites. Returns None to drop."""
    for prefix, new_prefix in mapping:
        if original.startswith(prefix):
            if new_prefix is None:
                return None
            rewritten = new_prefix + original[len(prefix) :]
            for src, dst in REWRITE_RULES:
                rewritten = rewritten.replace(src, dst)
            return rewritten
    return None  # unknown prefix — drop with warning


# ============================================================
# Conversion
# ============================================================


def _load_torch_state_dict(src: Path) -> Mapping[str, object]:
    """Load .safetensors or .pth state_dict.

    Prefers safetensors (memory-mapped, no pickle).
    """
    if src.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "safetensors not installed. Run: pip install safetensors"
            ) from e
        return load_file(str(src))

    if src.suffix in {".pth", ".pt", ".bin"}:
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch not installed. Run: pip install torch") from e
        loaded: Mapping[str, object] = torch.load(
            str(src), map_location="cpu", weights_only=True
        )
        return loaded

    raise ValueError(f"Unsupported weight file: {src}")


def convert_state_dict(
    state_dict: Mapping[str, object],
    component: Component,
    *,
    dropped_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Apply key mapping to a PyTorch state_dict.

    Returns a new dict with MLX-compatible keys. Values are still PyTorch
    tensors — call `_to_mlx_array` separately if converting to MLX.
    """
    mapping = KEY_MAPS[component]
    out: dict[str, object] = {}
    n_mapped = 0
    n_dropped = 0

    for original_key, tensor in state_dict.items():
        new_key = rewrite_key(original_key, mapping)
        if new_key is None:
            n_dropped += 1
            if dropped_callback is not None:
                dropped_callback(original_key)
            continue
        out[new_key] = tensor
        n_mapped += 1

    logger.info(f"Converted {n_mapped} keys, dropped {n_dropped} keys for {component}")
    return out


def convert_to_mlx(
    src: Path,
    dst: Path,
    component: Component,
) -> None:
    """End-to-end conversion: safetensors → MLX npz.

    This is the entry point called from CLI.

    Args:
        src: Path to .safetensors or diffusers component directory
            (in which case we look for `diffusion_pytorch_model.safetensors`
            or `model.safetensors`).
        dst: Output .npz path.
        component: Which key map to apply.
    """
    src_file = _resolve_src(src)
    logger.info(f"Loading {src_file}")
    state_dict = _load_torch_state_dict(src_file)

    dropped: list[str] = []
    converted = convert_state_dict(
        state_dict, component, dropped_callback=dropped.append
    )

    if dropped:
        logger.warning(
            f"Dropped {len(dropped)} unrecognised keys. First 5: {dropped[:5]}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    _save_as_mlx(converted, dst)
    logger.info(f"Wrote {dst}")


def _resolve_src(src: Path) -> Path:
    """Accept a directory or a direct file path."""
    if src.is_file():
        return src
    if src.is_dir():
        for candidate in (
            src / "diffusion_pytorch_model.safetensors",
            src / "model.safetensors",
            src / "pytorch_model.bin",
        ):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No weights found at {src}")


def _save_as_mlx(state: dict[str, object], dst: Path) -> None:
    """Convert PyTorch tensors to MLX arrays and save as .npz.

    On Linux (no MLX), fall back to numpy .npz. Either way the output
    is loadable by `mlx.core.load` on Apple Silicon.
    """
    import numpy as np

    # Convert each tensor to numpy first. Handles torch.bfloat16 → float32.
    numpy_state: dict[str, np.ndarray] = {}
    for key, tensor in state.items():
        numpy_state[key] = _tensor_to_numpy(tensor)

    try:
        import mlx.core as mx

        mlx_state = {k: mx.array(v) for k, v in numpy_state.items()}
        mx.savez(str(dst), **mlx_state)
    except ImportError:
        logger.warning(
            "mlx not available — saving via numpy.savez_compressed "
            "(still loadable by mlx.core.load)"
        )
        np.savez_compressed(str(dst), **numpy_state)  # type: ignore[arg-type]


def _tensor_to_numpy(tensor: object) -> np.ndarray:
    """Convert a PyTorch tensor (possibly bf16) to numpy float32/float16."""
    import numpy as np

    t = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor  # type: ignore[union-attr]

    # bf16이나 fp16이면 float32로 업캐스트 (numpy는 bf16 없음).
    if hasattr(t, "dtype"):
        import torch

        if t.dtype == torch.bfloat16:  # type: ignore[union-attr]
            t = t.to(torch.float32)  # type: ignore[union-attr]

    return np.asarray(t.numpy() if hasattr(t, "numpy") else t)


# ============================================================
# CLI
# ============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert LayerDiffuse/Marigold PyTorch weights to MLX"
    )
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=["unet", "vae", "text_encoder", "text_encoder_2", "marigold"],
        required=True,
    )
    args = parser.parse_args(argv)

    try:
        convert_to_mlx(args.src, args.dst, args.component)
    except (FileNotFoundError, ValueError, ImportError) as e:
        logger.error(f"{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

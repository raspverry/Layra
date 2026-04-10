"""PyTorch 체크포인트 → MLX weights 변환.

See-Through 원본은 PyTorch .safetensors로 배포된다.
이 모듈은 해당 가중치를 MLX 호환 형식으로 변환한다.

사용 예:
    python -m src.stage1_layerdiff.weights \\
        --src /path/to/layerdiff.safetensors \\
        --dst models/layerdiff/unet.npz
"""

from __future__ import annotations

from pathlib import Path

from src.common.logging import get_logger

logger = get_logger(__name__)


def convert_pytorch_to_mlx(
    src: Path,
    dst: Path,
    key_map: dict[str, str] | None = None,
) -> None:
    """PyTorch state_dict를 MLX npz로 변환.

    Args:
        src: .pth / .safetensors 경로.
        dst: .npz 저장 경로.
        key_map: 선택적 키 이름 변환 매핑 (e.g. PyTorch → MLX 표준 이름).

    Raises:
        FileNotFoundError: src가 없을 때.
        NotImplementedError: 구현 대기.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source weights not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # TODO(stage1): torch.load 또는 safetensors.torch.load_file로 읽고
    # TODO(stage1): mx.array(tensor.numpy())로 변환 후 mx.savez
    # TODO(stage1): key_map 적용 (BN → GroupNorm 등 필요시)
    raise NotImplementedError(
        "PyTorch → MLX weight conversion pending — see wiki/see-through-porting.md"
    )

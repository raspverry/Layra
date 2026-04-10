"""LayerDiffuse를 위한 MLX 레벨 프리미티브.

MLX SD 예제(ml-explore/mlx-examples/stable_diffusion)의 `unet.py`,
`vae.py`를 베이스로 LayerDiffuse의 3D(frame-conditioned) 구조에
맞춰 확장한 모듈들을 모아둔다.

Linux 환경에서는 mlx가 설치되지 않으므로 각 모듈은 mlx 관련
import를 `TYPE_CHECKING` 가드 또는 함수 내부 지연 import로
처리한다. 실제 연산은 맥북에서 채운다 (`NotImplementedError`).
"""

from src.stage1_layerdiff.mlx_ops.attention import (
    CrossFrameTransformerBlock,
    TransformerBlock,
)
from src.stage1_layerdiff.mlx_ops.blocks import (
    Transformer3DModel,
    UNetBlock2D,
)
from src.stage1_layerdiff.mlx_ops.embeddings import (
    SDXLAdditionEmbedding,
    TimestepEmbedding,
    timestep_embedding,
)
from src.stage1_layerdiff.mlx_ops.norms import AdaLayerNormSingle
from src.stage1_layerdiff.mlx_ops.resnet import ResnetBlock2D

__all__ = [
    "AdaLayerNormSingle",
    "CrossFrameTransformerBlock",
    "ResnetBlock2D",
    "SDXLAdditionEmbedding",
    "TimestepEmbedding",
    "Transformer3DModel",
    "TransformerBlock",
    "UNetBlock2D",
    "timestep_embedding",
]

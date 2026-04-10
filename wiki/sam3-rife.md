# SAM3 + RIFE 통합

## 상태: 🔴 미시작 (Stage 1 완료 후 시작)

---

## 참조: PachiPakuGen

레포: https://github.com/kazuya-bros/PachiPakuGen  
라이선스: MIT  
스택: Tauri (Rust) + Vite/TypeScript + **Python 5.7% (1 script)**  
주의: 원본은 Windows + DirectML 전용, SpriTalk 전용 출력

우리는 **로직만 참조**하고 독자 구현.

### PachiPakuGen Python 코드 실제 분석 (2026-04-10)

`scripts/extract_neck_mask.py` (유일한 Python 파일) 분석 결과:

```python
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    bpe_path=bpe_path,
    device=device,
    eval_mode=True,
    checkpoint_path=str(checkpoint_path),
    load_from_HF=False,
)
processor = Sam3Processor(model, confidence_threshold=0.3)
```

**핵심 차이점 — SAM3은 text prompt로 사용**:

```python
state = processor.set_image(image)
processor.reset_all_prompts(state)
state = processor.set_text_prompt(state=state, prompt="eye,mouth")
masks = state["masks"]  # or state["pred_masks"]
```

- 점 좌표나 박스가 아닌 **자연어 텍스트** (`"eye,mouth"`)
- confidence_threshold 0.3
- 마스크 후처리: `dilate` 2회 + Gaussian blur 7×7 → 경계선 매끄럽게

**우리 구현 상태 (2026-04-10, Phase 1d 완료)**:

`src/stage2_sam3/sam3_backend.py`에 `Sam3TextExtractor`를 구현해
PachiPakuGen과 동일한 패턴을 사용하도록 완료:

```python
from src.stage2_sam3 import Sam3TextExtractor, NeckExtractor, MouthExtractor

# 저수준: 임의 prompt
extractor = Sam3TextExtractor(config=stage2_cfg)
mask = extractor.extract(image_rgb, prompt="neck")  # HxW uint8 0..255

# 고수준: 파트 이름 → config.text_prompts 매핑
mask = extractor.extract_named(image_rgb, "neck")

# Convenience wrappers:
neck_mask = NeckExtractor(config=cfg).extract(image)
closed, opened = MouthExtractor(config=cfg).extract_pair(closed_img, open_img)
```

주요 설계:
- `build_sam3_image_model` + `Sam3Processor` lazy load
- `confidence_threshold=0.3` (PachiPakuGen과 동일 기본값)
- `combine_masks`: 여러 detection을 `np.maximum.reduce`로 결합
- `postprocess_mask`: `cv2.dilate` 2회 + `cv2.GaussianBlur` 7×7
  (PachiPakuGen과 동일)
- `config.text_prompts`로 파트 이름 → 프롬프트 매핑 (기본값 `"neck"`, `"mouth"`, `"eye"`)
- **`inject_processor`/`inject_backend`** — mock 객체 주입으로
  Linux 환경에서도 SAM3 없이 단위 테스트 가능 (21개 테스트 통과)
- Legacy point-based API는 `NeckExtractor.extract_from_point`로만 남김

나머지(PSD 파싱, RIFE)는 Rust 크레이트로 구현되어 있어
Python 참조 불가 — 우리 구현은 psd-tools + onnxruntime 기반.

---

## SAM3 통합

### 설치

```bash
# SAM3 (PyPI 미등록 — git+로 설치)
pip install git+https://github.com/facebookresearch/segment-anything-3.git

# 모델 weights (~3.2GB)
# https://github.com/facebookresearch/segment-anything-3
wget https://dl.fbaipublicfiles.com/segment_anything_3/sam3_vit_h.pth
```

### 목(neck) 추출 파이프라인

문제: See-Through의 neck 레이어는 outpainting으로 생성 → 경계 부자연스러움  
해결: 원본 이미지에서 SAM3으로 목 영역 직접 추출

```python
from segment_anything_3 import SAM3, SamPredictor
import numpy as np
from PIL import Image

class NeckExtractor:
    def __init__(self, model_path: str):
        self.model = SAM3()
        self.model.load_weights(model_path)
        self.predictor = SamPredictor(self.model)
    
    def extract_neck(
        self, 
        original_image: np.ndarray,
        point_hint: tuple[int, int] | None = None
    ) -> np.ndarray:
        """
        원본 이미지에서 목 영역 마스크 추출
        
        Args:
            original_image: See-Through 입력과 동일한 원본 이미지
            point_hint: 목 위치 힌트 (x, y). None이면 자동 감지
        
        Returns:
            neck_mask: HxW binary mask
        """
        self.predictor.set_image(original_image)
        
        if point_hint is None:
            # 얼굴 bbox 하단 중앙을 목으로 추정
            point_hint = self._estimate_neck_point(original_image)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([point_hint]),
            point_labels=np.array([1]),  # 1 = foreground
            multimask_output=True
        )
        
        # 가장 높은 score의 마스크 선택
        best_mask = masks[np.argmax(scores)]
        return best_mask
    
    def _estimate_neck_point(self, image: np.ndarray) -> tuple[int, int]:
        """얼굴 검출 기반 목 위치 자동 추정"""
        # TODO: 얼굴 bbox 검출 후 하단 중앙 반환
        h, w = image.shape[:2]
        return (w // 2, int(h * 0.45))  # 임시: 이미지 중앙 45% 지점
```

### 입(mouth) 추출 파이프라인

```python
class MouthExtractor:
    def __init__(self, model_path: str):
        self.predictor = SamPredictor(SAM3())
        self.predictor.model.load_weights(model_path)
    
    def extract_mouth_pair(
        self,
        closed_image: np.ndarray,
        open_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        닫힌 입, 열린 입 마스크 쌍 추출
        
        Returns:
            (closed_mask, open_mask)
        """
        closed_mask = self._extract_single(closed_image)
        open_mask = self._extract_single(open_image)
        return closed_mask, open_mask
    
    def _extract_single(self, image: np.ndarray) -> np.ndarray:
        self.predictor.set_image(image)
        mouth_point = self._estimate_mouth_point(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([mouth_point]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        return masks[np.argmax(scores)]
```

---

## RIFE 프레임 보간

### 설치

```bash
# ONNX Runtime (Mac에서 CPU/CoreML)
pip install onnxruntime

# RIFE ONNX 모델 변환 또는 다운로드
# PachiPakuGen 참조: ONNX Runtime 2.0 사용
```

### 프레임 보간 파이프라인

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

class RIFEInterpolator:
    def __init__(self, model_path: str):
        # Mac에서 CoreML provider 시도, 실패시 CPU
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers
        )
    
    def interpolate(
        self,
        frame_start: np.ndarray,
        frame_end: np.ndarray,
        n_frames: int = 8
    ) -> list[np.ndarray]:
        """
        두 프레임 사이 중간 프레임 생성
        
        Args:
            frame_start: 시작 프레임 (닫힘)
            frame_end: 끝 프레임 (열림)
            n_frames: 생성할 중간 프레임 수
        
        Returns:
            frames: [frame_start, ...중간..., frame_end] 리스트
        """
        frames = [frame_start]
        
        for i in range(1, n_frames - 1):
            t = i / (n_frames - 1)
            mid_frame = self._interpolate_single(frame_start, frame_end, t)
            frames.append(mid_frame)
        
        frames.append(frame_end)
        return frames
    
    def _interpolate_single(
        self, 
        img0: np.ndarray, 
        img1: np.ndarray, 
        timestep: float
    ) -> np.ndarray:
        # RIFE ONNX 입력 형식으로 변환
        inp = {
            "img0": self._preprocess(img0),
            "img1": self._preprocess(img1),
            "timestep": np.array([timestep], dtype=np.float32)
        }
        output = self.session.run(None, inp)
        return self._postprocess(output[0])
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # HxWxC → 1xCxHxW, 0-255 → 0-1
        img = img.astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)[np.newaxis]
    
    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        # 1xCxHxW → HxWxC, 0-1 → 0-255
        img = output[0].transpose(1, 2, 0)
        return (img * 255).clip(0, 255).astype(np.uint8)
```

### 눈 깜빡임 생성

```python
def generate_eye_blink_frames(
    eye_open: np.ndarray,
    eye_closed: np.ndarray,
    n_frames: int = 8,
    output_dir: str = "output/eye/"
) -> None:
    interpolator = RIFEInterpolator("models/rife.onnx")
    
    # 열림 → 닫힘
    frames = interpolator.interpolate(eye_open, eye_closed, n_frames)
    
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(f"{output_dir}/frame_{i+1:03d}.png")
```

### 입 모양 5모음 생성

```python
VOWELS = ["a", "i", "u", "e", "o"]  # あいうえお

def generate_mouth_frames(
    mouth_closed: np.ndarray,
    mouth_shapes: dict[str, np.ndarray],  # {"a": img, "i": img, ...}
    n_frames: int = 8,
    output_dir: str = "output/"
) -> None:
    interpolator = RIFEInterpolator("models/rife.onnx")
    
    for vowel in VOWELS:
        if vowel not in mouth_shapes:
            continue
        
        frames = interpolator.interpolate(
            mouth_closed, 
            mouth_shapes[vowel], 
            n_frames
        )
        
        vowel_dir = f"{output_dir}/mouth_{vowel}/"
        os.makedirs(vowel_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f"{vowel_dir}/frame_{i+1:03d}.png")
```

---

## PSD 레이어 파싱

Stage 1 출력 PSD를 Stage 2로 넘기기 위한 파싱:

```python
from psd_tools import PSDImage

def parse_see_through_psd(psd_path: str) -> dict[str, np.ndarray]:
    """
    See-Through 출력 PSD를 레이어별 numpy array로 파싱
    
    Returns:
        {"hair_front": ndarray, "face": ndarray, ...}
    """
    psd = PSDImage.open(psd_path)
    layers = {}
    
    for layer in psd.descendants():
        if layer.is_group():
            continue
        name = layer.name.lower().strip()
        composite = layer.composite()
        if composite is not None:
            layers[name] = np.array(composite)
    
    return layers

def extract_body_parts(layers: dict) -> tuple:
    """레이어 딕셔너리에서 body/hair 파트 추출"""
    body = layers.get("body") or layers.get("torso")
    hair_front = layers.get("hair_front") or layers.get("hair")
    hair_back = layers.get("hair_back")
    
    return body, hair_front, hair_back
```

---

## 엣지 케이스

| 케이스 | 문제 | 대응 |
|--------|------|------|
| hair_front/back 미분리 | 단일 hair 레이어만 존재 | depth 기반 자동 분리 시도, 실패시 단일 레이어로 출력 |
| 눈 레이어 합쳐짐 | eye_L, eye_R 미분리 | 수평 중심선으로 분할 |
| 입 레이어 미검출 | mouth 레이어 없음 | SAM3으로 원본에서 직접 추출 |
| 비정형 캐릭터 | 23개 표준 파트 매핑 불가 | 최선 매핑 후 경고 출력 |

---

## 진행 상황 로그

| 날짜 | 작업 | 상태 | 메모 |
|------|------|------|------|
| (미시작) | | | |

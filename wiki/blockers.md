# 블로커 / 미해결 이슈

## 🔴 크리티컬

### BLOCKER-001: See-Through weights 라이선스 미확인

**문제**: See-Through 코드는 Apache 2.0이지만, 학습된 weights의 라이선스가 명확하지 않음  
**영향**: 서비스화 시 상업적 사용 가능 여부 불명확  
**확인 방법**:
- LayerDiffuse weights 라이선스: https://github.com/lllyasviel/LayerDiffuse 확인
- Marigold weights 라이선스: https://github.com/prs-eth/marigold 확인
- See-Through 저자에게 직접 문의 (필요시)

**현재 상태**: 미확인  
**해결 기한**: Phase 4 시작 전

---

## 🟡 중간 우선순위

### ~~BLOCKER-002: MLX 포팅 가능성 미검증~~ ✅ 해결 (2026-04-10)

**해결 방법**: WebFetch로 원본 레포 정독. 자세한 내용은
`wiki/see-through-porting.md`의 "원본 코드 분석 결과" 섹션 참조.

**결론**: 포팅 가능성 매우 높음. 주요 근거:
- `xformers`, `triton`, `flash_attn`, `torch.compile`,
  `scaled_dot_product_attention`, `memory_efficient_attention` 전부 0건
- `KDiffusionStableDiffusionXLPipeline`은 diffusers 표준
  `StableDiffusionXLImg2ImgPipeline` 서브클래스
- `transformer3d.py`는 diffusers 표준 `Attention`, `BasicTransformerBlock`,
  `TemporalBasicTransformerBlock`, `AdaLayerNormSingle`만 사용
- 커스텀 CUDA 커널/Metal shader 재작성 불필요
- 유일한 CUDA 의존은 `.to(device='cuda')` 패턴뿐 → MLX는 unified memory라 제거만 하면 됨

**잔여 작업**: 실제 포팅은 맥북 도착 후. 12단계 포팅 맵은
`wiki/see-through-porting.md`의 "포팅 맵" 섹션에 정리.

---

### BLOCKER-003: RIFE ONNX 모델 Mac 호환성 미확인

**문제**: PachiPakuGen은 DirectML (Windows GPU) 사용. Mac CoreML/CPU에서 동작하는지 미확인  
**영향**: Stage 3 구현 방식 결정  
**확인 방법**:
```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)  # CoreMLExecutionProvider 있는지 확인
```

**현재 상태**: 미확인  
**대안**: CoreML 없으면 CPU fallback 사용 (느려지지만 동작은 함)

---

### BLOCKER-004: See-Through 처리 시간 기준 미확보

**문제**: M5 Pro MLX에서 실제 처리 시간 미측정. 성능 목표 달성 가능한지 불명확  
**확인 방법**: RunPod A100에서 원본 실행 → 기준 시간 확보 → MLX 목표 설정  
**현재 상태**: 미확인  
**해결 기한**: 맥북 도착 전 (RunPod에서 먼저 측정)

---

## 🟢 낮은 우선순위

### BLOCKER-005: Live2D SDK 상업 라이선스 비용

**문제**: 서비스화 시 Live2D 리깅 자동화 포함할 경우 SDK 상업 라이선스 필요  
**현재 판단**: MVP에서 리깅 제외했으므로 당장 블로커 아님  
**확인 기한**: Phase 4 시작 시

---

### BLOCKER-006: See-Through 엣지 케이스 처리

**문제**: 비표준 캐릭터(동물 귀, 특이한 헤어 스타일 등)에서 레이어 분해 품질 미확인  
**영향**: 서비스 지원 캐릭터 타입 제한 필요할 수 있음  
**확인 방법**: 다양한 스타일 이미지로 테스트  
**현재 상태**: 개인 아바타 단계에서 자연스럽게 확인됨

---

## 해결된 블로커

| 번호 | 내용 | 해결 방법 | 해결 날짜 |
|------|------|----------|----------|
| BLOCKER-002 | MLX 포팅 가능성 미검증 | WebFetch로 원본 정독, CUDA/xformers/triton/flash_attn 의존성 0건 확인, 12단계 포팅 맵 작성 | 2026-04-10 |
| MINOR-001 | Wiki SAM3 설치법 3곳 불일치 (PyPI 이름 vs git+) | 전부 git+로 통일 | 2026-04-10 |
| MINOR-002 | setup.sh SAM3 fallback echo가 항상 실행되던 쉘 버그 | if/then/fi 블록으로 재작성 | 2026-04-10 |
| MINOR-003 | requirements.txt의 onnxruntime>=2.0.0 존재하지 않는 버전 | >=1.17.0으로 수정 | 2026-04-10 |
| MINOR-004 | day1-checklist.md 자리표시자 URL | raspverry/layra로 교체 | 2026-04-10 |
| MINOR-005 | 루트 LICENSE, .gitignore, README.md 누락 | Apache 2.0 + 표준 .gitignore + README 추가 | 2026-04-10 |
| MINOR-006 | experiments/ 디렉토리 구조 wiki vs 실제 불일치 | results/ 하위로 통일 + 시드 파일 생성 | 2026-04-10 |

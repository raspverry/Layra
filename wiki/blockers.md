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

### BLOCKER-002: MLX 포팅 가능성 미검증

**문제**: LayerDiffuse가 표준 SDXL UNet인지, 커스텀 CUDA 커널 사용하는지 미확인  
**영향**: MLX 포팅 난이도 및 기간에 직접 영향  
**확인 방법**:
```bash
# See-Through 클론 후 확인
git clone https://github.com/shitagaki-lab/see-through
grep -r "cuda" src/ --include="*.py" | grep -v ".pyc"
grep -r "xformers" src/ --include="*.py"
grep -r "triton" src/ --include="*.py"
```

**현재 상태**: 미확인  
**해결 기한**: 맥북 도착 후 Day 1

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

(해결되면 여기로 이동)

| 번호 | 내용 | 해결 방법 | 해결 날짜 |
|------|------|----------|----------|
| | | | |

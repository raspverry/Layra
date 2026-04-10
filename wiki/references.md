# 참조 자료

## 핵심 레포지토리

### See-Through (Stage 1 베이스)
- **URL**: https://github.com/shitagaki-lab/see-through
- **라이선스**: Apache 2.0
- **논문**: arXiv:2602.03749 (SIGGRAPH 2026)
- **핵심 파일**:
  - `inference/scripts/inference_psd.py` — 메인 추론 스크립트
  - `src/layerdiff/` — LayerDiff 모델 정의
  - `src/marigold/` — Marigold depth 모델

### PachiPakuGen (Stage 2+3 로직 참조)
- **URL**: https://github.com/kazuya-bros/PachiPakuGen
- **라이선스**: MIT
- **스택**: Tauri + Vite/TypeScript + Python scripts
- **핵심**: See-Through PSD → SAM3 → RIFE → SpriTalk 소재
- **주의**: SpriTalk 전용 출력, Windows DirectML 전용 → 로직만 참조

### MLX
- **URL**: https://github.com/ml-explore/mlx
- **라이선스**: MIT
- **중요 서브 레포**: https://github.com/ml-explore/mlx-examples (SD 예제 포함)

### LayerDiffuse (See-Through 내부 모델)
- **URL**: https://github.com/lllyasviel/LayerDiffuse
- **라이선스**: Apache 2.0
- **역할**: Transparent image layer diffusion

### Marigold (Depth Estimation)
- **URL**: https://github.com/prs-eth/marigold
- **라이선스**: Apache 2.0
- **역할**: Diffusion 기반 단안 깊이 추정

### Segment Anything (SAM)
- **URL**: https://github.com/facebookresearch/segment-anything
- **라이선스**: Apache 2.0
- **역할**: 데이터 파이프라인 (See-Through 학습 데이터 생성 시 사용)

---

## 참조 레포지토리 (개발 도구)

### minimind
- **URL**: https://github.com/jingyaogong/minimind
- **역할**: PyTorch 네이티브 학습 루프 구조 참조
- **메모**: LLM 프로젝트지만 학습 코드 설계 방식 참고

### autoresearch-skill
- **URL**: https://github.com/olelehmann100kMRR/autoresearch-skill
- **역할**: Claude Code 자율 최적화 루프 skill
- **사용법**: `wiki/autoresearch.md` 참조

### ComfyUI-See-through
- **URL**: https://github.com/jtydhr88/ComfyUI-See-through
- **역할**: See-Through의 ComfyUI 통합 — 파이프라인 구조 참조용

---

## 논문

### See-Through
```
@article{lin2026seethrough,
  title={See-through: Single-image Layer Decomposition for Anime Characters},
  author={Lin, Jian and Li, Chengze and Qin, Haoyun and Chan, Kwun Wang 
          and Jin, Yanghua and Liu, Hanyuan and Choy, Stephen Chun Wang 
          and Liu, Xueting},
  journal={arXiv preprint arXiv:2602.03749},
  year={2026}
}
```
- arXiv: https://arxiv.org/abs/2602.03749
- SIGGRAPH 2026 Conditionally Accepted

### RIFE
- "Real-Time Intermediate Flow Estimation for Video Frame Interpolation"
- https://github.com/hzwer/RIFE

### Marigold
- "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation"
- CVPR 2024

---

## 유용한 링크

- [MLX 공식 문서](https://ml-explore.github.io/mlx/build/html/index.html)
- [Apple MLX 연구 블로그](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [ONNX Runtime CoreML Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Live2D Cubism 공식](https://www.live2d.com/en/cubism/)
- [Live2D SDK 라이선스](https://www.live2d.com/en/sdk/license/)
- [RunPod 공식](https://runpod.io)
- [Karpathy LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

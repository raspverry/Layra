# Stage 2: SAM3 통합

## 현재 구조 (스캐폴딩)

```
stage2_sam3/
├── __init__.py         # 공개 API: Stage2Pipeline, run_stage2, Extractors
├── pipeline.py         # PSD → SAM3 보정 → PNG 저장 오케스트레이션
├── neck_extractor.py   # SAM3 wrapper (lazy load)
├── mouth_extractor.py  # SAM3 wrapper (lazy load)
└── psd_parser.py       # LayerSet → body/hair/mouth/neck 파트 추출
```

## 상태

- `psd_parser`: Stage 1 PSD가 있으면 바로 동작 (PyTorch/MLX 불필요).
- `NeckExtractor` / `MouthExtractor`: SAM3 weights + `segment_anything_3`
  설치 필요. import를 함수 안에서 하여 테스트 환경에서 lazy 로드.
- `Stage2Pipeline`: `config` + `NeckExtractor` + (선택) `MouthExtractor` 조합.

## 시작 전 읽을 것

1. `wiki/sam3-rife.md`
2. PachiPakuGen scripts/: https://github.com/kazuya-bros/PachiPakuGen/tree/main/scripts

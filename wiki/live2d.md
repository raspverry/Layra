# Live2D 정보

## 라이선스 요약

| 용도 | 조건 |
|------|------|
| 개인 비상업 | **무료** |
| 상업용 | 매출 규모별 요금제, Live2D 사와 계약 필요 |

**현재 프로젝트**: 개인 비상업 → 무료 플랜 사용

공식 라이선스 페이지: https://www.live2d.com/en/sdk/license/

---

## Cubism 개인 플랜 설치

1. https://www.live2d.com/en/cubism/download/editor/ 에서 다운로드
2. 개인 무료 라이선스로 등록
3. 소재(PNG) 임포트 후 수동 리깅

---

## 우리 파이프라인 출력 → Cubism 임포트

### 필요한 소재 구조

```
캐릭터명/
├── body.png          # 몸통 (목 포함)
├── hair.png          # 앞머리
├── hair_back.png     # 뒷머리
├── eye/
│   ├── frame_001.png ~ frame_008.png  # 눈 깜빡임
└── mouth_a/ ~ mouth_o/
    ├── frame_001.png ~ frame_008.png  # 입 모양
```

### Cubism 임포트 순서

1. 새 프로젝트 생성
2. 소재 PNG를 레이어로 임포트 (drawing order 맞게)
3. 각 레이어에 mesh 생성 (자동 또는 수동)
4. 파라미터 설정 (EyeOpen, MouthOpenY 등)
5. 키프레임 바인딩
6. 프레임 시퀀스 연결
7. Physics 설정 (머리카락)

---

## 서비스화 시 검토 사항

서비스에서 `.moc3` 파일 직접 생성하려면:
- Cubism SDK (상업용) 라이선스 필요
- Live2D 사와 별도 계약
- 비용: 매출 규모에 따라 다름

**현재 판단**: MVP에서 `.moc3` 자동 생성 제외.  
소재(PNG) 생성까지만 제공, 리깅은 유저가 Cubism으로 직접.

---

## 대안 검토 (서비스화 시)

Live2D SDK 라이선스가 문제되면:
- **VRM 포맷**: 오픈소스, 3D 아바타 (2D 아님)
- **spine**: 별도 애니메이션 툴, 라이선스 유사한 구조
- **자체 렌더러**: 2.5D 레이어 렌더링 직접 구현 (공수 큼)

현재는 이 검토 불필요. Phase 4에서 재평가.

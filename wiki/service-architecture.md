# 서비스 아키텍처 (Phase 4 참조용)

> 현재는 개인 아바타 단계. 서비스화 결정 시 이 문서 기반으로 시작.

---

## 서비스 개요

**이름**: (미정)  
**타겟**: VTuber / Live2D 아티스트 (B2C) + 게임사/스튜디오 (B2B API)  
**모델**: Freemium

| 티어 | 내용 | 가격 |
|------|------|------|
| Free | PSD 출력만, 월 3회 | $0 |
| Pro | PSD + 애니메이션 프레임, 무제한 | $15/월 |
| Studio | Pro + API 접근 | $49/월 |

---

## 시스템 아키텍처

```
[유저 브라우저]
      ↓ HTTPS
[Next.js Frontend - Vercel]
      ↓ REST API
[FastAPI Backend - Railway]
      ↓
  ┌───────────────────────────┐
  │     작업 큐 (Celery)       │
  │     Redis (broker)         │
  └───────────┬───────────────┘
              ↓
      ┌───────┴────────┐
      │                │
  [로컬 작업]      [RunPod Serverless]
  (경량 처리)      (ML 추론 - A100)
      │                │
      └───────┬────────┘
              ↓
    [Supabase Storage]
    (PSD, PNG 저장)
```

---

## 주요 API 엔드포인트

```
POST /api/jobs
  body: { image: base64 }
  return: { job_id: string }

GET /api/jobs/{job_id}
  return: { status, progress, result_url }

GET /api/jobs/{job_id}/download
  return: PSD 파일 또는 ZIP (PNG 세트)
```

---

## 프론트엔드 구조

```
app/
├── page.tsx              # 랜딩 페이지
├── upload/
│   └── page.tsx          # 이미지 업로드
├── result/[jobId]/
│   └── page.tsx          # 결과 뷰어 (레이어 on/off)
├── dashboard/
│   └── page.tsx          # 유저 대시보드
└── api/
    └── ...               # API routes
```

**결과 뷰어 핵심 기능**:
- 레이어 on/off 토글
- drawing order 시각화
- PSD / PNG 다운로드 버튼

---

## 백엔드 구조

```
backend/
├── main.py               # FastAPI 앱
├── routers/
│   ├── jobs.py           # 작업 생성/조회
│   └── users.py          # 유저 관리
├── tasks/
│   ├── stage1.py         # See-Through 실행 (RunPod)
│   ├── stage2.py         # SAM3 실행
│   └── stage3.py         # RIFE 실행
├── models/
│   └── job.py            # DB 모델
└── services/
    ├── runpod.py         # RunPod API 클라이언트
    └── storage.py        # Supabase Storage
```

---

## RunPod Serverless 설정

```python
# runpod_handler.py (RunPod에 배포되는 코드)
import runpod
from src.stage1_layerdiff.inference import run_pipeline

def handler(job):
    image_b64 = job["input"]["image"]
    options = job["input"].get("options", {})
    
    result = run_pipeline(image_b64, **options)
    
    return {
        "psd_b64": result.psd_b64,
        "layers": result.layer_info,
        "processing_time": result.elapsed
    }

runpod.serverless.start({"handler": handler})
```

---

## 데이터베이스 스키마 (Supabase)

```sql
-- 작업 테이블
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    status TEXT NOT NULL DEFAULT 'pending',
    -- pending | processing | completed | failed
    input_image_url TEXT,
    result_psd_url TEXT,
    result_frames_url TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    processing_time_seconds FLOAT,
    error_message TEXT
);

-- 유저 사용량 테이블
CREATE TABLE usage (
    user_id UUID REFERENCES auth.users(id),
    month TEXT NOT NULL,  -- "2026-04"
    job_count INT DEFAULT 0,
    PRIMARY KEY (user_id, month)
);
```

---

## 배포 체크리스트 (서비스화 시)

- [ ] weights 라이선스 확인 완료
- [ ] Live2D SDK 라이선스 (리깅 포함 시)
- [ ] RunPod 계정 + API 키
- [ ] Supabase 프로젝트 생성
- [ ] Stripe 계정 + 상품 생성
- [ ] Clerk 설정
- [ ] Vercel 배포
- [ ] Railway 배포
- [ ] RunPod serverless endpoint 생성
- [ ] 도메인 연결
- [ ] VTuber 커뮤니티 베타 공개

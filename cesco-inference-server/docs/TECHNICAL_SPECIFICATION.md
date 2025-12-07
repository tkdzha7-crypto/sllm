# CESCO sLLM Inference Server 기술 명세서


## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [소스코드 디렉토리 구조](#2-소스코드-디렉토리-구조)
3. [데이터 흐름도 (Model Lineage)](#3-데이터-흐름도-model-lineage)
4. [Feature Table 세부 명세](#4-feature-table-세부-명세)
5. [모델 Weight 저장 구조](#5-모델-weight-저장-구조)
6. [모델 학습 상세 명세](#6-모델-학습-상세-명세)
7. [추론 서버 상세 명세](#7-추론-서버-상세-명세)
8. [소스코드 상세 분석](#8-소스코드-상세-분석)

---

## 1. 프로젝트 개요

### 1.1 시스템 목적

CESCO 추론 서버는 **고객 민원(VoC, Voice of Customer) 텍스트를 자동으로 분석하고 분류하는 sLLM(small Language Model) 기반 시스템**입니다. 이 시스템은 고객 상담 내용 등 텍스트를 입력받아 다음과 같은 정보를 자동으로 추출합니다:

- **클레임 여부 판정**: 고객 불만 여부 자동 분류
- **민원 내용 요약**: 핵심 내용 자동 요약
- **해충 종류 식별**: 해충 관련 민원의 경우 해충 종류 자동 분류
- **키워드 추출**: 주요 키워드 자동 추출
- **카테고리 분류**: 대분류/중분류/소분류 계층적 분류

### 1.2 주요 기능

| 기능 | 설명 | 관련 파일 |
|------|------|----------|
| **모델 학습** | Qwen-3-8B 기반 LoRA 파인튜닝 | `train.py` |
| **추론 서버** | FastAPI 기반 REST API 제공 | `app.py` |
| **단일 예측** | 개별 VoC 텍스트 분석 | `/predict` 엔드포인트 |
| **배치 처리** | 다수 VoC 동시 처리 | `/batch` 엔드포인트 |
| **자유 대화** | 커스텀 인스트럭션 기반 대화 | `/chat` 엔드포인트 |

### 1.3 기술 스택

#### 핵심 프레임워크

| 카테고리 | 기술 | 버전 | 용도 |
|---------|------|------|------|
| **웹 프레임워크** | FastAPI | ≥0.104.0 | REST API 서버 |
| **ASGI 서버** | Uvicorn | ≥0.24.0 | 비동기 서버 실행 |
| **딥러닝** | PyTorch | 2.8.0 | 모델 학습/추론 |
| **트랜스포머** | Transformers | ≥4.56.1 | LLM 모델 처리 |
| **모델 최적화** | Unsloth | 2025.10.12 | 학습 속도 최적화 |
| **파라미터 효율화** | PEFT | ≥0.17.1 | LoRA 구현 |

#### 전체 의존성 (`pyproject.toml` 기준)

```toml
dependencies = [
    "fastapi>=0.104.0",        # 웹 프레임워크
    "uvicorn[standard]>=0.24.0", # ASGI 서버
    "torch==2.8.0",            # 딥러닝 프레임워크
    "transformers>=4.56.1",    # 트랜스포머 모델
    "datasets>=4.1.0",         # 데이터셋 관리
    "pydantic>=2.4.0",         # 데이터 검증
    "peft>=0.17.1",            # LoRA 구현
    "unsloth==2025.10.12",     # 학습 최적화
    "unsloth-zoo==2025.10.13", # Unsloth 확장
    "accelerate>=0.24.0",      # 분산 학습
    "safetensors>=0.4.0",      # 모델 직렬화
    "trl",                     # SFT 트레이너
    "wandb>=0.22.2",           # 실험 추적
    "pyarrow>=21.0.0",         # 데이터 처리
    "rank-bm25>=0.2.2",        # BM25 검색
    "xlwings>=0.33.15",        # Excel 처리
    "pyodbc>=5.2.0",           # DB 연결 (미사용)
    "sqlalchemy>=2.0.0",       # ORM (미사용)
]
```

---

## 2. 소스코드 디렉토리 구조

### 2.1 전체 디렉토리 트리

```
cesco-inference-server/
│
├── app.py                      # [438줄] FastAPI 추론 서버 메인 애플리케이션
├── train.py                    # [414줄] 모델 학습 스크립트
│
├── pyproject.toml              # Python 프로젝트 설정 및 의존성 정의
├── .python-version             # Python 버전 지정 (3.10+)
│
├── Dockerfile                  # Docker 이미지 빌드 설정
├── docker-compose.yml          # Docker 오케스트레이션 설정
├── docker-build.sh             # Docker 빌드 스크립트
├── start_server.sh             # 서버 시작 스크립트
│
├── .env.example                # 환경변수 템플릿
├── .pre-commit-config.yaml     # Pre-commit 훅 설정
├── .gitignore                  # Git 무시 목록
└── .dockerignore               # Docker 빌드 무시 목록
```

### 2.2 각 파일/폴더별 상세 설명

#### 2.2.1 핵심 소스 파일

| 파일 | 줄 수 | 역할 | 주요 클래스/함수 |
|------|-------|------|-----------------|
| `app.py` | 438 | FastAPI 추론 서버 | `CESCOInference`, `predict()`, `batch_predict()` |
| `train.py` | 414 | 모델 학습 스크립트 | `CESCOTrainer`, `train()`, `prepare_dataset()` |

#### 2.2.2 설정 파일

| 파일 | 역할 | 주요 설정 |
|------|------|----------|
| `pyproject.toml` | 프로젝트 메타데이터 및 의존성 | Python ≥3.10, 의존성 26개 |
| `.env.example` | 환경변수 템플릿 | `MODEL_PATH`, `HOST`, `PORT`, `LOG_LEVEL` |
| `.pre-commit-config.yaml` | 코드 품질 관리 | ruff, mypy, trailing-whitespace |

#### 2.2.3 배포 관련 파일

| 파일 | 역할 | 주요 내용 |
|------|------|----------|
| `Dockerfile` | Docker 이미지 빌드 | NVIDIA CUDA 12.8 기반, Python 3.10 |
| `docker-compose.yml` | 컨테이너 오케스트레이션 | GPU 지원, 볼륨 마운트, 헬스체크 |
| `start_server.sh` | 프로덕션 서버 시작 | 환경변수 검증, uvicorn 실행 |

---

## 3. 데이터 흐름도 (Model Lineage)

### 3.1 학습 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           학습 데이터 흐름도                                  │
└─────────────────────────────────────────────────────────────────────────────┘

[1. 데이터 입력]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  JSON/JSONL 파일                                                            │
│  예: training_data.jsonl                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ {"input_text": "바퀴벌레가 나와요", "category_dict": "...",          │   │
│  │  "response": "{\"is_claim\": \"claim\", ...}"}                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  코드 위치: train.py:185-199 (prepare_dataset 함수)                         │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[2. 데이터 로딩]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  HuggingFace Dataset 변환                                                   │
│  - JSON/JSONL 파일 읽기                                                     │
│  - Dataset.from_list() 변환                                                 │
│  - Train/Validation 분할 (기본 90:10)                                       │
│  코드 위치: train.py:200-215                                                │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[3. 텍스트 전처리]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  input_text_cleaning() 함수                                                 │
│  - 특수문자 제거 (정규식: [^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s])                      │
│  - 다중 공백 정규화                                                         │
│  - 앞뒤 공백 제거                                                           │
│  코드 위치: train.py:104-113                                                │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[4. 프롬프트 포맷팅]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Alpaca 프롬프트 템플릿 적용                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Below is an instruction that describes a task...                    │   │
│  │ ### Instruction:                                                    │   │
│  │ {시스템 인스트럭션}                                                  │   │
│  │ ### Input:                                                          │   │
│  │ [Voice-of-Customer]: {정제된 텍스트}                                 │   │
│  │ [Category Dict]                                                     │   │
│  │ {카테고리 딕셔너리}                                                  │   │
│  │ ### Response:                                                       │   │
│  │ {정답 JSON}                                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  코드 위치: train.py:163-183 (format_prompt 함수)                           │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5. 토큰화]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Qwen 토크나이저 적용                                                       │
│  - 최대 시퀀스 길이: 8096 토큰                                              │
│  - Truncation 적용                                                          │
│  - Padding 적용 (pad_token = eos_token)                                     │
│  코드 위치: train.py:129-131                                                │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[6. LoRA 학습]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SFTTrainer 기반 학습                                                       │
│  - 기본 모델: Qwen/Qwen-3-8B (4-bit Quantization).                          │
│  - LoRA 어댑터 적용 (r=16, alpha=16)                                        │
│  - 타겟 모듈: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj │
│  - 옵티마이저: AdamW 8-bit                                                  │
│  - 스케줄러: Cosine                                                         │
│  코드 위치: train.py:217-302                                                │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[7. 모델 저장]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  모델 아티팩트 저장                                                         │
│  - final_model/: LoRA 어댑터 (~100MB)                                       │
│  - final_model_merged/: 병합된 전체 모델 (~16GB)                            │
│  - checkpoint-*/: 중간 체크포인트                                           │
│  - logs/: TensorBoard 로그                                                  │
│  코드 위치: train.py:308-330                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 추론 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           추론 데이터 흐름도                                  │
└─────────────────────────────────────────────────────────────────────────────┘

[1. API 요청 수신]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  POST /predict                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                   │   │
│  │   "input_text": "바퀴벌레가 계속 나와서 해지하고 싶습니다",         │   │
│  │   "max_new_tokens": 512,                                            │   │
│  │   "temperature": 0.1,                                               │   │
│  │   "top_p": 0.9                                                      │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  코드 위치: app.py:369-395                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[2. 요청 검증]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Pydantic 모델 검증 (InferenceRequest)                                      │
│  - input_text: 필수 (str)                                                   │
│  - input_categories: 선택 (str | None)                                      │
│  - max_new_tokens: 1-2048 (기본값: 512)                                     │
│  - temperature: 0.0-2.0 (기본값: 0.1)                                       │
│  - top_p: 0.0-1.0 (기본값: 0.9)                                             │
│  코드 위치: app.py:30-37                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[3. 텍스트 전처리]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  input_text_cleaning() 함수                                                 │
│  - 영문자(a-z, A-Z) 유지                                                    │
│  - 한글(가-힣, ㄱ-ㅎ, ㅏ-ㅣ) 유지                                           │
│  - 숫자(0-9) 유지                                                           │
│  - 공백 유지 및 정규화                                                      │
│  - 기타 특수문자 모두 제거                                                  │
│  코드 위치: app.py:130-139                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[4. 프롬프트 생성]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Alpaca 프롬프트 포맷 적용                                                   │
│  - [Voice-of-Customer] 레이블 추가                                          │
│  - [Category Dict] 카테고리 딕셔너리 추가                                   │
│  - 기본 카테고리 사용 (input_categories 미제공 시)                          │
│  코드 위치: app.py:196-204                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[5. 토큰화]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Qwen 토크나이저                                                            │
│  - return_tensors="pt"                                                      │
│  - truncation=True                                                          │
│  - max_length = max_seq_length - max_new_tokens                             │
│  - GPU 디바이스로 이동                                                      │
│  코드 위치: app.py:207-209                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[6. 모델 추론]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  model.generate() 호출                                                      │
│  - do_sample=True (샘플링 활성화)                                           │
│  - temperature, top_p 적용                                                  │
│  - use_cache=True (KV 캐시 사용)                                            │
│  - output_scores=True (신뢰도 계산용)                                       │
│  코드 위치: app.py:212-224                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[7. 신뢰도 점수 계산]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Transition Score 기반 신뢰도                                               │
│  - compute_transition_scores() 호출                                         │
│  - 로그 확률 → 확률 변환 (torch.exp)                                        │
│  - 평균 확률 계산                                                           │
│  코드 위치: app.py:226-231                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[8. 응답 디코딩]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  토큰 → 텍스트 변환                                                         │
│  - skip_special_tokens=True                                                 │
│  - "### Response:" 이후 텍스트 추출                                         │
│  코드 위치: app.py:234-240                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[9. JSON 파싱]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  parse_json_response() 함수                                                 │
│  - "{" ~ "}" 범위 추출                                                      │
│  - json.loads() 파싱                                                        │
│  - 파싱 실패 시 None 반환                                                   │
│  코드 위치: app.py:279-293                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
[10. API 응답 반환]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  InferenceResponse 반환                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                   │   │
│  │   "success": true,                                                  │   │
│  │   "confidence_score": 0.87,                                         │   │
│  │   "raw_response": "...",                                            │   │
│  │   "parsed_response": {                                              │   │
│  │     "is_claim": "claim",                                            │   │
│  │     "summary": "바퀴벌레 출현으로 인한 해약 요청",                   │   │
│  │     "bug_type": "바퀴",                                             │   │
│  │     "keywords": ["바퀴벌레", "해약"],                               │   │
│  │     "categories": [...]                                             │   │
│  │   },                                                                │   │
│  │   "error": null                                                     │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  코드 위치: app.py:389-391                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. API Request (Payload) 명세

### 4.1 입력 Payload 명세

#### 4.1.1 API 요청 Payload 파라미터

| 필드명 | 타입 | 필수 | 기본값 | 범위 | 설명 | 코드 위치 |
|--------|------|------|--------|------|------|----------|
| `input_text` | `str` | True | - | - | 분석할 고객 민원 텍스트 | app.py:33 |
| `input_categories` | `str \| None` | False | `None` | - | 커스텀 카테고리 딕셔너리 (JSON 문자열) | app.py:34 |
| `max_new_tokens` | `int` | False | `512` | 1-2048 | 생성할 최대 토큰 수 | app.py:35 |
| `temperature` | `float` | False | `0.1` | 0.0-2.0 | 샘플링 온도 (낮을수록 일관적) | app.py:36 |
| `top_p` | `float` | False | `0.9` | 0.0-1.0 | Nucleus sampling 파라미터 | app.py:37 |

#### 4.1.2 입력 텍스트 전처리 규칙

`input_text_cleaning()` 함수 (app.py:130-139, train.py:104-113)

| 처리 단계 | 정규식 | 설명 |
|----------|--------|------|
| 특수문자 제거 | `[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s]` | 영문, 한글, 숫자, 공백만 유지 |
| 공백 정규화 | `\s+` → ` ` | 연속 공백을 단일 공백으로 변환 |
| 앞뒤 공백 제거 | `.strip()` | 문자열 앞뒤 공백 제거 |

**전처리 예시**:
```
입력: "바퀴벌레가!!! 나와서...  해지하고 싶어요ㅠㅠ"
출력: "바퀴벌레가 나와서 해지하고 싶어요ㅠㅠ"
```

#### 4.1.3 학습 데이터 입력 형식

JSON/JSONL 파일 형식 (train.py:185-199)

```json
{
  "input_text": "바퀴벌레가 계속 나와서 해지하고 싶습니다",
  "category_dict": "{\"해충 문제\": {\"방문 요청\": [\"바퀴\", \"쥐\", \"개미\", \"기타\"]}}",
  "response": "{\"is_claim\": \"claim\", \"summary\": \"바퀴벌레 지속 출현으로 해지 요청\", \"bug_type\": \"바퀴\", \"keywords\": [\"바퀴벌레\", \"해지\"], \"categories\": [{\"대분류\": \"해충 문제\", \"중분류\": \"방문 요청\", \"소분류\": \"바퀴\", \"근거\": \"바퀴벌레 지속 출현\"}]}"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `input_text` | `str` | 원본 고객 민원 텍스트 |
| `category_dict` | `str` | 카테고리 딕셔너리 (JSON 문자열) |
| `response` | `str` | 정답 JSON (모델이 학습할 출력) |

### 4.2 출력 Response 명세

#### 4.2.1 모델 출력 JSON 구조

| 필드명 | 타입 | 필수 | 설명 | 예시 |
|--------|------|------|------|------|
| `is_claim` | `str` | True | 클레임 여부 | `"claim"` 또는 `"non-claim"` |
| `summary` | `str` | True | 민원 내용 요약 | `"바퀴벌레 출현으로 인한 해약 요청"` |
| `bug_type` | `str \| null` | False | 해충 종류 (해충 관련 시) | `"바퀴"`, `"쥐"`, `"개미"`, `"기타"`, `null` |
| `keywords` | `list[str]` | True | 주요 키워드 목록 | `["바퀴벌레", "해약", "불만"]` |
| `categories` | `list[object]` | True | 카테고리 분류 (최대 5개) | 아래 참조 |

#### 4.2.2 categories 객체 구조

| 필드명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `대분류` | `str` | 1차 분류 | `"해충 문제"` |
| `중분류` | `str` | 2차 분류 | `"방문 요청"` |
| `소분류` | `str` | 3차 분류 | `"바퀴"` |
| `근거` | `str` | 분류 근거 설명 | `"바퀴벌레 지속 출현"` |

#### 4.2.3 API 응답 구조 (InferenceResponse)

| 필드명 | 타입 | 설명 | 코드 위치 |
|--------|------|------|----------|
| `success` | `bool` | 요청 성공 여부 | app.py:53 |
| `raw_response` | `str` | 모델 출력 원문 | app.py:54 |
| `parsed_response` | `dict \| None` | 파싱된 JSON 객체 | app.py:55 |
| `error` | `str \| None` | 에러 메시지 (실패 시) | app.py:56 |
| `confidence_score` | `float` | 신뢰도 점수 (0.0-1.0) | app.py:57 |

---

## 5. 모델 Weight 저장 구조

### 5.1 저장 경로 및 파일 구조

모델 학습 완료 후 생성되는 디렉토리 구조입니다.

**저장 위치**: `{output_dir}_YYYYMMDD_HHMMSS/`

```
training_output_20251130_143000/
│
├── final_model/                        # LoRA 어댑터 (권장, ~100MB)
│   ├── adapter_config.json             # LoRA 하이퍼파라미터 설정
│   ├── adapter_model.safetensors       # LoRA 가중치 (핵심 파일)
│   ├── config.json                     # 모델 아키텍처 설정
│   ├── special_tokens_map.json         # 특수 토큰 매핑
│   ├── tokenizer.json                  # 토크나이저 데이터
│   └── tokenizer_config.json           # 토크나이저 설정
│
├── final_model_merged/                 # 병합된 전체 모델 (~16GB)
│   ├── config.json                     # 모델 아키텍처 설정
│   ├── generation_config.json          # 생성 파라미터 설정
│   ├── model-00001-of-00004.safetensors # 분할된 가중치 1 (~4GB)
│   ├── model-00002-of-00004.safetensors # 분할된 가중치 2 (~4GB)
│   ├── model-00003-of-00004.safetensors # 분할된 가중치 3 (~4GB)
│   ├── model-00004-of-00004.safetensors # 분할된 가중치 4 (~4GB)
│   ├── model.safetensors.index.json    # 분할 파일 인덱스
│   ├── special_tokens_map.json         # 특수 토큰 매핑
│   ├── tokenizer.json                  # 토크나이저 데이터
│   └── tokenizer_config.json           # 토크나이저 설정
│
├── checkpoint-100/                     # 100 스텝 체크포인트
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
│
├── checkpoint-200/                     # 200 스텝 체크포인트
│   └── ...
│
├── logs/                               # TensorBoard 로그
│   └── events.out.tfevents.1732945800.gpu-server
│
└── train_results.json                  # 학습 메트릭 결과
```

### 5.2 LoRA 어댑터 파일 명세

`final_model/` 디렉토리에 저장되는 파일입니다.

| 파일명 | 크기 (예상) | 설명 |
|--------|------------|------|
| `adapter_config.json` | ~1KB | LoRA 하이퍼파라미터 (r, alpha, dropout, target_modules) |
| `adapter_model.safetensors` | ~100MB | LoRA 가중치 (학습된 Low-Rank 행렬) |
| `config.json` | ~2KB | 기본 모델 아키텍처 정보 |
| `tokenizer.json` | ~7MB | 토크나이저 어휘 및 규칙 |
| `tokenizer_config.json` | ~1KB | 토크나이저 설정 |
| `special_tokens_map.json` | ~1KB | 특수 토큰 (pad, eos, bos) 매핑 |

**adapter_config.json 예시**:
```json
{
  "base_model_name_or_path": "Qwen/Qwen-3-8B",
  "bias": "none",
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "r": 16,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

### 5.3 병합 모델 파일 명세

`final_model_merged/` 디렉토리에 저장되는 파일입니다.

| 파일명 | 크기 (예상) | 설명 |
|--------|------------|------|
| `model-00001-of-00004.safetensors` | ~4GB | 모델 가중치 파트 1 |
| `model-00002-of-00004.safetensors` | ~4GB | 모델 가중치 파트 2 |
| `model-00003-of-00004.safetensors` | ~4GB | 모델 가중치 파트 3 |
| `model-00004-of-00004.safetensors` | ~4GB | 모델 가중치 파트 4 |
| `model.safetensors.index.json` | ~50KB | 레이어-파일 매핑 인덱스 |
| `config.json` | ~2KB | 모델 아키텍처 설정 |
| `generation_config.json` | ~1KB | 생성 기본 파라미터 |

### 5.4 체크포인트 구조

학습 중 `save_steps` 간격으로 저장되는 체크포인트입니다.

| 파일명 | 설명 |
|--------|------|
| `adapter_model.safetensors` | 해당 스텝의 LoRA 가중치 |
| `optimizer.pt` | 옵티마이저 상태 (모멘텀, 분산 등) |
| `scheduler.pt` | 학습률 스케줄러 상태 |
| `trainer_state.json` | 트레이너 상태 (현재 스텝, 에포크 등) |
| `training_args.bin` | 학습 인자 직렬화 |

### 5.5 저장 관련 코드 위치

| 기능 | 메서드 | 코드 위치 |
|------|--------|----------|
| LoRA 어댑터 저장 | `model.save_pretrained()` | train.py:312 |
| 토크나이저 저장 | `tokenizer.save_pretrained()` | train.py:313 |
| 병합 모델 저장 | `model.save_pretrained_merged()` | train.py:317-321 |
| 메트릭 저장 | `trainer.save_metrics()` | train.py:329 |
| 전체 모델 저장 (Full FT) | `model.save_pretrained()` | train.py:324 |

**저장 옵션** (train.py:320 `save_method` 파라미터):

| 옵션 | 설명 | 용량 |
|------|------|------|
| `merged_16bit` | 16비트 정밀도로 병합 (기본값) | ~16GB |
| `merged_4bit` | 4비트 양자화로 병합 | ~4GB |
| `lora` | LoRA 어댑터만 저장 | ~100MB |

---

## 6. 모델 학습 상세 명세

### 6.1 기본 모델 정보

| 항목 | 값 | 설명 |
|------|-----|------|
| **모델명** | `Qwen/Qwen-3-8B` | Alibaba Qwen 시리즈 8B 파라미터 모델 |
| **파라미터 수** | 약 80억 개 | 전체 모델 파라미터 |
| **학습 가능 파라미터** (LoRA) | 약 2천만 개 (~0.25%) | LoRA 어댑터 파라미터만 학습 |
| **양자화** | 4-bit | 메모리 효율을 위한 4비트 양자화 적용 |
| **최대 시퀀스 길이** | 8096 토큰 | 입력+출력 합계 최대 길이 |

### 6.2 학습 방식 (LoRA vs Full Fine-tuning)

#### 6.2.1 LoRA (Low-Rank Adaptation) - 기본값

| 설정 | 값 | 코드 위치 | 설명 |
|------|-----|----------|------|
| `use_lora` | `True` | train.py:31 | LoRA 활성화 |
| `lora_r` | `16` | train.py:32 | Low-Rank 차원 |
| `lora_alpha` | `16` | train.py:33 | 스케일링 팩터 |
| `lora_dropout` | `0.05` | train.py:34 | 드롭아웃 비율 |
| `target_modules` | 7개 모듈 | train.py:44-51 | 적용 대상 레이어 |

**target_modules 상세**:
```python
target_modules = [
    "q_proj",     # Query 프로젝션 (Attention)
    "k_proj",     # Key 프로젝션 (Attention)
    "v_proj",     # Value 프로젝션 (Attention)
    "o_proj",     # Output 프로젝션 (Attention)
    "gate_proj",  # Gate 프로젝션 (MLP)
    "up_proj",    # Up 프로젝션 (MLP)
    "down_proj",  # Down 프로젝션 (MLP)
]
```

**LoRA 수식**:
```
W' = W + (α/r) × BA

여기서:
- W: 원본 가중치 행렬 (동결)
- B: Low-Rank 행렬 B (학습)
- A: Low-Rank 행렬 A (학습)
- r: Rank (차원) = 16
- α: Scaling factor = 16
```

#### 6.2.2 Full Fine-tuning

| 설정 | 값 | 코드 위치 | 설명 |
|------|-----|----------|------|
| `--full_finetune` | CLI 플래그 | train.py:343-345 | 전체 파인튜닝 활성화 |

Full Fine-tuning 시 모든 파라미터가 학습 가능 상태로 설정됩니다 (train.py:160-161).

### 6.3 학습 파라미터 명세 (14개)

#### 6.3.1 모델 관련 파라미터

| 파라미터 | 기본값 | 타입 | 설명 | 코드 위치 |
|---------|-------|------|------|----------|
| `base_model` | `Qwen/Qwen-3-8B` | str | 기본 모델 이름/경로 | train.py:340 |
| `max_seq_length` | `8096` | int | 최대 시퀀스 길이 | train.py:341 |
| `use_lora` | `True` | bool | LoRA 사용 여부 | train.py:342 |

#### 6.3.2 LoRA 관련 파라미터

| 파라미터 | 기본값 | 타입 | 설명 | 코드 위치 |
|---------|-------|------|------|----------|
| `lora_r` | `16` | int | LoRA Rank | train.py:348 |
| `lora_alpha` | `16` | int | LoRA Alpha (스케일링) | train.py:349 |
| `lora_dropout` | `0.05` | float | LoRA Dropout | train.py:350 |

#### 6.3.3 데이터 관련 파라미터

| 파라미터 | 기본값 | 타입 | 설명 | 코드 위치 |
|---------|-------|------|------|----------|
| `data_path` | (필수) | str | 학습 데이터 경로 | train.py:353 |
| `validation_split` | `0.1` | float | 검증 데이터 비율 | train.py:354 |

#### 6.3.4 학습 관련 파라미터

| 파라미터 | 기본값 | 타입 | 설명 | 코드 위치 |
|---------|-------|------|------|----------|
| `output_dir` | `./outputs` | str | 출력 디렉토리 | train.py:357 |
| `num_epochs` | `3` | int | 학습 에포크 수 | train.py:358 |
| `batch_size` | `2` | int | 배치 크기 (per device) | train.py:359 |
| `gradient_accumulation_steps` | `4` | int | 그래디언트 누적 스텝 | train.py:361 |
| `learning_rate` | `2e-4` | float | 학습률 | train.py:362 |
| `warmup_steps` | `10` | int | Warmup 스텝 수 | train.py:363 |

#### 6.3.5 TrainingArguments 내부 파라미터

train.py:256-283에서 설정되는 추가 파라미터입니다:

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `optim` | `adamw_8bit` | 8-bit AdamW 옵티마이저 |
| `weight_decay` | `0.01` | 가중치 감쇠 |
| `lr_scheduler_type` | `cosine` | Cosine 학습률 스케줄러 |
| `save_steps` | `100` | 체크포인트 저장 간격 |
| `eval_steps` | `100` | 평가 간격 |
| `logging_steps` | `10` | 로깅 간격 |
| `save_total_limit` | `3` | 최대 체크포인트 보관 수 |
| `early_stopping_patience` | `3` | Early Stopping patience |

### 6.4 데이터 포맷 및 전처리

#### 6.4.1 지원 데이터 형식

| 형식 | 확장자 | 로딩 방식 | 코드 위치 |
|------|--------|----------|----------|
| JSON | `.json` | `json.load()` | train.py:194-195 |
| JSON Lines | `.jsonl` | 라인별 `json.loads()` | train.py:191-192 |

#### 6.4.2 데이터 스키마

```json
{
  "input_text": "string (필수) - 고객 민원 원문",
  "category_dict": "string (선택) - 카테고리 딕셔너리 JSON",
  "response": "string (필수) - 정답 JSON"
}
```

#### 6.4.3 프롬프트 템플릿 (Alpaca Format)

train.py:55-64에 정의된 템플릿:

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

#### 6.4.4 시스템 인스트럭션

train.py:67-99에 정의된 기본 인스트럭션:

```
You are a helpful assistant that analyzes customer interactions for CESCO.
Based on the customer's interaction summary, please respond in JSON format including the following information:

1. is_claim: 클레임 여부 (claim 또는 non-claim)
2. summary: 민원 내용 요약
3. bug_type: 해충 종류 (해충 관련인 경우만, 없으면 null)
4. keywords: 주요 키워드 리스트
5. categories: 대분류, 중분류, 소분류 조합 리스트 (복수 선택 가능, 최대 5개)

** HARD REQUIREMENTS **
- CAUTION: You can only choose categories from the provided [Category Dict].
- NEVER respond with categories that are not in the dictionary.
...
```

### 6.5 학습 실행 방법

#### 6.5.1 기본 실행

```bash
python train.py \
  --data_path ./data/training_data.jsonl \
  --output_dir ./outputs \
  --num_epochs 3
```

#### 6.5.2 전체 옵션 실행

```bash
python train.py \
  --base_model Qwen/Qwen-3-8B \
  --max_seq_length 8096 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --data_path ./data/training_data.jsonl \
  --validation_split 0.1 \
  --output_dir ./outputs \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --warmup_steps 10 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 100 \
  --early_stopping_patience 3 \
  --seed 42
```

#### 6.5.3 Full Fine-tuning 실행

```bash
python train.py \
  --full_finetune \
  --data_path ./data/training_data.jsonl \
  --output_dir ./outputs
```

### 6.6 코드 라인별 상세 설명

#### train.py 주요 섹션

| 라인 범위 | 내용 | 설명 |
|----------|------|------|
| 1-21 | Import 및 로깅 설정 | 필요 라이브러리 임포트 |
| 24-102 | `CESCOTrainer.__init__()` | 트레이너 초기화, 프롬프트 템플릿 정의 |
| 104-113 | `input_text_cleaning()` | 입력 텍스트 전처리 |
| 115-161 | `load_model()` | 모델 및 토크나이저 로딩, LoRA 적용 |
| 163-183 | `format_prompt()` | 학습 데이터를 Alpaca 포맷으로 변환 |
| 185-215 | `prepare_dataset()` | 데이터셋 로딩 및 분할 |
| 217-332 | `train()` | 학습 실행 및 모델 저장 |
| 335-414 | `main()` | CLI 인자 파싱 및 학습 실행 |

---

## 7. 추론 서버 상세 명세

### 7.1 API 엔드포인트 명세 (5개)

#### 7.1.1 엔드포인트 요약

| 엔드포인트 | 메서드 | 설명 | 코드 위치 |
|-----------|--------|------|----------|
| `/` | GET | 서버 상태 확인 | app.py:327-330 |
| `/health` | GET | 헬스 체크 | app.py:333-339 |
| `/predict` | POST | 구조화된 VoC 분석 | app.py:369-395 |
| `/chat` | POST | 자유형식 대화 | app.py:342-366 |
| `/batch` | POST | 배치 처리 | app.py:398-426 |

#### 7.1.2 GET / (루트)

**목적**: 서버 상태 확인

**응답**:
```json
{
  "message": "CESCO Inference API",
  "status": "running",
  "model_loaded": true
}
```

#### 7.1.3 GET /health

**목적**: 헬스 체크 (로드밸런서, 컨테이너 오케스트레이션용)

**응답 (성공)**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**응답 (실패)**: HTTP 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

#### 7.1.4 POST /predict

**목적**: 단일 VoC 텍스트 분석

**요청**:
```json
{
  "input_text": "바퀴벌레가 계속 나와서 해지하고 싶습니다",
  "input_categories": null,
  "max_new_tokens": 512,
  "temperature": 0.1,
  "top_p": 0.9
}
```

**응답 (성공)**:
```json
{
  "success": true,
  "confidence_score": 0.87,
  "raw_response": "{\"is_claim\": \"claim\", ...}",
  "parsed_response": {
    "is_claim": "claim",
    "summary": "바퀴벌레 지속 출현으로 해지 요청",
    "bug_type": "바퀴",
    "keywords": ["바퀴벌레", "해지"],
    "categories": [
      {
        "대분류": "해충 문제",
        "중분류": "방문 요청",
        "소분류": "바퀴",
        "근거": "바퀴벌레 지속 출현"
      }
    ]
  },
  "error": null
}
```

#### 7.1.5 POST /chat

**목적**: 커스텀 인스트럭션 기반 자유형식 대화

**요청**:
```json
{
  "instruction": "다음 텍스트를 영어로 번역해주세요",
  "input_text": "안녕하세요, CESCO입니다",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**응답**:
```json
{
  "success": true,
  "raw_response": "Hello, this is CESCO.",
  "parsed_response": null,
  "error": null,
  "confidence_score": 0.0
}
```

#### 7.1.6 POST /batch

**목적**: 다수의 VoC 동시 처리

**요청**:
```json
[
  {"input_text": "첫 번째 민원"},
  {"input_text": "두 번째 민원"},
  {"input_text": "세 번째 민원"}
]
```

**응답**: `InferenceResponse[]` 배열

### 7.2 추론 파라미터 설명

| 파라미터 | 범위 | 기본값 | 설명 |
|---------|------|--------|------|
| `max_new_tokens` | 1-2048 | 512 | 생성할 최대 토큰 수. 높을수록 긴 응답 가능 |
| `temperature` | 0.0-2.0 | 0.1 | 샘플링 온도. 낮을수록 결정적(일관적), 높을수록 창의적 |
| `top_p` | 0.0-1.0 | 0.9 | Nucleus sampling. 누적 확률 상위 p% 토큰에서 샘플링 |

**Temperature 가이드**:
| 값 | 용도 |
|----|------|
| 0.0-0.3 | 분류, 추출 등 일관성 필요 태스크 (권장) |
| 0.3-0.7 | 균형잡힌 출력 |
| 0.7-1.0 | 창의적 생성 |
| 1.0+ | 매우 다양한/무작위 출력 |

### 7.4 신뢰도 점수 계산 방식

app.py:226-231에서 계산됩니다.

**계산 과정**:

1. **Transition Score 계산**:
```python
transition_scores = self.model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)
```

2. **로그 확률 → 확률 변환**:
```python
probabilities = torch.exp(transition_scores)
```

3. **평균 확률 계산**:
```python
confidence_score = probabilities.mean().item()
```

**해석**:
- `confidence_score`는 0.0~1.0 범위
- 높을수록 모델이 출력에 확신을 가짐
- 일반적으로 0.7 이상이면 신뢰할 수 있는 출력

---

## 8. 소스코드 상세 분석

### 8.1 train.py 분석

#### 8.1.1 클래스 구조

```
CESCOTrainer
├── __init__(self, ...)                    # 초기화 (line 27-102)
├── input_text_cleaning(self, text)        # 텍스트 전처리 (line 104-113)
├── load_model(self)                       # 모델 로딩 (line 115-161)
├── format_prompt(self, example)           # 프롬프트 포맷팅 (line 163-183)
├── prepare_dataset(self, data_path, ...)  # 데이터셋 준비 (line 185-215)
└── train(self, ...)                       # 학습 실행 (line 217-332)
```

#### 8.1.2 주요 함수별 상세 설명

**`__init__()`** (line 27-102)
```python
def __init__(
    self,
    base_model_name: str = "Qwen/Qwen-3-8B",
    max_seq_length: int = 8096,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list | None = None,
):
```
- 모델 및 LoRA 파라미터 초기화
- Alpaca 프롬프트 템플릿 정의
- 기본 인스트럭션 정의

**`input_text_cleaning()`** (line 104-113)
```python
def input_text_cleaning(self, text: str) -> str:
    cleaned_text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text
```
- 특수문자 제거
- 공백 정규화

**`load_model()`** (line 115-161)
```python
def load_model(self):
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name=self.base_model_name,
        max_seq_length=self.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if self.use_lora:
        self.model = FastLanguageModel.get_peft_model(...)
```
- Unsloth로 모델 로딩 (4-bit 양자화)
- LoRA 어댑터 적용
- 학습 가능 파라미터 로깅

**`format_prompt()`** (line 163-183)
```python
def format_prompt(self, example: dict[str, Any]) -> str:
    cleaned_input = self.input_text_cleaning(example.get("input_text", ""))
    formatted_input = f"""[Voice-of-Customer]: {cleaned_input}

[Category Dict]
{category_dict}"""
    prompt = self.alpaca_prompt_template.format(...)
    return prompt
```
- 입력 텍스트 정제
- Alpaca 템플릿 적용

**`prepare_dataset()`** (line 185-215)
```python
def prepare_dataset(self, data_path: str, validation_split: float = 0.1):
    # JSON/JSONL 로딩
    # HuggingFace Dataset 변환
    # Train/Validation 분할
    return train_dataset, eval_dataset
```

**`train()`** (line 217-332)
```python
def train(self, train_dataset, eval_dataset, ...):
    training_args = TrainingArguments(...)
    trainer = SFTTrainer(...)
    train_result = trainer.train()
    # 모델 저장
    return trainer, train_result
```

### 8.2 app.py 분석

#### 8.2.1 클래스/함수 구조

```
Pydantic Models
├── InferenceRequest (line 30-37)
├── ChatRequest (line 40-47)
└── InferenceResponse (line 50-57)

CESCOInference (line 60-293)
├── __init__(self, model_path, max_seq_length)
├── input_text_cleaning(self, text)
├── _load_model(self)
├── generate_response(self, input_text, ...)
├── generate_free_response(self, instruction, ...)
└── parse_json_response(self, response)

FastAPI App
├── lifespan(app)                    # 앱 생명주기 관리 (line 296-316)
├── root()                           # GET / (line 327-330)
├── health()                         # GET /health (line 333-339)
├── chat(request)                    # POST /chat (line 342-366)
├── predict(request)                 # POST /predict (line 369-395)
└── batch_predict(requests)          # POST /batch (line 398-426)
```

#### 8.2.2 주요 함수 상세 설명

**`_load_model()`** (line 141-181)
```python
def _load_model(self):
    # LoRA 어댑터 파일 확인
    lora_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
    has_lora = any(os.path.exists(os.path.join(self.model_path, f)) for f in lora_files)

    # Unsloth로 로딩
    self.model, self.tokenizer = FastLanguageModel.from_pretrained(...)

    # 추론 모드 설정
    FastLanguageModel.for_inference(self.model)
```

**`generate_response()`** (line 183-242)
```python
def generate_response(self, input_text, input_categories, ...):
    # 1. 텍스트 정제
    cleaned_text = self.input_text_cleaning(input_text)

    # 2. 프롬프트 생성
    formatted_input = f"[Voice-of-Customer]: {cleaned_text}\n\n[Category Dict]\n{...}"
    prompt = self.alpaca_prompt_template.format(...)

    # 3. 토큰화
    inputs = self.tokenizer(prompt, return_tensors="pt", ...)

    # 4. 생성
    with torch.no_grad():
        outputs = self.model.generate(**inputs, ...)

    # 5. 신뢰도 계산
    transition_scores = self.model.compute_transition_scores(...)
    confidence_score = torch.exp(transition_scores).mean().item()

    # 6. 디코딩
    generated_text = self.tokenizer.decode(outputs.sequences[0], ...)

    return response, confidence_score
```

**`parse_json_response()`** (line 279-293)
```python
def parse_json_response(self, response: str):
    if "{" in response and "}" in response:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_str = response[json_start:json_end]
        return json.loads(json_str)
    return None
```

### 8.3 주요 클래스 및 함수 설명

#### 8.3.1 CESCOTrainer vs CESCOInference

| 항목 | CESCOTrainer | CESCOInference |
|------|--------------|----------------|
| 파일 | train.py | app.py |
| 목적 | 모델 학습 | 모델 추론 |
| 모델 로딩 방식 | `FastLanguageModel.from_pretrained()` | 동일 |
| LoRA 적용 | `get_peft_model()` | 자동 감지 |
| 모드 | 학습 모드 | `for_inference()` |
| 출력 | 모델 파일 저장 | JSON 응답 |

#### 8.3.2 공통 함수

| 함수 | 위치 | 설명 |
|------|------|------|
| `input_text_cleaning()` | train.py:104, app.py:130 | 동일한 텍스트 전처리 로직 |
| Alpaca 템플릿 | train.py:55, app.py:72 | 동일한 프롬프트 형식 |
| 인스트럭션 | train.py:67, app.py:84 | 동일한 시스템 프롬프트 |

---

## 9. Appendix

### 9.1 환경 변수 설정

`.env` 파일 또는 시스템 환경변수:

```bash
MODEL_PATH=./outputs_final/best_model  # 모델 디렉토리 경로
HOST=0.0.0.0                           # 서버 바인드 주소
PORT=8000                              # 서버 포트
LOG_LEVEL=info                         # 로깅 레벨 (debug, info, warning, error)
CUDA_VISIBLE_DEVICES=0                 # 사용할 GPU ID
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # CUDA 메모리 설정
```

### 9.2 Docker 배포

#### 9.2.1 Docker 빌드

```bash
docker build -t cesco-inference:latest .
```

#### 9.2.2 Docker Compose 실행

```bash
docker-compose up -d
```

#### 9.2.3 docker-compose.yml 주요 설정

```yaml
services:
  cesco-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs_final:/app/model:ro  # 모델 볼륨 (읽기 전용)
      - ./logs:/app/logs               # 로그 볼륨
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 9.3 문제 해결

#### 9.3.1 모델 로딩 실패

**증상**: `FileNotFoundError: Model path does not exist`

**해결**:
1. `MODEL_PATH` 환경변수 확인
2. 모델 파일 존재 여부 확인:
   ```bash
   ls -la $MODEL_PATH/
   # adapter_config.json, adapter_model.safetensors 확인
   ```

#### 9.3.2 CUDA 메모리 부족

**증상**: `CUDA out of memory`

**해결**:
1. 배치 크기 줄이기
2. `max_new_tokens` 줄이기
3. 다른 프로세스의 GPU 사용 확인:
   ```bash
   nvidia-smi
   ```

#### 9.3.3 JSON 파싱 실패

**증상**: `parsed_response`가 `null`

**원인**:
- 모델이 유효하지 않은 JSON 생성
- Temperature가 너무 높음

**해결**:
1. `temperature`를 0.1 이하로 설정
2. `raw_response` 확인하여 출력 형식 검토
3. 학습 데이터 품질 검토
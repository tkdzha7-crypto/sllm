# 기존 고객 추천 시스템 가이드

기존 고객 추천(User Product Recommendation) 시스템은 고객의 위치, 업종, 건물 규모 등을 기반으로 유사한 고객 그룹을 식별하고, 해당 그룹에서 인기 있는 계약과 상품을 추천합니다. 이 문서는 추천 파이프라인의 데이터 흐름, 피처 명세, 테이블 구조, 배치 스케줄 등을 상세히 설명합니다.

---

## 1. 용어 정의 (Terminology)

| 용어 | 정의 |
|------|------|
| CCOD | 고객코드(Customer Code)로, 고객을 고유하게 식별하는 값입니다. |
| K-Means | 비지도 학습 클러스터링 알고리즘입니다. 고객을 유사한 그룹으로 분류합니다. |
| PCA | 주성분 분석(Principal Component Analysis)으로, 산업분류 4차원을 2차원으로 축소합니다. |
| 클러스터 | K-Means로 분류된 고객 그룹입니다. 총 10개 클러스터를 사용합니다. |
| 유사 고객 | 동일 클러스터 내에서 유클리드 거리가 가장 가까운 고객입니다. |
| snapshot_month | 스냅샷 월로, 데이터가 수집된 기준 월입니다. 파티션 키로 사용됩니다. |
| StandardScaler | 피처 정규화 방법입니다. 평균 0, 표준편차 1로 변환합니다. |
| LabelEncoder | 카테고리 변수를 정수로 변환하는 인코더입니다. |

---

## 2. ERD (Entity Relationship Diagram)

기존 고객 추천 시스템은 3개의 주요 테이블로 구성됩니다.

```
┌─────────────────────────────────────┐
│     source.user_monthly_features    │
│     (고객 월별 원본 데이터)           │
├─────────────────────────────────────┤
│ PK: (snapshot_id, snapshot_month)   │
│     CCOD                            │
│     user_information (JSONB)        │
│     contract_info (JSONB)           │
│     purchase_logs (JSONB)           │
│     interaction_history (JSONB)     │
└─────────────────┬───────────────────┘
                  │
                  │ K-Means 클러스터링
                  ▼
┌─────────────────────────────────────┐
│     analytics.user_recommendation   │
│     (추천 결과)                      │
├─────────────────────────────────────┤
│ PK: (id, snapshot_month)            │
│     CCOD                            │
│     rec_contract_1~3 (추천 계약)     │
│     rec_product_1~3 (추천 상품)      │
│     user_cluster (클러스터 ID)       │
│     sim_CCOD (유사 고객 코드)         │
└─────────────────┬───────────────────┘
                  │
                  │ 클러스터 프로필 집계
                  ▼
┌─────────────────────────────────────┐
│     analytics.cluster_profile       │
│     (클러스터 프로필)                   │
├─────────────────────────────────────┤
│ PK: (id, snapshot_month)            │
│     cluster_id                      │
│     cluster_size                    │
│     top_contracts (JSONB)           │
│     top_purchases (JSONB)           │
└─────────────────────────────────────┘
```

---

## 3. 데이터 흐름도 (Model Lineage)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       기존 고객 추천 파이프라인                                   │
└─────────────────────────────────────────────────────────────────────────────┘

[원천 데이터]
    │
    ├── RODB.CESCOEIS
    │   ├── TB_고객 (고객 기본정보)
    │   └── TB_신계약_마스타/상세 (계약정보)
    │
    └── BIDB.IEIS / CX_CDM
        └── DA_M_MYLAB_PROFIT_DAILY_ITEM (구매 이력)
    │
    ▼
┌─────────────────────────────────────┐
│ [1단계] 데이터 수집 (Ingest)         │
│ ingest_users()                      │
│ ingest_contracts()                  │
│ ingest_purchase_logs()              │
│ - 청크 단위(2000명) 처리             │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│ [2단계] 데이터 처리 (Process)        │
│ process_and_load_chunk()            │
│ - JSON 직렬화                       │
│ - 고객정보 병합                     │
└─────────────────┬───────────────────┘
                  │
                  ▼
     source.user_monthly_features
                  │
                  ▼
┌─────────────────────────────────────┐
│ [3단계] 추천 생성 (Load)             │
│ recommend_users_for_chunk()         │
│ - 피처 추출 (8개)                   │
│ - K-Means 클러스터 할당             │
│ - 유사 고객 탐색                    │
│ - Top-3 계약/상품 추천              │
└─────────────────┬───────────────────┘
                  │
                  ▼
     analytics.user_recommendation
                  │
                  ▼
┌─────────────────────────────────────┐
│ [4단계] 클러스터 프로필 업데이트      │
│ update_cluster_info()               │
│ - 클러스터별 통계 집계              │
└─────────────────┬───────────────────┘
                  │
                  ▼
     analytics.cluster_profile
```

**데이터 흐름 요약:**

```
RODB + BIDB ⇒ source.user_monthly_features
           ⇒ K-Means 클러스터링 (8개 피처)
           ⇒ analytics.user_recommendation
           ⇒ analytics.cluster_profile
```

---

## 4. 피처(Feature) 상세 명세

추천 시스템에서 사용하는 피처는 총 8개입니다.

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| 위도 | FLOAT | 고객 소재지 위도입니다. | user_information에서 추출합니다. |
| 경도 | FLOAT | 고객 소재지 경도입니다. | user_information에서 추출합니다. |
| 시도_encoded | INT | 시도 카테고리 인코딩입니다. | LabelEncoder로 변환합니다. (0~16) |
| 시군구_encoded | INT | 시군구 카테고리 인코딩입니다. | LabelEncoder로 변환합니다. (0~250) |
| 분류_PCA1 | FLOAT | PCA 변환된 산업분류 1입니다. | 대/중/소/세분류 4차원 → 2차원 변환입니다. |
| 분류_PCA2 | FLOAT | PCA 변환된 산업분류 2입니다. | 대/중/소/세분류 4차원 → 2차원 변환입니다. |
| 업태_encoded | INT | 업태 카테고리 인코딩입니다. | LabelEncoder로 변환합니다. (0~20) |
| 평균_면적_category | INT | 건물 규모 카테고리입니다. | 면적 구간화 (0~5)입니다. |

### 면적 구간화 기준

| 카테고리 | 면적 범위 |
|---------|----------|
| 0 | ≤22 m² |
| 1 | 22~50 m² |
| 2 | 50~258 m² |
| 3 | 258~1600 m² |
| 4 | 1600~4950 m² |
| 5 | >4950 m² |

### 산업분류 PCA 변환

산업분류는 4개의 카테고리 변수(대분류, 중분류, 소분류, 세분류)로 구성됩니다. LabelEncoder로 각각 정수로 변환한 후, PCA를 적용하여 2차원(분류_PCA1, 분류_PCA2)으로 축소합니다.

```
대분류 (encoded) ─┐
중분류 (encoded) ─┼─→ PCA ─→ 분류_PCA1
소분류 (encoded) ─┤        분류_PCA2
세분류 (encoded) ─┘
```

---

## 5. 엔티티(테이블) 상세 명세

### 5.1 source.user_monthly_features

고객 월별 원본 데이터를 저장하는 테이블입니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| snapshot_id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| CCOD | VARCHAR | 50 | YES | - | 고객코드입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| user_information | JSONB | - | YES | - | 고객 메타데이터입니다. |
| contract_info | JSONB | - | YES | - | 계약 정보입니다. JSON 배열로 저장합니다. |
| purchase_logs | JSONB | - | YES | - | 구매 이력입니다. JSON 배열로 저장합니다. |
| interaction_history | JSONB | - | YES | - | 상호작용 이력입니다. |

**user_information JSON 구조:**

```json
{
  "고객코드": "CUS001",
  "고객명": "ABC사",
  "유형대": "일반사업체",
  "유형중": "중분류",
  "대표자명": "김철수",
  "우편번호": "12345",
  "주소1": "서울특별시 강남구 테헤란로 123",
  "주소2": "1층",
  "업태": "음식점업",
  "종목": "한식당",
  "사업자번호": "111-22-33333",
  "등록일자": "2020-01-15"
}
```

### 5.2 analytics.user_recommendation

추천 결과를 저장하는 테이블입니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| created_at | TIMESTAMPTZ | - | NO | NOW() | 레코드 생성일시입니다. |
| CCOD | VARCHAR | 50 | NO | - | 고객코드입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| rec_contract_1 | VARCHAR | 255 | YES | - | 추천 계약 1순위입니다. |
| rec_contract_1_reason | TEXT | - | YES | - | 추천 계약 1순위 이유입니다. |
| rec_contract_2 | VARCHAR | 255 | YES | - | 추천 계약 2순위입니다. |
| rec_contract_2_reason | TEXT | - | YES | - | 추천 계약 2순위 이유입니다. |
| rec_contract_3 | VARCHAR | 255 | YES | - | 추천 계약 3순위입니다. |
| rec_contract_3_reason | TEXT | - | YES | - | 추천 계약 3순위 이유입니다. |
| rec_product_1 | VARCHAR | 255 | YES | - | 추천 상품 1순위입니다. |
| rec_product_1_reason | TEXT | - | YES | - | 추천 상품 1순위 이유입니다. |
| rec_product_2 | VARCHAR | 255 | YES | - | 추천 상품 2순위입니다. |
| rec_product_2_reason | TEXT | - | YES | - | 추천 상품 2순위 이유입니다. |
| rec_product_3 | VARCHAR | 255 | YES | - | 추천 상품 3순위입니다. |
| rec_product_3_reason | TEXT | - | YES | - | 추천 상품 3순위 이유입니다. |
| user_cluster | INT | 4 | YES | - | 할당된 클러스터 ID입니다. (0~9) |
| cluster_similarity | FLOAT | 8 | YES | - | 클러스터 내 유사도입니다. (0~1) |
| sim_CCOD | VARCHAR | 50 | YES | - | 유사 고객 코드입니다. |
| sim_user_name | VARCHAR | 255 | YES | - | 유사 고객명입니다. |
| sim_user_contracts | JSONB | - | YES | - | 유사 고객의 계약 리스트입니다. |
| sim_user_products | JSONB | - | YES | - | 유사 고객의 구매 상품 리스트입니다. |

**총 컬럼 수**: 22개

### 5.3 analytics.cluster_profile

클러스터 프로필을 저장하는 테이블입니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| created_at | TIMESTAMPTZ | - | NO | NOW() | 레코드 생성일시입니다. |
| cluster_id | INT | 4 | YES | - | 클러스터 ID입니다. (0~9) |
| cluster_size | INT | 4 | YES | - | 클러스터 내 고객 수입니다. |
| avg_contracts_num | FLOAT | 8 | YES | - | 평균 계약 개수입니다. |
| avg_purchase_num | FLOAT | 8 | YES | - | 평균 구매 개수입니다. |
| top_contracts | JSONB | - | YES | - | 상위 계약 리스트입니다. |
| top_purchases | JSONB | - | YES | - | 상위 구매 상품 리스트입니다. |
| top_business_type | JSONB | - | YES | - | 상위 업태 리스트입니다. |
| top_first_contract_code | JSONB | - | YES | - | 최초 계약 상위 리스트입니다. |
| contracts_distribution | JSONB | - | YES | - | 계약 수 분포입니다. |
| purchase_distribution | JSONB | - | YES | - | 구매 수 분포입니다. |

**총 컬럼 수**: 13개

---

## 6. 배치 스케줄 명세

| Flow 명 | Cron 표현식 | 실행 주기 | 시간대 |
|---------|------------|----------|--------|
| Analyze Current Users | `0 2 1 * *` | 매월 1일 새벽 2시 | Asia/Seoul |

**상세 스케줄**

| 항목 | 내용 |
|------|------|
| 실행 시점 | 매월 1일 02:00에 실행됩니다. 연간 12회 실행됩니다. |
| 데이터 범위 | 전체 활성 고객을 대상으로 합니다. |
| 처리 방식 | 2000명씩 청크 단위로 처리합니다. |
| 의존성 | RODB, BIDB 연결이 가능해야 합니다. |

**예상 소요 시간 (2025-12-01 기준)**
- 개별 Worker가 1개의 Chunk (2,000개의 datapoint) 처리에 약 00:15:00 (HH:MM:SS)이 소요됩니다.
- 총 20개의 Worker로 실행시킬 시, 20개의 Chunk (40,000개의 datapoint) 처리에 약 00:15:00 (HH:MM:SS)이 소요됩니다.
- 전체 Datapoint의 개수를 5M (5,000,000건의 datapoint) 기준, 15ms x 125 = 1,875분 (약 30시간) 가량이 소요됩니다.
- 해약 예측 모델과 함께 Inference 되어, 중복으로 발생하지 않습니다.

---

## 7. 코드 설명

### 7.1 주요 파일 위치

| 파일 경로 | 설명 |
|----------|------|
| `flows/analyze_current_user/main.py` | 추천 파이프라인을 정의합니다. Prefect Flow와 Task가 포함되어 있습니다. |
| `flows/analyze_current_user/active_check.py` | 계약 활성/해약 상태 판정 로직을 정의합니다. |
| `src/models/recommendation/recommender.py` | UserRecommender 클래스를 정의합니다. 추천 엔진 메인 로직입니다. |
| `src/models/recommendation/train.py` | K-Means 모델 학습 스크립트입니다. |
| `src/models/recommendation/category_mapper.py` | 표준산업코드 → 업종명 매핑을 정의합니다. |
| `src/models/recommendation/mapping_constants.py` | 업태/종목 → 분류 매핑 상수를 정의합니다. |
| `src/models/recommendation/simple_clustering_model_pca.pkl` | 학습된 K-Means 모델 파일입니다. |

### 7.2 파이프라인 함수 설명

| 함수명 | 파일 | 역할 |
|--------|------|------|
| `ingest_users()` | main.py | RODB에서 모든 고객코드를 조회합니다. |
| `ingest_contracts()` | main.py | RODB에서 계약 정보를 수집합니다. |
| `ingest_purchase_logs()` | main.py | BIDB에서 구매 로그를 수집합니다. |
| `process_and_load_chunk()` | main.py | 데이터를 정제하고 source.user_monthly_features에 적재합니다. |
| `recommend_users_for_chunk()` | main.py | UserRecommender로 추천을 생성합니다. |
| `update_cluster_info()` | main.py | 클러스터 프로필을 업데이트합니다. |
| `analyze_current_user()` | main.py | 전체 파이프라인을 실행하는 Prefect Flow입니다. |

### 7.3 추천 엔진 클래스 설명

| 메서드명 | 파일 | 역할 |
|---------|------|------|
| `__init__()` | recommender.py | 모델과 인코더를 로드합니다. |
| `run_inference()` | recommender.py | 피처 추출 → 클러스터 할당 → 추천 생성을 수행합니다. |
| `extract_sido()` | recommender.py | 주소에서 시도명을 추출합니다. |
| `extract_sigungu()` | recommender.py | 주소에서 시군구명을 추출합니다. |

### 7.4 모델 학습 함수 설명

| 함수명 | 파일 | 역할 |
|--------|------|------|
| `load_training_data()` | train.py | source.user_monthly_features에서 학습 데이터를 로드합니다. |
| `prepare_features()` | train.py | 8개 피처를 추출하고 전처리합니다. |
| `train_clustering_model()` | train.py | K-Means 모델을 학습합니다. |
| `analyze_clusters()` | train.py | 클러스터별 Top-5 계약/상품을 집계합니다. |
| `save_model()` | train.py | 모델을 pickle 파일로 저장합니다. |

---

## 8. 디렉토리 구조

```
cesco-deskroom/
├── flows/
│   └── analyze_current_user/
│       ├── main.py               # 추천 파이프라인을 정의합니다.
│       └── active_check.py       # 계약 상태 판정 로직을 정의합니다.
├── src/
│   └── models/
│       └── recommendation/
│           ├── recommender.py    # 추천 엔진 클래스를 정의합니다.
│           ├── train.py          # 모델 학습 스크립트입니다.
│           ├── category_mapper.py # 산업분류 매핑을 정의합니다.
│           ├── mapping_constants.py # 매핑 상수를 정의합니다.
│           └── simple_clustering_model_pca.pkl # 학습된 모델 파일입니다.
├── entities/
│   └── analytics/
│       ├── user_recommendation.py   # 추천 결과 엔티티를 정의합니다.
│       └── cluster_profile.py       # 클러스터 프로필 엔티티를 정의합니다.
├── deploy.py                     # Prefect Flow 배포 설정과 스케줄을 정의합니다.
└── init.sql                      # PostgreSQL 테이블 DDL을 정의합니다.
```

### 추천 생성 프로세스

```
1. 고객 정보에서 8개 피처 추출
   ↓
2. StandardScaler로 정규화
   ↓
3. K-Means.predict()로 클러스터 할당
   ↓
4. 동일 클러스터 내 유클리드 거리 계산
   ↓
5. 최단 거리 고객을 유사 고객으로 선택
   ↓
6. 클러스터 내 계약/상품 사용률 기반 Top-3 추천
   ↓
7. 추천 이유 생성 ("고객 세그먼트 내 72.5%가 이용중")
```
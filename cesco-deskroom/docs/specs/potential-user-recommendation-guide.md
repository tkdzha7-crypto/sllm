# 잠재 고객 추천 시스템 가이드

잠재 고객 추천(Potential User Product Recommendation) 시스템은 koDATA에서 추출한 CESCO 서비스를 이용하지 않는 신규 사업체를 대상으로, 기존 고객 클러스터와 유사한 그룹을 식별하고 해당 그룹에서 인기 있는 계약과 상품을 추천합니다.

이 문서는 추천 파이프라인의 데이터 흐름, 피처 명세, 테이블 구조, 배치 스케줄 등을 상세히 설명합니다.

---

## 1. 용어 정의 (Terminology)

| 용어 | 정의 |
|------|------|
| 잠재 고객 | CESCO 서비스를 이용하지 않는 신규 사업체입니다. KODATA에서 수집합니다. |
| BZNO | 사업자번호입니다. 잠재 고객을 고유하게 식별하는 값입니다. |
| KEDCD | 통계청 사업체 코드입니다. 한국기업데이터에서 부여한 사업체 고유 식별자입니다. |
| BZPL_CD | 사업장 코드입니다. 동일 사업체의 복수 사업장을 구분합니다. |
| BZPL_SEQ | 사업장 일련번호입니다. 사업장 코드와 함께 사업장을 고유하게 식별합니다. |
| KODATA | 사업자정보 데이터베이스입니다. 전국 사업체 정보를 제공합니다. |
| K-Means | 비지도 학습 클러스터링 알고리즘입니다. 기존 고객 추천과 동일한 모델을 재사용합니다. |
| 유사 고객 | 동일 클러스터 내에서 유클리드 거리가 가장 가까운 기존 CESCO 고객입니다. |
| snapshot_month | 스냅샷 월로, 데이터가 수집된 기준 월입니다. 파티션 키로 사용됩니다. |
| inference_mode | UserRecommender의 동작 모드입니다. 잠재 고객은 'potential_users' 모드를 사용합니다. |

---

## 2. ERD (Entity Relationship Diagram)

잠재 고객 추천 시스템은 2개의 주요 테이블로 구성됩니다.

```
┌─────────────────────────────────────┐
│       source.potential_user         │
│       (잠재 고객 원본 데이터)            │
├─────────────────────────────────────┤
│ PK: (id, snapshot_month)            │
│     BZNO (사업자번호)                  │ 
│     KEDCD, BZPL_CD, BZPL_SEQ        │
│     ENP_NM (상호명)                   │
│     SIDO, SIGUNGU                   │
│     LAT, LOT (위도, 경도)             │
│     BF_BZC_CD (표준산업코드)           │
│     BSZE_METR, LSZE_METR (면적)      │
└─────────────────┬───────────────────┘
                  │
                  │ K-Means 클러스터링
                  │ (기존 고객 모델 재사용)
                  ▼
┌───────────────────────────────────────────┐
│  analytics.potential_user_recommendation  │
│  (추천 결과)                                │
├───────────────────────────────────────────┤
│ PK: (id, snapshot_month)                  │
│     BZNO, KEDCD, BZPL_CD, BZPL_SEQ        │
│     rec_contract_1~3 (추천 계약)            │
│     rec_product_1~3 (추천 상품)             │
│     user_cluster (클러스터 ID)              │
│     sim_CCOD (유사 기존 고객 코드)            │
└───────────────────────────────────────────┘
```

**기존 고객 추천과의 차이점:**
- 잠재 고객은 CCOD(고객코드)가 없으므로 BZNO(사업자번호)를 식별자로 사용합니다.
- KEDCD, BZPL_CD, BZPL_SEQ 조합으로 사업장을 고유하게 식별합니다.
- 클러스터 프로필 테이블(analytics.cluster_profile)은 기존 고객 추천에서 생성된 것을 공유합니다.

---

## 3. 데이터 흐름도 (Model Lineage)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       잠재 고객 추천 파이프라인                                   │
└─────────────────────────────────────────────────────────────────────────────┘

[원천 데이터]
    │
    └── KODATA (사업자정보DB)
    │   ├── TB_KODATA_KED50D5 (사업자 기본정보: BZNO, KEDCD, BZPL_CD, BZPL_SEQ)
    │   ├── TB_KODATA_KED50PC (위치정보: SIDO, SIGUNGU, LAT, LOT, RDNM_ADDR)
    │   └── TB_KODATA_KED50D1 (상호명, 표준산업코드: ENP_NM, BF_BZC_CD)  
    └── source.industry_codes (사업체 분류코드와 분류체계 명 매핑 테이블)
      
    │
    ▼
┌─────────────────────────────────────┐
│ [1단계] 데이터 수집 (Ingest)         │
│ ingest_new_users()                  │
│ - KODATA_QUERY 실행                 │
│ - 기존 CESCO 고객 제외               │
│ - 필수 정보 완전성 검증              │
│ - source.potential_user에 저장      │
└─────────────────┬───────────────────┘
                  │
                  ▼
       source.potential_user
                  │
                  ▼
┌─────────────────────────────────────┐
│ [2단계] 추천 생성 (Process + Load)     │
│ recommend_users_for_chunk()         │
│ - 피처 추출 (8개)                      │
│ - K-Means 클러스터 할당                │
│ - 유사 기존 고객 탐색                   │
│ - Top-3 계약/상품 추천                 │
└─────────────────┬───────────────────┘
                  │
                  ▼
  analytics.potential_user_recommendation
```

**데이터 흐름 요약:**

```
KODATA ⇒ source.potential_user
       ⇒ K-Means 클러스터링 (8개 피처, 기존 고객 모델 재사용)
       ⇒ analytics.potential_user_recommendation
```

**기존 고객 추천과의 파이프라인 비교:**

| 항목 | 기존 고객 추천 | 잠재 고객 추천 |
|------|--------------|--------------|
| 파이프라인 단계 | 4단계 (Ingest → Process → Load → Update) | 2단계 (Ingest → Process+Load) |
| 원천 데이터 | RODB + BIDB | KODATA |
| 클러스터 프로필 업데이트 | 포함 | 미포함 (기존 프로필 사용) |

---

## 4. 피처(Feature) 상세 명세

잠재 고객 추천에서 사용하는 피처는 기존 고객 추천과 동일한 8개입니다.

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| 위도 | FLOAT | 사업체 소재지 위도입니다. | KODATA LAT 컬럼에서 추출합니다. |
| 경도 | FLOAT | 사업체 소재지 경도입니다. | KODATA LOT 컬럼에서 추출합니다. |
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

표준산업코드(BF_BZC_CD)를 4개의 카테고리 변수(대분류, 중분류, 소분류, 세분류)로 분리합니다. LabelEncoder로 각각 정수로 변환한 후, PCA를 적용하여 2차원(분류_PCA1, 분류_PCA2)으로 축소합니다.

```
대분류 (encoded) ─┐
중분류 (encoded) ─┼─→ PCA ─→ 분류_PCA1
소분류 (encoded) ─┤        분류_PCA2
세분류 (encoded) ─┘
```

---

## 5. 엔티티(테이블) 상세 명세

### 5.1 source.potential_user

잠재 고객 원본 데이터를 저장하는 테이블입니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| KEDCD | VARCHAR | 50 | YES | - | 통계청 사업체 코드입니다. |
| BZPL_CD | VARCHAR | 50 | YES | - | 사업장 코드입니다. |
| BZPL_SEQ | VARCHAR | 50 | YES | - | 사업장 일련번호입니다. |
| BZNO | VARCHAR | 50 | NO | - | 사업자번호입니다. |
| ENP_NM | VARCHAR | 255 | YES | - | 상호명입니다. |
| created_at | TIMESTAMPTZ | - | NO | NOW() | 레코드 생성일시입니다. |
| SIDO | VARCHAR | 100 | YES | - | 시도명입니다. |
| SIGUNGU | VARCHAR | 100 | YES | - | 시군구명입니다. |
| LAT | FLOAT | 8 | YES | - | 위도입니다. |
| LOT | FLOAT | 8 | YES | - | 경도입니다. |
| BZPL_NM | VARCHAR | 255 | YES | - | 사업자명입니다. |
| BF_BZC_CD | VARCHAR | 50 | YES | - | 표준산업코드입니다. |
| BSZE_METR | FLOAT | 8 | YES | - | 사업소 면적(m²)입니다. |
| LSZE_METR | FLOAT | 8 | YES | - | 건물 규모(m²)입니다. |
| RDNM_ADDR | VARCHAR | 500 | YES | - | 도로명 주소입니다. |

**총 컬럼 수**: 17개

**파티션 방법:**
- `snapshot_month`를 기준으로 월별 파티션을 생성합니다.
- 파이프라인 실행 시 `ensure_partition_exists()` 함수가 자동으로 파티션을 생성합니다.

### 5.2 analytics.potential_user_recommendation

잠재 고객 추천 결과를 저장하는 테이블입니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| enp_nm | VARCHAR | 255 | YES | - | 상호명입니다. |
| created_at | TIMESTAMPTZ | - | NO | NOW() | 레코드 생성일시입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| KEDCD | VARCHAR | 50 | YES | - | 통계청 사업체 코드입니다. |
| BZPL_CD | VARCHAR | 50 | YES | - | 사업장 코드입니다. |
| BZPL_SEQ | VARCHAR | 50 | YES | - | 사업장 일련번호입니다. |
| BZNO | VARCHAR | 50 | NO | - | 사업자번호입니다. |
| ENP_NP | VARCHAR | 255 | YES | - | 대표자명입니다. |
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
| sim_CCOD | VARCHAR | 50 | YES | - | 유사 기존 고객 코드입니다. |
| sim_user_name | VARCHAR | 255 | YES | - | 유사 고객명입니다. |
| sim_user_contracts | JSONB | - | YES | - | 유사 고객의 계약 리스트입니다. |
| sim_user_products | JSONB | - | YES | - | 유사 고객의 구매 상품 리스트입니다. |

**총 컬럼 수**: 27개

**파티션 전략:**
- `snapshot_month`를 기준으로 월별 파티션을 생성합니다.

---

## 6. 배치 스케줄 명세

| Flow 명 | Cron 표현식 | 실행 주기 | 시간대 |
|---------|------------|----------|--------|
| Analyze Potential Users | `0 2 1 * *` | 매월 1일 새벽 2시 | Asia/Seoul |

**상세 스케줄:**

| 항목 | 내용 |
|------|------|
| 실행 시점 | 매월 1일 02:00에 실행됩니다. 연간 12회 실행됩니다. |
| 데이터 범위 | KODATA의 전체 사업체 중 기존 CESCO 고객을 제외한 사업체를 대상으로 합니다. |
| 처리 방식 | 2000명씩 Chunk 단위로 처리합니다. |
| 의존성 | RODB(KODATA 테이블) 연결이 가능해야 합니다. |

**예상 소요 시간 (2025-12-01 기준)**
- 개별 Worker가 1개의 Chunk (2,000개의 datapoint) 처리에 약 00:00:30 (HH:MM:SS)이 소요됩니다.
- 새롭게 등록된 잠재 고객을 대상으로 Inference 수행함에 따라, 월 평균 1,000명의 신규 추가된 잠재 고객이 등록됩니다.
- 따라서, 월 평균 Inference 시간은 약 30초 가량입니다. (단, 등록되는 신규 고객 숫자 및 Ingestion 이슈에 따라 달라질 수 있습니다.)

---

## 7. 코드 설명

### 7.1 주요 파일 위치

| 파일 경로 | 설명 |
|----------|------|
| `flows/analyze_potential_user/main.py` | 잠재 고객 추천 파이프라인을 정의합니다. Prefect Flow와 Task가 포함되어 있습니다. |
| `flows/common/db_utils.py` | 파티션 생성, 벌크 삽입 유틸리티를 정의합니다. |
| `src/models/recommendation/recommender.py` | UserRecommender 클래스를 정의합니다. (inference_mode='potential_users') |
| `src/models/recommendation/category_mapper.py` | 표준산업코드 → 업종명 매핑을 정의합니다. |
| `src/models/recommendation/mapping_constants.py` | 업태/종목 → 분류 매핑 상수를 정의합니다. |
| `src/models/recommendation/simple_clustering_model_pca.pkl` | 학습된 K-Means 모델 파일입니다. 기존 고객 추천과 동일한 모델을 사용합니다. |

### 7.2 파이프라인 함수 설명

| 함수명 | 파일 | 역할 |
|--------|------|------|
| `ingest_new_users()` | main.py | KODATA에서 신규 사업체를 조회하고 source.potential_user에 저장합니다. |
| `recommend_users_for_chunk()` | main.py | UserRecommender로 추천을 생성합니다. |
| `analyze_potential_user()` | main.py | 전체 파이프라인을 실행하는 Prefect Flow입니다. |


### 7.3 KODATA 테이블 조인 관계

```
(kodata_query.txt) 쿼리문 참조
```

### 7.4 추천 엔진 클래스 설명

UserRecommender 클래스는 `inference_mode` 파라미터로 추론 대상 고객을 기존 고객과 잠재 고객을 구분합니다.

| 메서드명 | 파일 | 역할 |
|---------|------|------|
| `__init__(inference_mode='potential_users')` | recommender.py | 잠재 고객 모드로 모델과 인코더를 로드합니다. |
| `run_inference()` | recommender.py | 피처 추출 → 클러스터 할당 → 추천 생성을 수행합니다. |
| `extract_sido()` | recommender.py | 주소에서 시도명을 추출합니다. (잠재 고객은 SIDO 컬럼 직접 사용) |
| `extract_sigungu()` | recommender.py | 주소에서 시군구명을 추출합니다. (잠재 고객은 SIGUNGU 컬럼 직접 사용) |
| `get_industry_classification()` | recommender.py | 표준산업코드를 대/중/소/세분류로 변환합니다. |

**inference_mode별 차이점:**

| 항목 | current_user | potential_users |
|------|-------------|-----------------|
| 위도/경도 | user_information에서 추출 | LAT/LOT 컬럼 직접 사용 |
| 시도/시군구 | 주소1에서 파싱 | SIDO/SIGUNGU 컬럼 직접 사용 |
| 산업분류 | 신고객분류코드 사용 | BF_BZC_CD에서 변환 |
| 출력 테이블 | analytics.user_recommendation | analytics.potential_user_recommendation |

---

## 8. 디렉토리 구조

```
cesco-deskroom/
├── flows/
│   ├── analyze_potential_user/
│   │   └── main.py               # 잠재 고객 추천 파이프라인을 정의합니다.
│   └── common/
│       └── db_utils.py           # DB 유틸리티를 정의합니다.
├── src/
│   ├── dataloader.py             # CescoRodbConnection 클래스를 정의합니다.
│   └── models/
│       └── recommendation/
│           ├── recommender.py    # 추천 엔진 클래스를 정의합니다.
│           ├── category_mapper.py # 산업분류 매핑을 정의합니다.
│           ├── mapping_constants.py # 매핑 상수를 정의합니다.
│           └── simple_clustering_model_pca.pkl # 학습된 모델 파일입니다.
├── entities/
│   └── analytics/
│       └── potential_user_recommendation.py # 추천 결과 엔티티를 정의합니다.
├── deploy.py                     # Prefect Flow 배포 설정과 스케줄을 정의합니다.
└── init.sql                      # PostgreSQL 테이블 DDL을 정의합니다.
```

---

## 부록 A: 기존 고객 추천과의 비교

| 항목 | 기존 고객 추천 | 잠재 고객 추천 |
|------|--------------|--------------|
| **입력 데이터 소스** | CESCO 기존 고객 DB (CCOD) | KODATA 신규 사업체 (BZNO) |
| **알고리즘** | K-Means (10 클러스터) | K-Means (동일 모델 재사용) |
| **피처 수** | 8개 | 8개 (동일) |
| **inference_mode** | 'current_user' | 'potential_users' |
| **파이프라인 단계** | 4단계 | 2단계 |
| **원본 테이블** | source.user_monthly_features | source.potential_user |
| **출력 테이블** | analytics.user_recommendation | analytics.potential_user_recommendation |
| **식별자** | CCOD (고객코드) | BZNO (사업자번호) |
| **추가 식별자** | 없음 | KEDCD, BZPL_CD, BZPL_SEQ |
| **배치 스케줄** | 매월 1일 02:00 | 매월 1일 02:00 |
| **클러스터 프로필** | 직접 업데이트 | 기존 프로필 참조 |

---

## 부록 B: 추천 알고리즘 구조

### 추천 생성 프로세스

```
1. KODATA에서 잠재 고객 정보 수집
   ↓
2. 8개 피처 추출 (위도, 경도, 시도, 시군구, PCA1, PCA2, 업태, 면적)
   ↓
3. StandardScaler로 정규화 (기존 고객 학습 파라미터 사용)
   ↓
4. K-Means.predict()로 클러스터 할당
   ↓
5. 동일 클러스터 내 기존 CESCO 고객과 유클리드 거리 계산
   ↓
6. 최단 거리 고객을 유사 고객으로 선택
   ↓
7. 클러스터 내 계약/상품 사용률 기반 Top-3 추천
   ↓
8. 추천 이유 생성 ("고객 세그먼트 내 72.5%가 이용중")
```

### 모델 아티팩트 재사용

잠재 고객 추천은 기존 고객 추천에서 학습된 모델 파일(`simple_clustering_model_pca.pkl`)을 그대로 재사용합니다.

| 아티팩트 | 용도 |
|---------|------|
| kmeans | 클러스터 할당 |
| scaler | 피처 정규화 |
| pca | 산업분류 차원축소 |
| le_sido | 시도 인코딩 |
| le_sigungu | 시군구 인코딩 |
| le_대분류~le_세분류 | 산업분류 인코딩 |
| le_업태 | 업태 인코딩 |
| cluster_recommendations | 클러스터별 추천 계약/상품 |
| clustering_df | 유사 고객 탐색용 기존 고객 데이터 |

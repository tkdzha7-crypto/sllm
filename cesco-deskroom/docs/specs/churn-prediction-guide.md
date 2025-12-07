# 해약 예측 시스템 가이드

해약 예측(Churn Prediction) 시스템은 고객의 서비스 해지 가능성을 사전에 예측하여 선제적 관리를 가능하게 합니다. 이 문서는 해약 예측 파이프라인의 데이터 흐름, 피처 명세, 테이블 구조, 배치 스케줄 등을 상세히 설명합니다.

---

## 1. 주요 용어 정의 (Terminology)

| 용어 | 정의 |
|------|------|
| Churn (해약) | 고객이 계약을 해지하는 것을 의미합니다.|
| CCOD | 고객코드(Customer Code)로, 고객을 고유하게 식별하는 값입니다. |
| CSI | 고객만족도(Customer Satisfaction Index)로, 서비스 만족도 설문 결과입니다. |
| snapshot_month | 스냅샷 월로, 데이터가 수집된 기준 월입니다. 파티션 키로 사용됩니다. |
| 시간 윈도우 | 30일/60일/90일 단위로 최근 활동을 집계하는 기간입니다. |
| XGBoost | 해약 예측에 사용되는 머신러닝 모델입니다. 이진 분류(Binary Classification)를 수행합니다. |
| SHAP | 피처 기여도를 계산하는 알고리즘입니다. 각 피처가 예측에 미친 영향을 정량화합니다. |

---

## 2. ERD (Entity Relationship Diagram)

해약 예측 시스템은 3개의 주요 테이블로 구성됩니다.

```
┌─────────────────────────────────────┐
│     source.user_monthly_features    │
│     (고객 월별 원본 데이터)              │
├─────────────────────────────────────┤
│ PK: (snapshot_id, snapshot_month)   │
│     CCOD                            │
│     contract_info (JSONB)           │
│     user_information (JSONB)        │
│     purchase_logs (JSONB)           │
│     interaction_history (JSONB)     │
└─────────────────┬───────────────────┘
                  │
                  │ 피처 엔지니어링
                  ▼
┌─────────────────────────────────────┐
│   analytics.user_monthly_property   │
│   (71개 피처 테이블 컬럼)                │
├─────────────────────────────────────┤
│ PK: (ccod, created_at)              │
│     작업 피처 28개                   │
│     VOC 피처 15개                   │
│     CSI 피처 8개                    │
│     파생 피처 7개                   │
└─────────────────┬───────────────────┘
                  │
                  │ XGBoost 모델 추론
                  ▼
┌─────────────────────────────────────┐
│   analytics.user_churn_prediction   │
│   (해약 예측 결과)                   │
├─────────────────────────────────────┤
│ PK: (id, snapshot_month)            │
│     CCOD                            │
│     churn_prob (해약 확률)           │
│     churn_label (0: 유지, 1: 해약)   │
│     feature_1~10_* (기여도 상위 10)  │
└─────────────────────────────────────┘
```

---

## 3. 데이터 흐름도 (Model Lineage)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          해약 예측 파이프라인                                    │
└─────────────────────────────────────────────────────────────────────────────┘

[원천 데이터]
    │
    ├── RODB.CESCOEIS
    │   ├── TB_고객 (고객 기본정보)
    │   ├── TB_신계약_마스타/상세 (계약정보)
    │   └── TB_VOC_Master (VOC)
    │
    └── BIDB.IEIS / CX_CDM
        ├── TB_WR_TOWR (작업 이력)
        ├── DA_M_SURV_DATA (CSI 설문)
        └── DA_M_MYLAB_PROFIT_DAILY_ITEM (구매)
    │
    ▼
┌─────────────────────────────────────┐
│ [1단계] 데이터 수집                  │
│ ingest_users()                      │
│ ingest_contracts()                  │
│ ingest_work_logs()                  │
│ ingest_purchase_logs()              │
│ - 청크 단위(2000명) 처리             │
└─────────────────┬───────────────────┘
                  │
                  ▼
     source.user_monthly_features
                  │
                  ▼
┌─────────────────────────────────────┐
│ [2단계] 피처 엔지니어링              │
│ extract_work_features()             │
│ extract_voc_features()              │
│ add_granular_features()             │
│ - 71개 피처 생성                    │
└─────────────────┬───────────────────┘
                  │
                  ▼
    analytics.user_monthly_property
                  │
                  ▼
┌─────────────────────────────────────┐
│ [3단계] 해약 예측                    │
│ predict_churn_for_chunk()           │
│ - XGBoost 모델 추론                 │
│ - 임계값: 0.5                      │
│ - SHAP 기여도 분석                  │
└─────────────────┬───────────────────┘
                  │
                  ▼
    analytics.user_churn_prediction
```

**데이터 흐름 요약:**

```
RODB + BIDB ⇒ source.user_monthly_features
           ⇒ 피처 엔지니어링 (71개)
           ⇒ analytics.user_monthly_property
           ⇒ XGBoost 모델
           ⇒ analytics.user_churn_prediction
```

---

## 4. 피처(Feature) 상세 명세

해약 예측에서 사용하는 피처는 총 71개입니다. (훈련 후 3개 제거하여 68개 사용)

### 4.1 작업 피처 (26개)

#### 전체 기간 지표 (5개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| unique_work_types | INT | 총 몇 가지 유형의 작업을 수행했는지를 의미합니다. | 작업명의 고유 개수입니다. |
| cancelled_work | INT | 총 취소된 작업 수입니다. | 취소여부==1인 작업 건수입니다. |
| cancellation_rate | FLOAT | 전체 작업 중 취소된 작업의 비율을 의미합니다. | cancelled_work / work_count입니다. |
| confirmation_rate | FLOAT | 전체 작업 중 확정된 작업의 비율을 의미합니다. | confirmed_work / work_count입니다. |
| avg_services_per_work | FLOAT | 작업당 평균 서비스 수입니다. | 총 서비스 수 / work_count입니다. |

cf. 서비스란, 하나의 작업 내에 수행된 세분화 된 과업으로 정의한 개념입니다. (참조 소스코드: feature_engineering.py)

#### 30일 윈도우 (7개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| work_last_30d | INT | 최근 30일 총 작업 수입니다. | `(last_date - 작업일자).days <= 30`인 작업 건수입니다. last_date는 해당 고객의 가장 최근 작업일자입니다. |
| confirmed_30d | INT | 최근 30일 확정 작업 수입니다. | 최근 30일 내 확정여부==True인 작업 건수입니다. |
| cancelled_30d | INT | 최근 30일 총 취소 작업 수입니다. | 최근 30일 내 취소여부==True인 작업 건수입니다. |
| services_30d | INT | 최근 30일 총 서비스 수입니다. | 최근 30일 내 각 작업의 서비스내역 배열 길이를 합산한 값입니다. |
| work_types_30d | INT | 최근 30일 작업 유형 수입니다. | 최근 30일 내 고유한 작업유형의 개수입니다. |
| cancellation_rate_30d | FLOAT | 최근 30일 취소 비율입니다. | cancelled_30d / work_last_30d입니다. work_last_30d가 0인 경우 0입니다. |
| confirmation_rate_30d | FLOAT | 최근 30일 확정 비율입니다. | confirmed_30d / work_last_30d입니다. work_last_30d가 0인 경우 0입니다. |

#### 60일 윈도우 (7개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| work_last_60d | INT | 최근 60일 작업 수입니다. | `(last_date - 작업일자).days <= 60`인 작업 건수입니다. |
| confirmed_60d | INT | 최근 60일 확정 수입니다. | 최근 60일 내 확정여부==True인 작업 건수입니다. |
| cancelled_60d | INT | 최근 60일 취소 수입니다. | 최근 60일 내 취소여부==True인 작업 건수입니다. |
| services_60d | INT | 최근 60일 서비스 수입니다. | 최근 60일 내 각 작업의 서비스내역 배열 길이를 합산한 값입니다. |
| work_types_60d | INT | 최근 60일 작업 유형 수입니다. | 최근 60일 내 고유한 작업유형의 개수입니다. |
| cancellation_rate_60d | FLOAT | 최근 60일 취소 비율입니다. | cancelled_60d / work_last_60d입니다. work_last_60d가 0인 경우 0입니다. |
| confirmation_rate_60d | FLOAT | 최근 60일 확정 비율입니다. | confirmed_60d / work_last_60d입니다. work_last_60d가 0인 경우 0입니다. |

#### 90일 윈도우 (7개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| work_last_90d | INT | 최근 90일 작업 수입니다. | `(last_date - 작업일자).days <= 90`인 작업 건수입니다. |
| confirmed_90d | INT | 최근 90일 확정 수입니다. | 최근 90일 내 확정여부==True인 작업 건수입니다. |
| cancelled_90d | INT | 최근 90일 취소 수입니다. | 최근 90일 내 취소여부==True인 작업 건수입니다. |
| services_90d | INT | 최근 90일 서비스 수입니다. | 최근 90일 내 각 작업의 서비스내역 배열 길이를 합산한 값입니다. |
| work_types_90d | INT | 최근 90일 작업 유형 수입니다. | 최근 90일 내 고유한 작업유형의 개수입니다. |
| cancellation_rate_90d | FLOAT | 최근 90일 취소 비율입니다. | cancelled_90d / work_last_90d입니다. work_last_90d가 0인 경우 0입니다. |
| confirmation_rate_90d | FLOAT | 최근 90일 확정 비율입니다. | confirmed_90d / work_last_90d입니다. work_last_90d가 0인 경우 0입니다. |

### 4.2 VOC 피처 (16개)

#### 전체 기간 지표 (4개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| num_interactions | INT | 총 상호작용 수입니다. | interaction_history 배열의 길이입니다. |
| voc_count | INT | VOC(고객의견) 수입니다. | event_type == "VOC"인 상호작용 건수입니다. |
| non_voc_count | INT | 비VOC 상호작용 수입니다. | num_interactions - voc_count입니다. |
| voc_ratio | FLOAT | VOC 비율입니다. | voc_count / num_interactions입니다. num_interactions가 0인 경우 0입니다. |

cf. 상호작용이란, 작업 이력, VOC 등 고객과의 모든 Interaction을 의미합니다.

#### 30일 윈도우 (4개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| interactions_30d | INT | 최근 30일 상호작용 수입니다. | `(last_date - event_date).days <= 30`인 상호작용 건수입니다. last_date는 해당 고객의 가장 최근 상호작용 일자입니다. |
| voc_30d | INT | 최근 30일 VOC 수입니다. | 최근 30일 내 event_type == "VOC"인 건수입니다. |
| non_voc_30d | INT | 최근 30일 비VOC 수입니다. | interactions_30d - voc_30d입니다. |
| voc_ratio_30d | FLOAT | 최근 30일 VOC 비율입니다. | voc_30d / interactions_30d입니다. interactions_30d가 0인 경우 0입니다. |

#### 60일 윈도우 (4개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| interactions_60d | INT | 최근 60일 상호작용 수입니다. | `(last_date - event_date).days <= 60`인 상호작용 건수입니다. |
| voc_60d | INT | 최근 60일 VOC 수입니다. | 최근 60일 내 event_type == "VOC"인 건수입니다. |
| non_voc_60d | INT | 최근 60일 비VOC 수입니다. | interactions_60d - voc_60d입니다. |
| voc_ratio_60d | FLOAT | 최근 60일 VOC 비율입니다. | voc_60d / interactions_60d입니다. interactions_60d가 0인 경우 0입니다. |

#### 90일 윈도우 (4개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| interactions_90d | INT | 최근 90일 상호작용 수입니다. | `(last_date - event_date).days <= 90`인 상호작용 건수입니다. |
| voc_90d | INT | 최근 90일 VOC 수입니다. | 최근 90일 내 event_type == "VOC"인 건수입니다. |
| non_voc_90d | INT | 최근 90일 비VOC 수입니다. | interactions_90d - voc_90d입니다. |
| voc_ratio_90d | FLOAT | 최근 90일 VOC 비율입니다. | voc_90d / interactions_90d입니다. interactions_90d가 0인 경우 0입니다. |

### 4.3 CSI (고객 만족) 피처 (8개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| avg_csi_score | FLOAT | 전체 평균 CSI 점수입니다. | 작업이력 내 서비스_만족도.평균_CSI_점수 값들의 평균입니다. CSI 점수가 없는 경우 0입니다. |
| csi_score_30d | FLOAT | 최근 30일 평균 CSI 점수입니다. | `(last_date - 작업일자).days <= 30`인 작업의 CSI 점수 평균입니다. 해당 기간 CSI 점수가 없는 경우 0입니다. |
| csi_score_60d | FLOAT | 최근 60일 평균 CSI 점수입니다. | `(last_date - 작업일자).days <= 60`인 작업의 CSI 점수 평균입니다. 해당 기간 CSI 점수가 없는 경우 0입니다. |
| csi_score_90d | FLOAT | 최근 90일 평균 CSI 점수입니다. | `(last_date - 작업일자).days <= 90`인 작업의 CSI 점수 평균입니다. 해당 기간 CSI 점수가 없는 경우 0입니다. |
| csi_survey_count | INT | 전체 설문 응답 수입니다. | 서비스_만족도.평균_CSI_점수가 존재하는 설문 응답의 총 개수입니다. |
| csi_survey_count_30d | INT | 최근 30일 설문 응답 수입니다. | 최근 30일 내 CSI 점수가 있는 설문 응답 개수입니다. |
| csi_survey_count_60d | INT | 최근 60일 설문 응답 수입니다. | 최근 60일 내 CSI 점수가 있는 설문 응답 개수입니다. |
| csi_survey_count_90d | INT | 최근 90일 설문 응답 수입니다. | 최근 90일 내 CSI 점수가 있는 설문 응답 개수입니다. |

### 4.4 파생 피처 (7개)

| 피처명 | 데이터 타입 | 설명 | 계산 방식 |
|--------|------------|------|----------|
| recent_90d_activity_ratio | FLOAT | 최근 90일 활동 비율입니다. | work_last_90d / work_count입니다. |
| recent_30d_activity_ratio | FLOAT | 최근 30일 활동 비율입니다. | work_last_30d / work_count입니다. |
| activity_density | FLOAT | 활동 밀도입니다. | work_count / (work_span_days + 1)입니다. |
| confirmation_rate_change | FLOAT | 확정 비율 변화입니다. | confirmation_rate_30d - confirmation_rate입니다. |
| cancellation_rate_change | FLOAT | 취소 비율 변화입니다. | cancellation_rate_30d - cancellation_rate입니다. |
| voc_to_work_ratio | FLOAT | VOC 대 작업 비율입니다. | voc_count / work_count입니다. |
| recent_30d_to_90d_ratio | FLOAT | 30일 대 90일 비율입니다. | work_last_30d / work_last_90d입니다. |

---

## 5. 엔티티(테이블) 상세 명세

### 5.1 source.user_monthly_features.py

고객 월별 원본 데이터를 저장하는 테이블 구조를 의미합니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| snapshot_id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| CCOD | VARCHAR | 50 | YES | - | 고객코드입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| contract_info | JSON | - | YES | - | 계약 정보입니다. JSON 배열로 저장합니다. |
| user_information | JSON | - | YES | - | 고객 메타데이터입니다. |
| purchase_logs | JSON | - | YES | - | 구매 이력입니다. JSON 배열로 저장합니다. |
| interaction_history | JSON | - | YES | - | 상호작용 이력입니다. VOC, 업무일지 등을 포함합니다. |

### 5.2 analytics.user_monthly_property

58개 피처와 1개의 created_at 컬럼, 총 59개의 피처를 정의하는 테이블입니다. 파티션 키는 `created_at`입니다.

| 컬럼명 | 데이터 타입 | Nullable | 설명 |
|--------|------------|----------|------|
| ccod | VARCHAR(255) | NO | 고객코드입니다. Primary Key입니다. |
| created_at | TIMESTAMP | NO | 생성일시입니다. Primary Key, 파티션 키입니다. 자동으로 생성되는 값으로, 해당 소스코드에는 없습니다. |
| unique_work_types | INTEGER | YES | 작업 유형 다양성입니다. |
| cancelled_work | INTEGER | YES | 취소된 작업 수입니다. |
| cancellation_rate | FLOAT | YES | 전체 취소 비율입니다. |
| confirmation_rate | FLOAT | YES | 전체 확정 비율입니다. |
| avg_services_per_work | FLOAT | YES | 작업당 평균 서비스 수입니다. |
| work_last_30d | INTEGER | YES | 최근 30일 작업 수입니다. |
| confirmed_30d | INTEGER | YES | 최근 30일 확정 수입니다. |
| cancelled_30d | INTEGER | YES | 최근 30일 취소 수입니다. |
| services_30d | INTEGER | YES | 최근 30일 서비스 수입니다. |
| work_types_30d | INTEGER | YES | 최근 30일 작업 유형 수입니다. |
| cancellation_rate_30d | FLOAT | YES | 최근 30일 취소 비율입니다. |
| confirmation_rate_30d | FLOAT | YES | 최근 30일 확정 비율입니다. |
| work_last_60d | INTEGER | YES | 최근 60일 작업 수입니다. |
| confirmed_60d | INTEGER | YES | 최근 60일 확정 수입니다. |
| cancelled_60d | INTEGER | YES | 최근 60일 취소 수입니다. |
| services_60d | INTEGER | YES | 최근 60일 서비스 수입니다. |
| work_types_60d | INTEGER | YES | 최근 60일 작업 유형 수입니다. |
| cancellation_rate_60d | FLOAT | YES | 최근 60일 취소 비율입니다. |
| confirmation_rate_60d | FLOAT | YES | 최근 60일 확정 비율입니다. |
| work_last_90d | INTEGER | YES | 최근 90일 작업 수입니다. |
| confirmed_90d | INTEGER | YES | 최근 90일 확정 수입니다. |
| cancelled_90d | INTEGER | YES | 최근 90일 취소 수입니다. |
| services_90d | INTEGER | YES | 최근 90일 서비스 수입니다. |
| work_types_90d | INTEGER | YES | 최근 90일 작업 유형 수입니다. |
| cancellation_rate_90d | FLOAT | YES | 최근 90일 취소 비율입니다. |
| confirmation_rate_90d | FLOAT | YES | 최근 90일 확정 비율입니다. |
| avg_csi_score | FLOAT | YES | 전체 평균 CSI 점수입니다. |
| csi_score_30d | FLOAT | YES | 최근 30일 평균 CSI 점수입니다. |
| csi_score_60d | FLOAT | YES | 최근 60일 평균 CSI 점수입니다. |
| csi_score_90d | FLOAT | YES | 최근 90일 평균 CSI 점수입니다. |
| csi_survey_count | INTEGER | YES | 전체 설문 응답 수입니다. |
| csi_survey_count_30d | INTEGER | YES | 최근 30일 설문 응답 수입니다. |
| csi_survey_count_60d | INTEGER | YES | 최근 60일 설문 응답 수입니다. |
| csi_survey_count_90d | INTEGER | YES | 최근 90일 설문 응답 수입니다. |
| num_interactions | INTEGER | YES | 총 상호작용 수입니다. |
| voc_count | INTEGER | YES | VOC 수입니다. |
| non_voc_count | INTEGER | YES | 비VOC 상호작용 수입니다. |
| voc_ratio | FLOAT | YES | VOC 비율입니다. |
| interactions_30d | INTEGER | YES | 최근 30일 상호작용 수입니다. |
| voc_30d | INTEGER | YES | 최근 30일 VOC 수입니다. |
| non_voc_30d | INTEGER | YES | 최근 30일 비VOC 수입니다. |
| voc_ratio_30d | FLOAT | YES | 최근 30일 VOC 비율입니다. |
| interactions_60d | INTEGER | YES | 최근 60일 상호작용 수입니다. |
| voc_60d | INTEGER | YES | 최근 60일 VOC 수입니다. |
| non_voc_60d | INTEGER | YES | 최근 60일 비VOC 수입니다. |
| voc_ratio_60d | FLOAT | YES | 최근 60일 VOC 비율입니다. |
| interactions_90d | INTEGER | YES | 최근 90일 상호작용 수입니다. |
| voc_90d | INTEGER | YES | 최근 90일 VOC 수입니다. |
| non_voc_90d | INTEGER | YES | 최근 90일 비VOC 수입니다. |
| voc_ratio_90d | FLOAT | YES | 최근 90일 VOC 비율입니다. |
| recent_90d_activity_ratio | FLOAT | YES | 최근 90일 활동 비율입니다. |
| recent_30d_activity_ratio | FLOAT | YES | 최근 30일 활동 비율입니다. |
| activity_density | FLOAT | YES | 활동 밀도입니다. |
| confirmation_rate_change | FLOAT | YES | 확정 비율 변화입니다. |
| cancellation_rate_change | FLOAT | YES | 취소 비율 변화입니다. |
| voc_to_work_ratio | FLOAT | YES | VOC 대 작업 비율입니다. |
| recent_30d_to_90d_ratio | FLOAT | YES | 30일 대 90일 비율입니다. |

### 5.3 analytics.user_churn_prediction

해약 예측 결과에 대한 총 64개의 추론 결과와 2개의 로깅 컬럼을 정의하는 테이블입니다. (created_at, id는 로깅 컬럼으로 파일에는 포함되어있지 않습니다.)

**기본 컬럼 (6개, 4개의 추론 결과 컬럼 + 2개의 created_at, id 로깅 컬럼)**

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| id | BIGSERIAL | 8 | NO | auto | Primary Key입니다. |
| created_at | TIMESTAMPTZ | - | NO | NOW() | 레코드 생성일시입니다. |
| snapshot_month | DATE | - | NO | - | 스냅샷 월입니다. 파티션 키입니다. |
| CCOD | VARCHAR | 50 | YES | - | 고객코드입니다. |
| churn_prob | FLOAT | 8 | YES | - | 해약 확률입니다. 0~1 범위입니다. |
| churn_label | INT | 4 | YES | - | 해약 여부입니다. 0: 유지, 1: 해약입니다. |

**피처 기여도 컬럼 (60개 = 10세트 × 6컬럼)**

기여도 1~10까지 동일한 구조가 반복됩니다. 아래는 전체 컬럼 목록입니다.

| 컬럼명 | 데이터 타입 | 길이 | Nullable | 기본값 | 설명 |
|--------|------------|------|----------|--------|------|
| feature_1_kor | VARCHAR | 50 | YES | - | 기여도 1위 피처명(한글)입니다. |
| feature_1_value | FLOAT | 8 | YES | - | 기여도 1위 고객 피처값입니다. |
| feature_1_normal_average | FLOAT | 8 | YES | - | 기여도 1위 피처에 대한 비해약 고객 평균입니다. |
| feature_1_churn_average | FLOAT | 8 | YES | - | 기여도 1위 피처에 대한 해약 고객 평균입니다. |
| feature_1_contrib | FLOAT | 8 | YES | - | 기여도 1위 피처의 기여도 값입니다. |
| feature_1_contrib_label | VARCHAR | 50 | YES | - | 기여도 1위 영향도입니다. 높음/보통/낮음입니다. |
| feature_2_kor | VARCHAR | 50 | YES | - | 기여도 2위 피처명(한글)입니다. |
| feature_2_value | FLOAT | 8 | YES | - | 기여도 2위 고객 피처값입니다. |
| feature_2_normal_average | FLOAT | 8 | YES | - | 기여도 2위 피처의 비해약 고객 평균입니다. |
| feature_2_churn_average | FLOAT | 8 | YES | - | 기여도 2위 피처의 해약 고객 평균입니다. |
| feature_2_contrib | FLOAT | 8 | YES | - | 기여도 2위 피처의 기여도 값입니다. |
| feature_2_contrib_label | VARCHAR | 50 | YES | - | 기여도 2위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_3_kor | VARCHAR | 50 | YES | - | 기여도 3위 피처명(한글)입니다. |
| feature_3_value | FLOAT | 8 | YES | - | 기여도 3위 고객 피처값입니다. |
| feature_3_normal_average | FLOAT | 8 | YES | - | 기여도 3위 피처의 비해약 고객 평균입니다. |
| feature_3_churn_average | FLOAT | 8 | YES | - | 기여도 3위 피처의 해약 고객 평균입니다. |
| feature_3_contrib | FLOAT | 8 | YES | - | 기여도 3위 피처의 기여도 값입니다. |
| feature_3_contrib_label | VARCHAR | 50 | YES | - | 기여도 3위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_4_kor | VARCHAR | 50 | YES | - | 기여도 4위 피처명(한글)입니다. |
| feature_4_value | FLOAT | 8 | YES | - | 기여도 4위 고객 피처값입니다. |
| feature_4_normal_average | FLOAT | 8 | YES | - | 기여도 4위 피처의 비해약 고객 평균입니다. |
| feature_4_churn_average | FLOAT | 8 | YES | - | 기여도 4위 피처의 해약 고객 평균입니다. |
| feature_4_contrib | FLOAT | 8 | YES | - | 기여도 4위 피처의 기여도 값입니다. |
| feature_4_contrib_label | VARCHAR | 50 | YES | - | 기여도 4위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_5_kor | VARCHAR | 50 | YES | - | 기여도 5위 피처명(한글)입니다. |
| feature_5_value | FLOAT | 8 | YES | - | 기여도 5위 고객 피처값입니다. |
| feature_5_normal_average | FLOAT | 8 | YES | - | 기여도 5위 피처의 비해약 고객 평균입니다. |
| feature_5_churn_average | FLOAT | 8 | YES | - | 기여도 5위 피처의 해약 고객 평균입니다. |
| feature_5_contrib | FLOAT | 8 | YES | - | 기여도 5위 피처의 기여도 값입니다. |
| feature_5_contrib_label | VARCHAR | 50 | YES | - | 기여도 5위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_6_kor | VARCHAR | 50 | YES | - | 기여도 6위 피처명(한글)입니다. |
| feature_6_value | FLOAT | 8 | YES | - | 기여도 6위 고객 피처값입니다. |
| feature_6_normal_average | FLOAT | 8 | YES | - | 기여도 6위 피처의 비해약 고객 평균입니다. |
| feature_6_churn_average | FLOAT | 8 | YES | - | 기여도 6위 피처의 해약 고객 평균입니다. |
| feature_6_contrib | FLOAT | 8 | YES | - | 기여도 6위 피처의 기여도 값입니다. |
| feature_6_contrib_label | VARCHAR | 50 | YES | - | 기여도 6위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_7_kor | VARCHAR | 50 | YES | - | 기여도 7위 피처명(한글)입니다. |
| feature_7_value | FLOAT | 8 | YES | - | 기여도 7위 고객 피처값입니다. |
| feature_7_normal_average | FLOAT | 8 | YES | - | 기여도 7위 피처의 비해약 고객 평균입니다. |
| feature_7_churn_average | FLOAT | 8 | YES | - | 기여도 7위 피처의 해약 고객 평균입니다. |
| feature_7_contrib | FLOAT | 8 | YES | - | 기여도 7위 피처의 기여도 값입니다. |
| feature_7_contrib_label | VARCHAR | 50 | YES | - | 기여도 7위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_8_kor | VARCHAR | 50 | YES | - | 기여도 8위 피처명(한글)입니다. |
| feature_8_value | FLOAT | 8 | YES | - | 기여도 8위 고객 피처값입니다. |
| feature_8_normal_average | FLOAT | 8 | YES | - | 기여도 8위 피처의 비해약 고객 평균입니다. |
| feature_8_churn_average | FLOAT | 8 | YES | - | 기여도 8위 피처의 해약 고객 평균입니다. |
| feature_8_contrib | FLOAT | 8 | YES | - | 기여도 8위 피처의 기여도 값입니다. |
| feature_8_contrib_label | VARCHAR | 50 | YES | - | 기여도 8위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_9_kor | VARCHAR | 50 | YES | - | 기여도 9위 피처명(한글)입니다. |
| feature_9_value | FLOAT | 8 | YES | - | 기여도 9위 고객 피처값입니다. |
| feature_9_normal_average | FLOAT | 8 | YES | - | 기여도 피처의 9위 비해약 고객 평균입니다. |
| feature_9_churn_average | FLOAT | 8 | YES | - | 기여도 9위 피처의 해약 고객 평균입니다. |
| feature_9_contrib | FLOAT | 8 | YES | - | 기여도 9위 피처의 기여도 값입니다. |
| feature_9_contrib_label | VARCHAR | 50 | YES | - | 기여도 9위 피처의 영향도입니다. 높음/보통/낮음입니다. |
| feature_10_kor | VARCHAR | 50 | YES | - | 기여도 10위 피처명(한글)입니다. |
| feature_10_value | FLOAT | 8 | YES | - | 기여도 10위 고객 피처값입니다. |
| feature_10_normal_average | FLOAT | 8 | YES | - | 기여도 10위 피처의 비해약 고객 평균입니다. |
| feature_10_churn_average | FLOAT | 8 | YES | - | 기여도 10위 피처의 해약 고객 평균입니다. |
| feature_10_contrib | FLOAT | 8 | YES | - | 기여도 10위 피처의 기여도 값입니다. |
| feature_10_contrib_label | VARCHAR | 50 | YES | - | 기여도 10위 피처의 영향도입니다. 높음/보통/낮음입니다. |

**총 컬럼 수**: 66개 (로깅 2개 + 기본 4개 + 피처별 6개 × 10개)

---

## 6. 배치 스케줄 명세

| Flow 명 | Cron 표현식 | 실행 주기 | 시간대 |
|---------|------------|----------|--------|
| Analyze Current Users | `0 2 1 * *` | 매월 1일 새벽 2시 | Asia/Seoul Timezone |

**상세 스케줄**

| 항목 | 내용 |
|------|------|
| 실행 시점 | 매월 1일 02:00에 실행됩니다. 즉, 연간 12회 실행됩니다. |
| 데이터 범위 | 전체 활성 고객을 대상으로 합니다. |
| 처리 방식 | 2000명씩 청크 단위로 처리합니다. |
| 의존성 | RODB, BIDB 연결이 가능해야 합니다. |

**예상 소요 시간 (2025-12-01 기준)**
- 개별 Worker가 1개의 Chunk (2,000개의 datapoint) 처리에 약 00:15:00 (HH:MM:SS)이 소요됩니다.
- 총 20개의 Worker로 실행시킬 시, 20개의 Chunk (40,000개의 datapoint) 처리에 약 00:15:00 (HH:MM:SS)이 소요됩니다.
- 전체 Datapoint의 개수를 5M (5,000,000건의 datapoint) 기준, 15ms x 125 = 1,875분 (약 30시간) 가량이 소요됩니다.
- 기존 고객 대상 제품 추천 모델과 함께 Inference 되어, 중복으로 발생하지 않습니다.

---

## 7. 코드 설명

### 7.1 주요 파일 위치

| 파일 경로 | 설명 |
|----------|------|
| `flows/analyze_current_user/main.py` | 해약 예측 파이프라인을 정의합니다. Prefect Flow와 Task가 포함되어 있습니다. |
| `flows/analyze_current_user/feature_engineering.py` | 로깅 컬럼을 제외한 57개의 피처 계산 로직을 정의합니다. |
| `flows/analyze_current_user/active_check.py` | 활성 고객 필터링 로직을 정의합니다. |
| `src/models/churn_prediction/predictor.py` | XGBoost 모델 추론 클래스를 정의합니다. |
| `src/models/churn_prediction/train.py` | 모델 Training 스크립트입니다. |
| `src/models/churn_model_with_csi.pkl` | Training된 XGBoost 모델 파일입니다. |
| `src/models/training_averages_with_csi.pkl` | Training 통계(해약/비해약 평균)입니다. |

### 7.2 파이프라인 함수 설명

| 함수명 | 파일 | 역할 |
|--------|------|------|
| `ingest_users()` | main.py | RODB에서 모든 고객코드를 조회합니다. |
| `ingest_contracts()` | main.py | RODB에서 계약 정보를 수집합니다. |
| `ingest_work_logs()` | main.py | BIDB에서 작업 이력을 수집합니다. |
| `ingest_purchase_logs()` | main.py | BIDB에서 구매 로그를 수집합니다. |
| `process_and_load_chunk()` | main.py | Chunk 단위로 데이터를 정제하고 source.user_monthly_features에 적재합니다. |
| `predict_churn_for_chunk()` | main.py | Chunk 단위로 ChurnPredictor로 해약 예측을 실행합니다. |
| `analyze_current_user()` | main.py | 전체 파이프라인을 실행하는 Prefect Flow입니다. |

### 7.3 피처 엔지니어링 함수 설명

| 함수명 | 파일 | 역할 |
|--------|------|------|
| `extract_work_features()` | feature_engineering.py | 작업 이력에서 37개 피처를 추출합니다. |
| `extract_voc_features()` | feature_engineering.py | 상호작용 이력에서 16개 피처를 추출합니다. |
| `add_granular_features()` | feature_engineering.py | 7개 파생 피처를 계산합니다. |

### 7.4 예측 클래스 설명

| 메서드명 | 파일 | 역할 |
|---------|------|------|
| `__init__()` | predictor.py | 모델과 훈련 통계를 로드합니다. |
| `run_inference()` | predictor.py | 피처 추출 → 모델 예측 → 기여도 분석을 수행합니다. |
| `get_feature_contributions()` | predictor.py | SHAP 또는 근사 방식으로 피처 기여도를 계산합니다. |

---

## 8. 디렉토리 구조

```
cesco-deskroom/
├── flows/
│   └── analyze_current_user/
│       ├── main.py               # 해약 예측 파이프라인을 정의합니다.
│       ├── feature_engineering.py # 피처 계산 로직을 정의합니다.
│       └── active_check.py       # 활성 고객 필터링 로직을 정의합니다.
├── src/
│   └── models/
│       ├── churn_prediction/
│       │   ├── predictor.py      # XGBoost 추론 클래스를 정의합니다.
│       │   └── train.py          # 모델 훈련 스크립트입니다.
│       ├── churn_model_with_csi.pkl       # 훈련된 모델 파일입니다.
│       └── training_averages_with_csi.pkl # 훈련 통계 파일입니다.
├── entities/
│   └── analytics/
│       ├── user_churn_prediction.py  # 해약 예측 결과 엔티티를 정의합니다.
│       └── user_monthly_property.py  # 월별 피처 엔티티를 정의합니다.
├── deploy.py                     # Prefect Flow 배포 설정과 스케줄을 정의합니다.
└── init.sql                      # PostgreSQL 테이블 DDL을 정의합니다.
```
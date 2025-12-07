# sLLM 재학습 시점 탐지 가이드

> VOC 분류 sLLM의 재학습 필요 여부를 판단하는 3가지 지표와 대응 방법

## 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│            analytics.voc_message_category                   │
│                    (소스 데이터)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            monitor_model_health Flow                        │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Confidence  │ │  Unknown    │ │    Correction       │   │
│  │   Drift     │ │   Rate      │ │      Rate           │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              monitoring.model_health                        │
│                    (결과 저장)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 재학습 필요 여부 탐지 지표

### 1.1 Confidence Drift (신뢰도 드리프트)

모델 신뢰도 분포의 변화를 측정합니다.

**측정 방법**
- KL-Divergence (Kullback-Leibler Divergence) 사용
- `scipy.stats.entropy(p_dist, q_dist)` 함수로 계산

**비교 기간**
| 구분 | 기간 |
|------|------|
| Baseline | 64일 전 ~ 32일 전 |
| Current | 32일 전 ~ 현재 |

**계산 과정**
1. `model_confidence` 값을 10개 bin (0.0 ~ 1.0)으로 히스토그램 생성
2. Baseline 분포와 Current 분포의 KL-Divergence 계산
3. epsilon(1e-10)을 더해 zero division 방지

**해석**
- 값이 상승하면 입력 데이터 분포가 변화했음을 의미
- 재학습 검토 필요

```python
bins = np.linspace(0, 1, 11)  # 10 bins
p_dist, _ = np.histogram(baseline_scores["model_confidence"], bins=bins, density=True)
q_dist, _ = np.histogram(current_scores["model_confidence"], bins=bins, density=True)
drift_scores = entropy(p_dist, q_dist)
```

---

### 1.2 Unknown Rate (기타 분류 비율)

모델이 분류하지 못한 데이터의 비율을 측정한다.

**측정 방법**
```
Unknown Rate = (main_category_1_name = '기타'인 레코드 수) / (전체 레코드 수)
```

**측정 기간**
- 최근 30일

**해석**
- 값이 상승하면 기존 학습 데이터에 없던 새로운 유형의 VOC가 유입되고 있음을 의미
- 재학습 검토 필요

```sql
-- 기타 분류 레코드 조회
SELECT *
FROM analytics.voc_message_category
WHERE main_category_1_name = '기타'
AND created_at BETWEEN :start_date AND :end_date
```

---

### 1.3 Correction Rate (Optional, 수정 비율)

사람이 모델 결과를 수정한 비율을 측정한다.

**측정 방법**
```
Correction Rate = (created_at != updated_at인 레코드 수) / (전체 레코드 수)
```

**측정 기간**
- 최근 30일

**해석**
- 값이 상승하면 모델 정확도가 저하되어 사람의 수정이 빈번해졌음을 의미
- 재학습 검토 필요

cf. 적재 결과에 대해서 유저가 업데이트가 있을 시에만 활용할 수 있습니다.

---

## 2. 데이터 테이블

### 소스 테이블: `analytics.voc_message_category`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| model_confidence | FLOAT | 모델 신뢰도 점수 (0~1) |
| main_category_1_name | VARCHAR(50) | 대분류 카테고리명 |
| created_at | TIMESTAMPTZ | 레코드 생성 시간 |
| updated_at | TIMESTAMPTZ | 레코드 수정 시간 |

### 결과 테이블: `monitoring.model_health`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | BIGSERIAL | PK |
| model_name | VARCHAR(100) | 모델명 (cesco_sLLM_Qwen_3) |
| model_version | VARCHAR(50) | 버전 |
| checked_at | TIMESTAMPTZ | 체크 시간 |
| unknown_rate | FLOAT | 기타 분류 비율 |
| correction_rate | FLOAT | 수정 비율 |
| confidence_drift | FLOAT | 신뢰도 드리프트 |

### 결과 조회 쿼리

```sql
SELECT
    model_name,
    checked_at,
    unknown_rate,
    correction_rate,
    confidence_drift
FROM monitoring.model_health
ORDER BY checked_at DESC
LIMIT 10;
```

---

## 3. 탐지 시 대응 액션

지표 상승이 확인되면 다음 절차로 재학습을 수행한다.

1. `train.py`에 신규 데이터 추가
2. Biased data (분류 오류가 발생한 케이스) 추가
3. 재학습 수행

---

## Tips: 성능 저하 시 대응 판단

Test Accuracy가 낮아진 상황에서:

**탐지 조건에 걸리는 경우 (Confidence Drift, Unknown Rate 상승)**
- sLLM이 학습한 분포와 현재 Inference에서 들어오는 데이터 분포가 다름
- → 재학습 수행

**탐지 조건에 걸리지 않는 경우**
- 데이터 분포는 동일하나 모델 출력이 부정확함
- → 프롬프트 엔지니어링으로 프롬프트 변경
- 프롬프트 위치: `cesco-inference-server/app.py`

---

## 참조 소스코드

| 파일 | 설명 |
|------|------|
| `flows/monitor_model_health/main.py` | 모니터링 Flow 전체 로직 |
| `init.sql` | 테이블 스키마 정의 |

### 코드 위치

| 함수 | 설명 | 라인 |
|------|------|------|
| `calculate_confidence_drift()` | Confidence Drift 계산 | 28-61 |
| `calculate_indefinite_answer_rate()` | Unknown Rate 계산 | 102-118 |
| `calculate_update_rate()` | Correction Rate 계산 | 121-139 |
| `monitor_model_health()` | 메인 Flow | 142-164 |

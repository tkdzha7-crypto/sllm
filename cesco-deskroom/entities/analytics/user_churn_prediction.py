from pydantic import BaseModel


class UserChurnPrediction(BaseModel):
    ccod: str  # 고객 코드
    churn_prob: float  # 이탈 확률
    churn_label: int  # 이탈 여부 (0: 유지, 1: 이탈)
    feature_1_kor: str | None
    feature_1_value: float | None
    feature_1_normal_average: float | None
    feature_1_churn_average: float | None
    feature_1_contrib: float | None
    feature_1_contrib_label: str | None
    feature_2_kor: str | None
    feature_2_value: float | None
    feature_2_normal_average: float | None
    feature_2_churn_average: float | None
    feature_2_contrib: float | None
    feature_2_contrib_label: str | None
    feature_3_kor: str | None
    feature_3_value: float | None
    feature_3_normal_average: float | None
    feature_3_churn_average: float | None
    feature_3_contrib: float | None
    feature_3_contrib_label: str | None
    feature_4_kor: str | None
    feature_4_value: float | None
    feature_4_normal_average: float | None
    feature_4_churn_average: float | None
    feature_4_contrib: float | None
    feature_4_contrib_label: str | None
    feature_5_kor: str | None
    feature_5_value: float | None
    feature_5_normal_average: float | None
    feature_5_churn_average: float | None
    feature_5_contrib: float | None
    feature_5_contrib_label: str | None
    feature_6_kor: str | None
    feature_6_value: float | None
    feature_6_normal_average: float | None
    feature_6_churn_average: float | None
    feature_6_contrib: float | None
    feature_6_contrib_label: str | None
    feature_7_kor: str | None
    feature_7_value: float | None
    feature_7_normal_average: float | None
    feature_7_churn_average: float | None
    feature_7_contrib: float | None
    feature_7_contrib_label: str | None
    feature_8_kor: str | None
    feature_8_value: float | None
    feature_8_normal_average: float | None
    feature_8_churn_average: float | None
    feature_8_contrib: float | None
    feature_8_contrib_label: str | None
    feature_9_kor: str | None
    feature_9_value: float | None
    feature_9_normal_average: float | None
    feature_9_churn_average: float | None
    feature_9_contrib: float | None
    feature_9_contrib_label: str | None
    feature_10_kor: str | None
    feature_10_value: float | None
    feature_10_normal_average: float | None
    feature_10_churn_average: float | None
    feature_10_contrib: float | None
    feature_10_contrib_label: str | None

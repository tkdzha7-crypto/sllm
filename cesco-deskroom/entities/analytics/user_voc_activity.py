from pydantic import BaseModel


class UserVocActivity(BaseModel):
    ccod: str  # 고객 코드
    aggregate_date: str  # 집계 기준일 (YYYY-MM-DD)
    category_code: str  # VOC 카테고리 코드
    category_name: str  # VOC 카테고리 이름
    recontact_agg_day: int  # 재접촉 집계 일수
    recontact_past_24h: int  # 재접촉 과거 24시간 건수
    recontact_past_3d: int  # 재접촉 과거 3일 건수
    recontact_past_7d: int  # 재접촉 과거 7일 건수
    recontact_past_30d: int  # 재접촉 과거 30일 건수

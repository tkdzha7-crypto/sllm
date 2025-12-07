from pydantic import BaseModel


class UserRecommendation(BaseModel):
    ccod: str  # 고객 코드
    rec_contract_1: str | None
    rec_contract_1_reason: str | None
    rec_contract_2: str | None
    rec_contract_2_reason: str | None
    rec_contract_3: str | None
    rec_contract_3_reason: str | None
    rec_product_1: str | None
    rec_product_1_reason: str | None
    rec_product_2: str | None
    rec_product_2_reason: str | None
    rec_product_3: str | None
    rec_product_3_reason: str | None
    user_cluster: int | None
    cluster_similarity: float | None
    sim_ccod: str | None
    sim_user_name: str | None
    sim_user_contracts: list[str] | None  # JSON 형태로 저장
    sim_user_products: list[str] | None  # JSON 형태로 저장

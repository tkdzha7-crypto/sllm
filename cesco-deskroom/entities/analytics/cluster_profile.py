from pydantic import BaseModel


class ClusterProfile(BaseModel):
    # snapshot_month: str
    cluster_id: int
    cluster_size: int
    avg_contracts_num: float
    avg_purchases_num: float
    top_contracts: list[str]  # JSON 형태로 저장
    top_purchases: list[str]  # JSON 형태로 저장
    top_business_type: list[str]  # JSON 형태로 저장
    top_first_contract_code: list[str]  # JSON 형태로 저장
    contracts_distribution: dict  # JSON 형태로 저장
    purchases_distribution: dict  # JSON 형태로 저장

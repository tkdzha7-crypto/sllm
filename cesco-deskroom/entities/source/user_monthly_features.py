from pydantic import BaseModel


class User(BaseModel):
    국적: str
    성별: str
    업태: str
    종목: str
    주소1: str
    주소2: str
    고객명: str
    유형대: str
    유형중: str
    고객코드: str
    담당부서: str
    대표자명: str
    등록일자: str
    우편번호: str
    사업자번호: str
    신고객분류코드: str


class Contract(BaseModel):
    면적: float
    순번: int
    면적M: float
    계약대상: str
    계약유형: str
    계약일자: str
    계약종류: str
    등록일자: str
    사용여부: str
    수정일자: str
    시작일자: str
    옵션코드: str
    외곽면적: float
    접수순번: int
    접수일자: str
    종료일자: str
    해약여부: str
    해약일자: str
    계약일련번호: str
    계약특이사항: str
    설치요청일자: str
    실적적용여부: str
    계약대상_대분류명: str
    계약대상_중분류명: str
    계약상세_사용여부: str
    계약대상_대분류코드: str
    계약대상_중분류코드: str


class Purchase(BaseModel):
    order_code: str
    order_sequence: int
    purchase_date: str
    purchase_month: str
    item_code: str
    item_name: str
    quantity: int
    buy_kind: str
    buy_kind_name: str
    sale_channel_code: str
    sale_channel_name: str
    sale_detail_code: str
    sale_detail_name: str
    user_id: str
    load_datetime: str


class Interaction(BaseModel):
    event_type: str
    접수일시: str
    접수내용: str
    고객코드: str


class UserMonthlyFeatures(BaseModel):
    ccod: str  # 고객 코드
    snapshot_month: str  # 스냅샷 월 (YYYY-MM 형식)
    user_information: User
    contract_info: list[Contract]
    purchase_logs: list[Purchase]
    interaction_history: list[Interaction]

from datetime import datetime

from pydantic import BaseModel


class Message(BaseModel):
    rcno: str  # 접수 번호
    received_at: datetime  # 접수 일시
    ccod: str  # 고객 코드
    content: str  # 메시지 내용

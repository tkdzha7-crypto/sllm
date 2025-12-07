from pydantic import BaseModel


class VOCCategory(BaseModel):
    voc_id: str
    name: str
    parent_id: str
    level: int

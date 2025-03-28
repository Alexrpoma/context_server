from typing import List

from pydantic import BaseModel


class DataItem(BaseModel):
    summary: str
    language: str
    url: str

class DataPayload(BaseModel):
    items: List[DataItem]
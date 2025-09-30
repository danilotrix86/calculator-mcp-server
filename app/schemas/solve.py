from pydantic import BaseModel, Field
from typing import Optional


class SolveRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class SolveImageRequest(BaseModel):
    image_data: str = Field(description="Base64 encoded image data")


class SolveResponse(BaseModel):
    answer: str
    used_tool: bool = False
    tool_called: Optional[str] = None
    tool_args: Optional[dict] = None
    error: Optional[str] = None

from pydantic import BaseModel, Field, field_validator
from typing import Optional

from app.config import rate_limit_config


class SolveRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    spoken_text: Optional[str] = Field(default=None, max_length=2000)


class SolveImageRequest(BaseModel):
    image_data: str = Field(description="Base64 encoded image data")

    @field_validator("image_data")
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        max_bytes = rate_limit_config.IMAGE_MAX_SIZE_BYTES
        if len(v) > max_bytes:
            raise ValueError(
                f"Image data exceeds maximum size of {max_bytes // (1024 * 1024)} MB"
            )
        return v


class SolveResponse(BaseModel):
    answer: str
    used_tool: bool = False
    tool_called: Optional[str] = None
    tool_args: Optional[dict] = None
    error: Optional[str] = None

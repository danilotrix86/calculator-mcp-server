import os
from dotenv import load_dotenv

load_dotenv()


class RateLimitConfig:
    SOLVE = os.environ.get("RATE_LIMIT_SOLVE", "10/minute")
    CHAT = os.environ.get("RATE_LIMIT_CHAT", "30/minute")
    MATRIX = os.environ.get("RATE_LIMIT_MATRIX", "30/minute")
    BLOG = os.environ.get("RATE_LIMIT_BLOG", "60/minute")
    ADMIN = os.environ.get("RATE_LIMIT_ADMIN", "20/minute")
    HEALTH = os.environ.get("RATE_LIMIT_HEALTH", "10/minute")
    DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "120/minute")

    IMAGE_MAX_SIZE_BYTES = int(os.environ.get("IMAGE_MAX_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB


rate_limit_config = RateLimitConfig()

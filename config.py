import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("gemini_api_key")
    LLAMAPARSE_API_KEY: str = os.getenv("llamaparse_api_key")

    class Config:
        env_file = ".env"

settings = Settings()
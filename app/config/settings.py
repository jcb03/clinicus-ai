import os
from pydantic_settings import BaseSettings
from typing import Optional, List
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    # API Keys - FIXED
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    
    # Rest of your settings...
    text_sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    text_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    audio_emotion_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    speech_to_text_model: str = "openai/whisper-base"
    
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))
    supported_audio_formats: List[str] = ["wav", "mp3", "ogg", "m4a", "webm"]
    supported_video_formats: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]
    supported_image_formats: List[str] = ["jpg", "jpeg", "png", "bmp", "webp"]
    
    openai_model: str = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
    max_tokens: int = int(os.getenv("MAX_TOKENS", 800))
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))
    
    streamlit_port: int = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", 8000))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()

# Debug output
print(f"🔑 Loaded API key length: {len(settings.openai_api_key)}")
print(f"🔑 API key starts with: {settings.openai_api_key[:10] if settings.openai_api_key else 'None'}")

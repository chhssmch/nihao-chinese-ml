from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import warnings

from ml_text.main import translate

warnings.filterwarnings("ignore")

app = FastAPI(
    title="NiHao ML Text API",
    description="API машинного перевода между английским и китайским на моделях Helsinki-NLP",
    version="1.0.0",
)


class TranslationRequest(BaseModel):
    """Модель запроса на перевод."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Текст для перевода (1–1000 символов)",
    )
    direction: Literal["en_to_zh", "zh_to_en"] = Field(
        default="en_to_zh",
        description="Направление перевода: 'en_to_zh' или 'zh_to_en'",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "direction": "en_to_zh",
            }
        }


class TranslationResponse(BaseModel):
    """Модель ответа с результатом перевода."""
    source_text: str
    translated_text: str
    direction: str
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Модель ответа health-check."""
    status: str
    service: str
    version: str


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Проверка доступности сервиса."""
    return HealthResponse(
        status="healthy",
        service="NiHao ML Text API",
        version="1.0.0",
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Переводит текст между английским и китайским
    """
    try:
        translated_text = translate(request.text, request.direction)

        return TranslationResponse(
            source_text=request.text,
            translated_text=translated_text,
            direction=request.direction,
            success=True,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перевода: {str(e)}")


@app.get("/translate")
async def translate_get(
    text: str,
    direction: Literal["en_to_zh", "zh_to_en"] = "en_to_zh",
):
    """
    GET-эндпоинт для быстрой проверки перевода
    """
    try:
        translated_text = translate(text, direction)

        return TranslationResponse(
            source_text=text,
            translated_text=translated_text,
            direction=direction,
            success=True,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перевода: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
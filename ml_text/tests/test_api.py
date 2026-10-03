import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from ml_text.api_views import app
from ml_text.main import detect_english_ratio, validate_language

client = TestClient(app)


class TestLanguageDetection:
    """Тесты функций определения языка."""

    def test_detect_english_ratio_pure_english(self):
        """Чистый английский текст — доля английских букв 1.0"""
        ratio = detect_english_ratio("Hello World")
        assert ratio == 1.0

    def test_detect_english_ratio_pure_chinese(self):
        """Чистый китайский текст — доля английских букв 0.0"""
        ratio = detect_english_ratio("你好世界")
        assert ratio == 0.0

    def test_detect_english_ratio_mixed(self):
        """Смешанный текст — доля английских букв между 0.6 и 0.8"""
        ratio = detect_english_ratio("Hello 你好")
        assert 0.6 < ratio < 0.8

    def test_detect_english_ratio_with_numbers(self):
        """Цифры и пунктуация не влияют на долю английских букв"""
        ratio = detect_english_ratio("Hello123!")
        assert ratio == 1.0

    def test_detect_english_ratio_empty_string(self):
        """Пустая строка — доля 0.0"""
        ratio = detect_english_ratio("")
        assert ratio == 0.0

    def test_validate_language_en_to_zh_valid(self):
        """Валидация EN→ZH: корректный английский текст проходит"""
        validate_language("Hello World", "en_to_zh")

    def test_validate_language_en_to_zh_invalid(self):
        """Валидация EN→ZH: китайский текст отклоняется"""
        with pytest.raises(ValueError) as exc_info:
            validate_language("你好世界", "en_to_zh")
        assert "английском языке" in str(exc_info.value)

    def test_validate_language_zh_to_en_valid(self):
        """Валидация ZH→EN: корректный китайский текст проходит"""
        validate_language("你好世界", "zh_to_en")

    def test_validate_language_zh_to_en_invalid(self):
        """Валидация ZH→EN: английский текст отклоняется"""
        with pytest.raises(ValueError) as exc_info:
            validate_language("Hello World", "zh_to_en")
        assert "китайском языке" in str(exc_info.value)

    def test_validate_language_mixed_text_en_to_zh(self):
        """Валидация EN→ZH: смешанный текст с преобладанием английского проходит"""
        validate_language("Hello 你好", "en_to_zh")

    def test_validate_language_mixed_text_zh_to_en(self):
        """Валидация ZH→EN: смешанный текст с преобладанием китайского проходит"""
        validate_language("你好你好你好你好Hello", "zh_to_en")

    def test_validate_language_mixed_text_en_to_zh_fails(self):
        """Валидация EN→ZH: смешанный текст с преобладанием китайского отклоняется"""
        with pytest.raises(ValueError) as exc_info:
            validate_language("Hello, 我爱中国!", "en_to_zh")
        assert "английском языке" in str(exc_info.value)

    def test_validate_language_mixed_text_zh_to_en_fails(self):
        """Валидация ZH→EN: смешанный текст с преобладанием английского отклоняется"""
        with pytest.raises(ValueError) as exc_info:
            validate_language("Hello World 你好", "zh_to_en")
        assert "китайском языке" in str(exc_info.value)


class TestHealthCheck:
    """Тесты эндпоинта health-check."""

    def test_health_check_success(self):
        """Health-check возвращает корректный ответ."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "NiHao ML Text API"
        assert data["version"] == "1.0.0"


class TestTranslationPOST:
    """Тесты эндпоинта POST /translate."""

    @patch('ml_text.api_views.translate')
    def test_translate_en_to_zh_success(self, mock_translate):
        """Успешный перевод EN→ZH."""
        mock_translate.return_value = "你好，你好吗？"

        response = client.post(
            "/translate",
            json={"text": "Hello, how are you?", "direction": "en_to_zh"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_text"] == "Hello, how are you?"
        assert data["translated_text"] == "你好，你好吗？"
        assert data["direction"] == "en_to_zh"
        assert data["success"] is True
        assert data["error"] is None
        mock_translate.assert_called_once_with("Hello, how are you?", "en_to_zh")

    @patch('ml_text.api_views.translate')
    def test_translate_zh_to_en_success(self, mock_translate):
        """Успешный перевод ZH→EN."""
        mock_translate.return_value = "Hello, how are you?"

        response = client.post(
            "/translate",
            json={"text": "你好世界123", "direction": "zh_to_en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_text"] == "你好世界123"
        assert data["translated_text"] == "Hello, how are you?"
        assert data["direction"] == "zh_to_en"
        assert data["success"] is True
        mock_translate.assert_called_once_with("你好世界123", "zh_to_en")

    @patch('ml_text.api_views.translate')
    def test_translate_default_direction(self, mock_translate):
        """По умолчанию направление — en_to_zh."""
        mock_translate.return_value = "翻译结果"

        response = client.post("/translate", json={"text": "Test text"})

        assert response.status_code == 200
        data = response.json()
        assert data["direction"] == "en_to_zh"
        mock_translate.assert_called_once_with("Test text", "en_to_zh")

    def test_translate_empty_text(self):
        """Пустой текст — ошибка валидации."""
        response = client.post(
            "/translate",
            json={"text": "", "direction": "en_to_zh"},
        )
        assert response.status_code == 422

    def test_translate_text_too_long(self):
        """Текст длиннее 1000 символов — ошибка валидации."""
        long_text = "a" * 1001
        response = client.post(
            "/translate",
            json={"text": long_text, "direction": "en_to_zh"},
        )
        assert response.status_code == 422

    def test_translate_invalid_direction(self):
        """Невалидное направление — ошибка валидации Pydantic."""
        response = client.post(
            "/translate",
            json={"text": "Test text", "direction": "invalid_direction"},
        )
        assert response.status_code == 422

    @patch('ml_text.api_views.translate')
    def test_translate_model_error(self, mock_translate):
        """Ошибка модели обрабатывается как 500."""
        mock_translate.side_effect = Exception("Model loading failed")

        response = client.post(
            "/translate",
            json={"text": "Test text", "direction": "en_to_zh"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Ошибка перевода" in data["detail"]

    @patch('ml_text.api_views.translate')
    def test_translate_value_error(self, mock_translate):
        """ValueError из translate обрабатывается как 400."""
        mock_translate.side_effect = ValueError("Invalid translation parameters")

        response = client.post(
            "/translate",
            json={"text": "Test text", "direction": "en_to_zh"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid translation parameters" in data["detail"]

    def test_translate_wrong_language_en_to_zh(self):
        """Китайский текст для EN→ZH отклоняется."""
        response = client.post(
            "/translate",
            json={"text": "你好，你好吗？", "direction": "en_to_zh"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "не соответствует выбранному направлению перевода" in data["detail"]
        assert "английском языке" in data["detail"]

    def test_translate_mixed_wrong_language_en_to_zh(self):
        """Смешанный текст с преобладанием китайского отклоняется для EN→ZH."""
        response = client.post(
            "/translate",
            json={"text": "Hello, 我爱中国!", "direction": "en_to_zh"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "не соответствует выбранному направлению перевода" in data["detail"]

    def test_translate_wrong_language_zh_to_en(self):
        """Английский текст для ZH→EN отклоняется."""
        response = client.post(
            "/translate",
            json={"text": "Hello, how are you?", "direction": "zh_to_en"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "не соответствует выбранному направлению перевода" in data["detail"]
        assert "китайском языке" in data["detail"]


class TestTranslationGET:
    """Тесты эндпоинта GET /translate."""

    @patch('ml_text.api_views.translate')
    def test_translate_get_en_to_zh_success(self, mock_translate):
        """Успешный GET-перевод EN→ZH."""
        mock_translate.return_value = "你好"

        response = client.get("/translate?text=Hello World&direction=en_to_zh")

        assert response.status_code == 200
        data = response.json()
        assert data["source_text"] == "Hello World"
        assert data["translated_text"] == "你好"
        assert data["direction"] == "en_to_zh"
        assert data["success"] is True
        mock_translate.assert_called_once_with("Hello World", "en_to_zh")

    @patch('ml_text.api_views.translate')
    def test_translate_get_default_direction(self, mock_translate):
        """GET без direction использует en_to_zh."""
        mock_translate.return_value = "翻译"

        response = client.get("/translate?text=Test")

        assert response.status_code == 200
        data = response.json()
        assert data["direction"] == "en_to_zh"
        mock_translate.assert_called_once_with("Test", "en_to_zh")

    def test_translate_get_invalid_direction(self):
        """GET с невалидным direction — ошибка валидации."""
        response = client.get("/translate?text=Test&direction=invalid")
        assert response.status_code == 422

    @patch('ml_text.api_views.translate')
    def test_translate_get_model_error(self, mock_translate):
        """Ошибка модели в GET обрабатывается как 500."""
        mock_translate.side_effect = Exception("Model error")

        response = client.get("/translate?text=Test")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Ошибка перевода" in data["detail"]

    def test_translate_get_wrong_language_en_to_zh(self):
        """GET EN→ZH с китайским текстом отклоняется."""
        response = client.get("/translate?text=你好&direction=en_to_zh")

        assert response.status_code == 400
        data = response.json()
        assert "не соответствует выбранному направлению перевода" in data["detail"]

    def test_translate_get_wrong_language_zh_to_en(self):
        """GET ZH→EN с английским текстом отклоняется."""
        response = client.get("/translate?text=Hello&direction=zh_to_en")

        assert response.status_code == 400
        data = response.json()
        assert "не соответствует выбранному направлению перевода" in data["detail"]

    def test_translate_get_mixed_wrong_language_zh_to_en(self):
        """GET ZH→EN со смешанным текстом с преобладанием английского отклоняется."""
        response = client.get("/translate?text=Hello World 你好&direction=zh_to_en")

        assert response.status_code == 400
        data = response.json()
        assert "не соответствует выбранному направлению перевода" in data["detail"]


class TestEdgeCases:
    """Граничные случаи."""

    @patch('ml_text.api_views.translate')
    def test_translate_max_length_text(self, mock_translate):
        """Текст ровно 1000 символов проходит."""
        mock_translate.return_value = "翻译结果"
        max_length_text = "a" * 1000

        response = client.post(
            "/translate",
            json={"text": max_length_text, "direction": "en_to_zh"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('ml_text.api_views.translate')
    def test_translate_min_length_text(self, mock_translate):
        """Текст из 1 символа проходит."""
        mock_translate.return_value = "翻"

        response = client.post(
            "/translate",
            json={"text": "a", "direction": "en_to_zh"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('ml_text.api_views.translate')
    def test_translate_special_characters(self, mock_translate):
        """Специальные символы не мешают переводу."""
        mock_translate.return_value = "特殊字符翻译"

        response = client.post(
            "/translate",
            json={"text": "Test @#$%^&*()", "direction": "en_to_zh"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch('ml_text.api_views.translate')
    def test_translate_unicode_characters(self, mock_translate):
        """Unicode-символы не мешают переводу."""
        mock_translate.return_value = "Unicode 翻译"

        response = client.post(
            "/translate",
            json={"text": "Test ñ 中文 🎉", "direction": "en_to_zh"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestAPISpecification:
    """Проверка документации и спецификации API."""

    def test_openapi_schema_exists(self):
        """OpenAPI-схема доступна."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_api_docs_available(self):
        """Swagger UI доступен."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_api_re_doc_available(self):
        """ReDoc доступен."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
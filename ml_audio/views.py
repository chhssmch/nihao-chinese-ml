import os
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from .main import speech_to_text

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ


@require_http_methods(["GET", "POST"])
def transcribe_view(request):
    """
    Страница распознавания китайской речи.
    GET  — форма загрузки.
    POST — принимаем файл, распознаём, показываем текст.
    """
    recognized_text = ""
    error = None

    if request.method == "POST":
        uploaded_file = request.FILES.get("audio_file")

        if not uploaded_file:
            error = "Выберите аудиофайл."
        else:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                error = f"Недопустимый формат: {ext}. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
            elif uploaded_file.size > MAX_FILE_SIZE:
                error = "Файл слишком большой. Максимум 10 МБ."
            else:
                save_dir = os.path.join(settings.MEDIA_ROOT, "audio")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, uploaded_file.name)

                with open(save_path, "wb+") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

                try:
                    recognized_text = speech_to_text(save_path, language="chinese")
                except Exception as e:
                    error = f"Ошибка при распознавании: {e}"
                finally:
                    if os.path.exists(save_path):
                        os.remove(save_path)

    context = {
        "recognized_text": recognized_text,
        "error": error,
    }
    return render(request, "ml_audio/transcribe.html", context)
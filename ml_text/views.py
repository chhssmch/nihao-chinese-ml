from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .main import translate


def home(request):
    """Главная страница сайта."""
    return render(request, "ml_text/home.html")

@require_http_methods(["GET", "POST"])
def translate_view(request):
    source_text = ""
    translated_text = ""
    error = None
    direction = request.POST.get("direction", "en_to_zh")  # По умолчанию EN -> ZH

    if request.method == "POST":
        source_text = request.POST.get("source_text", "").strip()
        direction = request.POST.get("direction", "en_to_zh")

        if not source_text:
            error = "Введите текст для перевода."
        elif len(source_text) > 1000:
            error = "Текст слишком длинный. Максимум 1000 символов."
        else:
            try:
                translated_text = translate(source_text, direction)
            except Exception as e:
                error = f"Ошибка при переводе: {e}"

    context = {
        "source_text": source_text,
        "translated_text": translated_text,
        "error": error,
        "direction": direction,
    }
    return render(request, "ml_text/translate.html", context)
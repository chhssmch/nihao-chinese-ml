import os
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .main import detect_objects, draw_boxes


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 МБ


@require_http_methods(["GET", "POST"])
def recognize_view(request):
    detected_items = []
    result_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.FILES.get("image_file")

        if not uploaded_file:
            error = "Выберите изображение."
        else:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                error = f"Недопустимый формат: {ext}. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
            elif uploaded_file.size > MAX_FILE_SIZE:
                error = "Файл слишком большой. Максимум 5 МБ."
            else:
                # Папки: media/images/  и  media/results/
                input_dir = os.path.join(settings.MEDIA_ROOT, "images")
                output_dir = os.path.join(settings.MEDIA_ROOT, "results")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                input_path = os.path.join(input_dir, uploaded_file.name)
                output_name = f"annotated_{uploaded_file.name}"
                output_path = os.path.join(output_dir, output_name)

                with open(input_path, "wb+") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

                try:
                    detected_items = detect_objects(input_path)

                    if not detected_items:
                        error = "На изображении не найдено объектов (порог уверенности 0.3)."
                    else:
                        draw_boxes(input_path, detected_items, output_path)
                        result_url = f"{settings.MEDIA_URL}results/{output_name}"
                except Exception as e:
                    error = f"Ошибка при детекции: {e}"
                finally:
                    # удаляем исходник, оставляем только результат
                    if os.path.exists(input_path):
                        os.remove(input_path)

    context = {
        "detected_items": detected_items,
        "result_url": result_url,
        "error": error,
    }
    return render(request, "ml_image/recognize.html", context)
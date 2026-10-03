import os
import shutil
import uuid
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .main import analyze_video


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 МБ


@require_http_methods(["GET", "POST"])
def analyze_view(request):
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.FILES.get("video_file")

        if not uploaded_file:
            error = "Выберите видеофайл."
        else:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                error = f"Недопустимый формат: {ext}. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
            elif uploaded_file.size > MAX_FILE_SIZE:
                error = "Файл слишком большой. Максимум 20 МБ."
            else:
                session_id = uuid.uuid4().hex[:8]
                work_dir = os.path.join(settings.MEDIA_ROOT, "video_work", session_id)
                frames_dir = os.path.join(settings.MEDIA_ROOT, "video_frames", session_id)
                os.makedirs(work_dir, exist_ok=True)

                video_path = os.path.join(work_dir, uploaded_file.name)

                with open(video_path, "wb+") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

                try:
                    raw = analyze_video(video_path, frames_dir, fps=1)

                    frame_urls = [
                        f"{settings.MEDIA_URL}video_frames/{session_id}/{os.path.basename(p)}"
                        for p in raw["annotated_frames"]
                    ]

                    result = {
                        "unique_objects": raw["unique_objects"],
                        "frame_urls": frame_urls,
                        "total_frames_processed": raw["total_frames_processed"],
                        "duration_seconds": raw["duration_seconds"],
                    }
                except Exception as e:
                    error = f"Ошибка при обработке видео: {e}"
                finally:
                    # удаляем исходное видео, оставляем только кадры
                    if os.path.exists(work_dir):
                        shutil.rmtree(work_dir, ignore_errors=True)

    context = {
        "result": result,
        "error": error,
    }
    return render(request, "ml_video/analyze.html", context)
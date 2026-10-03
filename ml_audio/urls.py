from django.urls import path

from . import views

app_name = "ml_audio"

urlpatterns = [
    path("transcribe/", views.transcribe_view, name="transcribe"),
]
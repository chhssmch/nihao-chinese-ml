from django.urls import path

from . import views

app_name = "ml_video"

urlpatterns = [
    path("analyze/", views.analyze_view, name="analyze"),
]
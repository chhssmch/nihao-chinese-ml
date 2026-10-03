from django.urls import path

from . import views

app_name = "ml_text"

urlpatterns = [
    path("", views.home, name="home"),
    path("translate/", views.translate_view, name="translate"),
]
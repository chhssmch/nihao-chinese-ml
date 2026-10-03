from django.urls import path

from . import views

app_name = "ml_image"

urlpatterns = [
    path("recognize/", views.recognize_view, name="recognize"),
]
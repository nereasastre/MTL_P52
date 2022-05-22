from django.urls import path

from pitch.views import index_view, process

app_name = "pitch"

urlpatterns = [
    path("", index_view, name="index"),
    path("audio/process", process, name="audio-process"),
]

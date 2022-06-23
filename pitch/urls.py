from django.urls import path

from pitch.views import index_view, audio_store

app_name = "pitch"

urlpatterns = [
    path("", index_view, name="index"),
    path('audio', audio_store, name="audio-process"),
]

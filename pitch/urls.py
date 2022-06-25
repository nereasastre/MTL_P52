from django.urls import path

from pitch.views import index_view, audio_store, tunner_view

app_name = "pitch"

urlpatterns = [
    path("", index_view, name="index"),
    path("tuner", tunner_view, name="tuner"),
    path("audio", audio_store, name="audio-process"),
]

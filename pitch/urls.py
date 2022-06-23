from django.urls import path

from pitch.views import index_view, audio_store, tunner_view

app_name = "pitch"

urlpatterns = [
    path("", index_view, name="index"),
    path("tunner", tunner_view, name="tunner"),
    path('audio', audio_store, name="audio-process"),
]

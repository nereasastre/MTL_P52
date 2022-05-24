from django.urls import path

from pitch.views import index_view, Audio_store

app_name = "pitch"

urlpatterns = [
    path("", index_view, name="index"),
    path('audio', Audio_store, name="audio-process"),
]

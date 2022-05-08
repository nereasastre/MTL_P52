from django.urls import path

from pitch.views import index_view

app_name = "pitch"

urlpatterns = [
    path("", index_view, name="index"),
]

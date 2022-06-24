from django.urls import path

from .consumers import WSConsumer

ws_urlpatterns = [
    path("ws/real_time/", WSConsumer.as_asgi()),
]

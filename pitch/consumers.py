import json
from threading import Event
from threading import Thread

from channels.generic.websocket import WebsocketConsumer
from django.conf import settings

from real_time_pitch_extractor.real_time_hps import real_time

condition = Event()


class WSConsumer(WebsocketConsumer):
    new_thread = None

    def scallback(self):
        print(
            f"Pitch detected: {settings.PITCH_DETECTED} --> Closest note: {settings.CLOSEST_NOTE} ({settings.CLOSEST_PITCH}) --> Pitch difference: {settings.PITCH_DIFF}"
        )
        self.send(json.dumps({"pitch": settings.PITCH_DETECTED}))

    def calc_pitch(self):
        print(real_time(self.scallback))

    def connect(self):
        self.accept()
        settings.RECORD = True
        self.new_thread = Thread(target=self.calc_pitch, args=())
        self.new_thread.start()

    def receive(self, text_data=None, bytes_data=None):
        settings.RECORD = False
        self.close()

    def disconnect(self, close_code):
        # Called when the socket closes
        settings.RECORD = False
        print("settings.RECORD", settings.RECORD)
        self.close()

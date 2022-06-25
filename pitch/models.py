# Create your models here.
import os

from django.db import models


class Audio_store(models.Model):
    record = models.FileField(upload_to="audios/")

    class Meta:
        db_table = "Audio_store"

    def filename(self):
        print(self.record.name)
        return os.path.basename(self.record.name)

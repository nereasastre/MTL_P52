# Create your models here.

from django.db import models


class Audio_store(models.Model):
    record = models.FileField(upload_to='audios/')

    class Meta:
        db_table = 'Audio_store'

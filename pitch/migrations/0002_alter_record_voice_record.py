# Generated by Django 4.0.4 on 2022-05-22 20:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("pitch", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="record",
            name="voice_record",
            field=models.FileField(upload_to="audios/"),
        ),
    ]

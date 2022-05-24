# Generated by Django 4.0.4 on 2022-05-24 16:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pitch', '0002_alter_record_voice_record'),
    ]

    operations = [
        migrations.CreateModel(
            name='Audio_store',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('record', models.FileField(upload_to='audios/')),
            ],
            options={
                'db_table': 'Audio_store',
            },
        ),
        migrations.DeleteModel(
            name='Record',
        ),
    ]

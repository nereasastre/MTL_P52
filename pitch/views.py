import logging
import os

from django.http import JsonResponse
from django.shortcuts import render
from scipy.io import wavfile

from offline_pitch_extractor.yin_extractor import yin_extractor
from pitch.forms import AudioForm
from pitch.models import Audio_store


def index_view(request):
    try:
        template_name = "design1.html"
        context = {}
        return render(request, template_name, context)
    except Exception as e:
        logging.error(f"[REFERRAL_DETAIL_VIEW] - ERROR 500 - {e}")


def audio_store(request):
    print("request.FILES", request.FILES)
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES or None)
        if form.is_valid():
            form.save()
            # Process Pitch
            # time.sleep(1)
            audio_path = Audio_store.objects.first().record.path
            print(audio_path)
            sr, audio = wavfile.read(audio_path)
            print("audio" , audio)
            #USE THIS IN CASE STERIO
            #audiodata = audio.astype(float)
            #final_audio = audiodata.sum(axis=1) / 2
            pitches = yin_extractor(audio=audio)
            print("audio pitches", pitches)
            os.remove(audio_path)
            Audio_store.objects.all().delete()
            return JsonResponse(
                {
                    "pitch": pitches,
                    "success": True,
                }
            )
    else:
        form = AudioForm()
    return JsonResponse(
        {
            "pitch": [0],
            "success": False,
        }
    )

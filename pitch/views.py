import logging
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render

from pitch.models import Record


def index_view(request):
    try:
        template_name = "design1.html"
        context = {}
        return render(request, template_name, context)
    except Exception as e:
        logging.error(f"[REFERRAL_DETAIL_VIEW] - ERROR 500 - {e}")


def process(request):
    print(request.FILES)
    print(request.POST)
    if request.method == "POST":
        audio_file = request.FILES.get("audio")
        #record = Record.objects.create(voice_record=audio_file)
        #record.save()
        #messages.success(request, "Audio recording successfully added!")
        return JsonResponse(
            {
                "pitch": 200,
                "success": True,
            }
        )

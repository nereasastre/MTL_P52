import logging

from django.http import JsonResponse
from django.shortcuts import render

from pitch.forms import AudioForm


def index_view(request):
    try:
        template_name = "design1.html"
        context = {}
        return render(request, template_name, context)
    except Exception as e:
        logging.error(f"[REFERRAL_DETAIL_VIEW] - ERROR 500 - {e}")


def Audio_store(request):
    print("request.FILES", request.FILES)
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES or None)
        if form.is_valid():
            form.save()
            return JsonResponse(
                {
                    "pitch": [100, 150, 200, 250, 300, 350, 400, 450, 500],
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
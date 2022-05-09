from django.shortcuts import render
import logging


def index_view(request):
    try:
        template_name = "pitch_extraction.html"
        print("hola mundo")
        context = {}
        return render(request, template_name, context)
    except Exception as e:
        logging.error(f"[REFERRAL_DETAIL_VIEW] - ERROR 500 - {e}")

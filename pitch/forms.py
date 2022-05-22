from django import forms

from pitch.models import Record


class AudioForm(forms.ModelForm):
    class Meta:
        model = Record
        fields = ['voice_record']

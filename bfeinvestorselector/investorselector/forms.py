from django import forms

class DescriptionForm(forms.Form):
    description = forms.CharField(widget=forms.Textarea, label="", help_text="", initial="new start up aiming at low income customers, dedicated in green energy")

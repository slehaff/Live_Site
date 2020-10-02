from PIL import Image
from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit

class PicForm(forms.Form):
    scannerid = forms.CharField(required=False, max_length=32, strip=True)
    cmd = forms.CharField(required=False, label='Tekst', max_length=50, strip=True)
    Pic1 = forms.ImageField(label="Billed", required=True)
    Pic2 = forms.ImageField(label="Billed2", required=True)
    
    def __init__(self, *args, **kwargs):
        super(PicForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        #self.helper.form_class = 'form-horizontal'
        #self.helper.label_class = 'col-sm-5 col-sm-offset-2'   # control-label
        #self.helper.field_class = 'col-sm-4'
        self.helper.form_tag = True
        self.helper.add_input(Submit('submit', 'Send'))
        self.helper.add_input(Submit('cancel', 'Fortryd', css_class='btn-secondary', formnovalidate='formnovalidate', formaction='/'))

      
import os 
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
#from scan_project.models import Scanner

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import PicForm
# from web.config import APP_NAME
# from scan_project.models import Scanner
# from scan_project.settings import MEDIA_ROOT

@csrf_exempt
def logon(request):
    RESPONSE = {
        'apiurl': 'http://django.holmnet.dk/api/',
        'sessionid': '12345'
    }
    deviceid = request.GET.get('deviceid')
    print("DeviceID", deviceid)
    if deviceid != None:
        scanner = Scanner.objects.get()
        print(scanner)
        print(scanner.Clinic)
        print(scanner.Clinic.ClinicName)
        jresponse = {
            **RESPONSE,
            'clinicname': scanner.Clinic.ClinicName,
            'clinicno': scanner.Clinic.ClinicNo,
            }
    else:
        print('No scanner found')
        jresponse = {
            **RESPONSE,
            'error': 1,
            'errortext': 'Scanner not found'
        }        
    return JsonResponse(jresponse)

def find_scanner_clinic(scannerid):
    print('Scannerid:',scannerid)
    return 1
@csrf_exempt
def save_uploaded_file(handle, filepath):
    print ('Handle: ', handle)
    print ('Filename', filepath)
    with open(filepath, 'wb+') as destination:
        for chunk in handle.chunks():
            destination.write(chunk)
    return

@csrf_exempt
def pic(request):

    print('picd')
    
    picform = PicForm()
    mycontext = {
        'title': "Test Upload",
        'pic': picform,
        'name': "Samir",
    }

    if request.method == 'POST':
        #print('FILES: ', request.FILES)
        picform = PicForm(request.POST, request.FILES)
        if picform.is_valid():
            print('valid')
            save_uploaded_file(request.FILES['Pic1'], '/home/samir/dblive/scan/static/scan_image_folder/black.png')
            save_uploaded_file(request.FILES['Pic2'], '/home/samir/dblive/scan/static/scan_image_folder/color.png')
            save_uploaded_file(request.FILES['Pic3'], '/home/samir/dblive/scan/static/scan_image_folder/high.png')
            save_uploaded_file(request.FILES['Pic4'], '/home/samir/dblive/scan/static/scan_image_folder/low.png')
            
            return JsonResponse({'result':"OK"})
        return render(request, 'api/pic.html', mycontext)
    return render(request, 'api/pic.html', mycontext)

def test(request):
    return render(request, "web/test.html")



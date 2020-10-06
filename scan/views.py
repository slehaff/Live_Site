import os 
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
#from scan_project.models import Scanner

# Create your views here.
from django.http import HttpResponse, JsonResponse
from .forms import PicForm
import myglobals
import numpy as np
import cv2

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
    # print ('Handle: ', handle)
    # print ('Filename', filepath)
    with open(filepath, 'wb+') as destination:
        for chunk in handle.chunks():
            destination.write(chunk)
    return


@csrf_exempt
def pic(request):
    f1 = '/home/samir/dblive/scan/static/scan_image_folder/black.png'
    f2 = '/home/samir/dblive/scan/static/scan_image_folder/color.png'
    f3 =  '/home/samir/dblive/scan/static/scan_image_folder/high.png'
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
            save_uploaded_file(request.FILES['Pic1'], f1)
            save_uploaded_file(request.FILES['Pic2'], f2)
            save_uploaded_file(request.FILES['Pic3'], f3)
            # save_uploaded_file(request.FILES['Pic4'], '/home/samir/dblive/scan/static/scan_image_folder/low.png')
            b1 = cv2.imread(f1, 0).astype(np.float32)
            b2 = cv2.imread(f2, 1).astype(np.float32)
            b3 = cv2.imread(f3, 0).astype(np.float32)
            myglobals.inque.append(b1)
            myglobals.inque.append(b2)
            myglobals.inque.append(b3)
            print('Queue:', len(myglobals.inque))
            return JsonResponse({'result':"OK"})
        return render(request, 'api/pic.html', mycontext)

    return render(request, 'api/pic.html', mycontext)

def test(request):
    return render(request, "web/test.html")



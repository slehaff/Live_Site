import os 
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
#from scan_project.models import Scanner

# Create your views here.
from django.http import HttpResponse, JsonResponse


context = {"home_page": "active"}
def index(request):
    return render(request, 'main.html', context)
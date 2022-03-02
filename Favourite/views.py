from django.shortcuts import render
from django.http import HttpResponse
import json

import sys
sys.path.append("/home/xyz/zjj/Django/AIServer/Favourite")
from AiCore.AiCore import _predict

# Create your views here.
def index(request):
    if request.method == 'GET':
        try:
            jsonList = json.loads(request.body)
            jsonList = _predict(jsonList)
            return HttpResponse(json.dumps(jsonList, ensure_ascii=False), content_type="application/json")
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed')
    return HttpResponse('njunb!!')
        
        
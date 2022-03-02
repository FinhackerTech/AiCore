from django.test import TestCase
import json

# Create your tests here.
from django.test import Client
c = Client()
response = c.post('/Favourite/',json.dumps([1,2,3], ensure_ascii=False),content_type="application/json")
response.status_code

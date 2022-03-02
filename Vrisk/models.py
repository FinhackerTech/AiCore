from django.db import models

# Create your models here.

class Features(models.Model):
    '''
     listcolid:features
    '''
    listcolid=models.CharField(max_length=32)
    
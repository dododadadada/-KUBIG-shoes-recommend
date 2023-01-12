from django.db import models

# Create your models here.
class Profile(models.Model):
    image = models.ImageField()

    def __str__(self):
        return self.image

class Musinsa_w(models.Model):
    prd_title = models.CharField(max_length=50, default='')
    category = models.CharField(max_length=50, default='')
    brand = models.CharField(max_length=50, default='')
    price = models.FloatField(max_length=50, default='')
    codi_url = models.CharField(max_length=50, default='')
    shoes_url = models.CharField(max_length=50, default='')
    codi_rec = models.CharField(max_length=50, default='')
    shoes_rec = models.CharField(max_length=50, default='')

class Musinsa_m(models.Model):
    prd_title = models.CharField(max_length=50, default='')
    category = models.CharField(max_length=50, default='')
    brand = models.CharField(max_length=50, default='')
    price = models.FloatField(max_length=50, default='')
    codi_url = models.CharField(max_length=50, default='')
    shoes_url = models.CharField(max_length=50, default='')
    codi_rec = models.CharField(max_length=50, default='')
    shoes_rec = models.CharField(max_length=50, default='')

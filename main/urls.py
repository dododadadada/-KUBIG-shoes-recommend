from django.urls import path

from . import views

app_name = "main"

urlpatterns = [
    path('', views.index, name='index'),
    path('manselect/', views.manselect, name='manselect'),
    path('womanselect/', views.womanselect, name='womanselect'),
    path('upload/', views.upload, name="upload"),
    path('upload_create/', views.upload_create,name="upload_create"),
    path('download_man/', views.downloadManMusinsa, name = 'downloadManMusinsa'),
    path('download_woman/', views.downloadWomanMusinsa, name = 'downloadWomanMusinsa'),
]


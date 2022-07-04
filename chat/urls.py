from django.urls import path
from . import views


urlpatterns = [
    path('',views.index, name="index"),
    path('processInput/<query>',views.processInput, name="process"),
    path('train/',views.train_data,name="train"),
]
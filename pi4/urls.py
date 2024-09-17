from django.contrib import admin
from django.urls import path
from endpoints.views import ponto_mais_proximo

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ponto-mais-proximo/', ponto_mais_proximo),
]


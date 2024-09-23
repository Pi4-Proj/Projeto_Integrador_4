import os
import json
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

class Queimadas:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path, delimiter=';')
        self.pipeline = self.treinar_modelo()

    def calcular_distancia(self, ponto1, ponto2):
        R = 6371.0
        lat1, lon1 = np.deg2rad(ponto1)
        lat2, lon2 = np.deg2rad(ponto2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def encontrar_ponto_mais_proximo(self, ponto_informado):
        print("encontrar ponto mais próximo ")
        df_fogo = self.df[self.df['Fogo'] == 1]
        print("antes do if")
        if df_fogo.empty:
            raise ValueError("Não há dados com 'Fogo' igual a 1.")
        distancias = df_fogo.apply(
            lambda row: self.calcular_distancia(ponto_informado, (row['longitude'], row['latitude'])),
            axis=1
        )
        indice_mais_proximo = distancias.idxmin()
        menor_distancia = distancias.min()
        ponto_proximo = df_fogo.loc[indice_mais_proximo]
        return menor_distancia, ponto_proximo


    def treinar_modelo(self):
        data_file = pd.read_csv(self.csv_path, delimiter=';')
        features_columns = [
            'Precipitação Total < 10mm',
            'Pressão Atmosférica entre 1015 e 1020 hPa',
            'Temperatura Bulbo seco ACIMA DE 30°C',
            'Temperatura Pt Orvalho abaixo de 10°C',
            'Vento com velocidade maior que 30Km/h',
            'Rajada max > 10 m/s',
            'Umidade relativa do ar < 30%',
            'Radiação Solar acima de  4 kWh/m²'
        ]
        X = data_file[features_columns]
        y = data_file['Fogo']

        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('model', DecisionTreeRegressor())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        return pipeline

    def testar_modelo(self, dados_teste):
        features_columns = [
            'Precipitação Total < 10mm',
            'Pressão Atmosférica entre 1015 e 1020 hPa',
            'Temperatura Bulbo seco ACIMA DE 30°C',
            'Temperatura Pt Orvalho abaixo de 10°C',
            'Vento com velocidade maior que 30Km/h',
            'Rajada max > 10 m/s',
            'Umidade relativa do ar < 30%',
            'Radiação Solar acima de  4 kWh/m²'
        ]
        X_teste = pd.DataFrame(dados_teste, columns=features_columns)
        predicoes = self.pipeline.predict(X_teste)
        return predicoes

@csrf_exempt
@require_POST
def ponto_mais_proximo(request):
    print("endpoint alcançado")

    try:
        data = json.loads(request.body)
        ponto_informado = (data['latitude'], data['longitude'])
        dados_teste = {
            'Precipitação Total < 10mm': data['precipitacao_total'],
            'Pressão Atmosférica entre 1015 e 1020 hPa': data['pressao_atmosferica'],
            'Temperatura Bulbo seco ACIMA DE 30°C': data['temp_bulbo_seco'],
            'Temperatura Pt Orvalho abaixo de 10°C': data['temp_orvalho'],
            'Vento com velocidade maior que 30Km/h': data['velocidade_vento'],
            'Rajada max > 10 m/s': data['rajada_max'],
            'Umidade relativa do ar < 30%': data['umidade_relativa'],
            'Radiação Solar acima de  4 kWh/m²': data['radiacao_solar']
        }
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, 'bb_queimadas_Macroregião_Araraquara_v17.csv')
        
        queimadas = Queimadas(csv_path)
        menor_distancia, ponto_proximo = queimadas.encontrar_ponto_mais_proximo(ponto_informado)
        predicao_fogo = queimadas.testar_modelo(dados_teste)

        resposta = {
            'menor_distancia': menor_distancia,
            'ponto_proximo': ponto_proximo.to_dict(),
            'predicao_fogo': predicao_fogo[0] 
        }
        return JsonResponse(resposta)

    except (KeyError, json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'erro': str(e)}, status=400)

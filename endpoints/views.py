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
        self.longitude = 0
        self.latitude = 0
        self.precipitacao_total_menos_10mm = 0
        self.pressao_atmosferica_entre_1015_1020_hPa = 0
        self.temperatura_bulbo_seco_acima_30C = 0
        self.temperatura_pt_orvalho_abaixo_10C = 0
        self.vento_maior_30Km_h = 0
        self.rajada_max_mais_10_m_s = 0
        self.umidade_relativa_abaixo_30 = 0
        self.radiacao_solar_acima_4_kWh_m2 = 0

        @property
        def BASE_DIR(self):
            return self._BASEDIR

        @BASE_DIR.setter
        def BASE_DIR(self, value):
            self._BASE_DIR = value

        @property
        def csv_path(self):
            return self._csv_path

        @csv_path.setter
        def csv_path(self, value):
            self._csv_path = value

        @property
        def longitude(self):
            return self._longitude

        @longitude.setter
        def longitude(self, value):
            self._longitude = value

        @property
        def latitude(self):
            return self._latitude

        @latitude.setter
        def latitude(self, value):
            self._latitude = value

        @property
        def precipitacao_total_menos_10mm(self):
            return self._precipitacao_total_menos_10mm

        @precipitacao_total_menos_10mm.setter
        def precipitacao_total_menos_10mm(self, value):
            self._precipitacao_total_menos_10mm = value

        @property
        def pressao_atmosferica_entre_1015_1020_hPa(self):
            return self._pressao_atmosferica_entre_1015_1020_hPa

        @pressao_atmosferica_entre_1015_1020_hPa.setter
        def pressao_atmosferica_entre_1015_1020_hPa(self, value):
            self._pressao_atmosferica_entre_1015_1020_hPa = value

        @property
        def temperatura_bulbo_seco_acima_30C(self):
            return self._temperatura_bulbo_seco_acima_30C

        @temperatura_bulbo_seco_acima_30C.setter
        def temperatura_bulbo_seco_acima_30C(self, value):
            self._temperatura_bulbo_seco_acima_30C = value

        @property
        def temperatura_pt_orvalho_abaixo_10C(self):
            return self._temperatura_pt_orvalho_abaixo_10C

        @temperatura_pt_orvalho_abaixo_10C.setter
        def temperatura_pt_orvalho_abaixo_10C(self, value):
            self._temperatura_pt_orvalho_abaixo_10C = value

        @property
        def vento_maior_30Km_h(self):
            return self._vento_maior_30Km_h

        @vento_maior_30Km_h.setter
        def vento_maior_30Km_h(self, value):
            self._vento_maior_30Km_h = value

        @property
        def rajada_max_mais_10_m_s(self):
            return self._rajada_max_mais_10_m_s

        @rajada_max_mais_10_m_s.setter
        def rajada_max_mais_10_m_s(self, value):
            self._rajada_max_mais_10_m_s = value

        @property
        def umidade_relativa_abaixo_30(self):
            return self._umidade_relativa_abaixo_30

        @umidade_relativa_abaixo_30.setter
        def umidade_relativa_abaixo_30(self, value):
            self._umidade_relativa_abaixo_30 = value

        @property
        def radiacao_solar_acima_4_kWh_m2(self):
            return self._radiacao_solar_acima_4_kWh_m2

        @radiacao_solar_acima_4_kWh_m2.setter
        def radiacao_solar_acima_4_kWh_m2(self, value):
            self._radiacao_solar_acima_4_kWh_m2 = value


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
        df_fogo = self.df[self.df['Fogo'] == 1]
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
        self.pipeline = pipeline 

        return pipeline  


    def testar_modelo(self):
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
        
        dados_teste = {
            'Precipitação Total < 10mm': self.precipitacao_total_menos_10mm,
            'Pressão Atmosférica entre 1015 e 1020 hPa': self.pressao_atmosferica_entre_1015_1020_hPa,
            'Temperatura Bulbo seco ACIMA DE 30°C': self.temperatura_bulbo_seco_acima_30C,
            'Temperatura Pt Orvalho abaixo de 10°C': self.temperatura_pt_orvalho_abaixo_10C,
            'Vento com velocidade maior que 30Km/h': self.vento_maior_30Km_h,
            'Rajada max > 10 m/s': self.rajada_max_mais_10_m_s,
            'Umidade relativa do ar < 30%': self.umidade_relativa_abaixo_30,
            'Radiação Solar acima de  4 kWh/m²': self.radiacao_solar_acima_4_kWh_m2
        }
        X_teste = pd.DataFrame([dados_teste], columns=features_columns)
        predicoes = self.pipeline.predict(X_teste)
        return predicoes

@csrf_exempt
@require_POST
def ponto_mais_proximo(request):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, 'bb_queimadas_Macroregião_Araraquara_v17.csv')
        queimadas = Queimadas(csv_path)
        data = json.loads(request.body)

        # Verificações e atribuições
        queimadas.latitude = data['latitude']
        queimadas.longitude = data['longitude']
        ponto_informado = (data['latitude'], data['longitude'])

        queimadas.precipitacao_total_menos_10mm = 1 if data['Precipitação Total < 10mm'] < 10 else 0
        queimadas.pressao_atmosferica_entre_1015_1020_hPa = 1 if 1015 <= data['Pressão Atmosférica entre 1015 e 1020 hPa'] <= 1020 else 0
        queimadas.temperatura_bulbo_seco_acima_30C = 1 if data['Temperatura Bulbo seco ACIMA DE 30°C'] > 30 else 0
        queimadas.temperatura_pt_orvalho_abaixo_10C = 1 if data['Temperatura Pt Orvalho abaixo de 10°C'] < 10 else 0
        queimadas.vento_maior_30Km_h = 1 if data['Vento com velocidade maior que 30Km/h'] > 30 else 0
        queimadas.rajada_max_mais_10_m_s = 1 if data['Rajada max > 10 m/s'] > 10 else 0
        queimadas.umidade_relativa_abaixo_30 = 1 if data['Umidade relativa do ar < 30%'] < 30 else 0
        queimadas.radiacao_solar_acima_4_kWh_m2 = 1 if data['Radiação Solar acima de  4 kWh/m²'] > 4 else 0

        menor_distancia, ponto_proximo = queimadas.encontrar_ponto_mais_proximo(ponto_informado)
        predicao_fogo = queimadas.testar_modelo()  # Removido argumento
        resposta = {
            'menor_distancia': menor_distancia,
            'ponto_proximo': ponto_proximo.to_dict(),
            'predicao_fogo': predicao_fogo[0]
        }
        return JsonResponse(resposta)

    except (KeyError, json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'erro': str(e)}, status=400)

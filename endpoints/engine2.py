import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

path_file = './bb_queimadas_Macroregião_Araraquara_v17.csv'
data_file = pd.read_csv(path_file)

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

categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler()) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
print(f"MAE usando Pipeline: {mae:.2f}")

print("Cinco previsões e características correspondentes:")
for i in range(5):
    features = X_test.iloc[i]
    
    print(f"\nPrevisão {i+1}: {predictions[i]}")
    print("Características usadas:")
    for col in features.index:
        print(f"{col}: {features[col]}")
        
print("\nColunas usadas para previsões:")
print(features_columns)


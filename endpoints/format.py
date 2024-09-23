import pandas as pd

# Carregar o CSV usando o separador correto (;)
file_path = 'bb_queimadas_Macroregião_Araraquara_v17.csv'
df = pd.read_csv(file_path, sep=';')

# Função para adicionar ponto após os dois primeiros dígitos
def insert_decimal(value):
    # Certificar-se de que o valor é uma string e remover espaços em branco
    value = str(value).strip()
    
    # Inserir o ponto após os dois primeiros dígitos, se o valor tiver mais de 2 dígitos
    if len(value) > 2:
        return value[:2] + '.' + value[2:]
    return value

# Aplicar a função nas colunas de latitude e longitude
df['latitude'] = df['latitude'].apply(insert_decimal)
df['longitude'] = df['longitude'].apply(insert_decimal)

# Exibir as primeiras linhas para verificar a modificação
print(df[['latitude', 'longitude']].head())

# Salvar o arquivo atualizado com ponto decimal nas coordenadas
df.to_csv('bb_queimadas_Macroregião_Araraquara_v17_decimal.csv', sep=';', index=False)

print("Arquivo salvo com o ponto decimal inserido.")


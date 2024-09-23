import pandas as pd

# Carregar o CSV
file_path = 'bb_queimadas_Macroregião_Araraquara_v17.csv'
df = pd.read_csv(file_path)

# Exibir os nomes das colunas
print(df.columns)


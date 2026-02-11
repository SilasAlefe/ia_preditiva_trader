import time
from cProfile import label
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_csv("dados.csv", sep=";", decimal=',', thousands='.')

#Unindo as colunas data e hora em uma só coluna
df['Data e hora'] = df['Data'] + ' ' +  df['Hora']

#Apagando as colunas individuas de data e hora
df = df.drop('Data', axis=1)
df = df.drop('Hora', axis=1)

#Mudando a ordem das colunas
df = df[['Data e hora', 'Abertura', 'Maximo', 'Minimo', 'Fechamento', 'Volume', 'Quantidade']]

#Colocando a coluna data e hora no formato ideal
df['Data e hora'] = pd.to_datetime(df['Data e hora'], dayfirst=True)

#Definindo a data e hora como o index da coluna
df = df.set_index("Data e hora")

#Plotando Tabela
df['Fechamento'].plot()
plt.show()

#Reorganizando tabela do mais antigo para o mais novo
df = df.sort_index(ascending=True)

#Criando o Target
df['Target'] = df['Fechamento'].shift(-1)


#Criando a média móvel
df['Shortmm'] = df.rolling(window=7).Fechamento.mean()
df['Longmm'] = df.rolling(window=21).Fechamento.mean()

df['Shortmm'].plot()
plt.show()
df['Longmm'].plot()
plt.show()

#Eliminando os valorez NAN
df = df.dropna()

#Criando as variáveis para treino
preditores = df.drop('Target', axis=1)
target = df['Target']

x_treino, x_teste, y_treino, y_teste = train_test_split(preditores, target, test_size=0.2, random_state=42, shuffle=False)
                                                        #Você define os dados a serem usados e a própria biblioteca separa em treino e teste levando em consideração o test_size
#Criando o modelo
#Metodo para analises mais simples, porem, mais passível de overfit -> arvore = DecisionTreeRegressor(max_depth=3)
arvore = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1) #n_jobs obriga a usar todos os núcleos disponíveis fezendo o processamento ser mais rápido, chegando a ser mais de 5X mais rápido
                                #Mais de 500 aumenta demais o tempo e o resultado é irrelevante
#Treinando o modelo
inicio = time.time()
arvore.fit(x_treino, y_treino)
fim = time.time()

#Gerando as previsões
previsao = arvore.predict(x_teste)

#Visualizando tempo, erro e maior erro
erro = np.sqrt(mean_squared_error(y_teste, previsao))
print(f'Pontuação erro: {erro}')
print(f'Tempo de treino: {fim-inicio} segundos')
#print(np.(y_treino))
#print(np.mean(previsao))

#Comparando dados testes com previsao

df_resultado = pd.DataFrame()
df_resultado['real'] = y_teste
df_resultado['previsao'] = previsao
plt.figure(figsize=(10,5))
plt.plot(df_resultado['real'], label='real')
plt.plot(df_resultado['previsao'], label='IA', linestyle='--')
plt.legend()
plt.show()

#print(y_teste)
#print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
#print(previsao)

maxdif = 0
for dado_real, dado_previsao in zip(y_teste, previsao):
    dif = dado_real-dado_previsao
    if np.absolute(dif) > maxdif:
        maxdif = np.absolute(dif)

print(maxdif)













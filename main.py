import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import MetaTrader5 as mt5


#Conectando ao MetaTrader com teste de conexão
if not mt5.initialize():
    print(f"Inicialização falhou, código de erro: {mt5.last_error()}")
else:
    print("Conexão bem sucedida")

x=0

while True:
    #Selecionando o Ativo
    mt5.symbol_select("WINJ26",True)

    #Copiando o Ativo
    dados_mt5 = mt5.copy_rates_from_pos("WINJ26", mt5.TIMEFRAME_M5, 0, 50000)

    #Traduzindo dados para o pandas
    df = pd.DataFrame(dados_mt5)
    df.rename(columns={'time': 'Data', 'open': 'Abertura', 'high': 'Max', 'low': 'Min', 'close': 'Fechamento', 'tick_volume': 'Volume', 'spread': 'Aumento', 'real_volume': 'Volume real'}, inplace=True)
    #Transformando data
    df['Data'] = pd.to_datetime(df['Data'], unit='s', origin='unix', dayfirst=True)


    #Definindo a data e hora como o index da coluna
    df = df.set_index("Data")

    """#Plotando Tabela
    df['Fechamento'].plot()
    plt.show()"""

    #Criando o Target
    df['Target'] = df['Fechamento'].shift(-1)

    #Criando a média móvel
    df['Shortmm'] = df.rolling(window=10).Fechamento.mean()
    df['longmm'] = df.rolling(window=28).Fechamento.mean()

    """#Plotando médias móveis
    df['Shortmm'].plot()
    plt.show()
    df['Longmm'].plot()
    plt.show()"""

    #Divisão dos dados
    df_historico = df.iloc[:-1]
    df_presente = df.iloc[[-1]]

    #print(df_historico.drop('Abertura', axis=1).drop('Max', axis=1).drop('Min', axis=1).drop('Volume', axis=1).drop('Aumento', axis=1).drop('Volume real', axis=1))
    #Eliminando os valorez NAN
    df_historico = df_historico.dropna()

    #print(df.drop('Abertura', axis=1).drop('Max', axis=1).drop('Min', axis=1).drop('Volume', axis=1).drop('Aumento', axis=1).drop('Volume real', axis=1))

    #Criando as variáveis para treino
    preditores = df_historico.drop('Target', axis=1)
    target = df_historico['Target']

    x_treino, x_teste, y_treino, y_teste = train_test_split(preditores, target, test_size=0.1, random_state=42, shuffle=False)
                                                            #Você define os dados a serem usados e a própria biblioteca separa em treino e teste levando em consideração o test_size
    #Criando o modelo
    #Metodo para analises mais simples, porem, mais passível de overfit -> arvore = DecisionTreeRegressor(max_depth=3)
    arvore = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1) #n_jobs obriga a usar todos os núcleos disponíveis fezendo o processamento ser mais rápido, chegando a ser mais de 5X mais rápido
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

    '''df_resultado = pd.DataFrame()
    df_resultado['real'] = y_teste
    df_resultado['previsao'] = previsao
    plt.figure(figsize=(10,5))
    plt.plot(df_resultado['real'], label='real')
    plt.plot(df_resultado['previsao'], label='IA', linestyle='--')
    plt.legend()
    plt.show()'''

    #print(y_teste)
    #print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    #print(previsao)

    maxdif = 0
    for dado_real, dado_previsao in zip(y_teste, previsao):
        dif = dado_real-dado_previsao
        if np.absolute(dif) > maxdif:
            maxdif = np.absolute(dif)

    print(maxdif)


    previsao_atual = arvore.predict(df_presente.drop('Target', axis=1))
    print(f"O candle deve fechar em: {previsao_atual}")

    if previsao_atual > df_historico['Target'].iloc[-1]:
        print("O próximo candle será maior: COMPRA")
    else:
        print("O próximo candle será menor: VENDA")
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    '''minagr = datetime.datetime.now().minute
    while True:
        if minagr != datetime.datetime.now().minute:
            time.sleep(2.5)
            break'''

    while datetime.datetime.now().minute%5==0:
        None
    while datetime.datetime.now().minute%5!=0:
        None
    time.sleep(2.5)
















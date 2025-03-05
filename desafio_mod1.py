# LEMBRAR DE INSTALAR  E IMPORTAR yahooquery
from yahooquery import Ticker
import warnings 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)

#===========================================================================================
def dados_acao ( acao = 'BBDC4.SA' ,data_inicio = "2008-1-1" , data_fim=None):
    # Definindo os tickers
    tickers = [acao]

    # Criando um DataFrame vazio
    new_data = pd.DataFrame()

    # Obtendo os dados históricos
    try:
        abev = Ticker(tickers)

        abev1 = pd.DataFrame(abev.history(start=data_inicio, end=data_fim))

        # Resetando o índice para transformar a coluna de datas em uma coluna regular
        abev1 = abev1.reset_index()
        # Convertendo a coluna 'Date' para o formato datetime
        abev1['date'] = pd.to_datetime(abev1['date'], utc=True).dt.tz_localize(None)

        # definindo um acoluna de retorno simples ((p1 - po)/po)*100 de forma a calcular a evolução em % ao longo do mês 
        abev1['retorno_simples']=(abev1['close']/abev1['close'].shift(1)-1)*100

        # definindo um acoluna de volume simples ((p1 - po)/po)*100 de forma a calcular a evolução em % ao longo do mês 
        abev1['volume_simples']=(abev1['volume']/abev1['volume'].shift(1)-1)*100

        # Define as features (X)  - apenas data e data Unix - numeros, seu quadrado 
        abev1['DateUnix'] = (abev1['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        abev1['DateUnix2'] = (abev1['DateUnix'])**2

        # Calcula a média móvel dos últimos 3 dias e o seu quadrado
        abev1['Med_movel'] = abev1['close'].rolling(window=3).mean()
        abev1['Med_movel_2'] = abev1['Med_movel']**2
        abev1 = abev1.dropna()
        
        # faz a limpeza dos dados
        abev1 = abev1[(abev1['volume']!=0) & abev1['close']!=0]
        abev1 = abev1[(abev1['volume_simples'].abs()<=300)]
        abev1 = abev1.dropna()

        # Definindo a coluna 'date' como índice
        abev1.set_index('date', inplace=True)

        #Sumarizando os dados por mês
        sumarizando = abev1.resample('M').agg({'close': ['mean','median'], 'volume': 'sum' , 'retorno_simples': 'sum' , 'volume_simples': 'sum' })

        # Reorganizando o MultiIndex
        sumarizando.columns = sumarizando.columns.map({
            ('close', 'mean'): 'fechamento_media',
            ('close', 'median'): 'fechamento_mediana',
            ('volume', 'sum'): 'volume_mensal',
            ('retorno_simples', 'sum'): 'retorno_porcent',
            ('volume_simples', 'sum'): 'volume_porcent',

        })

        # Convertendo o volume para milhões de reais
        sumarizando['volume_mensal'] = sumarizando['volume_mensal'] / 1_000_000

        #filtrando a sumarizacao
        sumarizando= sumarizando[
                                   # (sumarizando['volume_porcent'].abs() <= 100) & 
                                    (sumarizando['retorno_porcent'].abs() <= 100) &
                                    (sumarizando['volume_porcent'].abs() != 0) & 
                                    (sumarizando['retorno_porcent'].abs() != 0) 
                                    ]



        return  sumarizando, abev1,abev
    
    except Exception as e : #erro de excessão -> conexão com yahoo finance e outras possíveis
        print (f"Erro  : {e}")

#===========================================================================================
def grafico_uma_linha (x  , y, nome= 'Valores de Fechamento da Ação',label_y = 'Fechamento'):
        # Gráfico de linha com uma variável
        plt.figure(figsize=(12, 6))  # Ajusta o tamanho da figura
        plt.plot(x,y, linestyle='-', color='b',label=label_y) # Plota os dados
        plt.title(nome)
        plt.xlabel('Data')
        plt.ylabel('valor')
        plt.grid(True)
        plt.show()


#===========================================================================================
def grafico_de_barras (x  , y, nome, label_y1="Fechamento"):

    # Criando o gráfico de barras
    plt.bar(x, y, color='b', width=0.5)

    # Adicionando título e rótulos
    plt.title(nome)
    plt.xlabel('Categorias')
    plt.ylabel('Valores')

    # Incluindo rótulos inclinados
    plt.xticks(rotation=45)  # Inclina os rótulos em 45 graus
    # Exibindo o gráfico
    plt.grid(axis='y')
    plt.show()
#===========================================================================================
def grafico_de_dispersao (y1  , y2, titulo, nome_eixo_1, nome_eixo_2):
     # Plota o gráfico de dispersão entre 2 variáveis
    plt.figure(figsize=(12, 6))
    plt.scatter(y1, y2)
    plt.title(titulo)
    plt.xlabel(nome_eixo_1)
    plt.ylabel(nome_eixo_2)
    plt.grid(True)
    plt.show()

#===========================================================================
def grafico_2_linhas(x, y1, y2, titulo='Gráfico de Linhas', rotulo_x='Eixo X', rotulo_y='Eixo Y', label_y1="Y1", label_y2 = "Y2"):

    plt.figure(figsize=(10, 5))
    
    # Plotando as linhas
    plt.plot(x, y1, label=label_y1, color='blue', marker='o')
    plt.plot(x, y2, label=label_y2, color='red', marker='x')

    # Adicionando título e rótulos
    plt.title(titulo)
    plt.xlabel(rotulo_x)
    plt.ylabel(rotulo_y)

    # Adicionando legenda
    plt.legend()

    # Exibindo o gráfico
    plt.grid()
    plt.show()

#===========================================================================================
def modelo_regressao_scikit(dados, demais_dados, complemento):
     # Prepare os dados para o modelo de regressão
     #tamanho da predicao 1/3 dos dados
    tamanho = len(dados)//3
    
    # Define as features (X) e o target (y)
    X = (demais_dados.shift(-(tamanho))).dropna()  # Adicione mais features se desejar
    y = (dados.shift(-(tamanho))).dropna()  # Prever 1/3 do tamanho dos dados enviados

    # Divida os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crie e treine o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Faça previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avalie o modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Erro médio quadrático (MSE): {mse}')
    print(f'Raiz do erro médio quadrático (RMSE): {rmse}')
    resultado = pd.DataFrame((dados.tail(tamanho)).dropna())

    # Faça previsões no conjunto final
    y_pred = model.predict((demais_dados.tail(tamanho)).dropna())
    #resultado = resultado.dropna()
    #previd = pd.DataFrame(y_pred)
    resultado['previsao'] = y_pred
    resultado['erro_%'] = ((resultado['previsao'] - resultado['close'])/resultado['close'])*100.0
    grafico_uma_linha(resultado.index,resultado['erro_%'], f"Gráfico do Erro na previsão em (%) com (MSE): {mse}")
    grafico_2_linhas(resultado.index, resultado['close'], resultado['previsao'], f'Correto vs Previsto dos dados - DADOS TESTE - {complemento}', 'Data', 'Fechamento', 'Fechamento Real', 'Previsão')

    print(resultado)

    return model
#===========================================================================
def previsao_futuro(abev1,model,datas):

    # Resetando o índice para transformar a coluna de datas em uma coluna regular
    abev1 = abev1.reset_index()
    # Define as features (X)  - apenas data e data Unix - numeros, seu quadrado 
    novas_datas = pd.DataFrame({'date': datas})
    abev1 = pd.concat([abev1,novas_datas], ignore_index = True)
    for i in range(2,len(abev1['date'])):
        if i >= 3:  # Para garantir que temos 3 valores anteriores
            abev1.loc[i, 'close'] = abev1['close'].iloc[i-3:i].mean()
    
    abev1['DateUnix'] = (abev1['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    abev1['DateUnix2'] = (abev1['DateUnix'])**2

    # Calcula a média móvel dos últimos 3 dias e o seu quadrado
    abev1['Med_movel'] = abev1['close'].rolling(window=3).mean()
    abev1['Med_movel_2'] = abev1['Med_movel']**2

    # Definindo a coluna 'date' como índice
    abev1.set_index('date', inplace=True)
    abev1 = abev1.dropna()
    resultado = pd.DataFrame(abev1['close'])
    abev1.drop('close',axis=1,inplace = True)

    print('===============================================================')

    # Faça previsões no conjunto final
    y_pred = model.predict(abev1)
    resultado['retorno_sentido']=  (resultado['close']> resultado['close'].shift(1)).astype(int)
    resultado['previsao'] = y_pred

    grafico_uma_linha(resultado.index,resultado['previsao'], f"Gráfico de previsão para os próximos 60 dias ")

    print(resultado)
    return resultado   
          
#===========================================================================
#sumarizando, abev1, abev = dados_acao('BBDC4.SA')

# Entrada do usuário
acao_nome = input("Digite o nome da ação (ex: BBDC4.SA): ")
data_inicio_str = input("Digite a data de início (AAAA-MM-DD): ")
data_fim_str = input("Digite a data de fim (AAAA-MM-DD, deixe em branco para hoje): ")
# Verifica se a data final foi informada
data_fim = None
if data_fim_str:
  data_fim = data_fim_str

sumarizando, abev1, abev = dados_acao(acao_nome, data_inicio_str, data_fim)

grafico_uma_linha(abev1.index,abev1['close'], "Gráfico do Fechamento da Ação do Bradesco", 'Valor de Fechamento')
grafico_de_barras(sumarizando.index,sumarizando['volume_mensal'],'Gráfico do Volume Mensal retirado dados discrepantes')
grafico_de_dispersao (abev1['volume'] , abev1['close'] , "Relação entre a variação do Volume Negociado com o Preço de Fechamento", 'Volume', 'Preço de Fechamento')
print('===============================================================')
print('Simulando para previsão com dados open e volume, para previsões futuras precisaria definir eses 2 dados \n')
modelo_regressao_scikit(abev1['close'],abev1[['open','volume']],'Considerando dados open e volume')
print('===============================================================')
print('===============================================================')
print('Simulando para previsão com dados close que mé media móvel e data, para previsões futuras mais fácil de estimular\n')
model= modelo_regressao_scikit(abev1['close'],abev1[['DateUnix','DateUnix2', 'Med_movel', 'Med_movel_2']], 'Considerando dados simulados de close e apenas a data')

# Data atual
ultimo_dia = abev1.index[-1]
data_atual = ultimo_dia
proximo_dia_util = ultimo_dia + pd.tseries.offsets.BDay(1)
# Criando uma série de datas para os próximos 60 dias
datas_futuras = pd.date_range(start=proximo_dia_util, periods=60, freq='D') 
resultado = previsao_futuro(abev1[['close','DateUnix','DateUnix2', 'Med_movel', 'Med_movel_2']].tail(10),model,datas_futuras)
if not sumarizando.empty and not abev1.empty: 
    previsao_proximo_dia = resultado.loc[proximo_dia_util, 'previsao'] if proximo_dia_util in resultado.index else "Não disponível"
    retorno_sentido = resultado.loc[proximo_dia_util, 'retorno_sentido'] if proximo_dia_util in resultado.index else "Não disponível"
    print(f"\nPrevisão para o próximo dia útil ({proximo_dia_util.strftime('%Y-%m-%d')}): {previsao_proximo_dia}")
    print(f"Sentido (1 para subir, 0 para cair): {retorno_sentido}")

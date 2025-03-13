import seaborn as sns
from yahooquery import Ticker
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Ignorando warnings de 'FutureWarning' e 'SettingWithCopyWarning' para não poluir a saída
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)  # Adicionando ignorar para SettingWithCopyWarning


#===========================================================================================
def dados_acao(acao='BBDC4.SA', data_inicio="2010-1-1", data_fim=None):
    """
    Função para obter os dados históricos de uma ação específica através do Yahoo Finance.
    
    Parâmetros:
        acao (str): O ticker da ação que você quer buscar (ex: 'BBDC4.SA').
        data_inicio (str): Data de início para a busca de dados (formato 'AAAA-MM-DD').
        data_fim (str): Data de fim para a busca de dados (formato 'AAAA-MM-DD').
        
    Retorna:
        tuple: DataFrame com os dados sumarizados, DataFrame original com dados diários, e objeto Ticker.
    """
    # Definindo os tickers, aqui apenas um ticker é utilizado
    tickers = [acao]

    # Criando um DataFrame vazio para armazenar os dados
    new_data = pd.DataFrame()

    # Obtendo os dados históricos da ação
    try:
        # Criando o objeto Ticker
        abev = Ticker(tickers)

        # Obtendo histórico da ação em um DataFrame
        abev1 = pd.DataFrame(abev.history(start=data_inicio, end=data_fim))

        # Resetando o índice para transformar a coluna de datas em uma coluna regular
        abev1 = abev1.reset_index()
        # Convertendo a coluna 'Date' para o formato datetime e removendo o fuso horário
        abev1['date'] = pd.to_datetime(abev1['date'], utc=True).dt.tz_localize(None)

        # Calculando o retorno simples como porcentagem
        abev1['retorno_simples'] = (abev1['close'] / abev1['close'].shift(1) - 1) * 100
        
        # Calculando o volume simples como porcentagem
        abev1['volume_simples'] = (abev1['volume'] / abev1['volume'].shift(1) - 1) * 100

        # Definindo as features (X) - data e número Unix
        abev1['DateUnix'] = (abev1['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        abev1['DateUnix2'] = (abev1['DateUnix']) ** 2

        # Fazendo a limpeza dos dados, removendo zeros e valores extremos
        abev1 = abev1[(abev1['volume'] != 0) & (abev1['close'] != 0)]
        abev1 = abev1[(abev1['volume_simples'].abs() <= 300)]
        abev1 = abev1.dropna()

        # Definindo a coluna 'date' como índice
        abev1.set_index('date', inplace=True)

        # Sumarizando os dados por mês
        sumarizando = abev1.resample('M').agg({
            'close': ['mean', 'median'], 
            'volume': 'sum', 
            'retorno_simples': 'sum', 
            'volume_simples': 'sum'
        })

        # Reorganizando o MultiIndex para colunas com nomes mais amigáveis
        sumarizando.columns = sumarizando.columns.map({
            ('close', 'mean'): 'fechamento_media',
            ('close', 'median'): 'fechamento_mediana',
            ('volume', 'sum'): 'volume_mensal',
            ('retorno_simples', 'sum'): 'retorno_porcent',
            ('volume_simples', 'sum'): 'volume_porcent',
        })

        # Convertendo o volume para milhões de reais
        sumarizando['volume_mensal'] = sumarizando['volume_mensal'] / 1_000_000

        # Filtrando a sumarização para evitar dados discrepantes
        sumarizando = sumarizando[
            (sumarizando['retorno_porcent'].abs() <= 100) &
            (sumarizando['volume_porcent'].abs() != 0) &
            (sumarizando['retorno_porcent'].abs() != 0)
        ]

        return sumarizando, abev1, abev  # Retorna os dados sumarizados, o DataFrame original e o objeto Ticker

    except Exception as e:  # Trata exceções para conexões com o Yahoo Finance e outros possíveis erros
        print(f"Erro  : {e}")  # Exibe a mensagem de erro

#===========================================================================================
def grafico_uma_linha(x, y, nome='Valores de Fechamento da Ação', label_y='Fechamento'):
    """
    Plota um gráfico de linha com uma variável.

    Parâmetros:
        x (iterable): Valores do eixo X (ex: datas).
        y (iterable): Valores do eixo Y (ex: preços de fechamento).
        nome (str): Título do gráfico.
        label_y (str): Rótulo do eixo Y.
    """
    plt.figure(figsize=(12, 6))  # Ajusta o tamanho da figura
    plt.plot(x, y, linestyle='-', color='b', label=label_y)  # Plota os dados
    plt.title(nome)  # Define o título do gráfico
    plt.xlabel('Data')  # Rótulo do eixo X
    plt.ylabel('Valor')  # Rótulo do eixo Y
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.show()  # Exibe o gráfico

#===========================================================================================
def grafico_de_barras(x, y, nome, label_y1="Fechamento"):
    """
    Plota um gráfico de barras.

    Parâmetros:
        x (iterable): Valores do eixo X (ex: categorias).
        y (iterable): Valores do eixo Y (ex: quantidades).
        nome (str): Título do gráfico.
        label_y1 (str): Rótulo do eixo Y (opcional).
    """
    plt.figure(figsize=(10, 5))  # Ajusta o tamanho da figura
    sns.barplot(x=x, y=y)  # Plota os dados
    plt.title(nome)  # Define o título do gráfico
    plt.xlabel('Categorias')  # Rótulo do eixo X
    plt.ylabel(label_y1)  # Rótulo do eixo Y
    plt.show()  # Exibe o gráfico

#===========================================================================================
def grafico_de_dispersao(y1, y2, titulo, nome_eixo_1, nome_eixo_2):
    """
    Plota um gráfico de dispersão entre duas variáveis.

    Parâmetros:
        y1 (iterable): Valores do eixo Y1 (ex: Volume).
        y2 (iterable): Valores do eixo Y2 (ex: Preços).
        titulo (str): Título do gráfico.
        nome_eixo_1 (str): Rótulo do eixo X.
        nome_eixo_2 (str): Rótulo do eixo Y.
    """
    plt.figure(figsize=(12, 6))  # Ajusta o tamanho da figura
    plt.scatter(y1, y2)  # Plota os dados de dispersão
    plt.title(titulo)  # Define o título do gráfico
    plt.xlabel(nome_eixo_1)  # Rótulo do eixo X
    plt.ylabel(nome_eixo_2)  # Rótulo do eixo Y
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.show()  # Exibe o gráfico

#===========================================================================================
def grafico_2_linhas(x, y1, y2, titulo='Gráfico de Linhas', rotulo_x='Eixo X', rotulo_y='Eixo Y', label_y1="Y1", label_y2="Y2"):
    """
    Plota um gráfico de linhas com duas variáveis.

    Parâmetros:
        x (iterable): Valores do eixo X (ex: datas).
        y1 (iterable): Valores do eixo Y1 (ex: Fechamento real).
        y2 (iterable): Valores do eixo Y2 (ex: Previsão).
        titulo (str): Título do gráfico.
        rotulo_x (str): Rótulo do eixo X.
        rotulo_y (str): Rótulo do eixo Y.
        label_y1 (str): Rótulo da linha Y1.
        label_y2 (str): Rótulo da linha Y2.
    """
    plt.figure(figsize=(10, 5))  # Ajusta o tamanho da figura

    # Plotando as linhas
    plt.plot(x, y1, label=label_y1, color='blue', marker='o')  # Linha para y1
    plt.plot(x, y2, label=label_y2, color='red', marker='x')  #
    # Linha para y2

    # Adicionando título e rótulos
    plt.title(titulo)  # Define o título do gráfico
    plt.xlabel(rotulo_x)  # Rótulo do eixo X
    plt.ylabel(rotulo_y)  # Rótulo do eixo Y

    # Adicionando legenda para as linhas
    plt.legend()  # Mostra a legenda no gráfico

    # Exibindo o gráfico
    plt.grid()  # Adiciona grade ao gráfico
    plt.show()  # Exibe o gráfico

#===========================================================================================
def modelo_regressao_scikit(dados, demais_dados, complemento):
    """
    Cria e treina um modelo de regressão linear para prever preços de ações.

    Parâmetros:
        dados (pd.Series): Preços de fechamento da ação.
        demais_dados (pd.DataFrame): DataFrame com outras features (ex: abertura e volume).
        complemento (str): Descrição complementar para mensagens de saída.

    Retorna:
        model: O modelo de regressão treinado.
    """
    # Tamanho da amostra a ser prevista (1/3 dos dados)
    tamanho = len(dados) // 3

    # Define as features (X) e o target (y)
    X = (demais_dados.shift(-tamanho)).dropna()  # Features para prever
    y = (dados.shift(-tamanho)).dropna()  # Previsão a partir dos dados de fechamento

    # Divida os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crie e treine o modelo de regressão linear
    model = LinearRegression()  # Inicializa modelo de regressão linear
    model.fit(X_train, y_train)  # Treina o modelo

    # Faça previsões no conjunto de teste
    y_pred = model.predict(X_test)  # Previsões para o conjunto de teste

    # Avalie o modelo
    mse = mean_squared_error(y_test, y_pred)  # Erro médio quadrático
    rmse = np.sqrt(mse)  # Raiz do erro médio quadrático
    print(f'Erro médio quadrático (MSE): {mse}')  # Exibe MSE
    print(f'Raiz do erro médio quadrático (RMSE): {rmse}')  # Exibe RMSE

    # Cria um DataFrame para armazenar resultados
    resultado = pd.DataFrame((dados.tail(tamanho)).dropna())  # Dados reais

    # Faça previsões para o conjunto final
    y_pred = model.predict((demais_dados.tail(tamanho)).dropna())  # Previsões para dados finais

    resultado['previsao'] = y_pred  # Armazena as previsões
    # Calcula o erro percentual
    resultado['erro_%'] = ((resultado['previsao'] - resultado['close']) / resultado['close']) * 100.0

    # Gráfico do erro na previsão em porcentagem
    grafico_uma_linha(resultado.index, resultado['erro_%'], f"Gráfico do Erro na previsão em (%) com (MSE): {mse}")
    
    # Gráfico dos valores reais e previstos
    grafico_2_linhas(resultado.index, resultado['close'], resultado['previsao'], f'Correto vs Previsto dos dados - DADOS TESTE - {complemento}', 'Data', 'Fechamento', 'Fechamento Real', 'Previsão')

    print(resultado)  # Exibe resultados finais

    return model  # Retorna o modelo treinado

#===========================================================================
def previsao_futuro(abev1, model, datas):
    """
    Faz previsões futuras de preços com base em um modelo de regressão linear.

    Parâmetros:
        abev1 (pd.DataFrame): Histórico de preços da ação.
        model: O modelo de regressão treinado.
        datas (pd.date_range): Faixa de datas futuras para previsão.

    Retorna:
        resultado: DataFrame com previsões e dados originais.
    """
    # Resetando o índice para transformar a coluna de datas em uma coluna regular
    abev1 = abev1.reset_index()
    tamanho= len(abev1['date'])
    fechamento = abev1['close']
    # Define as features (X)  - apenas data e data Unix - numeros, seu quadrado
    novas_datas = pd.DataFrame({'date': datas})
    abev1 = pd.concat([abev1,novas_datas], ignore_index = True)
    for i in range(tamanho,len(abev1['date'])):
        if i >= 3:  # Para garantir que temos 60 valores anteriores
            abev1.loc[i, 'open'] = abev1['open'].iloc[i-60:i].mean()
            abev1.loc[i, 'volume'] = abev1['volume'].iloc[i-60:i].mean()
            abev1.loc[i, 'close'] = abev1['close'].iloc[i-60:i].mean()


     # Definindo a coluna 'date' como índice
    abev1.set_index('date', inplace=True)   
    print('===============================================================')

    # Faça previsões no conjunto final
    x_pred = abev1[['open','volume']]

    abev1['previsao'] = model.predict(x_pred)
 
    resultado = abev1.tail(len(abev1)-tamanho+1)

    #grafico_uma_linha(resultado.index,resultado['previsao'], f"Gráfico de previsão para os próximos 60 dias ")
    grafico_2_linhas(resultado.index, resultado['close'], resultado['previsao'], f'Previsto vs Média móvel dos dados ', 'Data', 'Fechamento', 'Fechamento Média Móvel', 'Previsão')

    resultado['retorno_sentido']=  (resultado['previsao']> resultado['previsao'].shift(1)).astype(int)

    return resultado

#===========================================================================================
# Código principal para execução do script

"""
Execução principal do script. Obtém dados de ações, gera gráficos e realiza previsões.
"""
# Solicitar informações do usuário
acao = input("Digite o ticker da ação (ex: BBDC4.SA): ")  # Ticker da ação desejada
if acao == "":
    acao = "BBDC4.SA"
data_inicio = input("Digite a data de início (AAAA-MM-DD): ")  # Data de início para busca de dados
if data_inicio=="":
    data_inicio = "2010-01-01"
data_fim = input("Digite a data de fim (AAAA-MM-DD): ")  # Data de fim para busca de dados
if data_fim=="":
    data_fim = None

# Obtendo e processando os dados da ação
sumarizado, dados, obj = dados_acao(acao, data_inicio, data_fim)  # Chama a função para obter dados

# Exibindo gráfico de fechamento da ação
grafico_uma_linha(dados.index, dados['close'], 'Fechamento das Ações', 'Fechamento')

# Exibindo gráfico de volume mensal das ações
grafico_de_barras(sumarizado.index, sumarizado['volume_mensal'], 'Volume Mensal das Ações')  # Gráfico de barras para volume mensal

# Exibindo gráfico de dispersão entre volume e fechamento
grafico_de_dispersao(dados['volume'], dados['close'], 'Dispersão entre Volume e Fechamento', 'Volume', 'Fechamento')  # Gráfico de dispersão

# Treinando o modelo de regressão
model = modelo_regressao_scikit(dados['close'], dados[['open', 'volume']], acao)  # Chama a função para modelar os dados

# Data Importantes
ultimo_dia = dados.index[-1]
data_atual = ultimo_dia
proximo_dia_util = ultimo_dia + pd.tseries.offsets.BDay(1)

# Previsão dos próximos 60 dias úteis
data_futura = pd.date_range(start=proximo_dia_util, periods=60, freq='D')  # Gera uma faixa de 60 dias úteis
previsao = previsao_futuro(dados, model, data_futura)  # Chama a função para prever futuros preços

print("Previsões para os próximos 60 dias:")
print(previsao[['previsao','retorno_sentido']])  # Exibindo as previsões finais

if not sumarizado.empty and not previsao.empty:
    previsao_proximo_dia = previsao.loc[proximo_dia_util, 'previsao'] if proximo_dia_util in previsao.index else "Não disponível"
    retorno_sentido = previsao.loc[proximo_dia_util, 'retorno_sentido'] if proximo_dia_util in previsao.index else "Não disponível"
    print(f"\nPrevisão para o próximo dia útil ({proximo_dia_util.strftime('%Y-%m-%d')}): {previsao_proximo_dia}")
    print(f"Sentido (1 para subir, 0 para cair): {retorno_sentido}")

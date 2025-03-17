# 1.Importar os csv do  EcomPlus SalesData Dataset no site https://leandrolessa.com.br/datasets/
# 2. Após a importação combinar os arquivos em um dataframe pandas
# 3. Utilizando o Dataframe mesclado, levantar as estatísticas solicitadas
# 4. Importar a seaborn para análise dos gráficos

import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

diretorio = os.path.join(os.path.dirname(os.path.abspath(__file__)),"")
print(diretorio)

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
def grafico_de_dispersao(x, y, hue, data, titulo, nome_eixo_1, nome_eixo_2):
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
    sns.scatterplot(x=x, y=y, hue=hue, data=data)  # Plota os dados de dispersão
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
def modelo_regressao_stats(y, X, complemento):
    X = sm.add_constant(X)

    # Crie o modelo
    model = sm.OLS(y, X)

    # Ajusta o modelo
    resultado = model.fit()

    # Imprime os resultados da regressão
    print(resultado.summary())

    return resultado

#===========================================================================================
def check_data_quality(df, df_name):
    print(f"Análise de Qualidade de Dados para '{df_name}':")
    print("\nDados faltantes:")
    print(df.isnull().sum())
    print("\nLinhas duplicadas:", df.duplicated().sum())
    # Exibindo dados duplicados (opcional, pode ser comentado se não precisar visualizar)
    #print("\nDados duplicados:\n", df[df.duplicated(keep=False)])
    print("-" * 40)
#===========================================================================================
def plot_grafico_pie(df,value_col, *groupby_cols):
    """
    Gera gráficos de pizza para o valor total da coluna especificada, agrupados pelas colunas fornecidas.

    Parameters:
    df (pd.DataFrame): DataFrame contendo os dados.
    value_col (str): Nome da coluna com valores float a ser somado.
    groupby_cols (str): Nome das colunas a serem usadas para agrupamento.
    """
    # Agrupando os dados
    grouped_data = df.groupby(list(groupby_cols))[value_col].sum().unstack()

    # Verificando se há mais de um grupo
    if len(grouped_data) == 0:
        print("Nenhum dado encontrado para plotagem.")
        return

    # Criando gráficos de pizza
    fig, axes = plt.subplots(1, len(grouped_data), figsize=(14, 7), squeeze=False)

    for i, (name, values) in enumerate(grouped_data.iterrows()):
        axes[0][i].pie(values, labels=values.index, autopct='%1.1f%%', startangle=90)
        axes[0][i].set_title(f'Distribuição - {name}')

    # Ajustando o layout
    plt.tight_layout()
    plt.show()

#===========================================================================================   

def correlacao_e_cluster(df, value_col, qty_col, *groupby_cols):
    """
    Analisa a correlação entre as variáveis numéricas e realiza clusterização.

    Parameters:
    df (pd.DataFrame): DataFrame contendo os dados.
    value_col (str): Nome da coluna com valores float a ser utilizado na análise.
    qty_col (str): Nome da coluna com Id dos Estados como inteiros a ser utilizado na análise.
    groupby_cols (str): Nome das colunas a serem usadas para agrupamento.
    """
    
    # Agrupando dados para somar os valores
    grouped_data = df.groupby(list(groupby_cols)).agg({value_col: 'sum', qty_col: 'sum'}).reset_index()
    
    # Análise de Correlação
    correlation_matrix = grouped_data[[value_col, qty_col]].corr()
    print("\nMatriz de correlação:\n", correlation_matrix)
    
    # Visualização da matriz de correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlação')
    plt.show()

    # Normalizando os dados para clusterização
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(grouped_data[[value_col, qty_col]])
    
    # Aplicando K-means
    kmeans = KMeans(n_clusters=3, random_state=42)  # Escolha o número de clusters que achar apropriado
    grouped_data['Cluster'] = kmeans.fit_predict(data_scaled)

    # Visualizando os clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=value_col, y=qty_col, hue='Cluster', data=grouped_data, palette='Set1')
    plt.title('Clusters de Compras')
    plt.xlabel(value_col)
    plt.ylabel(qty_col)
    plt.legend(title='Cluster')
    plt.show()


#===========================================================================================   
# Função para encontrar o produto mais e menos vendido em cada estado
def produtos_extremos(estado_id):
    produtos_estado = produtos_por_estado[produtos_por_estado['ID_Estado'] == estado_id]
    if not produtos_estado.empty:
      produto_mais_vendido = produtos_estado.loc[produtos_estado['Quantidade'].idxmax()]
      produto_menos_vendido = produtos_estado.loc[produtos_estado['Quantidade'].idxmin()]
      return produto_mais_vendido, produto_menos_vendido
    else:
      return None, None


#===========================================================================================  

def imprime_boxplot(regiao_imp):
  plt.figure(figsize=(12, 6))
  sns.boxplot(x='valor_venda', y='Categoria', data=regiao_imp)
  plt.title('Distribuição das Vendas por Categoria (Região Sudeste)')
  plt.xlabel('Categoria do Produto')
  plt.ylabel('Valor da Venda')
  plt.xticks(rotation=45, ha='right')  # Rotaciona os rótulos do eixo x para melhor visualização
  plt.show()

#===========================================================================================   
# 1. Importar os arquivos CSV
try:
    clientes = pd.read_csv(f'{diretorio}/clientes.csv') # Specify encoding if needed
    produtos = pd.read_csv(f'{diretorio}/ecommerce_produtos.csv')
    vendas = pd.read_csv(f'{diretorio}/vendas.csv')
    regiao = pd.read_csv(f'{diretorio}/estado_regiao.csv', encoding='latin-1', sep=";")
except FileNotFoundError:
    print("Um ou mais arquivos CSV não foram encontrados. Certifique-se de que os arquivos estejam no mesmo diretório ou forneça o caminho correto.")
    exit(0)

#renomeando colunas para relacionar arquivos
produtos = produtos.rename(columns={'SKU':'SKU_Produto'})
regiao = regiao.rename(columns={'id_estado': 'ID_Estado' })


# 2. Combinar os arquivos usando join (especifique as colunas de join)
# Exemplo: Supondo que 'cliente_id' esteja presente em todos os dataframes

try:
    # Merge vendas and clientes
    df_matriz = pd.merge(clientes, vendas, on='ID_Cliente', how='left')  

    # Merge com produtos
    df_matriz = pd.merge(df_matriz, produtos, on='SKU_Produto', how='left')
        
    # Merge com regiao
    
    df_matriz = pd.merge(df_matriz, regiao, on='ID_Estado', how='left')

except KeyError as e:
    print(f"Erro: Coluna de chave '{e}' não encontrada em um dos DataFrames. Verifique se as colunas de junção estão presentes e corretamente nomeadas.")
    exit()
except Exception as e:
    print(f"Um erro ocorreu durante o merge: {e}")
    exit()



df_matriz = df_matriz.dropna()
# Inclui coluna do valor de venda
df_matriz['valor_venda'] = df_matriz['Preco'].astype(float) * df_matriz['Quantidade'].astype(float)

print(df_matriz.columns)

print("================================================================================================")

# Verificando dados faltantes

# Aplicando a função para cada DataFrame
check_data_quality(clientes, "Clientes")
check_data_quality(produtos, "Produtos")
check_data_quality(vendas, "Vendas")
check_data_quality(regiao, "Região")
print("================================================================================================")


# Identificar o perfil demográfico dos clientes, incluindo distribuição por estado e canal de vendas preferido.
canal_regiao = df_matriz.groupby(['Canal','regiao'])['valor_venda'].sum().unstack().plot(kind='bar', stacked=True)

plt.title('Vendas por Canal e Região')
plt.ylabel('Valor Vendas')
plt.xlabel('Canal')
plt.legend(title='Região')
plt.show()
print("================================================================================================")

print(canal_regiao)
print("Gráfico do Valor da Venda por Canal")
plot_grafico_pie(df_matriz,'valor_venda', 'Canal', 'regiao')
print("================================================================================================")

print("Gráfico de Quantidade Vendida por Canal")
plot_grafico_pie(df_matriz,'Quantidade', 'Canal', 'regiao')
print("================================================================================================")



# Contagem de compras por canal
count_canal = df_matriz.groupby('Canal').size()
print("\nQuantidade de compras por canal:\n", count_canal)
print("================================================================================================")

# Quantidade de compras por canal
count_canal = df_matriz.groupby('Canal')['Quantidade'].sum().astype(int)
print("\nQuantidade comprada por canal:\n", count_canal)
print("================================================================================================")

# Contagem de compras por região
count_regiao = df_matriz.groupby('regiao').size()
print("\nQuantidade de compras por região:\n", count_regiao)

#Quantidade Comprada por região
count_regiao = df_matriz.groupby('regiao')['Quantidade'].sum().astype(int)
print("\nQuantidade comprada por região:\n", count_regiao)
print("================================================================================================")

# Agrupamento e soma das vendas por estado
vendas_por_estado = df_matriz.groupby('estado')['valor_venda'].sum()

# Encontrando os estados com mais e menos vendas
estado_mais_vendas = vendas_por_estado.idxmax()
estado_menos_vendas = vendas_por_estado.idxmin()

print(f"Estado com mais vendas: {estado_mais_vendas} (R${vendas_por_estado.max():.2f})")
print(f"Estado com menos vendas: {estado_menos_vendas} (R${vendas_por_estado.min():.2f})")
print("================================================================================================")

# Gráfico de barras para vendas por estado
plt.figure(figsize=(12, 6))
vendas_por_estado.sort_values().plot(kind='bar')
plt.title('Vendas por Estado')
plt.xlabel('Estado')
plt.ylabel('Valor Total de Vendas')
plt.show()
print("================================================================================================")


# Produtos mais e menos vendidos por estado
produtos_por_estado = df_matriz.groupby(['ID_Estado', 'SKU_Produto', 'Nome_Produto'])['Quantidade'].sum().reset_index()


print("================================================================================================")

# Aplicando a função para cada estado
for estado in df_matriz['ID_Estado'].unique():
    mais_vendido, menos_vendido = produtos_extremos(estado)
    nome_estado = df_matriz[df_matriz['ID_Estado']== estado ]['estado'].iloc[0]
    if mais_vendido is not None:
        print(f"\nEstado: {nome_estado}")
        print(f"Produto mais vendido: {mais_vendido['SKU_Produto']} - {mais_vendido['Nome_Produto']} (Quantidade: {mais_vendido['Quantidade']})")
        print(f"Produto menos vendido: {menos_vendido['SKU_Produto']} - {menos_vendido['Nome_Produto']} (Quantidade: {menos_vendido['Quantidade']})")

print("================================================================================================")

# Análise por canal
vendas_por_canal = df_matriz.groupby('Canal')['valor_venda'].sum()
print("\nVendas por canal:\n", vendas_por_canal)
print("================================================================================================")

# Análise por canal e estado
vendas_por_canal_estado = df_matriz.groupby(['estado','Canal'])['valor_venda'].sum().unstack()
print("\nVendas por canal e estado:\n", vendas_por_canal_estado)

print("================================================================================================")

# Análise por canal e região
vendas_por_canal_regiao = df_matriz.groupby(['Canal', 'regiao'])['valor_venda'].sum().unstack()
print("\nVendas por canal e região:\n", vendas_por_canal_regiao)

print("================================================================================================")

# Análise por região
vendas_por_regiao = df_matriz.groupby('regiao')['valor_venda'].sum()
print("\nVendas por região:\n", vendas_por_regiao)

print("================================================================================================")

# Análise de vendas por categoria com valor médio
preco_medio = df_matriz.groupby('Categoria')['Preco'].mean()
print("\nO Preco  médio  por categoria:\n", preco_medio)

print("================================================================================================")

regiao_maior_venda = df_matriz.groupby('regiao')['valor_venda'].sum().idxmax()
valor_maior_venda = df_matriz.groupby('regiao')['valor_venda'].sum().max()

print(f"A região com o maior valor total em vendas é {regiao_maior_venda} com R${valor_maior_venda:.2f}")

print("================================================================================================")

#Elabore um boxplot para representar a distribuição dos valores das vendas dos produtos da região sudeste, 
#organizados conforme sua categoria. Selecione o tipo de gráfico que melhor capture essa distribuição.

sudeste_df = df_matriz[df_matriz['regiao'] == 'Sudeste']
imprime_boxplot(sudeste_df)


print("================================================================================================")


df_marco = df_matriz[pd.to_datetime(df_matriz['Data']).dt.month == 3]

# Group by 'Canal' and sum 'valor_venda'
canal_vendas_marco = df_marco.groupby('Canal')['valor_venda'].sum()

# Find the channel with the highest sales
maior_canal_vendas = canal_vendas_marco.idxmax()

print(f"O canal com maior número de vendas em Março foi: {maior_canal_vendas}")

print("================================================================================================")
df_filtrado = df_matriz[(df_matriz['regiao'] == 'Sul') & (df_matriz['Canal'] == 'Online')]
grafico_de_dispersao('Preco', 'valor_venda', 'Categoria', df_filtrado, 'Gráfico de Dispersão: Preço vs. Valor da Venda (Região Sul, Compras Online)', 'Preco', 'Valor da Venda')


print("================================================================================================")

# Encontre o estado com o maior número de compras de produtos esportivos
compras_esporte_por_estado = df_matriz[df_matriz['Categoria'] == 'Esporte'].groupby('estado')['Quantidade'].sum()
print(compras_esporte_por_estado.sort_values(ascending = False))
estado_maior_compras_esporte = compras_esporte_por_estado.idxmax()
print(f"O estado com o maior número de compras de produtos esportivos é: {estado_maior_compras_esporte}")

print("================================================================================================")

# Encontrar o produto com o maior número de vendas
produto_mais_vendido = df_matriz.groupby('Nome_Produto')['Quantidade'].sum().idxmax()
quantidade_produto_mais_vendido = df_matriz.groupby('Nome_Produto')['Quantidade'].sum().max()

print(f"O produto mais vendido é '{produto_mais_vendido}' com {quantidade_produto_mais_vendido} unidades vendidas.")

print("================================================================================================")

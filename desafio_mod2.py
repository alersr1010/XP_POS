# 1.Importar os csv do  breast_cancer  Dataset no site https://leandrolessa.com.br/datasets/
# 2. Após a importação combinar os arquivos em um dataframe pandas
# 3. Utilizando o Dataframe mesclado, levantar as estatísticas solicitadas
# 4. Importar a seaborn para análise dos gráficos

import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks



diretorio = os.path.join(os.path.dirname(os.path.abspath(__file__)),"")
#diretorio = "sample_data"
print(diretorio)
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

def modelo_regressao_stats(y, X):
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
    print("-" * 80)
    print(f"Análise de Qualidade de Dados para '{df_name}':")
    print("\nDados Ausentes:")
    print(f"Nulo: {df.isnull().sum()}")
    print("-" * 80)
    print("\nLinhas duplicadas:", df.duplicated().sum())
    # Exibindo dados duplicados (opcional, pode ser comentado se não precisar visualizar)
    #print("\nDados duplicados:\n", df[df.duplicated(keep=False)])
    print("-" * 80)
#===========================================================================================

def analisar_dados(y, X):
    # Exemplo de uso
    # y = ... # suas labels
    # X = ... # seus dados de entrada
    # analisar_dados(y, X)
    undersample = RandomUnderSampler(random_state=42)
    x_under, y_under = undersample.fit_resample(X, y)

    t1 = TomekLinks(sampling_strategy='all') # inclui outliers
    X_balanced, y_balanced = t1.fit_resample(x_under, y_under)

    # Dividir os dados em conjuntos de treinamento (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

    # Balancear as classes no conjunto de treinamento usando SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   

    # Inicializar e treinar o modelo Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=100)
    rf.fit(X_train_balanced, y_train_balanced)

    # Fazer previsões
    y_pred = rf.predict(X_test)

    # Calcular a pontuação (acurácia)
    score = rf.score(X_test, y_test)
    print("-"*80)
    print(f"A pontuação (acurácia) do modelo é: {score:.4f}")
    print("-"*80)

    # Avaliar o modelo
    report = classification_report(y_test, y_pred, output_dict=True)

    # Obter a importância das features
    importancias = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Característica': X.columns,
        'Importância': importancias
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importância', ascending=False)

    print("\nImportância das Características:")
    print(feature_importance_df)

    print('-'*80)
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(cm)

    # Precisão para a classe saudável (assumindo que a classe 1 representa pessoas saudáveis)
    precision_saudavel = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print("\nA precisão associada para a classe de pessoas saudáveis é:", precision_saudavel)
    print('-'*80)

    # classe de pessoas classificadas como doentes

    print(f"Valor de Acerto para 1: saudável: {cm[0,0]}")
    print(f"Valor de Acerto para 2: doentes: {cm[1,1]}")

    # Extrair o recall da classe 2 (doentes) do relatório

    print('-'*80)

    print(f"Valor de Recall para 1: saudável: {report['1']['recall']:.4f}")
    print(f"Valor de Recall para 2: doentes: {report['2']['recall']:.4f}")
    print('-'*80)

    print('-'*80)
    # Encontrar a classe com maior suporte no relatório
    print(f"Valor de Suporte para 1: saudável: {report['1']['support']:.4f}")
    print(f"Valor de Suporte para 2: doentes: {report['2']['support']:.4f}")
    print('-'*80)

    # Encontrar a classe com maior suporte no relatório
    print(f"Valor de Precisão para 1: saudável: {report['1']['precision']:.4f}")
    print(f"Valor de Precisao para 2: doentes: {report['2']['precision']:.4f}")
    print('-'*80)

    return report


#===========================================================================================   
# 1. Importar os arquivos CSV
try:
    breast = pd.read_csv(f'{diretorio}/breast_cancer.csv') # Specify encoding if needed
except FileNotFoundError:
    print("Um ou mais arquivos CSV não foram encontrados. Certifique-se de que os arquivos estejam no mesmo diretório ou forneça o caminho correto.")
    exit(0)
print("="*80)
print(f" Tamanho original da base {len(breast)} linhas")
check_data_quality(breast, "Previsão de Câncer de Mama")
print("="*80)
breast_sem_dup = breast.drop_duplicates()
print(f"Tamanho final da base {len(breast_sem_dup)} linhas")
y = pd.DataFrame(breast_sem_dup['Classification']).astype(int)
X = pd.DataFrame(breast_sem_dup.drop(columns=["Classification"]))
analisar_dados(y,X)
print("="*80)
print("1 = Healthy controls: representa indivíduos saudáveis que não apresentam câncer de mama.")
print("2 = Patients: representa indivíduos que foram diagnosticados com câncer de mama. ")
print("Classificação por idade")
print(breast_sem_dup.groupby('Classification')['Age'].mean())

print("="*80)
print("1 = Healthy controls: representa indivíduos saudáveis que não apresentam câncer de mama.")
print("2 = Patients: representa indivíduos que foram diagnosticados com câncer de mama. ")
print("Classificação por Glucose")
print(breast_sem_dup.groupby('Classification')['Glucose'].mean())

print("="*80)
correlacao = breast_sem_dup.corr()
correlacao_com_class = correlacao['Classification'].sort_values(ascending=False)

print("Correlação com classificação:")
print(correlacao_com_class)

print("="*80)
plt.figure(figsize=(12, 6))
sns.histplot(data=breast_sem_dup[breast_sem_dup['Classification'] == 2], x='Age', bins=20)
plt.title('Distribuição de Idades de Pessoas Diagnosticadas com Câncer')
plt.xlabel('Idade')
plt.ylabel('Número de Pessoas')
plt.show()

print("="*80)
grafico_de_dispersao(x='Glucose', y='BMI', hue='Classification', data=breast_sem_dup, titulo='Relação entre Glucose e BMI', nome_eixo_1='Glucose', nome_eixo_2='BMI')

print("="*80)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Glucose', hue='Classification', data=breast_sem_dup,  gap=0.2, palette="flare")
plt.title('Boxplot da Glucose por Classificação')
plt.ylabel('Classificação')
plt.xlabel('Glucose')
plt.show()

print("="*80)

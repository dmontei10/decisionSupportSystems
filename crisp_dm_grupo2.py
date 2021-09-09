#Importação de bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm

#Ler o ficheiro .csv com os dados
dataframe=pd.read_csv(r"C:/Users/bruno/Desktop/bank-additional-full.csv",sep=";")
pd.set_option('display.max_columns', None)

#Renomear colunas, para português
dataframe.rename(columns = {'age':'idade', 
                            'job':'trabalho', 
                            'marital':'estado civil',
                            'education':'educacao',
                            'default':'incumprimento',
                            'housing':'emp.predial',
                            'loan':'emp.pessoal',
                            'contact':'contacto',
                            'month':'mes',
                            'day_of_week':'dia_da_semana',
                            'duration':'duracao',
                            'campaign':'campanha',
                            'pdays':'diasP',
                            'previous':'anterior',
                            'poutcome':'resultadoP',
                            'emp.var.rate':'taxa.var.emp',
                            'cons.price.idx':'ind.preco.cons',
                            'cons.conf.idx':'ind.conf.cons',
                            'nr.employed':'nr.empregados',
                            'y':'s'}, inplace = True)

#Analise descritiva
print(dataframe.info())      #Imprimir informacao sobre os dados
print(dataframe.head(5))     #Imprimir as primeiras 5 linhas do dataframe
print(dataframe.tail(5))     #Imprimir as ultimas 5 linhas do dataframe
print(dataframe.describe())  #Imprimir estatistica sobre os dados
print(pd.isna(dataframe))    #Imprimir valores nulos nos dados

#Definicao de variaveis continuas e impressao do nome das colunas
variaveis_cont = dataframe.describe().columns
print(variaveis_cont)

#Definicao de variaveis categoricas e impressao do nome das colunas
variaveis_categ = dataframe.describe(include=[object]).columns
print(variaveis_categ)

#Criacao dos histogramas, sobre as variaveis numericas
dataframe.hist(column = variaveis_cont, figsize = (20,20))
plt.show()

#Criacao dos graficos de barras, sobre as variaveis categoricas
fig, axes = plt.subplots(4, 3, figsize = (20, 20))
plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.7, hspace = 0.3)
for i, ax in enumerate(axes.ravel()):
    if i > 10:
        ax.set_visible(False)
        continue
    sns.countplot(y = variaveis_categ[i], data = dataframe, ax = ax)
plt.show()

#Criacao das boxplots, sobre as variaveis numericas
fig, axes = plt.subplots(4, 3, figsize = (20, 20))
plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.7, hspace = 0.3)
for i, ax in enumerate(axes.ravel()):
    if i > 9:
        ax.set_visible(False)
        continue
    sns.boxplot(x = variaveis_cont[i], data = dataframe, ax = ax)
plt.show()

#Converter coluna diasP de numerica para categorica
dataframe["cat_diasP"] = [0 if each == 999  else 1 for each in dataframe.diasP]
dataframe = dataframe.drop(["diasP"],axis = 1)

#Criacao da tabela de correlacao
correlacao = dataframe.corr(method = "pearson")
plt.figure(figsize = (25,10))
sns.heatmap(correlacao,vmax = 1, square = True, annot = True, cmap="YlOrRd")
plt.show()

#Definicao do fator de analise
fator_X=FactorAnalysis(n_components=1, max_iter=5000)
dataframe["fator_X"]=fator_X.fit_transform(dataframe[['euribor3m','nr.empregados','ind.preco.cons','taxa.var.emp']])
dataframe=dataframe.drop(["euribor3m","nr.empregados","ind.preco.cons","taxa.var.emp"],axis=1)

#Transformacao das colunas de variaveis categoricas do dataframe
#de forma a que sejam aumentadas as colunas, com todas as opcoes possiveis
#Conversao em forma de matriz com 0s e 1s, consoante os clientes
#Utilizacao da funcao get_dummies()
columns = dataframe.select_dtypes(include = [object]).columns
dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[columns])], axis = 1)
dataframe = dataframe.drop(['trabalho', 'estado civil', 'educacao', 'incumprimento', 'emp.predial', 'emp.pessoal',
       'contacto', 'mes', 'dia_da_semana','resultadoP', 's', 'duracao'], axis = 1)
print(dataframe.info(),"\n \n")

#Normalizacao dos dados, para nao se perder informacao ou 
#para nao distorcer diferencas nos intervalos de valores
#Utilizacao do metodo MinMaxScaler()
escalar_min_max = preprocessing.MinMaxScaler()
dados_escalados = pd.DataFrame(escalar_min_max.fit_transform(dataframe),columns = dataframe.columns)

#Criacao da variavel y, com os valores de cada uma das linhas da coluna s_yes
#de dados_escalados
y = dados_escalados.s_yes
dados_escalados = dados_escalados.drop(['s_yes','s_no'], axis = 1)

#Divisao do conjunto de dados em dados de treino e de teste, aleatoriamente
X_train, X_test, y_train, y_test = train_test_split(dados_escalados, y, test_size = 0.2, random_state = 42)

#Funcao para aumentar a amostragem para lidar com dados nao balanceados
#Funcao amostragem()
def amostragem(X_train, y_train):
    dataframe_total = pd.concat((X_train, pd.DataFrame({'value': y_train}, index = y_train.index)), axis = 1)
    
    dataframe_maior = dataframe_total [dataframe_total.value==0]
    dataframe_menor = dataframe_total [dataframe_total.value==1]
     
    # classe minoritaria amostragem
    dataframe_menor_amostragem = resample(dataframe_menor, 
                                     replace = True,     # amostra com substituicao
                                     n_samples = dataframe_maior.shape[0],    # igualar classe maioritaria
                                     random_state = 123) # resultados reproduziveis
    # Combinar a classe maioritaria com a classe de amostragem minoritaria
    dataframe_amostragem = pd.concat([dataframe_maior, dataframe_menor_amostragem], axis = 0)
    y_amostragem = dataframe_amostragem.value
    X_amostragem = dataframe_amostragem.drop('value', axis = 1)

    return X_amostragem, y_amostragem

#Aplicar a funcao amostragem() aos dados de treino e teste
X_train,y_train = amostragem(X_train, y_train)
print(X_train)
print(X_test)

"------------------------------------------------------------"
print("------------------------------------------------------------\n")

#Criacao do classificador Random Forest
rf = RandomForestClassifier(n_estimators=10 ,n_jobs=-1,
                            random_state=42, 
                            max_depth= 5)

#Aplicar esse classificador aos dados de treino e teste
#Criar variavel com a previsao dos dados de teste
rf.fit(X_train, y_train)
rf_predicao = rf.predict(X_test)

#Mostrar relatório de classificacao de Random Forest
print("Random Forest","\n")
print("Relatorio Classificacao","\n")
print(classification_report(y_test, rf_predicao))
print("\n")

#Criacao da matriz de confusao de Random Forest
resultado = round(accuracy_score(y_test, rf_predicao),3)
cm1 = cm(y_test, rf_predicao)
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.3, 
            square = True, cmap = 'YlOrRd')
plt.ylabel('Label Atual')
plt.xlabel('Label Prevista')
plt.title('Resultado Acuracia: {0}'.format(resultado), size = 12)
plt.show()

#Aplicar modelo K-Fold
#Obter dados estatisticos como acuracia, desvio padrao e erros
print("Resultados K-fold\n")
acuracias = cross_val_score(rf, X = X_train, y = y_train, cv = 10)
print("Acuracia (media): {:.2f} %" .format(acuracias.mean() * 100))
print("Desvio Padrao: {:.2f} %" .format(acuracias.std() * 100))

resultados = cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=10,)
print ("Erro Absoluto Medio: {:.2f} %\n" .format(resultados.mean()))

#Obter e visualizar os erros de previsao
print("Erros\n")
eam = mean_absolute_error(rf.predict(X_test), y_test)
eqm = mean_squared_error(rf.predict(X_test), y_test)
reqm = np.sqrt(eqm)
print('Erro Absoluto Medio (MAE): {:.2f} %' .format(eam))
print('Erro Quadratico Medio (MSE): {:.2f} %' .format(eqm))
print('Raiz Erro Quadratico Medio (RMSE): {:.2f} %'.format(reqm))

#Criacao de um ranking com as variaveis (features) mais importantes (influentes)
#do classificador Random Forest
ranking = rf.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = dados_escalados.columns
plt.title("Variaveis mais importantes baseadas no Regressor de Random Forest", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="aqua", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()
print("\n")

"------------------------------------------------------------"
print("------------------------------------------------------------\n")

#Criacao do classificador K-NN
k_NN = KNeighborsClassifier(n_neighbors=100,weights="distance")  
k_NN.fit(X_train, y_train)  
y_pred = k_NN.predict(X_test)  

#Mostrar relatorio de classificacao de K-NN
print("K-Nearest Neighbors\n")
print("Relatorio Classificacao\n")
print(classification_report(y_test, y_pred))
print("\n")

#Criacao da matriz de confusao de K-NN
resultado = round(accuracy_score(y_test, y_pred),3) 
cm1 = cm(y_test, y_pred)
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths =.3, 
        square = True, cmap = 'YlOrRd')
plt.ylabel('Label Atual')
plt.xlabel('Label Prevista')
plt.title('Resultado Acuracia: {0}'.format(resultado), size = 12)
plt.show()

#Aplicar modelo K-Fold
#Obter dados estatisticos como acuracia, desvio padrao e erros
print("Resultados K-fold\n")
acuracias = cross_val_score(k_NN, X = X_train, y = y_train, cv = 10)
print("Acuracia (media): {:.2f} %" .format(acuracias.mean() * 100))
print("Desvio Padrao: {:.2f} %" .format(acuracias.std() * 100))

resultados = cross_val_score(k_NN, X_train, y_train, scoring='neg_mean_absolute_error', cv = 10,)
print("Erro Absoluto Medio: {:.2f} %\n" .format(resultados.mean()))

#Obter e visualizar os erros de previsao
print("Erros\n")
eam = mean_absolute_error(k_NN.predict(X_test), y_test)
eqm = mean_squared_error(k_NN.predict(X_test), y_test)
reqm = np.sqrt(eqm)
print('Erro Absoluto Medio (MAE): {:.2f} %' .format(eam))
print('Erro Quadratico Medio (MSE): {:.2f} %' .format(eqm))
print('Raiz Erro Quadratico Medio (RMSE): {:.2f} %'.format(reqm))
print("\n")

"------------------------------------------------------------"
print("------------------------------------------------------------\n")

#Criacao do classificador Regressao Logistica
lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
lr.fit(X_train,y_train)
predicao = lr.predict(X_test)

#Mostrar relatorio de classificacao de Regressao Logistica
print("Regressao Logistica","\n")
print("Relatorio Classificacao","\n")
print(classification_report(y_test, predicao))
print("\n")

#Criacao da matriz de confusao de Regressao Logistica
resultado = round(accuracy_score(y_test, predicao),3) 
cm1 = cm(y_test, predicao)
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.3, 
        square = True, cmap = 'YlOrRd')
plt.ylabel('Label Atual')
plt.xlabel('Label Prevista')
plt.title('Resultado Acuracia: {0}'.format(resultado), size = 12)
plt.show()

#Selecao das variaveis do classificador Regessao Logistica
rfe=RFE(lr,10)
rfe=rfe.fit(X_train,y_train)
print("Selecao Features","\n")
print(X_train[X_train.columns[rfe.ranking_==1].values].columns,"\n")

#Aplicar modelo K-Fold
#Obter dados estatisticos como acuracia, desvio padrao e erros
print("Resultados K-fold\n")
acuracias = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)
print("Acuracia (media): {:.2f} %" .format(acuracias.mean() * 100))
print("Desvio Padrao: {:.2f} %" .format(acuracias.std() * 100))

resultados = cross_val_score(lr, X_train, y_train, scoring='neg_mean_absolute_error', cv=10,)
print ("Erro Absoluto Medio: {:.2f} %\n" .format(resultados.mean()))

#Obter e visualizar os erros de previsao
print("Erros\n")
eam = mean_absolute_error(lr.predict(X_test), y_test)
eqm = mean_squared_error(lr.predict(X_test), y_test)
reqm = np.sqrt(eqm)
print('Erro Absoluto Medio (MAE): {:.2f} %' .format(eam))
print('Erro Quadratico Medio (MSE): {:.2f} %' .format(eqm))
print('Raiz Erro Quadratico Medio (RMSE): {:.2f} %'.format(reqm))
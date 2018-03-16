#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:02:46 2018

@author: danielreyes
"""

import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

def f_importances(coef, names):
    #Transformando os valores do coeficiente em um dataframe de uma linha só
    importance = pd.DataFrame(coef)
    #Transformando os nomes das colunas (variáveis) em um dataframe de uma coluna só    
    feature = pd.DataFrame(names)
    
    #Invertendo linha por coluna no dataframe de importance
    importance = importance.transpose()
    #unindo os dois dataframes
    importances = pd.concat([feature, importance], axis=1, join_axes=[importance.index])
    #Renomeando as colunas
    importances.columns = ['feature', 'importance']
    #Ordenando o dataframe em ordem decrescente de acordo com o valor de importancia
    importances = importances.sort('importance', ascending=False)
    print(importances)
    
def plot_coefficients(classifier, feature_names, top_features=10):
    #Captura os valores de importancia
    coef = classifier.coef_.ravel()
    #pega os x maiores valores (passado por parâmetro)
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    #pega os x menores valores (passado por parâmetro)
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    #seta o tamanho da figura
    plt.figure(figsize=(13, 5))
    #configura a cor das barras para cada valor, quando negativo usa vermelho, quando positivo, azul
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    #Quantidade de barras usadas
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    #labels do eixo x
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    #labels do eixo y
    plt.yticks(np.arange(-1.3, 1.2, 0.3))
    plt.show()

#Função criada para plotar imagem da matriz de confusão e uma escala de cores
def plot_confusion_matrix(cm_Precision, cm_Recall, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    plt.subplots_adjust(wspace = 0.05)
    
    ax[0].imshow(cm_Precision, interpolation='nearest', cmap=cmap)
    ax[0].set_title("Precision")
    
    tick_marks = np.arange(len(classes))
    plt.sca(ax[0])
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #rótulos dos eixos 'x' e 'y'
    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Predito')
    plt.imshow(cm_Precision, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    
    fmt = '.2f' 
    thresh = cm_Precision.max() / 2.
    for i, j in itertools.product(range(cm_Precision.shape[0]), range(cm_Precision.shape[1])):
        ax[0].text(j, i, format(cm_Precision[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_Precision[i, j] > thresh else "black")

    ax[1].imshow(cm_Recall, interpolation='nearest', cmap=cmap)
    ax[1].set_title("Recall")
    plt.sca(ax[1])
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.imshow(cm_Recall, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    
#    #rótulos dos eixos 'x' e 'y'
    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Predito')

    fmt = '.2f' 
    thresh = cm_Recall.max() / 2.
    for i, j in itertools.product(range(cm_Recall.shape[0]), range(cm_Recall.shape[1])):
        ax[1].text(j, i, format(cm_Recall[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_Recall[i, j] > thresh else "black")

        
#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/BigData_DataScience/Cancer/CSVs')
#Lendo o arquivo para guardar os dados em um DataFrame
df = pd.read_csv("DadosCancerMama.csv", sep=";")

#Configuração do modelo que se quer utilizar
modelo = "svm"
#modelo = "dt"

#mapeando e substituindo os valores das duas classes: 1 - Cancer, 2 Não Cancer
rotulo = df['Label'].map({'Cancer': 1, 'NaoCancer': 0})

df["Label"] = rotulo

print(df.head())

colunas = list(df)

#criando um objeto que guarde a escala do dataset para poder normalizar ele deixando todas as variáveis 
#numa mesma escala
scaler = preprocessing.StandardScaler().fit(df)
    
df = pd.DataFrame(scaler.transform(df), columns=colunas)
del df["Label"]
del df["ID"]

colunas2 = list(df)

#dividindo o dataset importado em 80% em treino e 20% teste
df_treino, df_teste, rotulo_treino, rotulo_teste = train_test_split(
     df, rotulo, test_size=0.2, random_state=0)

#Caso o modelo seja arvore de decisão, configura os valores mínimo e máximo de profundidade da árvore
#a serem testados para obter o melhor resultado 
if(modelo == "dt"):
    parametros = {'max_depth': range(3, 10)}
    tree = tree.DecisionTreeClassifier()

#Treina o modelo apenas para descobrir as melhores variáveis
    tree.fit(df_treino, rotulo_treino)
#Guarda as melhores variáveis e seus valores de importancia para classificação
    feat_sel = tree.feature_importances_
#Cria um dataframe com o nome das variáveis e com os valores de importancia para classificação
    importances = pd.DataFrame({'feature':df_treino.columns,'importance':np.round(tree.feature_importances_,3)})

#Ordena os valores de maior ao menor, filtra só as variáveis com valores de importância acima de 0.01
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    importances = importances[(importances.T >= 0.01).any()]
    importances.plot.bar()

#Se for um modelo SVM (Support Vector Machine) testará os parâmetros de C e kernel passados 
elif(modelo == "svm"):
    c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    parametros = {'kernel':('linear', 'rbf'), 'C': c_values}
#    parametros = {'C': c_values}    
#    svm = svm.SVC(kernel = "linear")
    svm = svm.SVC()

#Treina o modelo apenas para descobrir as melhores variáveis, porém só é possível saber caso o kernel
#seja linear, por isso há a condição no código
    svm.fit(df_treino, rotulo_treino)
    kernel_svm = svm.kernel
#Caso seja linear, irá acessar os dados de coef e salvar em uma variável a ser passada para a função
#criada de plotagem de gráfico criada anteriormente    
    if(kernel_svm == "linear"):
        feat_sel = svm.coef_
        f_importances(svm.coef_, colunas2)
        plot_coefficients(svm, colunas2, 10)

#Escolhendo as métricas para avaliação dos modelos preditivos, neste caso precision e recall
scores = ['precision', 'recall']

#Para cada métrica ele fará o teste dos parâmetros e escolherá qual os melhores parâmetros
# a serem utilizados com a função "GridSearchCV"
#Também usa a função Cross-Validation com parâmetro CV=10 indicando que dividirá o dataset em 10 
#utilizando 9 partes para treino e 1 para teste e assim trocando até as 10 partes serem testadas
for score in scores:
    print()
    print("Métrica.: " + score)
    print()
    
    if(modelo=="dt"):
        clf = GridSearchCV(tree, parametros, cv=10,
                           scoring='%s' % score)
        clf.fit(df_treino, rotulo_treino)
    elif(modelo=="svm"):
        clf = GridSearchCV(svm, parametros, cv=10,
                           scoring='%s' % score)
        clf.fit(df_treino, rotulo_treino)

#O classificador possui uma propriedade que indica os melhores parâmetros e então acessamos ela para 
#descobrir quais são
    print("Melhores parâmetros.: {} " .format(clf.best_params_))

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        if(clf.best_params_ == params):
            print("%0.4f (+/-%0.04f) utilizando %r"
              % (mean, std * 2, params))

    predicoes = clf.predict(df_teste)
    
# Computando matriz de confusão referente a cada métrica pré-estabelecida (precision\recall)    
    if(score == 'precision'):
        cm_Precision = confusion_matrix(rotulo_teste, predicoes)
    elif(score == 'recall'):
        cm_recall = confusion_matrix(rotulo_teste, predicoes)

#Calcula a acurácia do modelo com os melhores parâmetros para as métricas pré-estabelecidas (precision\recall)
    accuracy = accuracy_score(rotulo_teste, predicoes)
    print("Acurácia.: " + str(accuracy))
    
    np.set_printoptions(precision=2)
    
# Plotando a matriz de confusão
class_names = ['Cancer', 'N-Cancer']
plot_confusion_matrix(cm_Precision, cm_recall, classes=class_names,
                          title='Confusion matrix')
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
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/BigData_DataScience/Cancer/CSVs')
#Lendo o arquivo para guardar os dados em um DataFrame
df = pd.read_csv("DadosCancerMama.csv", sep=";")

codificador_rotulos = preprocessing.LabelEncoder()
rotulo = codificador_rotulos.fit_transform(df["Label"])

df["Label"] = rotulo

histogram_example = plt.hist(df["Label"])
plt.show()

colunas = list(df)
scaler = preprocessing.StandardScaler().fit(df)
    
df = pd.DataFrame(scaler.transform(df), columns=colunas)
#df["Label"] = rotulo
del df["Label"]

#df_treino, df_teste, rotulo_treino, rotulo_teste = train_test_split(
#     df, rotulo, test_size=0.2, random_state=0)

parameters = {'max_depth': range(3, 10)}
Modelo_Arvore = tree.DecisionTreeClassifier()
clf = GridSearchCV(Modelo_Arvore, parameters)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
scores = cross_val_score(clf, df, rotulo, cv=cv)

print("Acurácia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#Treinando o modelo
#DecisionTree.fit(Treino, target_treino)
    
#prediz resultados com o modelo treinado
#predicted = DecisionTree.predict(Teste)




#Função criada para plotar imagem da matriz de confusão e uma escala de cores
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    print(cm)
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    
#    #rótulos dos eixos 'x' e 'y'
#    plt.ylabel('Rótulo Real')
#    plt.xlabel('Rótulo Predito')
#
## Computando matriz de confusão
#cnf_matrix = confusion_matrix(rotulos_teste, predicoes)
#np.set_printoptions(precision=2)
#
## Plotando a matriz de confusão
#class_names = ['Sobreviveu', 'N-Sobreviveu']
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix')
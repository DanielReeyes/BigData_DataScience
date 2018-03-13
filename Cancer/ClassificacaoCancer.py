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
from sklearn.metrics import classification_report

#Função criada para plotar imagem da matriz de confusão e uma escala de cores
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, 
                          score=""):
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.show()
    print(plt)
    
    #rótulos dos eixos 'x' e 'y'
    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Predito')
        
#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/BigData_DataScience/Cancer/CSVs')
#Lendo o arquivo para guardar os dados em um DataFrame
df = pd.read_csv("DadosCancerMama.csv", sep=";")

codificador_rotulos = preprocessing.LabelEncoder()
rotulo = codificador_rotulos.fit_transform(df["Label"])

df["Label"] = rotulo

#histogram_example = plt.hist(df["Label"])
#plt.show()

colunas = list(df)
scaler = preprocessing.StandardScaler().fit(df)
    
df = pd.DataFrame(scaler.transform(df), columns=colunas)
del df["Label"]

parametros = {'max_depth': range(3, 10)}
tree = tree.DecisionTreeClassifier()

df_treino, df_teste, rotulo_treino, rotulo_teste = train_test_split(
     df, rotulo, test_size=0.1, random_state=0)

# Set the parameters by cross-validation

scores = ['precision', 'recall']

for score in scores:
    print()
    print("Métrica.: " + score)
    print()

    clf = GridSearchCV(tree, parametros, cv=10,
                       scoring='%s' % score)
    clf.fit(df_treino, rotulo_treino)

    print("Melhores parâmetros.: {} " .format(clf.best_params_))

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        if(clf.best_params_ == params):
            print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    y_true, y_pred = rotulo_teste, clf.predict(df_teste)
#    print(classification_report(y_true, y_pred))
#    print()
    # Computando matriz de confusão
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    
    # Plotando a matriz de confusão
    class_names = ['Cancer', 'N-Cancer']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix', score=score)
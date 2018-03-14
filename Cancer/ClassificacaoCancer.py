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
#    #rótulos dos eixos 'x' e 'y'
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

codificador_rotulos = preprocessing.LabelEncoder()
rotulo = codificador_rotulos.fit_transform(df["Label"])

df["Label"] = rotulo

colunas = list(df)
scaler = preprocessing.StandardScaler().fit(df)
    
df = pd.DataFrame(scaler.transform(df), columns=colunas)
del df["Label"]
del df["ID"]

parametros = {'max_depth': range(3, 10)}
tree = tree.DecisionTreeClassifier()

df_treino, df_teste, rotulo_treino, rotulo_teste = train_test_split(
     df, rotulo, test_size=0.2, random_state=0)

tree.fit(df_treino, rotulo_treino)
feat_sel = tree.feature_importances_

importances = pd.DataFrame({'feature':df_treino.columns,'importance':np.round(tree.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances = importances[(importances.T >= 0.01).any()]
print (importances)
importances.plot.bar()

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
            print("%0.4f (+/-%0.04f) utilizando %r"
              % (mean, std * 2, params))

#    y_true, y_pred = rotulo_teste, clf.predict(df_teste)
    predicoes = clf.predict(df_teste)
    
    # Computando matriz de confusão
    if(score == 'precision'):
#        cm_Precision = confusion_matrix(y_true, y_pred)
        cm_Precision = confusion_matrix(rotulo_teste, predicoes)
    elif(score == 'recall'):
#        cm_recall = confusion_matrix(y_true, y_pred)
        cm_recall = confusion_matrix(rotulo_teste, predicoes)
        
    np.set_printoptions(precision=2)
    
# Plotando a matriz de confusão
class_names = ['Cancer', 'N-Cancer']
plot_confusion_matrix(cm_Precision, cm_recall, classes=class_names,
                          title='Confusion matrix')
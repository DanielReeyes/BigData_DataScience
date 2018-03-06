import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

os.chdir('/Users/danielreyes/Documents/BigData_DataScience/Titanic/CSVs') #Setando o caminho do arquivo de treino
Dados_Treino = pd.read_csv("train.csv") #Lendo o arquivo para guardar os valores

#print(Dados_Treino.head(3))

#Normalizando os dados de treino
#Cria uma nova coluna, quando a idade não foi informada (Null), passará a idade média dos dados (28)
colMediaIdade = np.where(Dados_Treino["Age"].isnull(),
                         28,
                         Dados_Treino["Age"])

#Substitui a coluna original dos dados pela coluna nova, sem dados faltantes
Dados_Treino["Age"] = colMediaIdade

codificador_rotulos = preprocessing.LabelEncoder()
rotulo_sexo = codificador_rotulos.fit_transform(Dados_Treino["Sex"])

RNA = MLPClassifier()

#RNA.fit(X=pd.DataFrame(rotulo_sexo),
#        y=Dados_Treino["Survived"])

#print(RNA)

# Make data frame of predictors
#variaveis = pd.DataFrame([rotulo_sexo, 
#                          Dados_Treino["Pclass"],
#                          Dados_Treino["Age"],
#                          Dados_Treino["Fare"]]
#                         ).T

variaveis = pd.DataFrame(Dados_Treino, columns=['Pclass','Age','Fare'])
variaveis['Sexo'] = rotulo_sexo

print("Amostra do novo DataFrame de Treino.: ")
print(variaveis.head())

RNA.fit(X=variaveis,
        y=Dados_Treino["Survived"])

print("Score do Treino.: ")
print(RNA.score(X=variaveis,
        y=Dados_Treino["Survived"]))

#DADOS DE TESTE
Dados_Teste = pd.read_csv("test.csv")    # Read the data

#Normalizando os dados de teste
#Cria uma nova coluna, quando a idade não foi informada (Null), passará a idade média dos dados (28)
new_age_var = np.where(Dados_Teste["Age"].isnull(), # teste se for nulo
                       28,                          # Valor se for nulo
                       Dados_Teste["Age"])          # se não for nulo, pega a idade 

Dados_Teste["Age"] = new_age_var 


# Convert test variables to match model features
encoded_sex_test = codificador_rotulos.fit_transform(Dados_Teste["Sex"])

#variaveis_teste = pd.DataFrame([encoded_sex_test,
#                              Dados_Teste["Pclass"],
#                              Dados_Teste["Age"],
#                              Dados_Teste["Fare"]]).T

variaveis_teste = pd.DataFrame(Dados_Teste, columns=['Pclass','Age','Fare'])
variaveis_teste['Sexo'] = encoded_sex_test

print("Amostra do novo DataFrame de Teste.: ")
print(variaveis_teste.head())

rotulos_teste = pd.read_csv("gender_submission.csv")
rotulos_teste = rotulos_teste["Survived"]

variaveis_teste["Fare"] = np.where(variaveis_teste["Fare"].isnull(), # teste se for nulo
                       0,                                            # Valor se for nulo
                       variaveis_teste["Fare"])                      # se não for nulo, pega o valor de Fare 

predicoes = RNA.predict(variaveis_teste)
accuracy = accuracy_score(rotulos_teste, predicoes)

print("Resultado dos dados de Teste.: ")
print(accuracy)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

    plt.tight_layout()
    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Predito')

# Computando matriz de confusão
cnf_matrix = confusion_matrix(rotulos_teste, predicoes)
np.set_printoptions(precision=2)

# Plotando a matriz de confusão
class_names = ['Sobreviveu', 'N-Sobreviveu']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
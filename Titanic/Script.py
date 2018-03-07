import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Setando o caminho dos arquivos de dados
os.chdir('/Users/danielreyes/Documents/BigData_DataScience/Titanic/CSVs')
#Lendo o arquivo para guardar os dados de treino em um DataFrame
Dados_Treino = pd.read_csv("train.csv") 

#Normalizando os dados de treino
#Quando a idade não foi informada (Null), passará a idade média dos dados
Dados_Treino['Age'] = Dados_Treino['Age'].fillna(Dados_Treino['Age'].median())

#Transformando os valores da coluna sexo de string para inteiro
codificador_rotulos = preprocessing.LabelEncoder()
sexo_cod_treino = codificador_rotulos.fit_transform(Dados_Treino["Sex"])

#Declara um modelo de Rede Neural Multi-Camada
#RNA = MLPClassifier(hidden_layer_sizes=(4,))
RNA = MLPClassifier()

#Criando um dataframe para treinar o modelo somente com as colunas escolhidas e
# adicionando a coluna sexo normalizada
Variaveis_Treino = pd.DataFrame(Dados_Treino, columns=['Pclass','Age','Fare'])
Variaveis_Treino['Sexo'] = sexo_cod_treino

print("Amostra do novo DataFrame de Treino.: ")
print(Variaveis_Treino.head())

#Treinando o modelo RNA com os dados de treino para que ele descubra ligações
#dos dados de treino com o rótulo de sobreviver
RNA.fit(X=Variaveis_Treino,
        y=Dados_Treino["Survived"])

print(RNA)

#Mensurando a capacidade de acerto do modelo treinado
print("Score do Treino.: ")
print(RNA.score(X=Variaveis_Treino,
        y=Dados_Treino["Survived"]))

#DADOS DE TESTE
#Lendo o arquivo para guardar os dados de teste em um DataFrame
Dados_Teste = pd.read_csv("test.csv")

#Normalizando os dados de teste
#Quando a idade não foi informada (Null), passará a idade média dos dados
Dados_Teste["Age"] = Dados_Teste['Age'].fillna(Dados_Teste['Age'].median()) 

Dados_Teste["Fare"] = Dados_Teste['Fare'].fillna(Dados_Teste['Fare'].median()) 

#Transformando os valores da coluna sexo de string para inteiro
sexo_cod_teste = codificador_rotulos.fit_transform(Dados_Teste["Sex"])

#Criando um dataframe de teste somente com as colunas escolhidas e
# adicionando a coluna sexo normalizada
variaveis_teste = pd.DataFrame(Dados_Teste, columns=['Pclass','Age','Fare'])
variaveis_teste['Sexo'] = sexo_cod_teste

#Normalizando os dados de treino e de teste na mesma escala com z-score
print("Normalizando os dados de -1 a 1:")
colunas = list(Variaveis_Treino)
scaler = preprocessing.StandardScaler().fit(Variaveis_Treino)
    
Variaveis_Treino = pd.DataFrame(scaler.transform(Variaveis_Treino), columns=colunas)
variaveis_teste = pd.DataFrame(scaler.transform(variaveis_teste), columns=colunas)

print("Amostra do novo DataFrame de Teste.: ")
print(variaveis_teste.head())

#Como os rótulos dos dados de teste estão em um outro CSV, é necessário lê-los 
#e guardar também em um array ou DataFrame para podermos comparar com as predições
#feitas
rotulos_teste = pd.read_csv("gender_submission.csv")
rotulos_teste = rotulos_teste["Survived"]

#Utiliza o modelo RNA declarado e treinado na parte superior do código para 
#efetuar predições com os dados de teste que não possuem vinculo algum 
#com os seus rótulos
predicoes = RNA.predict(variaveis_teste)

#Utiliza uma função para contabilizar a acurácia, para isso precisa passar como
#parâmetro os rótulos originais e os preditos
accuracy = accuracy_score(rotulos_teste, predicoes)

#Função criada para plotar imagem da matriz de confusão e uma escala de cores
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
    
    #rótulos dos eixos 'x' e 'y'
    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Predito')

# Computando matriz de confusão
cnf_matrix = confusion_matrix(rotulos_teste, predicoes)
np.set_printoptions(precision=2)

# Plotando a matriz de confusão
class_names = ['Sobreviveu', 'N-Sobreviveu']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

print("Resultado dos dados de Teste.: ")
print(accuracy)
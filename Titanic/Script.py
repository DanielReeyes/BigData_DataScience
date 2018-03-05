import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

os.chdir('/Users/danielreyes/Documents/ProjetosBigData/Titanic/CSVs') #Setando o caminho do arquivo de treino
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
variaveis = pd.DataFrame([rotulo_sexo, 
                          Dados_Treino["Pclass"],
                          Dados_Treino["Age"],
                          Dados_Treino["Fare"]]
                         ).T

RNA.fit(X=variaveis,
        y=Dados_Treino["Survived"])

print(RNA.score(X=variaveis,
        y=Dados_Treino["Survived"]))

#DADOS DE TESTE
Dados_Teste = pd.read_csv("test.csv")    # Read the data

# Impute median Age for NA Age values
new_age_var = np.where(Dados_Teste["Age"].isnull(), # Logical check
                       28,                       # Value if check is true
                       Dados_Teste["Age"])      # Value if check is false

Dados_Teste["Age"] = new_age_var 


# Convert test variables to match model features
encoded_sex_test = codificador_rotulos.fit_transform(Dados_Teste["Sex"])

variaveis_teste = pd.DataFrame([encoded_sex_test,
                              Dados_Teste["Pclass"],
                              Dados_Teste["Age"],
                              Dados_Teste["Fare"]]).T
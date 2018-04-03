"""
Created on Sat Mar 31 20:35:40 2018

@author: danielreyes
"""

import os
import csv
import re
import pandas as pd

os.chdir('/Users/danielreyes/Documents/BigData_DataScience/TOTVS/Dados')
data = []

strDescartaveis = ["complemento:", "dets:", "prod:", "emit:", "cnpj: 01.234.567000189", "enderEmit:",
"fone: 16509334902", "xBairro:", "xLgr: 650 Castro St. unit 210", "xMun: Mountain View", "ide:",
"xPais: United States", "uf: CA", "xFant: TOTVS Labs", "infAdic:", "infCpl:", "total:", "dhEmi:",
"icmsTot:", "versaoDocumento: 1.0", "vDesc: 0.0", "vFrete: 0.0", "vOutro: 0.0", "vSeg: 0.0", "vbc: 0.0", 
"vbcst: 0.0", "vcofins: 0.0",  "vicms: 0.0", "vicmsDeson: 0.0", "vii: 0.0", "vipi: 0.0", "vpis: 0.0",
"vst: 0.0", "natOp: VENDA"]

df = pd.DataFrame(columns=["valorTotal", "indTot", "qCom", "uCom", "vProd", "vUnCom", "xProd", "nItem", 
                           "indTot", "qCom", "uCom", "vProd", "vUnCom", "xProd", "date", "infCpl",  "vProd", 
                           "vSeg", "vTotTrib", "vnf"])

searchWord = str("date:")
i = 0
lst = list()

with open('sample.txt', newline='') as inputfile:
    for row in csv.reader(inputfile):
        row = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: .]', '', str(row))
        if not row.strip() == "" and row.strip() not in strDescartaveis:
            row = row.lstrip(" ")
            data.append(row)
            if row.find(searchWord, 0, len(row)) < 0:
                str1, str2 = row.split(":")
                str1 = str1.lstrip(" ")
                str2 = str2.lstrip(" ")
                print("Parte 1: " + str1)
                print("Parte 2: " + str2)
                lst = lst.append(str2)
#                df.loc[i] = str2
#                i = i+1    

            
            

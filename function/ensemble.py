import smote_variants as sv
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import math
from sklearn.cluster import KMeans  
from sklearn import svm
from openpyxl import load_workbook


def write_to_excel(file,sheetName,row):
                os.chdir("./data")
                wb = load_workbook('file') #現在 os 在 data 中 所以要加上該檔案的檔案名稱
                sheet = wb[sheetName]
                return 0;
# 資料讀取 之後可以改成 loop 該路徑底下所有的檔案 excel
def read_excel(path,folderName):
                os.chdir(path + folderName)
                dirs = os.listdir(path+ folderName)
                train = []
                test = []

                for i in dirs:
                                #print(i.split("-")[-1])
                                if("xlsx" in i):
                                                if("tra" in i):
                                                                train.append(i)

                                                elif("tst" in i):
                                                                test.append(i)
                train = sorted(train)
                test = sorted(test)
                return train,test;

# 找出大類 回傳大類的種類名稱
def find_maj(sample_class):
    counter = Counter(sample_class);
    maj = list(dict(counter.most_common(1)).keys())
    maj = "".join(maj)
    print(maj)
    return  maj

# 大類跟小類的差距 要生成多少小類的數值 大類數據數量 - 小類數據數量
def classprocess(output):
    c = Counter(output)
    datagap = []
    maj = find_maj(output)
    maj_num = dict(c)[find_maj(output)]
    for className, number in c.items(): 
        print(number)
        temp = np.array([className,(maj_num - number)])
        datagap.append(temp)
    return datagap
# 3 種 方法 用 cluster 取中心點混合，2 種 方法 只要做混合

def methodRatio(method): # method = [sv.polynom_fit_SMOTE(),sv.ProWSyn(),sv.SMOTE_IPF()]
                
                 return 0               

def Cluster(methodSingle,train,): 
                """參數
                methodSingle 表示使用的方法 為何種？ (2或3) 
                train 資料
                
                """
                # loop method 裡的方法 會是一個 array 進行取出 cluster 中心點
                # 2 種 5:5 3:7 2:8
                # 3 種 4:4:2  3:3:4  2:5:3  
                   
                alloverpolynom = []
                overpolynom = []

                centerpolynom = []
                countfor = 0;
                # 用 method =[]
                for ii,i in enumerate(train):
                                randomIndex = []
                                le = preprocessing.LabelEncoder()
                                data = pd.read_excel(i,index_col=0)
                                lastColumn = data.columns[-1]

                                data[lastColumn]= data[lastColumn].str.replace("\n", "").str.strip()
                                originlen = data.shape[0]
                                output = data.iloc[:,data.shape[1]-1];
                                classCount = classprocess(output)
                                finaldata = data.iloc[:,:data.shape[1]-1]

                                output = le.fit_transform(output)
                                finaldata.iloc[:,0] = le.fit_transform(finaldata.iloc[:,0])
    
                                #output.iloc[:] = le.fit_transform(output.iloc[:])

                                tempover = []
   
                                finaldata = np.array(finaldata)
                                output = np.array(output)
                                over = sv.polynom_fit_SMOTE()
    
                                X_polynom,y_polynom = over.sample(finaldata,output)
                                newDataCount = len(X_polynom) - len(data)  # 新生成的 data 數量
                                # 把 X_polynom 跟 y_polynom 和在一起
                                X_polynom = pd.DataFrame(X_polynom)
                                y_polynom = pd.DataFrame(y_polynom)
                                alloverpolynom = pd.concat([X_polynom,y_polynom],axis=1) # SMOTE 完後的數據
    
                overpolynom.append(alloverpolynom)



def Random():


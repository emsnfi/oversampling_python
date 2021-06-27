import os;

# 輸入 路徑跟檔案名字 切成 train test
class dataprocess:
    #path = "/Users/emily/Desktop/Research/oversampling_python/data/"
    #folderName = 'abalone19-5-fold' # yeast6-5-fold'#'haberman-5-fold' #'abalone19-5-fold' 
    def get_train(self,path,folderName):
        os.chdir(path + folderName)
        dirs = os.listdir(path + folderName)
        train = []

        for i in dirs:
            #print(i.split("-")[-1])
            if("xlsx" in i):
                if("tra" in i):
                    train.append(i)
        train = sorted(train)
        
    
        return train
    def get_test(self,path,folderName):
        os.chdir(path + folderName)
        dirs = os.listdir(path + folderName)
        test = []

        for i in dirs:
            #print(i.split("-")[-1])
            if("xlsx" in i):
                if("tst" in i):
                    test.append(i)
  
        test = sorted(test)
    
        return test
    
path = "/Users/emily/Desktop/Research/oversampling_python/data/"
folderName = 'abalone19-5-fold' # yeast6-5-fold'#'haberman-5-fold' #'abalone19-5-fold' 
d = dataprocess()
d.get_test(path,folderName)
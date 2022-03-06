# coding=utf-8
from os import name
import pandas as pd


def arff_to_csv(fpath):
    # 讀取arff資料
    if fpath.find('.arff') < 0:
        print('the file is not .arff file')
        return
    f = open(fpath)
    lines = f.readlines()
    content = []
    for l in lines:
        content.append(l)
    datas = []
    for c in content:
        cs = c.split(',')
        datas.append(cs)

    # 將資料存入csv檔案中
    df = pd.DataFrame(data=datas, index=None, columns=None)
    filename = fpath[:fpath.find('.arff')] + '.csv'
    df.to_csv(filename, index=None)
    return df


if __name__ == '__main__':
    df = arff_to_csv(
        '/Users/emily/Desktop/Research/oversampling_python/data/NSL-KDD/KDDTrain+.arff')
    print(df)

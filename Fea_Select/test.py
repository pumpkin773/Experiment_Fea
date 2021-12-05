#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import numpy as np
import pandas as pd
from pandas import read_csv, DataFrame

for srcFile in glob.glob('../data/*.csv'):
    #预处理
    df = read_csv(srcFile)
    fea = df.iloc[:, :-1]
    col = []
    for str in fea.columns:
        str = str.split('-')
        col.append(str[0])
    col = np.unique(col)
    # print(col)
    dic = {}
    for i in col:
        list = []
        for j in fea.columns:
            if i == j.split('-')[0]:
                list.append(j)
        dic[i]=list
    # print(dic)
    new_fea = pd.DataFrame()
    for key, value in dic.items():
        new_fea[key] = fea[value].apply(lambda x: x.sum(), axis=1)
        # print(new_fea)
    print(new_fea)


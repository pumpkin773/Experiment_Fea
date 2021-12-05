#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import pandas as pd
from scipy.stats import friedmanchisquare
import xlrd
import numpy as np
import matplotlib.pyplot as plt


for path1 in glob.glob('../golden_select_result/*'):
    for srcFile in glob.glob(path1 + '/*'):
        df = pd.read_csv(srcFile)
        filename = srcFile.split('\\')[-1]
        namelist = filename.split('_')
        print(namelist)
        newfile = namelist[0] + '_' + namelist[1] + '_result_' + namelist[3] + '_' +namelist[4]
        print(newfile)
        df = df.sort_values(by="1", ascending=False)
        print(df)


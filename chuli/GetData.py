#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import pandas as pd
import numpy as np
import os
import csv

df = pd.read_csv('../result_banlance/ant/1/allFea_ant_totalFeatures1.csv')
# 特征列
columns = list(df.columns)
columns = columns[:-1]


for path1 in glob.glob('../result_banlance/*'):
    for path2 in glob.glob(path1 +'/*'):
        la = path2.split('-')
        if len(la)>1:
            continue
        for srcFile in glob.glob(path2 + '/*'):
            filename = srcFile.split('\\')[-1]
            filename = filename.split('.')[0]
            label = filename.split('_')
            print(label)

            if label[0] == 'allFea':
                data_frame = pd.read_csv(srcFile)
                index = data_frame['F1'].idxmax()
                print(data_frame.iloc[index,:])

                result_path = '../get_result2/allfea.csv'
                data_row = data_frame.iloc[index:index + 1, :]
                data_row.to_csv(result_path, mode='a', header=None)
            elif label[0] == 'cluster':
                data_frame = pd.read_csv(srcFile)
                index = data_frame['F1'].idxmax()
                print(data_frame.iloc[index, :])

                result_path = '../get_result2/selfea.csv'
                data_row = data_frame.iloc[index, :]
                # print(len(data_row.shape))
                if len(data_row.shape) == 1:
                    data_row = data_frame.iloc[index:index + 1, :]
                data_row.to_csv(result_path, mode='a', header=None)
            elif label[0]=='general':
                data_frame = pd.read_csv(srcFile)
                index = data_frame.groupby(['feaSelect'])['F1'].idxmax()
                index_max = data_frame['F1'].idxmax()
                print(data_frame.iloc[index, :])
                print(data_frame.iloc[index_max, :])

                result_path = '../get_result2/selfea.csv'
                data_row = data_frame.iloc[index, :]
                # print(len(data_row.shape))
                if len(data_row.shape) == 1:
                    data_row = data_frame.iloc[index:index + 1, :]
                data_row.to_csv(result_path, mode='a', header=None)
            else:
                data_frame = pd.read_csv(srcFile)
                index = data_frame['F1'].idxmax()
                print(data_frame.iloc[index, :])

                result_path = '../get_result2/selfea.csv'
                data_row = data_frame.iloc[index, :]
                # print(len(data_row.shape))
                if len(data_row.shape) == 1:
                    data_row = data_frame.iloc[index:index + 1, :]
                data_row.to_csv(result_path, mode='a', header=None)



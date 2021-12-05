#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import pandas as pd
from scipy.stats import friedmanchisquare
import xlrd
import numpy as np
import matplotlib.pyplot as plt

#如果Friedman检验对多个方法的结果的p值小于0.05，
# 说明这些方法在任务中存在显著的绩效差异。
# 然后采用Nemenyi后置检验来区分哪些方法与其他方法有显著差异。

def postTest_10(value):
    for path1 in glob.glob('../get_result2/xzx/*'):
        allFea = []
        we = []
        Variance, Correlate, Chi2, Muinfor, recurElimination, Penalty_l1, Penalty_l1l2, GBDT = [], [], [], [], [], [], [], []
        SMOTE_we, RandomOverSampler_we, ADASYN_we, RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we = [], [], [], [], [], []
        for srcFile in glob.glob(path1 +'/*'):
            # 读取数据
            df = pd.read_csv(srcFile)
            filename = srcFile.split('\\')
            dataname = filename[-2]
            filename = filename[-1].split('_')
            for index, row in df.iterrows():
                if filename[0]=='allFea':
                    allFea.append(row[value])
                elif filename[0]=='cluster':
                    we.append(row[value])
                elif filename[0]=='general':
                    if row['feaSelect'] == 'Variance':
                        Variance.append(row[value])
                    elif row['feaSelect'] == 'Correlate':
                        Correlate.append(row[value])
                    elif row['feaSelect'] == 'Chi2':
                        Chi2.append(row[value])
                    elif row['feaSelect'] == 'Muinfor':
                        Muinfor.append(row[value])
                    elif row['feaSelect'] == 'recurElimination':
                        recurElimination.append(row[value])
                    elif row['feaSelect'] == 'Penalty_l1':
                        Penalty_l1.append(row[value])
                    elif row['feaSelect'] == 'Penalty_l1l2':
                        Penalty_l1l2.append(row[value])
                    elif row['feaSelect'] == 'GBDT':
                        GBDT.append(row[value])
                elif filename[0]=='banlance':
                    if row['balance_data'] == 'SMOTE':
                        SMOTE_we.append(row[value])
                    elif row['balance_data'] == 'RandomOverSampler':
                        RandomOverSampler_we.append(row[value])
                    elif row['balance_data'] == 'ADASYN':
                        ADASYN_we.append(row[value])
                    elif row['balance_data'] == 'RandomUnderSampler':
                        RandomUnderSampler_we.append(row[value])
                    elif row['balance_data'] == 'SMOTEENN':
                        SMOTEENN_we.append(row[value])
                    elif row['balance_data'] == 'SMOTETomek':
                        SMOTETomek_we.append(row[value])
        "allFea,we,Variance, Correlate, Chi2, Muinfor, " \
        "recurElimination, Penalty_l1, Penalty_l1l2, GBDT,SMOTE_we, " \
        "RandomOverSampler_we, ADASYN_we, RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we"
        print(dataname)
        stat, p = friedmanchisquare(allFea, we, Variance, Correlate, Chi2, Muinfor,
                                    recurElimination, Penalty_l1, Penalty_l1l2,
                                    GBDT, SMOTE_we, RandomOverSampler_we, ADASYN_we,
                                    RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
        matrix = np.array([allFea, we, Variance, Correlate, Chi2, Muinfor,
                                    recurElimination, Penalty_l1, Penalty_l1l2,
                                    GBDT, SMOTE_we, RandomOverSampler_we, ADASYN_we,
                                    RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we])
        return matrix

def postTest(vv):
    for path1 in glob.glob('../dataresult.xlsx'):
        allFea = []
        we = []
        Variance, Correlate, Chi2, Muinfor, recurElimination, Penalty_l1, Penalty_l1l2, GBDT = [], [], [], [], [], [], [], []
        SMOTE_we, RandomOverSampler_we, ADASYN_we, RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we = [], [], [], [], [], []

        table = xlrd.open_workbook('../dataresult.xlsx')
        sheet = table.sheet_by_name("Sheet1")
        count_nrows = sheet.nrows  # 获取总行数
        for i in range(1, count_nrows):
            file = sheet.cell(i, 1).value
            if sheet.cell(i, 2).value == 'NO':
                if sheet.cell(i, 4).value == -1:
                    allFea.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value == 'DBSCAN':
                    we.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value == 'Variance':
                    Variance.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'Correlate':
                    Correlate.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'Chi2':
                    Chi2.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'Muinfor':
                    Muinfor.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'recurElimination':
                    recurElimination.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'Penalty_l1':
                    Penalty_l1.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'Penalty_l1l2':
                    Penalty_l1l2.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 3).value ==  'GBDT':
                    GBDT.append(sheet.cell(i, vv).value)

            else:
                if sheet.cell(i, 2).value == 'SMOTE':
                    SMOTE_we.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 2).value == 'RandomOverSampler':
                    RandomOverSampler_we.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 2).value == 'ADASYN':
                    ADASYN_we.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 2).value == 'RandomUnderSampler':
                    RandomUnderSampler_we.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 2).value == 'SMOTEENN':
                    SMOTEENN_we.append(sheet.cell(i, vv).value)
                elif sheet.cell(i, 2).value == 'SMOTETomek':
                    SMOTETomek_we.append(sheet.cell(i, vv).value)

        "allFea,we,Variance, Correlate, Chi2, Muinfor, " \
        "recurElimination, Penalty_l1, Penalty_l1l2, GBDT,SMOTE_we, " \
        "RandomOverSampler_we, ADASYN_we, RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we"
        print(allFea)
        matrix = np.array([allFea, we, Variance, Correlate, Chi2, Muinfor,
                                    recurElimination, Penalty_l1, Penalty_l1l2,
                                    GBDT])
        print(matrix)
        '''
        matrix = np.array([allFea, we, Variance, Correlate, Chi2, Muinfor,
                           recurElimination, Penalty_l1, Penalty_l1l2,
                           GBDT, SMOTE_we, RandomOverSampler_we, ADASYN_we,
                           RandomUnderSampler_we, SMOTEENN_we, SMOTETomek_we])
        '''
        return matrix

'构造降序排序矩阵'
def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n + 1
            ## 相同排名评分序值
            if j < 3 and matrix[i, sorts[i, j]] == matrix[i, sorts[i, j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i, j]] != matrix[i, sorts[i, j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i, sorts[i, j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i, j]] = j + 1
                continue
    return matrix

"""
    Friedman检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回检验结果（对应于排序矩阵列顺序的一维数组）
"""
def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * (k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result / (n * (k - 1) - result)
    return result

"""
    Nemenyi检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回CD值
"""
def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))


def main():
    matrix = postTest(21)
    matrix_r = rank_matrix(matrix.T)
    Friedman = friedman(10, 4, matrix_r)
    CD = nemenyi(10, 4, 3.164)
    ##画CD图
    rank_x = list(map(lambda x: np.mean(x), matrix))
    name_y = ["allFea", "we", "Variance", "Correlate","Chi2","Muinfor",
              "recurElimination","Penalty_l1","Penalty_l1l2","GBDT"]
    min_ = [x for x in rank_x - CD / 2]
    max_ = [x for x in rank_x + CD / 2]

    plt.title("Friedman")
    plt.scatter(rank_x, name_y)
    plt.hlines(name_y, min_, max_)
    plt.show()

if __name__ == '__main__':
    main()
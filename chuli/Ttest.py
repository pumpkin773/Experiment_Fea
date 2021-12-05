#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from scipy import stats
import numpy as np
import scipy.stats

'''每个数据集中，每种方法都取最好的结果'''
def WeMaxdata(eva):
    AF_data, UFS_data = [], []
    CS_data, CFS_data, TFS_data, IG_data, L1_data, L2_data, VT_data, RFE_data = [], [], [], [], [], [], [], []

    for srcFile in glob.glob('../result_data/*.csv'):
        filename = srcFile.split('\\')[-1]
        filename = filename.split('_')
        # print(filename)
        if filename[0] == "Fea":
            pass
        else:
            print(filename)
            data_fram = pd.read_csv(srcFile)
            if filename[0] == "allFea":
                AF_data.append(data_fram[eva].max())
            elif filename[0] == "DBSCAN":
                UFS_data.append(data_fram[eva].max())
            else:
                model_data = data_fram.groupby(['feaSelect'])[eva].max()
                df_data = pd.DataFrame(model_data)
                # print(df_data)
                # print(df_data.loc['Chi2','F1'])
                if np.isnan(df_data.loc['Chi2', eva]):
                    print(df_data.loc['Chi2', eva])
                    CS_data.append(0)
                else:
                    CS_data.append(df_data.loc['Chi2', eva])

                if np.isnan(df_data.loc['Correlate', eva]):
                    print(df_data.loc['Correlate', eva])
                    CFS_data.append(0)
                else:
                    CFS_data.append(df_data.loc['Correlate', eva])

                if np.isnan(df_data.loc['GBDT', eva]):
                    print(df_data.loc['GBDT', eva])
                    TFS_data.append(0)
                else:
                    TFS_data.append(df_data.loc['GBDT', eva])

                if np.isnan(df_data.loc['Muinfor', eva]):
                    print(df_data.loc['Muinfor', eva])
                    IG_data.append(0)
                else:
                    IG_data.append(df_data.loc['Muinfor', eva])

                if np.isnan(df_data.loc['Penalty_l1', eva]):
                    print(df_data.loc['Penalty_l1', eva])
                    L1_data.append(0)
                else:
                    L1_data.append(df_data.loc['Penalty_l1', eva])

                if np.isnan(df_data.loc['Penalty_l1l2', eva]):
                    print(df_data.loc['Penalty_l1l2', eva])
                    L2_data.append(0)
                else:
                    L2_data.append(df_data.loc['Penalty_l1l2', eva])

                if np.isnan(df_data.loc['Variance', eva]):
                    print(df_data.loc['Variance', eva])
                    VT_data.append(0)
                else:
                    VT_data.append(df_data.loc['Variance', eva])

                if np.isnan(df_data.loc['recurElimination', eva]):
                    print(df_data.loc['recurElimination', eva])
                    RFE_data.append(0)
                else:
                    RFE_data.append(df_data.loc['recurElimination', eva])
                # ['AF', 'UFS', 'CS', 'CFS', 'TFS', 'IG', 'L1', 'L2', 'VT', 'RFE']
                # data = [AF_data, UFS_data, CS_data, CFS_data, TFS_data, IG_data, L1_data, L2_data, VT_data, RFE_data]
            data = [AF_data, UFS_data, CS_data, CFS_data, TFS_data, IG_data, L2_data, VT_data, RFE_data]

            '''model_index = data_fram['F1'].idxmin()
                    print(data_fram.iat[model_index, 9])
                    allFea_data.append(data_fram['F1'].min())
                    '''
    # data = [AF_data, UFS_data, CS_data, CFS_data, TFS_data, IG_data, L2_data, VT_data, RFE_data]
    data = [UFS_data, AF_data, VT_data, CS_data, CFS_data, IG_data, RFE_data, TFS_data, L2_data]
    return data

def timedata(eva):
    AF_data, UFS_data = [], []
    CS_data, CFS_data, TFS_data, IG_data, L1_data, L2_data, VT_data, RFE_data = [], [], [], [], [], [], [], []

    for srcFile in glob.glob('../result_data/*.csv'):
        filename = srcFile.split('\\')[-1]
        filename = filename.split('_')
        # print(filename)
        if filename[0] == "Fea":
            pass
        else:
            print(filename)
            data_fram = pd.read_csv(srcFile)
            if filename[0] == "DBSCAN":
                model_index = data_fram[eva].idxmax()
                UFS_data.append(data_fram.iat[model_index, 14])

            elif filename[0] == "general":
                model_index = data_fram.groupby(['feaSelect'])[eva].idxmax()
                model_data = data_fram.iloc[model_index, :]
                df_data = pd.DataFrame(model_data)
                CS_data.append(df_data[df_data['feaSelect'].isin(['Chi2'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['Chi2'])].iat[0, 14])
                CFS_data.append(df_data[df_data['feaSelect'].isin(['Correlate'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['Correlate'])].iat[0, 14])
                TFS_data.append(df_data[df_data['feaSelect'].isin(['GBDT'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['GBDT'])].iat[0, 14])
                IG_data.append(df_data[df_data['feaSelect'].isin(['Muinfor'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['Muinfor'])].iat[0, 14])
                L1_data.append(df_data[df_data['feaSelect'].isin(['Penalty_l1'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['Penalty_l1'])].iat[0, 14])
                L2_data.append(df_data[df_data['feaSelect'].isin(['Penalty_l1l2'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['Penalty_l1l2'])].iat[0, 14])
                VT_data.append(df_data[df_data['feaSelect'].isin(['Variance'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['Variance'])].iat[0, 14])
                RFE_data.append(df_data[df_data['feaSelect'].isin(['recurElimination'])].iat[0, 14])
                print(df_data[df_data['feaSelect'].isin(['recurElimination'])].iat[0, 14])
    data = [UFS_data, VT_data, CS_data, CFS_data, IG_data, RFE_data, TFS_data, L2_data]
    return data

def AUC_F1MCC(eva_index):
    AF_data, UFS_data = [], []
    CS_data, CFS_data, TFS_data, IG_data, L1_data, L2_data, VT_data, RFE_data = [], [], [], [], [], [], [], []

    for srcFile in glob.glob('../result_data/*.csv'):
        filename = srcFile.split('\\')[-1]
        filename = filename.split('_')
        # print(filename)
        if filename[0] == "Fea":
            pass
        else:
            print(filename)
            data_fram = pd.read_csv(srcFile)
            if filename[0] == "allFea":
                model_index = data_fram['AUC'].idxmax()
                AF_data.append(data_fram.iat[model_index, eva_index-1])
            elif filename[0] == "DBSCAN":
                model_index = data_fram['AUC'].idxmax()
                UFS_data.append(data_fram.iat[model_index, eva_index])
            elif filename[0] == "general":
                model_index = data_fram.groupby(['feaSelect'])['AUC'].idxmax()
                model_data = data_fram.iloc[model_index, :]
                df_data = pd.DataFrame(model_data)
                CS_data.append(df_data[df_data['feaSelect'].isin(['Chi2'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['Chi2'])].iat[0, eva_index])
                CFS_data.append(df_data[df_data['feaSelect'].isin(['Correlate'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['Correlate'])].iat[0, eva_index])
                TFS_data.append(df_data[df_data['feaSelect'].isin(['GBDT'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['GBDT'])].iat[0, eva_index])
                IG_data.append(df_data[df_data['feaSelect'].isin(['Muinfor'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['Muinfor'])].iat[0, eva_index])
                L1_data.append(df_data[df_data['feaSelect'].isin(['Penalty_l1'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['Penalty_l1'])].iat[0, eva_index])
                L2_data.append(df_data[df_data['feaSelect'].isin(['Penalty_l1l2'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['Penalty_l1l2'])].iat[0, eva_index])
                VT_data.append(df_data[df_data['feaSelect'].isin(['Variance'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['Variance'])].iat[0, eva_index])
                RFE_data.append(df_data[df_data['feaSelect'].isin(['recurElimination'])].iat[0, eva_index])
                print(df_data[df_data['feaSelect'].isin(['recurElimination'])].iat[0, eva_index])
    data = [UFS_data, AF_data, VT_data, CS_data, CFS_data, IG_data, RFE_data, TFS_data, L2_data]
    return data


def main():
    # 'select_Fea_time'
    EVA = ['AUC']
    # F1---23 mcc----27
    # EVA = ['F1','MCC']
    for eva in EVA:
        data = WeMaxdata(eva)
        # data = timedata(eva)
        if eva == 'F1':
            eva_index = 23
        elif eva == 'MCC':
            eva_index = 27
        # data = AUC_F1MCC(eva_index)
        ttest_pvalue_matrix = []
        for i in range(1, len(data)):
            t, pval = scipy.stats.ttest_ind(data[0], data[i])
            print(t, pval)
            # print(ja)
            ttest_pvalue_matrix.append(pval)
        # print(ttest_pvalue_matrix)
        ttest_pvalue_dataframe = pd.DataFrame(ttest_pvalue_matrix, index=['UFS_AF', 'UFS_VT', 'UFS_CS', 'UFS_CFS',
                                                  'UFS_IG', 'UFS_RFE', 'UFS_TFS', 'UFS_RFS'],
                                         columns=['t-test p value'])
        print(ttest_pvalue_dataframe)
        ttest_pvalue_dataframe.to_csv('../data/'+ eva +'_t-test_pvalue_result.csv')


if __name__ == "__main__":
    main()



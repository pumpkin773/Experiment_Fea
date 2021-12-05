import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xlrd

#导入数据
def with_in():
    name =  ['Recall', 'Precision', 'F-Measure', 'AUC']
    for n in name:
        path = "../result-data/" + "RQ1-" + n + ".csv"
        tips = pd.read_csv(path)
        plt.rcParams['figure.figsize'] = 5, 4
        sns.violinplot(x= "Approach",y = n,data=tips,
                       split=True,
                       linewidth = 2, #线宽
                       width = 0.5,   #箱之间的间隔比例
                       palette = 'muted', #设置调色板
                       order = ['Our approach', 'SimpleFeature' , 'Imbalance' , 'Our approach'], #筛选类别
                       # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       gridsize = 50, #设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                     #bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        plt.show()
        # 可以添加散点图
        # sns.swarmplot(x='day', y='total_bill', data=tips, color='k', size=3, alpha=0.8)

def feature():
    name = ['Cost', 'Recall', 'Precision', 'F-Measure', 'AUC']
    path = "/Users/xiuting/PycharmProjects/SBR-identification/result-data/All-feature.csv"

    tips = pd.read_csv(path)
    for n in name:
        sns.violinplot(x="Feature type", y=n, data=tips,
                       split=True,
                       linewidth=1,  # 线宽
                       width=0.8,  # 箱之间的间隔比例
                       palette='Pastel1',  # 设置调色板
                       order=['Ambari-Word frequency', 'Ambari-TF-IDF',
                              'Camel-Word frequency', 'Camel-TF-IDF',
                              'Derby-Word frequency', 'Derby-TF-IDF',
                              'Wicket-Word frequency', 'Wicket-TF-IDF'],  # 筛选类别
                       # order=['wf', 'tf'],  # 筛选类别
                       scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       gridsize=50,  # 设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                       # bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        plt.show()


def groupedFeature():
    name = ['Cost', 'Recall', 'Precision', 'F-Measure', 'AUC']
    path = "/Users/xiuting/PycharmProjects/SBR-identification/result-data/All-feature.csv"

    tips = pd.read_csv(path)
    for n in name:
        plt.rcParams['figure.figsize'] = 5, 4
        ax = sns.violinplot(x="Project",
                       y=n,
                       # split=True,
                       linewidth=2,  # 线宽
                       width=0.8,  # 箱之间的间隔比例
                       data=tips,
                       hue="Feature type",
                       palette='RdBu'
                       )
        # ax.legend_.remove()
        plt.legend(loc='lower right', ncol=1, markerscale=0.1, labelspacing=0, handlelength=0.5)
        plt.show()


def groupedStrategy():
    name = ['Cost', 'Recall', 'Precision', 'F-Measure', 'AUC']
    path = "/Users/xiuting/PycharmProjects/SBR-identification/result-data/strategy.csv"

    tips = pd.read_csv(path)

    for n in name:
        # plt.figure(dpi=50, figsize=(8, 8))
        plt.rcParams['figure.figsize'] = 6, 4
        ax = sns.violinplot(x="Project",
                       y=n,
                       # split=True,
                       linewidth=2,  # 线宽
                       width=0.6,  # 箱之间的间隔比例
                       data=tips,
                       hue="Selection Strategy",
                       palette='RdBu'
                       )
        ax.legend_.remove()
        # plt.legend(loc='lower right', ncol=1, markerscale=0.1, labelspacing=0, handlelength=0.5)
        plt.show()


def groupedSampling():
    name = ['Cost', 'Recall', 'Precision', 'F-Measure', 'AUC']
    path = "/Users/xiuting/PycharmProjects/SBR-identification/result-data/sampling.csv"

    tips = pd.read_csv(path)

    for n in name:
        # plt.figure(dpi=50, figsize=(8, 8))
        plt.rcParams['figure.figsize'] = 6, 4.5
        ax = sns.violinplot(x="Project",
                       y=n,
                       # split=True,
                       linewidth=2,  # 线宽
                       width=0.6,  # 箱之间的间隔比例
                       data=tips,
                       hue="Query strategy",
                       palette='RdBu'
                       )
        ax.legend_.remove()
        # plt.legend(loc='lower right', ncol=1, markerscale=0.1, labelspacing=0, handlelength=0.5)
        plt.show()


def test():
    name =  ['F1', 'AUC', 'MCC' ]
    for n in name:
        path = "../result-data/" + "RQ-F1" + ".csv"
        tips = pd.read_csv(path)
        plt.rcParams['figure.figsize'] = 6.5, 4
        sns.violinplot(x= "Feature selection techniques",y = n,data=tips,
                       split=True,
                       linewidth = 2, #线宽
                       width = 0.5,   #箱之间的间隔比例
                       palette = 'muted', #设置调色板
                       order = ['Our approach', 'AF' , 'VT' , 'CS', 'CFS' , 'IG', 'RFE' , 'TFS', 'RFS' ], #筛选类别
                       # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       gridsize = 50, #设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                     #bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        plt.show()
        # 可以添加散点图
        # sns.swarmplot(x='day', y='total_bill', data=tips, color='k', size=3, alpha=0.8)

def test_cost():
    name =  ['Cost (:s)']
    for n in name:
        path = "../data/" + "RQ-cost" + ".csv"
        tips = pd.read_csv(path)
        plt.rcParams['figure.figsize'] = 6, 4
        sns.violinplot(x= "Feature selection techniques",y = n,data=tips,
                       split=True,
                       linewidth = 2, #线宽
                       width = 0.5,   #箱之间的间隔比例
                       palette = 'muted', #设置调色板
                       order = ['Our approach', 'VT' , 'CS', 'CFS' , 'IG', 'RFE' , 'TFS', 'RFS' ], #筛选类别
                       # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       # gridsize = 50, #设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                     #bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        plt.show()
        # 可以添加散点图
        # sns.swarmplot(x='day', y='total_bill', data=tips, color='k', size=3, alpha=0.8)

def test_3():
    name =  ['F1', 'AUC', 'MCC', 'Cost (:s)']
    for n in name:
        path = "../data/" + "AUC-3" + ".csv"
        tips = pd.read_csv(path)
        plt.rcParams['figure.figsize'] = 6, 4
        sns.violinplot(x= "Feature selection techniques",y = n,data=tips,
                       split=True,
                       linewidth = 2, #线宽
                       width = 0.5,   #箱之间的间隔比例
                       palette = 'muted', #设置调色板
                       order = ['LDR', 'FCR', 'FDR' ], #筛选类别
                       # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       # gridsize = 50, #设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                     #bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        plt.show()
        # 可以添加散点图
        # sns.swarmplot(x='day', y='total_bill', data=tips, color='k', size=3, alpha=0.8)

def test_RQ5():
    name =  ['F1', 'AUC', 'MCC']
    for n in name:
        path = "../data/" + "RQ5" + ".csv"
        tips = pd.read_csv(path)
        plt.rcParams['figure.figsize'] = 6, 4
        sns.violinplot(x= "Machine learning models",y = n,data=tips,
                       split=True,
                       linewidth = 2, #线宽
                       width = 0.5,   #箱之间的间隔比例
                       palette = 'muted', #设置调色板
                       order = ['NB', 'LR', 'KNN', 'DT', 'RF', 'SVM' ], #筛选类别
                       # scale = 'count',  #测度小提琴图的宽度： area-面积相同,count-按照样本数量决定宽度,width-宽度一样
                       # gridsize = 50, #设置小提琴图的平滑度，越高越平滑
                       # inner = 'box', #设置内部显示类型 --> 'box','quartile','point','stick',None
                     #bw = 0.8      #控制拟合程度，一般可以不设置
                       )
        plt.show()
        # 可以添加散点图
        # sns.swarmplot(x='day', y='total_bill', data=tips, color='k', size=3, alpha=0.8)

if __name__ == '__main__':
    # with_in()
    # groupedFeature()
    # groupedStrategy()
    # groupedSampling()
    # test()
    # test_cost()
    # test_3()
    test_RQ5()


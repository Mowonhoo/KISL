# -*- coding: UTF-8 -*-
import os,re
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def bar(x, y, dirPrefix='bar', title='', xlabel='', ylabel='', showplot=False):
    
    width = 0.35# 柱子的宽度
    fig = plt.figure(figsize=(6, 6), dpi=100)# 创建画布, 并设置分辨率为 80像素/每英寸
    plt.subplot(111)# 创建一个子图
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.bar(x, y, width, color="#87CEFA") # 绘制柱状图, 每根柱子的颜色为紫罗兰色ind-width/2
    
    ax = plt.gca()  #gca:get current axis得到当前轴
    #设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    plt.tight_layout()
    
    fig.savefig(dirPrefix+".png", dpi=1080, bbox_inches='tight')
    fig.savefig(dirPrefix+".pdf", bbox_inches='tight')

    if ~showplot:
        plt.close()
    

def SilhouetteAnalysis(X_Dim, labels, SI, sample_silhouette_values
                       , dirPrefix='Silhouette analysis for clustering'
                       , suptitle=''
                       , colors=None
                       , D3=False
                       , showplot=True):
    n_clusters_ =len(np.unique(labels)) - (1 if -1 in labels else 0)
    # Silhouette analysis for clustering
    fig = plt.figure(figsize=(16, 6), dpi=200)# 创建画布, 并设置分辨率为 80像素/每英寸
    fig.set_size_inches(18, 7) 
    plt.rcParams.update({'font.family': 'Times New Roman'})
    plt.rcParams.update({'font.weight': 'normal'})
    plt.rcParams.update({'font.size': 20})
    
    ax1=fig.add_subplot(121)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(labels) + (n_clusters_ + 1) * 10])
    
    y_lower = 10
    for i in range(n_clusters_):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        if colors==None:
            color = cm.nipy_spectral(float(i)/n_clusters_)
        else:
            color = np.compress(np.array(labels == i), colors, axis=0)[0]#np.array(colors)[np.array(labels == i).astype(bool)][0]#
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                         ,ith_cluster_silhouette_values
                         ,facecolor=color
                         ,alpha=0.7
                         )
        # ax1.text(x=-0.05
        #          , y=y_lower + 0.5 * size_cluster_i
        #          , s=str(i)
        #          , fontsize=10
        #         )
        y_lower = y_upper + 10
    ax1.set_title("The Silhouette plot for the various clusters.", fontsize=18)
    ax1.set_xlabel("The Silhouette coefficient values", fontsize=16)
    ax1.set_ylabel("Clusters label", fontsize=16)
    ax1.axvline(x=SI, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    if colors==None:
        colors = cm.nipy_spectral(labels.astype(float)/n_clusters_)
    if D3:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d') #
        ax2.scatter3D(X_Dim[:, 0], X_Dim[:, 1], X_Dim[:, 2]
                    ,marker='o'
                    ,s=4
                    ,c=colors
                    )
        ax2.set_title("The visualization of the clustered data", fontsize=18)
        ax2.set_xlabel("Feature space for the 1st feature", fontsize=16)
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=16, rotation=38) #, rotation=38 y 轴名称旋转 38 度    
        ax2.set_zlabel("Feature space for the 3rd feature", fontsize=16)
    else:
        ax2 = fig.add_subplot(122) #
        ax2.scatter(X_Dim[:, 0], X_Dim[:, 1]
                    ,marker='o'
                    ,s=4
                    ,c=colors
                    )
        ax2.set_title("The visualization of the clustered data", fontsize=18)
        ax2.set_xlabel("Feature space for the 1st feature", fontsize=16)
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=16) #, rotation=38 y 轴名称旋转 38 度    
    plt.suptitle(suptitle, fontsize=20, fontweight='bold')
    
    fig = plt.gcf()
    
    fig.savefig(dirPrefix+".png", bbox_inches='tight', dpi=1080)  # 保存图片，如果不设置 bbox_inches='tight'，保存的图片有可能显示不全
    fig.savefig(dirPrefix+".pdf", bbox_inches='tight')  # 保存图片，如果不设置 bbox_inches='tight'，保存的图片有可能显示不全
    
    if ~showplot:
        plt.close()
    

# def draw_bar(values, labels=None):
#     width = 0.35# 柱子的宽度
#     ind = np.linspace(-0.1, len(values), len(values))
#     fig = plt.figure(figsize=(30, 10), dpi=80)# 创建画布, 并设置分辨率为 80像素/每英寸
#     plt.subplot(121)# 创建一个子图
#     # Bar Plot
#     values = -np.sort(-(ADJ.sum().values))# 包含每个柱子对应值的序列
#     plt.bar(ind-width/2, values, width, color="#87CEFA") # 绘制柱状图, 每根柱子的颜色为紫罗兰色
#
#     plt.subplot(122)
#     plt.scatter(np.log10(abs(sim).sum().values), np.log10(ADJ.sum().values), s=60, c='blue', marker='.',alpha=0.8)
#     # Set the ticks on x,y-axis
#     #ax.set_xticks(ind)# 添加纵横轴的刻度
#     #ax.set_xticklabels(labels)
#     #plt.yticks(np.arange(0, 8, 1))
#
#     # labels
#     #plt.xlabel('')# 设置横轴标签
#     #plt.ylabel('')# 设置纵轴标签
#
#     #plt.title('')# 添加标题
#     #plt.legend(loc="upper right")# 添加图例
#
#     plt.show()
#
#     return fig
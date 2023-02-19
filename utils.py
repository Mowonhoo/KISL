# -*- coding: UTF-8 -*-
import os,re
import pandas as pd
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

from dcor import distance_correlation as dcorr
from scipy.stats import pearsonr,spearmanr
from exceptions import EmptyClustersException
from constraints import preprocess_constraints
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

#from topological_overlap_measure import TOMsimilarity
from topological_overlap_measure import *
#from soothsayer.networks import topological_overlap_measure

def my_dcorr(X, solver="full"):
    """
    # ### Compute the distance correlation function ###
    # from dcor import distance_correlation as dcorr
    # input: pd.DataFrame.object (n*m)
    # >>> a = np.array
    # >>> b = np.array
    # >>> my_dcorr(a, b)
    # value: pd.DataFrame.object (n*n)
    # value:[0-1]
    """
    if solver=="full":
        newX = X
    else:
        newX = PCA(n_components=0.95, svd_solver="auto").fit_transform(X) # n_components="mle"
        newX = pd.DataFrame(newX, index=X.index)
    
    dcor = np.zeros((newX.shape[0], newX.shape[0]), dtype=float)
    for i in range(newX.shape[0]):  # enumerate
        for j in range(i, newX.shape[0]):
            dcor[i][j] = dcor[j][i] = dcorr(newX.iloc[i], newX.iloc[j])
    dcor = pd.DataFrame(dcor, index=newX.index, columns=newX.index)
    return dcor

def snn_sim_matrix(X, k=5):
    """
    :param X: array-like, shape = [samples_size, features_size]
    :param k: positive integer(default = 5)
    :return: snn
    """
    try:
        X = np.array(X)
    except:
        raise ValueError("The input must be a matrix.")
    samples_size, features_size = X.shape  
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_matrix = nbrs.kneighbors(X, return_distance=False) 
    sim_matrix = 0.5 + np.zeros((samples_size, samples_size))
    for i in range(samples_size):
        t = np.where(knn_matrix == i)[0]
        c = list(itertools.combinations(t, 2))
        for j in c:
            if j[0] not in knn_matrix[j[1]]:
                continue
            sim_matrix[j[0]][j[1]] += 1
    sim_matrix = 1 / sim_matrix
    sim_matrix = np.triu(sim_matrix)
    sim_matrix += sim_matrix.T - np.diag(sim_matrix.diagonal())
    return sim_matrix


class PCKMeans:
    def __init__(self, n_clusters=5, distance_type=None, w=1, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
        self.distance_type = distance_type

    def fit(self, X, y=None, ml=[], cl=[]):
        X = np.array(X)
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize centroids
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)
        
        # Repeat until convergence
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers
 
    def _distcorr(self, X, Y):
        """ Compute the distance correlation function

        >>> a = [1,2,3,4,5]
        >>> b = np.array([1,2,9,4,4])
        >>> distcorr(a, b)
        0.762676242417
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        
        return dcor    
    
    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w):
        #distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)
        #distance = np.linalg.norm(X[x_i] - centroids[c_i])
        #distance = self._distcorr(X[x_i], centroids[c_i]) # Compute the distance correlation

        if self.distance_type=='euclidean':
            distance = np.linalg.norm(X[x_i] - centroids[c_i])
        elif self.distance_type=='dcorr':
            distance = self._distcorr(X[x_i], centroids[c_i]) # Compute the distance correlation
        else:
            distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_penalty = 0
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w

        return distance + ml_penalty + cl_penalty

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for x_i in index:
            labels[x_i] = np.argmin([self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            print("Empty clusters")
            raise EmptyClustersException

        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    
def pickSoftThreshold(data, power=20, RsquaredCut = 0.85, dataIsExpr=False, prefix="01", outdir="./", verbose = 5):
    #=====================
    #Call R script to calculate K value
    #=====================
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    robjects.r['source']('./pickSoftThreshold.R')#, encoding = 'UTF-8'

    powers = robjects.IntVector(list(range(1, power+1)))
    #print(powers)
    #r_data = pandas2ri.py2ri(data)
    similarExpr = robjects.r['data.matrix'](data)
    sft = robjects.r['pickSoftThreshold'](similarExpr, dataIsExpr = dataIsExpr, powerVector = powers, RsquaredCut = RsquaredCut, prefix=prefix, outdir=outdir, verbose = verbose)#
    sft = {"powerEstimate": sft.rx2("powerEstimate")[0], "fitIndices": pd.DataFrame(sft.rx2("fitIndices"), index=sft.rx2("fitIndices").names).T}

    return(sft)


def three_sigma(data, col):
    """
    :param data:series
    :return:series
    """
    mean = data[col].mean()
    std = data[col].std()
    return data[((mean-2*std)<data[col]) & (data[col]<(mean+2*std))].reset_index(drop=True)

def stand_sca(data, col):
    print('mean:%.4f' % data[col].mean(), 'std:%.4f' % data[col].std())
    data[col]=(data[col]-data[col].mean())/data[col].std()
    return data

def minmax_sca(data, col):
    data_max = data[col].max()
    data_min = data[col].min()
    data[col]=(data[col]-data_min)/(data_max-data_min)
    return data

def get_pearsonr(Mat, Vect):
    pears = dict(coef = [], pval = [])
    for i in range(Mat.shape[0]):
        coef, pval = pearsonr(Mat[i, ], Vect)
        pears['coef'].append(coef)
        pears['pval'].append(pval)

    max_coef = pd.DataFrame.from_dict(pears)['coef'].max()
    index = pears['coef'].index(max_coef)
    return (pears['coef'][index], pears['pval'][index], index)

def get_spearmanr(Mat, Vect):
    spearm = dict(coef = [], pval = [])
    for i in range(Mat.shape[0]):
        coef, pval = spearmanr(Mat[i, ], Vect)
        spearm['coef'].append(coef)
        spearm['pval'].append(pval)

    max_coef = pd.DataFrame.from_dict(spearm)['coef'].max()
    index = spearm['coef'].index(max_coef)
    return (spearm['coef'][index], spearm['pval'][index], index)


def filter_feature_selection(X, y, label_type='classif', methods='mutual_info', threshold=0.05):
    if label_type=='classif':

        if methods=='mutual_info':
            from sklearn.feature_selection import mutual_info_classif as MIC
            X.drop(X.index[MIC(X.T, y) <= 0], axis=0, inplace=True)

        elif methods=='chi2':
            from sklearn.feature_selection import chi2
            chivalua, pvalues_chi = chi2(X.T, y)
            X.drop(X.index[pvalues_chi > threshold], axis=0, inplace=True)

        elif methods=='ANOVA':
            from sklearn.feature_selection import f_classif
            F, pvalues_f = f_classif(X.T, y)
            X.drop(X.index[pvalues_f > threshold], axis=0, inplace=True)

        else:
            print('Feature selection method parameter error!')   

    elif label_type=='regression':

        if methods=='mutual_info':
            from sklearn.feature_selection import mutual_info_regression as MIR
            X.drop(X.index[MIR(X.T, y) <= 0], axis=0, inplace=True)

        elif methods=='ANOVA':
            from sklearn.feature_selection import f_regression
            F, pvalues_f = f_regression(X.T, y)
            X.drop(X.index[pvalues_f > threshold], axis=0, inplace=True)

        else:
            print('Feature selection method parameter error!') 
    else:
        print('Label type parameter error!')

    return X

def model_feature_selection(X, y, method='svm', threshold=1e-5, random_state=None):
    #from sklearn.feature_selection import SelectFromModel
    
    if method=='rfc':
        from sklearn.ensemble import RandomForestClassifier as RFC
        RFC_ = RFC(n_estimators =10, random_state=random_state).fit(X.T, y)
        X.drop(X.index[RFC_.feature_importances_ < threshold], axis=0, inplace=True)
        
    elif method=='svm':
        from sklearn.svm import SVC
        clf= SVC(kernel = "linear", gamma="auto" , degree = 1 , cache_size=5000).fit(X.T, y)
        X.drop(X.index[abs(clf.coef_[0]) < threshold], axis=0, inplace=True)
        
    elif method=='LogisticR':
        from sklearn.linear_model import LogisticRegression as LR
        LR_ = LR(penalty="l2", solver="liblinear", C=0.9, random_state=0).fit(X.T, y)
        X.drop(X.index[abs(LR_.coef_[0]) < threshold], axis=0, inplace=True)
        
    elif method=='lasso':
        from sklearn.linear_model import Lasso        
        lasso_ = Lasso(alpha=0.01).fit(X.T, y)
        X.drop(X.index[abs(lasso_.coef_) < threshold], axis=0, inplace=True)
        
    elif method=='LinearR':
        from sklearn.linear_model import LinearRegression
        LinearR = LinearRegression().fit(X.T, y)
        X.drop(X.index[abs(LinearR.coef_[0]) < threshold], axis=0, inplace=True)
        
    elif method=='RFE':
        from sklearn.ensemble import RandomForestClassifier as RFC
        from sklearn.feature_selection import RFE, RFECV
        RFC_ = RFC(n_estimators =10, random_state=0)
        selector = RFECV(RFC_, min_features_to_select=X.shape[0], step=10, cv=5).fit(X.T, y)
        X.drop(X.index[selector.support_], axis=0, inplace=True)
        
    else:
        print('Methods in only %s, %s, %s, %s, %s, %s' % ['rfc', 'svm', 'LogisticR', 'lasso', 'LinearR', 'RFE']) 

    return X

def bar(x, y, dirPrefix='bar', title='', xlabel='', ylabel='', showplot=False):
    
    width = 0.35
    fig = plt.figure(figsize=(6, 6), dpi=100)
    plt.subplot(111)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.bar(x, y, width, color="#87CEFA") 
    
    ax = plt.gca()  #gca:get current axis
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
    fig = plt.figure(figsize=(16, 6), dpi=200)
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
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=16, rotation=38) #, rotation=38
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
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=16) #, rotation=38  
    plt.suptitle(suptitle, fontsize=20, fontweight='bold')
    
    fig = plt.gcf()
    
    fig.savefig(dirPrefix+".png", bbox_inches='tight', dpi=1080) 
    fig.savefig(dirPrefix+".pdf", bbox_inches='tight')
    
    if ~showplot:
        plt.close()
    
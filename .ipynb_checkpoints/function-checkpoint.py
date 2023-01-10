#!/python

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
###---------特征选择--------###

### 过滤法做特征选择 ###
def filter_feature_selection(X, y, label_type='classif', methods='mutual_info', threshold=0.05):
    ## 输入X--特征*记录
    if label_type=='classif':

        if methods=='mutual_info': # 捕捉线性和非线性依赖
            from sklearn.feature_selection import mutual_info_classif as MIC
            X.drop(X.index[MIC(X.T, y) <= 0], axis=0, inplace=True)

        elif methods=='chi2': # 捕捉线性依赖 
            from sklearn.feature_selection import chi2
            chivalua, pvalues_chi = chi2(X.T, y)
            X.drop(X.index[pvalues_chi > threshold], axis=0, inplace=True)

        elif methods=='ANOVA': # 捕捉线性依赖
            from sklearn.feature_selection import f_classif
            F, pvalues_f = f_classif(X.T, y)
            X.drop(X.index[pvalues_f > threshold], axis=0, inplace=True)

        else:
            print('Feature selection method parameter error!')   

    elif label_type=='regression':

        if methods=='mutual_info': # 捕捉线性和非线性依赖
            from sklearn.feature_selection import mutual_info_regression as MIR
            X.drop(X.index[MIR(X.T, y) <= 0], axis=0, inplace=True)

        elif methods=='ANOVA': # 捕捉线性依赖
            from sklearn.feature_selection import f_regression
            F, pvalues_f = f_regression(X.T, y)
            X.drop(X.index[pvalues_f > threshold], axis=0, inplace=True)

        else:
            print('Feature selection method parameter error!') 
    else:
        print('Label type parameter error!')

    return X

### 嵌入法做特征选择 ###
def model_feature_selection(X, y, method='svm', threshold=1e-5, random_state=None):
    ## 输入X--特征*记录
    #from sklearn.feature_selection import SelectFromModel
    
    if method=='rfc':
        ### 使用决策树做特征选择 ###
        from sklearn.ensemble import RandomForestClassifier as RFC
        RFC_ = RFC(n_estimators =10, random_state=random_state).fit(X.T, y)
        X.drop(X.index[RFC_.feature_importances_ < threshold], axis=0, inplace=True)
        
    elif method=='svm':
        ### 使用SVM做特征选择 ###
        from sklearn.svm import SVC
        clf= SVC(kernel = "linear", gamma="auto" , degree = 1 , cache_size=5000).fit(X.T, y)
        X.drop(X.index[abs(clf.coef_[0]) < threshold], axis=0, inplace=True)
        
    elif method=='LogisticR':
        ### 逻辑回归做特征选择 ###
        from sklearn.linear_model import LogisticRegression as LR
        LR_ = LR(penalty="l2", solver="liblinear", C=0.9, random_state=0).fit(X.T, y)
        X.drop(X.index[abs(LR_.coef_[0]) < threshold], axis=0, inplace=True)
        
    elif method=='lasso':
        ### lasso做特征选择 ###
        from sklearn.linear_model import Lasso        
        lasso_ = Lasso(alpha=0.01).fit(X.T, y)
        X.drop(X.index[abs(lasso_.coef_) < threshold], axis=0, inplace=True)
        
    elif method=='LinearR':
        ### 线性回归做特征选择 ###
        from sklearn.linear_model import LinearRegression
        LinearR = LinearRegression().fit(X.T, y)
        X.drop(X.index[abs(LinearR.coef_[0]) < threshold], axis=0, inplace=True)
        
    elif method=='RFE':
        ### 包装法做特征选择 ###
        from sklearn.ensemble import RandomForestClassifier as RFC
        from sklearn.feature_selection import RFE, RFECV
        RFC_ = RFC(n_estimators =10, random_state=0)
        selector = RFECV(RFC_, min_features_to_select=X.shape[0], step=10, cv=5).fit(X.T, y)
        X.drop(X.index[selector.support_], axis=0, inplace=True)
        
    else:
        print('Methods in only %s, %s, %s, %s, %s, %s' % ['rfc', 'svm', 'LogisticR', 'lasso', 'LinearR', 'RFE']) 

    return X
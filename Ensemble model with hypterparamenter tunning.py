#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:53:46 2021

@author: aprilliu
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


#cross validation on 10 folds to meausure the performance on train data with each classifier, and then chose the top 2 models
kfold = StratifiedKFold(n_splits=10)
random_state = 42
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier())
classifiers.append(XGBClassifier())
classifiers.append(LGBMClassifier())
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"Algorithm":["SVC","DecisionTree","RandomForest","GradientBoosting","ExtraTrees","XGB",'LGBM'],"CrossValMeans":cv_means,"CrossValerrors": cv_std})
cv_res


#Setting up of hypterparamenters in classifier
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
# Number of features to consider at every split
max_features = [ 20 ,'auto', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 30, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Create the random grid
rf_param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# Random forest Hyper parameter tuning with gridsearch
rf = RandomForestClassifier()

#fitting
Grid_s_rf = GridSearchCV(rf, param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = True)
Grid_s_rf.fit(X_train,y_train)
RFC_best = Grid_s_rf.best_estimator_

# Best score
Grid_s_rf.best_score_


max_depth = [int(x) for x in np.linspace(5, 20, num = 3)]
min_child_weight = [5,6,7]
eta = [.3, .2, .1, .05, .01, .005]

XGB_param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'eta': eta,
               'min_child_weight': min_child_weight,}

XGB = XGBClassifier()

Grid_s_XGB = GridSearchCV(XGB, param_grid = XGB_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = True)
Grid_s_XGB.fit(X_train,y_train)
XGB_best = Grid_s_XGB.best_estimator_

# Best score
Grid_s_XGB.best_score_


#Ensembling of 2 models
votingC = VotingClassifier(estimators=[('lg', rf),('xgb',XGB)], voting='soft', n_jobs=-1)
votingC = votingC.fit(X_train, y_train)
#To get the model performance on the training data
print('Score: ', votingC.score(X_train, y_train))


predictions = votingC.predict(X_test)
submission = pd.DataFrame({'PassengerId': test_data.PassengerId,
                           'Survived': predictions})
submission.to_csv('submission.csv', index = False)
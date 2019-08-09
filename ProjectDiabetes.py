#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:09:19 2019

@author: sachin
"""

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
 
# Data Pre-Processing.
dataset = pd.read_csv("diabetes.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

imputer = Imputer(missing_values = 0,
                  strategy = "mean",
                  axis = 0)
imputer = imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state = 0)                                                   

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
  
# Testing on different algorithms.

def SupportVectorMachine(X_train,X_test,y_train,y_test):
    support_vec = SVC(C = 6, 
                  kernel = 'rbf',
                  gamma = 0.00066)
    support_vec.fit(X_train, y_train)
 
    y_pred = support_vec.predict(X_test)


    confusion_m = confusion_matrix(y_test,y_pred)


    accuracy = cross_val_score(estimator = support_vec,
                               X = X_train,
                               y = y_train,
                               cv = 10,
                               n_jobs = -1)
    accuracy_mean = accuracy.mean()
    accuracy_std = accuracy.std()
    
    # Grid search for optaining best parameters.
    """
    parameters = [{'C': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'kernel': ['rbf','linear'],
               'gamma': [0.00061,0.00066,0.00062,0.00063,0.00064,0.00065]}]
    grid_search = GridSearchCV(estimator = support_vec,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               n_jobs = -1,
                               cv = 10)

    grid_search = grid_search.fit(X_train,y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_ """
    
    return ('SVM',y_pred,confusion_m,accuracy_mean)

def KNN(X_train,X_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors = 12,
                               weights = 'distance',
                               p = 1,
                               metric = 'minkowski',
                               algorithm = 'auto')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    confusion_m = confusion_matrix(y_test,y_pred)

    accuracy = cross_val_score(estimator = knn,
                           X = X_train,
                           y = y_train,
                           cv = 10,
                           n_jobs = -1)
    accuracy_mean = accuracy.mean()
    accuracy_std = accuracy.std()
    
    #Grid search for K-Neighbors-Classifier.
    """
    parameters = [{'weights': ['uniform','distance'], 'p': [1,2],
                   'algorithm': ['auto','ball_tree','kd_tree','brute'],
                   'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}]
    grid_search = GridSearchCV(estimator = knn,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               n_jobs = -1,
                               cv = 10)

    grid_search = grid_search.fit(X_train,y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_ """
    
    return ('KNeighbors',y_pred,confusion_m,accuracy_mean)

def RandomForest(X_train,X_test,y_train,y_test):
    
    random_forest = RandomForestClassifier(n_estimators = 300,
                                           criterion = "entropy",
                                           n_jobs = -1,
                                           random_state = 0)
    random_forest.fit(X_train,y_train)
    y_pred = random_forest.predict(X_test)


    confusion_m = confusion_matrix(y_test,y_pred)
 

    accuracy = cross_val_score(estimator = random_forest,
                               X = X_train,
                               y = y_train,
                               cv = 10,
                               n_jobs = -1)
    accuracy_mean = accuracy.mean()
    accuracy_std = accuracy.std()
    
    # Grid Search for Random Forest.
    """
    parameters = [{'n_estimators': [100,200,300,400,500,600,700,800,900],
                   'criterion': ["gini","entropy"]}]
    grid_search = GridSearchCV(estimator = random_forest,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               n_jobs = -1,
                               cv = 10)

    grid_search = grid_search.fit(X_train,y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_ """
    
    return ('RandomForest',y_pred,confusion_m,accuracy_mean)


def XGboost(X_train,X_test,y_train,y_test):
    
    classifier = XGBClassifier(n_estimators = 99,
                               max_depth = 3,
                               learning_rate = 0.1,
                               booster = 'gbtree',
                               random_state = 0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    confusion_m = confusion_matrix(y_test,y_pred)
 
    accuracy = cross_val_score(estimator = classifier,
                               X = X_train,
                               y = y_train,
                               cv = 10,
                               n_jobs = -1)
    accuracy_mean = accuracy.mean()
    accuracy_std = accuracy.std()
 
    # Grid Search for XG-Boost.
    """
    parameters = [{'n_estimators': [99,200,300],
                   'max_depth': [3,4,5],
                   'learning_rate': [0.1,0.11,0.12,0.13,0.14,0.15]}]
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               n_jobs = -1,
                               cv = 10)

    grid_search = grid_search.fit(X_train,y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_ """
    
    return ('XGboost',y_pred,confusion_m,accuracy_mean)

Accuracy_algo = list()

algo1, predictions1,confusion_m1, accuracy_mean_SVM = SupportVectorMachine(X_train,X_test,
                                                                           y_train,y_test)
Accuracy_algo.append((algo1,accuracy_mean_SVM))

algo2, predictions2,confusion_m2, accuracy_mean_KNN = KNN(X_train,X_test,y_train,y_test)
Accuracy_algo.append((algo2,accuracy_mean_KNN))

algo3, predictions3,confusion_m3, accuracy_mean_rf = RandomForest(X_train,X_test,
                                                                   y_train,y_test)
Accuracy_algo.append((algo3,accuracy_mean_rf))

algo4, predictions4,confusion_m4, accuracy_mean_xgb = XGboost(X_train,X_test,y_train,y_test)
Accuracy_algo.append((algo4,accuracy_mean_xgb))

print("\n\n\n")
for j,k in enumerate(Accuracy_algo):
    print("{}. {} Accuracy: {}".format(j+1,k[0],k[1]))
    




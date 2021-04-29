#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:36:29 2021

@author: aprilliu
"""

class Imputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.med_fare_ = X.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
        self.most_freq_embarked = X.Embarked.value_counts().index[0]
        return self
    def transform(self, X, y=None):
        #replacing missing values of Age with median Age for each class. 1 value for each Sex.
        X.Age = X.groupby(['Sex', 'Pclass'])['Age'].apply(lambda z: z.fillna(z.median()))
        # 1 only missing value for Fare. A Man in the third class with no family
        X.Fare = X.Fare.fillna(self.med_fare_)
        # filling Embarked with the most frequent 
        X.Embarked = X.Embarked.fillna(self.most_freq_embarked)
        
        return X
        
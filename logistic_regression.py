# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:30:17 2017

@author: I310036
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Read UCLA data set
admission = pd.read_csv("binary.csv")

X = admission.iloc[:,1:4]
Y = admission.iloc[:,0]
plt.hist(X.iloc[:,0])
plt.show()
plt.hist(X.iloc[:,1])
plt.show()
plt.hist(X.iloc[:,2])
plt.show()
plt.hist(Y)
plt.show()

# Shuffle the dataset
admissions = admission.loc[np.random.permutation(admission.index)]

num_train = 300
data_train = admission[:num_train]
data_test = admission[num_train:]

model = LogisticRegression()

model.fit(data_train[['gpa','gre','rank']],data_train['admit'])

train = model.predict_proba(data_train[['gpa','gre','rank']])[:,1]

plt.scatter(data_train['gre'], train, marker = '+', color = 'g')
plt.xlabel = 'gre'
plt.ylabel = 'Prob'
plt.show()

pred_train = model.predict(data_train[['gpa','gre','rank']])

accuracy_train = (pred_train == data_train['admit']).mean()
accuracy_train

pred_test = model.predict(data_test[['gpa','gre','rank']])

accuracy_test = (pred_test == data_test['admit']).mean()
accuracy_test

from sklearn.metrics import roc_curve, roc_auc_score

train_probs = model.predict_proba(data_train[['gpa', 'gre','rank']])[:,1]
test_probs = model.predict_proba(data_test[['gpa', 'gre','rank']])[:,1]

auc_train = roc_auc_score(data_train["admit"], train_probs)
auc_test = roc_auc_score(data_test["admit"], test_probs)

print('Auc_train: {}'.format(auc_train))
print('Auc_test: {}'.format(auc_test))

roc_train = roc_curve(data_train["admit"], train_probs)
roc_test = roc_curve(data_test["admit"], test_probs)

plt.plot(roc_train[0], roc_train[1])

plt.plot(roc_test[0], roc_test[1])
plt.show()
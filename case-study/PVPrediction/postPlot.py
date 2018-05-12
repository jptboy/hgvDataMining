# -*- coding: utf-8 -*-
"""
Created on Wed May 09 16:30:42 2018

@author: Administrator
"""

from sklearn.externals import joblib
import pandas as pd
import matplot.pyplot as plt

ann = joblib.load('ann.pkl')

# plot
cvDF = pd.DataFrame(ann.cv_results_)
identity = cvDF.loc[cvDF['param_reg__activation']=='identity']
# plot the training scores
identity[['mean_train_score','mean_test_score']].plot()
plt.title('Identity score')

# plot training time
plt.figure()
identity['mean_fit_time'].plot()
plt.title('Identity training time')

logistic = cvDF.loc[cvDF['param_reg__activation']=='logistic']
# plot the training scores
plt.figure()
logistic[['mean_train_score','mean_test_score']].plot()
plt.ylim(-600,100)

# plot training time
plt.figure()
logistic['mean_fit_time'].plot()
plt.title('Logistic training time')

tanh = cvDF.loc[cvDF['param_reg__activation']=='tanh']
# plot the training scores
plt.figure()
tanh[['mean_train_score','mean_test_score']].plot()
plt.ylim(-600,100)
plt.title('Tanh Score')

# plot training time
plt.figure()
tanh['mean_fit_time'].plot()
plt.title('Tanh training time')

relu = cvDF.loc[cvDF['param_reg__activation']=='relu']
# plot the training scores
plt.figure()
relu[['mean_train_score','mean_test_score']].plot()
plt.ylim(-600,100)
plt.title('Relu Score')

# plot training time
plt.figure()
relu['mean_fit_time'].plot()
plt.title('Relu training time')


# ######################################################################
# combine all the scores
#
plt.figure()
plt.plot(identity['mean_train_score'].values,label='identity')
plt.plot(logistic['mean_train_score'].values,label='logistic')
plt.plot(tanh['mean_train_score'].values,label='tanh')
plt.plot(relu['mean_train_score'].values,label='relu')
plt.ylabel('Score')
plt.legend()
plt.title('training')


# Test
plt.figure()
plt.plot(identity['mean_test_score'].values,label='identity')
plt.plot(logistic['mean_test_score'].values,label='logistic')
plt.plot(tanh['mean_test_score'].values,label='tanh')
plt.plot(relu['mean_test_score'].values,label='relu')
plt.ylabel('Score')
plt.legend()
plt.title('testing')


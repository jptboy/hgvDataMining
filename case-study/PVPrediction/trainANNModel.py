# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause


from __future__ import division
import time

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import os

# define the path
workDir ='C:/Github/hgvDataMining/'
data_dir = workDir+'case-study/PVPrediction/'

os.chdir(workDir)

# read traingn and testing data from pkl file
train_June = joblib.load(data_dir+'Train-June.pkl')
test_July = joblib.load(data_dir+'Test-July.pkl')

# drop NAN
# drop NAN
train = train_June.dropna(axis=0,how='any')
test = test_July.dropna(axis=0,how='any')

# ######################################################################################
# ####################    Read training data
# X has to be a m-by-n matrix
X_train = train[['IrrDiffuse (W/m2)','IrrDirect (W/m2)','IrrGlobal (W/m2)']].as_matrix()
# y has to be a m-by-1 matrix
Y_train = train[['CH3-Solar Input (F1)']].as_matrix()

# #############################################################################
# ###################    Read Testing data
X_test = test[['IrrDiffuse (W/m2)','IrrDirect (W/m2)','IrrGlobal (W/m2)']].as_matrix()
# y has to be a m-by-1 matrix
Y_test = test[['CH3-Solar Input (F1)']].as_matrix()

# #############################################################################
# Fit regression model
# First, normalize the data. Create a scaler based on training data
scaler = StandardScaler().fit(X_train)

# Dimension reduction

# Second, create a ANN estimator
ann = MLPRegressor(solver='lbfgs',alpha=0.001)

# Third, create steps
steps = [('normalize',scaler),('reg',ann)]

# Foruth, create pipeline for evaluation
pipe = Pipeline(steps)

# Fifth, perform model selection using grid search to find the best trained ANN model
# size of layers and nurons
layers = [1,2,3]
size = [10,20,30,40,50]
hidden_layer_sizes=[]
for i in layers:
    for j in size:
        a = (j,)*i
        hidden_layer_sizes.append(a)

estimator = GridSearchCV(pipe,
                   param_grid={'reg__hidden_layer_sizes':hidden_layer_sizes,
                               'reg__alpha':[1e-4,1e-3,1e-2,1e-1,1],
                               'reg__activation':['identity','logistic','tanh','relu']},
                   cv=5,scoring='neg_mean_absolute_error')

# fit the model using grid searching validation
t0 = time.time()
estimator.fit(X_train, Y_train)
ann_fit = time.time() - t0
print("ANN complexity and bandwidth selected and model fitted in %.3f s"
      % ann_fit)

t0 = time.time()
y_ann = estimator.predict(X_test)
ann_predict = time.time() - t0
print("ANN prediction for %d inputs in %.3f s"
      % (X_test.shape[0], ann_predict))
# #############################################################################
# Look at the results
r2 = r2_score(Y_test,y_ann)

plt.scatter(Y_test, y_ann, c='b',label='PV Generation')
plt.xlabel('Measurement [W]')
plt.ylabel('Prediction [W]')
plt.legend()

# Visualize training and prediction 
plt.figure()
plt.plot(Y_test,'-b')
plt.plot(y_ann,'-.r')
# Generate sample data
# Visualize learning
plt.show()
## Model persistence: save model in pickle
joblib.dump(estimator, 'ann.pkl')

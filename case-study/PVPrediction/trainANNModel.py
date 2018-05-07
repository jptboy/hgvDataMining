# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause


from __future__ import division
import time

import numpy as np

from sklearn.model_selection import GridSearchCV
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

rng = np.random.RandomState(0)

# #############################################################################
# Generate sample data
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

X_plot = np.linspace(0, 5, 100000)[:, None]

# Size of X, y, following numpy matrix array to avoid intorducing errors
y = y.reshape([len(y),1])

# #############################################################################
# Fit regression model
# First, normalize the data. Create a scaler based on training data
scaler = StandardScaler().fit(X)

# To reduce the computation time, here choose 100 samples as training data
train_size = 100

# Dimension reduction

# Second, create a ANN estimator
ann = MLPRegressor(solver='lbfgs',alpha=0.001)

# Third, create steps
steps = [('normalize',scaler),('reg',ann)]

# Foruth, create pipeline for evaluation
pipe = Pipeline(steps)

# Fifth, perform model selection using grid search to find the best trained ANN model
estimator = GridSearchCV(pipe,
                   param_grid={'reg__alpha':[1e-4,1e-3,1e-2,1e-1,1],
                               'reg__activation':['identity','logistic','tanh','relu']},
                   cv=5,scoring='neg_mean_absolute_error')

# fit the model using grid searching validation
t0 = time.time()
estimator.fit(X[:train_size], y[:train_size])
ann_fit = time.time() - t0
print("ANN complexity and bandwidth selected and model fitted in %.3f s"
      % ann_fit)

t0 = time.time()
y_ann = estimator.predict(X_plot)
ann_predict = time.time() - t0
print("ANN prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], ann_predict))
# #############################################################################
# Look at the results

plt.plot(X_plot, y_ann, c='b',
         label='ANN (fit: %.3fs, predict: %.3fs)' % (ann_fit, ann_predict))
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus ANN')
plt.legend()

# Visualize training and prediction time
plt.figure()

# Generate sample data

# Visualize learning
plt.show()
## Model persistence: save model in pickle
joblib.dump(estimator, 'ann.pkl')

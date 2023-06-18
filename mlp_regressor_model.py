#MLP Regressor model fit and scoring

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from helpers.models_helpers import *

cost = pd.read_csv('./data/cost_processed.csv')
tasks = pd.read_csv('./data/tasks_processed.csv')
suppliers = pd.read_csv('./data/suppliers_processed.csv')

# Prepare data for input and outputs of ML models
#Set Supplier ID as index for cost and tasks 
cost.set_index('Supplier ID', inplace=True)
suppliers.set_index('Supplier ID', inplace=True)

#Merge suppliers and cost in one new df
combined_df = pd.merge(suppliers, cost, left_index=True, right_index=True)

#Set Task ID as index for cost and tasks
combined_df.set_index('Task ID', inplace=True)
tasks.set_index('Task ID', inplace=True)

#Combine tasks with the new df
combined_df = pd.merge(tasks, combined_df, on='Task ID')

# Split dataset into x, y and groups
x = combined_df.iloc[:,1:-1]
y = combined_df['Cost']
groups = combined_df.index

#Produces same results as in the report
np.random.seed(47)

#Split dataset randomly into train and test (100 training : 20 test) (3.2)
x_train, x_test, y_train, y_test = random_split_train_test(groups, x, y)

#Train model (3.3)
mlp = MLPRegressor(max_iter=1000)
mlp.fit(x_train, y_train)

print("MLP - R2 Score(test):", mlp.score(x_test, y_test))

#3.4
print('MLP TEST DATA SCORES')
calculate_error_rmse(x_test, y_test, mlp)


#Leave-One-Group-Out cross-validation (4)
print('MLP CROSS VALIDATION - LEAVE ONE GROUP OUT RESULTS')
error_scorer = make_scorer(calculate_error)

logo = LeaveOneGroupOut()

#Cross Validation scores
error_scores = cross_val_score(mlp, x, y, cv=logo, groups=groups, scoring=error_scorer)
print('Selection Errors -',f"{error_scores.mean():0.2f} (+/- {error_scores.std()*2:0.2f})\n", error_scores)

rmse_cv = np.sqrt(np.mean(error_scores ** 2))
print('RMSE score:', rmse_cv)


#Hyper-Parameter Optimzation (5)
print('MLP HYPER-PARAMETER OPTIMIZATION - RESULTS')
# Parameters and values to tune
param_grid = dict(hidden_layer_sizes= [50, 100, (25,25), (50,50)],
                   solver = ['lbfgs','sgd', 'adam'])

model_tuning(mlp, param_grid, error_scorer, x, y, groups)
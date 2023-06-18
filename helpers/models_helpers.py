# - Preparation of  input/output data for ML models 
# - Helper functions for splitting train/test data, 
#   fitting and calculating error/RMSE of models and hyper-parameter tuning

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut


# Helper functions for ML models
#Split data to training and testing (3.2)
def random_split_train_test(groups, x, y): 
    testGroup = np.random.choice(np.unique(groups), 20, replace=False)
    
    x_test = x.loc[x.index.isin(testGroup),:]
    y_test = y.loc[y.index.isin(testGroup)]

    x_train = x.loc[~x.index.isin(testGroup),:]
    y_train = y.loc[~y.index.isin(testGroup)]

    return x_train, x_test, y_train, y_test

#Calculate Error and RMSE for a specific ML model (3.4)
def calculate_error_rmse(x, y, model):
    #Group input data by Task ID, predict cost of suppliers for each grouped task and 
    #find index (supplier) of minimum predicted cost for each task
    predictions_index_per_group = x.groupby('Task ID').apply(lambda group: np.argmin(model.predict(group)))
    #Update indexes to match locations of ungrouped data
    predictions_index = predictions_index_per_group + range(0, len(x), 64)
    #Locate real cost values of supplier chosen by ML
    predicted_supplier_cost = y.iloc[predictions_index]
    
    min_true_cost = y.groupby('Task ID').min()

    #Calculate  Selection Error
    error = min_true_cost - predicted_supplier_cost
    print('Selection Errors -',f"{error.mean():0.2f} (+/- {error.std()*2:0.2f})\n", error)

    #Calculate RMSE
    rmse = np.sqrt(np.mean(error ** 2))
    print('RMSE score:', rmse)

#Scoring function to calculate error between true min cost and predicted min cost
def calculate_error(y_true, y_pred):
    min_predict_supplier = np.argmin(y_pred)
    min_true_cost = y_true.min()
    
    return min_true_cost - y_true[min_predict_supplier]

#Hyper Parameter Tuning to find best parameters of model and 
#calculate RMSE of cross validation scores from best model
def model_tuning(model, param_grid, error_scorer, x, y, gps):
    logo = LeaveOneGroupOut()
    
    grid_search = GridSearchCV(model, param_grid, cv=logo, scoring=error_scorer, n_jobs=-1)
    
    #Run the search and fit best model
    grid_search.fit(x, y, groups=gps)
    
    print("Best parameter combination after the grid search:", grid_search.best_params_)
    
    #Find cross-validation results of best parameter combination
    gs_cv_results = pd.DataFrame(grid_search.cv_results_)
    gs_errors = gs_cv_results.iloc[grid_search.best_index_].iloc[8:-3]
    
    #Cross-validation errors
    print('Selection Errors -',f"{gs_errors.mean():0.2f} (+/- {gs_errors.std()*2:0.2f})\n", gs_errors)
    rmse_gs = np.sqrt(np.mean(gs_errors ** 2))
    print('RMSE score: ', rmse_gs)
        
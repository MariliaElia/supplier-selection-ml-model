#Data Tidying and Pre-processing - preperation for ML models
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.data_preparation_helpers import *

cost = pd.read_csv('./data/cost.csv')
tasks = pd.read_excel('./data/tasks.xlsx')
suppliers = pd.read_csv('./data/suppliers.csv')

# Convert dates to datetime objects
cost['Task ID'] = pd.to_datetime(cost['Task ID'], infer_datetime_format=True, format='%d/%m/%Y')
tasks['Task ID'] = pd.to_datetime(tasks['Task ID'], infer_datetime_format=True, format='%Y %m %d')


print("\n---------- Completeness of datasets verification ----------")
suppliers = remove_missing_ids(suppliers, cost, 'Supplier ID')
tasks = remove_missing_ids(tasks, cost, 'Task ID')

check_missing_values(suppliers, 'suppliers')
check_missing_values(tasks, 'tasks')
check_missing_values(cost, 'cost')
    
#Number of tasks, suppliers, features & cost values
print('\nNumber of tasks:', tasks.shape[0])
print('Number of task features:', tasks.shape[1])
print('Number of suppliers:', suppliers.shape[0])
print('Number of supplier features:', suppliers.shape[1])
print('Number of cost values:', cost.shape[0])


#Update indexes
suppliers.set_index('Supplier ID', inplace=True)
tasks.set_index('Task ID', inplace=True)

print("\n---------- Supplier Features - Descriptive Stats ----------")
descriptive_stats(suppliers)
print("\n---------- Task Features - Descriptive Stats ----------")
descriptive_stats(tasks)

print("\n---------- Features with low variance(<0.01) ----------")
print("\nSuppliers low variance check:")
suppliers = remove_low_variance_features(suppliers)
print("\nTasks low variance check:")
tasks = remove_low_variance_features(tasks)


#Scaling Data (Normalisation)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
suppliers[:] = scaler.fit_transform(suppliers)
tasks[:] = scaler.fit_transform(tasks)


print("\n---------- Absolute Correlation of task features ----------")
corr_matrix = tasks.corr().abs()
print("\nTasks Feature Correlation before removing large correlations(>0.08): \n",corr_matrix)

#Visualization of tasks features correlations
plt.figure(1, figsize =(19, 10))
sns.heatmap(corr_matrix, cmap='coolwarm',linewidth = 0.5, xticklabels=True)
plt.title( "Absolute Correlation of task features" )
plt.show()

# Remove large correlation of task features (>=0.08)
tasks_processed_df  = remove_large_corr_features(tasks, corr_matrix)

#Visualization of tasks features correlations after removing highly correlated features
updated_corr_matrix = tasks_processed_df.corr().abs()

plt.figure(2, figsize =(10, 8))
sns.heatmap(updated_corr_matrix, cmap='coolwarm',linewidth = 0.5, xticklabels=True)
plt.title( "Absolute Correlation of task features without high correlated ones" )
plt.show()

# Updates included - Removed missing IDs, low variance features, feature correlation >= 0.8, scaling
tasks_processed_df.to_csv('./data/tasks_processed.csv')
# Updates included - Scaling
suppliers.to_csv('./data/suppliers_processed.csv')


print("\n---------- Top 20 suppliers of each task ----------")
#Find top 20 suppliers per task
top_20_suppliers = cost.sort_values(by=['Task ID','Cost']).groupby('Task ID').head(20)
print(top_20_suppliers)

#Remove supplier IDs that don't appear in the top 20 of any task
suppliers.reset_index(inplace=True)
suppliers = remove_missing_ids(suppliers, top_20_suppliers, 'Supplier ID')
top_20_suppliers.set_index('Task ID', inplace=True)

# Write processed cost file - updates included: Date Format
cost.set_index('Task ID', inplace=True)
cost.to_csv('./data/cost_processed.csv')
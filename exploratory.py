#Exploratory Data Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cost = pd.read_csv('./data/cost.csv')
tasks = pd.read_csv('./data/tasks_processed.csv')

# Boxplots for each task feature
tasks.set_index('Task ID', inplace=True)
plt.figure(1, figsize =(15, 7))
plt.boxplot(tasks, labels=tasks.columns)
plt.title("Distribution of feature values for each task feature")
plt.xlabel("Task Features")
plt.ylabel("Task Feature Values")


#Selection Error
selection_error = cost[['Task ID', 'Supplier ID', 'Cost']]
#Find minimum cost(best supplier) for each task
min_cost_per_task = cost.groupby('Task ID')['Cost'].min()
selection_error.set_index('Task ID', inplace=True)
selection_error['Min_Cost'] = min_cost_per_task
#Calculate error of a supplier being selected to perform a task
selection_error['Error']  = selection_error['Min_Cost']  - selection_error['Cost']

#Calculate RMSE of each supplier
selection_error.reset_index(inplace=True)
selection_error_wide = selection_error.pivot(index ='Task ID',values = 'Error',columns = 'Supplier ID')
rmse = np.sqrt(np.mean(selection_error_wide ** 2, axis=0))

#Labels for boxplot
#Convert values to string and concatenate with Supplier ID to display on the boxplot as labels   
rmse_str = round(rmse, 4).values.astype(str)
labels = rmse.index + ' - ' + rmse_str

# Distribution of Errors for each supplier plot
plt.figure(2, figsize =(15, 9))
plt.boxplot(selection_error_wide, labels=labels)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.title("Distribution of Errors per Supplier")
plt.xlabel("Suppliers (annotated with RMSE)")
plt.ylabel("Selection Error")

#Heatmap plot showing cost values of tasks for each supplier
cost_wide = cost.pivot(index ='Task ID',values = 'Cost',columns = 'Supplier ID')

plt.figure(3, figsize =(17, 9))
sns.heatmap(cost_wide, linewidth=0.5, cmap='coolwarm')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.title( "Cost values of tasks and suppliers" )
plt.xlabel("Suppliers")
plt.ylabel("Tasks")
plt.show()
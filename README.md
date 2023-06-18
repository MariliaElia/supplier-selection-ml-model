# Supplier selection using Multi-Layer Perceptron and Random Forest Regressors

Developed as a course project in Business Analytics: Operational Research and Risk Analysis program at Alliance Manchester Business School.

# Project Overview

The objective of this project is to build an ML model that selects the most cost-efficient supplier among the sixty-four available to complete a specific task every day.

The two ML models developed are Multi-Layer Perceptron and Random Forest Regressors. 

The selection error made by the ML model was measured as:

*Equation 1:*

![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/bebfb036-a3df-4b31-a8ba-be152552156b)

where 𝑡 is a task, 𝑆 is the set of 64 suppliers, 𝑠𝑡′ is the supplier chosen by the ML model for task 𝑡, and 𝑐(𝑠,𝑡) is the cost if task 𝑡 is performed by supplier 𝑠. That is, the Error is the difference in cost between the supplier selected by the ML model and the actual best supplier for this task.

A score for each ML model was then computed using the root mean squared error (RMSE) over the tasks used for validation:

*Equation 2:*

![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/22556c64-8674-44ac-b263-98f89792f21b)

where 𝑇 is a set of tasks and Error(t) is the selection error calculated from Equation 1.

# Installation and Setup

## Codes and Resources Used
- **Editor Used:**  Spyder IDE
- **Python Version:** Python 3.9.13

## Python Packages Used
- **General Purpose:** `math`
- **Data Manipulation:** `pandas, numpy` 
- **Data Visualization:** `seaborn, matplotlib`
- **Machine Learning:** `scikit-learn`

# Data

## Source Data
- `tasks.xlsx`: contains one row per task and one column per task feature.
- `suppliers.csv`: contains one row per supplier and one column per supplier feature.
- `costs.csv`: contains data collected and/or estimated by the business about the cost of a task when performed by each supplier. Each row gives the cost value of one task performed by one supplier.

## Data Preparation

The below steps were followed in order to tidy and pre-process the data before building the ML models:
- Verified the completeness of each dataset and removed incomplete observations
- Checked for null values
- Removed features with variance less than 0.01
- Data scaling - normalisation of supplier and task features using MinMaxScaler 
- Removed highly correlated pairs between task features

# Code structure
- `./data`: contains initial data files and processed ones that are created by running `data_preparation.py`
- `./helpers`: two helper files have been created under `./helpers` folder, containing functions that assist data preparation and modelling. These files should be run first.
- `data_peparation.py`: Data tidying and pre-processing - preparation for ML models
- `exploratory.py`: EDA and graphs
- `mlp_regressor_model.py`: MLP Regressor model fit and scoring
- `random_forest_model.py`: Random Forest model fit and scoring

# Results and evaluation
## EDA

### Figure 1

![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/f2834611-6112-41ad-b670-3f5a30f9ef4b)

It can be observed that the distributions differ considerably from each other, a lot of outliers are defined for each task feature and only some features are normally distributed

### Figure 2

![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/e372ed10-4748-48a3-b820-47f027efdf8c)

It is observed that the maximum selection error of most suppliers is less than 0.10. Supplier 56 has the smallest error distribution with an RMSE of 0.0256, meaning that the costs of this supplier for each task are very close to the costs of the actual best suppliers and they would be a great choice if selected to perform all tasks. This supplier can be used as a benchmark for the ML models.

### Figure 3

![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/0db2753f-9948-423a-b512-dc82ef528c94)

Overview of the cost values for every supplier when performing each individual task. The below observation can be obtained:
- There is one specific task where the cost values are higher for all the suppliers.
- Task with ID ‘26/09/2019’ seems to have the lowest costs for all the suppliers.
- Suppliers 2 and 3 have the highest costs for the majority of the tasks.
- Most of the tasks have costs below 0.5.

## ML Models
The ML models were developed using the following final dataframe, where each task appears multiple times – once for each supplier – formulating a group. Each group corresponds to a specific task and includes its feature values, all the possible suppliers (with their features), and their cost to perform the task. The ML models were used to predict the output variable, which is supplier costs.

![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/f988e6e5-9b01-4abd-96a0-5fca4e116837)

### Fitting of the models
The 2 models were fitted using 3 different approaches:
- Random train/test split (20 task groups out of 120 as a test set)
- Leave-One-Group-Out Cross Validation
- Hyper-Parameter Optimisation

A random split cannot be indicative of the performance of a model (uncertainty about the information in the train dataset), cross-validation was performed with Equation 1 being the scoring function (Errors) and the RMSE as a metric this time. Cross–validation reduces bias and variance as most of the data is being used for fitting and as a validation set. The Leave-One-Group-Out method was used for the cross-validation, leaving one group of tasks out at a time for validation purposes.

The previous procedures were done for the default hyper-parameters of each model. Thus, as the last step, the Grid Search method (scikit-learn) was used for the hyper-parameter tuning of each model. The same approach regarding the scoring function (Equation 1), the performance metric (Equation 2), and the type of cross-validation (Leave-One-Group-Out) was followed for the hyper-parameter optimization.

Results obtained from each approach for each model can be seen below.

### Multi-Layer Perceptron Regressor Results
![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/256d9527-8acd-4afc-99e6-51d7857eb016)

### Random Forest Regressor Results
![image](https://github.com/MariliaElia/supplier-selection-ml-model/assets/24305018/61db0e22-64a8-4c53-899f-b0032fa642ee)

# Conclusions
Comparing the R-squared and RMSE scores obtained before cross-validation and hyper-parameter optimisation, it is noted that the RMSE of MLP Regressor is higher than the one of Random Forest Regressor, while the R-squared value is lower. This perceives that the Random Forest algorithm fits the data with better performance than the MLP Regressor.

Regarding the RMSEs of the ML models after cross-validation and hyper-parameter optimisation, the better option is still Random Forest regressor, but considering the suppliers RMSEs, supplier 56 remains the best choice for completing a task.

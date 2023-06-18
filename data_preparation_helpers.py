# Helper functions for data pre processing 
import pandas as pd
import numpy as np

#Verify IDs in one dataframe match with the other
#Checks whether each element in id_name column in df1 is in df2 - remove missing ones from df1
def remove_missing_ids(df1, df2, id_name): 
    if(df1[id_name].isin(df2[id_name]).all()):
        print(id_name + "s match in dataframes\n")
        df1 = df1
    else:
        #Remove missing IDs
        print("Removing" , len(df1[~df1[id_name].isin(df2[id_name])]),
              "row(s) for missing", id_name,"from the first dataframe.\n")
        df1 = df1[df1[id_name].isin(df2[id_name])]

    return df1

#Checks for null values
def check_missing_values(df, df_name):
    print("Does" ,df_name, "dataframe have any missing values?", np.any(df.isna()))
    if (np.any(df.isna())):
        print("Number of missing values per column:\n", df.isna().sum())
        
#Max / Min / Mean / Var
def descriptive_stats(x):
    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)
    mean = np.mean(x,axis=0)
    variance = np.var(x,axis=0)
  
    print("\nThe minimum for each feature in the dataset:\n",minimum)
    print("\nThe maximum for each feature in the dataset:\n",maximum)
    print("\nThe mean for each feature in the dataset:\n", mean)
    print("\nThe variance for each feature in the dataset:\n", variance)
    

def remove_low_variance_features(x):
    variance = np.var(x,axis=0)

    #Check for variances less than 0.01
    low_variance = variance[variance<0.01]

    if(low_variance.any()):
        print("Found",len(low_variance), "features with variance <0.01. Removing features from dataframe.")
        x = x.drop(labels=low_variance.index, axis='columns')
    else:
        print('No features found with variance <0.01')
    
    return x

#Absolute Correlation 

#Takes in dataframe and correlation matrix of df, removes high correlated features and returns new dataframe
def remove_large_corr_features(df, corr):
    #Create a long dataframe  with two columns for feature pairs and a column showing correlation of the pair
    corr_long_data = pd.melt(corr, ignore_index=False, 
                             value_vars=corr.columns, 
                             var_name="Feature", 
                             value_name = 'Correlation')

    #Remove correlations of same feature pairs from dataframe
    corr_long_data = corr_long_data[corr_long_data.index != corr_long_data['Feature']]

    #Check if there are any correlations bigger or equal to 0.8
    if(~np.any(corr_long_data['Correlation'] >= 0.8)):
        print("\nTasks Feature Correlation without large correlations (>0.8):\n", corr)
        return (df)
    else:
        #Find correlations larger than 0.8 and sort them in ascending order
        large_corr_pairs = corr_long_data[corr_long_data['Correlation'] >= 0.8].sort_values(by='Correlation', 
                                                                                            ascending=False)
 
        #Correlation sums per feature
        corr_feature_sums = corr_long_data.groupby('Feature')['Correlation'].sum()
        
        #Get feature names of feature pair with highest correlation in large_corr_pairs dataframe
        f1 = large_corr_pairs.first_valid_index()
        f2 = large_corr_pairs['Feature'].iloc[0]
            
        #Compare corr sums of each feture, find the one that has highest sum of correlations and remove it 
        if (corr_feature_sums[f1] > corr_feature_sums[f2]):
            df = df.drop(labels=f1, axis='columns')
        else:
            df = df.drop(labels=f2, axis='columns')
        
        #Recalculate correlation and repeat process until left with all correlations <= 0.8
        corr = df.corr().abs()
        return remove_large_corr_features(df, corr)
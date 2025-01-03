import pandas as pd
import numpy as np
import os 
from scipy.stats import zscore

train_file = "./data/raw/train.csv"
test_file = "./data/raw/test.csv"

train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)


def remove_outlier_with_zscore(df, columns=None, threshold=3):
    """
    Removes outliers from the dataframe based on Z-score.

    Parameters:
    - df (pd.DataFrame): The dataframe to process.
    - columns (list or None): List of columns to process. If None, numeric columns are considered.
    - threshold (float): The Z-score threshold to determine outliers (default is 3).

    Returns:
    - pd.DataFrame: Dataframe with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in columns:
        # Calculate Z-scores
        df['zscore'] = zscore(df[col])
        
        # Filter rows where the absolute Z-score is below the threshold
        df = df[abs(df['zscore']) <= threshold]
        
        # Drop the temporary Z-score column
        df = df.drop(columns=['zscore'])
    
    return df  
processed_train_data=remove_outlier_with_zscore(train_data)
processed_test_data=remove_outlier_with_zscore(test_data)
data_path=os.path.join("data","processed")
os.makedirs(data_path,exist_ok=True)
processed_train_data.to_csv(os.path.join(data_path,"processed_train_zscore.csv"),index=False)
processed_test_data.to_csv(os.path.join(data_path,"processed_test_zcore.csv"),index=False)
        
    
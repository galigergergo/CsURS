import torch
import os
import pandas as pd
from typing import Union


def read_dataset(data_path: str, file_name: str, targets: str, device: str) -> Union[torch.Tensor, torch.Tensor]:
    '''
    Reads CSV dataset, logs it to MLflow and returns X and y PyTorch tensors.
    '''    
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    
    # Move accident number columns to the right side of the dataframe 
    df = df[list(df.columns[~df.columns.str.contains(pat='rs_crashes_')]) + list(df.columns[df.columns.str.contains(pat='rs_crashes_')])]
    non_accident_dim = len(list(df.columns[~df.columns.str.contains(pat='rs_crashes_')]))

    return torch.tensor(df.drop(targets, axis=1).values, device=device).float(), torch.tensor(df[[targets]].values, device=device).float(), non_accident_dim


def read_dataset_config(data_path: str, file_name: str) -> pd.DataFrame:
    '''
    Reads CSV dataset config as pandas DataFrame.
    '''    
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)

    return df

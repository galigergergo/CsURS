# Crowdsourced Urban Road Safety Analysis

Source code for the submitted manuscript entitled: Crowdsourced data for urban road safety analysis via neural network interpretation

## Preliminaries for training models

1. Install mlflow for experiment tracking: `pip install mlflow`
2. Run mlflow tracking server in a directory outside of this project: `mlflow server --host 127.0.0.1 --port 8080`
3. Reach mlflow tracking server UI at: `http://127.0.0.1:8080/`

## Structure

- data: generating the proposed urban road safety dataset from crowdsourced data sources
- model: training and evaluating the proposed hybrid model for crash frequency prediction
1. **data**: main datasets + normalized datasets for NN training
2. **src**: source code for models, training and data handling
2. **01_DAE_train.ipynb**: training and evaluation of the first DAE module
3. **02_FCN_train.ipynb**: training and evaluation of the second FCN module

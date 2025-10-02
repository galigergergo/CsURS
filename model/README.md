# Graph Neural Networks for Crowdsourced Urban Road Safety Analysis

## Preliminaries for training models

1. Install mlflow for experiment tracking: `pip install mlflow`
2. Run mlflow tracking server in a directory outside of this project: `mlflow server --host 127.0.0.1 --port 8080`
3. Reach mlflow tracking server UI at: `http://127.0.0.1:8080/`

## Repository structure

1. **data**: main datasets + normalized datasets for NN training
2. **src**: source code for models, training and data handling
3. **00_create_NN_dataset.ipynb**: **!!! OLD CODE !!!** code for producing NN training datasets from main datasets 
4. **00_explore_data.ipynb**: initial data exploration for provided datasets
5. **01_DAE_train.ipynb**: training and evaluation of the first DAE module
6. **02_FCN_train.ipynb**: training and evaluation of the second FCN module - the AE_RUN_ID parameter should be set based on the mlflow run after running the **01_DAE_train.ipynb** notebook, e.g. e5a5b25fc25f4435bd9177664c21ff9a based on the output example below:
🏃 View run popular-slug-721 at: http://127.0.0.1:8080/#/experiments/775881003945806667/runs/e5a5b25fc25f4435bd9177664c21ff9a

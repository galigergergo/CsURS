import torch
import mlflow
import numpy as np
from tqdm import tqdm
from src.models.AEFCN import AE, FC, AEFCN


class AETrainer():
    def __init__(self, inp_dim, noise_factor, enc_dim=4, learning_rate=1e-3, weight_decay=1e-8):
        self.inp_dim = inp_dim
        self.noise_factor = noise_factor
        self.enc_dim = enc_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = AE(self.inp_dim, self.enc_dim, self.noise_factor)
        self.criterion = torch.nn.MSELoss()
        self.crit_text = 'mse'
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    
    def train(self, X_train, y_train, X_valid, y_valid, epochs):
        X_train, X_valid = X_train[:, :self.inp_dim], X_valid[:, :self.inp_dim]
        y_train = (X_train + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)).float()
        y_valid = (X_valid + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_valid.shape)).float()
        
        with torch.no_grad():
            model_signature = mlflow.models.infer_signature(X_train.numpy(), self.model(X_train).numpy())

        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            
            reconstr = self.model(X_train)
            
            loss = self.criterion(reconstr, y_train)
            loss.backward()
            
            self.optimizer.step()

            mlflow.log_metric('loss_mse_train', loss.item(), step=epoch)

            # validation
            with torch.no_grad():
                reconstr = self.model(X_valid)
                loss = self.criterion(reconstr, y_valid)
                mlflow.log_metric('loss_mse_validate', loss.item(), step=epoch)

        mlflow.pytorch.log_model(self.model, artifact_path=f'models/AE_ep{epoch}', signature=model_signature)

    
    def evaluate(self, X_test, y_test, eval_df_text, criterion=None, crit_text=None):
        assert not ((type(criterion) != type(None)) ^ (crit_text != None))
        with torch.no_grad():
            X_test = X_test[:, :self.inp_dim]
            y_test = (X_test + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)).float()
            pred = self.model(X_test)
            if type(criterion) == type(None):
                loss = self.criterion(pred, y_test).item()
                mlflow.log_metric(f'eval_{self.crit_text}_{eval_df_text}', loss)
            else:
                loss = criterion(pred, y_test).item()
                mlflow.log_metric(f'eval_{crit_text}_{eval_df_text}', loss)


class AEFCNTrainer():
    def __init__(self, autoencoder, inp_dim, learning_rate=1e-3):
        self.ae = autoencoder

        self.inp_dim = inp_dim
        self.learning_rate = learning_rate
        self.fc = FC(inp_dim)
        self.criterion = torch.nn.MSELoss()
        self.crit_text = 'mse'
        self.optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.learning_rate)
        
    
    def train(self, X_train, y_train, X_valid, y_valid, epochs, min_max_norms=None):
        # Split data for the autoencoder
        X_train_ae, X_valid_ae = X_train[:, :self.ae.inp_dim], X_valid[:, :self.ae.inp_dim]

        # Encode data with autoencoder
        X_train_enc = self.ae.encode(X_train_ae).detach()
        X_valid_enc = self.ae.encode(X_valid_ae).detach()

        # Concatenate encoded data with previous year accident data
        X_train_fc = torch.cat((X_train_enc, X_train[:, self.ae.inp_dim:]), 1)
        X_valid_fc = torch.cat((X_valid_enc, X_valid[:, self.ae.inp_dim:]), 1)

        # Train and validate FC network
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            
            reconstr = self.fc(X_train_fc)
            
            loss = self.criterion(reconstr, y_train)
            loss.backward()
            
            self.optimizer.step()

            if min_max_norms != None:
                with torch.no_grad():
                    pred = reconstr.clone().detach() * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                    targ = y_train.clone().detach() * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                    mlflow.log_metric('loss_mse_train', self.criterion(pred, targ).item(), step=epoch)
            else:
                mlflow.log_metric('loss_mse_train', loss.item(), step=epoch)

            # Validation
            with torch.no_grad():
                reconstr = self.fc(X_valid_fc)
                y_valid_n = y_valid.clone()
                if min_max_norms != None:
                    reconstr = reconstr * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                    y_valid_n = y_valid * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                loss = self.criterion(reconstr, y_valid_n)
                mlflow.log_metric('loss_mse_validate', loss.item(), step=epoch)

        # Define AEFCN model from trained AE and FC networks
        aefcn_model = AEFCN(self.ae, self.fc)
        self.model = aefcn_model
        
        # Infer model signature for MLflow
        with torch.no_grad():
            model_signature = mlflow.models.infer_signature(X_train.numpy(), aefcn_model(X_train).numpy())
        
        # Log model as an MLflow artifact
        mlflow.pytorch.log_model(self.model, artifact_path=f'models/AEFCN_ep{epoch}', signature=model_signature)
    
    def evaluate(self, X_test, y_test, eval_df_text, criterion=None, crit_text=None, min_max_norms=None, proc_func=None):
        assert not ((type(criterion) != type(None)) ^ (crit_text != None))
        with torch.no_grad():
            pred = self.model(X_test)
            if min_max_norms != None:
                pred = pred * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                y_test = y_test * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
            if proc_func != None:
                pred = proc_func(pred)
                y_test = proc_func(y_test)
            if type(criterion) == type(None):
                loss = self.criterion(pred, y_test).item()
                mlflow.log_metric(f'eval_{self.crit_text}_{eval_df_text}', loss)
            else:
                loss = criterion(pred, y_test).item()
                mlflow.log_metric(f'eval_{crit_text}_{eval_df_text}', loss)


class FCNTrainer():
    def __init__(self, inp_dim, learning_rate=1e-3):
        self.inp_dim = inp_dim
        self.learning_rate = learning_rate
        self.fc = FC(inp_dim)
        self.criterion = torch.nn.MSELoss()
        self.crit_text = 'mse'
        self.optimizer = torch.optim.Adam(self.fc.parameters(), lr=self.learning_rate)
        
    
    def train(self, X_train, y_train, X_valid, y_valid, epochs, min_max_norms=None):
        # Train and validate FC network
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            
            reconstr = self.fc(X_train)
            
            loss = self.criterion(reconstr, y_train)
            loss.backward()
            
            self.optimizer.step()

            if min_max_norms != None:
                with torch.no_grad():
                    pred = reconstr.clone().detach() * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                    targ = y_train.clone().detach() * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                    mlflow.log_metric('loss_mse_train', self.criterion(pred, targ).item(), step=epoch)
            else:
                mlflow.log_metric('loss_mse_train', loss.item(), step=epoch)

            # Validation
            with torch.no_grad():
                reconstr = self.fc(X_valid)
                y_valid_n = y_valid.clone()
                if min_max_norms != None:
                    reconstr = reconstr * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                    y_valid_n = y_valid * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                loss = self.criterion(reconstr, y_valid_n)
                mlflow.log_metric('loss_mse_validate', loss.item(), step=epoch)

        # Infer model signature for MLflow
        with torch.no_grad():
            model_signature = mlflow.models.infer_signature(X_train.numpy(), self.fc(X_train).numpy())
        
        # Log model as an MLflow artifact
        mlflow.pytorch.log_model(self.fc, artifact_path=f'models/AEFCN_ep{epoch}', signature=model_signature)
    
    def evaluate(self, X_test, y_test, eval_df_text, criterion=None, crit_text=None, min_max_norms=None, proc_func=None):
        assert not ((type(criterion) != type(None)) ^ (crit_text != None))
        with torch.no_grad():
            pred = self.fc(X_test)
            if min_max_norms != None:
                pred = pred * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
                y_test = y_test * (min_max_norms[1] - min_max_norms[0]) + min_max_norms[0]
            if proc_func != None:
                pred = proc_func(pred)
                y_test = proc_func(y_test)
            if type(criterion) == type(None):
                loss = self.criterion(pred, y_test).item()
                mlflow.log_metric(f'eval_{self.crit_text}_{eval_df_text}', loss)
            else:
                loss = criterion(pred, y_test).item()
                mlflow.log_metric(f'eval_{crit_text}_{eval_df_text}', loss)


############################################
# imports
############################################

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils import scaling
from src.dataloader import CrisprDatasetTrain, CrisprDatasetEval
from src.architectures import Crispr1DCNN, CrisprGRU


############################################
# training and evaluation
############################################  

def train_deep_model(model_type, X_train_val, y_train_val, features, group_ids_train, fold_outer, n_folds_inner, batch_size, max_epochs, patience, output_dir_model_fold_outer):
    
    # scale tabular features and save scaler for evaluation
    SCALE = StandardScaler()
    SCALE.fit(X_train_val[features])
    
    filename_scaler = output_dir_model_fold_outer + "scaler.pickle"    
    pickle.dump(SCALE, open(filename_scaler, "wb"))
    
    # get cv splitter for inner k-folds
    fold_inner = 0
    gkf_inner = GroupKFold(n_splits=n_folds_inner)
    
    for index_train, index_val in gkf_inner.split(X_train_val, y_train_val, groups=group_ids_train):

        # get training and validation set
        X_train, X_val = X_train_val.iloc[index_train], X_train_val.iloc[index_val]
        y_train, y_val = y_train_val.iloc[index_train], y_train_val.iloc[index_val]

        # standardize training and validation set
        X_scale_train = scaling(SCALE, X_train, features)
        y_train.reset_index(inplace=True, drop=True)

        X_scale_val = scaling(SCALE, X_val, features)
        y_val.reset_index(inplace=True, drop=True)

        # load training and validation set
        dataset_train = CrisprDatasetTrain(X_scale_train, y_train, features)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers = 6, shuffle = True, drop_last=True)
         
        dataset_val  = CrisprDatasetTrain(X_scale_val, y_val, features)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers = 6, drop_last=True)

        # setup early stopping, checkpoints and tensor board
        filename_model = 'model_fold_inner_' + str(fold_inner)
        
        early_stop_callback = EarlyStopping(
                monitor="val_loss", 
                min_delta=0.0, 
                patience=patience, 
                verbose=False, 
                mode="min")
        
        checkpoint_callback = ModelCheckpoint(
                monitor = 'val_loss',
                dirpath = output_dir_model_fold_outer,
                filename = filename_model,
                verbose = False,
                save_top_k = 1,
                mode = 'min',)

        #logger = TensorBoardLogger('tb_logCNN', name = 'CNN-' + str(fold_outer) + "-" + str(fold_inner))

        # train model
        if model_type =='1DCNN':
            model = Crispr1DCNN(len(features))
        
        elif model_type =='GRU':
            model = CrisprGRU(len(features))
        
        trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=max_epochs, check_val_every_n_epoch=1, progress_bar_refresh_rate = 0, weights_summary=None)
        trainer.fit(model, train_dataloader = loader_train, val_dataloaders = loader_val)  
            
        fold_inner += 1

        
############################################

def evaluate_deep_model(model_type, X_train_val, y_train_val, X_test, y_test, features, fold_outer, n_folds_inner, performance, output_dir_model_fold_outer):

    # define predictions array
    predictions_fold_inner_train = np.zeros((X_train_val.shape[0], n_folds_inner))
    predictions_fold_inner_test = np.zeros((X_test.shape[0], n_folds_inner))

    # scale tabular features 
    SCALE = StandardScaler()
    SCALE.fit(X_train_val[features])

    X_scale_train_val = scaling(SCALE, X_train_val, features)
    y_train_val.reset_index(inplace=True, drop=True)
    
    X_scale_test = scaling(SCALE, X_test, features)
    y_test.reset_index(inplace=True,drop=True)
    
    
    # load training/validation and test set
    dataset_train = CrisprDatasetTrain(X_scale_train_val, y_train_val, features)
    loader_train = DataLoader(dataset_train, batch_size=X_scale_train_val.shape[0], num_workers = 6, shuffle = False)
   
    dataset_test  = CrisprDatasetTrain(X_scale_test, y_test, features)
    loader_test = DataLoader(dataset_test, batch_size=X_scale_test.shape[0], num_workers = 6, shuffle = False)
  
    # send to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
    # evaluate models of inner CV-folds
    for fold_inner in range(n_folds_inner):
             
        filename_model = output_dir_model_fold_outer + 'model_fold_inner_' + str(fold_inner) + ".ckpt"
        
        # load model
        if model_type =='1DCNN':
            model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(features))
        
        elif model_type =='GRU':
            model = CrisprGRU.load_from_checkpoint(filename_model, num_features = len(features))
         
        # evaluate model
        model = model.to(device)
        model.eval()
        model.freeze()
                
        # predict with model
        predictions_train = []
        for x_sequence_40nt, x_features, y in loader_train:
            with torch.no_grad():
                predictions = model(x_sequence_40nt.to(device), x_features.to(device)).detach()
            predictions_train.extend(predictions.cpu().numpy())
        predictions_fold_inner_train[:,fold_inner] = predictions_train
                
        predictions_test = []
        for x_sequence_40nt, x_features, y in loader_test:
            with torch.no_grad():
                predictions = model(x_sequence_40nt.to(device), x_features.to(device)).detach()
            predictions_test.extend(predictions.cpu().numpy())
        predictions_fold_inner_test[:,fold_inner] = predictions_test

    y_hat_train = predictions_fold_inner_train.mean(axis=1)
    y_hat_test = predictions_fold_inner_test.mean(axis=1)

    performance.iloc[fold_outer,:] = [round(mean_squared_error(y_train_val, y_hat_train), 2), round(mean_squared_error(y_test, y_hat_test), 2), 
                                      round(spearmanr(y_train_val, y_hat_train)[0], 4),round(spearmanr(y_test, y_hat_test)[0], 4)]
    
    return performance, y_hat_test


############################################
# run experiments
############################################

def run_deep_model(model_type, X, Y, features, group_ids, gkf, n_folds_outer, n_folds_inner, batch_size, max_epochs, patience, output_dir_model, X_add = None, Y_add = None, group_ids_add = None):

    fold_outer = 0
    
    performance = pd.DataFrame({'mse_train': np.zeros(n_folds_outer), 'mse_test': np.zeros(n_folds_outer), 
                                'spearmanR_train': np.zeros(n_folds_outer), 'spearmanR_test': np.zeros(n_folds_outer)})
    predictions = pd.DataFrame({'fold_outer':np.zeros(X.shape[0]), 'geneid': np.zeros(X.shape[0]), 
                                'log2FC_target': np.zeros(X.shape[0]), 'log2FC_predicted': np.zeros(X.shape[0])})
    
    for index_train, index_test in gkf.split(X, Y, groups=group_ids):
        
        X_train_val, X_test = X.iloc[index_train], X.iloc[index_test]
        y_train_val, y_test = Y.iloc[index_train], Y.iloc[index_test]
        group_ids_train = [group_ids[i] for i in index_train] 
        group_ids_test = [group_ids[i] for i in index_test] 
        
        if (X_add is not None) & (Y_add is not None):
            
            gene_idx_filter = [i for i, gene in enumerate(group_ids_add) if gene in group_ids_test]
            group_ids_add_filtered = [i for j, i in enumerate(group_ids_add) if j not in gene_idx_filter]
        
            X_add_filtered = X_add.copy().drop(gene_idx_filter)
            X_add_filtered.reset_index(inplace=True,drop=True)
    
            Y_add_filtered = Y_add.copy().drop(gene_idx_filter)
            Y_add_filtered.reset_index(inplace=True,drop=True)
            
            X_train_val = pd.concat([X_train_val,X_add_filtered],axis=0)
            y_train_val = pd.concat([y_train_val,Y_add_filtered],axis=0)
            group_ids_train = group_ids_train + group_ids_add_filtered

        # define output directory for each outer fold
        output_dir_model_fold_outer = output_dir_model + "/fold_outer_" + str(fold_outer) + "/"
        os.makedirs(os.path.dirname(output_dir_model_fold_outer), exist_ok=True)
        
        # train and evaluate model
        train_deep_model(model_type, X_train_val.copy(), y_train_val.copy(), features, group_ids_train, fold_outer, n_folds_inner, batch_size, max_epochs, patience, output_dir_model_fold_outer)
        performance, y_hat_test = evaluate_deep_model(model_type, X_train_val.copy(), y_train_val.copy(), X_test.copy(), y_test.copy(), features, fold_outer, n_folds_inner, performance, output_dir_model_fold_outer)
        
        predictions.loc[index_test,'geneid'] = group_ids_test
        predictions.loc[index_test,'fold_outer'] = fold_outer
        predictions.loc[index_test,'log2FC_target'] = y_test
        predictions.loc[index_test,'log2FC_predicted'] = y_hat_test
            
        fold_outer += 1
        
    performance_mean = pd.DataFrame({'mean':performance.mean(axis=0).round(3), 'variance':performance.var(axis=0).round(3)}).transpose()
          
    return performance, predictions


############################################
# evaluation of new data with inner fold models
############################################ 

def evaluate_deep_model_new_data(model_type, X, features, fold_outer, n_folds_inner, input_dir_model_fold_outer):  
    
    predictions_fold_inner = np.zeros((X.shape[0], n_folds_inner))
        
    dataset  = CrisprDatasetEval(X, features)
    loader = DataLoader(dataset, batch_size=X.shape[0], num_workers = 3, shuffle = False)
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
    for fold_inner in range(n_folds_inner):
             
        filename_model = input_dir_model_fold_outer + 'model_fold_inner_' + str(fold_inner) + ".ckpt"    
        
        # load model
        if model_type =='1DCNN':
            model = Crispr1DCNN.load_from_checkpoint(filename_model, num_features = len(features))
        
        elif model_type =='GRU':
            model = CrisprGRU.load_from_checkpoint(filename_model, num_features = len(features))
        
        # evaluate model
        model = model.to(device)
        model.eval()
        model.freeze()
                       
        # predict with model
        predictions = []
        for x_sequence_40nt, x_features in loader:
            with torch.no_grad():
                pred = model( x_sequence_40nt.to(device),  x_features.to(device)).detach()
            predictions.extend(pred.cpu().numpy())
        predictions_fold_inner[:,fold_inner] = predictions
    
    predictions = predictions_fold_inner.mean(axis=1)
    
    return predictions
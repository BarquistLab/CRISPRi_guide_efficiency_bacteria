############################################
# imports
############################################

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from src.utils import scaling


############################################
# simple linear model
############################################

def lm_core(X_train, X_test, y_train, y_test, features, fold_outer, performance, output_dir_model, pp, plot_fold):
    
    # standardize training set
    SCALE = StandardScaler()
    SCALE.fit(X_train[features])

    X_scale_train = scaling(SCALE,X_train,features)
    y_train.reset_index(inplace=True,drop=True)

    X_scale_test = scaling(SCALE,X_test,features)
    y_test.reset_index(inplace=True,drop=True)

    # train linear model
    lm = LinearRegression()
    lm.fit(X_scale_train,y_train)

    # predict and calculate MSE and R2
    y_hat_train = lm.predict(X_scale_train)
    y_hat_test = lm.predict(X_scale_test)
    
    performance.iloc[fold_outer,:] = [round(mean_squared_error(y_train,y_hat_train),2), round(mean_squared_error(y_test,y_hat_test),2), 
                                round(spearmanr(y_train,y_hat_train)[0],4),round(spearmanr(y_test,y_hat_test)[0],4)]
    
    # save models and scaler
    filename_model = output_dir_model + "model_fold_outer" + str(fold_outer) + ".pickle"
    pickle.dump(lm, open(filename_model, 'wb'))
    
    filename_scaler = output_dir_model + "scaler_fold_outer" + str(fold_outer) + ".pickle"    
    pickle.dump(SCALE, open(filename_scaler, 'wb'))
    
    # create plots
    if fold_outer in plot_fold:
        n_coef_plot = 20

        # filter coefs
        coefs = pd.DataFrame({"variable": X_scale_test.columns, "coefficient": lm.coef_})
        coefs = coefs.sort_values(by=['coefficient'])
        ind = list(range(0,n_coef_plot,1))
        ind.extend(range(len(coefs["variable"]) - n_coef_plot, len(coefs["variable"]), 1))
        coefs = coefs.iloc[ind,:]
        
        # plot predictions and coefficients
        gs = gridspec.GridSpec(2, 2)
        plt.rcParams['figure.figsize'] = [15, 15]
        plt.figure()
            
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        sns1 = sns.regplot(x=y_train,y=y_hat_train,scatter_kws={"color": "lightgrey"})
        sns1.set(xlabel='true', ylabel='predicted', 
                 title='Training Set ( FOLD ' + str(fold_outer+1) +' ) with MSE: ' + str(round(mean_squared_error(y_train,y_hat_train),2)) + 
                       ' and spearman r: '+ str(round(spearmanr(y_train,y_hat_train)[0],4)))
            
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        sns2 = sns.regplot(x=y_test,y=y_hat_test, scatter_kws={"color": "lightgrey"})
        sns2.set(xlabel='true', ylabel='predicted', 
                 title='Test Set ( FOLD ' + str(fold_outer+1) +' ) with MSE: ' + str(round(mean_squared_error(y_test,y_hat_test),2)) + 
                       ' and spearman r: ' + str(round(spearmanr(y_test,y_hat_test)[0],4)))
            
        ax = plt.subplot(gs[1, :]) # row 1, span all columns
        sns_coef = sns.barplot(x="variable", y="coefficient", data=coefs)
        plt.xticks(rotation=90, fontsize=12)
        pp.savefig()

    return performance, y_hat_test
        

############################################
# lasso model
############################################

def lasso_core(X_train, X_test, y_train, y_test, features, group_ids_train, fold_outer, n_folds_inner, performance, penalty, output_dir_model, pp, plot_fold):
    
    # standardize training set
    SCALE = StandardScaler()
    SCALE.fit(X_train[features])

    X_scale_train = scaling(SCALE,X_train,features)
    y_train.reset_index(inplace=True,drop=True)

    X_scale_test = scaling(SCALE,X_test,features)
    y_test.reset_index(inplace=True,drop=True)

    # train Lasso model with optimization of regularization param
    gkf_inner = GroupKFold(n_splits=n_folds_inner)
    lasso = LassoCV(n_alphas=200, random_state=42, fit_intercept=True, cv = gkf_inner.split(X_scale_train, y_train, groups=group_ids_train), copy_X = True)
    lasso.fit(X_scale_train, y_train)

    # predict and calculate MSE and R2
    y_hat_train = lasso.predict(X_scale_train)
    y_hat_test = lasso.predict(X_scale_test)

    performance.iloc[fold_outer,:] =  [round(mean_squared_error(y_train,y_hat_train),2), round(mean_squared_error(y_test,y_hat_test),2), 
                                 round(spearmanr(y_train,y_hat_train)[0],4),round(spearmanr(y_test,y_hat_test)[0],4)]
    penalty.append(round(lm.alpha_,4))
    
    # save models and scaler
    lasso_best = Lasso()
    lasso_best.coef_ = lasso.coef_
    lasso_best.intercept_ = lasso.intercept_
    
    filename_model = output_dir_model + "model_fold_outer" + str(fold_outer) + ".pickle"
    pickle.dump(lasso_best, open(filename_model, 'wb'))
    
    filename_scaler = output_dir_model + "scaler_fold_outer" + str(fold_outer) + ".pickle"    
    pickle.dump(SCALE, open(filename_scaler, 'wb'))
        
    # create plots
    if fold_outer in plot_fold:
        n_coef_plot = 20

        # filter coefs
        coefs = pd.DataFrame({"variable": X_scale_test.columns, "coefficient": lasso.coef_})
        coefs = coefs.sort_values(by=['coefficient'])
        ind = list(range(0,n_coef_plot,1))
        ind.extend(range(len(coefs["variable"]) - n_coef_plot, len(coefs["variable"]), 1))
        coefs = coefs.iloc[ind,:]
        
        # plot predictions and coefficients
        gs = gridspec.GridSpec(2, 2)
        plt.rcParams['figure.figsize'] = [15, 15]
        plt.figure()
            
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        sns1 = sns.regplot(x=y_train,y=y_hat_train,scatter_kws={"color": "lightgrey"})
        sns1.set(xlabel='true', ylabel='predicted', 
                 title='Training Set ( FOLD ' + str(fold_outer+1) +' ) with MSE: ' + str(round(mean_squared_error(y_train,y_hat_train),2)) + 
                       ' and spearman r: '+ str(round(spearmanr(y_train,y_hat_train)[0],4)))
            
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        sns2 = sns.regplot(x=y_test,y=y_hat_test, scatter_kws={"color": "lightgrey"})
        sns2.set(xlabel='true', ylabel='predicted', 
                 title='Test Set ( FOLD ' + str(fold_outer+1) +' ) with MSE: ' + str(round(mean_squared_error(y_test,y_hat_test),2)) + 
                       ' and spearman r: ' + str(round(spearmanr(y_test,y_hat_test)[0],4)))
            
        ax = plt.subplot(gs[1, :]) # row 1, span all columns
        sns_coef = sns.barplot(x="variable", y="coefficient", data=coefs)
        plt.xticks(rotation=90, fontsize=12)
        pp.savefig()

    return performance, y_hat_test, penalty


############################################
# elnet model
############################################

def elnet_core(X_train, X_test, y_train, y_test, features, group_ids_train, fold_outer, n_folds_inner, performance, penalty, l1_l2_ratio, output_dir_model, pp, plot_fold):
    
    # standardize training set
    SCALE = StandardScaler()
    SCALE.fit(X_train[features])

    X_scale_train = scaling(SCALE,X_train,features)
    y_train.reset_index(inplace=True,drop=True)

    X_scale_test = scaling(SCALE,X_test,features)
    y_test.reset_index(inplace=True,drop=True)

    # train ElNet model with optimization of penalization and regularization param
    gkf_inner = GroupKFold(n_splits=n_folds_inner)
    elnet = ElasticNetCV(l1_ratio = [.001, .01, .1, .2, .5, .8, .9, .95, .99, 1], n_alphas=200, random_state=42, fit_intercept=True, cv = gkf_inner.split(X_scale_train, y_train, groups=group_ids_train))
    elnet.fit(X_scale_train, y_train)

    # predict and calculate MSE and R2
    y_hat_train = elnet.predict(X_scale_train)
    y_hat_test = elnet.predict(X_scale_test)

    performance.iloc[fold_outer,:] =  [round(mean_squared_error(y_train,y_hat_train),2), round(mean_squared_error(y_test,y_hat_test),2), 
                                 round(spearmanr(y_train,y_hat_train)[0],4),round(spearmanr(y_test,y_hat_test)[0],4)]
    penalty.append(round(elnet.alpha_,4))
    l1_l2_ratio.append(elnet.l1_ratio_)
    
    # save models
    elnet_best = ElasticNet()
    elnet_best.coef_ = elnet.coef_
    elnet_best.intercept_ = elnet.intercept_

    filename_model = output_dir_model + "model_fold_outer" + str(fold_outer) + ".pickle"
    pickle.dump(elnet_best, open(filename_model, 'wb'))
    
    filename_scaler = output_dir_model + "scaler_fold_outer" + str(fold_outer) + ".pickle"    
    pickle.dump(SCALE, open(filename_scaler, 'wb'))
        
    # create plots
    if fold_outer in plot_fold:
        n_coef_plot = 20

        # filter coefs
        coefs = pd.DataFrame({"variable": X_scale_test.columns, "coefficient": elnet.coef_})
        coefs = coefs.sort_values(by=['coefficient'])
        ind = list(range(0,n_coef_plot,1))
        ind.extend(range(len(coefs["variable"]) - n_coef_plot, len(coefs["variable"]), 1))
        coefs = coefs.iloc[ind,:]
        
        # plot predictions and coefficients
        gs = gridspec.GridSpec(2, 2)
        plt.rcParams['figure.figsize'] = [15, 15]
        plt.figure()
            
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        sns1 = sns.regplot(x=y_train,y=y_hat_train,scatter_kws={"color": "lightgrey"})
        sns1.set(xlabel='true', ylabel='predicted', 
                 title='Training Set ( FOLD ' + str(fold_outer+1) +' ) with MSE: ' + str(round(mean_squared_error(y_train,y_hat_train),2)) + 
                       ' and spearman r: '+ str(round(spearmanr(y_train,y_hat_train)[0],4)))
            
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        sns2 = sns.regplot(x=y_test,y=y_hat_test, scatter_kws={"color": "lightgrey"})
        sns2.set(xlabel='true', ylabel='predicted', 
                 title='Test Set ( FOLD ' + str(fold_outer+1) +' ) with MSE: ' + str(round(mean_squared_error(y_test,y_hat_test),2)) + 
                       ' and spearman r: ' + str(round(spearmanr(y_test,y_hat_test)[0],4)))
            
        ax = plt.subplot(gs[1, :]) # row 1, span all columns
        sns_coef = sns.barplot(x="variable", y="coefficient", data=coefs)
        plt.xticks(rotation=90, fontsize=12)
        pp.savefig()

    return performance, y_hat_test, penalty, l1_l2_ratio
    
    

############################################
# random forest model with recursive feature elimination
############################################

def rf_core(X_train, X_test, y_train, y_test, features, group_ids_train, fold_outer, n_folds_inner, random_grid, performance, performance_RFE, output_dir_model, pp, plot_fold):
    
    # RF models don't need standardization of features
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)

    # Random search of parameters
    gkf_inner = GroupKFold(n_splits=n_folds_inner)
    rf_random_search = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid, n_iter = 5, 
                                          cv = gkf_inner.split(X_train, y_train, groups=group_ids_train), verbose=0, random_state=42)
    rf_random_search.fit(X_train, y_train, groups = group_ids_train)
    rf = rf_random_search.best_estimator_

    # predict and calculate MSE and spearman
    y_hat_train = rf.predict(X_train)
    y_hat_test = rf.predict(X_test)

    performance.iloc[fold_outer,:] =  [round(mean_squared_error(y_train,y_hat_train),2), round(mean_squared_error(y_test,y_hat_test),2), 
                                 round(spearmanr(y_train,y_hat_train)[0],4), round(spearmanr(y_test,y_hat_test)[0],4)]

    # perform RFE
    rfe = RFECV(rf, cv=3, scoring='neg_mean_squared_error', step=0.1, min_features_to_select = 10)
    rfe.fit(X_train, y_train)
    print("Optimal number of features: {}".format(rfe.n_features_))
    
    X_train = X_train[X_train.columns[(rfe.ranking_ == 1)]]
    X_test = X_test[X_test.columns[(rfe.ranking_ == 1)]]

    rf_rfe = RandomForestRegressor(random_state=42)
    rf_rfe.set_params(**rf.get_params())
    rf_rfe.fit(X_train, y_train)

    # predict and calculate MSE and spearman
    y_hat_train_RFE = rf_rfe.predict(X_train)
    y_hat_test_RFE = rf_rfe.predict(X_test)

    performance_RFE.iloc[fold_outer,:] =  [round(mean_squared_error(y_train,y_hat_train_RFE),2), round(mean_squared_error(y_test,y_hat_test_RFE),2), 
                                 round(spearmanr(y_train,y_hat_train_RFE)[0],4), round(spearmanr(y_test,y_hat_test_RFE)[0],4)]

    # save models
    filename_model = output_dir_model + "model_fold_outer" + str(fold_outer) + ".pickle"
    pickle.dump(rf, open(filename_model, 'wb'))
    
    # create extra folder for FRE model
    folders = output_dir_model.split("/")
    output_dir_model_RFE = '/'.join(folders[:-2]) + "_RFE/" + folders[-2] + "/"
    filename_model_RFE = output_dir_model_RFE + "model_fold_outer" + str(fold_outer) + ".pickle"
    os.makedirs(os.path.dirname(filename_model_RFE), exist_ok=True)
    pickle.dump(rf_rfe, open(filename_model_RFE, 'wb'))

    # create plots
    if fold_outer in plot_fold:
        
        # plot predictions and coefficients
        gs = gridspec.GridSpec(2, 2)
        plt.rcParams['figure.figsize'] = [15, 15]
        plt.figure()
            
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        sns1 = sns.regplot(x=y_train,y=y_hat_train,scatter_kws={"color": "lightgrey"})
        sns1.set(xlabel='true', ylabel='predicted', 
                 title='RF Training Set ( FOLD ' + str(fold_outer+1)+' ) with MSE: ' + str(round(mean_squared_error(y_train,y_hat_train),2)) + 
                       ' and spearman r: '+ str(round(spearmanr(y_train,y_hat_train)[0],4)))
            
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        sns2 = sns.regplot(x=y_test,y=y_hat_test, scatter_kws={"color": "lightgrey"})
        sns2.set(xlabel='true', ylabel='predicted', 
                 title='RF Test Set ( FOLD ' + str(fold_outer+1)+' ) with MSE: ' + str(round(mean_squared_error(y_test,y_hat_test),2)) + 
                       ' and spearman r: ' + str(round(spearmanr(y_test,y_hat_test)[0],4)))
        
        ax = plt.subplot(gs[1, 0]) # row 1, col 0
        sns3 = sns.regplot(x=y_train,y=y_hat_train,scatter_kws={"color": "lightgrey"})
        sns3.set(xlabel='true', ylabel='predicted', 
                 title='RF+RFE Training Set ( FOLD ' + str(fold_outer+1)+' ) with MSE: ' + str(round(mean_squared_error(y_train,y_hat_train_RFE),2)) + 
                       ' and spearman r: '+ str(round(spearmanr(y_train,y_hat_train_RFE)[0],4)))
            
        ax = plt.subplot(gs[1, 1]) # row 1, col 1
        sns4 = sns.regplot(x=y_test,y=y_hat_test, scatter_kws={"color": "lightgrey"})
        sns4.set(xlabel='true', ylabel='predicted', 
                 title='RF+RFE Test Set ( FOLD ' + str(fold_outer+1)+' ) with MSE: ' + str(round(mean_squared_error(y_test,y_hat_test_RFE),2)) + 
                       ' and spearman r: ' + str(round(spearmanr(y_test,y_hat_test_RFE)[0],4)))
        
        pp.savefig()
        #print("---")
        #importances = rf.feature_importances_
        #index = np.argsort(importances)[::-1]

        #print("Feature ranking:")
        #top_features = 30
        #for feature in range(top_features):
        #    print("{}: {}".format(X_train.columns[index[feature]], round(importances[index[feature]],3)))

    return performance, performance_RFE, y_hat_test, y_hat_test_RFE, rf



############################################
# GBM model
############################################


def gbm_core(X_train, X_test, y_train, y_test, features, group_ids_train, fold_outer, n_folds_inner, random_grid, performance, output_dir_model, pp, plot_fold):
    
    # GBM models don't need standardization of features
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)

    # Random search of parameters
    gkf_inner = GroupKFold(n_splits=n_folds_inner)

    gbr_random_search = RandomizedSearchCV(estimator = GradientBoostingRegressor(), param_distributions = random_grid, n_iter = 5, 
                                           cv = gkf_inner.split(X_train, y_train, groups=group_ids_train), verbose=0, random_state=42)
    gbr_random_search.fit(X_train, y_train, groups = group_ids_train)
    gbr = gbr_random_search.best_estimator_

    # predict and calculate MSE and spearman
    y_hat_train = gbr.predict(X_train)
    y_hat_test = gbr.predict(X_test)

    performance.iloc[fold_outer,:] =  [round(mean_squared_error(y_train,y_hat_train),2), round(mean_squared_error(y_test,y_hat_test),2), 
                                 round(spearmanr(y_train,y_hat_train)[0],4), round(spearmanr(y_test,y_hat_test)[0],4)]

    # save models
    filename_model = output_dir_model + "model_fold_outer" + str(fold_outer) + ".pickle"
    pickle.dump(gbr, open(filename_model, 'wb'))
        
    # create plots
    if fold_outer in plot_fold:
        
        # plot predictions and coefficients
        gs = gridspec.GridSpec(1, 2)
        plt.rcParams['figure.figsize'] = [15, 7]
        plt.figure()
            
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        sns1 = sns.regplot(x=y_train,y=y_hat_train,scatter_kws={"color": "lightgrey"})
        sns1.set(xlabel='true', ylabel='predicted', 
                 title='Training Set ( FOLD ' + str(fold_outer+1)+' ) with MSE: ' + str(round(mean_squared_error(y_train,y_hat_train),2)) + 
                       ' and spearman r: '+ str(round(spearmanr(y_train,y_hat_train)[0],4)))
            
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        sns2 = sns.regplot(x=y_test,y=y_hat_test, scatter_kws={"color": "lightgrey"})
        sns2.set(xlabel='true', ylabel='predicted', 
                 title='Test Set ( FOLD ' + str(fold_outer+1)+' ) with MSE: ' + str(round(mean_squared_error(y_test,y_hat_test),2)) + 
                       ' and spearman r: ' + str(round(spearmanr(y_test,y_hat_test)[0],4)))
        pp.savefig()
        
    return performance, y_hat_test, gbr
  
    
############################################
# training
############################################  

def train_model(model, X, Y, features, group_ids, gkf, n_folds_outer, n_folds_inner, random_grid, output_dir_model, pp, X_add = None, Y_add = None, group_ids_add = None):

    fold_outer = 0
    plot_fold = [0,1,2,3,4]
    penalty = []
    l1_l2_ratio = []
    
    performance = pd.DataFrame({'mse_train': np.zeros(n_folds_outer), 'mse_test': np.zeros(n_folds_outer), 
                                'spearmanR_train': np.zeros(n_folds_outer), 'spearmanR_test': np.zeros(n_folds_outer)})
    performance_RFE = pd.DataFrame({'mse_train': np.zeros(n_folds_outer), 'mse_test': np.zeros(n_folds_outer), 
                                    'spearmanR_train': np.zeros(n_folds_outer), 'spearmanR_test': np.zeros(n_folds_outer)})

    predictions = pd.DataFrame({'fold_outer':np.zeros(X.shape[0]), 'geneid': np.zeros(X.shape[0]), 'log2FC_target': np.zeros(X.shape[0]), 'log2FC_predicted': np.zeros(X.shape[0])})
    predictions_RFE = pd.DataFrame({'fold_number':np.zeros(X.shape[0]), 'geneid': np.zeros(X.shape[0]), 'log2FC_target': np.zeros(X.shape[0]), 'log2FC_predicted': np.zeros(X.shape[0])})
   
    
    for index_train, index_test in gkf.split(X, Y, groups=group_ids):

        X_train, X_test = X.iloc[index_train], X.iloc[index_test]
        y_train, y_test = Y.iloc[index_train], Y.iloc[index_test]
        group_ids_train = [group_ids[i] for i in index_train] 
        group_ids_test = [group_ids[i] for i in index_test] 
        
        if (X_add is not None) & (Y_add is not None):
            
            gene_idx_filter = [i for i, gene in enumerate(group_ids_add) if gene in group_ids_test]
            group_ids_add_filtered = [i for j, i in enumerate(group_ids_add) if j not in gene_idx_filter]
        
            X_add_filtered = X_add.copy().drop(gene_idx_filter)
            X_add_filtered.reset_index(inplace=True,drop=True)
    
            Y_add_filtered = Y_add.copy().drop(gene_idx_filter)
            Y_add_filtered.reset_index(inplace=True,drop=True)
            
            X_train = pd.concat([X_train,X_add_filtered],axis=0)
            y_train = pd.concat([y_train,Y_add_filtered],axis=0)
            group_ids_train = group_ids_train + group_ids_add_filtered

        if model == "linear":
            performance, y_hat_test = lm_core(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), features, fold_outer, performance, output_dir_model, pp, plot_fold)
        
        elif model == "lasso":
            performance, y_hat_test, penalty = lasso_core(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), features, group_ids_train.copy(), fold_outer, n_folds_inner, performance, penalty, output_dir_model, pp, plot_fold)
        
        elif model == "elnet":
            performance, y_hat_test, penalty, l1_l2_ratio  = elnet_core(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), features, group_ids_train.copy(), fold_outer, n_folds_inner, performance, penalty, l1_l2_ratio, output_dir_model, pp, plot_fold)
        
        elif model == "random_forest":
            performance, performance_RFE, y_hat_test, y_hat_test_RFE, rf = rf_core(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), features, group_ids_train.copy(), 
                                                                                   fold_outer, n_folds_inner, random_grid, performance, performance_RFE, output_dir_model, pp, plot_fold)
            predictions_RFE.loc[index_test,'geneid'] = group_ids_test
            predictions_RFE.loc[index_test,'fold_outer'] = fold_outer
            predictions_RFE.loc[index_test,'log2FC_target'] = y_test
            predictions_RFE.loc[index_test,'log2FC_predicted'] = y_hat_test_RFE
        
        elif model == "GBM":
            performance, y_hat_test, gbr  = gbm_core(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(), features, group_ids_train.copy(), fold_outer, n_folds_inner, random_grid, performance, output_dir_model, pp, plot_fold)
        
        else:
            print("unknown model")
        
        predictions.loc[index_test,'geneid'] = group_ids_test
        predictions.loc[index_test,'fold_outer'] = fold_outer
        predictions.loc[index_test,'log2FC_target'] = y_test
        predictions.loc[index_test,'log2FC_predicted'] = y_hat_test

        fold_outer += 1  
    
    # print results
    print("Training with {}-fold grouped CV".format(n_folds_outer))
    performance_mean = pd.DataFrame({'mean':performance.mean(axis=0).round(3), 'variance':performance.var(axis=0).round(3)}).transpose()
    print(performance_mean)
    print(performance)
    
    if model == "lasso":
        print("amount of penalty:")
        print(penalty)
        
    elif model == "elnet":
        print("compromise between l1 and l2 penalization:")
        print(l1_l2_ratio)
        print("amount of penalty:")
        print(penalty)
        
    elif model == "random_forest":
        print("parameters of best RF model:")
        print(rf.get_params())
        performance_RFE_mean = pd.DataFrame({'mean':performance_RFE.mean(axis=0).round(3), 'variance':performance_RFE.var(axis=0).round(3)}).transpose()
        print("performance with RFE:")
        print(performance_RFE_mean)
        print(performance_RFE)
        
    elif model == "GBM":
        print("parameters of best GBM model:")
        print(gbr.get_params())
          
    return performance, predictions, performance_RFE, predictions_RFE

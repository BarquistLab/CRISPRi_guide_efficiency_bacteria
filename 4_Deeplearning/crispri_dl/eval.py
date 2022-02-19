############################################
# imports
############################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from scipy.stats import spearmanr
from scipy.stats import wilcoxon


############################################
# eval functions
############################################

def eval_top_n_guides_modelwise(model, dataset, predictions, top_n_guides_list, pp):
    
    if "rank" in dataset:
        order_ascending = False
    else: 
        order_ascending = True
        
    gs = gridspec.GridSpec(3, len(top_n_guides_list))
    plt.rcParams['figure.figsize'] = [21, 7*len(top_n_guides_list)]
    plt.figure()
    
    for n in range(len(top_n_guides_list)):
        top_n_guides = top_n_guides_list[n]
        
        target = predictions['log2FC_target']
        prediction = predictions['log2FC_predicted']
        
        predictions_top_n_guides = predictions.sort_values('log2FC_predicted',ascending = order_ascending).groupby('geneid').head(top_n_guides).copy()
        means_top_n_guides = predictions_top_n_guides.groupby('geneid')['log2FC_original'].mean().to_list()
        mean_of_means_top_n_guides = sum(means_top_n_guides) / len(means_top_n_guides)

        predictions_remaining_guides = predictions.sort_values('log2FC_predicted',ascending = order_ascending).groupby('geneid', group_keys=False).apply(lambda x:x.iloc[top_n_guides:])
        means_remaining_guides = predictions_remaining_guides.groupby('geneid')['log2FC_original'].mean().to_list()
        mean_of_means_remaining_guides = sum(means_remaining_guides) / len(means_remaining_guides)

        spearmanR = round(spearmanr(target, prediction)[0],4) 
        performance_increase = round(((mean_of_means_top_n_guides / mean_of_means_remaining_guides) - 1) * 100, 2)
        wilcoxon_w, wilcoxon_p = wilcoxon(means_remaining_guides, means_top_n_guides, alternative='greater')
        
            
        # plot results
        data = pd.DataFrame({"means":means_remaining_guides+means_top_n_guides,"guides":["remaining guides"]*len(means_remaining_guides)+["top {} guides".format(top_n_guides)]*len(means_top_n_guides)})

        title =  "Model: " + model + " for Dataset: " + dataset +"\n"
        title += "SpearmanR = {}".format(spearmanR) + "\n"
        title += "Wilcoxon Test: W = {} ".format(wilcoxon_w) + " with p-value {} ".format(wilcoxon_p) + "\n"
        title += "Performance increase {}% for mean of means of top {} vs remaining guides".format(performance_increase,top_n_guides) + "\n"

        ax = plt.subplot(gs[0, n])
        ax.text(0.5, 0.5, title, color="black", fontsize=10, ha='center')
        plt.axis('off')

        ax = plt.subplot(gs[1, n]) # row 0, col 0
        ax.set(ylim=(0, 25))
        sns1 = sns.histplot(data=data, x="means", hue="guides", bins=100)
        sns1.set(xlabel='mean log2FC of guides', title="Mean log2FC of top {} vs remaining guides".format(top_n_guides))

        ax = plt.subplot(gs[2, n]) # row 0, col 0
        ax.set(ylim=(-12, 2))
        sns2 = sns.violinplot(data=data, x="guides", y="means")
        sns2.set(ylabel='mean log2FC of guides', title="Mean log2FC of top {} vs remaining guides".format(top_n_guides))

    pp.savefig()
    

############################################

def eval_top_n_guides_genewise(top_n_guides, models_trained, dataset, predictions_of_gene, metrics_per_gene_per_model, ranking_per_gene_per_model, mean_FC_top_n_guides_per_gene_per_model, mean_mse_per_gene_per_model, spearmanR_per_gene_per_model, colnames_metrics, gene, output_plots, plot = False):
    
    if "rank" in dataset:
        order_ascending = False
    else: 
        order_ascending = True
    
    mean_top_n_guides_per_model = pd.DataFrame(columns = ["mean_top_n_guides"], index = models_trained)
    
    for model in models_trained:
            
        target = predictions_of_gene[model + "_" + 'log2FC_target']
        prediction = predictions_of_gene[model + "_" + 'log2FC_predicted']
        original = predictions_of_gene[model + "_" + 'log2FC_original']
        
        predictions_top_n_guides = predictions_of_gene.sort_values(model + "_" + 'log2FC_predicted',ascending = order_ascending).head(top_n_guides).copy()
        mean_top_n_guides = predictions_top_n_guides[model + "_" + 'log2FC_original'].mean()
        
        predictions_remaining_guides = predictions_of_gene.sort_values(model + "_" + 'log2FC_predicted',ascending = order_ascending).apply(lambda x:x.iloc[top_n_guides:])
        distribution = predictions_remaining_guides[model + "_" + 'log2FC_original']
        mean_remaining_guides = distribution.mean()
        
        
        #write to metrics table
        if "spearmanR" in colnames_metrics:
            # calulate spearman correlation
            spearmanR = round(spearmanr(target, prediction)[0],4) 
            metrics_per_gene_per_model.loc[gene, model + "_" + "spearmanR"] = spearmanR

        if "performance_increase" in colnames_metrics:
            # calculate performance increase
            performance_increase = round(((mean_top_n_guides / mean_remaining_guides) - 1) * 100, 2)
            metrics_per_gene_per_model.loc[gene, model + "_" + "performance_increase"] = performance_increase

        if "wilcoxon_p-value" in colnames_metrics:
            # calculate one sample wilcoxon p-value
            wilcoxon_w, wilcoxon_p = wilcoxon(distribution-mean_top_n_guides, alternative='greater')
            metrics_per_gene_per_model.loc[gene, model + "_" + "wilcoxon_p-value"] = wilcoxon_p
        
        mean_top_n_guides_per_model.loc[model, "mean_top_n_guides"] = mean_top_n_guides
        mean_FC_top_n_guides_per_gene_per_model.loc[gene, model] = mean_top_n_guides
        mse_per_guide = [(x - y)**2 for x,y in zip(target, prediction)]
        mean_mse_per_gene_per_model.loc[gene, model] = sum(mse_per_guide) / len(mse_per_guide)
        spearmanR_per_gene_per_model.loc[gene, model] = spearmanR
            
    ranking_models = mean_top_n_guides_per_model["mean_top_n_guides"].rank(ascending=True)
    ranking_per_gene_per_model.loc[gene, :] = ranking_models.to_list()
    
        
    if plot:
        filename_plots = output_plots + "top_" + str(top_n_guides) + "/" + gene + ".pdf"
        os.makedirs(os.path.dirname(filename_plots), exist_ok=True)
        pp = PdfPages(filename_plots)
        plot_guide_distribution_genewise(top_n_guides, models_trained.copy(), original, mean_top_n_guides_per_model, pp)
        pp.close()
        
    return metrics_per_gene_per_model, ranking_per_gene_per_model, mean_FC_top_n_guides_per_gene_per_model, mean_mse_per_gene_per_model, spearmanR_per_gene_per_model


############################################

def plot_guide_distribution_genewise(top_n_guides, models_trained, log2FC_original, mean_top_n_guides_per_model, pp):

    data = pd.DataFrame({"log2FC":log2FC_original})

    colors = sns.color_palette("deep", len(models_trained))

    gs = gridspec.GridSpec(1, 1)
    plt.rcParams['figure.figsize'] = [7, 7]
    plt.figure()

    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    ax.set(ylim=(0, 10))
    ax.set(xlim=(-14,2))
    sns1 = sns.histplot(x=log2FC_original, binwidth=0.1)
    sns1.set(xlabel='log2FC of guides', title="log2FC of top {} vs all guides".format(top_n_guides))
    
    #add lines for models
    for m in range(len(models_trained)):
        plt.axvline(mean_top_n_guides_per_model["mean_top_n_guides"].to_list()[m], color=colors[m], label=models_trained[m])
        plt.text(x = mean_top_n_guides_per_model["mean_top_n_guides"].to_list()[m]+0.1, y = (9-m), s = "model: {} with mean: {}".format(models_trained[m],round(mean_top_n_guides_per_model["mean_top_n_guides"].to_list()[m],2)), color = colors[m])
    
    pp.savefig()
    
    
############################################

def calculate_ranking(models_trained, dataset, metrics_per_gene_per_model, p_value_thr, ranking_per_model_per_dataset, number_sig_p_values_per_model_per_dataset):
    
    p_values_sig = []
    for model in models_trained:
        p_values = metrics_per_gene_per_model[model + "_wilcoxon_p-value"]
        p_values_sig.append(sum(p_values < p_value_thr))
        
    p_values_per_model = pd.DataFrame({"model": models_trained, "p_values_sig": p_values_sig})
    p_values_per_model["rank"] = p_values_per_model["p_values_sig"].rank(ascending=False)
    
    ranking_per_model_per_dataset.loc[dataset, :] = p_values_per_model["rank"].to_list()
    number_sig_p_values_per_model_per_dataset.loc[dataset, :] = p_values_per_model["p_values_sig"].to_list()
    
    return ranking_per_model_per_dataset, number_sig_p_values_per_model_per_dataset

    
############################################

def plot_ranking(ranking_per_model_per_dataset, number_sig_p_values_per_model_per_dataset, top_n_guides, pp):
    
    ranking_per_model_per_dataset["dataset"] = ranking_per_model_per_dataset.index
    ranking_ploting_table = pd.melt(ranking_per_model_per_dataset, id_vars='dataset')
    ranking_ploting_table.columns = ["dataset", "models", "rank"]
    
    
    number_sig_p_values_per_model_per_dataset["dataset"] = number_sig_p_values_per_model_per_dataset.index
    p_value_sig_ploting_table = pd.melt(number_sig_p_values_per_model_per_dataset, id_vars='dataset')
    p_value_sig_ploting_table.columns = ["dataset", "models", "p_values_sig"]

    gs = gridspec.GridSpec(1, 3)
    plt.rcParams['figure.figsize'] = [21, 7]
    plt.figure()
    
    
    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    sns1 = sns.boxplot(data=ranking_ploting_table, x="models", y="rank", color='white')
    sns1.set(ylabel='model ranking', xlabel="models", title="ranking of models over different datasets for top {} guides".format(top_n_guides))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    sns2 = sns.barplot(data=ranking_ploting_table, x="dataset", y="rank", hue="models", palette="GnBu")
    sns2.set(ylabel='rank', xlabel="dataset", title="ranking of models over different datasets for top {} guides".format(top_n_guides))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    ax = plt.subplot(gs[0, 2]) # row 0, col 2
    sns2 = sns.barplot(data=p_value_sig_ploting_table, x="dataset", y="p_values_sig", hue="models", palette="GnBu")
    sns2.set(ylabel='number of genes with significant p-values', xlabel="dataset", title="ranking of models over different datasets for top {} guides".format(top_n_guides))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    plt.tight_layout()
    
    pp.savefig()

    
############################################

def create_challengeR_dataset(metric_per_gene_per_model, dataset):
    metric_per_gene_per_model['TestCase'] = metric_per_gene_per_model.index
    metric_per_gene_per_model = pd.melt(metric_per_gene_per_model, id_vars='TestCase')
    metric_per_gene_per_model["Task"] = dataset
        
    metric_per_gene_per_model.columns = ["TestCase", "Algorithm", "MetricValue", "Task"]
    return(metric_per_gene_per_model[["Task", "TestCase", "Algorithm", "MetricValue"]])
    
    
############################################

def eval_top_n_guides_new_genes(model, dataset, predictions, top_n_guides, performance_metrics, pp):
    
    if "rank" in dataset:
        order_ascending = False
    else: 
        order_ascending = True
    
    predictions_top_n_guides = predictions.sort_values('Log2FC_predicted',ascending = order_ascending).head(top_n_guides).copy()
    means_top_n_guides = predictions_top_n_guides['Log2FC_original'].mean()
    
    predictions_remaining_guides = predictions.sort_values('Log2FC_predicted',ascending = order_ascending).apply(lambda x:x.iloc[top_n_guides:])
    means_remaining_guides = predictions_remaining_guides['Log2FC_original'].mean()
    
    spearmanr_value = round(spearmanr(predictions['Log2FC_original'],predictions['Log2FC_predicted'])[0],4)  
    performance_increase = round(((means_top_n_guides/means_remaining_guides)-1)*100,2)
    wilcoxon_w, wilcoxon_p = wilcoxon(predictions_remaining_guides['Log2FC_original'] - means_top_n_guides, alternative='greater')
    
    performance_metrics.loc[model+"_"+dataset,'SpearmanR'] = spearmanr_value
    performance_metrics.loc[model+"_"+dataset,'Performance_Increase'] = performance_increase
    performance_metrics.loc[model+"_"+dataset,'Wilcoxon_p-value'] = wilcoxon_p
    performance_metrics.loc[model+"_"+dataset,'mean_top_n_guides'] = means_top_n_guides
    
    
    data = pd.DataFrame({"Log2FC":predictions_remaining_guides['Log2FC_original'],"guides":["remaining guides"]*predictions_remaining_guides.shape[0]})
    
    title =  "Model: " + model + "\n"
    title +=  "for Dataset: " + dataset + "\n"
    title += "Wilcoxon Test: W = {} ".format(wilcoxon_w) + "\n"
    title += "with p-value {} ".format(wilcoxon_p) + "\n"
    title += "Performance increase {}% for mean of means of top {} vs remaining guides".format(performance_increase,top_n_guides) + "\n"
    title += "SpearmanR = {}".format(spearmanr_value) + "\n"
    
    
    gs = gridspec.GridSpec(2, 1)
    plt.rcParams['figure.figsize'] = [7, 15]
    plt.figure()
         
    ax = plt.subplot(gs[0, :])
    ax.text(0.5, 0.5, title, color="black", fontsize=12, ha='center')
    plt.axis('off')
 
       
    ax = plt.subplot(gs[1, 0]) # row 0, col 0
    ax.set(ylim=(0, 30))
    ax.set(xlim=(-8,1))
    sns1 = sns.histplot(data=data, x="Log2FC", binwidth=0.1)
    sns1.axvline(means_top_n_guides,color='darkred')
    sns1.set(xlabel='Log2FC of guides', title="Log2FC of top {} vs remaining guides".format(top_n_guides))
      
    pp.savefig()
    
    return performance_metrics


############################################

def plot_top_n_guides_new_genes(log2FC_original, top_n_guides_list, performance_metrics_list, pp):
    
    gs = gridspec.GridSpec(1, len(top_n_guides_list))
    plt.rcParams['figure.figsize'] = [7*len(top_n_guides_list), 7]
    plt.figure()
    
    for n in range(len(top_n_guides_list)):
        
        top_n_guides = top_n_guides_list[n]
        performance_metrics = performance_metrics_list[n].copy()
        
        data = pd.DataFrame({"log2FC":log2FC_original})
        colors = sns.color_palette("deep", performance_metrics.shape[0])

        ax = plt.subplot(gs[0, n])
        ax.set(ylim=(0, 10))
        ax.set(xlim=(-12,1))
        sns1 = sns.histplot(x=log2FC_original, binwidth=0.05)
        sns1.set(xlabel='log2FC of guides', title="log2FC of top {} vs all guides".format(top_n_guides))

        #add lines for models
        index_names = performance_metrics.index
        for m in range(len(index_names)):
            plt.axvline(performance_metrics.loc[index_names[m],'mean_top_n_guides'], color=colors[m], label=index_names[m].replace("rousset","cui"))
            plt.text(x = performance_metrics.loc[index_names[m],'mean_top_n_guides']+0.1, y = (8-m), s = "{} with mean: {}".format(index_names[m].replace("_median-sub_guide",""),
                                                                                                                                    round(performance_metrics.loc[index_names[m],'mean_top_n_guides'],2)), color = colors[m])
            
    pp.savefig()
        
  


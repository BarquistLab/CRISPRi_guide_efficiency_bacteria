
library(challengeR)

metric = "mean_mse" # "spearman", "ranking_top_1", "ranking_top_3"
target = "median-sub" # "rank
datasets = c("wang","rousset_E18")
in_dir_performance = "../reports/performance_eval/2021-4-21/"
out_dir_plots = "../reports/plots_eval/challengeR_two_datasets/"

file_name = paste(in_dir_performance,target,"/challengeR_",metric,".csv",sep="")
table = read.table(file = file_name,sep=",", header = T)

new_table = table[table$Task == paste(datasets[1],target,"guide",sep="_"),]
for(i in 2:length(datasets)){
  new_table = rbind(new_table,table[table$Task == paste(datasets[i],target,"guide",sep="_"),])
}
table = new_table

if(metric == "spearman"){
  smallBetter = FALSE
}else{
  smallBetter = TRUE
}


challenge <- as.challenge(table, by = "Task", algorithm = "Algorithm", case = "TestCase", value = "MetricValue", smallBetter = smallBetter)


ranking <- challenge%>%testThenRank(alpha = 0.05, # significance level
                                    p.adjust.method = "none", # method for adjustment for multiple testing, see ?p.adjus
                                    ties.method = "min" # a character string specifying how ties are treated, see ?base::rank
)

set.seed(1)
ranking_bootstrapped <- ranking%>%bootstrap(nboot = 1000)

meanRanks <- ranking%>%consensus(method = "euclidean") 

file_name_results = paste(out_dir_plots,target,"_",metric,".pdf",sep="")
ranking_bootstrapped %>% report(consensus = meanRanks, title = metric, file = file_name_results, format = "PDF", latex_engine = "pdflatex")






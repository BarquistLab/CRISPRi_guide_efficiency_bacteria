# CRISPRi-guide-efficiency-in-bacteria
This repository contains datasets and Python scripts that were used in the study "Automated machine learning and data integration improve prediction of CRISPRi guide efficiency in bacteria from genome-wide essentiality screens". *Each folder corresponds to each section of the paper.* 

For each Python script, **"-h" shows the detailed description, options and example to run the script.** The "test" folders are the example output of running the script. Please check the "log.txt" file for the input arguments of the tests. *For simplicity, please run the script at the same location as the script.*

###0_Datasets
It contains **3 collected datasets** used in the study with expanded feature set and **files related to the reference genome** to calculate the features used by "feature_enginnering.py".  

###1_Feature_engineering
It contains 2 Python scripts: one for optimizing the model using *auto-sklearn* (autosklearn_feature_engineering.py), the other for evaluating the optimized models from auto-sklearn (feature_sets.py). The choice of different feature sets can be specified with option **"-c / --choice"**.  

###2_Data_fusion
'''
It contains 3 Python scripts: one for optimizing the model using *auto-sklearn* (autosklearn_datafusion.py), one for evaluating the optimized models from auto-sklearn and other model tyeps (datafusion.py), and one for testing another automated machine learning tool H2O (h2o_crispri.py). The choice of different training dataset(s) can be specified with option **"-training"**.  The choice of model type can be specified with option **"-c / --choice"**.
'''
 

#Requirements

All scripts were written in Python (version 3.8). To install all Python dependencies, conda is recommended. 


## Python packages

  |Name             |      Version       |           
  |-----------------|--------------------|
  |python           |       3.8.4        | 
  |auto-sklearn     |       0.10.0       | 
  |scikit-learn     |       0.22.1       |
  |shap             |       0.39.0       | 
  |numpy            |       1.19.2       | 
  |merf             |       1.0          |
  |matplotlib       |       3.5.0        |  
  |seaborn          |       0.11.2       |
  |pandas           |       0.25.3       |
  |scipy            |       1.6.2        |
  |biopython        |       1.76         | 
  |h2o              |       3.30.1.1     |



## To run 0_Datasets/feature_engineering.py, it requires two versions of ViennaRNA package 2 (2.1.9h and 2.4.14)

Online instruction can be found in https://www.tbi.univie.ac.at/RNA/documentation.html

Because two versions are required, suffix for the program needs to be added. 


For version 2.4.14:
```
$ tar -zxvf ViennaRNA-2.4.14.tar.gz
  cd ViennaRNA-2.4.14
  ./configure --program-suffix=2.4.14
  make
  sudo make install
```
For version 2.1.9h:
```
$ tar -zxvf ViennaRNA-2.1.9h.tar.gz
  cd ViennaRNA-2.1.9h
  ./configure --program-suffix=2.1.9h
  make
  sudo make install
'''

	   



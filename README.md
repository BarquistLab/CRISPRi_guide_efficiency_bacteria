[![DOI](https://zenodo.org/badge/461471750.svg)](https://zenodo.org/doi/10.5281/zenodo.10262866)
# CRISPRi_guide_efficiency_bacteria
This repository contains datasets and Python scripts that were used in the study **"Improved prediction of bacterial CRISPRi guide efficiency from depletion screens through mixed-effect machine learning and data integration"**. *Each folder corresponds to a section of the paper.* 

For each Python script, **"-h" shows the detailed description, options and example to run the script.** The "test" folders are the example output of running the script. Please check the "log.txt" file for the input arguments of the tests. *For simplicity, please run the script at the same location as the script.*

### 0_Datasets
It contains **3 collected datasets** used in the study with an expanded feature set and **files related to the reference genome** to calculate the features used by "feature_enginnering.py".  

### 1_Feature_engineering
It contains 2 Python scripts: 
* autosklearn_feature_engineering.py: for optimizing the model using *auto-sklearn*, 
* feature_sets.py: for evaluating the optimized models from auto-sklearn. 

The choice of different feature sets can be specified with option **"-c / --choice"**.  

### 2_Data_fusion

It contains 3 Python scripts: 
* autosklearn_datafusion.py: for optimizing the model using *auto-sklearn*
* datafusion.py: for evaluating the optimized models from auto-sklearn and other model types
* h2o_crispri.py: for testing another automated machine learning tool H2O. 

The choice of different training dataset(s) can be specified with option **"-training"**.  The choice of model type can be specified with option **"-c / --choice"**.

### 3_Segregating_gene_effects

It contains 2 Python scripts: 
* MERF_crispri.py: for segregating gene and guide effects using *MERF*, followed by interpreting the model with SHAP. To test the simplified gene feature set, it can be specified by "-c CAI".
* median_subtracting_model.py: for segregating gene and guide effects using *Median subtracting method*. Either **rf or lasso** (random forest or LASSO model) can be specified by "-c / --choice".

The choice of different training dataset(s) can be specified with option **"-training"**.  The choice of train-test split can be specified with option **"-s / --split"**. To test the models without distance-associated features, please use "-s guide_dropdistance".

### 4_Deeplearning

It contains 1 Python script and supplementary scripts for deep learning model: 
* median_subtracting_model_DL.py:segregating gene and guide effects using *Median subtracting method*. Either **cnn or crispron** (CNN or CRISPRon model) can be specified by "-c / --choice".
* crispri_dl: it contains the scripts that are required to run the deep learning models, such as data loading and architectures.  

The choice of different training dataset(s) can be specified with option **"-training"**. 

### 5_gRNA_design

It contains 1 Python script for gRNA design and efficiency prediction similar to the webpage interface (https://ciao.helmholtz-hiri.de). The Python script version extends the function of the web tool to design gRNAs for **multiple sequences from FASTA input file** and **selected genes by gene name or gene ID** when *reference genome FASTA and GFF* files are provided.

Requirements can be installed by creating a conda environment from the file "environment.yml".
```
$ conda env create -f environment.yml
  conda activate guide_design
```
To obtain conda, please see the instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html


 

# Requirements

All scripts were written in Python (version 3.8). To install all Python dependencies, conda is recommended. 


## Python packages

  |Name             |      Version       |           
  |-----------------|--------------------|
  |python           |       3.8.12       | 
  |auto-sklearn     |       0.10.0       | 
  |scikit-learn     |       0.22.2       |
  |shap             |       0.39         | 
  |numpy            |       1.19.2       | 
  |merf             |       1.0          |
  |matplotlib       |       3.5.0        |  
  |seaborn          |       0.12.2       |
  |pandas           |       0.25.3       |
  |scipy            |       1.10.0       |
  |biopython        |       1.78         | 
  |pytorch          |       1.8.1        |
  |pytorch-lightning|       1.5.10       |



## To run 0_Datasets/feature_engineering.py and 5_gRNA_design, it requires two versions of ViennaRNA package 2 (2.1.9h and 2.4.14)

Online instruction can be found at https://www.tbi.univie.ac.at/RNA/documentation.html

Because two versions are required, a suffix for the program needs to be added. 


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
```
If you install it e.g. on a cluster without sudo rights you can follow this:
```
./configure --prefix=$HOME  --program-suffix=2.1.9h or 2.4.14    #this will create a bin folder in your $HOME directory
make
make install
export PATH=$HOME/bin:$PATH		# You may want to add this to your ~/.bashrc or ~/.bash_profile
```

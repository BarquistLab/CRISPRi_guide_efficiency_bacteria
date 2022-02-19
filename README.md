# CRISPRi-guide-efficiency-in-bacteria


#Installation

## Install system dependencies (ViennaRNA package 2)

To 0_Datasets/feature_engineering.py, it requires two versions of ViennaRNA package 2 (2.1.9h and 2.4.14)

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
  
```
## Install Python dependencies

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
  |flask            |       1.1.2        |



```  


#!/bin/bash

nohup python datafusion.py -o R75 -training 0 -c lasso_hyperopt &
nohup python datafusion.py -o C18 -training 1 -c lasso_hyperopt &
nohup python datafusion.py -o W -training 2 -c lasso_hyperopt &
nohup python datafusion.py -o R75C18 -training 0,1 -c lasso_hyperopt &
nohup python datafusion.py -o R75W -training 0,2 -c lasso_hyperopt &
nohup python datafusion.py -o C18W -training 1,2 -c lasso_hyperopt &
nohup python datafusion.py -o 3sets -training 0,1,2 -c lasso_hyperopt &

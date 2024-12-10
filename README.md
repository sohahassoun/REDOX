# REDOX
This repository contains the python script and datasets for redox potential prediction
# Environment Setup #
The python packages needed for this script are given in the file requirements.txt. You can use conda or pip to install these packages
# Usage #
The script train_redox.py in the folder code can be used to train and test a model on the appropriate dataset. You can change the dataset by changing the dataset variable in the script
# Dataset #
The datasets are in the folder data. Each dataset folder contains three files:
redox_expr.csv: experimentally measured redox potentials
qikprop.csv: ADME properties for molecules in the dataset
dft_energy.csv: HOMO/LUMO energies for the molecules in the dataset
# LICENSE #
This code is licensed under MIT license

Multi-Property Molecular Optimization using an Integrated Poly-Cycle Architecture code.

We run all training and experiments on Ubuntu 18.04.5 using one Nvidia GeForce RTX 2080 Ti 11GB GPU, two Intel
13 Xeon Gold 6230 2.10GHZ CPUs and 64GB RAM memory.


# Installation:
1. Install conda / minicaonda
2. From the main folder run:\
    i. conda env create -f environment.yml\
    ii. conda activate IPCA

All dataset files located in dataset folder.

# Training:
From the main folder run:

1. python train.py 2>error_train.txt

train.py is the main training file, contains most of the hyper-parameter and configuration setting.
After training, the checkpoints will be located in the checkpoints folder, training plots will be located in the plots_output folder.

Main setting:\
property (line 31) [property selection] -> M (= Multi-property).


# Inference  :
From the main folder run:

1. python test.py 2>error_test.txt

test.py is the main testing file, contains most of the hyper-parameter and configuration setting.

Main setting:\
check_testset (line 24) [Main Results: Molecule Optimization] -> True / False.\
property (line 26) [property selection] -> M (= Multi-property).


# Ablation Experiments  :
Druing training change unified_destination and fixed_loss_coef flages (lines 57-58 in train.py) according to:
1. Unified target domains		  -> unified_destination to True.
2. Non-adaptive loss 			  -> fixed_loss_coef to True.

For testing, run regularly except for these changes:
1. Unified target domains		  -> unified_destination to True (line 54 in test.py).
2. NO embedding 				  -> no_embedding to True (line 45 in test.py).